"""Adapter for *official* lyuwenyu/RT-DETR PyTorch RT-DETR-R18VD.

Install official repository separately (not vendored). We generate a dataset-
specific YAML config that inherits the official rtdetr_r18vd_6x_coco.yml.
For its raw COCO category_id=1 and remap=False, num_classes=2 is required
(category index 0 remains unused).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import torch
import yaml
from PIL import Image
import torchvision.transforms as T


def build_rtdetr_config(official_repo: str, dataset_dir: str, output_dir: str,
                        batch: int=8, epochs: int=100,
                        workers: int=4, seed: int=42) -> Path:
    root=Path(official_repo).resolve()
    if (root/"rtdetr_pytorch").is_dir():
        root=root/"rtdetr_pytorch"
    base=root/"configs"/"rtdetr"/"rtdetr_r18vd_6x_coco.yml"
    if not base.exists():
        raise FileNotFoundError(f"Official R18 config missing: {base}")
    data=Path(dataset_dir).resolve()
    train=json.loads((data/"annotations"/"train.json").read_text())
    n=len(train["images"])
    if n == 0:
        raise ValueError("Empty RT-DETR training set")
    if epochs < 1:
        raise ValueError("epochs must be positive")
    dest=Path(output_dir).resolve()
    dest.mkdir(parents=True,exist_ok=True)
    cfg={
        "__include__":[str(base)],
        "output_dir":str(dest),
        "num_classes":2, # Raw COCO category id=1 (not remapped)
        "remap_mscoco_category":False,
        "epoches":epochs,
        # Official DetSolver.fit evaluates on Val every epoch but only writes
        # named epoch checkpoints at checkpoint_step intervals. Set it to 1
        # so the best Val epoch always has a matching saved checkpoint.
        "checkpoint_step":1,
        "train_dataloader":{
            "dataset":{"img_folder":str(data/"images"/"train"),
                       "ann_file":str(data/"annotations"/"train.json")},
            "batch_size":batch,"num_workers":workers},
        "val_dataloader":{
            "dataset":{"img_folder":str(data/"images"/"val"),
                       "ann_file":str(data/"annotations"/"val.json")},
            "batch_size":batch,"num_workers":workers},
    }
    path=dest/"rtdetr_r18_experiment.yml"
    path.write_text(yaml.safe_dump(cfg,sort_keys=False),encoding="utf-8")
    return path


def train_rtdetr(official_repo: str, dataset_dir: str, out_dir: str, *,
                 batch: int=8, epochs: int=100, workers: int=4,
                 seed: int=42, gpu: str="0", amp: bool=True) -> Path:
    root=Path(official_repo).resolve()
    if (root/"rtdetr_pytorch").is_dir():
        root=root/"rtdetr_pytorch"
    config=build_rtdetr_config(str(root),dataset_dir,out_dir,batch,epochs,workers,seed)
    command=[sys.executable,"tools/train.py","-c",str(config),"--seed",str(seed)]
    if amp:
        command.append("--amp")
    env=os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"]=str(gpu)
    subprocess.run(command,cwd=root,env=env,check=True)
    out=Path(out_dir)
    # Our config enforces checkpoint_step=1; official engine validates on Val
    # every epoch. Upstream RT-DETR does NOT implement patience early stopping.
    records=out/"log.txt"
    if records.exists():
        best=(-float("inf"),None)
        for line in records.read_text().splitlines():
            try:
                entry=json.loads(line)
                metric=entry.get("test_coco_eval_bbox")
                epoch=entry.get("epoch")
                path=out/f"checkpoint{int(epoch):04d}.pth"
                if metric and metric[0] > best[0] and path.exists():
                    best=(float(metric[0]),path)
            except (ValueError,TypeError,KeyError):
                continue
        if best[1] is not None:
            return best[1]
    raise RuntimeError(
        "No Val-selected RT-DETR checkpoint found in log.txt and checkpoint{epoch:04d}.pth. "
        "Check official solver logs and checkpoint_step=1; refusing to test a "
        "last-epoch checkpoint without Val model selection."
    )


@torch.no_grad()
def predict_rtdetr(official_repo: str, config_path: str,
                   checkpoint: str, dataset_dir: str, split: str,
                   output_json: str, device: str="cuda") -> Path:
    """Official model + postprocessor adapted from its tools/infer.py."""
    root=Path(official_repo).resolve()
    if (root/"rtdetr_pytorch").is_dir():
        root=root/"rtdetr_pytorch"
    if not (root/"src"/"core").is_dir():
        raise FileNotFoundError(f"Not an RT-DETR PyTorch checkout: {root}")
    sys.path.insert(0,str(root))
    # import the official project's 'src' package only after sys.path insertion
    from src.core import YAMLConfig  # type: ignore
    config=YAMLConfig(str(Path(config_path).resolve()),resume=str(checkpoint))
    payload=torch.load(checkpoint,map_location="cpu",weights_only=False)
    state=payload["ema"]["module"] if "ema" in payload else payload["model"]
    config.model.load_state_dict(state)
    model=config.model.deploy()
    processor=config.postprocessor.deploy()
    device=torch.device(device if torch.cuda.is_available() or "cpu" in device else "cpu")
    model=model.to(device).eval()
    processor=processor.to(device).eval()
    preprocess=T.Compose([T.Resize((640,640)),T.ToTensor()])
    data=Path(dataset_dir)
    records=json.loads((data/"annotations"/f"{split}.json").read_text())
    output=[]
    for meta in records["images"]:
        image=Image.open(data/"images"/split/meta["file_name"]).convert("RGB")
        w,h=image.size
        x=preprocess(image).unsqueeze(0).to(device)
        orig=torch.tensor([[w,h]],device=device,dtype=torch.int64)
        raw=model(x)
        labels,boxes,scores=processor(raw,orig)
        for cls,box,conf in zip(labels[0].cpu().tolist(),
                                boxes[0].cpu().tolist(),scores[0].cpu().tolist()):
            if int(cls)!=1 or float(conf)<.001:
                continue
            x1,y1,x2,y2=box
            output.append(dict(image_id=meta["id"],category_id=1,
                               bbox=[x1,y1,x2-x1,y2-y1],score=float(conf)))
    output_path=Path(output_json)
    output_path.parent.mkdir(parents=True,exist_ok=True)
    output_path.write_text(json.dumps(output),encoding="utf-8")
    return output_path
