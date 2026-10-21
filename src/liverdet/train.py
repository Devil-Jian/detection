"""Train YOLOv8s and torchvision RetinaNet; export comparable COCO predictions.

RT-DETR-R18 is handled by the external official-repo adapter (see rtdetr.py).
No silent substitution of an RT-DETR-L model under an R18 name.
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from .metrics import evaluate_predictions


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _epochs(n_items:int, batch:int, target_steps:int) -> int:
    return max(1,int(math.ceil(target_steps / max(1, math.ceil(n_items/batch)))))


def train_yolo(dataset_dir: str, out_dir: str, *, target_steps: int=2000,
               batch: int=8, image_size: int=640, seed: int=42, device: str="0",
               pretrained: bool=True, patience:int=15) -> Path:
    from ultralytics import YOLO
    path=Path(dataset_dir)
    train_gt=json.loads((path/"annotations"/"train.json").read_text())
    n=len(train_gt["images"])
    if not n:
        raise ValueError("Empty train set")
    seed_everything(seed)
    model=YOLO("yolov8s.pt" if pretrained else "yolov8s.yaml")
    epochs=_epochs(n,batch,target_steps)
    # Explicitly disable implicit YOLO augmentations in all groups.
    # REAL-AUG is the independently exported 1:1 augmented-view dataset.
    model.train(data=str(path/"dataset.yaml"),imgsz=image_size,batch=batch,
        epochs=epochs,patience=patience,seed=seed,deterministic=True,
        device=device,project=str(Path(out_dir).parent),name=Path(out_dir).name,
        exist_ok=True,verbose=True,
        mosaic=0.0,mixup=0.0,copy_paste=0.0,
        degrees=0.0,translate=0.0,scale=0.0,shear=0.0,
        fliplr=0.0,flipud=0.0,
        hsv_h=0.0,hsv_s=0.0,hsv_v=0.0)
    ckpt=Path(out_dir)/"weights"/"best.pt"
    if not ckpt.is_file():
        raise RuntimeError(f"YOLO best checkpoint missing: {ckpt}")
    return ckpt


class CocoTorchDataset(Dataset):
    """A standard Torchvision-compatible dataset; all images uint8 PNG normalized to [0,1]."""
    def __init__(self, data_dir: str, split: str):
        self.base=Path(data_dir)
        data=json.loads((self.base/"annotations"/f"{split}.json").read_text())
        self.images=data["images"]
        self.annotations={im["id"]:[] for im in self.images}
        self.split=split
        for a in data["annotations"]:
            self.annotations[a["image_id"]].append(a)

    def __len__(self): return len(self.images)

    def __getitem__(self,index):
        info=self.images[index]
        im=np.asarray(Image.open(self.base/"images"/self.split/info["file_name"]).convert("RGB"))
        tensor=torch.from_numpy(im.copy()).permute(2,0,1).float()/255.
        a=self.annotations[info["id"]]
        boxes=[[x,y,x+w,y+h] for ann in a for x,y,w,h in [ann["bbox"]]]
        target={
            "boxes":torch.tensor(boxes,dtype=torch.float32).reshape(-1,4),
            "labels":torch.ones(len(a),dtype=torch.int64),
            "image_id":torch.tensor(info["id"],dtype=torch.int64)
        }
        return tensor,target


def collate(batch):
    return tuple(zip(*batch))


def create_retinanet(pretrained_backbone: bool):
    from torchvision.models import ResNet50_Weights
    from torchvision.models.detection import retinanet_resnet50_fpn
    weights_backbone=ResNet50_Weights.IMAGENET1K_V1 if pretrained_backbone else None
    return retinanet_resnet50_fpn(
        weights=None,weights_backbone=weights_backbone,num_classes=2)


@torch.no_grad()
def retina_predictions(model, dataset_dir: str, split: str, device: torch.device,
                       batch_size:int=4, workers:int=0) -> list[dict]:
    ds=CocoTorchDataset(dataset_dir,split)
    loader=DataLoader(ds,batch_size=batch_size,num_workers=workers,
                      shuffle=False,collate_fn=collate)
    output=[]
    model.eval()
    for images,targets in loader:
        pred=model([x.to(device) for x in images])
        for t,p in zip(targets,pred):
            image_id=int(t["image_id"])
            for b,score,label in zip(
                    p["boxes"].cpu().tolist(),p["scores"].cpu().tolist(),p["labels"].cpu().tolist()):
                if int(label)!=1:
                    continue
                x1,y1,x2,y2=b
                output.append(dict(image_id=image_id,category_id=1,
                            bbox=[x1,y1,x2-x1,y2-y1],score=float(score)))
    return output


def train_retinanet(dataset_dir: str, out_dir: str, *, target_steps: int=2000,
                    batch:int=8, seed:int=42, lr:float=1e-4,
                    device:str="cuda", workers:int=4, pretrained:bool=True,
                    patience:int=15) -> Path:
    seed_everything(seed)
    device=torch.device(device if torch.cuda.is_available() or "cpu" in device else "cpu")
    ds=CocoTorchDataset(dataset_dir,"train")
    if not len(ds):
        raise ValueError("Empty training set")
    train_loader=DataLoader(ds,batch_size=batch,shuffle=True,num_workers=workers,collate_fn=collate)
    net=create_retinanet(pretrained).to(device)
    optimizer=torch.optim.AdamW(net.parameters(),lr=lr,weight_decay=1e-4)
    scaler=torch.amp.GradScaler("cuda",enabled=(device.type=="cuda"))
    epochs=_epochs(len(ds),batch,target_steps)
    out=Path(out_dir)
    out.mkdir(parents=True,exist_ok=True)
    best=-float("inf")
    stale=0
    for epoch in range(1,epochs+1):
        net.train()
        total_loss=0.
        for images, targets in train_loader:
            x=[im.to(device) for im in images]
            y=[{k:v.to(device) for k,v in t.items()} for t in targets]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,
                                enabled=device.type=="cuda"):
                losses=net(x,y)
                loss=sum(losses.values())
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite RetinaNet loss at epoch {epoch}")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss+=loss.item()
        preds=retina_predictions(net,dataset_dir,"val",device,batch_size=max(1,batch//2),workers=workers)
        pred_path=out/"val_predictions.json"
        pred_path.write_text(json.dumps(preds),encoding="utf-8")
        val=evaluate_predictions(str(Path(dataset_dir)/"annotations"/"val.json"),str(pred_path))
        score=val["ap50"]
        print(f"Epoch {epoch}/{epochs}: train_loss={total_loss/max(len(train_loader),1):.5f}, val_AP50={score:.4f}",flush=True)
        if score>best:
            best=score
            stale=0
            torch.save({"model":net.state_dict(),"epoch":epoch,"ap50":score},out/"best.pth")
        else:
            stale+=1
        if stale>=patience:
            break
    return out/"best.pth"


def predict_yolo(checkpoint:str, dataset_dir:str, split:str, output_json:str,
                 image_size:int=640, device:str="0") -> Path:
    from ultralytics import YOLO
    yolo=YOLO(checkpoint)
    base=Path(dataset_dir)
    meta=json.loads((base/"annotations"/f"{split}.json").read_text())
    detections=[]
    for image in meta["images"]:
        filename=base/"images"/split/image["file_name"]
        res=yolo.predict(str(filename),imgsz=image_size,conf=.001,iou=.7,
                         device=device,verbose=False)[0]
        if res.boxes is None:
            continue
        for b,score,cls in zip(res.boxes.xyxy.cpu().tolist(),
                                res.boxes.conf.cpu().tolist(),
                                res.boxes.cls.cpu().tolist()):
            if int(cls)!=0:
                continue
            x1,y1,x2,y2=b
            detections.append(dict(image_id=image["id"],category_id=1,
                                   bbox=[x1,y1,x2-x1,y2-y1],score=float(score)))
    out=Path(output_json)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(detections),encoding="utf-8")
    return out


def predict_retina(checkpoint:str, dataset_dir:str, split:str, output_json:str,
                   device:str="cuda", batch:int=4, workers:int=0) -> Path:
    device=torch.device(device if torch.cuda.is_available() or "cpu" in device else "cpu")
    net=create_retinanet(False)
    payload=torch.load(checkpoint,map_location="cpu",weights_only=True)
    net.load_state_dict(payload["model"])
    net=net.to(device)
    preds=retina_predictions(net,dataset_dir,split,device,batch,workers)
    out=Path(output_json)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(preds),encoding="utf-8")
    return out
