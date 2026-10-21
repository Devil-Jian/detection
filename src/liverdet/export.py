"""One canonical manifest -> five training groups -> COCO + YOLO exports.

Every configuration has the same validation/test patients and annotations.
Test annotations must be manually reviewed unless explicitly doing smoke tests.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from PIL import Image
from .common import GROUPS, boxes_load, clip_boxes, image_size, read_csv, save_csv, check_no_patient_overlap


def _view_augmentation(source: str, boxes: list[list[float]], dest: Path, seed: int):
    """Deterministic geometric augmentation; 1 augmented view per real image."""
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("REAL-AUG needs pip install -e '.[augment]'") from exc
    image = np.asarray(Image.open(source).convert("L"))
    h, w = image.shape
    rng = np.random.default_rng(seed)
    angle = rng.uniform(-10, 10)
    scale = rng.uniform(.9, 1.1)
    translate_x, translate_y = rng.uniform(-.05,.05)*w, rng.uniform(-.05,.05)*h
    affine = cv2.getRotationMatrix2D((w/2,h/2), float(angle),float(scale))
    affine[:,2] += (translate_x,translate_y)
    aug = cv2.warpAffine(image, affine, (w,h), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    result = []
    for x1,y1,x2,y2 in boxes:
        p = np.array([[x1,y1,1],[x1,y2,1],[x2,y1,1],[x2,y2,1]],dtype=float)
        warped = (affine @ p.T).T
        result.extend(clip_boxes([[warped[:,0].min(),warped[:,1].min(),
                                  warped[:,0].max(),warped[:,1].max()]],w,h))
    dest.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(aug,mode="L").save(dest)
    return result


def _row_to_record(row: pd.Series, source: str) -> dict:
    if source == "real":
        return {"path":str(row.real_path), "boxes":boxes_load(row.boxes_real), "origin":source}
    if source == "syn":
        if not str(row.get("syn_path","")).strip():
            raise ValueError(f"SYN is missing {row.sample_id}. Run attach-syn.")
        if row.get("match_qc") != "pass":
            raise ValueError(f"SYN match not reviewed: {row.sample_id}")
        return {"path":str(row.syn_path), "boxes":boxes_load(row.boxes_syn), "origin":source}
    if source == "ncct":
        if not str(row.get("ncct_path","")).strip():
            raise ValueError(f"NCCT missing for {row.sample_id}. Run match-ncct.")
        if row.get("match_qc") != "pass":
            raise ValueError(f"NCCT match not reviewed: {row.sample_id}")
        return {"path":str(row.ncct_path), "boxes":boxes_load(row.boxes_ncct), "origin":source}
    raise ValueError(source)


def _symlink_or_copy(source: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    # Symlinks avoid expensive duplicates; Windows fallback is copy.
    try:
        os.symlink(Path(source).resolve(),dest)
    except OSError:
        shutil.copy2(source,dest)


def _label_is_reviewed(label: str) -> bool:
    return label in ("manual","verified_negative")


def export_group(manifest: str, out_root: str, group: str, phase: str | None = None, *,
                 allow_pseudo_eval: bool = False, seed: int = 42,
                 min_box_area: float = 1) -> Path:
    if group not in GROUPS:
        raise ValueError(f"Unknown group: {group}")
    if group == "ncct":
        phase = "ncct"
    elif phase not in ("ap","pvp"):
        raise ValueError("Specify --phase ap or pvp for enhanced groups")
    df = read_csv(manifest, ["patient_id","phase","split","sample_id",
                             "real_path","boxes_real","label_source"])
    check_no_patient_overlap(df)
    if group == "ncct":
        # One NCCT baseline, preferentially use AP reference per patient.
        frame = df[df.phase == "ap"].copy()
        pids_with_ap = set(frame.patient_id)
        frame = pd.concat([frame,df[(df.phase=="pvp") & ~df.patient_id.isin(pids_with_ap)]],
                          ignore_index=True)
        # In the event that multiple CECT target slices match one NCCT slice:
        # require explicit resolution, rather than silently using wrong labels.
        if "ncct_slice" not in frame:
            raise ValueError("NCCT slice matches have not been generated.")
        dupe = frame.duplicated(["patient_id","ncct_slice"])
        if dupe.any():
            raise ValueError("Multiple CECT slices map to one NCCT slice. "
                             "Review and consolidate the labels before NCCT baseline export.")
    else:
        frame = df[df.phase == phase].copy()

    if frame.empty:
        raise ValueError(f"No samples for {group}/{phase}")
    test = frame[frame.split.isin(["val","test"])]
    if not allow_pseudo_eval and not all(_label_is_reviewed(v) for v in test.label_source):
        raise ValueError("Validation/test must have manual ground truth, not teacher pseudo-labels. "
                         "For pipeline smoke tests only, set --allow-pseudo-eval.")

    base = Path(out_root)/f"{group}_{phase}"
    base.mkdir(parents=True, exist_ok=True)
    split_stats = []
    for split in ("train","val","test"):
        subset = frame[frame.split == split].sort_values("sample_id")
        images, annotations = [], []
        label_dir = base/"labels"/split
        img_dir = base/"images"/split
        label_dir.mkdir(parents=True,exist_ok=True)
        img_dir.mkdir(parents=True,exist_ok=True)
        image_id, annot_id = 0, 0
        for _,row in subset.iterrows():
            sources = ["real"]
            if group == "ncct":
                sources = ["ncct"]
            elif split == "train":
                if group == "cect_syn":
                    sources = ["syn"]
                elif group == "cect_real_syn":
                    sources = ["real","syn"]
                elif group == "cect_real_aug":
                    sources = ["real","aug"]
            for source in sources:
                if source == "aug":
                    record = _row_to_record(row,"real")
                    key = f"{seed}/{row.sample_id}"
                    sample_seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little")% (2**32)
                    aug_path = base/"augmented"/f"{row.sample_id}.png"
                    record["boxes"] = _view_augmentation(record["path"],record["boxes"],aug_path,sample_seed)
                    record["path"] = str(aug_path)
                else:
                    record = _row_to_record(row,source)
                if not Path(record["path"]).is_file():
                    raise FileNotFoundError(record["path"])
                width,height = image_size(record["path"])
                boxes = clip_boxes(record["boxes"],width,height,min_box_area)
                image_id += 1
                basename = f"{image_id:07d}_{row.sample_id}_{source}.png"
                _symlink_or_copy(record["path"],img_dir/basename)
                images.append(dict(id=image_id, file_name=basename,
                                   width=width,height=height,
                                   patient_id=str(row.patient_id),
                                   sample_id=str(row.sample_id)))
                yolo_labels = []
                for box in boxes:
                    x1,y1,x2,y2=box
                    annot_id += 1
                    annotations.append(dict(id=annot_id,image_id=image_id,
                        category_id=1,bbox=[x1,y1,x2-x1,y2-y1],
                        area=(x2-x1)*(y2-y1),iscrowd=0))
                    yolo_labels.append(f"0 {(x1+x2)/(2*width):.8f} {(y1+y2)/(2*height):.8f} "
                                       f"{(x2-x1)/width:.8f} {(y2-y1)/height:.8f}")
                (label_dir/Path(basename).with_suffix(".txt").name).write_text(
                    "\n".join(yolo_labels)+("\n" if yolo_labels else ""),encoding="utf-8")
        coco = dict(images=images,annotations=annotations,
                    categories=[dict(id=1,name="liver_lesion",supercategory="lesion")])
        ann_path = base/"annotations"/f"{split}.json"
        ann_path.parent.mkdir(parents=True,exist_ok=True)
        ann_path.write_text(json.dumps(coco,indent=2),encoding="utf-8")
        split_stats.append(dict(group=group,phase=phase,split=split,
                                images=len(images),boxes=len(annotations)))
    # data.yaml for Ultralytics
    (base/"dataset.yaml").write_text(
        "path: "+str(base.resolve())+"\ntrain: images/train\nval: images/val\n"
        "test: images/test\nnames:\n  0: liver_lesion\n",encoding="utf-8")
    save_csv(pd.DataFrame(split_stats),base/"counts.csv")
    if not allow_pseudo_eval:
        (base/"PAPER_EVAL_READY").write_text("Evaluation images require manual or verified-negative labels.\n")
    else:
        (base/"SMOKE_TEST_ONLY").write_text("WARNING: PSEUDO-LABEL EVALUATION IS NOT A PAPER RESULT.\n")
    return base
