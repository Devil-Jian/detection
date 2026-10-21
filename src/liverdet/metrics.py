"""Unified COCO bbox metrics and clearly labeled slice-object FROC.

NOTE: slice-FROC is NOT patient/3-D lesion FROC. The latter requires manually
linked 3-D lesion IDs across slices and is deliberately not fabricated here.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def intersection_over_union(a: list[float], b: list[float]) -> float:
    """Boxes in COCO xywh."""
    ax1,ay1,aw,ah=a
    bx1,by1,bw,bh=b
    overlap_w=max(0.,min(ax1+aw,bx1+bw)-max(ax1,bx1))
    overlap_h=max(0.,min(ay1+ah,by1+bh)-max(ay1,by1))
    inter=overlap_w*overlap_h
    return inter/max(aw*ah+bw*bh-inter,1e-9)


def slice_froc(gt: dict, detections: list[dict], thresholds=(.05,.1,.2,.3,.5,.7,.9),
               iou_threshold=.5) -> list[dict]:
    all_imgs={im["id"] for im in gt["images"]}
    by_id={i:[] for i in all_imgs}
    preds={i:[] for i in all_imgs}
    for ann in gt["annotations"]:
        by_id[ann["image_id"]].append(ann["bbox"])
    for det in detections:
        if det["image_id"] in preds:
            preds[det["image_id"]].append(det)
    total_gt=len(gt["annotations"])
    records=[]
    for threshold in thresholds:
        tp=fp=0
        for image_id in all_imgs:
            used=set()
            for det in sorted(
                (x for x in preds[image_id] if float(x["score"])>=threshold),
                key=lambda x:float(x["score"]),reverse=True):
                scores=[intersection_over_union(det["bbox"],b) if i not in used else -1
                        for i,b in enumerate(by_id[image_id])]
                best=int(np.argmax(scores)) if scores else -1
                if best>=0 and scores[best]>=iou_threshold:
                    tp+=1
                    used.add(best)
                else:
                    fp+=1
        records.append(dict(threshold=threshold,
                            slice_object_sensitivity=tp/max(total_gt,1),
                            false_positives_per_slice=fp/max(len(all_imgs),1),
                            true_positives=tp,false_positives=fp,total_gt=total_gt))
    return records


def evaluate_predictions(gt_path: str, predictions_path: str,
                         out_path: str | None = None) -> dict:
    gt=json.loads(Path(gt_path).read_text(encoding="utf-8"))
    detections=json.loads(Path(predictions_path).read_text(encoding="utf-8"))
    if not isinstance(detections,list):
        raise ValueError("predictions.json must be a list of COCO detection objects")
    coco=COCO(gt_path)
    if len(detections):
        predicted=coco.loadRes(detections)
        evaluator=COCOeval(coco,predicted,"bbox")
        evaluator.params.imgIds=[im["id"] for im in gt["images"]]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
        ap,ap50,ap75=[float(evaluator.stats[j]) for j in (0,1,2)]
    else:
        ap=ap50=ap75=0.
    result={
        "ap_50_95":ap,"ap50":ap50,"ap75":ap75,
        "froc_definition":"2-D object sensitivity versus false positives per slice",
        "slice_froc":slice_froc(gt,detections),
        "images":len(gt["images"]),"objects":len(gt["annotations"]),
    }
    if out_path:
        p=Path(out_path)
        p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps(result,indent=2),encoding="utf-8")
    return result
