"""Command-line entry for the complete detection data flow."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import yaml

from .inventory import scan_cases
from .cohorts import build_cohorts
from .audit import audit_cases
from .prepare import prepare_slices
from .matching import match_ncct, apply_reviews, synthesis_requests, attach_synthetic
from .export import export_group
from .metrics import evaluate_predictions


def configure() -> argparse.ArgumentParser:
    p=argparse.ArgumentParser(prog="liverdet",
        description="CECT detection experiments: prepare -> match -> synth -> export -> train -> evaluate")
    commands=p.add_subparsers(dest="action",required=True)

    s=commands.add_parser("scan",help="Discover AP/PVP and NIfTI pseudo-mask cases")
    s.add_argument("--images",required=True)
    s.add_argument("--labels-ap",required=True)
    s.add_argument("--labels-pvp",required=True)
    s.add_argument("--ncct-root")
    s.add_argument("--out",required=True)

    s=commands.add_parser("build-cohorts",
        help="Build TRAIN=pseudo and VAL/TEST=manual from private case-ID CSV, without scanning the whole registered root")
    s.add_argument("--case-list",required=True,
                   help="CSV columns patient_id,cohort_type,ap_image,ap_mask,ap_mask_type,pvp_image,pvp_mask,pvp_mask_type")
    s.add_argument("--workdir",required=True)
    s.add_argument("--labels-ap",default="/dataset/detection/ap")
    s.add_argument("--labels-pvp",default="/dataset/detection/pvp")
    s.add_argument("--val-fraction",type=float,default=0.5,
                   help="Fraction of ground-truth patients for Val; remainder Test")
    s.add_argument("--seed",type=int,default=42)
    s.add_argument("--skip-missing-pseudo",action="store_true",
                   help="Exclude any pseudo training patient without BOTH valid phase images/masks, audit exclusion in cohort_inventory.csv")

    s=commands.add_parser("audit",help="Read-only NIfTI path, shape and affine audit before prepare")
    s.add_argument("--cases",required=True)
    s.add_argument("--out",required=True)
    s.add_argument("--inspect-masks",action="store_true",
                   help="Read full lesion-mask volumes to count positive slices and check mask value ranges")
    s.add_argument("--check-liver-mask",action="store_true",
                   help="Optional old workflow: audit liver mask geometry and pseudo lesions outside liver")
    s.add_argument("--affine-tolerance",type=float,default=1e-3)

    s=commands.add_parser("prepare",help="NIfTI -> PNG + slice manifest")
    s.add_argument("--cases",required=True)
    s.add_argument("--workdir",required=True)
    s.add_argument("--splits",help="CSV patient_id,split; strongly recommended")
    s.add_argument("--trusted-negatives",help="CSV patient_id,phase,slice_index")
    s.add_argument("--seed",type=int,default=42)
    s.add_argument("--hu-min",type=float,default=-160)
    s.add_argument("--hu-max",type=float,default=240)
    s.add_argument("--min-component-pixels",type=int,default=4,
                   help="Keep teacher pseudo 2D connected components with at least N pixels (default 4); manual GT is untouched")
    s.add_argument("--filter-pseudo-liver-mask",action="store_true",
                   help="Optional old workflow ONLY: clip pseudo masks by same-phase liver mask before extracting 2D boxes")
    s.add_argument("--print-train-slices",action="store_true",
                   help="Print sample_id, original NIfTI path, PNG path and zero-based slice_index for every saved Train slice; always write manifests/train_slices.json")

    s=commands.add_parser("match-ncct",help="Local spatial+NMI match, all QC initially pending")
    s.add_argument("--manifest",required=True)
    s.add_argument("--workdir",required=True)
    s.add_argument("--radius-mm",type=float,default=15)
    s.add_argument("--hu-min",type=float,default=-160)
    s.add_argument("--hu-max",type=float,default=240)

    s=commands.add_parser("review-matches",help="Apply human QC: sample_id,match_qc")
    s.add_argument("--manifest",required=True)
    s.add_argument("--reviews",required=True)

    s=commands.add_parser("synthesis-requests",help="Output request list for external generator")
    s.add_argument("--manifest",required=True)
    s.add_argument("--syn-root",required=True)
    s.add_argument("--out",required=True)

    s=commands.add_parser("attach-syn",help="Attach generated PNGs with geometry checks")
    s.add_argument("--manifest",required=True)
    s.add_argument("--syn-root",required=True)
    s.add_argument("--allow-unreviewed-syn",action="store_true",
                   help="Engineering smoke test ONLY; exports still require approved matches")

    s=commands.add_parser("export",help="Create COCO and YOLO for one group/phase")
    s.add_argument("--manifest",required=True)
    s.add_argument("--out-root",required=True)
    s.add_argument("--group",required=True,
                   choices=["ncct","cect_real","cect_syn","cect_real_aug","cect_real_syn"])
    s.add_argument("--phase",choices=["ap","pvp"])
    s.add_argument("--seed",type=int,default=42)
    s.add_argument("--allow-pseudo-eval",action="store_true",
                   help="Engineering smoke test ONLY, forbidden for final paper tables")

    s=commands.add_parser("matrix",help="Print fixed 27 experiment names")

    s=commands.add_parser("train",help="Train a detector on an exported group")
    s.add_argument("--model",required=True,
                   choices=["yolov8s","retinanet_r50","rtdetr_r18"])
    s.add_argument("--dataset",required=True)
    s.add_argument("--out",required=True)
    s.add_argument("--steps",type=int,default=2000)
    s.add_argument("--batch",type=int,default=8)
    s.add_argument("--seed",type=int,default=42)
    s.add_argument("--device",default="0")
    s.add_argument("--workers",type=int,default=4)
    s.add_argument("--lr",type=float,default=1e-4)
    s.add_argument("--imgsz",type=int,default=640)
    s.add_argument("--patience",type=int,default=15)
    s.add_argument("--no-pretrained",action="store_true")
    s.add_argument("--rtdetr-repo",help="Path to official RT-DETR checkout")

    s=commands.add_parser("predict",help="Convert model detections to COCO JSON")
    s.add_argument("--model",required=True,
                   choices=["yolov8s","retinanet_r50","rtdetr_r18"])
    s.add_argument("--dataset",required=True)
    s.add_argument("--checkpoint",required=True)
    s.add_argument("--out",required=True)
    s.add_argument("--split",choices=["val","test"],default="test")
    s.add_argument("--device",default="0")
    s.add_argument("--workers",type=int,default=0)
    s.add_argument("--batch",type=int,default=4)
    s.add_argument("--imgsz",type=int,default=640)
    s.add_argument("--rtdetr-repo",help="Path to official RT-DETR checkout")
    s.add_argument("--rtdetr-config",help="Auto-generated config from the training run")

    s=commands.add_parser("evaluate",help="COCO AP + slice-FROC from prediction JSON")
    s.add_argument("--dataset",required=True)
    s.add_argument("--predictions",required=True)
    s.add_argument("--out",required=True)
    s.add_argument("--split",choices=["val","test"],default="test")
    return p


def matrix() -> list[str]:
    names=[]
    for model in ("yolov8s","retinanet_r50","rtdetr_r18"):
        names.append(f"{model}_ncct_ncct")
        for phase in ("ap","pvp"):
            for group in ("cect_real","cect_syn","cect_real_aug","cect_real_syn"):
                names.append(f"{model}_{phase}_{group}")
    assert len(names)==27
    return names


def main(argv: list[str] | None=None) -> None:
    args=configure().parse_args(argv)
    if args.action=="scan":
        out=scan_cases(args.images,args.labels_ap,args.labels_pvp,args.out,args.ncct_root)
        print(f"Discovered {len(out)} phase cases. REVIEW cases.csv before preparing.")
    elif args.action=="build-cohorts":
        build_cohorts(args.case_list,args.workdir,
                      pseudo_ap_root=args.labels_ap,pseudo_pvp_root=args.labels_pvp,
                      val_fraction=args.val_fraction,seed=args.seed,
                      allow_missing_pseudo=args.skip_missing_pseudo)
    elif args.action=="audit":
        audit_cases(args.cases,args.out,inspect_masks=args.inspect_masks,
                    affine_tolerance=args.affine_tolerance,
                    check_liver_mask=args.check_liver_mask)
    elif args.action=="prepare":
        frame=prepare_slices(args.cases,args.workdir,split_csv=args.splits,
            trusted_negatives_csv=args.trusted_negatives,seed=args.seed,
            hu_min=args.hu_min,hu_max=args.hu_max,
            min_component_pixels=args.min_component_pixels,
            filter_pseudo_liver_mask=args.filter_pseudo_liver_mask,
            print_train_slices=args.print_train_slices)
        print(f"Prepared {len(frame)} verified positive/negative slices.")
    elif args.action=="match-ncct":
        frame=match_ncct(args.manifest,args.workdir,radius_mm=args.radius_mm,
                         hu_min=args.hu_min,hu_max=args.hu_max)
        print(f"Matched {sum(frame.ncct_path.astype(str).str.len()>0)} target slices; ALL require manual QC.")
    elif args.action=="review-matches":
        frame=apply_reviews(args.manifest,args.reviews)
        print(frame.match_qc.value_counts().to_string())
    elif args.action=="synthesis-requests":
        frame=synthesis_requests(args.manifest,args.syn_root,args.out)
        print(f"{len(frame)} external synthesis requests written.")
    elif args.action=="attach-syn":
        frame=attach_synthetic(args.manifest,args.syn_root,args.allow_unreviewed_syn)
        print(f"{sum(frame.syn_path.astype(str).str.len()>0)} Syn images attached.")
    elif args.action=="export":
        result=export_group(args.manifest,args.out_root,args.group,args.phase,
                            seed=args.seed,allow_pseudo_eval=args.allow_pseudo_eval)
        print(f"COCO+YOLO exported: {result}")
    elif args.action=="matrix":
        print("\n".join(matrix()))
    elif args.action=="train":
        from .train import train_yolo,train_retinanet
        if args.model=="yolov8s":
            result=train_yolo(args.dataset,args.out,target_steps=args.steps,batch=args.batch,
                 image_size=args.imgsz,seed=args.seed,device=args.device,
                 pretrained=not args.no_pretrained,patience=args.patience)
        elif args.model=="retinanet_r50":
            device=f"cuda:{args.device}" if str(args.device).isdigit() else args.device
            result=train_retinanet(args.dataset,args.out,target_steps=args.steps,batch=args.batch,
                 seed=args.seed,device=device,workers=args.workers,lr=args.lr,
                 pretrained=not args.no_pretrained,patience=args.patience)
        else:
            if not args.rtdetr_repo:
                raise ValueError("RT-DETR-R18 requires --rtdetr-repo path to official checkout")
            from .rtdetr import train_rtdetr
            result=train_rtdetr(args.rtdetr_repo,args.dataset,args.out,batch=args.batch,
                target_steps=args.steps,workers=args.workers,seed=args.seed,gpu=args.device)
        ckpt_file=Path(args.out)/"best_checkpoint.txt"
        ckpt_file.parent.mkdir(parents=True,exist_ok=True)
        ckpt_file.write_text(str(Path(result).resolve())+"\n",encoding="utf-8")
        print(f"Checkpoint: {result}")
    elif args.action=="predict":
        from .train import predict_yolo,predict_retina
        if args.model=="yolov8s":
            result=predict_yolo(args.checkpoint,args.dataset,args.split,args.out,
                                image_size=args.imgsz,device=args.device)
        elif args.model=="retinanet_r50":
            device=f"cuda:{args.device}" if str(args.device).isdigit() else args.device
            result=predict_retina(args.checkpoint,args.dataset,args.split,args.out,
                                  device=device,batch=args.batch,workers=args.workers)
        else:
            if not args.rtdetr_repo or not args.rtdetr_config:
                raise ValueError("RT-DETR requires --rtdetr-repo and --rtdetr-config")
            from .rtdetr import predict_rtdetr
            device=f"cuda:{args.device}" if str(args.device).isdigit() else args.device
            result=predict_rtdetr(args.rtdetr_repo,args.rtdetr_config,args.checkpoint,
                                   args.dataset,args.split,args.out,device=device)
        print(f"Predictions: {result}")
    elif args.action=="evaluate":
        result=evaluate_predictions(str(Path(args.dataset)/"annotations"/f"{args.split}.json"),
                                     args.predictions,args.out)
        print(json.dumps({k:v for k,v in result.items() if k!="slice_froc"},indent=2))


if __name__ == "__main__":
    main()
