#!/usr/bin/env python3
"""Sequential execution plan for 27 configurations.

Prepare/export datasets before running. By default this refuses data exports
made with unreviewed pseudo-evaluation labels.
"""
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path
import subprocess
import sys

MODELS=("yolov8s","retinanet_r50","rtdetr_r18")
GROUPS=("cect_real","cect_syn","cect_real_aug","cect_real_syn")


def jobs():
    for model in MODELS:
        yield model,"ncct","ncct"
        for phase in ("ap","pvp"):
            for group in GROUPS:
                yield model,group,phase


def run(args):
    root=Path(args.exports)
    checkpoint_root=Path(args.runs)
    results_root=Path(args.results)
    records=[]
    for model,group,phase in jobs():
        if args.models and model not in args.models:
            continue
        if args.groups and group not in args.groups:
            continue
        if args.phases and phase not in args.phases:
            continue
        key=f"{model}_{phase}_{group}_seed{args.seed}"
        data=root/f"{group}_{phase}"
        if not (data/"annotations"/"test.json").exists():
            raise FileNotFoundError(f"First run 'liverdet export' for {data}")
        if (data/"SMOKE_TEST_ONLY").exists() and not args.allow_smoke:
            raise ValueError(f"{data} was exported using pseudo validation/test labels")
        run_dir=checkpoint_root/key
        ckpt_txt=run_dir/"best_checkpoint.txt"
        preds=results_root/f"{key}_predictions.json"
        metrics=results_root/f"{key}_metrics.json"
        train_cmd=[
            "liverdet","train","--model",model,"--dataset",str(data),
            "--out",str(run_dir),"--epochs",str(args.epochs),
            "--batch",str(args.batch),"--seed",str(args.seed),
            "--device",str(args.device)]
        if model=="rtdetr_r18":
            if not args.rtdetr_repo:
                raise ValueError("--rtdetr-repo required for RT-DETR-R18")
            train_cmd+=["--rtdetr-repo",args.rtdetr_repo]
        predict_cmd=[
            "liverdet","predict","--model",model,"--dataset",str(data),
            "--split","test","--out",str(preds),"--device",str(args.device)]
        if model=="rtdetr_r18":
            predict_cmd+=["--rtdetr-repo",args.rtdetr_repo,
                "--rtdetr-config",str(run_dir/"rtdetr_r18_experiment.yml")]
        eval_cmd=["liverdet","evaluate","--dataset",str(data),
                  "--predictions",str(preds),"--out",str(metrics)]
        print(f"\n=== {key} ===",flush=True)
        if args.dry_run:
            print("TRAIN:", " ".join(train_cmd))
            print("PREDICT: checkpoint path from",ckpt_txt)
            print("EVAL:", " ".join(eval_cmd))
            continue
        if not (args.skip_completed and ckpt_txt.is_file()):
            subprocess.run(train_cmd,check=True)
        if not ckpt_txt.is_file():
            raise FileNotFoundError(f"No saved checkpoint list at {ckpt_txt}")
        checkpoint=ckpt_txt.read_text().strip()
        if not Path(checkpoint).is_file():
            raise FileNotFoundError(checkpoint)
        # New train CLI already predicts and evaluates Test once, after the
        # best Val checkpoint is fixed. Copy these files instead of testing
        # again. Legacy runs without them retain the old predict/eval fallback.
        auto_preds=run_dir/"test_predictions.json"
        auto_metrics=run_dir/"test_metrics.json"
        if auto_preds.is_file() and auto_metrics.is_file():
            results_root.mkdir(parents=True,exist_ok=True)
            if not (args.skip_completed and preds.is_file()):
                shutil.copy2(auto_preds,preds)
            if not (args.skip_completed and metrics.is_file()):
                shutil.copy2(auto_metrics,metrics)
        else:
            if not (args.skip_completed and preds.is_file()):
                subprocess.run(predict_cmd+["--checkpoint",checkpoint],check=True)
            if not (args.skip_completed and metrics.is_file()):
                subprocess.run(eval_cmd,check=True)
        records.append(dict(experiment=key,model=model,group=group,phase=phase,
                            checkpoint=checkpoint,predictions=str(preds),
                            metrics=str(metrics)))
    if not args.dry_run:
        results_root.mkdir(parents=True,exist_ok=True)
        dest=results_root/"experiment_manifest.csv"
        with dest.open("w",newline="") as handle:
            writer=csv.DictWriter(handle,fieldnames=[
                "experiment","model","group","phase","checkpoint","predictions","metrics"])
            writer.writeheader()
            writer.writerows(records)
        print(f"Saved {len(records)} experiment records to {dest}")


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument("--exports",required=True,help="Directory with 9 exported group_phase folders")
    parser.add_argument("--runs",required=True)
    parser.add_argument("--results",required=True)
    parser.add_argument("--rtdetr-repo")
    parser.add_argument("--models",nargs="*",choices=MODELS)
    parser.add_argument("--groups",nargs="*",choices=("ncct",)+GROUPS)
    parser.add_argument("--phases",nargs="*",choices=("ncct","ap","pvp"))
    parser.add_argument("--epochs",type=int,default=100)
    parser.add_argument("--batch",type=int,default=8)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--device",default="0")
    parser.add_argument("--skip-completed",action="store_true")
    parser.add_argument("--allow-smoke",action="store_true",
                        help="Engineering smoke test ONLY")
    parser.add_argument("--dry-run",action="store_true")
    run(parser.parse_args(argv))


if __name__=="__main__":
    main()
