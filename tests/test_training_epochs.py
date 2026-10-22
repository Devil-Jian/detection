"""Contract tests: epoch-based training, per-epoch Val, one final Test."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from liverdet import cli
from liverdet.train import train_yolo
from liverdet.rtdetr import build_rtdetr_config


def _dataset(tmp_path: Path) -> Path:
    root = tmp_path / "dataset"
    ann = root / "annotations"
    ann.mkdir(parents=True)
    (ann / "train.json").write_text(json.dumps({"images": [{"id": 1}]}))
    (ann / "test.json").write_text(json.dumps({"images": [], "annotations": []}))
    return root


def test_train_cli_epoch_defaults():
    parser = cli.configure()
    args = parser.parse_args([
        "train", "--model", "yolov8s", "--dataset", "/tmp/data", "--out", "/tmp/run"
    ])
    assert args.epochs == 100
    assert args.patience == 15
    assert not hasattr(args, "steps")
    chosen = parser.parse_args([
        "train", "--model", "yolov8s", "--dataset", "/tmp/data",
        "--out", "/tmp/run", "--epochs", "70", "--patience", "8",
    ])
    assert chosen.epochs == 70
    assert chosen.patience == 8
    with pytest.raises(SystemExit):
        parser.parse_args([
            "train", "--model", "yolov8s", "--dataset", "/tmp/data",
            "--out", "/tmp/run", "--steps", "2000",
        ])


def test_yolo_training_explicit_val_each_epoch_best_checkpoint(tmp_path, monkeypatch):
    dataset = _dataset(tmp_path)
    (dataset / "dataset.yaml").write_text("path: /tmp/data\ntrain: images/train\nval: images/val\n")
    calls = []
    out = tmp_path / "run"

    class FakeYOLO:
        def __init__(self, weights):
            assert weights == "yolov8s.pt"

        def train(self, **kw):
            calls.append(kw)
            (out / "weights").mkdir(parents=True)
            (out / "weights" / "best.pt").write_text("best")

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=FakeYOLO))
    checkpoint = train_yolo(str(dataset), str(out), epochs=100, patience=15)
    assert checkpoint == out / "weights" / "best.pt"
    assert len(calls) == 1
    assert calls[0]["epochs"] == 100
    assert calls[0]["patience"] == 15
    assert calls[0]["val"] is True
    assert calls[0]["save"] is True
    assert calls[0]["mosaic"] == 0.0


def test_cli_auto_test_once_after_best_val_checkpoint(tmp_path, monkeypatch):
    dataset = _dataset(tmp_path)
    out = tmp_path / "runs" / "yolo_ap"
    order = []

    def fake_train(dataset_dir, out_dir, **kw):
        order.append(("train", kw["epochs"], kw["patience"]))
        ckpt = Path(out_dir) / "weights" / "best.pt"
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        ckpt.write_text("best")
        return ckpt

    def fake_predict(checkpoint, dataset_dir, split, output_json, **kw):
        order.append(("predict", split, Path(checkpoint).name))
        Path(output_json).write_text("[]")
        return Path(output_json)

    def fake_evaluate(gt, preds, metrics):
        order.append(("evaluate", Path(gt).name))
        Path(metrics).write_text(json.dumps({"ap50": 0.4, "ap_50_95": 0.2}))
        return {"ap50": 0.4, "ap_50_95": 0.2}

    monkeypatch.setattr("liverdet.train.train_yolo", fake_train)
    monkeypatch.setattr("liverdet.train.predict_yolo", fake_predict)
    monkeypatch.setattr(cli, "evaluate_predictions", fake_evaluate)
    cli.main([
        "train", "--model", "yolov8s", "--dataset", str(dataset),
        "--out", str(out), "--epochs", "100", "--patience", "15",
    ])
    assert order == [
        ("train", 100, 15),
        ("predict", "test", "best.pt"),
        ("evaluate", "test.json"),
    ]
    assert (out / "test_predictions.json").is_file()
    assert (out / "test_metrics.json").is_file()
    assert (out / "best_checkpoint.txt").read_text().strip() == str(
        (out / "weights" / "best.pt").resolve()
    )


def test_cli_prevents_test_on_unreviewed_pseudo_gt(tmp_path, monkeypatch):
    dataset = _dataset(tmp_path)
    (dataset / "SMOKE_TEST_ONLY").touch()
    def not_called(*args, **kwargs):
        raise AssertionError("must not train on smoke evaluation cohort")
    monkeypatch.setattr("liverdet.train.train_yolo", not_called)
    with pytest.raises(ValueError, match="SMOKE_TEST_ONLY"):
        cli.main([
            "train", "--model", "yolov8s", "--dataset", str(dataset),
            "--out", str(tmp_path / "run")
        ])


def test_rtdetr_config_100_epochs_val_checkpoint_per_epoch(tmp_path):
    official = tmp_path / "rtdetr_pytorch"
    base = official / "configs" / "rtdetr" / "rtdetr_r18vd_6x_coco.yml"
    base.parent.mkdir(parents=True)
    base.write_text("task: detection\n")
    data = _dataset(tmp_path)
    cfg_path = build_rtdetr_config(
        str(official), str(data), str(tmp_path / "runs"), epochs=100
    )
    cfg = yaml.safe_load(cfg_path.read_text())
    assert cfg["epoches"] == 100
    assert cfg["checkpoint_step"] == 1
    assert cfg["val_dataloader"]["dataset"]["ann_file"] == str(
        data / "annotations" / "val.json"
    )


def test_rtdetr_best_checkpoint_is_selected_from_val(tmp_path, monkeypatch):
    from liverdet.rtdetr import train_rtdetr
    official = tmp_path / "rtdetr_pytorch"
    base = official / "configs" / "rtdetr" / "rtdetr_r18vd_6x_coco.yml"
    base.parent.mkdir(parents=True)
    base.write_text("task: detection\n")
    data = _dataset(tmp_path)
    out = tmp_path / "runs"
    def fake_official_training(*args, **kwargs):
        out.mkdir(parents=True, exist_ok=True)
        for epoch in (0, 1):
            (out / f"checkpoint{epoch:04d}.pth").write_text("checkpoint")
        records = [
            {"epoch": 0, "test_coco_eval_bbox": [0.25]},
            {"epoch": 1, "test_coco_eval_bbox": [0.15]},
        ]
        (out / "log.txt").write_text("\n".join(json.dumps(x) for x in records))
    monkeypatch.setattr("liverdet.rtdetr.subprocess.run", fake_official_training)
    chosen = train_rtdetr(str(official), str(data), str(out), epochs=100)
    assert chosen == out / "checkpoint0000.pth"
