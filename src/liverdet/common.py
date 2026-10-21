"""Shared manifest and image/box helpers. All boxes are XYXY pixels."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image

PHASES = ("ap", "pvp")
GROUPS = ("ncct", "cect_real", "cect_syn", "cect_real_aug", "cect_real_syn")
REQUIRED_CASE_COLUMNS = {"patient_id", "phase", "image_path", "mask_path"}
REQUIRED_SLICE_COLUMNS = {"sample_id", "patient_id", "phase", "split", "real_path", "boxes_real", "label_source"}


def read_csv(path: str | Path, required: Iterable[str] = ()) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"patient_id": str, "sample_id": str}, keep_default_na=False)
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing required columns: {sorted(missing)}")
    return frame


def save_csv(frame: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def boxes_load(value: str | list) -> list[list[float]]:
    if isinstance(value, str):
        if not value.strip():
            return []
        value = json.loads(value)
    result = []
    for box in value:
        if len(box) != 4:
            raise ValueError(f"Invalid xyxy bbox {box}")
        x1, y1, x2, y2 = [float(v) for v in box]
        if not all(math.isfinite(v) for v in (x1, y1, x2, y2)) or x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid xyxy bbox {box}")
        result.append([x1, y1, x2, y2])
    return result


def boxes_dump(boxes: list[list[float]]) -> str:
    return json.dumps(boxes, separators=(",", ":"))


def window_uint8(array: np.ndarray, lo: float, hi: float) -> np.ndarray:
    if hi <= lo:
        raise ValueError("HU window max must exceed min")
    array = np.asarray(array, dtype=np.float32)
    array = np.nan_to_num(array, nan=lo, posinf=hi, neginf=lo)
    return np.rint((np.clip(array, lo, hi) - lo) / (hi - lo) * 255).astype(np.uint8)


def save_png(image: np.ndarray, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image, mode="L").save(path)


def image_size(path: str | Path) -> tuple[int, int]:
    with Image.open(path) as im:
        return im.size


def clip_boxes(boxes: list[list[float]], width: int, height: int, min_area: float = 1) -> list[list[float]]:
    out = []
    for x1, y1, x2, y2 in boxes:
        x1, x2 = float(np.clip(x1, 0, width)), float(np.clip(x2, 0, width))
        y1, y2 = float(np.clip(y1, 0, height)), float(np.clip(y2, 0, height))
        if (x2 - x1) * (y2 - y1) >= min_area and x1 < x2 and y1 < y2:
            out.append([x1, y1, x2, y2])
    return out


def check_no_patient_overlap(frame: pd.DataFrame) -> None:
    groups = frame.groupby("patient_id")["split"].nunique()
    leaked = groups[groups > 1]
    if not leaked.empty:
        raise ValueError(f"Patient leakage: {list(leaked.index[:20])}")


def assert_manual_eval(frame: pd.DataFrame, allow_pseudo: bool = False) -> None:
    if allow_pseudo:
        return
    eval_rows = frame[frame.split.isin(["val", "test"])]
    bad = eval_rows[~eval_rows.label_source.isin(["manual", "verified_negative"])]
    if not bad.empty:
        raise ValueError(
            f"{len(bad)} validation/test slices are not manually annotated. "
            "Use manually reviewed ground truth for paper metrics; "
            "--allow-pseudo-eval is ONLY for an engineering smoke test."
        )
