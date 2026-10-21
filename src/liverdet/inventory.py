"""Discover registered NIfTI image volumes and pseudo-mask volumes.

The scanner is conservative: ambiguous patient or label matches cause errors. The
resulting cases.csv is always reviewable/editable before preparing 2-D slices.
"""
from __future__ import annotations

import re
from pathlib import Path
import pandas as pd

from .common import PHASES, save_csv

TOKEN = {
    "ap": re.compile(r"(?i)(?:^|[^a-z0-9])(ap|arterial)(?:[^a-z0-9]|$)"),
    "pvp": re.compile(r"(?i)(?:^|[^a-z0-9])(pvp|portal|venous)(?:[^a-z0-9]|$)"),
    "ncct": re.compile(r"(?i)(?:^|[^a-z0-9])(ncct|noncontrast|non_contrast|plain)(?:[^a-z0-9]|$)"),
}
SUFFIX = re.compile(r"(?i)(?:[_\-. ](?:registered|reg|image|img|mask|label|seg|arterial|portal|venous|ap|pvp|ncct|noncontrast|plain))+$")


def nii_files(root: str | Path) -> list[Path]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(root)
    return sorted(p for p in root.rglob("*") if p.is_file() and (p.name.endswith(".nii.gz") or p.suffix == ".nii"))


def strip_nii(name: str) -> str:
    return name[:-7] if name.endswith(".nii.gz") else Path(name).stem


def infer_patient(path: Path, root: Path) -> str:
    rel = path.relative_to(root)
    stem = strip_nii(path.name)
    if len(rel.parts) >= 2:
        # Patient-per-directory convention; leave nonstandard layouts to manual CSV.
        return rel.parts[0]
    cleaned = SUFFIX.sub("", stem)
    if cleaned == stem:
        cleaned = re.sub(r"(?i)(?:[_\-. ](?:ap|pvp|arterial|portal|venous|ncct|plain))(?=[_\-. ]|$)", "", stem).strip("_-. ")
    return cleaned


def phase_of(path: Path) -> str | None:
    value = "/".join(path.parts[-3:])
    matches = [k for k, pattern in TOKEN.items() if pattern.search(value)]
    if len(matches) > 1:
        raise ValueError(f"Ambiguous phase tokens in {path}: {matches}")
    return matches[0] if matches else None


def scan_cases(image_root: str, label_ap_root: str, label_pvp_root: str,
               out: str, ncct_root: str | None = None) -> pd.DataFrame:
    root = Path(image_root)
    images = {}
    for path in nii_files(root):
        phase = phase_of(path)
        if phase not in PHASES:
            continue
        key = (infer_patient(path, root), phase)
        if key in images:
            raise ValueError(f"Multiple image volumes for {key}: {images[key]}, {path}. Create cases.csv manually.")
        images[key] = path
    if not images:
        raise ValueError("No AP/PVP volumes discovered. Supply cases.csv with explicit paths.")

    labels = {}
    for phase, folder in (("ap", label_ap_root), ("pvp", label_pvp_root)):
        label_root = Path(folder)
        for path in nii_files(label_root):
            key = (infer_patient(path, label_root), phase)
            if key in labels:
                raise ValueError(f"Multiple label masks for {key}; edit cases.csv manually.")
            labels[key] = path
    ncct = {}
    if ncct_root:
        ncct_base = Path(ncct_root)
        for path in nii_files(ncct_base):
            pid = infer_patient(path, ncct_base)
            if pid in ncct:
                raise ValueError(f"Duplicate NCCT for patient {pid}: {path}")
            ncct[pid] = path

    rows = []
    for (pid, phase), image in sorted(images.items()):
        label = labels.get((pid, phase))
        if label is None:
            raise ValueError(f"No NIfTI mask for patient={pid}, phase={phase}. "
                             "If labels are JSON/TXT, convert to 3-D mask or prepare cases.csv manually.")
        rows.append(dict(patient_id=pid, phase=phase, image_path=str(image),
                         mask_path=str(label), ncct_path=str(ncct.get(pid, "")),
                         label_source="pseudo"))
    frame = pd.DataFrame(rows)
    save_csv(frame, out)
    return frame
