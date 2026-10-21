"""Discover registered AP/PVP image volumes and corresponding NIfTI pseudo labels.

Supported research-server layout (NCCT is in EACH patient folder):

    image_root/<patient_id>/ap_registered.nii.gz
    image_root/<patient_id>/pvp_registered.nii.gz
    image_root/<patient_id>/ncct_zcrop.nii.gz

    label_ap_root/<patient_id>.nii.gz       OR
    label_ap_root/<patient_id>/*.nii.gz
    label_pvp_root/<patient_id>.nii.gz       OR
    label_pvp_root/<patient_id>/*.nii.gz

Flat label filenames can also end in recognized phase/label suffixes, e.g.
A0_1_ap_lesion_mask.nii.gz. Ambiguous/missing matches cause an error rather
than silently assigning one patient's tumor mask to another's image.
"""
from __future__ import annotations

from pathlib import Path
import re

import pandas as pd

from .common import save_csv

# Only these files represent actual CT image volumes. The *mask_registered*
# files inside image_root are NOT teacher-generated lesion pseudo labels.
PHASE_FILES = {
    "ap": ("ap_registered.nii.gz", "ap_registered.nii"),
    "pvp": ("pvp_registered.nii.gz", "pvp_registered.nii"),
}
NCCT_FILES = ("ncct_zcrop.nii.gz", "ncct_zcrop.nii")

# Phase-specific label roots are given, so 'ap' and 'pvp' are optional
# filename suffix tokens, not sources from which patient IDs are inferred.
LABEL_SUFFIXES = frozenset({
    "ap", "pvp", "mask", "registered", "reg", "lesion", "tumor",
    "seg", "segmentation", "pseudo", "label", "labels",
    "pred", "prediction", "predictions",
})


def nii_files(root: str | Path) -> list[Path]:
    root = Path(root)
    if not root.is_dir():
        raise NotADirectoryError(f"Missing NIfTI directory: {root}")
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and (p.name.lower().endswith(".nii.gz") or p.name.lower().endswith(".nii"))
    )


def strip_nii(name: str) -> str:
    name_lower = name.lower()
    if name_lower.endswith(".nii.gz"):
        return name[:-7]
    if name_lower.endswith(".nii"):
        return name[:-4]
    raise ValueError(f"Not a NIfTI filename: {name}")


def _find_exact_file(folder: Path, filenames: tuple[str, ...], *, required: bool = True) -> Path | None:
    matches = [folder / name for name in filenames if (folder / name).is_file()]
    if len(matches) > 1:
        raise ValueError(f"Both .nii and .nii.gz found in {folder}: {matches}")
    if not matches and required:
        raise FileNotFoundError(f"Missing {' or '.join(filenames)} in {folder}")
    return matches[0] if matches else None


def _flat_filename_matches_patient(path: Path, patient_id: str) -> bool:
    """No fuzzy prefix matching: A0_1 and A0_10 must never collide."""
    stem = strip_nii(path.name)
    if stem == patient_id:
        return True
    if not stem.startswith(patient_id):
        return False
    tail = stem[len(patient_id):]
    if not tail or tail[0] not in "_.-":
        return False
    suffix_tokens = [t.lower() for t in re.split(r"[_\-.]+", tail) if t]
    return bool(suffix_tokens) and all(t in LABEL_SUFFIXES for t in suffix_tokens)


def _label_candidates(patient_id: str, label_root: Path, all_labels: list[Path]) -> list[Path]:
    matches = []
    for file in all_labels:
        rel = file.relative_to(label_root)
        if len(rel.parts) == 1:
            if _flat_filename_matches_patient(file, patient_id):
                matches.append(file)
        elif rel.parts[0] == patient_id:
            # Nested patient folders must contain exactly one NIfTI label
            # in each phase-specific root; multiple masks are ambiguous.
            matches.append(file)
    return matches


def scan_cases(
    image_root: str,
    label_ap_root: str,
    label_pvp_root: str,
    out: str,
    ncct_root: str | None = None,
) -> pd.DataFrame:
    """Generate cases.csv in the patient's folder layout.

    ncct_root is deprecated for this layout; NCCT is discovered as
    <image_root>/<patient_id>/ncct_zcrop.nii.gz. If provided, ncct_root
    must equal image_root, to avoid accidentally overriding this source.
    """
    root = Path(image_root)
    if not root.is_dir():
        raise NotADirectoryError(f"Missing registered patient root: {root}")
    if ncct_root and Path(ncct_root).resolve() != root.resolve():
        raise ValueError(
            "NCCT is stored with AP/PVP per patient. Do not pass a separate --ncct-root; "
            "ncct_zcrop.nii.gz is discovered automatically from image_root."
        )

    label_roots = {"ap": Path(label_ap_root), "pvp": Path(label_pvp_root)}
    label_files = {phase: nii_files(folder) for phase, folder in label_roots.items()}
    if not all(label_files.values()):
        raise ValueError("One or both pseudo-label directories contain no NIfTI files.")

    patient_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    if not patient_dirs:
        raise ValueError(f"No patient subfolders in {root}")
    rows = []
    for folder in patient_dirs:
        pid = folder.name  # A0_1, etc. Preserve exact case and underscores.
        phase_images = {
            phase: _find_exact_file(folder, filenames, required=False)
            for phase, filenames in PHASE_FILES.items()
        }
        if not any(phase_images.values()):
            # Ignore non-patient folders, but never accept a mask as an image.
            continue
        missing_phases = [phase for phase, path in phase_images.items() if path is None]
        if missing_phases:
            raise FileNotFoundError(f"{pid}: missing registered phase(s): {missing_phases}")
        ncct = _find_exact_file(folder, NCCT_FILES, required=True)
        for phase, image_path in phase_images.items():
            assert image_path is not None
            matches = _label_candidates(pid, label_roots[phase], label_files[phase])
            if len(matches) != 1:
                raise ValueError(
                    f"{pid}/{phase}: expected exactly one tumor pseudo-label NIfTI under "
                    f"{label_roots[phase]}, got {len(matches)}: {matches[:8]}. "
                    "Check flat filenames or nested patient folders; "
                    "do NOT substitute ap_mask_registered/pvp_mask_registered."
                )
            rows.append({
                "patient_id": pid,
                "phase": phase,
                "image_path": str(image_path.resolve()),
                "mask_path": str(matches[0].resolve()),
                "ncct_path": str(ncct.resolve()),
                "label_source": "pseudo",
            })

    if not rows:
        raise ValueError(
            "No complete patient AP/PVP folders found. Expected e.g. "
            "image_root/A0_1/ap_registered.nii.gz and pvp_registered.nii.gz."
        )
    frame = pd.DataFrame(rows).sort_values(["patient_id", "phase"]).reset_index(drop=True)
    save_csv(frame, out)
    return frame
