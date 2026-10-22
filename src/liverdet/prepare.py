"""Convert NIfTI image + aligned 3-D NIfTI mask into auditable 2-D samples.

Coordinate convention:
    NIfTI arr[x,y,k] -> PNG arr[x,y,k].T (rows y, columns x).
    XYXY bbox coordinates in PNG pixel space.
We never interpret missing pseudo detections as verified negatives.
By default, pseudo lesions receive only a small 2D component-area filter.
Same-phase liver masking is available only with an explicit opt-in.
Manual ground truth is preserved without component-size filtering.
"""
from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import nibabel as nib
import pandas as pd
from scipy import ndimage as ndi

from .common import (
    REQUIRED_CASE_COLUMNS, boxes_dump, read_csv, save_csv, save_png,
    window_uint8, check_no_patient_overlap
)


def boxes_from_mask(slice_mask: np.ndarray, min_component_pixels: int = 4) -> list[list[float]]:
    """Mask is a 2-D PNG-oriented array: rows=y, cols=x. Connected 8-neighborhood."""
    binary = np.asarray(slice_mask) > 0
    connected, count = ndi.label(binary, structure=np.ones((3, 3), dtype=int))
    boxes = []
    for index in range(1, count + 1):
        ys, xs = np.nonzero(connected == index)
        if len(xs) < min_component_pixels:
            continue
        boxes.append([float(xs.min()), float(ys.min()),
                      float(xs.max() + 1), float(ys.max() + 1)])
    return boxes


def _assign_splits(cases: pd.DataFrame, seed: int, train_fraction: float,
                   val_fraction: float, split_file: str | None) -> pd.DataFrame:
    if split_file:
        mapping = read_csv(split_file, ["patient_id", "split"])
        if mapping.patient_id.duplicated().any():
            raise ValueError("Duplicate patient_id in split CSV")
        cases = cases.drop(columns=["split"], errors="ignore").merge(mapping, on="patient_id", how="left")
        if cases.split.isna().any():
            raise ValueError("Some patients have no split")
    elif "split" not in cases or (cases.split.astype(str).str.len() == 0).any():
        if not (0 < train_fraction < 1 and 0 < val_fraction < 1 - train_fraction):
            raise ValueError("Invalid split fractions")
        patients = sorted(cases.patient_id.unique())
        rng = np.random.default_rng(seed)
        rng.shuffle(patients)
        n = len(patients)
        n_train = int(np.floor(n * train_fraction))
        n_val = int(np.floor(n * val_fraction))
        if n_train == 0 or n_val == 0 or n - n_train - n_val == 0:
            raise ValueError("Need enough patients for train/val/test")
        mapping = {
            pid: "train" if i < n_train else ("val" if i < n_train+n_val else "test")
            for i, pid in enumerate(patients)
        }
        cases["split"] = cases.patient_id.map(mapping)
    if not set(cases.split.unique()).issubset({"train", "val", "test"}):
        raise ValueError("Invalid split value")
    check_no_patient_overlap(cases)
    return cases


def prepare_slices(cases_csv: str, workdir: str, *,
                   split_csv: str | None = None, seed: int = 42,
                   train_fraction: float = .7, val_fraction: float = .15,
                   hu_min: float = -100, hu_max: float = 300,
                   min_component_pixels: int = 4,
                   trusted_negatives_csv: str | None = None,
                   filter_pseudo_liver_mask: bool = False) -> pd.DataFrame:
    """Write real PNGs and slices.csv; all positive slices; negative only if verified.

    trusted_negatives_csv columns: patient_id,phase,slice_index. For manual
    masks empty slices are also accepted as verified negative.
    """
    if min_component_pixels < 1:
        raise ValueError("min_component_pixels must be >= 1")
    cases = read_csv(cases_csv, REQUIRED_CASE_COLUMNS)
    if cases.duplicated(["patient_id", "phase"]).any():
        raise ValueError("Duplicate patient-phase in cases.csv")
    cases = _assign_splits(cases, seed, train_fraction, val_fraction, split_csv)
    if "label_source" not in cases:
        cases["label_source"] = "pseudo"
    if "ncct_path" not in cases:
        cases["ncct_path"] = ""
    if not set(cases.phase.unique()).issubset({"ap", "pvp"}):
        raise ValueError("Only ap and pvp phases accepted")
    save_csv(cases, Path(workdir)/"manifests"/"cases_with_splits.csv")
    verified = set()
    if trusted_negatives_csv:
        neg = read_csv(trusted_negatives_csv, ["patient_id", "phase", "slice_index"])
        verified = {(str(r.patient_id), str(r.phase), int(r.slice_index)) for r in neg.itertuples(index=False)}

    rows = []
    liver_qc_rows = []
    component_qc_rows = []
    for case in cases.itertuples(index=False):
        case_label = str(case.label_source)
        if case_label not in ("manual", "pseudo", "verified_negative"):
            raise ValueError(f"Invalid label_source {case_label}")
        image = nib.load(case.image_path)
        if len(image.shape) != 3:
            raise ValueError(f"Expected 3-D NIfTI: {case.image_path}")
        mask = None
        if str(case.mask_path).strip():
            mask = nib.load(case.mask_path)
            if image.shape != mask.shape or not np.allclose(image.affine, mask.affine, rtol=0, atol=1e-3):
                raise ValueError(f"Image/mask geometry mismatch: {case.image_path} vs {case.mask_path}")
        elif case_label != "verified_negative":
            raise ValueError(f"Missing mask not trusted as negative: {case.patient_id}/{case.phase}")

        # Organ masks are NEVER used as lesion ground truth. Liver clipping
        # is opt-in only; the default keeps all pseudo lesions, regardless
        # of whether they lie inside a liver segmentation.
        liver = None
        liver_path = Path(case.image_path).parent / f"{case.phase}_mask_registered.nii.gz"
        before_voxels = after_voxels = 0
        before_slices = after_slices = 0
        if case_label == "pseudo" and filter_pseudo_liver_mask:
            if not liver_path.is_file():
                raise FileNotFoundError(
                    f"Missing {case.phase.upper()} liver mask for pseudo training case "
                    f"{case.patient_id}: {liver_path}"
                )
            liver = nib.load(str(liver_path))
            if len(liver.shape) != 3 or liver.shape != image.shape or not np.allclose(
                liver.affine, image.affine, rtol=0, atol=1e-3
            ):
                raise ValueError(
                    f"Liver mask geometry mismatch: {liver_path} vs {case.image_path}"
                )

        total_components = kept_components = positive_slices_before = positive_slices_after = 0
        for k in range(image.shape[2]):
            lesion_slice = None if mask is None else np.asarray(mask.dataobj[:, :, k]) > 0
            if liver is not None:
                # Logical 3D intersection performed slice-wise on the same voxel grid.
                liver_data = np.asarray(liver.dataobj[:, :, k])
                if not np.isfinite(liver_data).all():
                    raise ValueError(f"Nonfinite liver mask: {liver_path}, slice={k}")
                positive = bool(lesion_slice.any())
                before_voxels += int(lesion_slice.sum())
                before_slices += int(positive)
                lesion_slice = lesion_slice & (liver_data > 0)
                after_voxels += int(lesion_slice.sum())
                after_slices += int(lesion_slice.any())
            if lesion_slice is None or not lesion_slice.any():
                boxes = []
            elif case_label == "pseudo":
                # Only teacher pseudo labels receive an area threshold.
                # Manual Val/Test lesions must not be discarded, however small.
                oriented = lesion_slice.T
                total_components += len(boxes_from_mask(oriented, 1))
                positive_slices_before += 1
                boxes = boxes_from_mask(oriented, min_component_pixels)
                kept_components += len(boxes)
                positive_slices_after += int(bool(boxes))
            else:
                boxes = boxes_from_mask(lesion_slice.T, 1)
            confirmed_negative = (
                case_label in ("manual", "verified_negative")
                or (str(case.patient_id), str(case.phase), k) in verified
            )
            if not boxes and not confirmed_negative:
                continue  # Unknown, NOT a negative training label.
            sample_id = f"{case.patient_id}_{case.phase}_{k:04d}"
            out = Path(workdir)/"data"/"real"/case.phase/f"{sample_id}.png"
            if not out.exists():
                save_png(window_uint8(np.asarray(image.dataobj[:, :, k]).T, hu_min, hu_max), out)
            rows.append(dict(
                sample_id=sample_id, patient_id=case.patient_id, phase=case.phase,
                split=case.split, slice_index=k, real_path=str(out.resolve()),
                boxes_real=boxes_dump(boxes), label_source=case_label if boxes else "manual" if confirmed_negative and case_label == "manual" else "verified_negative",
                source_volume=str(Path(case.image_path).resolve()),
                liver_mask_path=str(liver_path.resolve()) if liver is not None else "",
                ncct_volume=str(Path(case.ncct_path).resolve()) if str(case.ncct_path).strip() else "",
                ncct_slice="", ncct_path="", boxes_ncct="[]", match_nmi="",
                match_offset_mm="", match_qc="unmatched", syn_path="", boxes_syn="[]"
            ))
        if case_label == "pseudo":
            component_qc_rows.append(dict(
                patient_id=case.patient_id, phase=case.phase, split=case.split,
                pseudo_mask_path=str(Path(case.mask_path).resolve()),
                min_component_pixels=min_component_pixels,
                components_before=total_components,
                components_kept=kept_components,
                components_removed=total_components-kept_components,
                positive_slices_before=positive_slices_before,
                positive_slices_after=positive_slices_after,
                liver_filter_enabled=filter_pseudo_liver_mask,
            ))
        if liver is not None:
            liver_qc_rows.append(dict(
                patient_id=case.patient_id, phase=case.phase, split=case.split,
                pseudo_mask_path=str(Path(case.mask_path).resolve()),
                liver_mask_path=str(liver_path.resolve()),
                pseudo_voxels_before=before_voxels,
                pseudo_voxels_inside_liver=after_voxels,
                pseudo_voxels_removed=before_voxels - after_voxels,
                positive_slices_before=before_slices,
                positive_slices_after=after_slices,
            ))
    if component_qc_rows:
        qc_path = Path(workdir) / "manifests" / "small_component_filter_qc.csv"
        save_csv(pd.DataFrame(component_qc_rows), qc_path)
        print(f"Pseudo-label small-component QC: {qc_path}")
    if liver_qc_rows:
        qc_path = Path(workdir) / "manifests" / "liver_filter_qc.csv"
        save_csv(pd.DataFrame(liver_qc_rows), qc_path)
        print(f"Pseudo-label liver mask filtering QC: {qc_path}")
    if not rows:
        raise ValueError("No verified slices. Pseudo masks with no predictions cannot create negatives.")
    frame = pd.DataFrame(rows)
    check_no_patient_overlap(frame)
    save_csv(frame, Path(workdir)/"manifests"/"slices.csv")
    return frame
