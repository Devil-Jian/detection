"""Read-only audit of AP/PVP lesion labels and phase-specific liver mask geometry.

Audit runs before slice extraction, writes a CSV even when errors exist, and
NEVER assumes pseudo-mask zeros are verified negative slices.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from .common import read_csv, save_csv

EXPECTED = {"patient_id", "phase", "image_path", "mask_path", "ncct_path"}


def _format_shape(shape) -> str:
    return "x".join(map(str, shape))


def _load_3d(path: str) -> nib.spatialimages.SpatialImage:
    if not path or not Path(path).is_file():
        raise FileNotFoundError(f"Missing file: {path!r}")
    volume = nib.load(path)
    if len(volume.shape) != 3:
        raise ValueError(f"Expected 3D NIfTI at {path}, got shape {volume.shape}")
    if not np.all(np.isfinite(volume.affine)):
        raise ValueError(f"Nonfinite affine at {path}")
    return volume


def audit_cases(cases_csv: str, output_csv: str, *,
                inspect_masks: bool = False,
                affine_tolerance: float = 1e-3,
                fail_on_error: bool = True) -> pd.DataFrame:
    """Inspect paths, image/mask geometry, and optional pseudo-mask contents.

    inspect_masks=True loads each full 3D pseudo-mask and may take time;
    shape/affine checks use only NIfTI metadata.
    """
    cases = read_csv(cases_csv, EXPECTED)
    if not set(cases.phase.astype(str)).issubset({"ap", "pvp"}):
        raise ValueError("Only AP and PVP phases may be audited.")
    if cases.duplicated(["patient_id", "phase"]).any():
        raise ValueError("Duplicate patient/phase in cases.csv")
    counts = cases.groupby("patient_id").phase.nunique()
    bad_patients = set(counts[counts != 2].index.astype(str))

    results = []
    for row in cases.itertuples(index=False):
        problems = []
        warnings = []
        pid, phase = str(row.patient_id), str(row.phase)
        if pid in bad_patients:
            problems.append("incomplete_phase_pair")
        image, mask, ncct, liver = None, None, None, None
        is_pseudo = str(getattr(row, "label_source", "")).strip() == "pseudo"
        liver_path = str(Path(str(row.image_path)).parent / f"{phase}_mask_registered.nii.gz") if is_pseudo else ""
        shape_image = shape_mask = shape_ncct = liver_shape = ""
        liver_affine_diff = outside_liver_voxels = inside_liver_voxels = ""
        affine_max_diff = ""
        nonzero_voxels = positive_slices = ""
        mask_min = mask_max = ""

        for field, path in (("real", str(row.image_path)),
                            ("pseudo_mask", str(row.mask_path)),
                            ("ncct", str(row.ncct_path))):
            try:
                volume = _load_3d(path)
                if field == "real":
                    image = volume
                    shape_image = _format_shape(volume.shape)
                elif field == "pseudo_mask":
                    mask = volume
                    shape_mask = _format_shape(volume.shape)
                else:
                    ncct = volume
                    shape_ncct = _format_shape(volume.shape)
            except Exception as exc:
                problems.append(f"{field}:{type(exc).__name__}:{exc}")

        # Check the same-phase registered liver mask ONLY for pseudo cases.
        # Ground-truth Val/Test lesions must not be liver-mask filtered.
        if is_pseudo:
            try:
                liver = _load_3d(liver_path)
                liver_shape = _format_shape(liver.shape)
                if image is not None:
                    if image.shape != liver.shape:
                        problems.append("real_vs_liver_shape_mismatch")
                    liver_affine_diff = float(np.max(np.abs(image.affine - liver.affine)))
                    if liver_affine_diff > affine_tolerance:
                        problems.append("real_vs_liver_affine_mismatch")
            except Exception as exc:
                problems.append(f"liver_mask:{type(exc).__name__}:{exc}")

        if image is not None and mask is not None:
            if image.shape != mask.shape:
                problems.append("real_vs_mask_shape_mismatch")
            affine_max_diff = float(np.max(np.abs(image.affine - mask.affine)))
            if affine_max_diff > affine_tolerance:
                problems.append("real_vs_mask_affine_mismatch")
        if image is not None and ncct is not None:
            # NCCT is fixed reference; full image/mask geometry equality is
            # NOT required, but the matching stage must check anatomy later.
            if not np.allclose(image.affine, ncct.affine, atol=affine_tolerance, rtol=0):
                warnings.append("real_ncct_affine_differs_requires_matching_qc")
            if image.shape != ncct.shape:
                warnings.append("real_ncct_shape_differs_requires_matching_qc")

        if inspect_masks and mask is not None:
            try:
                data = np.asarray(mask.dataobj)
                finite = np.isfinite(data)
                if not finite.all():
                    problems.append("mask_nonfinite")
                clean = np.where(finite, data, 0)
                nz = clean > 0
                nonzero_voxels = int(np.count_nonzero(nz))
                positive_slices = int(np.count_nonzero(nz.any(axis=(0, 1))))
                mask_min, mask_max = float(clean.min()), float(clean.max())
                if is_pseudo and liver is not None and mask.shape == liver.shape:
                    liver_data = np.asarray(liver.dataobj)
                    if not np.isfinite(liver_data).all():
                        problems.append("liver_mask_nonfinite")
                    inside = nz & (liver_data > 0)
                    inside_liver_voxels = int(np.count_nonzero(inside))
                    outside_liver_voxels = int(nonzero_voxels - inside_liver_voxels)
                    if outside_liver_voxels:
                        warnings.append("pseudo_lesions_outside_liver_will_be_filtered")
                if nonzero_voxels == 0:
                    warnings.append("pseudo_mask_empty_not_verified_negative")
                if mask_min < 0:
                    warnings.append("negative_mask_values_check_label_semantics")
                if mask_max > 1:
                    warnings.append("mask_values_gt_1_check_binary_or_instance_mask")
                if (mask_max > 0 and mask_max < 1):
                    warnings.append("floating_pseudo_probabilities_need_threshold")
            except Exception as exc:
                problems.append(f"mask_read:{type(exc).__name__}:{exc}")

        results.append({
            "patient_id": pid, "phase": phase,
            "image_path": str(row.image_path),
            "mask_path": str(row.mask_path),
            "ncct_path": str(row.ncct_path),
            "liver_mask_path": liver_path, "liver_shape": liver_shape,
            "liver_affine_max_abs_diff": liver_affine_diff,
            "pseudo_voxels_inside_liver": inside_liver_voxels,
            "pseudo_voxels_outside_liver": outside_liver_voxels,
            "real_shape": shape_image, "mask_shape": shape_mask,
            "ncct_shape": shape_ncct, "affine_max_abs_diff": affine_max_diff,
            "pseudo_nonzero_voxels": nonzero_voxels,
            "pseudo_positive_slices": positive_slices,
            "pseudo_min": mask_min, "pseudo_max": mask_max,
            "status": "error" if problems else ("warning" if warnings else "ok"),
            "errors": "; ".join(problems), "warnings": "; ".join(warnings),
        })
    report = pd.DataFrame(results)
    save_csv(report, output_csv)
    totals = Counter(report.status)
    print(f"NIfTI audit: cases={len(report)}, patients={cases.patient_id.nunique()}, "
          f"ok={totals['ok']}, warning={totals['warning']}, error={totals['error']}.")
    print(f"Report: {Path(output_csv).resolve()}")
    if totals["error"] and fail_on_error:
        raise ValueError(
            f"{totals['error']} NIfTI cases failed preflight. "
            f"Fix problems in {output_csv} before 'liverdet prepare'.")
    return report
