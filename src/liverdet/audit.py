"""Read-only audit of actual AP/PVP pseudo-label paths and NIfTI geometry.

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
        image, mask, ncct = None, None, None
        shape_image = shape_mask = shape_ncct = ""
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
