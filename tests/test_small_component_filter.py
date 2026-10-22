"""Pseudo-only small-component filtering without any liver-mask dependency."""
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from liverdet.audit import audit_cases
from liverdet.common import boxes_load
from liverdet.prepare import prepare_slices


def _nii(path: Path, data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data.astype(np.float32), np.eye(4)), str(path))


def _cases(tmp_path: Path, *, large_pseudo: bool = True) -> Path:
    rows = []
    for pid, split, label in (
        ("P_PSEUDO", "train", "pseudo"),
        ("P_VAL", "val", "manual"),
        ("P_TEST", "test", "manual"),
    ):
        for phase in ("ap", "pvp"):
            volume = np.zeros((18, 18, 3))
            lesion = np.zeros_like(volume)
            lesion[1, 1, 1] = 1  # one-pixel noise for pseudo; small GT lesion for manual
            if label == "pseudo" and large_pseudo:
                lesion[8:11, 9:12, 1] = 1  # true pseudo candidate, area = 9
            image_path = tmp_path / pid / f"{phase}_registered.nii.gz"
            mask_path = tmp_path / "teacher" / phase / f"{pid}.nii.gz"
            if label == "manual":
                mask_path = tmp_path / pid / f"{phase}_tumor_mask_registered.nii.gz"
            _nii(image_path, volume)
            _nii(mask_path, lesion)
            _nii(tmp_path / pid / "ncct_zcrop.nii.gz", volume)
            # Intentionally do NOT create any liver mask.
            rows.append(
                dict(patient_id=pid, phase=phase, split=split, image_path=str(image_path),
                     mask_path=str(mask_path), ncct_path=str(tmp_path / pid / "ncct_zcrop.nii.gz"),
                     label_source=label)
            )
    csv = tmp_path / "cases.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    return csv


def test_no_liver_required_pseudo_noise_removed_gt_intact(tmp_path):
    csv = _cases(tmp_path)
    audit = audit_cases(str(csv), str(tmp_path / "audit.csv"), inspect_masks=True)
    assert set(audit.status) == {"ok"}
    assert audit.liver_mask_path.eq("").all()

    output = tmp_path / "output"
    samples = prepare_slices(str(csv), str(output), min_component_pixels=4)
    pseudo = samples[(samples.split == "train") & (samples.phase == "ap")]
    assert len(pseudo) == 1
    assert pseudo.iloc[0].slice_index == 1
    assert boxes_load(pseudo.iloc[0].boxes_real) == [[8.0, 9.0, 11.0, 12.0]]
    assert samples.liver_mask_path.eq("").all()
    for split in ("val", "test"):
        manual = samples[(samples.split == split) &
                         (samples.phase == "ap") &
                         (samples.slice_index == 1)]
        assert boxes_load(manual.iloc[0].boxes_real) == [[1.0, 1.0, 2.0, 2.0]]
    qc = pd.read_csv(output / "manifests" / "small_component_filter_qc.csv")
    assert len(qc) == 2
    assert set(qc.components_before) == {2}
    assert set(qc.components_kept) == {1}
    assert set(qc.components_removed) == {1}
    assert set(qc.liver_filter_enabled.astype(str)) == {"False"}
    assert not (output / "manifests" / "liver_filter_qc.csv").exists()


def test_all_pseudo_components_too_small_are_unknown_not_negatives(tmp_path):
    csv = _cases(tmp_path, large_pseudo=False)
    out = tmp_path / "out"
    samples = prepare_slices(str(csv), str(out), min_component_pixels=4)
    assert (samples.split == "train").sum() == 0
    assert not samples[(samples.split == "train") &
                       (samples.label_source == "verified_negative")].shape[0]
    qc = pd.read_csv(out / "manifests" / "small_component_filter_qc.csv")
    assert set(qc.components_removed) == {1}
    assert set(qc.positive_slices_after) == {0}


def test_one_pixel_pseudo_component_remains_when_min_is_one(tmp_path):
    csv = _cases(tmp_path, large_pseudo=False)
    samples = prepare_slices(str(csv), str(tmp_path / "out"), min_component_pixels=1)
    pseudo = samples[(samples.split == "train") & (samples.phase == "ap")]
    assert len(pseudo) == 1
    assert boxes_load(pseudo.iloc[0].boxes_real) == [[1.0, 1.0, 2.0, 2.0]]


def test_invalid_area_threshold_rejected(tmp_path):
    csv = _cases(tmp_path)
    with pytest.raises(ValueError, match="min_component_pixels"):
        prepare_slices(str(csv), str(tmp_path / "out"), min_component_pixels=0)
