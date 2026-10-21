"""Regression tests for user's registered patient folder layout.

The folder-local 'ap_mask_registered.nii.gz' is deliberately a distinct,
wrong-shaped mask so picking it as a lesion pseudo-label would fail later.
"""
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from liverdet.inventory import scan_cases
from liverdet.prepare import prepare_slices


def write_nii(path: Path, shape=(12, 14, 3)):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.zeros(shape, dtype=np.float32)
    data[3:8, 4:10, 1] = 1.0
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(path))


def make_registered_root(root: Path, pid: str):
    patient = root / pid
    for name in (
        "ap_registered.nii.gz",
        "pvp_registered.nii.gz",
        "ncct_zcrop.nii.gz",
        "ap_mask_registered.nii.gz",  # anatomical mask, must never be used as lesion label
        "pvp_mask_registered.nii.gz",
        "ncct_mask_zcrop.nii.gz",
        "dp_registered.nii.gz",
        "dp_mask_registered.nii.gz",
    ):
        write_nii(patient / name)


def test_flat_masks_ncct_in_patient_folder(tmp_path):
    root = tmp_path / "dataset" / "gx_all_registered"
    labels = tmp_path / "detection"
    for pid in ("A0_1", "A0_10", "B_1"):
        make_registered_root(root, pid)
        write_nii(labels / "ap" / f"{pid}.nii.gz")
        write_nii(labels / "pvp" / f"{pid}_pvp_lesion_mask.nii.gz")

    out = tmp_path / "cases.csv"
    cases = scan_cases(str(root), str(labels / "ap"), str(labels / "pvp"), str(out))
    assert len(cases) == 6
    assert out.is_file()
    for row in cases.itertuples(index=False):
        assert row.patient_id in ("A0_1", "A0_10", "B_1")
        assert row.image_path.endswith(f"{row.phase}_registered.nii.gz")
        assert row.ncct_path.endswith(f"{row.patient_id}/ncct_zcrop.nii.gz")
        assert str(labels / row.phase) in row.mask_path
        assert "mask_registered" not in row.mask_path

    # Verify the resulting CSV is consumable by the existing NIfTI->2D converter.
    splits = tmp_path / "splits.csv"
    pd.DataFrame([{"patient_id": "A0_1", "split": "train"},
                  {"patient_id": "A0_10", "split": "val"},
                  {"patient_id": "B_1", "split": "test"}]).to_csv(splits, index=False)
    slices = prepare_slices(str(out), str(tmp_path/"output"), split_csv=str(splits))
    assert len(slices) == 6  # positive slice of each patient x 2 phases
    assert slices.ncct_volume.astype(str).str.endswith("ncct_zcrop.nii.gz").all()


def test_nested_labels(tmp_path):
    root = tmp_path / "gx_all_registered"
    labels = tmp_path / "detection"
    make_registered_root(root, "A0_1")
    write_nii(labels / "ap" / "A0_1" / "pseudo.nii.gz")
    write_nii(labels / "pvp" / "A0_1" / "lesion_mask.nii.gz")
    frame = scan_cases(str(root), str(labels/"ap"), str(labels/"pvp"),
                       str(tmp_path/"cases.csv"))
    assert len(frame) == 2


def test_missing_or_ambiguous_labels_are_rejected(tmp_path):
    root = tmp_path/"gx_all_registered"
    labels = tmp_path/"detection"
    make_registered_root(root, "A0_1")
    write_nii(labels/"ap"/"A0_1.nii.gz")
    write_nii(labels/"pvp"/"A0_1.nii.gz")
    write_nii(labels/"pvp"/"A0_1_mask.nii.gz")
    with pytest.raises(ValueError, match="expected exactly one"):
        scan_cases(str(root),str(labels/"ap"),str(labels/"pvp"),str(tmp_path/"cases.csv"))


def test_no_fuzzy_prefix_patient_mix(tmp_path):
    root = tmp_path/"gx_all_registered"
    labels = tmp_path/"detection"
    make_registered_root(root,"A0_1")
    make_registered_root(root,"A0_10")
    for phase in ("ap","pvp"):
        write_nii(labels/phase/"A0_1.nii.gz")
    with pytest.raises(ValueError, match="A0_10"):
        scan_cases(str(root),str(labels/"ap"),str(labels/"pvp"),str(tmp_path/"cases.csv"))


def test_patient_folder_masks_not_auto_pseudolabels(tmp_path):
    root = tmp_path/"gx_all_registered"
    labels = tmp_path/"detection"
    make_registered_root(root,"A0_1")
    (labels/"ap").mkdir(parents=True)
    (labels/"pvp").mkdir(parents=True)
    with pytest.raises(ValueError, match="contain no NIfTI"):
        scan_cases(str(root),str(labels/"ap"),str(labels/"pvp"),str(tmp_path/"cases.csv"))
