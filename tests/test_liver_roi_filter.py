"""Pseudo lesion labels MUST be intersected with registered liver ROI;
manual ground truth must NOT be clipped to the liver ROI.
"""
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from liverdet.prepare import prepare_slices
from liverdet.common import boxes_load
from liverdet.audit import audit_cases


def nifti(path: Path, array, *, x_shift=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    affine = np.eye(4)
    affine[0, 3] = x_shift
    nib.save(nib.Nifti1Image(array.astype(np.float32), affine), str(path))


def make_cases(tmp_path, *, exterior_only=False, missing_liver=False, bad_liver_affine=False):
    reg = tmp_path / "dataset" / "gx_all_registered"
    pseudo_root = tmp_path / "detection"
    rows = []
    for pid, split, source in (("A0_1004","train","pseudo"),
                               ("A1_2001","val","manual"),
                               ("A1_2002","test","manual")):
        patient = reg / pid
        for phase in ("ap", "pvp"):
            volume = np.zeros((20, 20, 3), dtype=np.float32)
            lesion = np.zeros_like(volume)
            if not exterior_only or source == "manual":
                lesion[2:5, 3:6, 1] = 1    # inside liver
            lesion[12:15, 13:16, 1] = 1  # outside liver
            liver = np.zeros_like(volume)
            liver[:9,:9,:] = 1
            image_path = patient / f"{phase}_registered.nii.gz"
            nifti(image_path, volume)
            ncct_path = patient / "ncct_zcrop.nii.gz"
            nifti(ncct_path, volume)
            if source == "pseudo":
                lesion_path = pseudo_root / phase / f"{pid}.nii.gz"
                if not missing_liver:
                    nifti(patient / f"{phase}_mask_registered.nii.gz", liver,
                          x_shift=7 if bad_liver_affine else 0)
            else:
                lesion_path = patient / f"{phase}_tumor_mask_registered.nii.gz"
                # Deliberately NO liver mask for manual Val/Test.
            nifti(lesion_path, lesion)
            rows.append(dict(patient_id=pid, phase=phase, split=split, image_path=str(image_path),
                             mask_path=str(lesion_path), ncct_path=str(ncct_path),
                             label_source=source))
    csv = tmp_path / "cases.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    return csv


def test_filter_pseudo_inside_liver_only_and_keep_manual_gt(tmp_path):
    cases = make_cases(tmp_path)
    out = tmp_path / "processed"
    audit = audit_cases(str(cases), str(tmp_path/"audit.csv"),inspect_masks=True)
    assert "pseudo_lesions_outside_liver_will_be_filtered" in audit.loc[
        (audit.patient_id=="A0_1004") & (audit.phase=="ap"), "warnings"].iloc[0]
    assert audit.loc[(audit.patient_id=="A0_1004") &
                     (audit.phase=="ap"), "pseudo_voxels_outside_liver"].iloc[0] == 9
    assert audit.loc[(audit.patient_id=="A1_2001") &
                     (audit.phase=="ap"), "liver_mask_path"].iloc[0] == ""

    df = prepare_slices(str(cases), str(out))
    pseudo = df[(df.split=="train") & (df.phase=="ap")]
    assert len(pseudo) == 1
    assert pseudo.iloc[0].slice_index == 1
    assert boxes_load(pseudo.iloc[0].boxes_real) == [[2.0,3.0,5.0,6.0]]
    assert pseudo.iloc[0].liver_mask_path.endswith("ap_mask_registered.nii.gz")
    for split in ("val", "test"):
        manual = df[(df.split==split) & (df.phase=="ap") & (df.slice_index==1)]
        assert len(manual) == 1
        assert boxes_load(manual.iloc[0].boxes_real) == [
            [2.0,3.0,5.0,6.0], [12.0,13.0,15.0,16.0]
        ]
        assert manual.iloc[0].liver_mask_path == ""

    qc = pd.read_csv(out/"manifests"/"liver_filter_qc.csv")
    assert len(qc) == 2  # only pseudo Train AP and PVP, not manual GT
    assert set(qc.pseudo_voxels_before) == {18}
    assert set(qc.pseudo_voxels_inside_liver) == {9}
    assert set(qc.pseudo_voxels_removed) == {9}


def test_outside_only_pseudo_does_not_become_negative(tmp_path):
    cases = make_cases(tmp_path, exterior_only=True)
    df=prepare_slices(str(cases),str(tmp_path/"processed"))
    assert (df.split=="train").sum() == 0
    qc=pd.read_csv(tmp_path/"processed"/"manifests"/"liver_filter_qc.csv")
    assert set(qc.positive_slices_before)=={1}
    assert set(qc.positive_slices_after)=={0}


@pytest.mark.parametrize("missing,affine",[(True,False),(False,True)])
def test_missing_or_misaligned_liver_mask_fails_closed(tmp_path,missing,affine):
    cases = make_cases(tmp_path,missing_liver=missing,bad_liver_affine=affine)
    with pytest.raises((FileNotFoundError,ValueError),
                       match="liver mask|Liver mask"):
        prepare_slices(str(cases),str(tmp_path/"processed"))
    with pytest.raises(ValueError,match="failed preflight"):
        audit_cases(str(cases),str(tmp_path/"audit.csv"))
