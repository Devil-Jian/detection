"""The external AP/PVP labels are EXACT <patient_id>.nii.gz."""
from pathlib import Path
import nibabel as nib
import numpy as np
import pytest
import pandas as pd

from liverdet.inventory import scan_cases
from liverdet.audit import audit_cases


def nii(path: Path, *, shape=(16, 14, 4), shift=0, empty=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    a = np.zeros(shape, dtype=np.float32)
    if not empty:
        a[2:6, 4:8, 1] = 1
    affine = np.eye(4)
    affine[0, 3] = shift
    nib.save(nib.Nifti1Image(a, affine), str(path))


def layout(tmp_path, ids=("A0_1004", "A0_1252")):
    base=tmp_path/"dataset"/"dataset"/"gx_all_registered"
    pseudo=tmp_path/"dataset"/"detection"
    for pid in ids:
        for phase in ("ap", "pvp"):
            nii(base/pid/f"{phase}_registered.nii.gz")
            nii(pseudo/phase/f"{pid}.nii.gz")
        nii(base/pid/"ncct_zcrop.nii.gz")
        # Phase-matched liver ROI masks, distinct from external lesion pseudo labels.
        nii(base/pid/"ap_mask_registered.nii.gz")
        nii(base/pid/"pvp_mask_registered.nii.gz")
    cases=tmp_path/"cases.csv"
    frame=scan_cases(str(base),str(pseudo/"ap"),str(pseudo/"pvp"),str(cases))
    return base,pseudo,cases,frame


def test_exact_pseudo_label_filename_and_audit(tmp_path):
    _,_,cases,frame=layout(tmp_path)
    assert len(frame) == 4
    assert set(frame.patient_id)=={"A0_1004", "A0_1252"}
    assert all(Path(row.mask_path).name == row.patient_id + ".nii.gz"
               for row in frame.itertuples(index=False))
    assert all(Path(row.ncct_path).name == "ncct_zcrop.nii.gz"
               for row in frame.itertuples(index=False))
    report=audit_cases(str(cases),str(tmp_path/"audit.csv"),inspect_masks=True)
    assert set(report.status)=={"ok"}
    assert set(report.pseudo_positive_slices)=={1}
    assert set(report.pseudo_nonzero_voxels)=={16}


def test_audit_reports_affine_mismatch_without_modifying(tmp_path):
    _,_,cases,frame=layout(tmp_path,("A0_1004",))
    wrong=Path(frame[frame.phase=="ap"].iloc[0].mask_path)
    nii(wrong,shift=30)
    dest=tmp_path/"bad-audit.csv"
    with pytest.raises(ValueError,match="failed preflight"):
        audit_cases(str(cases),str(dest))
    assert dest.exists()
    report=pd.read_csv(dest)
    row=report[report.phase=="ap"].iloc[0]
    assert "affine_mismatch" in row.errors
    assert report[report.phase=="pvp"].iloc[0].status == "ok"


def test_empty_pseudo_mask_is_not_verified_negative(tmp_path):
    _,_,cases,frame=layout(tmp_path,("A0_1004",))
    path=Path(frame[frame.phase=="pvp"].iloc[0].mask_path)
    nii(path,empty=True)
    report=audit_cases(str(cases),str(tmp_path/"audit.csv"),inspect_masks=True)
    row=report[report.phase=="pvp"].iloc[0]
    assert row.status=="warning"
    assert "pseudo_mask_empty_not_verified_negative" in row.warnings
