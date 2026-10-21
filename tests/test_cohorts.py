from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from liverdet.cohorts import build_cohorts
from liverdet.prepare import prepare_slices
from liverdet.export import export_group


def create_nii(path: Path, *, shape=(20, 24, 3)):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.zeros(shape, dtype=np.float32)
    data[4:8, 6:11, 1] = 1
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(path))


def create_manifest(tmp_path, pseudo_ids=("A0_1004","A0_1252","A2_1"),
                    ground_ids=("A1_100","A1_104","A1_105","A1_106")):
    base=tmp_path/"registered"
    detect=tmp_path/"detection"
    rows=[]
    for pid in pseudo_ids+ground_ids:
        gt=pid in ground_ids
        folder=base/pid
        ap=folder/"ap_registered.nii.gz"
        pvp=folder/"pvp_registered.nii.gz"
        create_nii(ap)
        create_nii(pvp)
        create_nii(folder/"ncct_zcrop.nii.gz")
        if gt:
            ap_mask=folder/"ap_tumor_mask_registered.nii.gz"
            pvp_mask=folder/"pvp_tumor_mask_registered.nii.gz"
            create_nii(ap_mask)
            create_nii(pvp_mask)
            ap_path,pvp_path=str(ap_mask),str(pvp_mask)
            ap_type=pvp_type="ground_truth"
            cohort="ground_truth"
        else:
            for phase in ("ap","pvp"):
                create_nii(detect/phase/f"{pid}.nii.gz")
            # Mimic user's actual CSV: most pending pseudo paths are empty.
            if pid=="A0_1004":
                ap_path=str(detect/"ap"/f"{pid}.nii.gz")
                pvp_path=str(detect/"pvp"/f"{pid}.nii.gz")
                ap_type=pvp_type="pseudo_label"
            else:
                ap_path=pvp_path=""
                ap_type=pvp_type="pending"
            cohort="pseudo_label"
        rows.append(dict(patient_id=pid,cohort_type=cohort,
                         ap_image=str(ap), ap_mask=ap_path,ap_mask_type=ap_type,
                         pvp_image=str(pvp),pvp_mask=pvp_path,pvp_mask_type=pvp_type,
                         source_folder=str(folder)))
    csv=tmp_path/"detection_cases.csv"
    pd.DataFrame(rows).to_csv(csv,index=False)
    return csv,detect


def test_pseudo_only_train_manual_only_valtest(tmp_path):
    source,labels=create_manifest(tmp_path)
    out=tmp_path/"output"
    cases,splits,inventory=build_cohorts(
        str(source),str(out),
        pseudo_ap_root=str(labels/"ap"),pseudo_pvp_root=str(labels/"pvp"),
        seed=42,val_fraction=.5
    )
    assert (splits.split=="train").sum()==3
    assert (splits.split=="val").sum()==2
    assert (splits.split=="test").sum()==2
    assert set(splits.loc[splits.split=="train","cohort_type"])=={"pseudo_label"}
    assert set(splits.loc[splits.split!="train","cohort_type"])=={"ground_truth"}
    assert all(cases.loc[cases.split=="train","label_source"]=="pseudo")
    assert all(cases.loc[cases.split!="train","label_source"]=="manual")
    assert not cases.mask_path.eq("").any()
    assert set(cases.phase)=={"ap","pvp"}
    assert len(inventory)==14

    # Existing prepare and exporters must accept manual Val/Test without
    # --allow-pseudo-eval; GT patients never enter train.
    slices=prepare_slices(str(out/"manifests"/"cases.csv"),str(out))
    assert set(slices.loc[slices.split=="train","label_source"])=={"pseudo"}
    assert set(slices.loc[slices.split=="val","label_source"]).issubset({"manual"})
    assert set(slices.loc[slices.split=="test","label_source"]).issubset({"manual"})
    exp=export_group(str(out/"manifests"/"slices.csv"),str(out/"exports"),"cect_real","ap")
    assert (exp/"PAPER_EVAL_READY").exists()
    assert not (exp/"SMOKE_TEST_ONLY").exists()


def test_missing_pseudo_is_not_silently_accepted(tmp_path):
    source,labels=create_manifest(tmp_path)
    (labels/"ap"/"A0_1252.nii.gz").unlink()
    out=tmp_path/"output"
    with pytest.raises(ValueError,match="pseudo patients lack"):
        build_cohorts(str(source),str(out),pseudo_ap_root=str(labels/"ap"),
                      pseudo_pvp_root=str(labels/"pvp"))
    assert (out/"manifests"/"cohort_inventory.csv").exists()
    cases,splits,inventory=build_cohorts(
        str(source),str(out),pseudo_ap_root=str(labels/"ap"),
        pseudo_pvp_root=str(labels/"pvp"),allow_missing_pseudo=True)
    assert not (cases.patient_id=="A0_1252").any()
    assert "missing_pseudo_mask" in set(inventory.mask_status)


def test_missing_ground_truth_never_substitutes_pseudo(tmp_path):
    source,labels=create_manifest(tmp_path)
    table=pd.read_csv(source)
    table.loc[table.patient_id=="A1_100","ap_mask"]=""
    table.to_csv(source,index=False)
    with pytest.raises(ValueError,match="manual ground-truth inputs"):
        build_cohorts(str(source),str(tmp_path/"out"),
                      pseudo_ap_root=str(labels/"ap"),pseudo_pvp_root=str(labels/"pvp"))


def test_duplicate_patient_rejected(tmp_path):
    source,labels=create_manifest(tmp_path)
    table=pd.read_csv(source)
    table=pd.concat([table,table.head(1)],ignore_index=True)
    table.to_csv(source,index=False)
    with pytest.raises(ValueError,match="Duplicate patient IDs"):
        build_cohorts(str(source),str(tmp_path/"out"),
                      pseudo_ap_root=str(labels/"ap"),pseudo_pvp_root=str(labels/"pvp"))


def test_split_repeatable(tmp_path):
    source,labels=create_manifest(tmp_path)
    args={"pseudo_ap_root":str(labels/"ap"),"pseudo_pvp_root":str(labels/"pvp"),"seed":1337}
    _,a,_=build_cohorts(str(source),str(tmp_path/"one"),**args)
    _,b,_=build_cohorts(str(source),str(tmp_path/"two"),**args)
    pd.testing.assert_frame_equal(a,b)
