import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from PIL import Image
import pytest

from liverdet.prepare import boxes_from_mask, prepare_slices
from liverdet.common import boxes_load, read_csv
from liverdet.export import export_group
from liverdet.matching import nmi, match_ncct, apply_reviews, synthesis_requests, attach_synthetic
from liverdet.metrics import evaluate_predictions


def _create_vol(path, arr):
    nib.save(nib.Nifti1Image(arr, np.eye(4)), str(path))


def test_mask_bbox_orientation():
    mask=np.zeros((30,40),dtype=np.uint8)
    mask[2:6,15:20]=1
    assert boxes_from_mask(mask)==[[15.,2.,20.,6.]]


def test_nmi_prefers_identical():
    rng=np.random.default_rng(42)
    image=rng.integers(0,256,size=(64,64),dtype=np.uint8)
    other=rng.integers(0,256,size=(64,64),dtype=np.uint8)
    assert nmi(image,image)>nmi(image,other)


def test_full_three_patient_smoke(tmp_path):
    image_dir=tmp_path/"volumes"
    image_dir.mkdir()
    rows=[]
    for pid, split in (("P1","train"),("P2","val"),("P3","test")):
        arr=np.ones((24,32,4),dtype=np.float32)*50
        mask=np.zeros_like(arr,dtype=np.uint8)
        mask[3:9,10:17,1]=1
        # note x=3:9, y=10:17; PNG bbox x=3..9 y=10..17
        arr[3:9,10:17,1]=160
        for phase in ("ap","pvp"):
            vol=image_dir/f"{pid}_{phase}.nii.gz"
            ann=image_dir/f"{pid}_{phase}_mask.nii.gz"
            _create_vol(vol,arr)
            _create_vol(ann,mask)
            rows.append(dict(patient_id=pid,phase=phase,image_path=str(vol),
                             mask_path=str(ann),ncct_path=str(vol),
                             split=split,label_source="manual"))
    cases=tmp_path/"cases.csv"
    pd.DataFrame(rows).to_csv(cases,index=False)
    work=tmp_path/"work"
    manifest=work/"manifests"/"slices.csv"
    prepared=prepare_slices(str(cases),str(work))
    assert len(prepared)==3*2*4
    assert boxes_load(prepared[(prepared.patient_id=="P1") &
                               (prepared.phase=="ap") &
                               (prepared.slice_index==1)].iloc[0].boxes_real)==[[3.,10.,9.,17.]]

    base=export_group(str(manifest),str(work/"exports"),"cect_real","ap")
    annotations=json.loads((base/"annotations"/"test.json").read_text())
    assert len(annotations["images"])==4
    assert len(annotations["annotations"])==1
    pred=[dict(image_id=annotations["annotations"][0]["image_id"],category_id=1,
               bbox=annotations["annotations"][0]["bbox"],score=.99)]
    predictions=tmp_path/"pred.json"
    predictions.write_text(json.dumps(pred))
    metrics=evaluate_predictions(str(base/"annotations"/"test.json"),str(predictions))
    assert metrics["ap50"]>.99

    matched=match_ncct(str(manifest),str(work))
    assert set(matched.match_qc)=={"pending"}
    reviews=tmp_path/"reviews.csv"
    matched[["sample_id"]].assign(match_qc="pass").to_csv(reviews,index=False)
    apply_reviews(str(manifest),str(reviews))
    requests=synthesis_requests(str(manifest),str(work/"syn"),str(work/"requests.csv"))
    assert len(requests)==len(prepared)
    for r in requests.itertuples(index=False):
        path=Path(r.syn_output)
        path.parent.mkdir(parents=True,exist_ok=True)
        Image.open(r.ncct_input).save(path)
    attach_synthetic(str(manifest),str(work/"syn"))

    syn=export_group(str(manifest),str(work/"exports"),"cect_syn","ap")
    mix=export_group(str(manifest),str(work/"exports"),"cect_real_syn","ap")
    ncct=export_group(str(manifest),str(work/"exports"),"ncct")
    aug=export_group(str(manifest),str(work/"exports"),"cect_real_aug","ap")
    real_train=json.loads((base/"annotations"/"train.json").read_text())
    syn_train=json.loads((syn/"annotations"/"train.json").read_text())
    mix_train=json.loads((mix/"annotations"/"train.json").read_text())
    ncct_train=json.loads((ncct/"annotations"/"train.json").read_text())
    aug_train=json.loads((aug/"annotations"/"train.json").read_text())
    assert len(real_train["images"])==4
    assert len(syn_train["images"])==4
    assert len(mix_train["images"])==8
    assert len(ncct_train["images"])==4
    assert len(aug_train["images"])==8
    assert (mix/"dataset.yaml").exists()


def test_pseudo_eval_is_blocked(tmp_path):
    row=dict(sample_id="P_ap_0000",patient_id="P",phase="ap",split="test",
             real_path=str(tmp_path/"image.png"),boxes_real="[]",label_source="pseudo")
    Image.fromarray(np.zeros((8,8),dtype=np.uint8)).save(row["real_path"])
    csv=tmp_path/"slices.csv"
    pd.DataFrame([row]).to_csv(csv,index=False)
    with pytest.raises(ValueError,match="manual ground truth"):
        export_group(str(csv),str(tmp_path/"out"),"cect_real","ap")
