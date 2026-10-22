"""Pseudo-only small-component filtering without any liver-mask dependency."""
from pathlib import Path
import json

from PIL import Image

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

    # Each actually exported Train PNG is discoverable with exact patient,
    # phase, original volume, pseudo-mask file and zero-based slice index.
    json_path = output / "manifests" / "train_slices.json"
    index = json.loads(json_path.read_text(encoding="utf-8"))
    assert isinstance(index, list)
    assert len(index) == 2
    assert list(index[0]) == [
        "sample_id", "patient_id", "phase", "slice_index",
        "source_volume", "mask_path", "real_path", "label_source",
    ]
    assert [row["sample_id"] for row in index] == [
        "P_PSEUDO_ap_0001", "P_PSEUDO_pvp_0001"
    ]
    assert all(row["slice_index"] == 1 for row in index)
    assert all(type(row["slice_index"]) is int for row in index)
    assert {row["label_source"] for row in index} == {"pseudo"}
    assert {row["phase"] for row in index} == {"ap", "pvp"}
    for row in index:
        assert Path(row["source_volume"]).exists()
        assert Path(row["mask_path"]).exists()
        assert Path(row["real_path"]).is_file()
        assert row["sample_id"] in Path(row["real_path"]).name
    assert {row["sample_id"] for row in index} == set(samples.loc[samples.split == "train", "sample_id"])
    assert not (output / "manifests" / "train_slices.csv").exists()

    # HU=0 maps to uint8 round((0+160)/400*255) = 102 by default.
    with Image.open(index[0]["real_path"]) as im:
        assert im.mode == "L"
        assert np.all(np.asarray(im) == 102)



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
    # A valid empty JSON array is written even without Train positives.
    train_index = json.loads((out / "manifests" / "train_slices.json").read_text())
    assert train_index == []


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


def test_train_slice_optional_stdout_includes_requested_identifiers(tmp_path, capsys):
    csv = _cases(tmp_path)
    prepare_slices(
        str(csv), str(tmp_path / "out"), min_component_pixels=4,
        print_train_slices=True,
    )
    output = capsys.readouterr().out
    for phase in ("ap", "pvp"):
        assert (
            f"TRAIN_SLICE sample_id=P_PSEUDO_{phase}_0001 "
            f"phase={phase} slice_index=1 source_volume="
        ) in output
        assert f"/P_PSEUDO_{phase}_0001.png" in output
    assert "TRAIN_SLICE sample_id=P_VAL" not in output
    assert "Train slice index:" in output


def test_default_and_explicit_hu_windows(tmp_path):
    csv = _cases(tmp_path)
    default = prepare_slices(str(csv), str(tmp_path / "default"))
    chosen = default[(default.split == "train") & (default.phase == "ap")].iloc[0]
    with Image.open(chosen.real_path) as im:
        assert int(np.asarray(im)[0, 0]) == 102

    explicit = prepare_slices(str(csv), str(tmp_path / "override"),
                              hu_min=-100, hu_max=300)
    other = explicit[(explicit.split == "train") & (explicit.phase == "ap")].iloc[0]
    with Image.open(other.real_path) as im:
        assert int(np.asarray(im)[0, 0]) == 64

def test_matching_default_uses_same_hu_window():
    from inspect import signature
    from liverdet.matching import match_ncct
    assert signature(prepare_slices).parameters["hu_min"].default == -160
    assert signature(prepare_slices).parameters["hu_max"].default == 240
    assert signature(match_ncct).parameters["hu_min"].default == -160
    assert signature(match_ncct).parameters["hu_max"].default == 240

def test_cli_hu_defaults_and_json_index():
    from liverdet.cli import configure
    parser = configure()
    for action in ("prepare", "match-ncct"):
        args = [action, "--workdir", "/tmp/out", "--manifest", "/tmp/slices.csv"] if action == "match-ncct" else [
            action, "--cases", "/tmp/cases.csv", "--workdir", "/tmp/out"
        ]
        options = parser.parse_args(args)
        assert options.hu_min == -160
        assert options.hu_max == 240
    assert parser.parse_args([
        "prepare", "--cases", "/tmp/cases.csv", "--workdir", "/tmp/out",
        "--hu-min", "-100", "--hu-max", "300"
    ]).hu_min == -100
