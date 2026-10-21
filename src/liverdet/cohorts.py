"""Build fixed TRAIN=pseudo, VAL/TEST=manual patient splits from private case-ID CSV.

No unrestricted root directory crawl. The case list is authoritative. A
missing/pending pseudo-mask path can be resolved ONLY as
  /dataset/detection/{ap,pvp}/<patient_id>.nii.gz
Manual ground truth comes ONLY from the case list's *_mask fields, never
falls back to teacher pseudo labels.

The source case list contains sensitive research identifiers and is not
uploaded into the public source repository.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from .common import read_csv, save_csv, check_no_patient_overlap

REQUIRED = {
    "patient_id", "cohort_type",
    "ap_image", "ap_mask", "ap_mask_type",
    "pvp_image", "pvp_mask", "pvp_mask_type",
}
REAL_TYPES = {"ground_truth", "manual"}
PSEUDO_TYPES = {"pseudo_label", "pseudo"}
IGNORE_TYPES = {"pending", "", "missing", "none"}


def _resolve_pseudo(path_value: str, mask_type: str, root: Path, pid: str,
                    *, allow_missing: bool) -> tuple[str, str]:
    """Return candidate path and status; do not mistake pending as verified mask."""
    if mask_type not in PSEUDO_TYPES | IGNORE_TYPES:
        raise ValueError(f"{pid}: pseudo cohort has unexpected mask_type={mask_type!r}")
    expected = (root / f"{pid}.nii.gz").resolve()
    supplied = str(path_value).strip()
    # Cases listed as pseudo_label but not yet populated are resolved by ID.
    if supplied:
        supplied_path = Path(supplied)
        if supplied_path.resolve() != expected:
            raise ValueError(
                f"{pid}: pseudo mask path {supplied!r} differs from exact per-patient "
                f"mask {expected}; refusing implicit label substitution.")
    if not expected.is_file():
        if allow_missing:
            return "", "missing_pseudo_mask"
        return str(expected), "missing_pseudo_mask"
    return str(expected), "ok"


def _resolve_manual(path_value: str, mask_type: str, pid: str, phase: str,
                    image_path: str) -> tuple[str, str]:
    if mask_type not in REAL_TYPES:
        raise ValueError(f"{pid}/{phase}: ground_truth patient must have real ground-truth mask")
    if not path_value:
        return "", "missing_ground_truth_mask_path"
    p = Path(path_value).resolve()
    expected_name = f"{phase}_tumor_mask_registered.nii.gz"
    if p.name != expected_name:
        raise ValueError(
            f"{pid}/{phase}: expected manual lesion mask {expected_name}, got {p.name}")
    if p.parent != Path(image_path).resolve().parent:
        raise ValueError(f"{pid}/{phase}: manual mask is not in the image patient folder")
    return str(p), ("ok" if p.is_file() else "missing_ground_truth_mask_file")


def build_cohorts(
    case_list: str,
    workdir: str,
    *,
    pseudo_ap_root: str = "/dataset/detection/ap",
    pseudo_pvp_root: str = "/dataset/detection/pvp",
    val_fraction: float = 0.5,
    seed: int = 42,
    allow_missing_pseudo: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Make 3 output CSVs: cases.csv, splits.csv, cohort_inventory.csv.

    Only validation and test are divided randomly; no manual GT case is
    trained. Missing pseudo masks fail by default, optional explicit skip.

    Validation/test cases are NEVER skipped when a manual mask is missing.
    """
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")
    raw = read_csv(case_list, REQUIRED)
    if raw.empty:
        raise ValueError("Empty case list")
    if raw.patient_id.duplicated().any():
        dupes = raw.loc[raw.patient_id.duplicated(keep=False), "patient_id"].tolist()
        raise ValueError(f"Duplicate patient IDs in case list: {dupes[:8]}")
    raw["cohort_type"] = raw["cohort_type"].str.strip().str.lower()
    if not set(raw.cohort_type).issubset(PSEUDO_TYPES | REAL_TYPES):
        raise ValueError(f"Unsupported cohort types: {set(raw.cohort_type) - (PSEUDO_TYPES | REAL_TYPES)}")
    for field in ("ap_mask_type", "pvp_mask_type"):
        raw[field] = raw[field].astype(str).str.strip().str.lower()

    ground_ids = sorted(raw.loc[raw.cohort_type.isin(REAL_TYPES), "patient_id"].astype(str))
    pseudo_ids = sorted(raw.loc[raw.cohort_type.isin(PSEUDO_TYPES), "patient_id"].astype(str))
    if set(ground_ids) & set(pseudo_ids):
        raise ValueError("Same patient included as both pseudo and ground truth")
    if len(ground_ids) < 2:
        raise ValueError("At least two ground-truth patients required for val/test")

    # Stable, reproducible patient-level split; no GT enters train.
    rng = np.random.default_rng(seed)
    shuffled = np.asarray(ground_ids, dtype=object)
    rng.shuffle(shuffled)
    n_val = int(round(len(shuffled) * val_fraction))
    n_val = min(max(1, n_val), len(shuffled) - 1)
    val_ids = set(shuffled[:n_val].tolist())
    test_ids = set(shuffled[n_val:].tolist())
    roots = {"ap": Path(pseudo_ap_root), "pvp": Path(pseudo_pvp_root)}

    case_rows: list[dict] = []
    inventory_rows: list[dict] = []
    missing_pseudo_ids: set[str] = set()
    missing_gt: list[str] = []
    for row in raw.itertuples(index=False):
        pid = str(row.patient_id)
        cohort = "ground_truth" if row.cohort_type in REAL_TYPES else "pseudo_label"
        split = "train" if cohort == "pseudo_label" else ("val" if pid in val_ids else "test")
        entries = []
        for phase in ("ap", "pvp"):
            image_path = str(getattr(row, f"{phase}_image")).strip()
            mask_path = str(getattr(row, f"{phase}_mask")).strip()
            mask_type = str(getattr(row, f"{phase}_mask_type")).strip()
            if not image_path:
                raise ValueError(f"{pid}/{phase}: missing image path in case list")
            source_folder = str(getattr(row, "source_folder", "")).strip()
            if source_folder and Path(image_path).resolve().parent != Path(source_folder).resolve():
                raise ValueError(f"{pid}/{phase}: image outside case list source_folder")
            expected_image = f"{phase}_registered.nii.gz"
            if Path(image_path).name != expected_image:
                raise ValueError(f"{pid}/{phase}: expected image filename {expected_image}")

            if cohort == "pseudo_label":
                resolved_mask, mask_status = _resolve_pseudo(
                    mask_path, mask_type, roots[phase], pid,
                    allow_missing=allow_missing_pseudo
                )
                label_source = "pseudo"
                if mask_status != "ok":
                    missing_pseudo_ids.add(pid)
            else:
                resolved_mask, mask_status = _resolve_manual(
                    mask_path, mask_type, pid, phase, image_path
                )
                label_source = "manual"
                if mask_status != "ok":
                    missing_gt.append(f"{pid}/{phase}: {mask_status}")

            image_status = "ok" if Path(image_path).is_file() else "missing_image_file"
            if image_status != "ok":
                if cohort == "ground_truth":
                    missing_gt.append(f"{pid}/{phase}: {image_status}")
                else:
                    missing_pseudo_ids.add(pid)
            entries.append({
                "patient_id": pid, "cohort_type": cohort, "split": split,
                "phase": phase, "image_path": str(Path(image_path).resolve()),
                "mask_path": resolved_mask, "mask_type": mask_type,
                "ncct_path": str(Path(image_path).resolve().parent / "ncct_zcrop.nii.gz"),
                "label_source": label_source,
                "image_status": image_status, "mask_status": mask_status,
            })
        inventory_rows.extend(entries)

    work = Path(workdir)/"manifests"
    inventory = pd.DataFrame(inventory_rows).sort_values(["patient_id", "phase"]).reset_index(drop=True)
    # Always output diagnostic inventory, even if labeled data is incomplete.
    save_csv(inventory, work/"cohort_inventory.csv")
    if missing_gt:
        raise ValueError(
            f"{len(missing_gt)} manual ground-truth inputs are missing; "
            f"inspect {work/'cohort_inventory.csv'}. "
            "VAL/TEST manual cases cannot be dropped or replaced with pseudo labels."
        )
    if missing_pseudo_ids and not allow_missing_pseudo:
        raise ValueError(
            f"{len(missing_pseudo_ids)} pseudo patients lack required AP/PVP images or masks. "
            f"Inspect {work/'cohort_inventory.csv'}; "
            "use --skip-missing-pseudo to explicitly exclude these training patients."
        )

    active = inventory[~inventory.patient_id.isin(missing_pseudo_ids)].copy()
    if not (active.split == "train").any():
        raise ValueError("No complete pseudo-labeled training patients; cannot train")
    if active.loc[active.split.isin(["val", "test"]), "label_source"].ne("manual").any():
        raise ValueError("Manual ground truth not enforced for val/test")
    if active.loc[active.split == "train", "label_source"].ne("pseudo").any():
        raise ValueError("Pseudo-only training violated")
    check_no_patient_overlap(active)

    cases = active[[
        "patient_id", "phase", "image_path", "mask_path", "ncct_path",
        "split", "label_source", "cohort_type"
    ]].copy()
    splits = (cases[["patient_id", "split", "cohort_type"]]
              .drop_duplicates().sort_values(["split", "patient_id"]).reset_index(drop=True))
    if splits.patient_id.duplicated().any():
        raise ValueError("Patient leakage in split output")
    save_csv(cases, work/"cases.csv")
    save_csv(splits, work/"splits.csv")
    print("Cohort split summary (patients):")
    print(splits.groupby(["split", "cohort_type"]).size().to_string())
    print(f"Missing pseudo patients excluded: {len(missing_pseudo_ids)}")
    print(f"Files: {work/'cases.csv'}, {work/'splits.csv'}, {work/'cohort_inventory.csv'}")
    return cases, splits, inventory
