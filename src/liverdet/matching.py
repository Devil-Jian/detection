"""Spatially constrained NCCT slice matching and synthesis request preparation.

Do not mistake a high NMI score for a verified anatomical match. The result
remains 'pending' until manual QC is recorded in an explicit review CSV.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import nibabel as nib
import pandas as pd
from PIL import Image
from scipy.ndimage import zoom

from .common import boxes_load, boxes_dump, clip_boxes, read_csv, save_csv, save_png, window_uint8


def nmi(a: np.ndarray, b: np.ndarray, bins: int = 32) -> float:
    """2*MI/(H(A)+H(B)); robust enough for candidate ranking, not ground truth."""
    x, y = a.reshape(-1).astype(float), b.reshape(-1).astype(float)
    valid = np.isfinite(x) & np.isfinite(y)
    if not valid.any():
        return 0.
    hist, _, _ = np.histogram2d(x[valid], y[valid], bins=bins)
    p = hist / max(hist.sum(), 1)
    px, py = p.sum(1), p.sum(0)
    nz = p > 0
    denom = np.outer(px, py)
    mi = float((p[nz] * np.log(p[nz]/np.maximum(denom[nz], 1e-12))).sum())
    hx = -float((px[px>0] * np.log(px[px>0])).sum())
    hy = -float((py[py>0] * np.log(py[py>0])).sum())
    return max(0., 2*mi/max(hx+hy, 1e-12))


def as_same_size(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if image.shape == shape:
        return image
    resized = zoom(image, (shape[0]/image.shape[0], shape[1]/image.shape[1]), order=1)
    return resized[:shape[0], :shape[1]]


def ncct_idx_for_world(affine: np.ndarray, world: np.ndarray) -> float:
    voxel = np.linalg.inv(affine) @ world
    return float(voxel[2])


def reproject_boxes(boxes: list[list[float]], src_affine: np.ndarray, src_z: int,
                    dst_affine: np.ndarray, width: int, height: int) -> list[list[float]]:
    """Affine 2-D XY bbox reprojection; requires separate anatomical quality review."""
    mat = np.linalg.inv(dst_affine) @ src_affine
    result = []
    for x1,y1,x2,y2 in boxes:
        coords = np.array([[x1,y1,src_z,1],[x1,y2,src_z,1],
                           [x2,y1,src_z,1],[x2,y2,src_z,1]],dtype=float).T
        dest = (mat @ coords)[:2, :]
        result.extend(clip_boxes([[dest[0].min(),dest[1].min(),
                                   dest[0].max(),dest[1].max()]],width,height))
    return result


def match_ncct(manifest: str, workdir: str, *, radius_mm: float = 15,
               hu_min: float = -100, hu_max: float = 300) -> pd.DataFrame:
    df = read_csv(manifest, ["sample_id", "patient_id", "phase", "slice_index",
                             "source_volume", "ncct_volume", "boxes_real", "real_path"])
    if radius_mm <= 0:
        raise ValueError("radius_mm must be positive")
    required = ["ncct_slice", "ncct_path", "boxes_ncct", "match_nmi",
                "match_offset_mm", "match_qc"]
    for field in required:
        if field not in df:
            df[field] = ""
    cache = {}
    for i, row in df.iterrows():
        if not str(row.ncct_volume).strip():
            continue
        src, target = str(row.source_volume), str(row.ncct_volume)
        if src not in cache:
            cache[src] = nib.load(src)
        if target not in cache:
            cache[target] = nib.load(target)
        real, ncct = cache[src], cache[target]
        k = int(row.slice_index)
        world = real.affine @ np.array([real.shape[0]/2, real.shape[1]/2, k, 1.])
        estimate = ncct_idx_for_world(ncct.affine, world)
        # Slice spacing normal to NCCT image plane, in mm.
        z_step_mm = float(np.linalg.norm(ncct.affine[:3,2]))
        search = max(1, int(np.ceil(radius_mm / z_step_mm)))
        candidates = [j for j in range(max(0,int(round(estimate))-search),
                        min(ncct.shape[2],int(round(estimate))+search+1))]
        if not candidates:
            raise ValueError(f"No NCCT candidate for {row.sample_id}")
        real_slice = np.asarray(Image.open(row.real_path).convert("L"), dtype=np.float32)
        best = None
        for j in candidates:
            offset = abs(j-estimate)*z_step_mm
            if offset > radius_mm:
                continue
            candidate = window_uint8(np.asarray(ncct.dataobj[:,:,j]).T, hu_min,hu_max)
            resized = as_same_size(candidate, real_slice.shape)
            score = nmi(real_slice, resized) - 0.15 * offset/radius_mm
            if best is None or score > best[0]:
                best = (score, j, offset, candidate, nmi(real_slice, resized))
        if best is None:
            raise ValueError(f"NCCT candidates outside radius for {row.sample_id}")
        _, j, offset, matched, raw_nmi = best
        out = Path(workdir)/"data"/"ncct"/str(row.patient_id)/f"{j:04d}.png"
        if not out.exists():
            save_png(matched, out)
        boxes = reproject_boxes(
            boxes_load(row.boxes_real), real.affine, k, ncct.affine,
            ncct.shape[0], ncct.shape[1])
        df.at[i,"ncct_slice"] = j
        df.at[i,"ncct_path"] = str(out.resolve())
        df.at[i,"boxes_ncct"] = boxes_dump(boxes)
        df.at[i,"match_nmi"] = round(raw_nmi,6)
        df.at[i,"match_offset_mm"] = round(offset,3)
        df.at[i,"match_qc"] = "pending"  # Never auto-pass based on similarity alone.
    save_csv(df, manifest)
    return df


def apply_reviews(manifest: str, reviews_csv: str) -> pd.DataFrame:
    """Reviews CSV: sample_id,match_qc (pass|fail); no silent approvals."""
    frame = read_csv(manifest, ["sample_id", "match_qc"])
    reviews = read_csv(reviews_csv, ["sample_id", "match_qc"])
    if reviews.sample_id.duplicated().any():
        raise ValueError("Duplicate sample_id in QC reviews")
    if not set(reviews.match_qc).issubset({"pass","fail"}):
        raise ValueError("match_qc must be pass or fail")
    m = dict(zip(reviews.sample_id, reviews.match_qc))
    unknown = set(m) - set(frame.sample_id)
    if unknown:
        raise ValueError(f"Unknown sample_id in review CSV: {list(unknown)[:5]}")
    frame["match_qc"] = [m.get(s,old) for s,old in zip(frame.sample_id,frame.match_qc)]
    save_csv(frame,manifest)
    return frame


def synthesis_requests(manifest: str, syn_root: str, output_csv: str) -> pd.DataFrame:
    """Produce generation worklist; external generator must write 8-bit gray PNG.

    Output paths: SYN_ROOT/<phase>/<patient_id>/<slice_index:04d>.png
    Generated images must have same XY geometry as matched NCCT PNG.
    """
    df = read_csv(manifest, ["sample_id","patient_id","phase","slice_index",
                             "ncct_path","match_qc","boxes_ncct"])
    out = []
    for row in df.itertuples(index=False):
        if not row.ncct_path or row.match_qc == "fail":
            continue
        dest = Path(syn_root)/row.phase/row.patient_id/f"{int(row.slice_index):04d}.png"
        out.append(dict(sample_id=row.sample_id, patient_id=row.patient_id,
                        phase=row.phase, ncct_input=row.ncct_path,
                        syn_output=str(dest), qc=row.match_qc))
    result = pd.DataFrame(out, columns=["sample_id","patient_id","phase","ncct_input","syn_output","qc"])
    save_csv(result, output_csv)
    return result


def attach_synthetic(manifest: str, syn_root: str,
                     allow_unreviewed: bool = False) -> pd.DataFrame:
    df = read_csv(manifest, ["sample_id","patient_id","phase","slice_index",
                             "ncct_path","boxes_ncct","match_qc"])
    for col in ("syn_path","boxes_syn"):
        if col not in df:
            df[col] = ""
    count = 0
    for i, row in df.iterrows():
        if not str(row.ncct_path).strip() or str(row.match_qc) == "fail":
            continue
        dest = Path(syn_root)/str(row.phase)/str(row.patient_id)/f"{int(row.slice_index):04d}.png"
        if not dest.is_file():
            continue
        if row.match_qc != "pass" and not allow_unreviewed:
            raise ValueError(f"Match for {row.sample_id} not manually approved ({row.match_qc}).")
        with Image.open(dest) as syn, Image.open(row.ncct_path) as ncct:
            if syn.mode not in ("L","I;16","I"):
                raise ValueError(f"Syn output must be grayscale PNG: {dest}")
            if syn.size != ncct.size:
                raise ValueError(f"Syn/NCCT output geometry mismatch: {dest} {syn.size}!={ncct.size}")
            if syn.mode != "L":
                raise ValueError(f"Synthetic PNG must be uint8 grayscale with SAME fixed intensity preprocessing: {dest}")
        df.at[i,"syn_path"] = str(dest.resolve())
        df.at[i,"boxes_syn"] = row.boxes_ncct
        count += 1
    if count == 0:
        raise ValueError(f"No matching Syn PNGs in {syn_root}")
    save_csv(df, manifest)
    return df
