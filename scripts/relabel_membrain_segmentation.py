#!/usr/bin/env python3
"""
Relabel a MemBrain-seg MRC volume using distance constraints from manual annotations.

Output label scheme:
  0 - no segmentation (original 0)
  1 - generic membrane (original 1, not assigned to a subclass)
  2 - presynaptic membrane (original 1, near presynaptic membrane cloud)
  3 - postsynaptic membrane (original 1, near postsynaptic membrane cloud)
  4 - presynaptic active zone (label 2 voxels near presynaptic active-zone points)
  5 - postsynaptic active zone (label 3 voxels near postsynaptic active-zone points)
  6 - vesicles (near vesicle point cloud / GLB vertices)

Outputs are written under ``<alignment_dir>/STT_results/membranes_labeled/``:
  - one multi-label MRC (labels 0–6)
  - XY max-projection PNG overview
  - optional per-label binary MRCs with ``--write-separate-masks``

Reference geometry is loaded in tomogram coordinates (nm), matching STT / aunps annotations.
By default 1 voxel = 1 nm (--voxel-size-nm 1.0), so --distance-nm 5 corresponds to 5 voxels.

Examples:
  # Single tomogram
  PYTHONPATH=src python scripts/relabel_membrain_segmentation.py \\
    --tomogram-path data/15F1/TOP_TOMOS/20231017_EGmilled24-2_68 \\
    --alignment-dir best_alignment \\
    --distance-nm 5

  # Batch from STT tomograms.csv
  PYTHONPATH=src python scripts/relabel_membrain_segmentation.py \\
    --csv tomogram_csv_files/tomograms_15F1-H12Cys_FINAL.csv \\
    --distance-nm 5
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import mrcfile
import numpy as np
import pandas as pd
import trimesh
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from scipy.ndimage import distance_transform_edt


def require_alignment_dir(alignment_dir, *, context: str = "") -> str:
    if alignment_dir is None:
        msg = "alignment_dir is required and cannot be None."
        if context:
            msg = f"{msg} ({context})"
        raise ValueError(msg)
    s = str(alignment_dir).strip()
    if not s or s.lower() in ("nan", "none"):
        raise ValueError("alignment_dir must be a non-empty string.")
    return s

GLB_AXIS_PERM = [0, 2, 1]
GLB_SCALE = np.array([10.0, -10.0, 10.0], dtype=float)

LABEL_NAMES = {
    0: "background",
    1: "membrane",
    2: "presynaptic_membrane",
    3: "postsynaptic_membrane",
    4: "presynaptic_active_zone",
    5: "postsynaptic_active_zone",
    6: "vesicle",
}

# RGB colors for XY overview (label id → color); background is transparent/black.
LABEL_COLORS = {
    0: (0.0, 0.0, 0.0),
    1: (0.55, 0.55, 0.55),  # membrane — grey
    2: (0.90, 0.15, 0.15),  # presynaptic membrane — red
    3: (0.15, 0.75, 0.25),  # postsynaptic membrane — green
    4: (0.95, 0.40, 0.40),  # presynaptic AZ — lighter red
    5: (0.40, 0.90, 0.45),  # postsynaptic AZ — lighter green
    6: (0.98, 0.55, 0.55),  # vesicle — pastel / light red
}

# Labels written as separate binary MRCs when --write-separate-masks is set.
SEPARATE_MASK_LABELS = (1, 2, 3, 4, 5, 6)


def _load_txt_points(path: Path) -> np.ndarray:
    if not path.is_file():
        return np.zeros((0, 3), dtype=float)
    pts = np.atleast_2d(np.loadtxt(path, delimiter=None))
    if pts.size == 0:
        return np.zeros((0, 3), dtype=float)
    return pts.astype(float)


def _transform_glb_vertices(vertices: np.ndarray) -> np.ndarray:
    """FindingAMPA/Blender GLB export -> tomogram nm coordinates (X, Y, Z)."""
    v = np.atleast_2d(np.asarray(vertices, dtype=float))
    return v[:, GLB_AXIS_PERM] * GLB_SCALE


def load_glb_vertices(path: Path, *, apply_glb_transform: bool = True) -> np.ndarray:
    if not path.is_file():
        return np.zeros((0, 3), dtype=float)
    with open(path, "rb") as f:
        scene = trimesh.exchange.gltf.load_glb(f)
    parts: list[np.ndarray] = []
    for mesh in scene["geometry"].values():
        verts = np.asarray(mesh["vertices"], dtype=float)
        if apply_glb_transform:
            verts = _transform_glb_vertices(verts)
        parts.append(verts)
    if not parts:
        return np.zeros((0, 3), dtype=float)
    return np.vstack(parts)


def _concat_point_sets(parts: Iterable[np.ndarray]) -> np.ndarray:
    arrays = [np.atleast_2d(p) for p in parts if p is not None and len(np.atleast_2d(p))]
    if not arrays:
        return np.zeros((0, 3), dtype=float)
    return np.vstack(arrays)


def nm_to_voxel_indices(coords_nm: np.ndarray, voxel_size_nm: float) -> np.ndarray:
    """
    Convert X,Y,Z in nm to Z,Y,X integer voxel indices for MRC arrays (mrcfile order).
    """
    xyz = np.atleast_2d(np.asarray(coords_nm, dtype=float))
    if len(xyz) == 0:
        return np.zeros((0, 3), dtype=int)
    scale = float(voxel_size_nm)
    ix = np.rint(xyz[:, 0] / scale).astype(int)
    iy = np.rint(xyz[:, 1] / scale).astype(int)
    iz = np.rint(xyz[:, 2] / scale).astype(int)
    return np.column_stack([iz, iy, ix])


def proximity_mask_from_points(
    shape: tuple[int, ...],
    points_nm: np.ndarray,
    distance_nm: float,
    voxel_size_nm: float,
    *,
    origin_nm: np.ndarray | None = None,
) -> np.ndarray:
    """
    Boolean mask of voxels whose centers are within ``distance_nm`` of any reference point.
    """
    if len(np.atleast_2d(points_nm)) == 0:
        return np.zeros(shape, dtype=bool)
    if distance_nm <= 0:
        return np.zeros(shape, dtype=bool)

    pts = np.atleast_2d(np.asarray(points_nm, dtype=float))
    if origin_nm is not None:
        pts = pts - np.asarray(origin_nm, dtype=float).reshape(1, 3)

    seeds = np.zeros(shape, dtype=bool)
    vox = nm_to_voxel_indices(pts, voxel_size_nm)
    z, y, x = vox[:, 0], vox[:, 1], vox[:, 2]
    valid = (z >= 0) & (z < shape[0]) & (y >= 0) & (y < shape[1]) & (x >= 0) & (x < shape[2])
    if not np.any(valid):
        return np.zeros(shape, dtype=bool)
    seeds[z[valid], y[valid], x[valid]] = True

    radius_vox = float(distance_nm) / float(voxel_size_nm)
    dist = distance_transform_edt(~seeds)
    return dist <= radius_vox


def distance_field_from_points(
    shape: tuple[int, ...],
    points_nm: np.ndarray,
    voxel_size_nm: float,
    *,
    origin_nm: np.ndarray | None = None,
) -> np.ndarray:
    """Voxel-wise Euclidean distance (nm) to the nearest reference point."""
    pts = np.atleast_2d(np.asarray(points_nm, dtype=float))
    if len(pts) == 0:
        return np.full(shape, np.inf, dtype=float)
    if origin_nm is not None:
        pts = pts - np.asarray(origin_nm, dtype=float).reshape(1, 3)

    seeds = np.zeros(shape, dtype=bool)
    vox = nm_to_voxel_indices(pts, voxel_size_nm)
    z, y, x = vox[:, 0], vox[:, 1], vox[:, 2]
    valid = (z >= 0) & (z < shape[0]) & (y >= 0) & (y < shape[1]) & (x >= 0) & (x < shape[2])
    if not np.any(valid):
        return np.full(shape, np.inf, dtype=float)
    seeds[z[valid], y[valid], x[valid]] = True
    return distance_transform_edt(~seeds) * float(voxel_size_nm)


def load_membrane_points(aunps_dir: Path, side: str) -> np.ndarray:
    """Load presynaptic or postsynaptic membrane points (GLB preferred, TXT fallback)."""
    glb_name = f"{side}membranes.glb"
    glb_path = aunps_dir / glb_name
    if glb_path.is_file():
        return load_glb_vertices(glb_path)

    txt_paths = sorted(aunps_dir.glob(f"{side}membranes_*.txt"))
    return _concat_point_sets(_load_txt_points(p) for p in txt_paths)


def load_vesicle_points(aunps_dir: Path) -> np.ndarray:
    glb_path = aunps_dir / "synapticvesicles.glb"
    if glb_path.is_file():
        return load_glb_vertices(glb_path)
    txt_paths = sorted(aunps_dir.glob("synapticvesicles_*.txt"))
    return _concat_point_sets(_load_txt_points(p) for p in txt_paths)


def load_active_zone_side_points(
    tomogram_path: Path,
    alignment_dir: str,
    side: str,
) -> np.ndarray:
    """
    Combine active-zone point clouds for one side across all zones.

    ``side`` is ``"presynaptic"`` or ``"postsynaptic"``.
    Uses outer + inner txt files from STT_results/activezone/.
    """
    active_zone_dir = tomogram_path / alignment_dir / "STT_results" / "activezone"
    if not active_zone_dir.is_dir():
        print(f"Warning: active zone directory not found: {active_zone_dir}")
        return np.zeros((0, 3), dtype=float)

    parts: list[np.ndarray] = []
    if side == "presynaptic":
        for path in sorted(active_zone_dir.glob("*_pre_outer.txt")):
            zone_name = path.stem[:-10] if path.stem.endswith("_pre_outer") else path.stem
            parts.append(_load_txt_points(path))
            parts.append(_load_txt_points(active_zone_dir / f"{zone_name}_pre_inner.txt"))
    else:
        for path in sorted(active_zone_dir.glob("*_post_outer.txt")):
            zone_name = path.stem[:-11] if path.stem.endswith("_post_outer") else path.stem
            parts.append(_load_txt_points(path))
            parts.append(_load_txt_points(active_zone_dir / f"{zone_name}_post_inner.txt"))
    return _concat_point_sets(parts)


def prepare_membrane_binary(
    seg: np.ndarray,
    *,
    membrane_threshold: float | None = None,
) -> tuple[np.ndarray, str]:
    """
    Convert input MemBrain volume to a boolean membrane mask.

    Hard label maps (exact 0/1) use ``seg == 1``. Soft/probability maps (many unique
    values) are thresholded. If ``membrane_threshold`` is None, auto-pick:
    ``> 0.5`` when max ≤ 1.5, else ``> 150`` (override with ``--membrane-threshold``).
    """
    seg = np.asarray(seg)
    unique = np.unique(seg)
    hard_labels = (
        len(unique) <= 8
        and np.all(np.isclose(unique, np.round(unique)))
        and (1 in set(int(round(u)) for u in unique) or np.any(seg == 1))
    )
    if hard_labels and membrane_threshold is None:
        mask = seg == 1
        return mask, "hard labels (seg == 1)"

    if membrane_threshold is None:
        vmax = float(np.nanmax(seg)) if seg.size else 0.0
        membrane_threshold = 0.5 if vmax <= 1.5 else 150.0
    mask = seg > float(membrane_threshold)
    return mask, f"threshold > {membrane_threshold:g}"


def relabel_segmentation(
    seg: np.ndarray,
    *,
    presyn_pts: np.ndarray,
    postsyn_pts: np.ndarray,
    presyn_az_pts: np.ndarray,
    postsyn_az_pts: np.ndarray,
    vesicle_pts: np.ndarray,
    distance_nm: float,
    voxel_size_nm: float,
    origin_nm: np.ndarray | None = None,
    membrane_threshold: float | None = None,
) -> tuple[np.ndarray, str]:
    """Apply distance-based relabeling; returns (labels, membrane-mask description)."""
    seg = np.asarray(seg)
    out = np.zeros(seg.shape, dtype=np.uint8)

    membrane, membrane_desc = prepare_membrane_binary(
        seg, membrane_threshold=membrane_threshold
    )
    out[membrane] = 1

    if np.any(membrane) and (len(presyn_pts) or len(postsyn_pts)):
        dist_pre = distance_field_from_points(
            seg.shape, presyn_pts, voxel_size_nm, origin_nm=origin_nm
        )
        dist_post = distance_field_from_points(
            seg.shape, postsyn_pts, voxel_size_nm, origin_nm=origin_nm
        )
        near_pre = membrane & (dist_pre <= distance_nm)
        near_post = membrane & (dist_post <= distance_nm)
        presyn_only = near_pre & ~near_post
        postsyn_only = near_post & ~near_pre
        overlap = near_pre & near_post
        out[presyn_only] = 2
        out[postsyn_only] = 3
        if np.any(overlap):
            out[overlap] = np.where(
                dist_pre[overlap] <= dist_post[overlap], 2, 3
            ).astype(np.uint8)

    if len(presyn_az_pts):
        az_pre = proximity_mask_from_points(
            seg.shape, presyn_az_pts, distance_nm, voxel_size_nm, origin_nm=origin_nm
        )
        out[(out == 2) & az_pre] = 4

    if len(postsyn_az_pts):
        az_post = proximity_mask_from_points(
            seg.shape, postsyn_az_pts, distance_nm, voxel_size_nm, origin_nm=origin_nm
        )
        out[(out == 3) & az_post] = 5

    if len(vesicle_pts):
        ves_mask = proximity_mask_from_points(
            seg.shape, vesicle_pts, distance_nm, voxel_size_nm, origin_nm=origin_nm
        )
        out[ves_mask] = 6

    return out, membrane_desc


def save_xy_projection_png(
    labels: np.ndarray,
    output_path: Path,
    *,
    title: str,
) -> Path:
    """
    Save a top-down XY overview: for each (y,x), take the max label along Z
    (preferring non-zero labels).
    """
    labels = np.asarray(labels)
    # Prefer non-background labels along Z for a readable overview.
    nonzero = labels > 0
    # Where any non-zero exists, take max among non-zero; else 0.
    masked = np.where(nonzero, labels, 0)
    proj = masked.max(axis=0)

    cmap_colors = [LABEL_COLORS[i] for i in range(7)]
    cmap = ListedColormap(cmap_colors)

    fig, ax = plt.subplots(figsize=(8.5, 8.0))
    ax.imshow(proj, cmap=cmap, vmin=0, vmax=6, origin="lower", interpolation="nearest")
    ax.set_xlabel("X (voxels)")
    ax.set_ylabel("Y (voxels)")
    ax.set_title(title)
    present = sorted({int(v) for v in np.unique(proj) if int(v) in LABEL_NAMES})
    handles = [
        Patch(facecolor=LABEL_COLORS[i], edgecolor="0.2", label=f"{i}: {LABEL_NAMES[i]}")
        for i in present
        if i != 0
    ]
    if handles:
        ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def write_mrc_with_voxel_size(
    path: Path,
    data: np.ndarray,
    *,
    voxel_size_angstrom: float | tuple[float, float, float] = 10.0,
) -> Path:
    """Write an MRC and set isotropic (or XYZ) voxel size in Angstroms in the header."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if np.isscalar(voxel_size_angstrom):
        vx = vy = vz = float(voxel_size_angstrom)
    else:
        vx, vy, vz = (float(v) for v in voxel_size_angstrom)
    with mrcfile.new(path, overwrite=True) as mrc:
        mrc.set_data(np.ascontiguousarray(data))
        mrc.voxel_size = (vx, vy, vz)
    return path


def read_voxel_size_angstrom(
    mrc,
    *,
    fallback_nm: float = 1.0,
) -> tuple[float, float, float]:
    """
    Read voxel size (Å) from an open MRC. If missing/zero, use ``fallback_nm * 10``
    (1 nm = 10 Å), matching the usual BIN4 tomogram pixel size.
    """
    vs = mrc.voxel_size
    vx, vy, vz = float(vs.x), float(vs.y), float(vs.z)
    if vx > 0 and vy > 0 and vz > 0:
        return (vx, vy, vz)
    ang = float(fallback_nm) * 10.0
    return (ang, ang, ang)


def write_separate_label_mrcs(
    labels: np.ndarray,
    out_dir: Path,
    *,
    tomogram_name: str,
    voxel_size_angstrom: float | tuple[float, float, float] = 10.0,
) -> list[Path]:
    """Write one binary MRC per present membrane-related label."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for label_id in SEPARATE_MASK_LABELS:
        mask = (labels == label_id).astype(np.int16)
        if not np.any(mask):
            continue
        name = LABEL_NAMES[label_id]
        path = out_dir / f"{tomogram_name}_label{label_id}_{name}.mrc"
        write_mrc_with_voxel_size(path, mask, voxel_size_angstrom=voxel_size_angstrom)
        written.append(path)
    return written


def _parse_origin(text: str) -> np.ndarray:
    parts = [float(x.strip()) for x in text.split(",")]
    if len(parts) != 3:
        raise ValueError("Origin must be three comma-separated values: x,y,z in nm")
    return np.array(parts, dtype=float)


def _discover_segmentation_mrc(tomogram_path: Path, alignment_dir: str) -> Optional[Path]:
    membrain_dir = tomogram_path / alignment_dir / "membrain"
    if not membrain_dir.is_dir():
        return None
    candidates = sorted(membrain_dir.glob("*.mrc"))
    # Prefer hard/segmented maps but skip prior relabeled outputs.
    candidates = [p for p in candidates if "relabeled" not in p.name.lower()]
    if not candidates:
        return None
    preferred = [p for p in candidates if "seg" in p.name.lower()]
    return preferred[0] if preferred else candidates[0]


def _label_counts(labels: np.ndarray) -> dict[int, int]:
    unique, counts = np.unique(labels, return_counts=True)
    return {int(u): int(c) for u, c in zip(unique, counts)}


def default_data_root() -> Path:
    return Path(os.environ.get("TOMO_ROOT_BASE") or "data")


def load_csv_jobs(
    csv_path: Path,
    *,
    data_dir: Path,
    set_name: str | None = None,
) -> list[tuple[Path, str, str]]:
    """
    Load STT tomograms.csv rows as (tomogram_path, alignment_dir, set_name).

    Required columns: tomoname, set, alignment_dir.
    Paths are ``<data_dir>/<set>/TOP_TOMOS/<tomoname>``.
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    required = {"tomoname", "set", "alignment_dir"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV {csv_path} missing required columns: {sorted(missing)}. "
            "Expected STT tomograms.csv format (tomoname, set, alignment_dir, ...)."
        )

    if set_name:
        df = df[df["set"].astype(str) == str(set_name)]

    jobs: list[tuple[Path, str, str]] = []
    for _, row in df.iterrows():
        tomoname = str(row["tomoname"]).strip()
        row_set = str(row["set"]).strip()
        try:
            alignment_dir = require_alignment_dir(
                row["alignment_dir"], context=f"tomogram {tomoname}"
            )
        except ValueError as exc:
            print(f"Skipping {tomoname}: {exc}")
            continue
        if not tomoname or not row_set or tomoname.lower() == "nan":
            print(f"Skipping row with missing tomoname/set: {row.to_dict()}")
            continue
        tomogram_path = Path(data_dir) / row_set / "TOP_TOMOS" / tomoname
        jobs.append((tomogram_path, alignment_dir, row_set))
    return jobs


def process_one_tomogram(
    tomogram_path: Path,
    alignment_dir: str,
    *,
    distance_nm: float,
    voxel_size_nm: float,
    origin_nm: np.ndarray,
    membrane_threshold: float | None = None,
    skip_vesicles: bool = False,
    skip_active_zones: bool = False,
    segmentation_mrc: Path | None = None,
    output_dir: Path | None = None,
    output_mrc: Path | None = None,
    presyn_glb: Path | None = None,
    postsyn_glb: Path | None = None,
    vesicle_glb: Path | None = None,
    write_separate_masks: bool = False,
    rerun: bool = False,
) -> Path:
    """Relabel one tomogram; returns the output directory written."""
    tomogram_path = Path(tomogram_path)
    alignment_dir = require_alignment_dir(alignment_dir)
    tomogram_name = tomogram_path.name

    out_dir = (
        Path(output_dir)
        if output_dir
        else tomogram_path / alignment_dir / "STT_results" / "membranes_labeled"
    )
    out_path = (
        Path(output_mrc)
        if output_mrc
        else out_dir / f"{tomogram_name}_membranes_relabeled.mrc"
    )
    png_path = out_dir / f"{tomogram_name}_membranes_relabeled_xy_projection.png"

    if (
        not rerun
        and out_path.is_file()
        and out_path.stat().st_size > 0
        and png_path.is_file()
        and png_path.stat().st_size > 0
    ):
        print(f"SKIP (already complete): {out_path}")
        return Path(out_dir)

    aunps_dir = tomogram_path / alignment_dir / "aunps"
    if not aunps_dir.is_dir():
        raise FileNotFoundError(f"AuNPs directory not found: {aunps_dir}")

    seg_path = (
        Path(segmentation_mrc)
        if segmentation_mrc
        else _discover_segmentation_mrc(tomogram_path, alignment_dir)
    )
    if seg_path is None:
        raise FileNotFoundError(
            "Segmentation MRC not found. Pass --segmentation-mrc or place a .mrc under membrain/."
        )
    seg_path = Path(seg_path)
    if not seg_path.is_absolute() and not seg_path.is_file():
        cand = tomogram_path / seg_path
        if cand.is_file():
            seg_path = cand
    if not seg_path.is_file():
        raise FileNotFoundError(
            f"Segmentation MRC not found: {seg_path}. "
            "Pass --segmentation-mrc or place a .mrc under membrain/."
        )

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Tomogram:      {tomogram_path}")
    print(f"Alignment:     {alignment_dir}")
    print(f"Segmentation:  {seg_path}")
    print(f"Output dir:    {out_dir}")
    print(f"Output MRC:    {out_path}")
    print(f"Output PNG:    {png_path}")
    print(f"Distance:      {distance_nm} nm")
    print(f"Voxel size:    {voxel_size_nm} nm")
    print(f"Origin (nm):   {origin_nm}")

    if presyn_glb:
        presyn_pts = load_glb_vertices(Path(presyn_glb))
    else:
        presyn_pts = load_membrane_points(aunps_dir, "presynaptic")
    if postsyn_glb:
        postsyn_pts = load_glb_vertices(Path(postsyn_glb))
    else:
        postsyn_pts = load_membrane_points(aunps_dir, "postsynaptic")

    if vesicle_glb:
        vesicle_pts = load_glb_vertices(Path(vesicle_glb))
    elif skip_vesicles:
        vesicle_pts = np.zeros((0, 3))
    else:
        vesicle_pts = load_vesicle_points(aunps_dir)

    if skip_active_zones:
        presyn_az_pts = np.zeros((0, 3))
        postsyn_az_pts = np.zeros((0, 3))
    else:
        presyn_az_pts = load_active_zone_side_points(
            tomogram_path, alignment_dir, "presynaptic"
        )
        postsyn_az_pts = load_active_zone_side_points(
            tomogram_path, alignment_dir, "postsynaptic"
        )

    print(
        f"Reference points: presyn membrane {len(presyn_pts):,}, "
        f"postsyn membrane {len(postsyn_pts):,}"
    )
    print(
        f"                  presyn AZ {len(presyn_az_pts):,}, "
        f"postsyn AZ {len(postsyn_az_pts):,}"
    )
    print(f"                  vesicles {len(vesicle_pts):,}")

    with mrcfile.open(seg_path, permissive=True) as mrc:
        seg = np.asarray(mrc.data)
        voxel_size_angstrom = read_voxel_size_angstrom(
            mrc, fallback_nm=float(voxel_size_nm)
        )

    print(
        f"Volume shape (Z,Y,X): {seg.shape}; "
        f"input value range: [{np.nanmin(seg):g}, {np.nanmax(seg):g}]"
    )
    print(
        f"MRC voxel size (Å): ({voxel_size_angstrom[0]:g}, "
        f"{voxel_size_angstrom[1]:g}, {voxel_size_angstrom[2]:g})"
    )

    labeled, membrane_desc = relabel_segmentation(
        seg,
        presyn_pts=presyn_pts,
        postsyn_pts=postsyn_pts,
        presyn_az_pts=presyn_az_pts,
        postsyn_az_pts=postsyn_az_pts,
        vesicle_pts=vesicle_pts,
        distance_nm=float(distance_nm),
        voxel_size_nm=float(voxel_size_nm),
        origin_nm=origin_nm,
        membrane_threshold=membrane_threshold,
    )
    print(f"Membrane mask: {membrane_desc}")

    counts = _label_counts(labeled)
    print("Output label counts:")
    for label_id in sorted(counts):
        name = LABEL_NAMES.get(label_id, "unknown")
        print(f"  {label_id} ({name}): {counts[label_id]:,} voxels")

    write_mrc_with_voxel_size(
        out_path,
        labeled.astype(np.int16),
        voxel_size_angstrom=voxel_size_angstrom,
    )
    print(f"Wrote relabeled segmentation -> {out_path}")

    if write_separate_masks:
        separate_paths = write_separate_label_mrcs(
            labeled,
            out_dir,
            tomogram_name=tomogram_name,
            voxel_size_angstrom=voxel_size_angstrom,
        )
        for path in separate_paths:
            print(f"Wrote separate label MRC -> {path}")

    save_xy_projection_png(
        labeled,
        png_path,
        title=(
            f"{tomogram_name}\nMemBrain relabel XY projection "
            f"(distance ≤ {distance_nm:g} nm)"
        ),
    )
    print(f"Wrote XY projection PNG -> {png_path}")
    return out_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Relabel MemBrain segmentation MRC using annotation distance constraints.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help=(
            "STT tomograms.csv batch file (columns: tomoname, set, alignment_dir, ...). "
            "Paths are <data-dir>/<set>/TOP_TOMOS/<tomoname>."
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Data root for CSV batch mode (default: $TOMO_ROOT_BASE or 'data')",
    )
    parser.add_argument(
        "--set",
        dest="set_name",
        default=None,
        help="Optional set filter when using --csv",
    )
    parser.add_argument(
        "--tomogram-path",
        type=Path,
        default=None,
        help="Single tomogram root directory (contains alignment_dir/aunps, etc.)",
    )
    parser.add_argument(
        "--alignment-dir",
        default="best_alignment",
        help="Alignment subdirectory name (single-tomogram mode)",
    )
    parser.add_argument(
        "--segmentation-mrc",
        type=Path,
        default=None,
        help="Input MemBrain segmentation MRC (single-tomogram only; default: auto-discover)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory override (single-tomogram only; "
            "default / CSV mode: <alignment_dir>/STT_results/membranes_labeled)"
        ),
    )
    parser.add_argument(
        "--output-mrc",
        type=Path,
        default=None,
        help="Optional explicit path for the combined relabeled MRC (single-tomogram only)",
    )
    parser.add_argument(
        "--distance-nm",
        type=float,
        default=5.0,
        help="Distance threshold in nm for assigning subclass labels",
    )
    parser.add_argument(
        "--voxel-size-nm",
        type=float,
        default=1.0,
        help="Voxel edge length in nm (1.0 => 5 nm distance ≈ 5 voxels)",
    )
    parser.add_argument(
        "--origin-nm",
        type=str,
        default="0,0,0",
        help="Subtract this X,Y,Z offset (nm) from annotation coordinates before voxel indexing",
    )
    parser.add_argument(
        "--membrane-threshold",
        type=float,
        default=None,
        help=(
            "Threshold for soft MemBrain maps (voxels > threshold = membrane). "
            "Default: auto (hard 0/1 maps use seg==1; soft maps use >150; "
            "probability maps with max≤1.5 use >0.5)."
        ),
    )
    parser.add_argument(
        "--skip-vesicles",
        action="store_true",
        help="Do not assign label 6 (vesicles)",
    )
    parser.add_argument(
        "--skip-active-zones",
        action="store_true",
        help="Do not assign labels 4/5 (active zones)",
    )
    parser.add_argument(
        "--write-separate-masks",
        action="store_true",
        help="Also write one binary MRC per present label (default: only the multi-label MRC)",
    )
    parser.add_argument(
        "--presyn-glb",
        type=Path,
        default=None,
        help="Override presynaptic membrane GLB path (single-tomogram only)",
    )
    parser.add_argument(
        "--postsyn-glb",
        type=Path,
        default=None,
        help="Override postsynaptic membrane GLB path (single-tomogram only)",
    )
    parser.add_argument(
        "--vesicle-glb",
        type=Path,
        default=None,
        help="Override vesicle GLB path (single-tomogram only)",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="In CSV batch mode, abort on the first failure (default: continue)",
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Recompute even if relabeled MRC + PNG already exist",
    )
    args = parser.parse_args(argv)

    if bool(args.csv) == bool(args.tomogram_path):
        parser.error("Provide exactly one of --csv or --tomogram-path")

    origin_nm = _parse_origin(args.origin_nm)
    common_kwargs = dict(
        distance_nm=float(args.distance_nm),
        voxel_size_nm=float(args.voxel_size_nm),
        origin_nm=origin_nm,
        membrane_threshold=args.membrane_threshold,
        skip_vesicles=bool(args.skip_vesicles),
        skip_active_zones=bool(args.skip_active_zones),
        write_separate_masks=bool(args.write_separate_masks),
        rerun=bool(args.rerun),
    )

    if args.tomogram_path:
        process_one_tomogram(
            Path(args.tomogram_path),
            require_alignment_dir(args.alignment_dir),
            segmentation_mrc=args.segmentation_mrc,
            output_dir=args.output_dir,
            output_mrc=args.output_mrc,
            presyn_glb=args.presyn_glb,
            postsyn_glb=args.postsyn_glb,
            vesicle_glb=args.vesicle_glb,
            **common_kwargs,
        )
        return 0

    if args.output_dir or args.output_mrc or args.segmentation_mrc:
        print(
            "Note: --output-dir / --output-mrc / --segmentation-mrc are ignored in "
            "--csv batch mode (each tomogram uses its own membrain/ and "
            "STT_results/membranes_labeled/)."
        )
    if args.presyn_glb or args.postsyn_glb or args.vesicle_glb:
        print("Note: GLB path overrides are ignored in --csv batch mode.")

    data_dir = Path(args.data_dir) if args.data_dir else default_data_root()
    jobs = load_csv_jobs(Path(args.csv), data_dir=data_dir, set_name=args.set_name)
    if not jobs:
        print(f"No tomogram rows to process from {args.csv}", file=sys.stderr)
        return 1

    print(f"Batch: {len(jobs)} tomogram(s) from {args.csv} (data root: {data_dir})")
    ok = 0
    failed: list[tuple[str, str]] = []
    for i, (tomogram_path, alignment_dir, set_name) in enumerate(jobs, start=1):
        label = f"{tomogram_path.name}__{alignment_dir}"
        print(f"\n{'=' * 60}")
        print(f"[{i}/{len(jobs)}] {set_name} / {label}")
        print(f"{'=' * 60}")
        try:
            process_one_tomogram(tomogram_path, alignment_dir, **common_kwargs)
            ok += 1
        except Exception as exc:
            print(f"FAILED {label}: {exc}")
            failed.append((label, str(exc)))
            if args.stop_on_error:
                raise

    print(f"\nDone. Succeeded: {ok}/{len(jobs)}")
    if failed:
        print(f"Failed ({len(failed)}):")
        for label, msg in failed:
            print(f"  {label}: {msg}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())