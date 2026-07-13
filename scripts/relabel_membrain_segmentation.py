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

Reference geometry is loaded in tomogram coordinates (nm), matching STT / aunps annotations.
By default 1 voxel = 1 nm (--voxel-size-nm 1.0), so --distance-nm 5 corresponds to 5 voxels.

Example:
  PYTHONPATH=src python scripts/relabel_membrain_segmentation.py \\
    --tomogram-path data/15F1/TOP_TOMOS/20231017_EGmilled24-2_68 \\
    --alignment-dir best_alignment \\
    --segmentation-mrc best_alignment/membrain/20231017_EGmilled24-2_68_full_rec_BP_3DCTF_BIN4_ddw_MemBrain_seg_v10_alpha.ckpt_segmented_smooth.mrc \\
    --distance-nm 5 \\
    --output-mrc best_alignment/membrain/20231017_EGmilled24-2_68_relabeled.mrc
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Optional

import mrcfile
import numpy as np
import trimesh
from scipy.ndimage import distance_transform_edt

# Allow running from repo root without install (script is standalone; no package __init__ import).
_REPO_ROOT = Path(__file__).resolve().parent.parent


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
) -> np.ndarray:
    """Apply distance-based relabeling; ``seg`` is the original MemBrain labels."""
    seg = np.asarray(seg)
    out = np.zeros(seg.shape, dtype=np.uint8)
    out[seg == 0] = 0

    membrane = seg == 1
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

    return out


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
    if not candidates:
        return None
    preferred = [p for p in candidates if "seg" in p.name.lower()]
    return preferred[0] if preferred else candidates[0]


def _label_counts(labels: np.ndarray) -> dict[int, int]:
    unique, counts = np.unique(labels, return_counts=True)
    return {int(u): int(c) for u, c in zip(unique, counts)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Relabel MemBrain segmentation MRC using annotation distance constraints.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--tomogram-path",
        type=Path,
        required=True,
        help="Tomogram root directory (contains alignment_dir/aunps, etc.)",
    )
    parser.add_argument(
        "--alignment-dir",
        default="best_alignment",
        help="Alignment subdirectory name",
    )
    parser.add_argument(
        "--segmentation-mrc",
        type=Path,
        default=None,
        help="Input MemBrain segmentation MRC (default: auto-discover under membrain/)",
    )
    parser.add_argument(
        "--output-mrc",
        type=Path,
        default=None,
        help="Output relabeled MRC (default: <input>_relabeled.mrc beside input)",
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
        "--presyn-glb",
        type=Path,
        default=None,
        help="Override presynaptic membrane GLB path",
    )
    parser.add_argument(
        "--postsyn-glb",
        type=Path,
        default=None,
        help="Override postsynaptic membrane GLB path",
    )
    parser.add_argument(
        "--vesicle-glb",
        type=Path,
        default=None,
        help="Override vesicle GLB path (default: aunps/synapticvesicles.glb or *.txt)",
    )
    args = parser.parse_args(argv)

    tomogram_path = Path(args.tomogram_path)
    alignment_dir = require_alignment_dir(args.alignment_dir)
    aunps_dir = tomogram_path / alignment_dir / "aunps"
    if not aunps_dir.is_dir():
        raise FileNotFoundError(f"AuNPs directory not found: {aunps_dir}")

    seg_path = Path(args.segmentation_mrc) if args.segmentation_mrc else _discover_segmentation_mrc(
        tomogram_path, alignment_dir
    )
    if seg_path is None or not seg_path.is_file():
        raise FileNotFoundError(
            "Segmentation MRC not found. Pass --segmentation-mrc or place a .mrc under membrain/."
        )

    out_path = (
        Path(args.output_mrc)
        if args.output_mrc
        else seg_path.with_name(f"{seg_path.stem}_relabeled.mrc")
    )
    origin_nm = _parse_origin(args.origin_nm)

    print(f"Tomogram:      {tomogram_path}")
    print(f"Alignment:     {alignment_dir}")
    print(f"Segmentation:  {seg_path}")
    print(f"Output:        {out_path}")
    print(f"Distance:      {args.distance_nm} nm")
    print(f"Voxel size:    {args.voxel_size_nm} nm")
    print(f"Origin (nm):   {origin_nm}")

    if args.presyn_glb:
        presyn_pts = load_glb_vertices(Path(args.presyn_glb))
    else:
        presyn_pts = load_membrane_points(aunps_dir, "presynaptic")
    if args.postsyn_glb:
        postsyn_pts = load_glb_vertices(Path(args.postsyn_glb))
    else:
        postsyn_pts = load_membrane_points(aunps_dir, "postsynaptic")

    if args.vesicle_glb:
        vesicle_pts = load_glb_vertices(Path(args.vesicle_glb))
    elif args.skip_vesicles:
        vesicle_pts = np.zeros((0, 3))
    else:
        vesicle_pts = load_vesicle_points(aunps_dir)

    if args.skip_active_zones:
        presyn_az_pts = np.zeros((0, 3))
        postsyn_az_pts = np.zeros((0, 3))
    else:
        presyn_az_pts = load_active_zone_side_points(tomogram_path, alignment_dir, "presynaptic")
        postsyn_az_pts = load_active_zone_side_points(tomogram_path, alignment_dir, "postsynaptic")

    print(f"Reference points: presyn membrane {len(presyn_pts):,}, postsyn membrane {len(postsyn_pts):,}")
    print(f"                  presyn AZ {len(presyn_az_pts):,}, postsyn AZ {len(postsyn_az_pts):,}")
    print(f"                  vesicles {len(vesicle_pts):,}")

    with mrcfile.open(seg_path, permissive=True) as mrc:
        seg = np.asarray(mrc.data)

    print(f"Volume shape (Z,Y,X): {seg.shape}; unique input labels: {np.unique(seg)}")

    labeled = relabel_segmentation(
        seg,
        presyn_pts=presyn_pts,
        postsyn_pts=postsyn_pts,
        presyn_az_pts=presyn_az_pts,
        postsyn_az_pts=postsyn_az_pts,
        vesicle_pts=vesicle_pts,
        distance_nm=float(args.distance_nm),
        voxel_size_nm=float(args.voxel_size_nm),
        origin_nm=origin_nm,
    )

    counts = _label_counts(labeled)
    print("Output label counts:")
    for label_id in sorted(counts):
        name = LABEL_NAMES.get(label_id, "unknown")
        print(f"  {label_id} ({name}): {counts[label_id]:,} voxels")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    mrcfile.write(out_path, labeled.astype(np.int16), overwrite=True)

    print(f"Wrote relabeled segmentation -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
