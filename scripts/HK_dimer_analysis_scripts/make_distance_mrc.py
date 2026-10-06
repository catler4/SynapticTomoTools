#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a 3D "distance heatmap" MRC volume from pre/post membrane point clouds.

Instead of a 2D projection PNG, this writes an actual MRC file on the SAME
grid (shape + voxel size + origin) as a reference tomogram, where each voxel
holds the nearest 3D distance from the post membrane to the pre membrane
(splatted + Gaussian-blurred so nearby empty voxels get smooth values too).

This MRC can be opened directly in ChimeraX alongside a membrane surface
model built from the same tomogram, and colored onto that surface with:

    open distance_map.mrc
    color sample #<surface_model_id> map #<distance_map_model_id> \\
        palette blue:white:red range 0,30

("color sample" reads the map's value at each surface vertex position and
colors the surface by it -- this is the ChimeraX equivalent of "color by
property" for a scalar field defined on a volume grid.)

Usage:
    python make_distance_mrc.py \\
        --pre pre_membrane.txt \\
        --post post_membrane.txt \\
        --reference-mrc tomogram_bin4.mrc \\
        --out-mrc distance_map.mrc \\
        --sigma 3.0
"""

import argparse
from pathlib import Path
from typing import List

import numpy as np
import mrcfile
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


# ============================================================
# IO
# ============================================================
def load_xyz_txt(file_path: Path) -> np.ndarray:
    coords: List[List[float]] = []
    with file_path.open("r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            if len(parts) < 3:
                continue
            try:
                x, y, z = map(float, parts[:3])
            except Exception:
                continue
            coords.append([x, y, z])
    if coords:
        return np.asarray(coords, dtype=float)
    return np.zeros((0, 3), dtype=float)


# ============================================================
# Distance computation (post -> nearest pre point)
# ============================================================
def compute_post_to_pre_distance(post_xyz: np.ndarray, pre_xyz: np.ndarray) -> np.ndarray:
    """Nearest 3D Euclidean distance from every post point to the pre point cloud."""
    if post_xyz.size == 0:
        return np.zeros((0,), dtype=float)
    if pre_xyz.size == 0:
        return np.full((len(post_xyz),), np.nan, dtype=float)

    tree = cKDTree(pre_xyz)
    dists, _ = tree.query(post_xyz, k=1)
    return dists.astype(float)


# ============================================================
# Remove small disconnected point clusters (floating fragments)
# ============================================================
def remove_small_clusters(
    post_xyz: np.ndarray,
    cluster_radius: float,
    min_cluster_size: int,
    keep_largest_only: bool,
) -> np.ndarray:
    """
    Group post_xyz points into connected components: two points are
    'connected' if they are within cluster_radius (voxels) of each other,
    transitively (i.e. a chain of nearby points forms one cluster).

    Returns a boolean keep-mask of length len(post_xyz).

    - If keep_largest_only: only the single biggest connected component survives.
    - Else: any component with fewer than min_cluster_size points is dropped.
    """
    n = len(post_xyz)
    if n == 0:
        return np.zeros((0,), dtype=bool)

    tree = cKDTree(post_xyz)
    pairs = tree.query_pairs(r=cluster_radius, output_type="ndarray")

    if len(pairs) == 0:
        # no connections at all -> every point is isolated (its own cluster of size 1)
        labels = np.arange(n)
        n_components = n
    else:
        row, col = pairs[:, 0], pairs[:, 1]
        data = np.ones(len(row), dtype=np.uint8)
        graph = csr_matrix((data, (row, col)), shape=(n, n))
        n_components, labels = connected_components(csgraph=graph, directed=False)

    sizes = np.bincount(labels, minlength=n_components)

    if keep_largest_only:
        biggest = int(np.argmax(sizes))
        keep_component = np.zeros(n_components, dtype=bool)
        keep_component[biggest] = True
    else:
        keep_component = sizes >= min_cluster_size

    keep_mask = keep_component[labels]

    print(f"[INFO] cluster filter: {n_components} connected component(s) found "
          f"(radius={cluster_radius} voxels); sizes range {sizes.min()}-{sizes.max()}")
    return keep_mask


# ============================================================
# 3D splat + blur volume construction
# ============================================================
def build_distance_volume(
    post_xyz: np.ndarray,
    distances: np.ndarray,
    shape_zyx,
    sigma: float,
    percentile_clip: float,
    empty_value: float,
) -> np.ndarray:
    """
    Splat (x,y,z) distance values onto a (nz,ny,nx) grid, then Gaussian-blur
    both the value-sum and the count separately (same 'blurred average'
    trick used for the 2D heatmap PNGs), so voxels near real data points get
    smoothly-interpolated values while far-away empty regions stay flagged.

    Returns a float32 volume of shape shape_zyx. Voxels with no nearby
    coverage are set to `empty_value` (e.g. np.nan).
    """
    nz, ny, nx = shape_zyx
    sum_map = np.zeros((nz, ny, nx), dtype=np.float32)
    cnt_map = np.zeros((nz, ny, nx), dtype=np.float32)

    if len(post_xyz) == 0:
        return np.full((nz, ny, nx), empty_value, dtype=np.float32)

    x = np.rint(post_xyz[:, 0]).astype(int)
    y = np.rint(post_xyz[:, 1]).astype(int)
    z = np.rint(post_xyz[:, 2]).astype(int)

    keep = (
        (x >= 0) & (x < nx) &
        (y >= 0) & (y < ny) &
        (z >= 0) & (z < nz) &
        np.isfinite(distances)
    )
    x, y, z, v = x[keep], y[keep], z[keep], distances[keep].astype(np.float32)

    if len(v) == 0:
        print("[WARNING] No post points fell inside the reference volume bounds "
              "(or all distances were non-finite) -- output will be entirely empty.")
        return np.full((nz, ny, nx), empty_value, dtype=np.float32)

    np.add.at(sum_map, (z, y, x), v)
    np.add.at(cnt_map, (z, y, x), 1.0)

    sum_blur = gaussian_filter(sum_map, sigma=sigma)
    cnt_blur = gaussian_filter(cnt_map, sigma=sigma)

    with np.errstate(invalid="ignore", divide="ignore"):
        volume = sum_blur / np.maximum(cnt_blur, 1e-6)

    covered = cnt_blur > 1e-6

    if np.any(covered) and percentile_clip is not None:
        vmax_clip = np.percentile(volume[covered], percentile_clip)
        volume = np.clip(volume, None, vmax_clip)

    volume[~covered] = empty_value

    return volume.astype(np.float32)


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Build a 3D post->pre membrane distance MRC volume for ChimeraX 'color sample'."
    )

    parser.add_argument("--pre", required=True, type=str, help="Pre membrane txt (x y z, voxel coords)")
    parser.add_argument("--post", required=True, type=str, help="Post membrane txt (x y z, voxel coords)")
    parser.add_argument("--reference-mrc", required=True, type=str,
                        help="Tomogram MRC whose grid (shape, voxel size, origin) the output volume will match.")
    parser.add_argument("--out-mrc", required=True, type=str, help="Output distance-map MRC path")

    parser.add_argument("--sigma", type=float, default=3.0,
                        help="Gaussian blur sigma in voxels (3D). Larger = smoother/more filled-in. (default: 3.0)")
    parser.add_argument("--percentile_clip", type=float, default=99.0,
                        help="Clip extreme high distance values at this percentile (default: 99.0). "
                             "Use a negative value to disable clipping.")
    parser.add_argument("--empty_value", type=float, default=np.nan,
                        help="Value to assign to voxels with no nearby post-membrane coverage. "
                             "Default: NaN (ChimeraX 'color sample' typically leaves these uncolored). "
                             "Pass a specific number (e.g. -1) if your workflow needs a finite sentinel instead.")
    parser.add_argument("--coords_are_centered", action="store_true",
                        help="Set this if pre/post txt coordinates are relative to the tomogram's CENTER "
                             "(e.g. Warp/Dynamo/RELION style, where (0,0,0) = volume center, values can be "
                             "negative) rather than the array corner (0,0,0) = first voxel. When set, "
                             "coordinates are shifted by +size/2 (using the reference mrc's shape) before "
                             "being placed on the voxel grid.")
    parser.add_argument("--flip_x", action="store_true",
                        help="Mirror the X axis (x -> nx-1-x) before placing points. Use to test/fix an "
                             "axis-direction mismatch (e.g. different tools disagreeing on which corner is 0,0,0).")
    parser.add_argument("--flip_y", action="store_true",
                        help="Mirror the Y axis (y -> ny-1-y). Common when image-row convention (y down) "
                             "vs Cartesian (y up) differ between the tool that made the txt and the mrc.")
    parser.add_argument("--flip_z", action="store_true",
                        help="Mirror the Z axis (z -> nz-1-z). Common if the coordinates were picked on a "
                             "tomogram reconstructed with a different Z-flip convention "
                             "(e.g. AreTomo3 '-FlipVol 1' applied on one version but not the other).")
    parser.add_argument("--shift_x", type=float, default=0.0,
                        help="Add this many voxels to every X coordinate (can be negative). Use to test/fix "
                             "a plain translation offset between the txt coordinate frame and the reference mrc.")
    parser.add_argument("--shift_y", type=float, default=0.0,
                        help="Add this many voxels to every Y coordinate (can be negative).")
    parser.add_argument("--shift_z", type=float, default=0.0,
                        help="Add this many voxels to every Z coordinate (can be negative).")
    parser.add_argument("--max_dist", type=float, default=40.0,
                        help="Only keep post points whose nearest distance to the pre membrane is "
                             "<= this value (voxels). Post points farther than this (no nearby pre "
                             "membrane) are dropped entirely -- not splatted into the volume at all. "
                             "Use a negative value to disable this filter and keep all post points. "
                             "(default: 40.0)")
    parser.add_argument("--cluster_radius", type=float, default=5.0,
                        help="Two post points are considered 'connected' (same cluster) if they are "
                             "within this many voxels of each other. Used by --min_cluster_size / "
                             "--keep_largest_only to drop small floating fragments. (default: 5.0)")
    parser.add_argument("--min_cluster_size", type=int, default=0,
                        help="Drop any connected component of post points smaller than this many points "
                             "(e.g. a small floating mesh fragment). 0 = disabled (default). "
                             "Try 20-50 as a starting point, or check the printed cluster sizes and adjust.")
    parser.add_argument("--keep_largest_only", action="store_true",
                        help="Ignore --min_cluster_size and keep ONLY the single largest connected "
                             "component of post points, dropping every other fragment no matter its size.")

    args = parser.parse_args()

    pre_path = Path(args.pre).expanduser().resolve()
    post_path = Path(args.post).expanduser().resolve()
    ref_path = Path(args.reference_mrc).expanduser().resolve()
    out_path = Path(args.out_mrc).expanduser().resolve()

    for p in [pre_path, post_path, ref_path]:
        if not p.exists():
            raise FileNotFoundError(f"Not found: {p}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    percentile_clip = None if args.percentile_clip is not None and args.percentile_clip < 0 else args.percentile_clip

    print(f"[INFO] pre            : {pre_path}")
    print(f"[INFO] post           : {post_path}")
    print(f"[INFO] reference_mrc  : {ref_path}")
    print(f"[INFO] out_mrc        : {out_path}")
    print(f"[INFO] sigma          : {args.sigma}")
    print(f"[INFO] percentile_clip: {percentile_clip}")
    print(f"[INFO] empty_value    : {args.empty_value}")

    pre_xyz = load_xyz_txt(pre_path)
    post_xyz = load_xyz_txt(post_path)
    print(f"[INFO] loaded pre  coords: {len(pre_xyz)}")
    print(f"[INFO] loaded post coords: {len(post_xyz)}")

    with mrcfile.open(ref_path, permissive=True) as ref:
        shape_zyx = ref.data.shape  # (nz, ny, nx)
        voxel_size = ref.voxel_size.copy()
        nxstart = ref.header.nxstart
        nystart = ref.header.nystart
        nzstart = ref.header.nzstart
        origin = ref.header.origin.copy() if hasattr(ref.header, "origin") else None

    print(f"[INFO] reference shape (z,y,x): {shape_zyx}")

    if args.coords_are_centered:
        nz, ny, nx = shape_zyx
        shift_xyz = np.array([nx / 2.0, ny / 2.0, nz / 2.0])
        pre_xyz = pre_xyz + shift_xyz
        post_xyz = post_xyz + shift_xyz
        print(f"[INFO] --coords_are_centered set: shifted pre/post coords by +{shift_xyz.tolist()} "
              f"(center -> corner convention)")

    if args.flip_x or args.flip_y or args.flip_z:
        nz, ny, nx = shape_zyx
        if args.flip_x:
            pre_xyz[:, 0] = (nx - 1) - pre_xyz[:, 0]
            post_xyz[:, 0] = (nx - 1) - post_xyz[:, 0]
            print(f"[INFO] --flip_x set: mirrored X axis (nx-1={nx-1})")
        if args.flip_y:
            pre_xyz[:, 1] = (ny - 1) - pre_xyz[:, 1]
            post_xyz[:, 1] = (ny - 1) - post_xyz[:, 1]
            print(f"[INFO] --flip_y set: mirrored Y axis (ny-1={ny-1})")
        if args.flip_z:
            pre_xyz[:, 2] = (nz - 1) - pre_xyz[:, 2]
            post_xyz[:, 2] = (nz - 1) - post_xyz[:, 2]
            print(f"[INFO] --flip_z set: mirrored Z axis (nz-1={nz-1})")

    if args.shift_x != 0.0 or args.shift_y != 0.0 or args.shift_z != 0.0:
        shift = np.array([args.shift_x, args.shift_y, args.shift_z])
        pre_xyz = pre_xyz + shift
        post_xyz = post_xyz + shift
        print(f"[INFO] applied manual shift (x,y,z) = {shift.tolist()} voxels")

    distances = compute_post_to_pre_distance(post_xyz, pre_xyz)
    n_finite = int(np.sum(np.isfinite(distances)))
    print(f"[INFO] computed distances: {len(distances)} (finite: {n_finite})")

    if args.max_dist is not None and args.max_dist >= 0:
        keep = np.isfinite(distances) & (distances <= args.max_dist)
        n_before = len(post_xyz)
        post_xyz = post_xyz[keep]
        distances = distances[keep]
        n_after = len(post_xyz)
        print(f"[INFO] --max_dist {args.max_dist}: kept {n_after}/{n_before} post points "
              f"(dropped {n_before - n_after} with no pre membrane within range)")
    else:
        print("[INFO] --max_dist disabled (negative value): keeping all post points")

    if args.keep_largest_only or args.min_cluster_size > 0:
        n_before = len(post_xyz)
        keep_cluster = remove_small_clusters(
            post_xyz=post_xyz,
            cluster_radius=args.cluster_radius,
            min_cluster_size=args.min_cluster_size,
            keep_largest_only=args.keep_largest_only,
        )
        post_xyz = post_xyz[keep_cluster]
        distances = distances[keep_cluster]
        n_after = len(post_xyz)
        print(f"[INFO] cluster filter: kept {n_after}/{n_before} post points "
              f"(dropped {n_before - n_after} in small/disconnected fragments)")

    volume = build_distance_volume(
        post_xyz=post_xyz,
        distances=distances,
        shape_zyx=shape_zyx,
        sigma=args.sigma,
        percentile_clip=percentile_clip,
        empty_value=args.empty_value,
    )

    n_covered = int(np.sum(np.isfinite(volume))) if np.isnan(args.empty_value) else volume.size
    print(f"[INFO] output volume shape: {volume.shape}, dtype: {volume.dtype}")
    if np.isnan(args.empty_value):
        print(f"[INFO] voxels with real (non-NaN) data: {n_covered} / {volume.size}")

    with mrcfile.new(out_path, overwrite=True) as mrc_out:
        mrc_out.set_data(volume)
        mrc_out.voxel_size = voxel_size
        mrc_out.header.nxstart = nxstart
        mrc_out.header.nystart = nystart
        mrc_out.header.nzstart = nzstart
        if origin is not None:
            mrc_out.header.origin = origin

        # NOTE: mrc_out.update_header_stats() does NOT ignore NaN correctly and
        # can leave dmin/dmax as garbage sentinel values (e.g. float64 max/-max),
        # which makes ChimeraX pick a nonsensical default color/intensity range.
        # Compute stats manually over only the finite (non-NaN) voxels instead.
        finite_vals = volume[np.isfinite(volume)]
        if finite_vals.size > 0:
            mrc_out.header.dmin = float(np.min(finite_vals))
            mrc_out.header.dmax = float(np.max(finite_vals))
            mrc_out.header.dmean = float(np.mean(finite_vals))
        else:
            mrc_out.header.dmin = 0.0
            mrc_out.header.dmax = 0.0
            mrc_out.header.dmean = 0.0

    print(f"[OK] saved distance-map MRC: {out_path}")
    print("[ALL DONE]")
    print()
    print("ChimeraX usage example:")
    print(f"    open {ref_path.name}")
    print(f"    open {out_path.name}")
    print("    # after building/opening your membrane surface model as #1, and this map as #2:")
    print("    color sample #1 map #2 palette blue:white:red range 0,30")


if __name__ == "__main__":
    main()