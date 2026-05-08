#!/usr/bin/env python3
"""
Test active-zone GLB pipeline + one mid-Z tomogram overlay.

Uses ``synaptic_tomo_tools.activezone.find_active_zones_from_glb`` (distance gate, then vertex
normal dot for outer vs inner — see that docstring).

The saved PNG overlays **only** active-zone outer and inner points on the tomogram slice (not full
membranes).

Example (conda env ``synaptictomo``):

  conda activate synaptictomo
  pip install -e .

  python -u scripts/test_active_zone_overlay.py \\
    --csv tomogram_csv_files/tomograms-test-15.csv \\
    --root-data-dir data \\
    --output-dir results/active_zone_test
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from synaptic_tomo_tools.activezone import (
    find_active_zones_from_glb,
    import_membrane_segmentations_from_glb,
    save_active_zone_segmentations,
)
from synaptic_tomo_tools.visualization import (
    filter_coords_in_slice,
    load_tomogram_slice,
)


def _load_optional_az_txt(path: Path) -> np.ndarray:
    if not path.exists():
        return np.zeros((0, 3))
    try:
        arr = np.atleast_2d(np.loadtxt(path, delimiter=None))
        if arr.size == 0 or arr.shape[1] < 3:
            return np.zeros((0, 3))
        return arr
    except Exception:
        return np.zeros((0, 3))


def _load_all_active_zone_surfaces(tomo_path: Path, alignment_dir: str):
    """Load every paired *_pre_outer / *_post_outer zone plus inner txts."""
    az_dir = tomo_path / alignment_dir / "STT_results" / "activezone"
    azs_pre, azs_post, azs_pre_inner, azs_post_inner = [], [], [], []
    for pre_file in sorted(az_dir.glob("active_zone_*_pre_outer.txt")):
        zone_name = pre_file.name.replace("_pre_outer.txt", "")
        post_file = az_dir / f"{zone_name}_post_outer.txt"
        if not post_file.exists():
            continue
        pre_coords = np.atleast_2d(np.loadtxt(pre_file))
        post_coords = np.atleast_2d(np.loadtxt(post_file))
        if pre_coords.size == 0 or post_coords.size == 0:
            continue
        azs_pre.append(pre_coords)
        azs_post.append(post_coords)
        azs_pre_inner.append(_load_optional_az_txt(az_dir / f"{zone_name}_pre_inner.txt"))
        azs_post_inner.append(_load_optional_az_txt(az_dir / f"{zone_name}_post_inner.txt"))
    return azs_pre, azs_post, azs_pre_inner, azs_post_inner


def main():
    parser = argparse.ArgumentParser(description="Test active zone GLB path + center-slice overlay")
    parser.add_argument(
        "--csv",
        type=Path,
        default=_REPO_ROOT / "tomogram_csv_files" / "tomograms-test-15.csv",
        help="Tomogram CSV with tomoname, set, alignment_dir",
    )
    parser.add_argument(
        "--root-data-dir",
        type=Path,
        default=_REPO_ROOT / "data",
        help="Dataset root containing <set>/TOP_TOMOS/<tomoname>",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_REPO_ROOT / "results" / "active_zone_test",
        help="Where to write the PNG",
    )
    parser.add_argument(
        "--distance-range",
        type=float,
        nargs=2,
        default=(10.0, 40.0),
        metavar=("MIN_NM", "MAX_NM"),
        help="Active zone distance band (default 10 40)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Less console output",
    )
    args = parser.parse_args()
    verbose = not args.quiet

    df = pd.read_csv(args.csv)
    if df.empty:
        raise SystemExit(f"No rows in {args.csv}")
    row = df.iloc[0]
    tomoname = str(row["tomoname"]).strip()
    set_name = str(row["set"]).strip()
    alignment_dir = str(row["alignment_dir"]).strip()

    tomogram_path = args.root_data_dir / set_name / "TOP_TOMOS" / tomoname
    if not tomogram_path.is_dir():
        raise SystemExit(f"Tomogram directory not found: {tomogram_path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_png = args.output_dir / f"{tomoname}_active_zone_center_slice_overlay.png"

    print(f"Tomogram: {tomogram_path}", flush=True)
    print(f"Alignment: {alignment_dir}", flush=True)
    print(
        "[pipeline] Active zone: synaptic_tomo_tools.activezone.find_active_zones_from_glb",
        flush=True,
    )

    t0 = time.perf_counter()
    print(
        "\n[1/4] GLB → find_active_zones_from_glb → save_active_zone_segmentations…",
        flush=True,
    )
    membranes = import_membrane_segmentations_from_glb(tomogram_path, alignment_dir=alignment_dir)
    if verbose:
        print(
            f"      Loaded {len(membranes['presynaptic'])} presynaptic / "
            f"{len(membranes['postsynaptic'])} postsynaptic GLB mesh(es)",
            flush=True,
        )

    active_zones = find_active_zones_from_glb(
        membranes,
        distance_range=(float(args.distance_range[0]), float(args.distance_range[1])),
    )
    save_active_zone_segmentations(active_zones, tomogram_path, alignment_dir=alignment_dir)
    print(f"      … detection + save done in {time.perf_counter() - t0:.1f}s\n", flush=True)

    if not active_zones.get("active_zones"):
        raise SystemExit("No active zones produced; check GLBs and distance range.")

    t1 = time.perf_counter()
    print("[2/4] Loading tomogram slice (mid-Z) and active zone txt coordinates…", flush=True)
    slice2d, z_center = load_tomogram_slice(tomogram_path, None, alignment_dir=alignment_dir)
    if slice2d is None:
        raise SystemExit(
            f"No ddw.mrc under {tomogram_path / alignment_dir}; cannot render slice overlay."
        )

    azs_pre, azs_post, azs_pre_inner, azs_post_inner = _load_all_active_zone_surfaces(
        tomogram_path, alignment_dir
    )
    print(
        f"      … slice z={z_center}, {len(azs_pre)} active zone pair(s) "
        f"({time.perf_counter() - t1:.1f}s)\n",
        flush=True,
    )

    t2 = time.perf_counter()
    print("[3/4] Filtering points to slice thickness…", flush=True)
    z_thresh_az = 1.0
    azs_pre_xy = filter_coords_in_slice(azs_pre, z_center, z_thresh_az, None)
    azs_post_xy = filter_coords_in_slice(azs_post, z_center, z_thresh_az, None)
    azs_pre_inner_xy = filter_coords_in_slice(azs_pre_inner, z_center, z_thresh_az, None)
    azs_post_inner_xy = filter_coords_in_slice(azs_post_inner, z_center, z_thresh_az, None)
    print(f"      … done ({time.perf_counter() - t2:.1f}s)\n", flush=True)

    t3 = time.perf_counter()
    print("[4/4] Rendering overlay and saving PNG…", flush=True)
    vmin, vmax = np.percentile(slice2d, [2, 98])
    inner_pre_rgb = (1.0, 0.52, 0.52)
    inner_post_rgb = (0.52, 1.0, 0.52)
    inner_az_alpha = 0.06

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(slice2d, cmap="gray", vmin=vmin, vmax=vmax, origin="lower")

    for coords in azs_pre_inner_xy:
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            color=inner_pre_rgb,
            s=3,
            alpha=inner_az_alpha,
            rasterized=True,
        )
    for coords in azs_post_inner_xy:
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            color=inner_post_rgb,
            s=3,
            alpha=inner_az_alpha,
            rasterized=True,
        )

    for coords in azs_pre_xy:
        ax.scatter(coords[:, 0], coords[:, 1], c="red", s=3, alpha=0.12, rasterized=True)
    for coords in azs_post_xy:
        ax.scatter(coords[:, 0], coords[:, 1], c="lime", s=3, alpha=0.12, rasterized=True)

    ax.set_title(
        f"{tomoname} — mid-Z slice z={z_center} nm\n"
        f"Red/green: AZ outer (pre/post) | Pink/light-green: AZ inner"
    )
    ax.set_xlabel("X (nm)")
    ax.set_ylabel("Y (nm)")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_png} ({time.perf_counter() - t3:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
