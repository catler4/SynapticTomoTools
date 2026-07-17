#!/usr/bin/env python3
"""
Compute convex-hull packing density for AuNP DBSCAN clusters already assigned
by ``synaptic_tomo_tools.aunps.analyze_aunps``, and plot a histogram of
nm^2-per-AuNP (convex-hull area / AuNP count) across clusters.

Reads per-AuNP cluster assignments from each tomogram's
``<alignment_dir>/STT_results/aunps/aunp_clusters.star`` (written by
``analyze_aunps``, column ``aunp_cluster``) instead of rerunning DBSCAN, then
applies the same convex-hull density calculation as the ``aunp_clusters.csv``
block of ``analyze_aunps``.

Single-tomogram mode:

  python -u scripts/test_aunp_cluster_density_histogram.py \\
    --tomo-path /nrs/elferich/gouaux_tomo/15F1/TOP_TOMOS/20241030_AMmilled12-1_15 \\
    --alignment-dir patch_tracking \\
    --output-dir results/aunp_cluster_density_test

Batch mode over many tomograms/alignment dirs, discovered from a glob of
``aunp_clusters.star`` files (each match's grandparent directories give the
tomogram path and alignment dir):

  python -u scripts/test_aunp_cluster_density_histogram.py \\
    --glob "/nrs/elferich/gouaux_tomo/15F1/TOP_TOMOS/*/*/STT_results/aunps/aunp_clusters.star" \\
    --output-dir results/aunp_cluster_density_test
"""

from __future__ import annotations

import argparse
import glob as glob_module
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import starfile
from matplotlib import pyplot as plt
from scipy.spatial import ConvexHull, distance_matrix

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

COORD_COLS = ["faCoordinateX", "faCoordinateY", "faCoordinateZ"]
CLUSTER_STAR_RELPATH = Path("STT_results") / "aunps" / "aunp_clusters.star"


def compute_cluster_density_summary(
    coords: np.ndarray,
    cluster_labels: np.ndarray,
) -> pd.DataFrame:
    """
    Per-cluster AuNP count, convex-hull area (nm^2), and packing density
    (AuNPs per nm^2) for each non-noise cluster label.

    ``coords`` are AuNP 3D coordinates (nm); the convex hull is computed in
    3D and its surface area halved to approximate the single-sided area of
    the (roughly membrane-shaped) AuNP point cloud. Clusters with fewer than
    3 points get NaN area/density (a hull needs >= 3 points).
    """
    cluster_rows = []
    for label in np.unique(cluster_labels):
        if label == -1:
            continue  # Skip noise
        cluster_points = coords[cluster_labels == label]
        n_points = len(cluster_points)
        if n_points < 3:
            area = np.nan
            max_dim = np.nan
        else:
            try:
                hull = ConvexHull(cluster_points)
                area = hull.area / 2.0
            except Exception:
                area = np.nan
            try:
                dists = distance_matrix(cluster_points, cluster_points)
                max_dim = np.nanmax(dists)
            except Exception:
                max_dim = np.nan
        density = n_points / area if area and area > 0 else np.nan
        area_per_aunp = area / n_points if area and n_points > 0 else np.nan
        cluster_rows.append({
            "cluster_label": label,
            "n_aunps": n_points,
            "cluster_area_nm2": area,
            "cluster_max_dimension_nm": max_dim,
            "cluster_density_aunps_per_nm2": density,
            "cluster_area_per_aunp_nm2": area_per_aunp,
        })
    columns = [
        "cluster_label",
        "n_aunps",
        "cluster_area_nm2",
        "cluster_max_dimension_nm",
        "cluster_density_aunps_per_nm2",
        "cluster_area_per_aunp_nm2",
    ]
    return pd.DataFrame(cluster_rows, columns=columns)


def load_cluster_star_dataframe(star_path: Path) -> Optional[pd.DataFrame]:
    """Read an ``aunp_clusters.star`` file and return its DataFrame (or None)."""
    if not star_path.is_file():
        return None
    star_data = starfile.read(star_path)
    if isinstance(star_data, dict):
        for v in star_data.values():
            if isinstance(v, pd.DataFrame):
                return v
        return None
    if isinstance(star_data, pd.DataFrame):
        return star_data
    return None


def discover_tomogram_jobs(pattern: str) -> List[Tuple[Path, str]]:
    """
    Discover (tomogram_path, alignment_dir) pairs from a glob of
    ``aunp_clusters.star`` files at
    ``<tomo_path>/<alignment_dir>/STT_results/aunps/aunp_clusters.star``.
    """
    pairs = set()
    for f in glob_module.glob(pattern):
        aunps_dir = Path(f).resolve().parent  # STT_results/aunps
        stt_results_dir = aunps_dir.parent  # STT_results
        alignment_dir_path = stt_results_dir.parent  # <alignment_dir>
        tomo_path = alignment_dir_path.parent  # <tomo_path>
        pairs.add((tomo_path, alignment_dir_path.name))
    return sorted(pairs, key=lambda x: (str(x[0]), x[1]))


def write_histogram(areas_per_aunp: np.ndarray, png_path: Path, title: str, n_bins: int) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(areas_per_aunp, bins=min(n_bins, max(5, len(areas_per_aunp))), color="#4C72B0", edgecolor="white")
    ax.set_xlabel("AuNP density (nm² / AuNP)")
    ax.set_ylabel("Cluster count")
    ax.set_title(title)
    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {png_path}")


def _project_to_plane(points: np.ndarray) -> np.ndarray:
    """Project 3D points onto their best-fit 2D plane (top 2 PCA components), for visualization only."""
    centered = points - points.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[:2].T


def select_representative_clusters(pool: List[dict], n: int) -> List[dict]:
    """
    Evenly sample ``n`` clusters spanning the range of ``cluster_area_per_aunp_nm2``
    in ``pool``, so the diagnostic plot shows tightly- to loosely-packed examples
    rather than an arbitrary/biased subset.
    """
    valid = [e for e in pool if np.isfinite(e["area_per_aunp_nm2"])]
    if not valid:
        return []
    valid.sort(key=lambda e: e["area_per_aunp_nm2"])
    if len(valid) <= n:
        return valid
    idxs = sorted({int(round(i)) for i in np.linspace(0, len(valid) - 1, n)})
    return [valid[i] for i in idxs]


def write_cluster_hull_diagnostic(entries: List[dict], png_path: Path, n_cols: int = 5) -> None:
    """Grid of projected AuNP clusters with their convex-hull outline, for visual sanity-checking."""
    if not entries:
        return
    n = len(entries)
    n_cols = min(n_cols, n)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.8 * n_cols, 2.8 * n_rows), squeeze=False)
    axes_flat = axes.ravel()

    for ax, entry in zip(axes_flat, entries):
        proj = _project_to_plane(entry["points"])
        ax.scatter(proj[:, 0], proj[:, 1], s=14, color="#4C72B0", zorder=3)
        if len(proj) >= 3:
            try:
                hull = ConvexHull(proj)
                hull_pts = proj[hull.vertices]
                hull_pts = np.vstack([hull_pts, hull_pts[:1]])
                ax.plot(hull_pts[:, 0], hull_pts[:, 1], color="#DD8452", linewidth=1.5, zorder=2)
                ax.fill(hull_pts[:, 0], hull_pts[:, 1], color="#DD8452", alpha=0.15, zorder=1)
            except Exception:
                pass
        ax.set_title(
            f"{entry['tomo_name']}\n{entry['alignment_dir']} #{entry['cluster_label']}\n"
            f"n={entry['n_aunps']}, {entry['area_per_aunp_nm2']:.0f} nm²/AuNP",
            fontsize=7,
        )
        ax.set_aspect("equal", adjustable="datalim")
        ax.tick_params(labelsize=6)

    for ax in axes_flat[len(entries):]:
        ax.axis("off")

    fig.suptitle("Representative AuNP clusters (PCA-projected convex hulls)", fontsize=11)
    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {png_path}")


def compute_clusters_for_tomogram(
    tomo_path: Path,
    alignment_dir: str,
    *,
    min_cluster_aunps: int = 1,
) -> Optional[Tuple[pd.DataFrame, Dict[int, np.ndarray]]]:
    """
    Load one tomogram's pre-computed cluster assignments and return its density
    summary plus a ``{cluster_label: points}`` map for the clusters that pass
    the ``min_cluster_aunps`` filter (or None if no usable data).
    """
    tomo_name = tomo_path.name
    star_path = tomo_path / alignment_dir / CLUSTER_STAR_RELPATH
    df = load_cluster_star_dataframe(star_path)
    if df is None:
        print(f"  {tomo_name} ({alignment_dir}): no aunp_clusters.star found at {star_path}")
        return None
    if "aunp_cluster" not in df.columns:
        print(f"  {tomo_name} ({alignment_dir}): 'aunp_cluster' column not found in {star_path}")
        return None

    coords = np.asarray(df[COORD_COLS]).astype(float)
    cluster_labels = pd.to_numeric(df["aunp_cluster"], errors="coerce").fillna(-1).astype(int).to_numpy()

    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)

    cluster_df = compute_cluster_density_summary(coords, cluster_labels)
    n_before = len(cluster_df)
    cluster_df = cluster_df[cluster_df["n_aunps"] >= min_cluster_aunps].reset_index(drop=True)
    print(
        f"  {tomo_name} ({alignment_dir}): {n_clusters} AuNP clusters (from {star_path.name}), "
        f"{len(cluster_df)}/{n_before} with >= {min_cluster_aunps} AuNPs"
    )

    cluster_df["tomogram_name"] = tomo_name
    cluster_df["alignment_dir"] = alignment_dir

    points_by_label = {
        int(label): coords[cluster_labels == label]
        for label in cluster_df["cluster_label"]
    }
    return cluster_df, points_by_label


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AuNP cluster convex-hull density histogram(s), using existing analyze_aunps cluster assignments"
    )
    parser.add_argument("--tomo-path", type=Path, help="Tomogram directory (single-tomogram mode)")
    parser.add_argument("--alignment-dir", type=str, help="Alignment subdirectory name (single-tomogram mode)")
    parser.add_argument(
        "--glob",
        type=str,
        help=(
            "Glob pattern matching aunp_clusters.star files, used to discover "
            "(tomogram, alignment_dir) pairs for batch mode, e.g. "
            "'/nrs/elferich/gouaux_tomo/15F1/TOP_TOMOS/*/*/STT_results/aunps/aunp_clusters.star'"
        ),
    )
    parser.add_argument("--n-bins", type=int, default=20, help="Number of histogram bins (default: 20)")
    parser.add_argument(
        "--min-cluster-aunps",
        type=int,
        default=6,
        help="Only include clusters with at least this many AuNPs (default: 6)",
    )
    parser.add_argument(
        "--per-tomogram-plots",
        action="store_true",
        help="In batch (--glob) mode, also write a histogram PNG per tomogram (always written in single-tomogram mode)",
    )
    parser.add_argument(
        "--n-diagnostic-clusters",
        type=int,
        default=20,
        help="Number of representative clusters to show convex hulls for (default: 20; 0 disables)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_REPO_ROOT / "results" / "aunp_cluster_density_test",
        help="Where to write cluster CSV(s) and histogram PNG(s)",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.glob:
        jobs = discover_tomogram_jobs(args.glob)
        if not jobs:
            print(f"No aunp_clusters.star files found matching glob {args.glob!r}")
            return 1
        print(f"Discovered {len(jobs)} (tomogram, alignment_dir) pairs")
    elif args.tomo_path and args.alignment_dir:
        jobs = [(args.tomo_path, args.alignment_dir)]
    else:
        parser.error("Provide --glob, or both --tomo-path and --alignment-dir")
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_individual_plots = args.per_tomogram_plots or len(jobs) == 1

    all_frames: List[pd.DataFrame] = []
    diagnostic_pool: List[dict] = []
    for tomo_path, alignment_dir in jobs:
        result = compute_clusters_for_tomogram(
            tomo_path, alignment_dir, min_cluster_aunps=args.min_cluster_aunps
        )
        if result is None:
            continue
        cluster_df, points_by_label = result
        if cluster_df.empty:
            continue

        tomo_name = tomo_path.name
        csv_path = args.output_dir / f"{tomo_name}__{alignment_dir}_aunp_clusters.csv"
        cluster_df.to_csv(csv_path, index=False)
        print(f"  Wrote {csv_path}")

        all_frames.append(cluster_df)

        for row in cluster_df.itertuples():
            diagnostic_pool.append({
                "tomo_name": tomo_name,
                "alignment_dir": alignment_dir,
                "cluster_label": row.cluster_label,
                "n_aunps": row.n_aunps,
                "area_per_aunp_nm2": row.cluster_area_per_aunp_nm2,
                "points": points_by_label[row.cluster_label],
            })

        if write_individual_plots:
            areas_per_aunp = cluster_df["cluster_area_per_aunp_nm2"].dropna().to_numpy()
            if len(areas_per_aunp) == 0:
                print("  No clusters with a valid convex-hull density; skipping histogram.")
            else:
                write_histogram(
                    areas_per_aunp,
                    args.output_dir / f"{tomo_name}__{alignment_dir}_aunp_cluster_density_histogram.png",
                    title=f"{tomo_name}\n{alignment_dir}, {len(areas_per_aunp)} clusters",
                    n_bins=args.n_bins,
                )

    if len(jobs) > 1:
        if not all_frames:
            print("No clusters found across any tomogram.")
            return 0
        combined = pd.concat(all_frames, ignore_index=True)
        combined_csv = args.output_dir / "combined_aunp_clusters.csv"
        combined.to_csv(combined_csv, index=False)
        n_tomos = combined[["tomogram_name", "alignment_dir"]].drop_duplicates().shape[0]
        print(f"Wrote {combined_csv} ({len(combined)} clusters from {n_tomos} tomograms)")

        areas_per_aunp = combined["cluster_area_per_aunp_nm2"].dropna().to_numpy()
        if len(areas_per_aunp) == 0:
            print("No clusters with a valid convex-hull density across the whole batch; skipping combined histogram.")
        else:
            write_histogram(
                areas_per_aunp,
                args.output_dir / "combined_aunp_cluster_density_histogram.png",
                title=f"All tomograms (n={n_tomos})\n{len(areas_per_aunp)} clusters",
                n_bins=args.n_bins,
            )

    if args.n_diagnostic_clusters > 0:
        representative = select_representative_clusters(diagnostic_pool, args.n_diagnostic_clusters)
        if representative:
            write_cluster_hull_diagnostic(
                representative,
                args.output_dir / "aunp_cluster_hull_diagnostic.png",
            )
        else:
            print("No clusters with a valid convex hull; skipping diagnostic plot.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
