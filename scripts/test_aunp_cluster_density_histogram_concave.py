#!/usr/bin/env python3
"""
Fork of test_aunp_cluster_density_histogram.py using a dilated concave hull
instead of a convex hull for the per-cluster AuNP packing density.

Methodology change from the convex-hull version: instead of
``scipy.spatial.ConvexHull`` on the 3D points, each cluster's AuNPs are
projected onto their best-fit 2D plane (PCA), then:

  1. A locally adaptive alpha-shape concave hull is traced around the
     projected AuNP positions themselves -- a taut boundary that follows the
     natural shape of the point cloud and bends inward at gaps, unlike a
     convex hull's straight bridges. Unlike a textbook alpha shape (one
     global alpha for the whole cluster), each Delaunay triangle is compared
     against its *own* local alpha (``--alpha-multiplier`` x the mean
     nearest-neighbor distance of its 3 vertices): a single global alpha
     would have to loosen everywhere just to bridge out to one sparse/
     outlier AuNP, which flattens real concavities elsewhere in the same
     cluster. With a local alpha, dense regions stay tight/concave while the
     boundary only loosens near the actually-sparse points, and every AuNP
     ends up enclosed without needing a global compromise.
  2. That boundary is dilated (offset) outward by ``--buffer-radius`` nm
     (default 5nm) to give every AuNP a margin/halo, since a hull through
     bare point centers ignores the AuNPs' physical size.

Area is computed by rasterizing the dilated region on a grid
(``--grid-resolution`` nm/pixel): points inside the raw alpha-shape polygon,
or within ``--buffer-radius`` of its boundary, count as inside. Clusters too
small for a Delaunay-based alpha shape (< 3 points) fall back to a plain
``--buffer-radius``-nm halo around each point.

Reads per-AuNP cluster assignments from each tomogram's
``<alignment_dir>/STT_results/aunps/aunp_clusters.star`` (written by
``analyze_aunps``, column ``aunp_cluster``) instead of rerunning DBSCAN.

Single-tomogram mode:

  python -u scripts/test_aunp_cluster_density_histogram_concave.py \\
    --tomo-path /nrs/elferich/gouaux_tomo/15F1/TOP_TOMOS/20241030_AMmilled12-1_15 \\
    --alignment-dir patch_tracking \\
    --output-dir results/aunp_cluster_density_concave_test

Batch mode over many tomograms/alignment dirs, discovered from a glob of
``aunp_clusters.star`` files (each match's grandparent directories give the
tomogram path and alignment dir):

  python -u scripts/test_aunp_cluster_density_histogram_concave.py \\
    --glob "/nrs/elferich/gouaux_tomo/15F1/TOP_TOMOS/*/*/STT_results/aunps/aunp_clusters.star" \\
    --output-dir results/aunp_cluster_density_concave_test
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
from matplotlib.path import Path as MplPath
from scipy.spatial import ConvexHull, Delaunay, cKDTree, distance_matrix

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

COORD_COLS = ["faCoordinateX", "faCoordinateY", "faCoordinateZ"]
CLUSTER_STAR_RELPATH = Path("STT_results") / "aunps" / "aunp_clusters.star"


def _project_to_plane(points: np.ndarray) -> np.ndarray:
    """Project 3D points onto their best-fit 2D plane (top 2 PCA components)."""
    centered = points - points.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[:2].T


def _point_segment_distance(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Vectorized distance from each of ``points`` to the segment a->b."""
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom == 0.0:
        return np.linalg.norm(points - a, axis=1)
    t = np.clip(((points - a) @ ab) / denom, 0.0, 1.0)
    proj = a + np.outer(t, ab)
    return np.linalg.norm(points - proj, axis=1)


def _distance_to_polygon_boundary(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Vectorized distance from each of ``points`` to the closed polyline ``polygon``."""
    dmin = np.full(len(points), np.inf)
    for i in range(len(polygon) - 1):
        d = _point_segment_distance(points, polygon[i], polygon[i + 1])
        dmin = np.minimum(dmin, d)
    return dmin


def _local_nn_distances(points_2d: np.ndarray) -> np.ndarray:
    """Per-point nearest-neighbor distance (not a single global median)."""
    tree = cKDTree(points_2d)
    nn_dist, _ = tree.query(points_2d, k=2)
    return nn_dist[:, 1]


def _locally_adaptive_alpha_boundary_edges(
    points: np.ndarray,
    nn_dist: np.ndarray,
    alpha_multiplier: float,
) -> set:
    """
    Boundary edges of a *locally adaptive* alpha shape: like the standard
    Delaunay/circumradius alpha shape, but each triangle is compared against
    its own local alpha (``alpha_multiplier`` x the mean nearest-neighbor
    distance of its 3 vertices) instead of one global alpha for the whole
    point set. This keeps the boundary tight (and concave) through dense
    regions while still being permissive enough to reach sparser/outlier
    points nearby, instead of a single global alpha that has to loosen
    everywhere just to reach the sparsest point.
    """
    tri = Delaunay(points)
    edge_count: Dict[Tuple[int, int], int] = {}

    def add_edge(i: int, j: int) -> None:
        key = (i, j) if i < j else (j, i)
        edge_count[key] = edge_count.get(key, 0) + 1

    for ia, ib, ic in tri.simplices:
        pa, pb, pc = points[ia], points[ib], points[ic]
        a = np.linalg.norm(pb - pc)
        b = np.linalg.norm(pa - pc)
        c = np.linalg.norm(pa - pb)
        s = 0.5 * (a + b + c)
        area = np.sqrt(max(s * (s - a) * (s - b) * (s - c), 0.0))
        if area < 1e-9:
            continue
        circum_r = (a * b * c) / (4.0 * area)
        local_alpha = alpha_multiplier * float(np.mean([nn_dist[ia], nn_dist[ib], nn_dist[ic]]))
        if circum_r <= local_alpha:
            add_edge(ia, ib)
            add_edge(ib, ic)
            add_edge(ic, ia)

    return {edge for edge, count in edge_count.items() if count == 1}


def _order_boundary_edge_loops(edges: set) -> List[List[int]]:
    """Group alpha-shape boundary edges into closed vertex-index loops."""
    import networkx as nx

    graph = nx.Graph()
    graph.add_edges_from(edges)

    loops = []
    for component in nx.connected_components(graph):
        if len(component) < 3:
            continue
        subgraph = graph.subgraph(component)
        try:
            cycle = nx.find_cycle(subgraph)
        except nx.NetworkXNoCycle:
            continue
        loops.append([u for u, _ in cycle])
    return loops


def compute_locally_adaptive_concave_hull_polygons(
    points_2d: np.ndarray,
    alpha_multiplier: float,
) -> List[np.ndarray]:
    """Locally adaptive alpha-shape boundary polygon(s), see ``_locally_adaptive_alpha_boundary_edges``."""
    if len(points_2d) < 3:
        return []
    nn_dist = _local_nn_distances(points_2d)
    edges = _locally_adaptive_alpha_boundary_edges(points_2d, nn_dist, alpha_multiplier)
    loops = _order_boundary_edge_loops(edges)
    polygons = []
    for loop in loops:
        poly = points_2d[loop]
        poly = np.vstack([poly, poly[:1]])
        polygons.append(poly)
    return polygons


def compute_full_coverage_concave_hull(
    points_2d: np.ndarray,
    alpha: Optional[float],
    alpha_multiplier: float,
    growth_factor: float = 1.5,
    max_iterations: int = 25,
) -> List[np.ndarray]:
    """
    Locally adaptive alpha-shape polygon(s) around ``points_2d`` that enclose
    every point (see ``compute_locally_adaptive_concave_hull_polygons``).

    Because alpha is local to each triangle's own neighborhood, this usually
    already covers every point on the first try -- dense regions stay tight/
    concave while the boundary only loosens near actually-sparse points.
    ``alpha_multiplier`` is grown (for this cluster only) as a rare safety
    net if some points are still left outside, falling back to the convex
    hull (which trivially encloses every point) if it doesn't converge
    within ``max_iterations``. The ``alpha`` parameter is accepted for CLI
    symmetry with the global-alpha helpers but is not used here.

    Returns an empty list if there are fewer than 3 points (no hull is
    geometrically possible).
    """
    if len(points_2d) < 3:
        return []
    current_multiplier = alpha_multiplier
    for _ in range(max_iterations):
        polygons = compute_locally_adaptive_concave_hull_polygons(points_2d, current_multiplier)
        if polygons:
            inside = np.zeros(len(points_2d), dtype=bool)
            for poly in polygons:
                # Polygon vertices are themselves data points, i.e. sit exactly
                # on the boundary -- contains_points' point-in-polygon test is
                # not reliable exactly on the boundary, so nudge with a tiny
                # radius rather than undercounting coverage on every call.
                inside |= MplPath(poly).contains_points(points_2d, radius=1e-6)
            if inside.all():
                return polygons
        current_multiplier *= growth_factor

    hull = ConvexHull(points_2d)
    poly = points_2d[hull.vertices]
    poly = np.vstack([poly, poly[:1]])
    return [poly]


def rasterize_dilated_concave_hull(
    points_2d: np.ndarray,
    buffer_radius: float,
    grid_resolution: float,
    alpha: Optional[float],
    alpha_multiplier: float,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[np.ndarray]]]:
    """
    Rasterize the alpha-shape concave hull of ``points_2d`` (grown to enclose
    every point, see ``compute_full_coverage_concave_hull``), dilated outward
    by ``buffer_radius`` nm so every point additionally gets a margin.

    Returns (xs, ys, grid_x, grid_y, mask, polygons) where ``mask`` is a
    boolean grid (True = inside the dilated hull) and ``polygons`` are the
    raw (non-dilated) full-coverage hull boundary loops (empty only when
    there are fewer than 3 points, in which case ``mask`` falls back to a
    plain buffer_radius halo around each point). Returns None if
    ``points_2d`` is empty.
    """
    if len(points_2d) == 0:
        return None
    pad = buffer_radius + grid_resolution
    min_xy = points_2d.min(axis=0) - pad
    max_xy = points_2d.max(axis=0) + pad
    xs = np.arange(min_xy[0], max_xy[0] + grid_resolution, grid_resolution)
    ys = np.arange(min_xy[1], max_xy[1] + grid_resolution, grid_resolution)
    grid_x, grid_y = np.meshgrid(xs, ys)
    grid_pts = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    polygons = compute_full_coverage_concave_hull(points_2d, alpha, alpha_multiplier)

    if polygons:
        inside = np.zeros(len(grid_pts), dtype=bool)
        dist_to_boundary = np.full(len(grid_pts), np.inf)
        for poly in polygons:
            inside |= MplPath(poly).contains_points(grid_pts)
            dist_to_boundary = np.minimum(dist_to_boundary, _distance_to_polygon_boundary(grid_pts, poly))
        dilated = inside | (dist_to_boundary <= buffer_radius)
    else:
        # Fewer than 3 points: no polygon is geometrically possible, so the
        # hull is just the union of each point's own margin.
        tree = cKDTree(points_2d)
        dist_to_points, _ = tree.query(grid_pts, k=1)
        dilated = dist_to_points <= buffer_radius

    mask = dilated.reshape(grid_x.shape)
    return xs, ys, grid_x, grid_y, mask, polygons


def compute_dilated_concave_hull_area_nm2(
    points_2d: np.ndarray,
    buffer_radius: float,
    grid_resolution: float,
    alpha: Optional[float],
    alpha_multiplier: float,
) -> float:
    """Area (nm^2) of the alpha-shape concave hull of ``points_2d``, dilated by ``buffer_radius`` nm."""
    result = rasterize_dilated_concave_hull(points_2d, buffer_radius, grid_resolution, alpha, alpha_multiplier)
    if result is None:
        return np.nan
    _, _, _, _, mask, _ = result
    return float(mask.sum()) * (grid_resolution ** 2)


def compute_cluster_density_summary(
    coords: np.ndarray,
    cluster_labels: np.ndarray,
    *,
    buffer_radius: float = 5.0,
    grid_resolution: float = 1.0,
    alpha: Optional[float] = None,
    alpha_multiplier: float = 2.5,
) -> pd.DataFrame:
    """
    Per-cluster AuNP count, dilated concave-hull area (nm^2), and packing
    density (AuNPs per nm^2) for each non-noise cluster label.

    ``coords`` are AuNP 3D coordinates (nm); each cluster is projected onto
    its best-fit 2D plane, then the hull area is the alpha-shape concave
    hull of the projected AuNPs, dilated outward by ``buffer_radius`` nm.
    """
    cluster_rows = []
    for label in np.unique(cluster_labels):
        if label == -1:
            continue  # Skip noise
        cluster_points = coords[cluster_labels == label]
        n_points = len(cluster_points)
        proj = _project_to_plane(cluster_points)
        try:
            area = compute_dilated_concave_hull_area_nm2(
                proj, buffer_radius, grid_resolution, alpha, alpha_multiplier
            )
        except Exception:
            area = np.nan
        try:
            dists = distance_matrix(cluster_points, cluster_points)
            max_dim = np.nanmax(dists) if dists.size else np.nan
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
    ax.set_xlabel("AuNP density (nm² / AuNP, dilated concave hull)")
    ax.set_ylabel("Cluster count")
    ax.set_title(title)
    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {png_path}")


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


def write_cluster_hull_diagnostic(
    entries: List[dict],
    png_path: Path,
    *,
    buffer_radius: float,
    grid_resolution: float,
    alpha: Optional[float],
    alpha_multiplier: float,
    n_cols: int = 5,
) -> None:
    """
    Grid of projected AuNP clusters with their raw alpha-shape outline
    (dashed) and the dilated concave hull used for the density calculation
    (filled), for visual sanity-checking.
    """
    if not entries:
        return
    n = len(entries)
    n_cols = min(n_cols, n)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.8 * n_cols, 2.8 * n_rows), squeeze=False)
    axes_flat = axes.ravel()

    for ax, entry in zip(axes_flat, entries):
        proj = _project_to_plane(entry["points"])
        result = rasterize_dilated_concave_hull(proj, buffer_radius, grid_resolution, alpha, alpha_multiplier)
        if result is not None:
            _, _, grid_x, grid_y, mask, polygons = result
            mask_f = mask.astype(float)
            ax.contourf(grid_x, grid_y, mask_f, levels=[0.5, 1.5], colors=["#DD8452"], alpha=0.25, zorder=1)
            ax.contour(grid_x, grid_y, mask_f, levels=[0.5], colors="#DD8452", linewidths=1.5, zorder=2)
            for poly in polygons:
                ax.plot(poly[:, 0], poly[:, 1], color="#55A868", linewidth=1.0, linestyle="--", zorder=2.5)
        ax.scatter(proj[:, 0], proj[:, 1], s=14, color="#4C72B0", zorder=3)
        ax.set_title(
            f"{entry['tomo_name']}\n{entry['alignment_dir']} #{entry['cluster_label']}\n"
            f"n={entry['n_aunps']}, {entry['area_per_aunp_nm2']:.0f} nm²/AuNP",
            fontsize=7,
        )
        ax.set_aspect("equal", adjustable="datalim")
        ax.tick_params(labelsize=6)

    for ax in axes_flat[len(entries):]:
        ax.axis("off")

    fig.suptitle(
        f"Representative AuNP clusters (dashed = alpha-shape hull, filled = +{buffer_radius:.0f}nm margin)",
        fontsize=11,
    )
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
    buffer_radius: float = 5.0,
    grid_resolution: float = 1.0,
    alpha: Optional[float] = None,
    alpha_multiplier: float = 2.5,
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

    cluster_df = compute_cluster_density_summary(
        coords,
        cluster_labels,
        buffer_radius=buffer_radius,
        grid_resolution=grid_resolution,
        alpha=alpha,
        alpha_multiplier=alpha_multiplier,
    )
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
        description=(
            "AuNP cluster dilated-concave-hull density histogram(s), using existing "
            "analyze_aunps cluster assignments"
        )
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
    parser.add_argument(
        "--buffer-radius",
        type=float,
        default=5.0,
        help="Margin (nm) the alpha-shape hull is dilated outward by, around each AuNP (default: 5.0)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Alpha-shape circumradius cutoff (nm). Default: auto (2.5x median nearest-neighbor distance)",
    )
    parser.add_argument(
        "--alpha-multiplier",
        type=float,
        default=2.5,
        help="Multiplier used to derive alpha automatically when --alpha is not given (default: 2.5)",
    )
    parser.add_argument(
        "--grid-resolution",
        type=float,
        default=1.0,
        help="Rasterization grid spacing (nm/pixel) used to compute the dilated-hull area (default: 1.0)",
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
        help="Number of representative clusters to show dilated concave hulls for (default: 20; 0 disables)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_REPO_ROOT / "results" / "aunp_cluster_density_concave_test",
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
            tomo_path,
            alignment_dir,
            min_cluster_aunps=args.min_cluster_aunps,
            buffer_radius=args.buffer_radius,
            grid_resolution=args.grid_resolution,
            alpha=args.alpha,
            alpha_multiplier=args.alpha_multiplier,
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
                print("  No clusters with a valid concave-hull density; skipping histogram.")
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
            print("No clusters with a valid concave-hull density across the whole batch; skipping combined histogram.")
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
                buffer_radius=args.buffer_radius,
                grid_resolution=args.grid_resolution,
                alpha=args.alpha,
                alpha_multiplier=args.alpha_multiplier,
            )
        else:
            print("No clusters with a valid concave hull; skipping diagnostic plot.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
