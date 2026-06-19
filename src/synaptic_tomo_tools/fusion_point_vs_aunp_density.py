"""
Fusion-point vs AuNP density analysis with presynaptic tangential-shuffle controls.

For each close/fusing vesicle fusion point, samples random tangential offsets at
several distances d on the presynaptic active zone, snaps to the AZ point cloud,
and looks up packing density from the same scan-vertex tables used in production.

Also runs standardized bivariate Ripley H12 on postsynaptic-projected fusion,
control, and AuNP positions (fusion vs controls per d; label-permutation null),
and geodesic Ripley's O via membrain-stats (same two analysis modes).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .aunps import (
    build_packing_density_at_fusion_points_dataframe,
    calculate_packing_density_using_sliding_cylinder,
    enumerate_close_vesicle_fusion_points,
)
from .vesicles import import_presynaptic_membranes_and_active_zones

# Probe radii (nm) for fusion-point control lookups and Ripley analysis (not used for main packing heat map).
PACKING_DENSITY_PROBE_RADII_NM = (10.0, 20.0, 30.0, 40.0, 50.0)


def packing_density_radius_tag(radius_nm: float) -> str:
    """Filename suffix for a probe radius, e.g. ``10.0`` -> ``r10nm``."""
    return f"r{int(round(radius_nm))}nm"


DEFAULT_OFFSET_DISTANCES_NM = (10.0, 20.0, 30.0, 40.0, 50.0)
DEFAULT_PROBE_RADII_NM = PACKING_DENSITY_PROBE_RADII_NM
DEFAULT_PROBE_RADIUS_NM = 25.0
DEFAULT_N_DIRECTIONS = 100
DEFAULT_ANALYSIS_SEED = 42

FUSION_POINT_VS_AUNP_DENSITY_SUBDIR = "fusion_point_vs_aunp_density"
COMBINED_RESULTS_DIR = Path("results/aunps/fusion_point_vs_aunp_density")
RIPLEY_H12_VESICLE_CURVES_NPZ = "ripley_h12_vesicle_curves.npz"
RIPLEY_O_VESICLE_CURVES_NPZ = "ripley_o_vesicle_curves.npz"


def _random_tangent_direction(
    center: np.ndarray,
    neighbors: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Unit tangent direction in the local AZ plane (PCA: two largest-variance axes)."""
    if len(neighbors) < 3:
        direction = rng.normal(size=3)
        direction[2] *= 0.2
        return direction / np.linalg.norm(direction)
    centered = neighbors - center
    cov = centered.T @ centered / len(centered)
    _, evecs = np.linalg.eigh(cov)
    t1 = evecs[:, 2]
    t2 = evecs[:, 1]
    t1 /= np.linalg.norm(t1)
    t2 /= np.linalg.norm(t2)
    theta = rng.uniform(0.0, 2.0 * np.pi)
    direction = np.cos(theta) * t1 + np.sin(theta) * t2
    return direction / np.linalg.norm(direction)


def sample_tangential_control_on_az(
    fusion_xyz: np.ndarray,
    az_xyz: np.ndarray,
    az_tree: cKDTree,
    offset_nm: float,
    rng: np.random.Generator,
    *,
    neighbor_radius_nm: float = 30.0,
    min_separation_from_fusion_nm: float = 5.0,
    max_snap_distance_nm: float = 10.0,
    max_attempts: int = 40,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Move fusion_xyz by offset_nm in a random tangent direction on the presynaptic AZ,
    then snap to the nearest presynaptic active-zone point.

    Retries up to max_attempts when the snapped point is too close to the fusion point
    or the candidate lies farther than max_snap_distance_nm from the AZ surface.
    Returns None if no valid control is found within max_attempts.
    """
    neighbor_idx = az_tree.query_ball_point(fusion_xyz, r=neighbor_radius_nm)
    if len(neighbor_idx) < 5:
        neighbor_idx = az_tree.query_ball_point(fusion_xyz, r=neighbor_radius_nm * 2.0)
    neighbors = az_xyz[np.asarray(neighbor_idx, dtype=int)]

    for _ in range(max_attempts):
        direction = _random_tangent_direction(fusion_xyz, neighbors, rng)
        candidate = fusion_xyz + offset_nm * direction
        snap_dist_nm, snap_idx = az_tree.query(candidate, k=1)
        if snap_dist_nm > max_snap_distance_nm:
            continue
        snapped = az_xyz[int(snap_idx)]
        if np.linalg.norm(snapped - fusion_xyz) >= min_separation_from_fusion_nm:
            return snapped, direction
    return None, None


def lookup_packing_at_point(
    xyz: np.ndarray,
    scan_df: pd.DataFrame,
) -> dict:
    """Nearest scan-vertex packing lookup (same as production fusion-point table)."""
    scan_xyz = scan_df[["vertex_x_nm", "vertex_y_nm", "vertex_z_nm"]].to_numpy(dtype=float)
    tree = cKDTree(scan_xyz)
    dist_nm, idx = tree.query(xyz, k=1)
    row = scan_df.iloc[int(idx)]
    return {
        "query_point_x_nm": float(xyz[0]),
        "query_point_y_nm": float(xyz[1]),
        "query_point_z_nm": float(xyz[2]),
        "packing_coefficient": float(row["packing_coefficient"]),
        "aunp_count_in_cylinder": int(row["aunp_count_in_cylinder"]),
        "aunp_density_per_nm2": float(row["aunp_density_per_nm2"]),
        "nearest_scan_active_zone_name": row["active_zone_name"],
        "nearest_scan_vertex_index": int(row["scan_vertex_index"]),
        "nearest_scan_vertex_x_nm": float(row["vertex_x_nm"]),
        "nearest_scan_vertex_y_nm": float(row["vertex_y_nm"]),
        "nearest_scan_vertex_z_nm": float(row["vertex_z_nm"]),
        "nearest_scan_vertex_distance_nm": float(dist_nm),
    }


def load_or_compute_scan_df(
    tomogram_path: Path,
    alignment_dir: str,
    probe_radius_nm: float,
    *,
    vertex_sampling_step: int = 50,
    receptor_crosssection: float = 122.0,
    aunps_per_receptor: float = 2.0,
    scan_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Load scan CSV if present; otherwise compute from GLB zones or use provided scan_df."""
    if scan_df is not None and not scan_df.empty:
        rad_col = "probe_radius_nm" if "probe_radius_nm" in scan_df.columns else "cylinder_radius_nm"
        if rad_col in scan_df.columns:
            subset = scan_df[np.isclose(scan_df[rad_col], probe_radius_nm)]
            if not subset.empty:
                return subset
        return scan_df.copy()

    aunps_dir = tomogram_path / alignment_dir / "STT_results" / "aunps"
    tag = packing_density_radius_tag(probe_radius_nm)
    per_radius_csv = aunps_dir / f"packing_density_scan_vertices_{tag}.csv"
    legacy_csv = aunps_dir / "packing_density_scan_vertices.csv"

    if per_radius_csv.is_file():
        return pd.read_csv(per_radius_csv)

    if legacy_csv.is_file():
        legacy = pd.read_csv(legacy_csv)
        rad_col = "probe_radius_nm" if "probe_radius_nm" in legacy.columns else "cylinder_radius_nm"
        if rad_col in legacy.columns:
            subset = legacy[np.isclose(legacy[rad_col], probe_radius_nm)]
            if not subset.empty:
                return subset

    print(f"  Computing scan vertices for probe r={probe_radius_nm:.0f} nm...")
    from .activezone import (
        define_active_zonogram,
        find_active_zones_from_glb,
        import_membrane_segmentations_from_glb,
    )
    import starfile

    membrane_data = import_membrane_segmentations_from_glb(tomogram_path, alignment_dir=alignment_dir)
    active_zones_glb = find_active_zones_from_glb(membrane_data, distance_range=(10.0, 40.0))
    zonogram_results = define_active_zonogram(active_zones_glb)

    aunps_dir_pick = tomogram_path / alignment_dir / "aunps"
    star_files = sorted(aunps_dir_pick.glob("aunp_tm_BP_active_zone_*_manual_refined.star"))
    if not star_files:
        raise FileNotFoundError(f"No AuNP STAR files in {aunps_dir_pick}")
    aunp_coords = starfile.read(star_files[0])[["faCoordinateX", "faCoordinateY", "faCoordinateZ"]].to_numpy()

    tomogram_name = tomogram_path.name
    rows: list[dict] = []
    for zone_name, zone_data in active_zones_glb["active_zones"].items():
        if "active_postsynaptic_mesh" not in zone_data:
            continue
        if zone_name not in zonogram_results["zonogram_data"]:
            continue
        v_array, n_aunps, aunp_dens, coeff = calculate_packing_density_using_sliding_cylinder(
            zone_data,
            zonogram_results["zonogram_data"][zone_name],
            aunp_coords,
            cylinder_radius=probe_radius_nm,
            receptor_crosssection_nm_squared=receptor_crosssection,
            aunps_per_receptor=aunps_per_receptor,
            vertex_sampling_step=vertex_sampling_step,
        )
        for scan_idx, (vertex, n, dens, c) in enumerate(zip(v_array, n_aunps, aunp_dens, coeff)):
            rows.append(
                {
                    "tomogram_name": tomogram_name,
                    "alignment_dir": alignment_dir,
                    "active_zone_name": zone_name,
                    "scan_vertex_index": scan_idx,
                    "vertex_x_nm": float(vertex[0]),
                    "vertex_y_nm": float(vertex[1]),
                    "vertex_z_nm": float(vertex[2]),
                    "aunp_count_in_cylinder": int(n),
                    "aunp_density_per_nm2": float(dens),
                    "packing_coefficient": float(c),
                    "probe_radius_nm": float(probe_radius_nm),
                    "cylinder_radius_nm": float(probe_radius_nm),
                }
            )
    return pd.DataFrame(rows)


def build_control_table(
    fusion_rows: list[dict],
    membrane_az_pairs: dict,
    scan_by_radius: dict[float, pd.DataFrame],
    offset_distances_nm: tuple[float, ...],
    n_directions: int,
    seed: int,
    *,
    max_snap_distance_nm: float = 10.0,
) -> pd.DataFrame:
    """Real fusion rows (offset=0) + tangential AZ controls for each d and direction."""
    rng = np.random.default_rng(seed)
    out_rows: list[dict] = []

    for fp in fusion_rows:
        membrane = fp.get("closest_membrane")
        if not membrane or membrane not in membrane_az_pairs:
            continue
        az_xyz = membrane_az_pairs[membrane]["active_zone_points"]
        if az_xyz is None or len(az_xyz) == 0:
            continue
        az_tree = cKDTree(az_xyz)
        fusion_xyz = np.array(
            [fp["fusion_point_x_nm"], fp["fusion_point_y_nm"], fp["fusion_point_z_nm"]],
            dtype=float,
        )

        for probe_radius, scan_df in scan_by_radius.items():
            if scan_df.empty:
                continue

            real_lookup = lookup_packing_at_point(fusion_xyz, scan_df)
            out_rows.append(
                {
                    **fp,
                    **real_lookup,
                    "point_type": "fusion",
                    "control_offset_nm": 0.0,
                    "control_direction_index": -1,
                    "control_offset_distance_actual_nm": 0.0,
                    "probe_radius_nm": float(probe_radius),
                    "random_seed": seed,
                }
            )

            for offset_nm in offset_distances_nm:
                if offset_nm <= 0:
                    continue
                for dir_idx in range(n_directions):
                    control_xyz, direction = sample_tangential_control_on_az(
                        fusion_xyz,
                        az_xyz,
                        az_tree,
                        offset_nm,
                        rng,
                        max_snap_distance_nm=max_snap_distance_nm,
                    )
                    if control_xyz is None:
                        continue
                    ctrl_lookup = lookup_packing_at_point(control_xyz, scan_df)
                    out_rows.append(
                        {
                            **fp,
                            **ctrl_lookup,
                            "point_type": "control",
                            "control_offset_nm": float(offset_nm),
                            "control_direction_index": int(dir_idx),
                            "control_offset_distance_actual_nm": float(
                                np.linalg.norm(control_xyz - fusion_xyz)
                            ),
                            "control_direction_x": float(direction[0]) if direction is not None else np.nan,
                            "control_direction_y": float(direction[1]) if direction is not None else np.nan,
                            "control_direction_z": float(direction[2]) if direction is not None else np.nan,
                            "fusion_point_x_nm": float(fp["fusion_point_x_nm"]),
                            "fusion_point_y_nm": float(fp["fusion_point_y_nm"]),
                            "fusion_point_z_nm": float(fp["fusion_point_z_nm"]),
                            "probe_radius_nm": float(probe_radius),
                            "random_seed": seed,
                        }
                    )
    return pd.DataFrame(out_rows)


def _find_precalculated_zonogram_mrc(
    tomogram_path: Path,
    alignment_dir: str,
    zone_name: str,
) -> Path | None:
    """Locate saved active zonogram MRC from a prior visualization run."""
    az_dir = (
        tomogram_path
        / alignment_dir
        / "STT_results"
        / "visualizations"
        / "active_zonograms"
    )
    if not az_dir.is_dir():
        return None
    matches = sorted(az_dir.glob(f"{tomogram_path.name}_active_zonogram_{zone_name}*.mrc"))
    return matches[0] if matches else None


def _load_zone_transform_from_active_zone_results(
    tomogram_path: Path,
    alignment_dir: str,
    zone_name: str,
) -> dict:
    """
    Reconstruct zonogram center / transformation / extent from saved active-zone analysis.

    Matches the coordinate system used when STT visualizations render active zonograms
    (same define_active_zonogram path as visualization.run_combined_zonogram_analysis).
    """
    from .activezone import (
        define_active_zonogram,
        find_active_zones_from_glb,
        import_membrane_segmentations_from_glb,
    )

    np.random.seed(42)
    membrane_data = import_membrane_segmentations_from_glb(
        str(tomogram_path),
        alignment_dir=alignment_dir,
    )
    active_zones_data = find_active_zones_from_glb(membrane_data, distance_range=(10.0, 40.0))
    zonogram_results = define_active_zonogram(active_zones_data)
    zonogram_data = zonogram_results.get("zonogram_data", {})
    if zone_name not in zonogram_data:
        available = ", ".join(sorted(zonogram_data))
        raise KeyError(f"Zone '{zone_name}' not in zonogram data (available: {available})")
    return zonogram_data[zone_name]


def _load_precalculated_zonogram_for_zone(
    tomogram_path: Path,
    alignment_dir: str,
    zone_name: str,
):
    """Return (zonogram_findingampa, zone_data) using saved MRC + active-zone transform."""
    import mrcfile
    import torch
    from .visualization import transform_positions_to_zonogram_coords

    mrc_path = _find_precalculated_zonogram_mrc(tomogram_path, alignment_dir, zone_name)
    if mrc_path is None:
        raise FileNotFoundError(
            f"No precalculated zonogram MRC for zone '{zone_name}' under "
            f"{tomogram_path / alignment_dir / 'STT_results' / 'visualizations' / 'active_zonograms'}"
        )

    with mrcfile.open(mrc_path, mode="r") as mrc:
        vol = torch.tensor(np.asarray(mrc.data, dtype=np.float32))
    zone_data = _load_zone_transform_from_active_zone_results(
        tomogram_path,
        alignment_dir,
        zone_name,
    )
    zonogram_findingampa = (np.eye(3), np.zeros(3), vol, ())
    return zonogram_findingampa, zone_data, mrc_path, transform_positions_to_zonogram_coords


def _plot_fusion_vs_control_zonogram(
    real: pd.DataFrame,
    ctrl: pd.DataFrame,
    *,
    tomogram_path: Path,
    alignment_dir: str,
    probe_radius_nm: float,
    output_path: Path,
) -> None:
    """All tangential controls on one active zonogram, colored by offset d."""
    from matplotlib import cm
    from .visualization import render_mini_zonogram_xy_only

    zone_col = "nearest_scan_active_zone_name"
    if zone_col not in real.columns:
        print("Skipping zonogram XY plot: nearest_scan_active_zone_name missing.")
        return

    offsets = sorted(d for d in ctrl["control_offset_nm"].unique() if d > 0)
    if not offsets:
        return

    zone_names = sorted(real[zone_col].dropna().unique())
    cmap = cm.get_cmap("turbo", len(offsets))

    for zone_name in zone_names:
        try:
            zonogram_findingampa, zone_data, mrc_path, transform_fn = _load_precalculated_zonogram_for_zone(
                tomogram_path,
                alignment_dir,
                str(zone_name),
            )
        except (FileNotFoundError, KeyError) as exc:
            print(f"Skipping zonogram XY plot for {zone_name}: {exc}")
            continue

        sub_f = real[
            (real["probe_radius_nm"] == probe_radius_nm)
            & (real[zone_col] == zone_name)
        ].drop_duplicates(subset=["vesicle_id"])
        if sub_f.empty:
            continue

        fusion_xyz = sub_f[["fusion_point_x_nm", "fusion_point_y_nm", "fusion_point_z_nm"]].to_numpy(dtype=float)
        fusion_zono = transform_fn(fusion_xyz, zonogram_findingampa, zone_data)

        fig, ax = render_mini_zonogram_xy_only(zonogram_findingampa, include_legend_space=True)

        for i, offset_nm in enumerate(offsets):
            sub_c = ctrl[
                (ctrl["probe_radius_nm"] == probe_radius_nm)
                & (ctrl["control_offset_nm"] == offset_nm)
                & (ctrl[zone_col] == zone_name)
            ]
            if sub_c.empty:
                continue
            control_xyz = sub_c[["query_point_x_nm", "query_point_y_nm", "query_point_z_nm"]].to_numpy(dtype=float)
            control_zono = transform_fn(control_xyz, zonogram_findingampa, zone_data)
            color = cmap(i)
            ax.scatter(
                control_zono[:, 0],
                control_zono[:, 1],
                c=[color],
                s=14,
                alpha=0.5,
                edgecolors="none",
                label=f"d={int(offset_nm)} nm (n={len(control_zono)})",
                zorder=2,
            )

        ax.scatter(
            fusion_zono[:, 0],
            fusion_zono[:, 1],
            c="red",
            s=140,
            marker="*",
            edgecolors="white",
            linewidths=0.6,
            label="fusion points",
            zorder=5,
        )
        for row, pt in zip(sub_f.itertuples(), fusion_zono):
            ax.annotate(
                f"v{int(row.vesicle_id)}",
                (pt[0], pt[1]),
                fontsize=8,
                color="yellow",
                ha="center",
                va="bottom",
                xytext=(0, 6),
                textcoords="offset points",
                zorder=6,
            )

        ax.set_title(
            f"Fusion vs tangential controls on active zonogram\n"
            f"{zone_name} | r={int(probe_radius_nm)} nm | controls colored by d"
        )
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7, framealpha=0.9)
        fig.tight_layout()

        if len(zone_names) == 1:
            save_path = output_path
        else:
            save_path = output_path.with_name(f"{output_path.stem}_{zone_name}{output_path.suffix}")
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Zonogram overlay from {mrc_path.name} -> {save_path}")


DEFAULT_RIPLEY_R_MAX_NM = 150.0
DEFAULT_RIPLEY_R_STEP_NM = 5.0
DEFAULT_RIPLEY_N_PERM = 499
RIPLEY_PERCENTILE_LO = 2.5
RIPLEY_PERCENTILE_HI = 97.5
UNCERTAINTY_METHOD_PERCENTILE = "percentile_2p5_97p5"
DEFAULT_RIPLEY_O_MESH_MAX_VERTS = 4000
DEFAULT_RIPLEY_O_GEODESIC_METHOD = "fast"


def _load_az_surface_txt(path: Path) -> np.ndarray:
    if not path.is_file():
        return np.zeros((0, 3), dtype=float)
    data = np.atleast_2d(np.loadtxt(path, delimiter=None))
    if data.size == 0:
        return np.zeros((0, 3), dtype=float)
    return data.astype(float)


def _load_postsynaptic_active_zone_surface(
    tomogram_path: Path,
    alignment_dir: str,
    zone_name: str,
) -> np.ndarray:
    """Full postsynaptic active-zone patch (outer + inner) for one zone."""
    az_dir = tomogram_path / alignment_dir / "STT_results" / "activezone"
    parts: list[np.ndarray] = []
    for suffix in ("post_outer", "post_inner"):
        path = az_dir / f"{zone_name}_{suffix}.txt"
        surf = _load_az_surface_txt(path)
        if len(surf):
            parts.append(surf)
    if not parts:
        raise FileNotFoundError(f"No postsynaptic AZ surfaces found for {zone_name} in {az_dir}")
    return np.vstack(parts)


def _project_points_to_surface(xyz: np.ndarray, surface_tree: cKDTree, surface_xyz: np.ndarray) -> np.ndarray:
    """Nearest-neighbor projection onto a postsynaptic surface point cloud."""
    xyz = np.atleast_2d(np.asarray(xyz, dtype=float))
    if len(xyz) == 0:
        return np.zeros((0, 3), dtype=float)
    _, idx = surface_tree.query(xyz, k=1)
    return surface_xyz[np.asarray(idx, dtype=int)]


def _estimate_planar_window_area_nm2(coords: np.ndarray) -> float:
    """Convex-hull area (nm²) after PCA flattening of surface points."""
    from scipy.spatial import ConvexHull

    coords = np.atleast_2d(np.asarray(coords, dtype=float))
    if len(coords) < 3:
        return 1.0
    centered = coords - coords.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    plane = centered @ vt[:2].T
    return float(ConvexHull(plane).volume)


def _ripley_r_grid(r_max_nm: float, r_step_nm: float) -> np.ndarray:
    n_steps = max(1, int(np.floor(r_max_nm / r_step_nm)))
    return np.arange(r_step_nm, r_max_nm + 0.5 * r_step_nm, r_step_nm, dtype=float)


def cross_k12(
    x: np.ndarray,
    y: np.ndarray,
    r_vals: np.ndarray,
    window_area_nm2: float,
) -> np.ndarray:
    """
    Empirical bivariate cross-K on a surface point pattern.

    Uses 3D chord distances between projected surface points with 2D normalization
    (πr²), appropriate for active-zone patches embedded in 3D.
    """
    x = np.atleast_2d(np.asarray(x, dtype=float))
    y = np.atleast_2d(np.asarray(y, dtype=float))
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0 or window_area_nm2 <= 0:
        return np.full(len(r_vals), np.nan)

    tree = cKDTree(y)
    counts = np.zeros(len(r_vals), dtype=float)
    r_max = float(r_vals[-1])
    for xi in x:
        neighbor_idx = tree.query_ball_point(xi, r=r_max)
        if not neighbor_idx:
            continue
        dists = np.linalg.norm(y[np.asarray(neighbor_idx)] - xi, axis=1)
        for k, r in enumerate(r_vals):
            counts[k] += np.sum(dists < r)
    return (window_area_nm2 / (n1 * n2)) * counts


def ripley_h12(k12: np.ndarray, r_vals: np.ndarray) -> np.ndarray:
    """Standardized bivariate Ripley H: sqrt(K12 / π) − r."""
    k12 = np.maximum(np.asarray(k12, dtype=float), 0.0)
    return np.sqrt(k12 / np.pi) - r_vals


def ripley_h12_from_points(
    x: np.ndarray,
    y: np.ndarray,
    r_vals: np.ndarray,
    window_area_nm2: float,
) -> np.ndarray:
    return ripley_h12(cross_k12(x, y, r_vals, window_area_nm2), r_vals)


def _import_membrain_ripley():
    try:
        from membrain_stats.utils.ripley_utils import (
            aggregate_ripleys_stats,
            compute_ripleys_stats,
        )
    except ImportError as exc:
        raise ImportError(
            "membrain-stats is required for Ripley's O analysis. "
            "Install with: pip install membrain-stats potpourri3d"
        ) from exc
    return compute_ripleys_stats, aggregate_ripleys_stats


def _membrain_aggregate_ripley_o(
    ripley_stats: list,
    *,
    bin_size_nm: float,
    num_bins: int,
):
    """
    membrain-stats aggregate_ripleys_stats(ripley_type='O') with a fix for
    bivariate (n_start != n_target) distance matrices.
    """
    import membrain_stats.utils.ripley_utils as ru

    def _fixed_get_number_of_points(distance_matrices: list[np.ndarray]):
        num_starting_points = sum(len(dm) for dm in distance_matrices)
        avg_starting_points = num_starting_points / len(distance_matrices)
        num_reachable_points = [
            [np.sum(dm[:, i] < np.inf) for i in range(dm.shape[1])]
            for dm in distance_matrices
        ]
        avg_reachable_points = [np.mean(entry) for entry in num_reachable_points]
        return avg_starting_points, avg_reachable_points

    original = ru.get_number_of_points
    ru.get_number_of_points = _fixed_get_number_of_points
    try:
        return ru.aggregate_ripleys_stats(
            ripley_stats,
            ripley_type="O",
            bin_size=bin_size_nm,
            num_bins=num_bins,
        )
    finally:
        ru.get_number_of_points = original


def _build_membrain_o_analysis_mesh(
    surface_xyz: np.ndarray,
    anchor_xyz: np.ndarray,
    *,
    r_patch_nm: float,
    max_vertices: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Coarse triangular manifold for membrain-stats geodesic Ripley O.

    Delaunay triangulation of a subsampled postsynaptic AZ patch near the
    analysis points (membrain-stats uses potpourri3d heat-method geodesics).
    """
    from scipy.spatial import Delaunay

    surface_xyz = np.atleast_2d(np.asarray(surface_xyz, dtype=float))
    anchor_xyz = np.atleast_2d(np.asarray(anchor_xyz, dtype=float))
    dist, _ = cKDTree(anchor_xyz).query(surface_xyz, k=1)
    patch = surface_xyz[dist <= r_patch_nm]
    if len(patch) < 3:
        raise ValueError("Too few postsynaptic surface points in Ripley O patch.")
    if len(patch) > max_vertices:
        patch = patch[rng.choice(len(patch), max_vertices, replace=False)]
    centered = patch - patch.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    plane = centered @ vt[:2].T
    tri = Delaunay(plane)
    return patch.astype(float), tri.simplices.astype(int)


def _snap_points_to_mesh_vertices(points: np.ndarray, mesh_verts: np.ndarray) -> np.ndarray:
    points = np.atleast_2d(np.asarray(points, dtype=float))
    if len(points) == 0:
        return np.zeros((0, 3), dtype=float)
    _, idx = cKDTree(mesh_verts).query(points, k=1)
    return mesh_verts[np.asarray(idx, dtype=int)]


def _membrain_ripley_o_on_r_grid(
    start_xyz: np.ndarray,
    target_xyz: np.ndarray,
    mesh_verts: np.ndarray,
    mesh_faces: np.ndarray,
    r_vals: np.ndarray,
    *,
    geodesic_method: str = DEFAULT_RIPLEY_O_GEODESIC_METHOD,
    bin_size_nm: float | None = None,
) -> np.ndarray:
    """Bivariate geodesic Ripley's O (membrain-stats) interpolated onto r_vals."""
    compute_ripleys_stats, _aggregate_ripleys_stats = _import_membrain_ripley()
    start_xyz = np.atleast_2d(np.asarray(start_xyz, dtype=float))
    target_xyz = np.atleast_2d(np.asarray(target_xyz, dtype=float))
    if len(start_xyz) == 0 or len(target_xyz) == 0:
        return np.full(len(r_vals), np.nan)

    if bin_size_nm is None:
        bin_size_nm = float(r_vals[1] - r_vals[0]) if len(r_vals) > 1 else DEFAULT_RIPLEY_R_STEP_NM

    mesh_dict = {
        "verts": np.asarray(mesh_verts, dtype=float),
        "faces": np.asarray(mesh_faces, dtype=int),
        "positions_start": _snap_points_to_mesh_vertices(start_xyz, mesh_verts),
        "positions_target": _snap_points_to_mesh_vertices(target_xyz, mesh_verts),
    }
    ripley_stat = compute_ripleys_stats(mesh_dict, method=geodesic_method)
    x_vals, o_vals = _membrain_aggregate_ripley_o(
        [ripley_stat],
        bin_size_nm=bin_size_nm,
        num_bins=max(10, len(r_vals)),
    )
    r_max = float(r_vals[-1])
    valid = x_vals <= r_max + 0.5 * bin_size_nm
    x_vals = np.asarray(x_vals, dtype=float)[valid]
    o_vals = np.asarray(o_vals, dtype=float)[: len(x_vals)]
    if len(x_vals) < 2:
        return np.full(len(r_vals), np.nan)
    return np.interp(r_vals, x_vals, o_vals, left=np.nan, right=np.nan)


def _ripley_o_from_points(
    start_xyz: np.ndarray,
    target_xyz: np.ndarray,
    mesh_verts: np.ndarray,
    mesh_faces: np.ndarray,
    r_vals: np.ndarray,
    *,
    geodesic_method: str = DEFAULT_RIPLEY_O_GEODESIC_METHOD,
) -> np.ndarray:
    return _membrain_ripley_o_on_r_grid(
        start_xyz,
        target_xyz,
        mesh_verts,
        mesh_faces,
        r_vals,
        geodesic_method=geodesic_method,
    )


def _per_vesicle_o_curves(
    points_by_vesicle: dict[int, np.ndarray],
    target_xyz: np.ndarray,
    mesh_verts: np.ndarray,
    mesh_faces: np.ndarray,
    r_vals: np.ndarray,
    *,
    geodesic_method: str = DEFAULT_RIPLEY_O_GEODESIC_METHOD,
) -> np.ndarray:
    curves = [
        _ripley_o_from_points(
            pts, target_xyz, mesh_verts, mesh_faces, r_vals, geodesic_method=geodesic_method
        )
        for _vid in sorted(points_by_vesicle)
        for pts in [points_by_vesicle[_vid]]
        if len(pts) > 0
    ]
    if not curves:
        return np.empty((0, len(r_vals)))
    return np.vstack(curves)


def _paired_vesicle_o_curves(
    fusion_by_vesicle: dict[int, np.ndarray],
    control_by_vesicle: dict[int, np.ndarray],
    target_xyz: np.ndarray,
    mesh_verts: np.ndarray,
    mesh_faces: np.ndarray,
    r_vals: np.ndarray,
    *,
    geodesic_method: str = DEFAULT_RIPLEY_O_GEODESIC_METHOD,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    common_ids = sorted(set(fusion_by_vesicle) & set(control_by_vesicle))
    fusion_curves: list[np.ndarray] = []
    control_curves: list[np.ndarray] = []
    used_ids: list[int] = []
    for vesicle_id in common_ids:
        f_pts = fusion_by_vesicle[vesicle_id]
        c_pts = control_by_vesicle[vesicle_id]
        if len(f_pts) == 0 or len(c_pts) == 0:
            continue
        fusion_curves.append(
            _ripley_o_from_points(
                f_pts, target_xyz, mesh_verts, mesh_faces, r_vals, geodesic_method=geodesic_method
            )
        )
        control_curves.append(
            _ripley_o_from_points(
                c_pts, target_xyz, mesh_verts, mesh_faces, r_vals, geodesic_method=geodesic_method
            )
        )
        used_ids.append(vesicle_id)
    if not fusion_curves:
        empty = np.empty((0, len(r_vals)))
        return empty, empty, []
    return np.vstack(fusion_curves), np.vstack(control_curves), used_ids


def _label_permutation_o_curves(
    pool_xyz: np.ndarray,
    n_type_a: int,
    mesh_verts: np.ndarray,
    mesh_faces: np.ndarray,
    r_vals: np.ndarray,
    n_perm: int,
    rng: np.random.Generator,
    *,
    geodesic_method: str = DEFAULT_RIPLEY_O_GEODESIC_METHOD,
) -> np.ndarray:
    pool_xyz = np.atleast_2d(np.asarray(pool_xyz, dtype=float))
    curves = np.empty((n_perm, len(r_vals)), dtype=float)
    for p in range(n_perm):
        labels = np.zeros(len(pool_xyz), dtype=bool)
        labels[rng.choice(len(pool_xyz), n_type_a, replace=False)] = True
        curves[p] = _ripley_o_from_points(
            pool_xyz[labels],
            pool_xyz[~labels],
            mesh_verts,
            mesh_faces,
            r_vals,
            geodesic_method=geodesic_method,
        )
    return curves


def _project_points_by_vesicle(
    df: pd.DataFrame,
    xyz_cols: tuple[str, str, str],
    post_tree: cKDTree,
    post_surface: np.ndarray,
) -> dict[int, np.ndarray]:
    points_by_vesicle: dict[int, np.ndarray] = {}
    for vesicle_id, group in df.groupby("vesicle_id"):
        xyz = group[list(xyz_cols)].to_numpy(dtype=float)
        if len(xyz):
            points_by_vesicle[int(vesicle_id)] = _project_points_to_surface(xyz, post_tree, post_surface)
    return points_by_vesicle


def _per_vesicle_h12_curves(
    points_by_vesicle: dict[int, np.ndarray],
    aunp_post: np.ndarray,
    r_vals: np.ndarray,
    window_area_nm2: float,
) -> np.ndarray:
    curves = [
        ripley_h12_from_points(pts, aunp_post, r_vals, window_area_nm2)
        for _vid in sorted(points_by_vesicle)
        for pts in [points_by_vesicle[_vid]]
        if len(pts) > 0
    ]
    if not curves:
        return np.empty((0, len(r_vals)))
    return np.vstack(curves)


def _paired_vesicle_h12_curves(
    fusion_by_vesicle: dict[int, np.ndarray],
    control_by_vesicle: dict[int, np.ndarray],
    aunp_post: np.ndarray,
    r_vals: np.ndarray,
    window_area_nm2: float,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Per-vesicle H₁₂ curves aligned on common vesicle IDs."""
    common_ids = sorted(set(fusion_by_vesicle) & set(control_by_vesicle))
    fusion_curves: list[np.ndarray] = []
    control_curves: list[np.ndarray] = []
    used_ids: list[int] = []
    for vesicle_id in common_ids:
        f_pts = fusion_by_vesicle[vesicle_id]
        c_pts = control_by_vesicle[vesicle_id]
        if len(f_pts) == 0 or len(c_pts) == 0:
            continue
        fusion_curves.append(ripley_h12_from_points(f_pts, aunp_post, r_vals, window_area_nm2))
        control_curves.append(ripley_h12_from_points(c_pts, aunp_post, r_vals, window_area_nm2))
        used_ids.append(vesicle_id)
    if not fusion_curves:
        empty = np.empty((0, len(r_vals)))
        return empty, empty, []
    return np.vstack(fusion_curves), np.vstack(control_curves), used_ids


def _monte_carlo_p_two_sided(observed: np.ndarray, null_samples: np.ndarray) -> np.ndarray:
    """Two-sided permutation p-value at each radius (includes +1 correction)."""
    observed = np.asarray(observed, dtype=float)
    null_samples = np.asarray(null_samples, dtype=float)
    if null_samples.ndim != 2:
        raise ValueError("null_samples must be (n_samples, n_r)")
    n_samples = null_samples.shape[0]
    return (np.sum(np.abs(null_samples) >= np.abs(observed), axis=0) + 1) / (n_samples + 1)


def _neglog10_p(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.log10(np.clip(p_values, 1e-300, 1.0))


def _wilcoxon_min_achievable_p(n_pairs: int) -> float:
    """Smallest two-sided Wilcoxon p-value achievable with n paired replicates."""
    if n_pairs < 2:
        return float("nan")
    from scipy import stats

    try:
        return float(stats.wilcoxon(np.ones(n_pairs), alternative="two-sided", zero_method="wilcox").pvalue)
    except ValueError:
        return float("nan")


def _decorate_significance_axis(
    ax,
    r_vals: np.ndarray,
    p_vals: np.ndarray,
    *,
    alpha: float = 0.05,
    marginal_alpha: float = 0.10,
    panel_note: str | None = None,
) -> None:
    """Plot -log10(p) with tiered markers and panel annotation."""
    p_vals = np.asarray(p_vals, dtype=float)
    neglog = _neglog10_p(p_vals)
    ax.plot(r_vals, neglog, color="C0", lw=1.8)

    sig_strict = p_vals < alpha
    sig_marginal = (p_vals >= alpha) & (p_vals < marginal_alpha)
    if np.any(sig_strict):
        ax.scatter(
            r_vals[sig_strict],
            neglog[sig_strict],
            color="C3",
            s=30,
            zorder=4,
            edgecolors="k",
            linewidths=0.3,
            label=f"p < {alpha:g}",
        )
    if np.any(sig_marginal):
        ax.scatter(
            r_vals[sig_marginal],
            neglog[sig_marginal],
            color="C1",
            s=22,
            zorder=3,
            edgecolors="k",
            linewidths=0.2,
            label=f"p < {marginal_alpha:g}",
        )

    ax.axhline(-np.log10(alpha), color="k", ls="--", lw=1.0, alpha=0.7)
    ax.axhline(-np.log10(marginal_alpha), color="0.55", ls=":", lw=1.0, alpha=0.7)
    ax.set_ylim(bottom=0.0)

    finite = p_vals[np.isfinite(p_vals)]
    min_p = float(np.min(finite)) if len(finite) else float("nan")
    note_lines = [f"min p = {min_p:.3g}", f"{int(sig_strict.sum())} @ p<{alpha:g}"]
    if np.any(sig_marginal):
        note_lines.append(f"{int(sig_marginal.sum())} @ p<{marginal_alpha:g}")
    if panel_note:
        note_lines.append(panel_note)
    ax.text(
        0.02,
        0.98,
        "\n".join(note_lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=7,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, pad=0.25),
    )


def _vesicle_paired_pvalues(
    fusion_curves: np.ndarray,
    control_curves: np.ndarray,
) -> np.ndarray:
    """Wilcoxon signed-rank p-value per radius on paired per-vesicle H₁₂ curves."""
    from scipy import stats

    if len(fusion_curves) < 2 or len(control_curves) < 2:
        return np.full(fusion_curves.shape[1] if fusion_curves.ndim == 2 else 0, np.nan)
    diffs = fusion_curves - control_curves
    p_vals = np.empty(diffs.shape[1], dtype=float)
    for k in range(diffs.shape[1]):
        sample = diffs[:, k]
        if np.allclose(sample, 0.0):
            p_vals[k] = 1.0
            continue
        try:
            res = stats.wilcoxon(sample, alternative="two-sided", zero_method="wilcox")
            p_vals[k] = float(res.pvalue)
        except ValueError:
            p_vals[k] = 1.0
    return p_vals


def _plot_significance_panels(
    r_vals: np.ndarray,
    p_by_label: dict[str, np.ndarray],
    *,
    title: str,
    output_path: Path,
    alpha: float = 0.05,
    marginal_alpha: float = 0.10,
    panel_notes: dict[str, str] | None = None,
) -> None:
    n_panels = len(p_by_label)
    ncols = min(3, n_panels)
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.8 * nrows), squeeze=False)
    axes_flat = axes.flatten()
    panel_notes = panel_notes or {}
    for ax, (label, p_vals) in zip(axes_flat, p_by_label.items()):
        _decorate_significance_axis(
            ax,
            r_vals,
            p_vals,
            alpha=alpha,
            marginal_alpha=marginal_alpha,
            panel_note=panel_notes.get(label),
        )
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("r (nm)")
        ax.set_ylabel("-log10(p)")
    for ax in axes_flat[n_panels:]:
        ax.set_visible(False)
    fig.suptitle(title, y=1.02, fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_significance_single(
    r_vals: np.ndarray,
    p_vals: np.ndarray,
    *,
    title: str,
    output_path: Path,
    label: str = "two-sided p",
    alpha: float = 0.05,
    marginal_alpha: float = 0.10,
    panel_note: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    _decorate_significance_axis(
        ax,
        r_vals,
        p_vals,
        alpha=alpha,
        marginal_alpha=marginal_alpha,
        panel_note=panel_note,
    )
    ax.set_xlabel("r (nm)")
    ax.set_ylabel("-log10(p)")
    ax.set_title(title)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _replicate_mean(curves: np.ndarray) -> np.ndarray:
    """Mean across replicate curves (n_replicates × n_r)."""
    curves = np.asarray(curves, dtype=float)
    if curves.ndim != 2 or len(curves) == 0:
        n_r = curves.shape[1] if curves.ndim == 2 and curves.size else 0
        return np.full(n_r, np.nan)
    return curves.mean(axis=0)


def _replicate_percentile_band(
    curves: np.ndarray,
    lo_pct: float = RIPLEY_PERCENTILE_LO,
    hi_pct: float = RIPLEY_PERCENTILE_HI,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Percentile band and median across replicate curves (n_replicates × n_r)."""
    curves = np.asarray(curves, dtype=float)
    if curves.ndim != 2 or len(curves) == 0:
        n_r = curves.shape[1] if curves.ndim == 2 and curves.size else 0
        nan = np.full(n_r, np.nan)
        return nan, nan, nan
    return (
        np.percentile(curves, lo_pct, axis=0),
        np.median(curves, axis=0),
        np.percentile(curves, hi_pct, axis=0),
    )


def _plot_ripley_dual_envelope_panel(
    ax: plt.Axes,
    r_vals: np.ndarray,
    *,
    secondary_lo: np.ndarray,
    secondary_med: np.ndarray,
    secondary_hi: np.ndarray,
    secondary_band_label: str,
    secondary_median_label: str,
    primary_lo: np.ndarray,
    primary_med: np.ndarray,
    primary_hi: np.ndarray,
    primary_band_label: str,
    primary_median_label: str,
    ylabel: str,
    title: str | None = None,
    panel_title: str | None = None,
    refline: float | None = None,
    refline_label: str | None = None,
    legend_fontsize: float = 8,
    secondary_show_envelope: bool = True,
    secondary_line: np.ndarray | None = None,
    secondary_line_label: str | None = None,
    primary_show_envelope: bool = True,
    primary_line: np.ndarray | None = None,
    primary_line_label: str | None = None,
) -> None:
    """Symmetric 2.5–97.5% envelopes with median center lines for two curve groups."""
    secondary_plot = secondary_line if secondary_line is not None else secondary_med
    secondary_label = (
        secondary_line_label if secondary_line_label is not None else secondary_median_label
    )
    if secondary_show_envelope:
        ax.fill_between(
            r_vals,
            secondary_lo,
            secondary_hi,
            color="0.85",
            zorder=1,
            label=secondary_band_label,
        )
        ax.plot(
            r_vals,
            secondary_plot,
            color="0.45",
            lw=1.8,
            zorder=2,
            label=secondary_label,
        )
    else:
        ax.plot(
            r_vals,
            secondary_plot,
            color="0.45",
            lw=1.8,
            zorder=2,
            label=secondary_label,
        )
    fusion_line = primary_line if primary_line is not None else primary_med
    fusion_label = primary_line_label if primary_line_label is not None else primary_median_label
    if primary_show_envelope:
        ax.fill_between(
            r_vals,
            primary_lo,
            primary_hi,
            color="C3",
            alpha=0.25,
            zorder=3,
            label=primary_band_label,
        )
        ax.plot(
            r_vals,
            fusion_line,
            color="C3",
            lw=2.0,
            zorder=4,
            label=fusion_label,
        )
    else:
        ax.plot(
            r_vals,
            fusion_line,
            color="C3",
            lw=2.0,
            zorder=4,
            label=fusion_label,
        )
    if refline is not None:
        ax.axhline(refline, color="k", lw=0.8, alpha=0.5, label=refline_label)
    if panel_title is not None:
        ax.set_title(panel_title)
    elif title is not None:
        ax.set_title(title)
    ax.set_xlabel("r (nm)")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=legend_fontsize)


def _plot_label_permutation_envelope_panel(
    ax: plt.Axes,
    r_vals: np.ndarray,
    *,
    null_lo: np.ndarray,
    null_med: np.ndarray,
    null_hi: np.ndarray,
    n_perm: int,
    obs_lo: np.ndarray,
    obs_med: np.ndarray,
    obs_hi: np.ndarray,
    n_obs_replicates: int,
    ylabel: str,
    title: str,
    refline: float | None = None,
    refline_label: str | None = None,
    fusion_mean_only: bool = False,
    obs_mean: np.ndarray | None = None,
    null_mean: np.ndarray | None = None,
) -> None:
    _plot_ripley_dual_envelope_panel(
        ax,
        r_vals,
        secondary_lo=null_lo,
        secondary_med=null_med,
        secondary_hi=null_hi,
        secondary_band_label=f"label null 2.5–97.5% (n={n_perm})",
        secondary_median_label="label null median",
        primary_lo=obs_lo,
        primary_med=obs_med,
        primary_hi=obs_hi,
        primary_band_label=f"observed 2.5–97.5% (n={n_obs_replicates} vesicles)",
        primary_median_label="observed median (per-vesicle)",
        ylabel=ylabel,
        title=title,
        refline=refline,
        refline_label=refline_label,
        secondary_show_envelope=not fusion_mean_only,
        secondary_line=null_mean,
        secondary_line_label=(
            f"label null mean (n={n_perm})" if fusion_mean_only else None
        ),
        primary_show_envelope=not fusion_mean_only,
        primary_line=obs_mean,
        primary_line_label=(
            f"fusion mean (n={n_obs_replicates} vesicles)" if fusion_mean_only else None
        ),
    )


def _plot_fusion_vs_control_ripley_panel(
    ax: plt.Axes,
    r_vals: np.ndarray,
    *,
    ctrl_lo: np.ndarray,
    ctrl_med: np.ndarray,
    ctrl_hi: np.ndarray,
    n_control_vesicles: int,
    offset_nm: float,
    fusion_lo: np.ndarray,
    fusion_med: np.ndarray,
    fusion_hi: np.ndarray,
    n_fusion_vesicles: int,
    ylabel: str,
    refline: float | None = None,
    refline_label: str | None = None,
    fusion_mean_only: bool = False,
    fusion_mean: np.ndarray | None = None,
    ctrl_mean: np.ndarray | None = None,
) -> None:
    _plot_ripley_dual_envelope_panel(
        ax,
        r_vals,
        secondary_lo=ctrl_lo,
        secondary_med=ctrl_med,
        secondary_hi=ctrl_hi,
        secondary_band_label=(
            f"controls d={int(offset_nm)} nm 2.5–97.5% (n={n_control_vesicles} vesicles)"
        ),
        secondary_median_label=f"controls d={int(offset_nm)} nm median",
        primary_lo=fusion_lo,
        primary_med=fusion_med,
        primary_hi=fusion_hi,
        primary_band_label=f"fusion 2.5–97.5% (n={n_fusion_vesicles} vesicles)",
        primary_median_label="fusion median (per-vesicle)",
        ylabel=ylabel,
        panel_title=f"d = {int(offset_nm)} nm",
        refline=refline,
        refline_label=refline_label,
        legend_fontsize=7,
        secondary_show_envelope=not fusion_mean_only,
        secondary_line=ctrl_mean,
        secondary_line_label=(
            f"controls d={int(offset_nm)} nm mean (n={n_control_vesicles} vesicles)"
            if fusion_mean_only
            else None
        ),
        primary_show_envelope=not fusion_mean_only,
        primary_line=fusion_mean,
        primary_line_label=(
            f"fusion mean (n={n_fusion_vesicles} vesicles)" if fusion_mean_only else None
        ),
    )


def _label_permutation_h12_curves(
    pool_xyz: np.ndarray,
    n_type_a: int,
    r_vals: np.ndarray,
    window_area_nm2: float,
    n_perm: int,
    rng: np.random.Generator,
) -> np.ndarray:
    pool_xyz = np.atleast_2d(np.asarray(pool_xyz, dtype=float))
    curves = np.empty((n_perm, len(r_vals)), dtype=float)
    for p in range(n_perm):
        labels = np.zeros(len(pool_xyz), dtype=bool)
        labels[rng.choice(len(pool_xyz), n_type_a, replace=False)] = True
        curves[p] = ripley_h12_from_points(pool_xyz[labels], pool_xyz[~labels], r_vals, window_area_nm2)
    return curves


def _label_permutation_h12_envelope(
    pool_xyz: np.ndarray,
    n_type_a: int,
    r_vals: np.ndarray,
    window_area_nm2: float,
    n_perm: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Null H12(r) envelope from random fusion/AuNP label reassignment on fixed positions."""
    pool_xyz = np.atleast_2d(np.asarray(pool_xyz, dtype=float))
    if len(pool_xyz) < n_type_a or n_type_a <= 0:
        nan = np.full(len(r_vals), np.nan)
        return nan, nan
    curves = _label_permutation_h12_curves(
        pool_xyz, n_type_a, r_vals, window_area_nm2, n_perm, rng
    )
    return np.percentile(curves, 2.5, axis=0), np.percentile(curves, 97.5, axis=0)


def _load_aunp_coordinates_for_zone(
    tomogram_path: Path,
    alignment_dir: str,
    zone_name: str,
    post_surface_tree: cKDTree,
    *,
    max_snap_to_surface_nm: float = 50.0,
) -> np.ndarray:
    import starfile
    from .activezone import load_active_zone_mapping

    aunps_dir = tomogram_path / alignment_dir / "STT_results" / "aunps"
    star_path = aunps_dir / "aunp_clusters.star"
    if not star_path.is_file():
        picks = sorted((tomogram_path / alignment_dir / "aunps").glob("aunp_tm_BP_active_zone_*_manual_refined.star"))
        if not picks:
            raise FileNotFoundError(f"No AuNP STAR file found under {tomogram_path / alignment_dir}")
        star_path = picks[0]

    star_data = starfile.read(star_path)
    if isinstance(star_data, dict):
        aunp_df = next(v for v in star_data.values() if isinstance(v, pd.DataFrame))
    else:
        aunp_df = star_data

    if "active_zone" in aunp_df.columns:
        mapping = load_active_zone_mapping(tomogram_path, alignment_dir)
        if mapping:
            mapping = {int(k): v for k, v in mapping.items()}
            az_ids = [idx for idx, zname in mapping.items() if zname == zone_name]
            if az_ids:
                aunp_df = aunp_df[aunp_df["active_zone"].isin(az_ids)]

    coords = aunp_df[["faCoordinateX", "faCoordinateY", "faCoordinateZ"]].to_numpy(dtype=float)
    if len(coords) == 0:
        return np.zeros((0, 3), dtype=float)

    snap_dist, _ = post_surface_tree.query(coords, k=1)
    keep = snap_dist <= max_snap_to_surface_nm
    return coords[keep]


def _dedupe_rows_by_xyz(df: pd.DataFrame, cols: tuple[str, str, str], *, decimals: int = 3) -> pd.DataFrame:
    if df.empty:
        return df
    rounded = df[list(cols)].round(decimals)
    return df.loc[~rounded.duplicated()].copy()


def run_ripley_postsynaptic_analysis(
    df: pd.DataFrame,
    tomogram_path: Path,
    alignment_dir: str,
    output_dir: Path,
    *,
    r_max_nm: float = DEFAULT_RIPLEY_R_MAX_NM,
    r_step_nm: float = DEFAULT_RIPLEY_R_STEP_NM,
    n_perm: int = DEFAULT_RIPLEY_N_PERM,
    seed: int = 42,
    probe_radius_for_coords: float | None = None,
) -> pd.DataFrame | None:
    """
    Bivariate Ripley H on postsynaptic-projected fusion / control / AuNP positions.

    #1: fusion vs controls-at-d — per-vesicle 2.5–97.5% bands + median, per d.
    #2: label-permutation null on pooled fusion + AuNP projected positions.
    """
    zone_col = "nearest_scan_active_zone_name"
    if zone_col not in df.columns:
        print("Skipping Ripley analysis: nearest_scan_active_zone_name missing.")
        return None

    real = df[df["point_type"] == "fusion"].copy()
    ctrl = df[df["point_type"] == "control"].copy()
    if real.empty or ctrl.empty:
        print("Skipping Ripley analysis: no fusion or control rows.")
        return None

    if probe_radius_for_coords is None:
        probe_radius_for_coords = float(sorted(df["probe_radius_nm"].unique())[len(df["probe_radius_nm"].unique()) // 2])

    r_vals = _ripley_r_grid(r_max_nm, r_step_nm)
    rng = np.random.default_rng(seed)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    all_result_rows: list[dict] = []
    offsets = sorted(d for d in ctrl["control_offset_nm"].unique() if d > 0)

    for zone_name in sorted(real[zone_col].dropna().unique()):
        try:
            post_surface = _load_postsynaptic_active_zone_surface(tomogram_path, alignment_dir, str(zone_name))
        except FileNotFoundError as exc:
            print(f"Skipping Ripley for {zone_name}: {exc}")
            continue

        post_tree = cKDTree(post_surface)
        window_area = _estimate_planar_window_area_nm2(post_surface)

        try:
            aunp_xyz = _load_aunp_coordinates_for_zone(
                tomogram_path, alignment_dir, str(zone_name), post_tree
            )
        except FileNotFoundError as exc:
            print(f"Skipping Ripley for {zone_name}: {exc}")
            continue
        if len(aunp_xyz) < 3:
            print(f"Skipping Ripley for {zone_name}: too few AuNPs near postsynaptic surface.")
            continue

        aunp_post = _project_points_to_surface(aunp_xyz, post_tree, post_surface)

        sub_f = _dedupe_rows_by_xyz(
            real[
                (real["probe_radius_nm"] == probe_radius_for_coords)
                & (real[zone_col] == zone_name)
            ],
            ("fusion_point_x_nm", "fusion_point_y_nm", "fusion_point_z_nm"),
        )
        fusion_xyz = sub_f[["fusion_point_x_nm", "fusion_point_y_nm", "fusion_point_z_nm"]].to_numpy(dtype=float)
        fusion_post = _project_points_to_surface(fusion_xyz, post_tree, post_surface)

        if len(fusion_post) == 0:
            print(f"Skipping Ripley for {zone_name}: no fusion points.")
            continue

        fusion_by_vesicle = _project_points_by_vesicle(
            sub_f,
            ("fusion_point_x_nm", "fusion_point_y_nm", "fusion_point_z_nm"),
            post_tree,
            post_surface,
        )
        fusion_vesicle_curves = _per_vesicle_h12_curves(
            fusion_by_vesicle, aunp_post, r_vals, window_area
        )
        h12_fusion_lo, h12_fusion_med, h12_fusion_hi = _replicate_percentile_band(fusion_vesicle_curves)
        h12_fusion_mean = _replicate_mean(fusion_vesicle_curves)

        # --- #2 Label permutation (fusion vs AuNP) ---
        pool = np.vstack([fusion_post, aunp_post])
        h12_obs = ripley_h12_from_points(fusion_post, aunp_post, r_vals, window_area)
        perm_curves = _label_permutation_h12_curves(
            pool, len(fusion_post), r_vals, window_area, n_perm, rng
        )
        h12_null_lo, h12_null_med, h12_null_hi = _replicate_percentile_band(perm_curves)
        h12_null_mean = _replicate_mean(perm_curves)
        h12_obs_lo, h12_obs_med, h12_obs_hi = _replicate_percentile_band(fusion_vesicle_curves)
        p_label_two = _monte_carlo_p_two_sided(h12_obs, perm_curves)
        p_label_greater = (np.sum(perm_curves >= h12_obs, axis=0) + 1) / (n_perm + 1)
        p_label_less = (np.sum(perm_curves <= h12_obs, axis=0) + 1) / (n_perm + 1)

        fig, ax = plt.subplots(figsize=(7, 5))
        _plot_label_permutation_envelope_panel(
            ax,
            r_vals,
            null_lo=h12_null_lo,
            null_med=h12_null_med,
            null_hi=h12_null_hi,
            n_perm=n_perm,
            obs_lo=h12_obs_lo,
            obs_med=h12_obs_med,
            obs_hi=h12_obs_hi,
            n_obs_replicates=len(fusion_by_vesicle),
            ylabel="Ripley H₁₂(r) = √(K₁₂/π) − r",
            title=(
                f"Label-permutation null: fusion vs AuNP on postsynaptic AZ\n"
                f"{zone_name} (p-values: pooled observed vs null)"
            ),
            refline=0.0,
            refline_label="H₁₂ = 0",
        )
        fig.tight_layout()
        fig.savefig(figures_dir / f"ripley_h12_label_permutation_{zone_name}.png", dpi=150)
        plt.close(fig)

        fig_mean, ax_mean = plt.subplots(figsize=(7, 5))
        _plot_label_permutation_envelope_panel(
            ax_mean,
            r_vals,
            null_lo=h12_null_lo,
            null_med=h12_null_med,
            null_hi=h12_null_hi,
            n_perm=n_perm,
            obs_lo=h12_obs_lo,
            obs_med=h12_obs_med,
            obs_hi=h12_obs_hi,
            n_obs_replicates=len(fusion_by_vesicle),
            ylabel="Ripley H₁₂(r) = √(K₁₂/π) − r",
            title=(
                f"Label-permutation null: fusion vs AuNP on postsynaptic AZ\n"
                f"{zone_name} (null and fusion means only)"
            ),
            refline=0.0,
            refline_label="H₁₂ = 0",
            fusion_mean_only=True,
            obs_mean=h12_fusion_mean,
            null_mean=h12_null_mean,
        )
        fig_mean.tight_layout()
        fig_mean.savefig(
            figures_dir / f"ripley_h12_label_permutation_{zone_name}_fusion_mean.png",
            dpi=150,
        )
        plt.close(fig_mean)

        _plot_significance_single(
            r_vals,
            p_label_two,
            title=(
                f"Label-permutation significance (fusion vs AuNP)\n"
                f"{zone_name} | two-sided Monte Carlo p, n={n_perm}"
            ),
            output_path=figures_dir / f"ripley_h12_pvalues_label_permutation_{zone_name}.png",
            panel_note=f"n_fusion={len(fusion_post)}, n_aunp={len(aunp_post)}",
        )
        fig, ax = plt.subplots(figsize=(7, 4))
        neglog_greater = _neglog10_p(p_label_greater)
        neglog_less = _neglog10_p(p_label_less)
        ax.plot(r_vals, neglog_greater, color="C3", lw=1.8, label="p enrichment (H₁₂ ≥ null)")
        ax.plot(r_vals, neglog_less, color="C0", lw=1.8, label="p depletion (H₁₂ ≤ null)")
        for p_dir, neglog, color in (
            (p_label_greater, neglog_greater, "C3"),
            (p_label_less, neglog_less, "C0"),
        ):
            sig = p_dir < 0.05
            marginal = (p_dir >= 0.05) & (p_dir < 0.10)
            if np.any(sig):
                ax.scatter(r_vals[sig], neglog[sig], color=color, s=24, zorder=4, edgecolors="k", linewidths=0.2)
            if np.any(marginal):
                ax.scatter(r_vals[marginal], neglog[marginal], color="C1", s=16, zorder=3, edgecolors="k", linewidths=0.2)
        ax.axhline(-np.log10(0.05), color="k", ls="--", lw=1.0, alpha=0.7)
        ax.axhline(-np.log10(0.10), color="0.55", ls=":", lw=1.0, alpha=0.7)
        ax.set_xlabel("r (nm)")
        ax.set_ylabel("-log10(p)")
        ax.set_title(f"Directional label-permutation p-values\n{zone_name}")
        ax.legend(fontsize=8)
        ax.set_ylim(bottom=0.0)
        fig.tight_layout()
        fig.savefig(figures_dir / f"ripley_h12_pvalues_label_permutation_directional_{zone_name}.png", dpi=150)
        plt.close(fig)

        for r, obs, n_lo, n_med, n_hi, o_lo, o_med, o_hi, f_mean, p2, pg, pl in zip(
            r_vals,
            h12_obs,
            h12_null_lo,
            h12_null_med,
            h12_null_hi,
            h12_obs_lo,
            h12_obs_med,
            h12_obs_hi,
            h12_fusion_mean,
            p_label_two,
            p_label_greater,
            p_label_less,
        ):
            all_result_rows.append(
                {
                    "zone_name": zone_name,
                    "analysis": "label_permutation",
                    "uncertainty_method": UNCERTAINTY_METHOD_PERCENTILE,
                    "control_offset_nm": np.nan,
                    "r_nm": float(r),
                    "h12_observed": float(obs),
                    "h12_null_lo": float(n_lo),
                    "h12_null_hi": float(n_hi),
                    "h12_null_median": float(n_med),
                    "h12_obs_lo": float(o_lo),
                    "h12_obs_hi": float(o_hi),
                    "h12_obs_median": float(o_med),
                    "p_value_two_sided": float(p2),
                    "p_value_enrichment": float(pg),
                    "p_value_depletion": float(pl),
                    "h12_fusion_mean": float(f_mean),
                    "h12_fusion_sem": np.nan,
                    "h12_fusion_lo": float(o_lo),
                    "h12_fusion_hi": float(o_hi),
                    "h12_fusion_median": float(o_med),
                    "h12_control_mean": np.nan,
                    "h12_control_sem": np.nan,
                    "n_fusion": len(fusion_post),
                    "n_fusion_vesicles": len(fusion_by_vesicle),
                    "n_aunp": len(aunp_post),
                    "window_area_nm2": window_area,
                }
            )

        n_panels = len(offsets)
        ncols = min(3, n_panels)
        nrows = int(np.ceil(n_panels / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4 * nrows), squeeze=False)
        fig_mean, axes_mean = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4 * nrows), squeeze=False)
        axes_flat = axes.flatten()
        axes_mean_flat = axes_mean.flatten()
        p_by_d: dict[str, np.ndarray] = {}
        panel_notes: dict[str, str] = {}
        fvc_by_offset: dict[float, tuple[np.ndarray, np.ndarray]] = {}

        for ax, ax_mean, offset_nm in zip(axes_flat, axes_mean_flat, offsets):
            sub_c = _dedupe_rows_by_xyz(
                ctrl[
                    (ctrl["probe_radius_nm"] == probe_radius_for_coords)
                    & (ctrl["control_offset_nm"] == offset_nm)
                    & (ctrl[zone_col] == zone_name)
                ],
                ("query_point_x_nm", "query_point_y_nm", "query_point_z_nm"),
            )
            if sub_c.empty:
                ax.set_title(f"d={int(offset_nm)} nm (no controls)")
                ax_mean.set_title(f"d={int(offset_nm)} nm (no controls)")
                continue

            control_by_vesicle = _project_points_by_vesicle(
                sub_c,
                ("query_point_x_nm", "query_point_y_nm", "query_point_z_nm"),
                post_tree,
                post_surface,
            )
            control_post = np.vstack(list(control_by_vesicle.values()))

            fusion_curves, ctrl_curves, paired_ids = _paired_vesicle_h12_curves(
                fusion_by_vesicle,
                control_by_vesicle,
                aunp_post,
                r_vals,
                window_area,
            )
            h12_ctrl_lo, h12_ctrl_med, h12_ctrl_hi = _replicate_percentile_band(ctrl_curves)
            h12_ctrl_mean = _replicate_mean(ctrl_curves)
            p_fusion_vs_ctrl = _vesicle_paired_pvalues(fusion_curves, ctrl_curves)
            fvc_by_offset[float(offset_nm)] = (fusion_curves, ctrl_curves)

            p_by_d[f"d={int(offset_nm)} nm"] = p_fusion_vs_ctrl
            min_p_floor = _wilcoxon_min_achievable_p(len(paired_ids))
            panel_notes[f"d={int(offset_nm)} nm"] = (
                f"n_paired={len(paired_ids)}"
                + (
                    f"\nWilcoxon floor={min_p_floor:.3g}"
                    if np.isfinite(min_p_floor) and min_p_floor > 0.05
                    else ""
                )
            )

            _plot_fusion_vs_control_ripley_panel(
                ax,
                r_vals,
                ctrl_lo=h12_ctrl_lo,
                ctrl_med=h12_ctrl_med,
                ctrl_hi=h12_ctrl_hi,
                n_control_vesicles=len(control_by_vesicle),
                offset_nm=float(offset_nm),
                fusion_lo=h12_fusion_lo,
                fusion_med=h12_fusion_med,
                fusion_hi=h12_fusion_hi,
                n_fusion_vesicles=len(fusion_by_vesicle),
                ylabel="H₁₂(r)",
                refline=0.0,
                refline_label="H₁₂ = 0",
            )
            _plot_fusion_vs_control_ripley_panel(
                ax_mean,
                r_vals,
                ctrl_lo=h12_ctrl_lo,
                ctrl_med=h12_ctrl_med,
                ctrl_hi=h12_ctrl_hi,
                n_control_vesicles=len(control_by_vesicle),
                offset_nm=float(offset_nm),
                fusion_lo=h12_fusion_lo,
                fusion_med=h12_fusion_med,
                fusion_hi=h12_fusion_hi,
                n_fusion_vesicles=len(fusion_by_vesicle),
                ylabel="H₁₂(r)",
                refline=0.0,
                refline_label="H₁₂ = 0",
                fusion_mean_only=True,
                fusion_mean=h12_fusion_mean,
                ctrl_mean=h12_ctrl_mean,
            )

            for r, f_lo, f_med, f_hi, f_mean, c_lo, c_med, c_hi, p_val in zip(
                r_vals,
                h12_fusion_lo,
                h12_fusion_med,
                h12_fusion_hi,
                h12_fusion_mean,
                h12_ctrl_lo,
                h12_ctrl_med,
                h12_ctrl_hi,
                p_fusion_vs_ctrl,
            ):
                all_result_rows.append(
                    {
                        "zone_name": zone_name,
                        "analysis": "fusion_vs_control",
                        "uncertainty_method": UNCERTAINTY_METHOD_PERCENTILE,
                        "control_offset_nm": float(offset_nm),
                        "r_nm": float(r),
                        "h12_fusion_lo": float(f_lo),
                        "h12_fusion_median": float(f_med),
                        "h12_fusion_hi": float(f_hi),
                        "h12_control_lo": float(c_lo),
                        "h12_control_median": float(c_med),
                        "h12_control_hi": float(c_hi),
                        "h12_fusion_mean": float(f_mean),
                        "h12_fusion_sem": np.nan,
                        "h12_control_mean": float(c_med),
                        "h12_control_sem": np.nan,
                        "p_value_two_sided": float(p_val),
                        "n_fusion": len(fusion_post),
                        "n_fusion_vesicles": len(fusion_by_vesicle),
                        "n_control": len(control_post),
                        "n_control_vesicles": len(control_by_vesicle),
                        "n_paired_vesicles": len(paired_ids),
                        "n_aunp": len(aunp_post),
                        "window_area_nm2": window_area,
                    }
                )

        for ax in axes_flat[n_panels:]:
            ax.set_visible(False)
        fig.suptitle(
            f"Ripley H₁₂ on postsynaptic AZ: fusion vs controls "
            f"(2.5–97.5% bands + median across per-vesicle H₁₂ curves)\n{zone_name}",
            y=1.02,
        )
        fig.tight_layout()
        fig.savefig(
            figures_dir / f"ripley_h12_fusion_vs_controls_by_d_{zone_name}.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)

        for ax_mean in axes_mean_flat[n_panels:]:
            ax_mean.set_visible(False)
        fig_mean.suptitle(
            f"Ripley H₁₂ on postsynaptic AZ: fusion vs controls "
            f"(fusion and control means only)\n{zone_name}",
            y=1.02,
        )
        fig_mean.tight_layout()
        fig_mean.savefig(
            figures_dir / f"ripley_h12_fusion_vs_controls_by_d_{zone_name}_fusion_mean.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig_mean)

        if p_by_d:
            n_paired_global = len(fusion_by_vesicle)
            floor_note = ""
            floor_p = _wilcoxon_min_achievable_p(n_paired_global)
            if np.isfinite(floor_p) and floor_p > 0.05:
                floor_note = f" | Wilcoxon floor p={floor_p:.3g} (n={n_paired_global})"
            _plot_significance_panels(
                r_vals,
                p_by_d,
                title=(
                    f"Fusion vs controls significance (paired Wilcoxon on per-vesicle H₁₂)\n"
                    f"{zone_name} | red p<0.05, orange p<0.10{floor_note}"
                ),
                output_path=figures_dir / f"ripley_h12_pvalues_fusion_vs_control_{zone_name}.png",
                panel_notes=panel_notes,
            )

        _save_ripley_h12_vesicle_artifacts(
            output_dir / RIPLEY_H12_VESICLE_CURVES_NPZ,
            r_vals=r_vals,
            zone_name=str(zone_name),
            tomogram_name=tomogram_path.name,
            fusion_vesicle_curves=fusion_vesicle_curves,
            label_perm_null_curves=perm_curves,
            h12_obs=h12_obs,
            fusion_vs_control_by_offset=fvc_by_offset,
        )

        print(
            f"  Ripley H₁₂: {zone_name} — {len(fusion_post)} fusion, {len(aunp_post)} AuNPs "
            f"(projected to postsynaptic AZ, area≈{window_area:.0f} nm²)"
        )

    if not all_result_rows:
        return None
    out_df = pd.DataFrame(all_result_rows)
    out_df.to_csv(output_dir / "ripley_h12_postsynaptic.csv", index=False)
    return out_df


def run_ripley_o_membrain_postsynaptic_analysis(
    df: pd.DataFrame,
    tomogram_path: Path,
    alignment_dir: str,
    output_dir: Path,
    *,
    r_max_nm: float = DEFAULT_RIPLEY_R_MAX_NM,
    r_step_nm: float = DEFAULT_RIPLEY_R_STEP_NM,
    n_perm: int = DEFAULT_RIPLEY_N_PERM,
    mesh_max_vertices: int = DEFAULT_RIPLEY_O_MESH_MAX_VERTS,
    geodesic_method: str = DEFAULT_RIPLEY_O_GEODESIC_METHOD,
    seed: int = 42,
    probe_radius_for_coords: float | None = None,
) -> pd.DataFrame | None:
    """
    Bivariate geodesic Ripley's O (membrain-stats) on postsynaptic-projected fusion,
    control, and AuNP positions.

    Uses membrain_stats.compute_ripleys_stats + aggregate_ripleys_stats (ripley_type='O')
    on a coarse Delaunay mesh of the postsynaptic AZ patch near the analysis points.
    """
    try:
        _import_membrain_ripley()
    except ImportError as exc:
        print(f"Skipping Ripley's O analysis: {exc}")
        return None

    zone_col = "nearest_scan_active_zone_name"
    if zone_col not in df.columns:
        print("Skipping Ripley's O analysis: nearest_scan_active_zone_name missing.")
        return None

    real = df[df["point_type"] == "fusion"].copy()
    ctrl = df[df["point_type"] == "control"].copy()
    if real.empty or ctrl.empty:
        print("Skipping Ripley's O analysis: no fusion or control rows.")
        return None

    if probe_radius_for_coords is None:
        probe_radius_for_coords = float(sorted(df["probe_radius_nm"].unique())[len(df["probe_radius_nm"].unique()) // 2])

    r_vals = _ripley_r_grid(r_max_nm, r_step_nm)
    r_patch_nm = r_max_nm + r_step_nm
    rng = np.random.default_rng(seed)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    all_result_rows: list[dict] = []
    offsets = sorted(d for d in ctrl["control_offset_nm"].unique() if d > 0)

    for zone_name in sorted(real[zone_col].dropna().unique()):
        try:
            post_surface = _load_postsynaptic_active_zone_surface(tomogram_path, alignment_dir, str(zone_name))
        except FileNotFoundError as exc:
            print(f"Skipping Ripley's O for {zone_name}: {exc}")
            continue

        post_tree = cKDTree(post_surface)

        try:
            aunp_xyz = _load_aunp_coordinates_for_zone(
                tomogram_path, alignment_dir, str(zone_name), post_tree
            )
        except FileNotFoundError as exc:
            print(f"Skipping Ripley's O for {zone_name}: {exc}")
            continue
        if len(aunp_xyz) < 3:
            print(f"Skipping Ripley's O for {zone_name}: too few AuNPs near postsynaptic surface.")
            continue

        aunp_post = _project_points_to_surface(aunp_xyz, post_tree, post_surface)

        sub_f = _dedupe_rows_by_xyz(
            real[
                (real["probe_radius_nm"] == probe_radius_for_coords)
                & (real[zone_col] == zone_name)
            ],
            ("fusion_point_x_nm", "fusion_point_y_nm", "fusion_point_z_nm"),
        )
        fusion_xyz = sub_f[["fusion_point_x_nm", "fusion_point_y_nm", "fusion_point_z_nm"]].to_numpy(dtype=float)
        fusion_post = _project_points_to_surface(fusion_xyz, post_tree, post_surface)

        if len(fusion_post) == 0:
            print(f"Skipping Ripley's O for {zone_name}: no fusion points.")
            continue

        try:
            mesh_verts, mesh_faces = _build_membrain_o_analysis_mesh(
                post_surface,
                np.vstack([fusion_post, aunp_post]),
                r_patch_nm=r_patch_nm,
                max_vertices=mesh_max_vertices,
                rng=rng,
            )
        except ValueError as exc:
            print(f"Skipping Ripley's O for {zone_name}: {exc}")
            continue

        fusion_by_vesicle = _project_points_by_vesicle(
            sub_f,
            ("fusion_point_x_nm", "fusion_point_y_nm", "fusion_point_z_nm"),
            post_tree,
            post_surface,
        )
        fusion_vesicle_curves = _per_vesicle_o_curves(
            fusion_by_vesicle,
            aunp_post,
            mesh_verts,
            mesh_faces,
            r_vals,
            geodesic_method=geodesic_method,
        )
        o_fusion_lo, o_fusion_med, o_fusion_hi = _replicate_percentile_band(fusion_vesicle_curves)
        o_fusion_mean = _replicate_mean(fusion_vesicle_curves)

        # --- #2 Label permutation (fusion vs AuNP) ---
        pool = np.vstack([fusion_post, aunp_post])
        o_obs = _ripley_o_from_points(
            fusion_post, aunp_post, mesh_verts, mesh_faces, r_vals, geodesic_method=geodesic_method
        )
        perm_curves = _label_permutation_o_curves(
            pool, len(fusion_post), mesh_verts, mesh_faces, r_vals, n_perm, rng, geodesic_method=geodesic_method
        )
        o_null_lo, o_null_med, o_null_hi = _replicate_percentile_band(perm_curves)
        o_null_mean = _replicate_mean(perm_curves)
        o_obs_lo, o_obs_med, o_obs_hi = _replicate_percentile_band(fusion_vesicle_curves)
        p_label_two = _monte_carlo_p_two_sided(o_obs, perm_curves)
        p_label_greater = (np.sum(perm_curves >= o_obs, axis=0) + 1) / (n_perm + 1)
        p_label_less = (np.sum(perm_curves <= o_obs, axis=0) + 1) / (n_perm + 1)

        fig, ax = plt.subplots(figsize=(7, 5))
        _plot_label_permutation_envelope_panel(
            ax,
            r_vals,
            null_lo=o_null_lo,
            null_med=o_null_med,
            null_hi=o_null_hi,
            n_perm=n_perm,
            obs_lo=o_obs_lo,
            obs_med=o_obs_med,
            obs_hi=o_obs_hi,
            n_obs_replicates=len(fusion_by_vesicle),
            ylabel="Ripley's O(r) [membrain-stats geodesic]",
            title=(
                f"Label-permutation null: Ripley's O fusion vs AuNP\n"
                f"{zone_name} (p-values: pooled observed vs null)"
            ),
            refline=1.0,
            refline_label="CSR (O=1)",
        )
        fig.tight_layout()
        fig.savefig(figures_dir / f"ripley_o_label_permutation_{zone_name}.png", dpi=150)
        plt.close(fig)

        fig_mean, ax_mean = plt.subplots(figsize=(7, 5))
        _plot_label_permutation_envelope_panel(
            ax_mean,
            r_vals,
            null_lo=o_null_lo,
            null_med=o_null_med,
            null_hi=o_null_hi,
            n_perm=n_perm,
            obs_lo=o_obs_lo,
            obs_med=o_obs_med,
            obs_hi=o_obs_hi,
            n_obs_replicates=len(fusion_by_vesicle),
            ylabel="Ripley's O(r) [membrain-stats geodesic]",
            title=(
                f"Label-permutation null: Ripley's O fusion vs AuNP\n"
                f"{zone_name} (null and fusion means only)"
            ),
            refline=1.0,
            refline_label="CSR (O=1)",
            fusion_mean_only=True,
            obs_mean=o_fusion_mean,
            null_mean=o_null_mean,
        )
        fig_mean.tight_layout()
        fig_mean.savefig(
            figures_dir / f"ripley_o_label_permutation_{zone_name}_fusion_mean.png",
            dpi=150,
        )
        plt.close(fig_mean)

        _plot_significance_single(
            r_vals,
            p_label_two,
            title=(
                f"Ripley's O label-permutation significance\n"
                f"{zone_name} | two-sided Monte Carlo p, n={n_perm}"
            ),
            output_path=figures_dir / f"ripley_o_pvalues_label_permutation_{zone_name}.png",
            panel_note=f"n_fusion={len(fusion_post)}, n_aunp={len(aunp_post)}",
        )
        fig, ax = plt.subplots(figsize=(7, 4))
        neglog_greater = _neglog10_p(p_label_greater)
        neglog_less = _neglog10_p(p_label_less)
        ax.plot(r_vals, neglog_greater, color="C3", lw=1.8, label="p enrichment (O ≥ null)")
        ax.plot(r_vals, neglog_less, color="C0", lw=1.8, label="p depletion (O ≤ null)")
        for p_dir, neglog, color in (
            (p_label_greater, neglog_greater, "C3"),
            (p_label_less, neglog_less, "C0"),
        ):
            sig = p_dir < 0.05
            marginal = (p_dir >= 0.05) & (p_dir < 0.10)
            if np.any(sig):
                ax.scatter(r_vals[sig], neglog[sig], color=color, s=24, zorder=4, edgecolors="k", linewidths=0.2)
            if np.any(marginal):
                ax.scatter(r_vals[marginal], neglog[marginal], color="C1", s=16, zorder=3, edgecolors="k", linewidths=0.2)
        ax.axhline(-np.log10(0.05), color="k", ls="--", lw=1.0, alpha=0.7)
        ax.axhline(-np.log10(0.10), color="0.55", ls=":", lw=1.0, alpha=0.7)
        ax.set_xlabel("r (nm)")
        ax.set_ylabel("-log10(p)")
        ax.set_title(f"Directional Ripley's O label-permutation p-values\n{zone_name}")
        ax.legend(fontsize=8)
        ax.set_ylim(bottom=0.0)
        fig.tight_layout()
        fig.savefig(figures_dir / f"ripley_o_pvalues_label_permutation_directional_{zone_name}.png", dpi=150)
        plt.close(fig)

        for r, obs, n_lo, n_med, n_hi, o_lo, o_med, o_hi, f_mean, p2, pg, pl in zip(
            r_vals,
            o_obs,
            o_null_lo,
            o_null_med,
            o_null_hi,
            o_obs_lo,
            o_obs_med,
            o_obs_hi,
            o_fusion_mean,
            p_label_two,
            p_label_greater,
            p_label_less,
        ):
            all_result_rows.append(
                {
                    "zone_name": zone_name,
                    "analysis": "label_permutation",
                    "uncertainty_method": UNCERTAINTY_METHOD_PERCENTILE,
                    "control_offset_nm": np.nan,
                    "r_nm": float(r),
                    "o_observed": float(obs),
                    "o_null_lo": float(n_lo),
                    "o_null_hi": float(n_hi),
                    "o_null_median": float(n_med),
                    "o_obs_lo": float(o_lo),
                    "o_obs_hi": float(o_hi),
                    "o_obs_median": float(o_med),
                    "p_value_two_sided": float(p2),
                    "p_value_enrichment": float(pg),
                    "p_value_depletion": float(pl),
                    "o_fusion_mean": float(f_mean),
                    "o_fusion_sem": np.nan,
                    "o_fusion_lo": float(o_lo),
                    "o_fusion_hi": float(o_hi),
                    "o_fusion_median": float(o_med),
                    "o_control_mean": np.nan,
                    "o_control_sem": np.nan,
                    "n_fusion": len(fusion_post),
                    "n_fusion_vesicles": len(fusion_by_vesicle),
                    "n_aunp": len(aunp_post),
                    "mesh_vertices": len(mesh_verts),
                    "mesh_faces": len(mesh_faces),
                    "geodesic_method": geodesic_method,
                }
            )

        n_panels = len(offsets)
        ncols = min(3, n_panels)
        nrows = int(np.ceil(n_panels / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4 * nrows), squeeze=False)
        fig_mean, axes_mean = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4 * nrows), squeeze=False)
        axes_flat = axes.flatten()
        axes_mean_flat = axes_mean.flatten()
        p_by_d: dict[str, np.ndarray] = {}
        panel_notes: dict[str, str] = {}
        fvc_by_offset: dict[float, tuple[np.ndarray, np.ndarray]] = {}

        for ax, ax_mean, offset_nm in zip(axes_flat, axes_mean_flat, offsets):
            sub_c = _dedupe_rows_by_xyz(
                ctrl[
                    (ctrl["probe_radius_nm"] == probe_radius_for_coords)
                    & (ctrl["control_offset_nm"] == offset_nm)
                    & (ctrl[zone_col] == zone_name)
                ],
                ("query_point_x_nm", "query_point_y_nm", "query_point_z_nm"),
            )
            if sub_c.empty:
                ax.set_title(f"d={int(offset_nm)} nm (no controls)")
                ax_mean.set_title(f"d={int(offset_nm)} nm (no controls)")
                continue

            control_by_vesicle = _project_points_by_vesicle(
                sub_c,
                ("query_point_x_nm", "query_point_y_nm", "query_point_z_nm"),
                post_tree,
                post_surface,
            )
            control_post = np.vstack(list(control_by_vesicle.values()))

            fusion_curves, ctrl_curves, paired_ids = _paired_vesicle_o_curves(
                fusion_by_vesicle,
                control_by_vesicle,
                aunp_post,
                mesh_verts,
                mesh_faces,
                r_vals,
                geodesic_method=geodesic_method,
            )
            o_ctrl_lo, o_ctrl_med, o_ctrl_hi = _replicate_percentile_band(ctrl_curves)
            o_ctrl_mean = _replicate_mean(ctrl_curves)
            p_fusion_vs_ctrl = _vesicle_paired_pvalues(fusion_curves, ctrl_curves)
            fvc_by_offset[float(offset_nm)] = (fusion_curves, ctrl_curves)

            p_by_d[f"d={int(offset_nm)} nm"] = p_fusion_vs_ctrl
            min_p_floor = _wilcoxon_min_achievable_p(len(paired_ids))
            panel_notes[f"d={int(offset_nm)} nm"] = (
                f"n_paired={len(paired_ids)}"
                + (
                    f"\nWilcoxon floor={min_p_floor:.3g}"
                    if np.isfinite(min_p_floor) and min_p_floor > 0.05
                    else ""
                )
            )

            _plot_fusion_vs_control_ripley_panel(
                ax,
                r_vals,
                ctrl_lo=o_ctrl_lo,
                ctrl_med=o_ctrl_med,
                ctrl_hi=o_ctrl_hi,
                n_control_vesicles=len(control_by_vesicle),
                offset_nm=float(offset_nm),
                fusion_lo=o_fusion_lo,
                fusion_med=o_fusion_med,
                fusion_hi=o_fusion_hi,
                n_fusion_vesicles=len(fusion_by_vesicle),
                ylabel="Ripley's O(r)",
                refline=1.0,
                refline_label="CSR (O=1)",
            )
            _plot_fusion_vs_control_ripley_panel(
                ax_mean,
                r_vals,
                ctrl_lo=o_ctrl_lo,
                ctrl_med=o_ctrl_med,
                ctrl_hi=o_ctrl_hi,
                n_control_vesicles=len(control_by_vesicle),
                offset_nm=float(offset_nm),
                fusion_lo=o_fusion_lo,
                fusion_med=o_fusion_med,
                fusion_hi=o_fusion_hi,
                n_fusion_vesicles=len(fusion_by_vesicle),
                ylabel="Ripley's O(r)",
                refline=1.0,
                refline_label="CSR (O=1)",
                fusion_mean_only=True,
                fusion_mean=o_fusion_mean,
                ctrl_mean=o_ctrl_mean,
            )

            for r, f_lo, f_med, f_hi, f_mean, c_lo, c_med, c_hi, p_val in zip(
                r_vals,
                o_fusion_lo,
                o_fusion_med,
                o_fusion_hi,
                o_fusion_mean,
                o_ctrl_lo,
                o_ctrl_med,
                o_ctrl_hi,
                p_fusion_vs_ctrl,
            ):
                all_result_rows.append(
                    {
                        "zone_name": zone_name,
                        "analysis": "fusion_vs_control",
                        "uncertainty_method": UNCERTAINTY_METHOD_PERCENTILE,
                        "control_offset_nm": float(offset_nm),
                        "r_nm": float(r),
                        "o_fusion_lo": float(f_lo),
                        "o_fusion_median": float(f_med),
                        "o_fusion_hi": float(f_hi),
                        "o_control_lo": float(c_lo),
                        "o_control_median": float(c_med),
                        "o_control_hi": float(c_hi),
                        "o_fusion_mean": float(f_mean),
                        "o_fusion_sem": np.nan,
                        "o_control_mean": float(c_med),
                        "o_control_sem": np.nan,
                        "p_value_two_sided": float(p_val),
                        "n_fusion": len(fusion_post),
                        "n_fusion_vesicles": len(fusion_by_vesicle),
                        "n_control": len(control_post),
                        "n_control_vesicles": len(control_by_vesicle),
                        "n_paired_vesicles": len(paired_ids),
                        "n_aunp": len(aunp_post),
                        "mesh_vertices": len(mesh_verts),
                        "mesh_faces": len(mesh_faces),
                        "geodesic_method": geodesic_method,
                    }
                )

        for ax in axes_flat[n_panels:]:
            ax.set_visible(False)
        fig.suptitle(
            f"Ripley's O (membrain-stats geodesic): fusion vs controls "
            f"(2.5–97.5% bands + median across per-vesicle O curves)\n{zone_name}",
            y=1.02,
        )
        fig.tight_layout()
        fig.savefig(
            figures_dir / f"ripley_o_fusion_vs_controls_by_d_{zone_name}.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)

        for ax_mean in axes_mean_flat[n_panels:]:
            ax_mean.set_visible(False)
        fig_mean.suptitle(
            f"Ripley's O (membrain-stats geodesic): fusion vs controls "
            f"(fusion and control means only)\n{zone_name}",
            y=1.02,
        )
        fig_mean.tight_layout()
        fig_mean.savefig(
            figures_dir / f"ripley_o_fusion_vs_controls_by_d_{zone_name}_fusion_mean.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig_mean)

        if p_by_d:
            n_paired_global = len(fusion_by_vesicle)
            floor_note = ""
            floor_p = _wilcoxon_min_achievable_p(n_paired_global)
            if np.isfinite(floor_p) and floor_p > 0.05:
                floor_note = f" | Wilcoxon floor p={floor_p:.3g} (n={n_paired_global})"
            _plot_significance_panels(
                r_vals,
                p_by_d,
                title=(
                    f"Fusion vs controls Ripley's O significance (paired Wilcoxon on per-vesicle O)\n"
                    f"{zone_name} | red p<0.05, orange p<0.10{floor_note}"
                ),
                output_path=figures_dir / f"ripley_o_pvalues_fusion_vs_control_{zone_name}.png",
                panel_notes=panel_notes,
            )

        _save_ripley_o_vesicle_artifacts(
            output_dir / RIPLEY_O_VESICLE_CURVES_NPZ,
            r_vals=r_vals,
            zone_name=str(zone_name),
            tomogram_name=tomogram_path.name,
            fusion_vesicle_curves=fusion_vesicle_curves,
            label_perm_null_curves=perm_curves,
            o_obs=o_obs,
            fusion_vs_control_by_offset=fvc_by_offset,
        )

        print(
            f"  Ripley's O: {zone_name} — {len(fusion_post)} fusion, {len(aunp_post)} AuNPs "
            f"(mesh {len(mesh_verts)} verts / {len(mesh_faces)} faces, geodesic={geodesic_method})"
        )

    if not all_result_rows:
        return None
    out_df = pd.DataFrame(all_result_rows)
    out_df.to_csv(output_dir / "ripley_o_membrain_postsynaptic.csv", index=False)
    return out_df


def plot_results(
    df: pd.DataFrame,
    output_dir: Path,
    *,
    tomogram_path: Path | None = None,
    alignment_dir: str = "best_alignment",
) -> None:
    """Quick diagnostic figures."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if df.empty:
        print("No rows to plot.")
        return

    real = df[df["point_type"] == "fusion"].copy()
    ctrl = df[df["point_type"] == "control"].copy()

    probe_radii = sorted(df["probe_radius_nm"].unique())
    n_panels = len(probe_radii)
    ncols = min(3, n_panels)
    nrows = int(np.ceil(n_panels / ncols))
    mid_radius = probe_radii[len(probe_radii) // 2]

    # 1) Paired delta (real - control mean) per vesicle vs offset, one probe radius panel
    n_panels = len(probe_radii)
    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4), sharey=True)
    if n_panels == 1:
        axes = [axes]
    for ax, probe_radius in zip(axes, probe_radii):
        deltas = []
        offsets = []
        for vesicle_id in real["vesicle_id"].unique():
            r_row = real[
                (real["vesicle_id"] == vesicle_id) & (real["probe_radius_nm"] == probe_radius)
            ]
            if r_row.empty:
                continue
            real_val = float(r_row["packing_coefficient"].iloc[0])
            c_sub = ctrl[
                (ctrl["vesicle_id"] == vesicle_id) & (ctrl["probe_radius_nm"] == probe_radius)
            ]
            for offset_nm, grp in c_sub.groupby("control_offset_nm"):
                offsets.append(offset_nm)
                deltas.append(real_val - float(grp["packing_coefficient"].mean()))
        if deltas:
            delta_df = pd.DataFrame({"offset": offsets, "delta": deltas})
            summary = delta_df.groupby("offset")["delta"].agg(mean="mean", std="std", count="count")
            summary["sem"] = summary["std"] / np.sqrt(summary["count"].clip(lower=1))
            ax.errorbar(
                summary.index,
                summary["mean"],
                yerr=summary["sem"],
                marker="o",
                capsize=3,
                color="C3",
            )
            ax.axhline(0.0, color="k", linewidth=0.8, alpha=0.5)
        ax.set_title(f"probe r={int(probe_radius)} nm")
        ax.set_xlabel("Control offset d (nm)")
    axes[0].set_ylabel("Δ packing (fusion − control mean)")
    fig.suptitle("Paired fusion minus control packing", y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "delta_packing_vs_offset.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2) AuNP density per nm² vs offset — faceted like plot #1
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4 * nrows), squeeze=False)
    axes_flat = axes.flatten()
    for ax, probe_radius in zip(axes_flat, probe_radii):
        sub_ctrl = ctrl[ctrl["probe_radius_nm"] == probe_radius]
        summary = (
            sub_ctrl.groupby("control_offset_nm")["aunp_density_per_nm2"]
            .agg(mean="mean", std="std", count="count")
            .reset_index()
        )
        summary["sem"] = summary["std"] / np.sqrt(summary["count"].clip(lower=1))
        ax.errorbar(
            summary["control_offset_nm"],
            summary["mean"],
            yerr=summary["sem"],
            marker="s",
            markersize=7,
            capsize=5,
            capthick=1.5,
            elinewidth=1.5,
            color="C0",
            label="control mean ± SEM",
        )
        fusion_vals = real.loc[real["probe_radius_nm"] == probe_radius, "aunp_density_per_nm2"]
        fusion_mean = float(fusion_vals.mean())
        fusion_sem = (
            float(fusion_vals.std() / np.sqrt(len(fusion_vals)))
            if len(fusion_vals) > 1
            else 0.0
        )
        ax.axhline(fusion_mean, color="C3", linestyle="--", linewidth=1.5, label="fusion mean")
        if fusion_sem > 0:
            ax.axhspan(fusion_mean - fusion_sem, fusion_mean + fusion_sem, color="C3", alpha=0.2)
        ax.set_title(f"probe r = {int(probe_radius)} nm")
        ax.set_xlabel("Control offset d (nm)")
        ax.set_ylabel("AuNP density (per nm²)")
        ax.legend(fontsize=7, loc="best")
    for ax in axes_flat[n_panels:]:
        ax.set_visible(False)
    fig.suptitle("AuNP density at controls vs offset (mean ± SEM at each d)", y=1.02, fontsize=11)
    fig.tight_layout()
    fig.savefig(output_dir / "aunp_density_vs_control_offset.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    if not ctrl.empty and tomogram_path is not None:
        _plot_fusion_vs_control_zonogram(
            real,
            ctrl,
            tomogram_path=tomogram_path,
            alignment_dir=alignment_dir,
            probe_radius_nm=mid_radius,
            output_path=output_dir / "fusion_vs_control_zonogram.png",
        )

    print(f"Saved figures to {output_dir}")


def presynaptic_membrane_name_for_zone(zone_name: str, zone_data: dict | None = None) -> str:
    """Map active zone name to presynaptic membrane key used by vesicle results."""
    if zone_data and zone_data.get("presynaptic_membrane_index") is not None:
        pre_idx = int(zone_data["presynaptic_membrane_index"])
    else:
        # active_zone_pre1_post1 -> 1
        pre_part = zone_name.split("_")[2]
        pre_idx = int(pre_part.replace("pre", ""))
    return f"presynapticmembranes_{pre_idx}"


def load_presynaptic_az_points_for_zone(
    tomogram_path: Path,
    alignment_dir: str,
    zone_name: str,
) -> np.ndarray:
    """Presynaptic outer + inner active-zone points for one zone."""
    az_dir = tomogram_path / alignment_dir / "STT_results" / "activezone"
    parts: list[np.ndarray] = []
    for suffix in ("pre_outer", "pre_inner"):
        path = az_dir / f"{zone_name}_{suffix}.txt"
        if path.is_file():
            surf = np.atleast_2d(np.loadtxt(path, delimiter=None))
            if surf.size:
                parts.append(surf.astype(float))
    if not parts:
        return np.zeros((0, 3), dtype=float)
    return np.vstack(parts)


def membrane_az_pairs_for_zone(
    tomogram_path: Path,
    alignment_dir: str,
    zone_name: str,
    zone_data: dict | None,
    membrane_az_pairs: dict,
) -> dict:
    """Presynaptic membrane entry with zone-specific active-zone points for controls."""
    membrane_name = presynaptic_membrane_name_for_zone(zone_name, zone_data)
    if membrane_name not in membrane_az_pairs:
        return {}
    az_xyz = load_presynaptic_az_points_for_zone(tomogram_path, alignment_dir, zone_name)
    if len(az_xyz) == 0:
        return {}
    base = membrane_az_pairs[membrane_name]
    return {
        membrane_name: {
            **base,
            "active_zone_points": az_xyz,
        }
    }


def filter_fusion_rows_for_zone(
    fusion_rows: Sequence[dict],
    membrane_name: str,
) -> list[dict]:
    return [row for row in fusion_rows if row.get("closest_membrane") == membrane_name]


def default_probe_radius_for_coords(probe_radii: Sequence[float] | None = None) -> float:
    """Middle probe radius for Ripley coordinate filtering."""
    radii = sorted(probe_radii or PACKING_DENSITY_PROBE_RADII_NM)
    return float(radii[len(radii) // 2])


def load_scan_by_radius_for_tomogram(
    tomogram_path: Path,
    alignment_dir: str,
    *,
    probe_radii: Sequence[float] = PACKING_DENSITY_PROBE_RADII_NM,
    cache_dir: Path | None = None,
    vertex_sampling_step: int = 50,
    receptor_crosssection: float = 122.0,
    aunps_per_receptor: float = 2.0,
) -> dict[float, pd.DataFrame]:
    """Load or compute scan-vertex tables at each probe radius (original script behaviour)."""
    tomogram_path = Path(tomogram_path)
    cache_dir = cache_dir or fusion_point_vs_aunp_density_output_dir(
        tomogram_path, alignment_dir, "_shared_scans"
    )
    cache_dir.mkdir(parents=True, exist_ok=True)

    scan_by_radius: dict[float, pd.DataFrame] = {}
    for probe_radius in probe_radii:
        tag = packing_density_radius_tag(float(probe_radius))
        cache_path = cache_dir / f"scan_vertices_{tag}.csv"
        if cache_path.is_file():
            scan_by_radius[float(probe_radius)] = pd.read_csv(cache_path)
            continue
        scan_df = load_or_compute_scan_df(
            tomogram_path,
            alignment_dir,
            float(probe_radius),
            vertex_sampling_step=vertex_sampling_step,
            receptor_crosssection=receptor_crosssection,
            aunps_per_receptor=aunps_per_receptor,
        )
        scan_df.to_csv(cache_path, index=False)
        scan_by_radius[float(probe_radius)] = scan_df
    return scan_by_radius


def scan_by_radius_for_zone(
    scan_by_radius: dict[float, pd.DataFrame],
    zone_name: str,
) -> dict[float, pd.DataFrame]:
    """Restrict tomogram-wide scan tables to one active zone."""
    zone_scans: dict[float, pd.DataFrame] = {}
    for radius, scan_df in scan_by_radius.items():
        if scan_df.empty or "active_zone_name" not in scan_df.columns:
            continue
        subset = scan_df[scan_df["active_zone_name"] == zone_name]
        if not subset.empty:
            zone_scans[float(radius)] = subset
    return zone_scans


def fusion_point_vs_aunp_density_output_dir(
    tomogram_path: Path,
    alignment_dir: str,
    zone_name: str,
) -> Path:
    """Per-tomogram active-zonogram directory for one zone's fusion-point analysis."""
    return (
        tomogram_path
        / alignment_dir
        / "STT_results"
        / "visualizations"
        / "active_zonograms"
        / FUSION_POINT_VS_AUNP_DENSITY_SUBDIR
        / zone_name
    )


def run_fusion_point_vs_aunp_density_for_zone(
    tomogram_path: Path,
    alignment_dir: str,
    zone_name: str,
    *,
    scan_by_radius: dict[float, pd.DataFrame],
    fusion_rows: Sequence[dict],
    membrane_az_pairs: dict,
    zone_data: dict | None = None,
    output_dir: Path,
    offset_distances_nm: Sequence[float] = DEFAULT_OFFSET_DISTANCES_NM,
    n_directions: int = DEFAULT_N_DIRECTIONS,
    seed: int = DEFAULT_ANALYSIS_SEED,
    max_snap_distance_nm: float = 10.0,
    write_figures: bool = False,
    skip_ripley: bool = False,
    skip_ripley_o: bool = False,
) -> pd.DataFrame | None:
    """Run tangential-shuffle control analysis for one active zone."""
    if not scan_by_radius:
        return None

    membrane_name = presynaptic_membrane_name_for_zone(zone_name, zone_data)
    zone_fusion_rows = filter_fusion_rows_for_zone(fusion_rows, membrane_name)
    if not zone_fusion_rows:
        print(f"  No fusion points for {zone_name} ({membrane_name}), skipping fusion-point analysis")
        return None

    zone_membrane_pairs = membrane_az_pairs_for_zone(
        tomogram_path, alignment_dir, zone_name, zone_data, membrane_az_pairs
    )
    if not zone_membrane_pairs:
        print(f"  No presynaptic AZ points for {zone_name}, skipping fusion-point analysis")
        return None

    df = build_control_table(
        zone_fusion_rows,
        zone_membrane_pairs,
        scan_by_radius,
        tuple(float(d) for d in offset_distances_nm),
        int(n_directions),
        int(seed),
        max_snap_distance_nm=max_snap_distance_nm,
    )
    if df.empty:
        return None

    df = df.copy()
    df["active_zone_name"] = zone_name
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "fusion_point_vs_aunp_density.csv", index=False)

    meta = {
        "tomogram_name": tomogram_path.name,
        "alignment_dir": alignment_dir,
        "active_zone_name": zone_name,
        "probe_radii_nm": sorted(scan_by_radius),
        "offset_distances_nm": list(offset_distances_nm),
        "n_directions": int(n_directions),
        "max_snap_distance_nm": float(max_snap_distance_nm),
        "seed": int(seed),
        "n_fusion_vesicles": len(zone_fusion_rows),
        "n_table_rows": len(df),
    }
    with open(output_dir / "run_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    if write_figures:
        plot_results(
            df,
            output_dir / "figures",
            tomogram_path=tomogram_path,
            alignment_dir=alignment_dir,
        )

    probe_for_coords = default_probe_radius_for_coords(sorted(scan_by_radius))
    if not skip_ripley:
        run_ripley_postsynaptic_analysis(
            df,
            tomogram_path,
            alignment_dir,
            output_dir,
            seed=seed,
            probe_radius_for_coords=probe_for_coords,
        )
    if not skip_ripley_o:
        run_ripley_o_membrain_postsynaptic_analysis(
            df,
            tomogram_path,
            alignment_dir,
            output_dir,
            seed=seed,
            probe_radius_for_coords=probe_for_coords,
        )

    print(
        f"  Fusion-point vs AuNP density ({zone_name}): {len(df)} rows "
        f"({(df['point_type'] == 'fusion').sum()} fusion) -> {output_dir}"
    )
    return df


def run_fusion_point_vs_aunp_density_for_tomogram(
    tomogram_path: Path,
    alignment_dir: str,
    *,
    fusion_rows: Sequence[dict],
    active_zones_glb: dict,
    probe_radii: Sequence[float] = PACKING_DENSITY_PROBE_RADII_NM,
    receptor_crosssection: float = 122.0,
    aunps_per_receptor: float = 2.0,
    vertex_sampling_step: int = 50,
    offset_distances_nm: Sequence[float] = DEFAULT_OFFSET_DISTANCES_NM,
    n_directions: int = DEFAULT_N_DIRECTIONS,
    seed: int = DEFAULT_ANALYSIS_SEED,
    max_snap_distance_nm: float = 10.0,
    write_figures: bool = True,
) -> list[pd.DataFrame]:
    """Run fusion-point vs AuNP density analysis for each active zone in one tomogram."""
    tomogram_path = Path(tomogram_path)
    if not fusion_rows:
        return []

    print(
        f"  Fusion-point analysis probe radii: "
        f"{', '.join(str(int(r)) for r in probe_radii)} nm"
    )
    shared_cache = (
        tomogram_path
        / alignment_dir
        / "STT_results"
        / "visualizations"
        / "active_zonograms"
        / FUSION_POINT_VS_AUNP_DENSITY_SUBDIR
        / "_scan_cache"
    )
    scan_by_radius = load_scan_by_radius_for_tomogram(
        tomogram_path,
        alignment_dir,
        probe_radii=probe_radii,
        cache_dir=shared_cache,
        vertex_sampling_step=vertex_sampling_step,
        receptor_crosssection=receptor_crosssection,
        aunps_per_receptor=aunps_per_receptor,
    )

    membrane_az_pairs = import_presynaptic_membranes_and_active_zones(
        tomogram_path, alignment_dir=alignment_dir
    )
    zone_frames: list[pd.DataFrame] = []
    for zone_name, zone_data in active_zones_glb.get("active_zones", {}).items():
        zone_scans = scan_by_radius_for_zone(scan_by_radius, zone_name)
        if not zone_scans:
            continue
        out_dir = fusion_point_vs_aunp_density_output_dir(tomogram_path, alignment_dir, zone_name)
        df_zone = run_fusion_point_vs_aunp_density_for_zone(
            tomogram_path,
            alignment_dir,
            zone_name,
            scan_by_radius=zone_scans,
            fusion_rows=fusion_rows,
            membrane_az_pairs=membrane_az_pairs,
            zone_data=zone_data,
            output_dir=out_dir,
            offset_distances_nm=offset_distances_nm,
            n_directions=n_directions,
            seed=seed,
            max_snap_distance_nm=max_snap_distance_nm,
            write_figures=write_figures,
        )
        if df_zone is not None and not df_zone.empty:
            zone_frames.append(df_zone)
    return zone_frames


def _read_optional_csv(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    df = pd.read_csv(path)
    return df if not df.empty else None


def collect_per_tomogram_fusion_point_vs_aunp_density_tables(
    tomo_paths: Iterable[tuple[Any, Any, Any, str]],
) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
    """Load per-zone fusion-point and Ripley CSVs from all tomograms."""
    fusion_frames: list[pd.DataFrame] = []
    h12_frames: list[pd.DataFrame] = []
    o_frames: list[pd.DataFrame] = []

    for tomo, _set_name, _aunp_active_zones, alignment_dir in tomo_paths:
        tomogram_path = Path(tomo)
        base = (
            tomogram_path
            / alignment_dir
            / "STT_results"
            / "visualizations"
            / "active_zonograms"
            / FUSION_POINT_VS_AUNP_DENSITY_SUBDIR
        )
        if not base.is_dir():
            continue
        for zone_dir in sorted(p for p in base.iterdir() if p.is_dir()):
            fusion_csv = zone_dir / "fusion_point_vs_aunp_density.csv"
            df_f = _read_optional_csv(fusion_csv)
            if df_f is not None:
                fusion_frames.append(df_f)
            df_h12 = _read_optional_csv(zone_dir / "ripley_h12_postsynaptic.csv")
            if df_h12 is not None:
                h12_frames.append(df_h12)
            df_o = _read_optional_csv(zone_dir / "ripley_o_membrain_postsynaptic.csv")
            if df_o is not None:
                o_frames.append(df_o)

    fusion_combined = pd.concat(fusion_frames, ignore_index=True) if fusion_frames else pd.DataFrame()
    h12_combined = pd.concat(h12_frames, ignore_index=True) if h12_frames else None
    o_combined = pd.concat(o_frames, ignore_index=True) if o_frames else None
    return fusion_combined, h12_combined, o_combined


def _offset_artifact_key(offset_nm: float) -> str:
    return f"offset_{int(round(offset_nm))}nm"


def _save_ripley_h12_vesicle_artifacts(
    path: Path,
    *,
    r_vals: np.ndarray,
    zone_name: str,
    tomogram_name: str,
    fusion_vesicle_curves: np.ndarray,
    label_perm_null_curves: np.ndarray,
    h12_obs: np.ndarray,
    fusion_vs_control_by_offset: dict[float, tuple[np.ndarray, np.ndarray]],
) -> None:
    payload: dict[str, Any] = {
        "r_vals": r_vals,
        "zone_name": np.array(zone_name),
        "tomogram_name": np.array(tomogram_name),
        "fusion_vesicle_curves": fusion_vesicle_curves,
        "label_perm_null_curves": label_perm_null_curves,
        "h12_obs": h12_obs,
    }
    for offset_nm, (fusion_curves, ctrl_curves) in fusion_vs_control_by_offset.items():
        tag = _offset_artifact_key(offset_nm)
        payload[f"fusion_{tag}"] = fusion_curves
        payload[f"control_{tag}"] = ctrl_curves
    np.savez_compressed(path, **payload)


def _save_ripley_o_vesicle_artifacts(
    path: Path,
    *,
    r_vals: np.ndarray,
    zone_name: str,
    tomogram_name: str,
    fusion_vesicle_curves: np.ndarray,
    label_perm_null_curves: np.ndarray,
    o_obs: np.ndarray,
    fusion_vs_control_by_offset: dict[float, tuple[np.ndarray, np.ndarray]],
) -> None:
    payload: dict[str, Any] = {
        "r_vals": r_vals,
        "zone_name": np.array(zone_name),
        "tomogram_name": np.array(tomogram_name),
        "fusion_vesicle_curves": fusion_vesicle_curves,
        "label_perm_null_curves": label_perm_null_curves,
        "o_obs": o_obs,
    }
    for offset_nm, (fusion_curves, ctrl_curves) in fusion_vs_control_by_offset.items():
        tag = _offset_artifact_key(offset_nm)
        payload[f"fusion_{tag}"] = fusion_curves
        payload[f"control_{tag}"] = ctrl_curves
    np.savez_compressed(path, **payload)


def _load_ripley_vesicle_artifact(path: Path) -> dict[str, np.ndarray | str]:
    data = np.load(path, allow_pickle=True)
    out: dict[str, np.ndarray | str] = {}
    for key in data.files:
        val = data[key]
        if key in {"zone_name", "tomogram_name"}:
            out[key] = str(val.item()) if val.ndim == 0 else str(val)
        else:
            out[key] = np.asarray(val)
    return out


def _collect_ripley_vesicle_artifacts_by_zone(
    tomo_paths: Iterable[tuple[Any, Any, Any, str]],
    artifact_name: str,
) -> dict[str, list[dict[str, np.ndarray | str]]]:
    """Load saved per-vesicle Ripley artifacts grouped by active zone name."""
    by_zone: dict[str, list[dict[str, np.ndarray | str]]] = {}
    for tomo, _set_name, _aunp_active_zones, alignment_dir in tomo_paths:
        base = (
            Path(tomo)
            / alignment_dir
            / "STT_results"
            / "visualizations"
            / "active_zonograms"
            / FUSION_POINT_VS_AUNP_DENSITY_SUBDIR
        )
        if not base.is_dir():
            continue
        for zone_dir in sorted(p for p in base.iterdir() if p.is_dir() and not p.name.startswith("_")):
            artifact_path = zone_dir / artifact_name
            if not artifact_path.is_file():
                continue
            artifact = _load_ripley_vesicle_artifact(artifact_path)
            zone_name = str(artifact.get("zone_name", zone_dir.name))
            by_zone.setdefault(zone_name, []).append(artifact)
    return by_zone


def _stack_nonempty_curves(parts: list[np.ndarray]) -> np.ndarray:
    valid = [p for p in parts if p.ndim == 2 and len(p) > 0]
    if not valid:
        return np.empty((0, 0))
    return np.vstack(valid)


def _offset_keys_from_artifact(artifact: dict[str, np.ndarray | str]) -> list[float]:
    offsets: list[float] = []
    for key in artifact:
        if not isinstance(key, str) or not key.startswith("fusion_offset_"):
            continue
        tag = key.removeprefix("fusion_")
        nm = tag.removeprefix("offset_").removesuffix("nm")
        offsets.append(float(nm))
    return sorted(offsets)


def _unpaired_curve_pvalues(fusion_curves: np.ndarray, ctrl_curves: np.ndarray) -> np.ndarray:
    """Mann–Whitney p-value per radius comparing pooled per-vesicle curves."""
    from scipy import stats

    if fusion_curves.ndim != 2 or ctrl_curves.ndim != 2:
        return np.array([])
    p_vals = np.empty(fusion_curves.shape[1], dtype=float)
    for k in range(fusion_curves.shape[1]):
        f_sample = fusion_curves[:, k]
        c_sample = ctrl_curves[:, k]
        if len(f_sample) < 2 or len(c_sample) < 2:
            p_vals[k] = 1.0
            continue
        try:
            p_vals[k] = float(stats.mannwhitneyu(f_sample, c_sample, alternative="two-sided").pvalue)
        except ValueError:
            p_vals[k] = 1.0
    return p_vals


def plot_pooled_ripley_h12_from_vesicle_artifacts(
    tomo_paths: Iterable[tuple[Any, Any, Any, str]],
    output_dir: Path,
    *,
    file_tag: str = "pooled",
) -> pd.DataFrame | None:
    """Combine saved per-vesicle H₁₂ curves across tomograms (no Ripley recomputation)."""
    by_zone = _collect_ripley_vesicle_artifacts_by_zone(tomo_paths, RIPLEY_H12_VESICLE_CURVES_NPZ)
    if not by_zone:
        print("Skipping pooled Ripley H₁₂: no saved vesicle-curve artifacts found.")
        return None

    figures_dir = output_dir / "figures" / "pooled_ripley"
    figures_dir.mkdir(parents=True, exist_ok=True)
    all_result_rows: list[dict] = []

    for zone_name, artifacts in sorted(by_zone.items()):
        r_vals = np.asarray(artifacts[0]["r_vals"], dtype=float)
        fusion_parts = [_stack_nonempty_curves([np.asarray(a["fusion_vesicle_curves"])]) for a in artifacts]
        null_parts = [_stack_nonempty_curves([np.asarray(a["label_perm_null_curves"])]) for a in artifacts]
        obs_parts = [np.asarray(a["h12_obs"], dtype=float) for a in artifacts]

        fusion_vesicle_curves = _stack_nonempty_curves(fusion_parts)
        perm_curves = _stack_nonempty_curves(null_parts)
        if fusion_vesicle_curves.size == 0 or perm_curves.size == 0:
            print(f"Skipping pooled Ripley H₁₂ for {zone_name}: empty vesicle curves.")
            continue

        h12_obs = np.mean(np.vstack(obs_parts), axis=0)
        h12_fusion_lo, h12_fusion_med, h12_fusion_hi = _replicate_percentile_band(fusion_vesicle_curves)
        h12_null_lo, h12_null_med, h12_null_hi = _replicate_percentile_band(perm_curves)
        p_label_two = _monte_carlo_p_two_sided(h12_obs, perm_curves)
        n_null = perm_curves.shape[0]
        p_label_greater = (np.sum(perm_curves >= h12_obs, axis=0) + 1) / (n_null + 1)
        p_label_less = (np.sum(perm_curves <= h12_obs, axis=0) + 1) / (n_null + 1)

        fig, ax = plt.subplots(figsize=(7, 5))
        _plot_label_permutation_envelope_panel(
            ax,
            r_vals,
            null_lo=h12_null_lo,
            null_med=h12_null_med,
            null_hi=h12_null_hi,
            n_perm=n_null,
            obs_lo=h12_fusion_lo,
            obs_med=h12_fusion_med,
            obs_hi=h12_fusion_hi,
            n_obs_replicates=len(fusion_vesicle_curves),
            ylabel="Ripley H₁₂(r) = √(K₁₂/π) − r",
            title=(
                f"Pooled label-permutation null: fusion vs AuNP on postsynaptic AZ\n"
                f"{zone_name} | n_tomograms={len(artifacts)}, n_fusion_vesicles={len(fusion_vesicle_curves)}"
            ),
            refline=0.0,
            refline_label="H₁₂ = 0",
        )
        fig.tight_layout()
        fig.savefig(figures_dir / f"ripley_h12_label_permutation_{file_tag}_{zone_name}.png", dpi=150)
        plt.close(fig)

        _plot_significance_single(
            r_vals,
            p_label_two,
            title=(
                f"Pooled label-permutation significance (fusion vs AuNP)\n"
                f"{zone_name} | two-sided Monte Carlo p, n_null={n_null}"
            ),
            output_path=figures_dir / f"ripley_h12_pvalues_label_permutation_{file_tag}_{zone_name}.png",
            panel_note=(
                f"n_fusion_vesicles={len(fusion_vesicle_curves)}, n_tomograms={len(artifacts)}"
            ),
        )

        for r, obs, n_lo, n_med, n_hi, o_lo, o_med, o_hi, p2, pg, pl in zip(
            r_vals,
            h12_obs,
            h12_null_lo,
            h12_null_med,
            h12_null_hi,
            h12_fusion_lo,
            h12_fusion_med,
            h12_fusion_hi,
            p_label_two,
            p_label_greater,
            p_label_less,
        ):
            all_result_rows.append(
                {
                    "scope": file_tag,
                    "zone_name": zone_name,
                    "analysis": "label_permutation",
                    "uncertainty_method": UNCERTAINTY_METHOD_PERCENTILE,
                    "control_offset_nm": np.nan,
                    "r_nm": float(r),
                    "h12_observed": float(obs),
                    "h12_null_lo": float(n_lo),
                    "h12_null_hi": float(n_hi),
                    "h12_null_median": float(n_med),
                    "h12_obs_lo": float(o_lo),
                    "h12_obs_hi": float(o_hi),
                    "h12_obs_median": float(o_med),
                    "p_value_two_sided": float(p2),
                    "p_value_enrichment": float(pg),
                    "p_value_depletion": float(pl),
                    "n_fusion_vesicles": len(fusion_vesicle_curves),
                    "n_tomograms": len(artifacts),
                }
            )

        offsets = _offset_keys_from_artifact(artifacts[0])
        n_panels = len(offsets)
        ncols = min(3, n_panels)
        nrows = int(np.ceil(n_panels / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4 * nrows), squeeze=False)
        axes_flat = axes.flatten()
        p_by_d: dict[str, np.ndarray] = {}
        panel_notes: dict[str, str] = {}

        for ax, offset_nm in zip(axes_flat, offsets):
            tag = _offset_artifact_key(offset_nm)
            fusion_parts_d: list[np.ndarray] = []
            ctrl_parts_d: list[np.ndarray] = []
            for artifact in artifacts:
                f_key, c_key = f"fusion_{tag}", f"control_{tag}"
                if f_key in artifact and c_key in artifact:
                    fusion_parts_d.append(np.asarray(artifact[f_key]))
                    ctrl_parts_d.append(np.asarray(artifact[c_key]))
            fusion_curves = _stack_nonempty_curves(fusion_parts_d)
            ctrl_curves = _stack_nonempty_curves(ctrl_parts_d)
            if fusion_curves.size == 0 or ctrl_curves.size == 0:
                ax.set_title(f"d={int(offset_nm)} nm (no controls)")
                continue

            h12_ctrl_lo, h12_ctrl_med, h12_ctrl_hi = _replicate_percentile_band(ctrl_curves)
            p_fusion_vs_ctrl = _unpaired_curve_pvalues(fusion_curves, ctrl_curves)
            p_by_d[f"d={int(offset_nm)} nm"] = p_fusion_vs_ctrl
            panel_notes[f"d={int(offset_nm)} nm"] = (
                f"n_fusion_vesicles={len(fusion_curves)}, n_control_vesicles={len(ctrl_curves)}, "
                f"n_tomograms={len(artifacts)}"
            )

            _plot_fusion_vs_control_ripley_panel(
                ax,
                r_vals,
                ctrl_lo=h12_ctrl_lo,
                ctrl_med=h12_ctrl_med,
                ctrl_hi=h12_ctrl_hi,
                n_control_vesicles=len(ctrl_curves),
                offset_nm=float(offset_nm),
                fusion_lo=h12_fusion_lo,
                fusion_med=h12_fusion_med,
                fusion_hi=h12_fusion_hi,
                n_fusion_vesicles=len(fusion_curves),
                ylabel="H₁₂(r)",
                refline=0.0,
                refline_label="H₁₂ = 0",
            )

            for r, f_lo, f_med, f_hi, c_lo, c_med, c_hi, p_val in zip(
                r_vals,
                h12_fusion_lo,
                h12_fusion_med,
                h12_fusion_hi,
                h12_ctrl_lo,
                h12_ctrl_med,
                h12_ctrl_hi,
                p_fusion_vs_ctrl,
            ):
                all_result_rows.append(
                    {
                        "scope": file_tag,
                        "zone_name": zone_name,
                        "analysis": "fusion_vs_control",
                        "uncertainty_method": UNCERTAINTY_METHOD_PERCENTILE,
                        "control_offset_nm": float(offset_nm),
                        "r_nm": float(r),
                        "h12_fusion_lo": float(f_lo),
                        "h12_fusion_median": float(f_med),
                        "h12_fusion_hi": float(f_hi),
                        "h12_control_lo": float(c_lo),
                        "h12_control_median": float(c_med),
                        "h12_control_hi": float(c_hi),
                        "p_value_two_sided": float(p_val),
                        "n_fusion_vesicles": len(fusion_curves),
                        "n_control_vesicles": len(ctrl_curves),
                        "n_tomograms": len(artifacts),
                    }
                )

        for ax in axes_flat[n_panels:]:
            ax.set_visible(False)
        fig.suptitle(
            f"Pooled Ripley H₁₂: fusion vs controls across all tomograms\n{zone_name}",
            y=1.02,
        )
        fig.tight_layout()
        fig.savefig(
            figures_dir / f"ripley_h12_fusion_vs_controls_by_d_{file_tag}_{zone_name}.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)

        if p_by_d:
            _plot_significance_panels(
                r_vals,
                p_by_d,
                title=(
                    f"Pooled fusion vs controls significance (Mann–Whitney on per-vesicle H₁₂)\n"
                    f"{zone_name} | all tomograms combined"
                ),
                output_path=figures_dir / f"ripley_h12_pvalues_fusion_vs_control_{file_tag}_{zone_name}.png",
                panel_notes=panel_notes,
            )

        print(
            f"  Pooled Ripley H₁₂ ({zone_name}): {len(fusion_vesicle_curves)} fusion vesicles from "
            f"{len(artifacts)} tomograms"
        )

    if not all_result_rows:
        return None
    out_df = pd.DataFrame(all_result_rows)
    out_df.to_csv(output_dir / f"ripley_h12_postsynaptic_{file_tag}.csv", index=False)
    return out_df


def plot_pooled_ripley_o_from_vesicle_artifacts(
    tomo_paths: Iterable[tuple[Any, Any, Any, str]],
    output_dir: Path,
    *,
    file_tag: str = "pooled",
) -> pd.DataFrame | None:
    """Combine saved per-vesicle Ripley's O curves across tomograms (no Ripley recomputation)."""
    by_zone = _collect_ripley_vesicle_artifacts_by_zone(tomo_paths, RIPLEY_O_VESICLE_CURVES_NPZ)
    if not by_zone:
        print("Skipping pooled Ripley's O: no saved vesicle-curve artifacts found.")
        return None

    figures_dir = output_dir / "figures" / "pooled_ripley"
    figures_dir.mkdir(parents=True, exist_ok=True)
    all_result_rows: list[dict] = []

    for zone_name, artifacts in sorted(by_zone.items()):
        r_vals = np.asarray(artifacts[0]["r_vals"], dtype=float)
        fusion_parts = [_stack_nonempty_curves([np.asarray(a["fusion_vesicle_curves"])]) for a in artifacts]
        null_parts = [_stack_nonempty_curves([np.asarray(a["label_perm_null_curves"])]) for a in artifacts]
        obs_parts = [np.asarray(a["o_obs"], dtype=float) for a in artifacts]

        fusion_vesicle_curves = _stack_nonempty_curves(fusion_parts)
        perm_curves = _stack_nonempty_curves(null_parts)
        if fusion_vesicle_curves.size == 0 or perm_curves.size == 0:
            print(f"Skipping pooled Ripley's O for {zone_name}: empty vesicle curves.")
            continue

        o_obs = np.mean(np.vstack(obs_parts), axis=0)
        o_fusion_lo, o_fusion_med, o_fusion_hi = _replicate_percentile_band(fusion_vesicle_curves)
        o_null_lo, o_null_med, o_null_hi = _replicate_percentile_band(perm_curves)
        p_label_two = _monte_carlo_p_two_sided(o_obs, perm_curves)
        n_null = perm_curves.shape[0]
        p_label_greater = (np.sum(perm_curves >= o_obs, axis=0) + 1) / (n_null + 1)
        p_label_less = (np.sum(perm_curves <= o_obs, axis=0) + 1) / (n_null + 1)

        for r, obs, n_lo, n_med, n_hi, o_lo, o_med, o_hi, p2, pg, pl in zip(
            r_vals,
            o_obs,
            o_null_lo,
            o_null_med,
            o_null_hi,
            o_fusion_lo,
            o_fusion_med,
            o_fusion_hi,
            p_label_two,
            p_label_greater,
            p_label_less,
        ):
            all_result_rows.append(
                {
                    "scope": file_tag,
                    "zone_name": zone_name,
                    "analysis": "label_permutation",
                    "uncertainty_method": UNCERTAINTY_METHOD_PERCENTILE,
                    "control_offset_nm": np.nan,
                    "r_nm": float(r),
                    "o_observed": float(obs),
                    "o_null_lo": float(n_lo),
                    "o_null_hi": float(n_hi),
                    "o_null_median": float(n_med),
                    "o_obs_lo": float(o_lo),
                    "o_obs_hi": float(o_hi),
                    "o_obs_median": float(o_med),
                    "p_value_two_sided": float(p2),
                    "p_value_enrichment": float(pg),
                    "p_value_depletion": float(pl),
                    "n_fusion_vesicles": len(fusion_vesicle_curves),
                    "n_tomograms": len(artifacts),
                }
            )

        fig, ax = plt.subplots(figsize=(7, 5))
        _plot_label_permutation_envelope_panel(
            ax,
            r_vals,
            null_lo=o_null_lo,
            null_med=o_null_med,
            null_hi=o_null_hi,
            n_perm=n_null,
            obs_lo=o_fusion_lo,
            obs_med=o_fusion_med,
            obs_hi=o_fusion_hi,
            n_obs_replicates=len(fusion_vesicle_curves),
            ylabel="Ripley's O(r) [membrain-stats geodesic]",
            title=(
                f"Pooled label-permutation null: Ripley's O fusion vs AuNP\n"
                f"{zone_name} | n_tomograms={len(artifacts)}"
            ),
            refline=1.0,
            refline_label="CSR (O=1)",
        )
        fig.tight_layout()
        fig.savefig(figures_dir / f"ripley_o_label_permutation_{file_tag}_{zone_name}.png", dpi=150)
        plt.close(fig)

        _plot_significance_single(
            r_vals,
            p_label_two,
            title=f"Pooled Ripley's O label-permutation significance\n{zone_name}",
            output_path=figures_dir / f"ripley_o_pvalues_label_permutation_{file_tag}_{zone_name}.png",
            panel_note=f"n_tomograms={len(artifacts)}, n_null={n_null}",
        )

        offsets = _offset_keys_from_artifact(artifacts[0])
        n_panels = len(offsets)
        ncols = min(3, n_panels)
        nrows = int(np.ceil(n_panels / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4 * nrows), squeeze=False)
        axes_flat = axes.flatten()
        p_by_d: dict[str, np.ndarray] = {}

        for ax, offset_nm in zip(axes_flat, offsets):
            tag = _offset_artifact_key(offset_nm)
            fusion_parts_d: list[np.ndarray] = []
            ctrl_parts_d: list[np.ndarray] = []
            for artifact in artifacts:
                f_key, c_key = f"fusion_{tag}", f"control_{tag}"
                if f_key in artifact and c_key in artifact:
                    fusion_parts_d.append(np.asarray(artifact[f_key]))
                    ctrl_parts_d.append(np.asarray(artifact[c_key]))
            fusion_curves = _stack_nonempty_curves(fusion_parts_d)
            ctrl_curves = _stack_nonempty_curves(ctrl_parts_d)
            if fusion_curves.size == 0 or ctrl_curves.size == 0:
                ax.set_title(f"d={int(offset_nm)} nm (no controls)")
                continue

            o_ctrl_lo, o_ctrl_med, o_ctrl_hi = _replicate_percentile_band(ctrl_curves)
            p_fusion_vs_ctrl = _unpaired_curve_pvalues(fusion_curves, ctrl_curves)
            p_by_d[f"d={int(offset_nm)} nm"] = p_fusion_vs_ctrl

            _plot_fusion_vs_control_ripley_panel(
                ax,
                r_vals,
                ctrl_lo=o_ctrl_lo,
                ctrl_med=o_ctrl_med,
                ctrl_hi=o_ctrl_hi,
                n_control_vesicles=len(ctrl_curves),
                offset_nm=float(offset_nm),
                fusion_lo=o_fusion_lo,
                fusion_med=o_fusion_med,
                fusion_hi=o_fusion_hi,
                n_fusion_vesicles=len(fusion_curves),
                ylabel="Ripley's O(r)",
                refline=1.0,
                refline_label="CSR (O=1)",
            )

            for r, f_lo, f_med, f_hi, c_lo, c_med, c_hi, p_val in zip(
                r_vals,
                o_fusion_lo,
                o_fusion_med,
                o_fusion_hi,
                o_ctrl_lo,
                o_ctrl_med,
                o_ctrl_hi,
                p_fusion_vs_ctrl,
            ):
                all_result_rows.append(
                    {
                        "scope": file_tag,
                        "zone_name": zone_name,
                        "analysis": "fusion_vs_control",
                        "uncertainty_method": UNCERTAINTY_METHOD_PERCENTILE,
                        "control_offset_nm": float(offset_nm),
                        "r_nm": float(r),
                        "o_fusion_lo": float(f_lo),
                        "o_fusion_median": float(f_med),
                        "o_fusion_hi": float(f_hi),
                        "o_control_lo": float(c_lo),
                        "o_control_median": float(c_med),
                        "o_control_hi": float(c_hi),
                        "p_value_two_sided": float(p_val),
                        "n_fusion_vesicles": len(fusion_curves),
                        "n_control_vesicles": len(ctrl_curves),
                        "n_tomograms": len(artifacts),
                    }
                )

        for ax in axes_flat[n_panels:]:
            ax.set_visible(False)
        fig.suptitle(
            f"Pooled Ripley's O: fusion vs controls across all tomograms\n{zone_name}",
            y=1.02,
        )
        fig.tight_layout()
        fig.savefig(
            figures_dir / f"ripley_o_fusion_vs_controls_by_d_{file_tag}_{zone_name}.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)

        if p_by_d:
            _plot_significance_panels(
                r_vals,
                p_by_d,
                title=f"Pooled Ripley's O fusion vs controls significance\n{zone_name}",
                output_path=figures_dir / f"ripley_o_pvalues_fusion_vs_control_{file_tag}_{zone_name}.png",
            )

        print(
            f"  Pooled Ripley's O ({zone_name}): {len(fusion_vesicle_curves)} fusion vesicles from "
            f"{len(artifacts)} tomograms"
        )

    if not all_result_rows:
        return None
    out_df = pd.DataFrame(all_result_rows)
    out_df.to_csv(output_dir / f"ripley_o_membrain_postsynaptic_{file_tag}.csv", index=False)
    return out_df


def aggregate_fusion_point_vs_aunp_density_visualizations(
    tomo_paths: Iterable[tuple[Any, Any, Any, str]],
    *,
    results_dir: Path | str = COMBINED_RESULTS_DIR,
) -> None:
    """
    Combine per-tomogram fusion-point vs AuNP density outputs and regenerate summary plots.

    Per-tomogram Ripley CSVs are concatenated only (not recomputed). Dataset-level pooled
    Ripley figures stack saved per-vesicle curves from each tomogram (no Ripley recomputation).

    Called from the visualization pipeline after active zonograms exist (for zonogram overlays).
    """
    results_dir = Path(results_dir)
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    fusion_df, h12_df, o_df = collect_per_tomogram_fusion_point_vs_aunp_density_tables(tomo_paths)
    if fusion_df.empty:
        print("No fusion-point vs AuNP density tables found to aggregate.")
        return

    fusion_df.to_csv(results_dir / "fusion_point_vs_aunp_density_combined.csv", index=False)
    print(
        f"Combined fusion-point vs AuNP density table: "
        f"{len(fusion_df)} rows -> {results_dir / 'fusion_point_vs_aunp_density_combined.csv'}"
    )

    plot_results(
        fusion_df,
        figures_dir,
        tomogram_path=None,
        alignment_dir="",
    )

    if h12_df is not None:
        h12_df.to_csv(results_dir / "ripley_h12_postsynaptic_combined.csv", index=False)
    if o_df is not None:
        o_df.to_csv(results_dir / "ripley_o_membrain_postsynaptic_combined.csv", index=False)

    print("Combining saved per-vesicle Ripley curves across all tomograms...")
    plot_pooled_ripley_h12_from_vesicle_artifacts(tomo_paths, results_dir)
    plot_pooled_ripley_o_from_vesicle_artifacts(tomo_paths, results_dir)

    real = fusion_df[fusion_df["point_type"] == "fusion"].copy()
    ctrl = fusion_df[fusion_df["point_type"] == "control"].copy()
    if real.empty or ctrl.empty:
        return

    probe_radii = sorted(fusion_df["probe_radius_nm"].unique())
    mid_radius = float(probe_radii[len(probe_radii) // 2])

    for tomo, _set_name, _aunp_active_zones, alignment_dir in tomo_paths:
        tomogram_path = Path(tomo)
        tomogram_name = tomogram_path.name
        sub_real = real[real["tomogram_name"] == tomogram_name]
        sub_ctrl = ctrl[ctrl["tomogram_name"] == tomogram_name]
        if sub_real.empty or sub_ctrl.empty:
            continue
        try:
            _plot_fusion_vs_control_zonogram(
                sub_real,
                sub_ctrl,
                tomogram_path=tomogram_path,
                alignment_dir=alignment_dir,
                probe_radius_nm=mid_radius,
                output_path=figures_dir / f"fusion_vs_control_zonogram_{tomogram_name}.png",
            )
        except Exception as exc:
            print(f"  Warning: zonogram overlay failed for {tomogram_name}: {exc}")

    print(f"Combined fusion-point vs AuNP density figures -> {figures_dir}")


def main() -> None:
    import argparse

    repo_root = Path(__file__).resolve().parent.parent.parent
    default_tomo = repo_root / "data/15F1/TOP_TOMOS/20240111_WaffleHipp_116"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tomogram-path",
        type=Path,
        default=default_tomo,
        help="Tomogram directory",
    )
    parser.add_argument("--alignment-dir", default="best_alignment")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "results/fusion_point_vs_aunp_density/20240111_WaffleHipp_116",
    )
    parser.add_argument(
        "--probe-radii",
        type=float,
        nargs="+",
        default=list(PACKING_DENSITY_PROBE_RADII_NM),
        help="Sliding-cylinder probe radii (nm) for density lookup",
    )
    parser.add_argument(
        "--offset-distances",
        type=float,
        nargs="+",
        default=list(DEFAULT_OFFSET_DISTANCES_NM),
        help="Tangential shuffle distances d (nm)",
    )
    parser.add_argument(
        "--n-directions",
        type=int,
        default=100,
        help="Random tangent directions per (vesicle, d, probe radius)",
    )
    parser.add_argument(
        "--fusing-only",
        action="store_true",
        help="Include only vesicles classified as fusing (exclude close)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--vesicle-distance-threshold",
        type=float,
        default=20.0,
    )
    parser.add_argument("--fusion-point-threshold", type=float, default=20.0)
    parser.add_argument(
        "--max-snap-distance",
        type=float,
        default=10.0,
        help="Reject controls whose tangential candidate snaps > this distance (nm) from the AZ",
    )
    parser.add_argument(
        "--ripley-r-max",
        type=float,
        default=DEFAULT_RIPLEY_R_MAX_NM,
        help="Maximum radius r (nm) for Ripley H12 analysis on postsynaptic AZ",
    )
    parser.add_argument(
        "--ripley-r-step",
        type=float,
        default=DEFAULT_RIPLEY_R_STEP_NM,
        help="Step size (nm) for Ripley H12 radius grid",
    )
    parser.add_argument(
        "--ripley-n-perm",
        type=int,
        default=DEFAULT_RIPLEY_N_PERM,
        help="Label-permutation replicates for Ripley null envelope",
    )
    parser.add_argument(
        "--skip-ripley",
        action="store_true",
        help="Skip postsynaptic Ripley H12 analysis",
    )
    parser.add_argument(
        "--skip-ripley-o",
        action="store_true",
        help="Skip postsynaptic Ripley's O analysis (membrain-stats geodesic)",
    )
    parser.add_argument(
        "--ripley-o-mesh-max-verts",
        type=int,
        default=DEFAULT_RIPLEY_O_MESH_MAX_VERTS,
        help="Max vertices in coarse AZ mesh for membrain-stats geodesic Ripley's O",
    )
    parser.add_argument(
        "--ripley-o-geodesic-method",
        choices=("fast", "exact"),
        default=DEFAULT_RIPLEY_O_GEODESIC_METHOD,
        help="Geodesic solver for membrain-stats Ripley's O (fast=heat method, exact=pygeodesic)",
    )
    args = parser.parse_args()
    probe_radii = [float(r) for r in args.probe_radii]

    tomogram_path = args.tomogram_path.resolve()
    if not tomogram_path.is_dir():
        raise SystemExit(f"Tomogram not found: {tomogram_path}")

    print(f"Tomogram: {tomogram_path}")
    print(f"Probe radii: {probe_radii}")
    print(f"Control offsets d: {args.offset_distances}")
    print(f"Directions per (vesicle, d): {args.n_directions}")
    print(f"Max snap distance: {args.max_snap_distance} nm")

    fusion_rows = enumerate_close_vesicle_fusion_points(
        tomogram_path,
        alignment_dir=args.alignment_dir,
        vesicle_distance_threshold=args.vesicle_distance_threshold,
        fusion_point_threshold=args.fusion_point_threshold,
    )
    if args.fusing_only:
        fusion_rows = [
            r
            for r in fusion_rows
            if r.get("is_fusing") or r.get("vesicle_distance_class") == "fusing"
        ]
    print(f"Close/fusing vesicles with fusion points: {len(fusion_rows)}", end="")
    if args.fusing_only:
        print(" (fusing only)", end="")
    print()
    if not fusion_rows:
        raise SystemExit("No fusion points found.")

    membrane_az_pairs = import_presynaptic_membranes_and_active_zones(
        tomogram_path, alignment_dir=args.alignment_dir
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    scan_by_radius: dict[float, pd.DataFrame] = {}
    for probe_radius in probe_radii:
        cache_path = args.output_dir / f"scan_vertices_{packing_density_radius_tag(float(probe_radius))}.csv"
        if cache_path.exists():
            scan_by_radius[float(probe_radius)] = pd.read_csv(cache_path)
            print(f"  Loaded cached scan for r={probe_radius:.0f} nm: {len(scan_by_radius[float(probe_radius)])}")
        else:
            scan_by_radius[float(probe_radius)] = load_or_compute_scan_df(
                tomogram_path,
                args.alignment_dir,
                float(probe_radius),
            )
            scan_by_radius[float(probe_radius)].to_csv(cache_path, index=False)
            print(f"  Scan vertices for r={probe_radius:.0f} nm: {len(scan_by_radius[float(probe_radius)])}")

    df = build_control_table(
        fusion_rows,
        membrane_az_pairs,
        scan_by_radius,
        tuple(args.offset_distances),
        args.n_directions,
        args.seed,
        max_snap_distance_nm=args.max_snap_distance,
    )
    print(f"Built table: {len(df)} rows ({(df['point_type']=='fusion').sum()} fusion, "
          f"{(df['point_type']=='control').sum()} control)")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "fusion_point_vs_aunp_density.csv"
    df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}")

    # Compare with production fusion-only table for one radius if available
    for probe_radius in probe_radii:
        scan_df = scan_by_radius[float(probe_radius)]
        prod = build_packing_density_at_fusion_points_dataframe(
            scan_df,
            fusion_rows,
            cylinder_radius=probe_radius,
            receptor_crosssection=122.0,
            aunps_per_receptor=2.0,
            vertex_sampling_step=50,
        )
        if not prod.empty:
            prod_path = args.output_dir / f"production_fusion_lookup_r{int(round(probe_radius))}nm.csv"
            prod.to_csv(prod_path, index=False)

    meta = {
        "tomogram_path": str(tomogram_path),
        "alignment_dir": args.alignment_dir,
        "probe_radii_nm": probe_radii,
        "offset_distances_nm": args.offset_distances,
        "n_directions": args.n_directions,
        "max_snap_distance_nm": args.max_snap_distance,
        "ripley_r_max_nm": args.ripley_r_max,
        "ripley_r_step_nm": args.ripley_r_step,
        "ripley_n_perm": args.ripley_n_perm,
        "ripley_o_mesh_max_verts": args.ripley_o_mesh_max_verts,
        "ripley_o_geodesic_method": args.ripley_o_geodesic_method,
        "seed": args.seed,
        "n_fusion_vesicles": len(fusion_rows),
        "n_table_rows": len(df),
    }
    with open(args.output_dir / "run_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    plot_results(
        df,
        args.output_dir / "figures",
        tomogram_path=tomogram_path,
        alignment_dir=args.alignment_dir,
    )

    if not args.skip_ripley:
        run_ripley_postsynaptic_analysis(
            df,
            tomogram_path,
            args.alignment_dir,
            args.output_dir,
            r_max_nm=args.ripley_r_max,
            r_step_nm=args.ripley_r_step,
            n_perm=args.ripley_n_perm,
            seed=args.seed,
        )

    if not args.skip_ripley_o:
        run_ripley_o_membrain_postsynaptic_analysis(
            df,
            tomogram_path,
            args.alignment_dir,
            args.output_dir,
            r_max_nm=args.ripley_r_max,
            r_step_nm=args.ripley_r_step,
            n_perm=args.ripley_n_perm,
            mesh_max_vertices=args.ripley_o_mesh_max_verts,
            geodesic_method=args.ripley_o_geodesic_method,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
