"""
Fusion-point vs AuNP density analysis with presynaptic tangential-shuffle controls.

For each close/fusing vesicle fusion point, samples random tangential offsets at
several distances d on the presynaptic synaptic cleft, snaps to the AZ point cloud,
and looks up packing density from the same scan-vertex tables used in production.

Also runs standardized bivariate Ripley H12 on postsynaptic-projected fusion,
control, and AuNP positions (fusion vs controls per d; label-permutation null),
and geodesic Ripley's O via membrain-stats (same two analysis modes).
"""

from __future__ import annotations

import contextlib
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .alignment_utils import require_alignment_dir
from .aunps import (
    DEFAULT_AUNP_PICK_STAR_PATTERN,
    build_packing_density_at_fusion_points_dataframe,
    calculate_packing_density_using_sliding_cylinder,
    discover_aunp_pick_star_files,
    enumerate_close_vesicle_fusion_points,
    load_aunp_pick_star_dataframes,
    normalize_aunp_pick_star_pattern,
)
from .vesicles import import_presynaptic_membranes_and_clefts

# Probe radii (nm) for fusion-point control lookups and Ripley analysis (not used for main packing heat map).
PACKING_DENSITY_PROBE_RADII_NM = (10.0, 20.0, 30.0, 40.0, 50.0)


def packing_density_radius_tag(radius_nm: float) -> str:
    """Filename suffix for a probe radius, e.g. ``10.0`` -> ``r10nm``."""
    return f"r{int(round(radius_nm))}nm"


def _scan_cache_subdir_name(aunp_pick_star_pattern: str | None = None) -> str:
    """Cache folder name for scan-vertex tables; encodes non-default AuNP pick patterns."""
    pat = normalize_aunp_pick_star_pattern(aunp_pick_star_pattern)
    if pat == DEFAULT_AUNP_PICK_STAR_PATTERN:
        return "_scan_cache"
    slug = re.sub(r"[^\w.-]+", "_", pat.replace("*", "idx"))
    return f"_scan_cache_{slug}"


def _combined_aunp_pick_coordinates(
    tomogram_path: Path,
    alignment_dir: str,
    *,
    aunp_pick_star_pattern: str | None = None,
) -> np.ndarray:
    """Load and concatenate AuNP coordinates from per-zone pick STAR files."""
    aunps_dir = tomogram_path / alignment_dir / "aunps"
    pick_pattern = normalize_aunp_pick_star_pattern(aunp_pick_star_pattern)
    star_dfs = load_aunp_pick_star_dataframes(aunps_dir, pattern=pick_pattern)
    if not star_dfs:
        raise FileNotFoundError(
            f"No AuNP pick STAR files matching pattern {pick_pattern!r} in {aunps_dir}"
        )
    df = pd.concat(star_dfs, ignore_index=True)
    return df[["faCoordinateX", "faCoordinateY", "faCoordinateZ"]].to_numpy(dtype=float)


DEFAULT_OFFSET_DISTANCES_NM = (10.0, 20.0, 30.0, 40.0, 50.0)
FUSION_POINT_SHIFT_OFFSET_NM = 40.0
FUSION_POINT_AZ_MAX_SNAP_DISTANCE_NM = 5.0
DEFAULT_FUSION_POINT_NULL_REPLICATES_N = 10
DEFAULT_FUSION_POINT_LABEL_PERM_N = DEFAULT_FUSION_POINT_NULL_REPLICATES_N
DEFAULT_PROBE_RADII_NM = PACKING_DENSITY_PROBE_RADII_NM
DEFAULT_PROBE_RADIUS_NM = 25.0
DEFAULT_N_DIRECTIONS = 100
DEFAULT_ANALYSIS_SEED = 42
DEFAULT_RIPLEY_N_PERM = 499

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
    then snap to the nearest presynaptic synaptic-cleft point.

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
        "nearest_scan_cleft_name": row["cleft_name"],
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
    aunp_pick_star_pattern: str | None = None,
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
    from .cleft import (
        define_active_zonogram,
        find_active_zones_from_glb,
        import_membrane_segmentations_from_glb,
    )
    membrane_data = import_membrane_segmentations_from_glb(tomogram_path, alignment_dir=alignment_dir)
    clefts_glb = find_active_zones_from_glb(membrane_data, distance_range=(10.0, 40.0))
    zonogram_results = define_active_zonogram(clefts_glb)

    aunp_coords = _combined_aunp_pick_coordinates(
        tomogram_path,
        alignment_dir,
        aunp_pick_star_pattern=aunp_pick_star_pattern,
    )

    tomogram_name = tomogram_path.name
    rows: list[dict] = []
    for zone_name, zone_data in clefts_glb["clefts"].items():
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
                    "cleft_name": zone_name,
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
        az_xyz = membrane_az_pairs[membrane]["cleft_points"]
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


def zone_name_for_presynaptic_membrane(membrane_name: str | None) -> str | None:
    """Map ``presynapticmembranes_N`` key to ``cleft_preN_postN`` zone name."""
    if not membrane_name or not str(membrane_name).startswith("presynapticmembranes_"):
        return None
    try:
        idx = int(str(membrane_name).removeprefix("presynapticmembranes_"))
    except ValueError:
        return None
    return f"cleft_pre{idx}_post{idx}"


def compute_40nm_shifted_fusion_point_aunp_pairwise_distances(
    tomogram_path: Path,
    alignment_dir: str,
    aunp_coords: np.ndarray,
    *,
    vesicle_distance_threshold: float = 20.0,
    fusion_point_threshold: float = 20.0,
    offset_nm: float = FUSION_POINT_SHIFT_OFFSET_NM,
    n_shifts: int = DEFAULT_FUSION_POINT_NULL_REPLICATES_N,
    seed: int = DEFAULT_ANALYSIS_SEED,
    max_snap_distance_nm: float = FUSION_POINT_AZ_MAX_SNAP_DISTANCE_NM,
) -> pd.DataFrame:
    """
    Per-(vesicle, AuNP) distances using tangential AZ controls at ``offset_nm``.

    Draws ``n_shifts`` independent random tangent directions per fusing vesicle
    (same placement logic as ``build_control_table`` / ``sample_tangential_control_on_az``),
    retrying failed placements until all replicates succeed or a safety attempt cap.
    """
    tomogram_path = Path(tomogram_path)
    alignment_dir = require_alignment_dir(alignment_dir)
    aunp_coords = np.atleast_2d(np.asarray(aunp_coords, dtype=float))
    if aunp_coords.size == 0:
        return pd.DataFrame()

    fusion_rows = enumerate_close_vesicle_fusion_points(
        tomogram_path,
        alignment_dir=alignment_dir,
        vesicle_distance_threshold=vesicle_distance_threshold,
        fusion_point_threshold=fusion_point_threshold,
        fusing_only=True,
    )
    if not fusion_rows:
        return pd.DataFrame()

    membrane_az_pairs = import_presynaptic_membranes_and_clefts(
        tomogram_path, alignment_dir=alignment_dir
    )
    rng = np.random.default_rng(seed)
    long_rows: list[dict] = []

    for fp in fusion_rows:
        membrane = fp.get("closest_membrane")
        if not membrane or membrane not in membrane_az_pairs:
            continue
        az_xyz = membrane_az_pairs[membrane]["cleft_points"]
        if az_xyz is None or len(az_xyz) == 0:
            continue
        az_tree = cKDTree(az_xyz)
        fusion_xyz = np.array(
            [fp["fusion_point_x_nm"], fp["fusion_point_y_nm"], fp["fusion_point_z_nm"]],
            dtype=float,
        )
        zone_name = zone_name_for_presynaptic_membrane(membrane)
        n_shifts_target = int(n_shifts)
        shift_replicate_id = 0
        max_placement_attempts = n_shifts_target * 40
        placement_attempts = 0

        while shift_replicate_id < n_shifts_target and placement_attempts < max_placement_attempts:
            placement_attempts += 1
            shifted_xyz, _direction = sample_tangential_control_on_az(
                fusion_xyz,
                az_xyz,
                az_tree,
                float(offset_nm),
                rng,
                max_snap_distance_nm=max_snap_distance_nm,
            )
            if shifted_xyz is None:
                continue
            distances = np.linalg.norm(aunp_coords - shifted_xyz, axis=1)
            for j, dist in enumerate(distances):
                long_rows.append(
                    {
                        "tomogram_name": fp["tomogram_name"],
                        "alignment_dir": alignment_dir,
                        "cleft_name": zone_name,
                        "shift_replicate_id": int(shift_replicate_id),
                        "vesicle_id": fp["vesicle_id"],
                        "vesicle_name": fp["vesicle_name"],
                        "aunp_index": j,
                        "distance_to_presynaptic_az_nm": fp["distance_to_presynaptic_az_nm"],
                        "fusion_point_x_nm": float(fusion_xyz[0]),
                        "fusion_point_y_nm": float(fusion_xyz[1]),
                        "fusion_point_z_nm": float(fusion_xyz[2]),
                        "query_point_x_nm": float(shifted_xyz[0]),
                        "query_point_y_nm": float(shifted_xyz[1]),
                        "query_point_z_nm": float(shifted_xyz[2]),
                        "control_offset_nm": float(offset_nm),
                        "fusion_point_to_aunp_distance_nm": float(dist),
                    }
                )
            shift_replicate_id += 1

        if shift_replicate_id < n_shifts_target:
            print(
                f"  Warning: 40 nm shift — vesicle {fp['vesicle_id']} in {fp['tomogram_name']}: "
                f"only {shift_replicate_id}/{n_shifts_target} controls placed after "
                f"{placement_attempts} attempts."
            )
    return pd.DataFrame(long_rows)


def compute_label_permutation_fusion_point_aunp_pairwise_distances(
    tomogram_path: Path,
    alignment_dir: str,
    aunp_coords: np.ndarray,
    aunp_cleft_ids: np.ndarray,
    az_mapping: dict[int, str],
    *,
    vesicle_distance_threshold: float = 20.0,
    fusion_point_threshold: float = 20.0,
    n_perm: int = DEFAULT_FUSION_POINT_NULL_REPLICATES_N,
    seed: int = DEFAULT_ANALYSIS_SEED,
) -> pd.DataFrame:
    """
    Per-(permuted fusion site, AuNP) distances under Ripley-style label permutation.

    For each synaptic cleft and permutation replicate (default 10), fusion vs AuNP labels
    are reassigned on the pooled fusion + zone AuNP positions (same index pool as
    Ripley H₁₂ label-permutation null). Exactly ``n_fusion`` positions receive the
    fusion label each round; each is projected onto the presynaptic synaptic cleft
    surface before 3D distance calculation to every AuNP in that zone.
    """
    tomogram_path = Path(tomogram_path)
    alignment_dir = require_alignment_dir(alignment_dir)
    aunp_coords = np.atleast_2d(np.asarray(aunp_coords, dtype=float))
    aunp_cleft_ids = np.asarray(aunp_cleft_ids, dtype=int)
    if aunp_coords.size == 0:
        return pd.DataFrame()

    az_mapping = {int(k): v for k, v in az_mapping.items()}
    fusion_rows = enumerate_close_vesicle_fusion_points(
        tomogram_path,
        alignment_dir=alignment_dir,
        vesicle_distance_threshold=vesicle_distance_threshold,
        fusion_point_threshold=fusion_point_threshold,
        fusing_only=True,
    )
    if not fusion_rows:
        return pd.DataFrame()

    fusion_by_zone: dict[str, list[dict]] = {}
    for fp in fusion_rows:
        zone_name = zone_name_for_presynaptic_membrane(fp.get("closest_membrane"))
        if zone_name:
            fusion_by_zone.setdefault(zone_name, []).append(fp)

    rng = np.random.default_rng(seed)
    long_rows: list[dict] = []

    for zone_name, zone_fusion_rows in fusion_by_zone.items():
        az_ids = [idx for idx, zname in az_mapping.items() if zname == zone_name]
        if not az_ids:
            continue
        aunp_mask = np.isin(aunp_cleft_ids, az_ids)
        if not np.any(aunp_mask):
            continue
        zone_aunp_coords = aunp_coords[aunp_mask]
        zone_aunp_global_idx = np.flatnonzero(aunp_mask)

        pre_surface = load_presynaptic_az_points_for_zone(
            tomogram_path, alignment_dir, zone_name
        )
        if len(pre_surface) == 0:
            continue
        pre_tree = cKDTree(pre_surface)

        fusion_world = np.array(
            [
                [fp["fusion_point_x_nm"], fp["fusion_point_y_nm"], fp["fusion_point_z_nm"]]
                for fp in zone_fusion_rows
            ],
            dtype=float,
        )
        if len(fusion_world) == 0 or len(zone_aunp_coords) == 0:
            continue

        pool_world = np.vstack([fusion_world, zone_aunp_coords])
        n_fusion = len(fusion_world)
        n_pool = len(pool_world)
        if n_pool < n_fusion + 1:
            continue

        tomogram_name = zone_fusion_rows[0]["tomogram_name"]

        for perm_id in range(int(n_perm)):
            labels = np.zeros(n_pool, dtype=bool)
            labels[rng.choice(n_pool, n_fusion, replace=False)] = True
            fusion_labeled_idx = np.flatnonzero(labels)
            if len(fusion_labeled_idx) != n_fusion:
                continue

            fusion_source_world = pool_world[fusion_labeled_idx]
            fusion_on_pre = _project_points_to_surface(
                fusion_source_world, pre_tree, pre_surface
            )

            for fusion_site_idx, (pool_idx, query_xyz, source_xyz) in enumerate(
                zip(fusion_labeled_idx, fusion_on_pre, fusion_source_world)
            ):
                pool_idx = int(pool_idx)
                if pool_idx < n_fusion:
                    fp = zone_fusion_rows[pool_idx]
                    pool_source = "original_fusion"
                    vesicle_id = fp["vesicle_id"]
                    vesicle_name = fp["vesicle_name"]
                    distance_to_az = fp["distance_to_presynaptic_az_nm"]
                else:
                    pool_source = "original_aunp"
                    vesicle_id = np.nan
                    vesicle_name = None
                    distance_to_az = np.nan

                distances = np.linalg.norm(zone_aunp_coords - query_xyz, axis=1)
                for local_aunp_idx, dist in enumerate(distances):
                    row = {
                        "tomogram_name": tomogram_name,
                        "alignment_dir": alignment_dir,
                        "cleft_name": zone_name,
                        "permutation_id": int(perm_id),
                        "fusion_site_index": int(fusion_site_idx),
                        "pool_index": pool_idx,
                        "pool_source": pool_source,
                        "aunp_index": int(zone_aunp_global_idx[local_aunp_idx]),
                        "source_point_x_nm": float(source_xyz[0]),
                        "source_point_y_nm": float(source_xyz[1]),
                        "source_point_z_nm": float(source_xyz[2]),
                        "query_point_x_nm": float(query_xyz[0]),
                        "query_point_y_nm": float(query_xyz[1]),
                        "query_point_z_nm": float(query_xyz[2]),
                        "fusion_point_to_aunp_distance_nm": float(dist),
                    }
                    if pool_source == "original_fusion":
                        row["vesicle_id"] = vesicle_id
                        row["vesicle_name"] = vesicle_name
                        row["distance_to_presynaptic_az_nm"] = distance_to_az
                        row["fusion_point_x_nm"] = float(fp["fusion_point_x_nm"])
                        row["fusion_point_y_nm"] = float(fp["fusion_point_y_nm"])
                        row["fusion_point_z_nm"] = float(fp["fusion_point_z_nm"])
                    else:
                        row["vesicle_id"] = vesicle_id
                        row["vesicle_name"] = vesicle_name
                        row["distance_to_presynaptic_az_nm"] = distance_to_az
                    long_rows.append(row)
    return pd.DataFrame(long_rows)


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
        / "cleft_MIPs"
    )
    if not az_dir.is_dir():
        return None
    matches = sorted(az_dir.glob(f"{tomogram_path.name}_cleft_MIP_{zone_name}*.mrc"))
    return matches[0] if matches else None


def _load_zone_transform_from_cleft_results(
    tomogram_path: Path,
    alignment_dir: str,
    zone_name: str,
) -> dict:
    """
    Reconstruct zonogram center / transformation / extent from saved synaptic-cleft analysis.

    Matches the coordinate system used when STT visualizations render active zonograms
    (same define_active_zonogram path as visualization.run_combined_zonogram_analysis).
    """
    from .cleft import (
        define_active_zonogram,
        find_active_zones_from_glb,
        import_membrane_segmentations_from_glb,
    )

    np.random.seed(42)
    membrane_data = import_membrane_segmentations_from_glb(
        str(tomogram_path),
        alignment_dir=alignment_dir,
    )
    clefts_data = find_active_zones_from_glb(membrane_data, distance_range=(10.0, 40.0))
    zonogram_results = define_active_zonogram(clefts_data)
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
    """Return (zonogram_findingampa, zone_data) using saved MRC + synaptic-cleft transform."""
    import mrcfile
    import torch
    from .visualization import transform_positions_to_zonogram_coords

    mrc_path = _find_precalculated_zonogram_mrc(tomogram_path, alignment_dir, zone_name)
    if mrc_path is None:
        raise FileNotFoundError(
            f"No precalculated zonogram MRC for zone '{zone_name}' under "
            f"{tomogram_path / alignment_dir / 'STT_results' / 'visualizations' / 'cleft_MIPs'}"
        )

    with mrcfile.open(mrc_path, mode="r") as mrc:
        vol = torch.tensor(np.asarray(mrc.data, dtype=np.float32))
    zone_data = _load_zone_transform_from_cleft_results(
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

    zone_col = "nearest_scan_cleft_name"
    if zone_col not in real.columns:
        print("Skipping zonogram XY plot: nearest_scan_cleft_name missing.")
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

        zone_note = _fusing_vesicle_sample_note(sub_f, default_n_tomograms=1)
        ax.set_title(
            f"Fusion vs tangential controls on active zonogram\n"
            f"{zone_name} | r={int(probe_radius_nm)} nm | controls colored by d\n"
            f"{zone_note}"
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


def _count_cleft_instances(df: pd.DataFrame) -> int | None:
    """Count tomogram×active-zone instances (additive across tomograms), not unique zone names."""
    if df.empty:
        return None
    zone_col = None
    for col in ("cleft_name", "nearest_scan_cleft_name"):
        if col in df.columns:
            zone_col = col
            break
    if zone_col is None:
        return None
    if "tomogram_name" in df.columns:
        return int(df[["tomogram_name", zone_col]].dropna(how="any").drop_duplicates().shape[0])
    return int(df[zone_col].dropna().nunique())


def _fusing_vesicle_sample_note(
    real: pd.DataFrame,
    *,
    default_n_tomograms: int | None = None,
) -> str:
    """Human-readable fusing-vesicle and tomogram counts for plot annotations."""
    if real.empty:
        n_tomograms = 0 if default_n_tomograms is None else int(default_n_tomograms)
        return f"n_fusing_vesicles=0, n_tomograms={n_tomograms}"

    if "tomogram_name" in real.columns:
        n_vesicles = len(real[["tomogram_name", "vesicle_id"]].drop_duplicates())
        n_tomograms = (
            int(default_n_tomograms)
            if default_n_tomograms is not None
            else int(real["tomogram_name"].nunique())
        )
    else:
        n_vesicles = int(real["vesicle_id"].nunique())
        n_tomograms = 1 if default_n_tomograms is None else int(default_n_tomograms)

    parts = [f"n_fusing_vesicles={n_vesicles}", f"n_tomograms={n_tomograms}"]
    n_zones = _count_cleft_instances(real)
    if n_zones is not None and n_zones > 0:
        parts.append(f"n_clefts={n_zones}")
    return ", ".join(parts)


def _add_figure_sample_note(fig, note: str, *, y: float = 0.01) -> None:
    """Bottom-right annotation listing how many fusing vesicles are in the plot."""
    if not note:
        return
    fig.text(
        0.99,
        y,
        note,
        ha="right",
        va="bottom",
        fontsize=8,
        transform=fig.transFigure,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, pad=0.3),
    )


DEFAULT_RIPLEY_R_MAX_NM = 500.0
DEFAULT_RIPLEY_R_STEP_NM = 5.0
DEFAULT_RIPLEY_N_PERM = 499
RIPLEY_PLOT_X_MAX_NM = 100.0
RIPLEY_PERCENTILE_LO = 2.5
RIPLEY_PERCENTILE_HI = 97.5
UNCERTAINTY_METHOD_PERCENTILE = "percentile_2p5_97p5"
UNCERTAINTY_METHOD_MEAN_SD = "mean_sd"
DEFAULT_RIPLEY_O_MESH_MAX_VERTS = 4000
DEFAULT_RIPLEY_O_GEODESIC_METHOD = "fast"


def _load_az_surface_txt(path: Path) -> np.ndarray:
    if not path.is_file():
        return np.zeros((0, 3), dtype=float)
    data = np.atleast_2d(np.loadtxt(path, delimiter=None))
    if data.size == 0:
        return np.zeros((0, 3), dtype=float)
    return data.astype(float)


def _load_postsynaptic_cleft_surface(
    tomogram_path: Path,
    alignment_dir: str,
    zone_name: str,
) -> np.ndarray:
    """Full postsynaptic synaptic-cleft patch (outer + inner) for one zone."""
    az_dir = tomogram_path / alignment_dir / "STT_results" / "cleft"
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
    (πr²), appropriate for synaptic-cleft patches embedded in 3D.
    """
    x = np.atleast_2d(np.asarray(x, dtype=float))
    y = np.atleast_2d(np.asarray(y, dtype=float))
    r_vals = np.asarray(r_vals, dtype=float)
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0 or window_area_nm2 <= 0:
        return np.full(len(r_vals), np.nan)

    tree = cKDTree(y)
    counts = np.zeros(len(r_vals), dtype=float)
    r_max = float(r_vals[-1])
    neighbor_lists = tree.query_ball_point(x, r=r_max)
    for i, neighbor_idx in enumerate(neighbor_lists):
        if not neighbor_idx:
            continue
        dists = np.linalg.norm(y[np.asarray(neighbor_idx, dtype=int)] - x[i], axis=1)
        counts += (dists[:, None] < r_vals[None, :]).sum(axis=0).astype(float)
    return (window_area_nm2 / (n1 * n2)) * counts


def ripley_h12(k12: np.ndarray, r_vals: np.ndarray) -> np.ndarray:
    """Standardized bivariate Ripley H: sqrt(K12 / π) − r."""
    k12 = np.maximum(np.asarray(k12, dtype=float), 0.0)
    return np.sqrt(k12 / np.pi) - r_vals


def ripley_k12_from_h12(h12: np.ndarray, r_vals: np.ndarray) -> np.ndarray:
    """Invert H₁₂ → K₁₂: K = π·(H+r)² (non-negative radius argument)."""
    h12 = np.asarray(h12, dtype=float)
    r_vals = np.asarray(r_vals, dtype=float)
    return np.pi * np.maximum(h12 + r_vals, 0.0) ** 2


def mean_h12_from_h_curves(h_curves: np.ndarray, r_vals: np.ndarray) -> np.ndarray:
    """Average replicate H₁₂ on the K scale (via H→K invert), then convert once with ``ripley_h12``."""
    r_vals = np.asarray(r_vals, dtype=float)
    h_mat = np.atleast_2d(np.asarray(h_curves, dtype=float))
    if h_mat.size == 0 or h_mat.shape[0] == 0:
        return np.full(len(r_vals), np.nan)
    k_mat = ripley_k12_from_h12(h_mat, r_vals[None, :])
    return ripley_h12(np.nanmean(k_mat, axis=0), r_vals)


def _h12_mean_sd_band(h_curves: np.ndarray, r_vals: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean ± SD on K (from inverted H), mapped once through ``ripley_h12``."""
    r_vals = np.asarray(r_vals, dtype=float)
    h_mat = np.atleast_2d(np.asarray(h_curves, dtype=float))
    if h_mat.size == 0 or h_mat.shape[0] == 0:
        nan = np.full(len(r_vals), np.nan)
        return nan, nan, nan
    k_mat = ripley_k12_from_h12(h_mat, r_vals[None, :])
    mean_k = np.nanmean(k_mat, axis=0)
    n_valid = np.sum(~np.isnan(k_mat), axis=0)
    with np.errstate(invalid="ignore"):
        sd_k = np.nanstd(k_mat, axis=0, ddof=1)
    sd_k = np.where(n_valid > 1, sd_k, 0.0)
    return (
        ripley_h12(mean_k - sd_k, r_vals),
        ripley_h12(mean_k, r_vals),
        ripley_h12(mean_k + sd_k, r_vals),
    )


def ripley_h12_from_points(
    x: np.ndarray,
    y: np.ndarray,
    r_vals: np.ndarray,
    window_area_nm2: float,
) -> np.ndarray:
    return ripley_h12(cross_k12(x, y, r_vals, window_area_nm2), r_vals)


@contextlib.contextmanager
def _suppress_membrain_mesh_stdout():
    """Silence membrain-stats debug print in split_mesh_into_connected_components."""
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        yield


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
    with _suppress_membrain_mesh_stdout():
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
    _apply_ripley_xlim(ax)


def _apply_ripley_xlim(ax) -> None:
    ax.set_xlim(0.0, RIPLEY_PLOT_X_MAX_NM)


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


def _replicate_mean_sd_band(curves: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean ± sample SD band across replicate curves (n_replicates × n_r)."""
    curves = np.asarray(curves, dtype=float)
    if curves.ndim != 2 or len(curves) == 0:
        n_r = curves.shape[1] if curves.ndim == 2 and curves.size else 0
        nan = np.full(n_r, np.nan)
        return nan, nan, nan
    mean = curves.mean(axis=0)
    if len(curves) > 1:
        sd = curves.std(axis=0, ddof=1)
    else:
        sd = np.zeros_like(mean)
    return mean - sd, mean, mean + sd


def _replicate_envelope_band(
    curves: np.ndarray,
    *,
    method: str = "percentile",
    r_vals: np.ndarray | None = None,
    average_h12_on_k: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Envelope band across replicates.

    For H₁₂ with ``average_h12_on_k=True`` and ``method='mean_sd'``, mean±SD are computed
    on K (inverting H) then mapped through ``ripley_h12``. Percentile bands always stay on
    the input curve scale (H of each replicate).
    """
    if method == "mean_sd":
        if average_h12_on_k:
            if r_vals is None:
                raise ValueError("r_vals required when average_h12_on_k=True")
            return _h12_mean_sd_band(curves, r_vals)
        return _replicate_mean_sd_band(curves)
    return _replicate_percentile_band(curves)


def _label_perm_envelope_labels(
    *,
    method: str,
    n_perm: int,
    n_obs_replicates: int,
) -> dict[str, str]:
    if method == "mean_sd":
        return {
            "secondary_band_label": f"label null mean ± SD (n={n_perm})",
            "secondary_median_label": "label null mean",
            "primary_band_label": f"fusion mean ± SD (n={n_obs_replicates} vesicles)",
            "primary_median_label": "fusion mean (per-vesicle)",
        }
    return {
        "secondary_band_label": f"label null 2.5–97.5% (n={n_perm})",
        "secondary_median_label": "label null median",
        "primary_band_label": f"observed 2.5–97.5% (n={n_obs_replicates} vesicles)",
        "primary_median_label": "observed median (per-vesicle)",
    }


def _fusion_vs_control_envelope_labels(
    *,
    method: str,
    offset_nm: float,
    n_control_vesicles: int,
    n_fusion_vesicles: int,
) -> dict[str, str]:
    d = int(offset_nm)
    if method == "mean_sd":
        return {
            "secondary_band_label": f"controls d={d} nm mean ± SD (n={n_control_vesicles} vesicles)",
            "secondary_median_label": f"controls d={d} nm mean",
            "primary_band_label": f"fusion mean ± SD (n={n_fusion_vesicles} vesicles)",
            "primary_median_label": "fusion mean (per-vesicle)",
        }
    return {
        "secondary_band_label": f"controls d={d} nm 2.5–97.5% (n={n_control_vesicles} vesicles)",
        "secondary_median_label": f"controls d={d} nm median",
        "primary_band_label": f"fusion 2.5–97.5% (n={n_fusion_vesicles} vesicles)",
        "primary_median_label": "fusion median (per-vesicle)",
    }


def _ripley_envelope_style_specs() -> list[tuple[str, str, str]]:
    return [
        ("percentile", "", "2.5–97.5% bands + median"),
        ("mean_sd", "_mean_sd", "mean ± SD bands"),
    ]


def _save_label_perm_ripley_figures(
    figures_dir: Path,
    filename_stem: str,
    *,
    r_vals: np.ndarray,
    null_curves: np.ndarray,
    obs_curves: np.ndarray,
    n_perm: int,
    n_obs_replicates: int,
    ylabel: str,
    title_prefix: str,
    refline: float | None = None,
    refline_label: str | None = None,
    sample_note: str | None = None,
    average_h12_on_k: bool = False,
) -> None:
    for method, suffix, band_note in _ripley_envelope_style_specs():
        null_lo, null_c, null_hi = _replicate_envelope_band(
            null_curves, method=method, r_vals=r_vals, average_h12_on_k=average_h12_on_k
        )
        obs_lo, obs_c, obs_hi = _replicate_envelope_band(
            obs_curves, method=method, r_vals=r_vals, average_h12_on_k=average_h12_on_k
        )
        labels = _label_perm_envelope_labels(
            method=method,
            n_perm=n_perm,
            n_obs_replicates=n_obs_replicates,
        )
        fig, ax = plt.subplots(figsize=(7, 5))
        _plot_label_permutation_envelope_panel(
            ax,
            r_vals,
            null_lo=null_lo,
            null_med=null_c,
            null_hi=null_hi,
            n_perm=n_perm,
            obs_lo=obs_lo,
            obs_med=obs_c,
            obs_hi=obs_hi,
            n_obs_replicates=n_obs_replicates,
            ylabel=ylabel,
            title=f"{title_prefix}\n({band_note})",
            refline=refline,
            refline_label=refline_label,
            secondary_band_label=labels["secondary_band_label"],
            secondary_median_label=labels["secondary_median_label"],
            primary_band_label=labels["primary_band_label"],
            primary_median_label=labels["primary_median_label"],
        )
        fig.tight_layout()
        if sample_note:
            _add_figure_sample_note(fig, sample_note)
        fig.savefig(figures_dir / f"{filename_stem}{suffix}.png", dpi=150)
        plt.close(fig)


def _save_fusion_vs_control_by_d_figures(
    figures_dir: Path,
    filename_stem: str,
    *,
    r_vals: np.ndarray,
    fusion_vesicle_curves: np.ndarray,
    panel_specs: Sequence[dict[str, Any]],
    ylabel: str,
    suptitle_base: str,
    refline: float | None = None,
    refline_label: str | None = None,
    sample_note: str | None = None,
    average_h12_on_k: bool = False,
) -> None:
    n_panels = len(panel_specs)
    if n_panels == 0:
        return
    ncols = min(3, n_panels)
    nrows = int(np.ceil(n_panels / ncols))

    for method, suffix, band_note in _ripley_envelope_style_specs():
        fusion_lo, fusion_c, fusion_hi = _replicate_envelope_band(
            fusion_vesicle_curves,
            method=method,
            r_vals=r_vals,
            average_h12_on_k=average_h12_on_k,
        )
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4 * nrows), squeeze=False)
        axes_flat = axes.flatten()
        for ax, spec in zip(axes_flat, panel_specs):
            offset_nm = float(spec["offset_nm"])
            ctrl_curves = spec["ctrl_curves"]
            if ctrl_curves is None:
                ax.set_title(f"d={int(offset_nm)} nm (no controls)")
                continue
            ctrl_lo, ctrl_c, ctrl_hi = _replicate_envelope_band(
                ctrl_curves,
                method=method,
                r_vals=r_vals,
                average_h12_on_k=average_h12_on_k,
            )
            labels = _fusion_vs_control_envelope_labels(
                method=method,
                offset_nm=offset_nm,
                n_control_vesicles=int(spec["n_control_vesicles"]),
                n_fusion_vesicles=int(spec["n_fusion_vesicles"]),
            )
            _plot_fusion_vs_control_ripley_panel(
                ax,
                r_vals,
                ctrl_lo=ctrl_lo,
                ctrl_med=ctrl_c,
                ctrl_hi=ctrl_hi,
                n_control_vesicles=int(spec["n_control_vesicles"]),
                offset_nm=offset_nm,
                fusion_lo=fusion_lo,
                fusion_med=fusion_c,
                fusion_hi=fusion_hi,
                n_fusion_vesicles=int(spec["n_fusion_vesicles"]),
                ylabel=ylabel,
                refline=refline,
                refline_label=refline_label,
                secondary_band_label=labels["secondary_band_label"],
                secondary_median_label=labels["secondary_median_label"],
                primary_band_label=labels["primary_band_label"],
                primary_median_label=labels["primary_median_label"],
            )
        for ax in axes_flat[n_panels:]:
            ax.set_visible(False)
        fig.suptitle(f"{suptitle_base}\n({band_note})", y=1.02)
        fig.tight_layout()
        if sample_note:
            _add_figure_sample_note(fig, sample_note)
        fig.savefig(
            figures_dir / f"{filename_stem}{suffix}.png",
            dpi=150,
            bbox_inches="tight",
        )
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
    _apply_ripley_xlim(ax)


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
    secondary_band_label: str | None = None,
    secondary_median_label: str | None = None,
    primary_band_label: str | None = None,
    primary_median_label: str | None = None,
    fusion_mean_only: bool = False,
    obs_mean: np.ndarray | None = None,
    null_mean: np.ndarray | None = None,
) -> None:
    if secondary_band_label is None:
        secondary_band_label = f"label null 2.5–97.5% (n={n_perm})"
    if secondary_median_label is None:
        secondary_median_label = "label null median"
    if primary_band_label is None:
        primary_band_label = f"observed 2.5–97.5% (n={n_obs_replicates} vesicles)"
    if primary_median_label is None:
        primary_median_label = "observed median (per-vesicle)"
    _plot_ripley_dual_envelope_panel(
        ax,
        r_vals,
        secondary_lo=null_lo,
        secondary_med=null_med,
        secondary_hi=null_hi,
        secondary_band_label=secondary_band_label,
        secondary_median_label=secondary_median_label,
        primary_lo=obs_lo,
        primary_med=obs_med,
        primary_hi=obs_hi,
        primary_band_label=primary_band_label,
        primary_median_label=primary_median_label,
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
    secondary_band_label: str | None = None,
    secondary_median_label: str | None = None,
    primary_band_label: str | None = None,
    primary_median_label: str | None = None,
    fusion_mean_only: bool = False,
    fusion_mean: np.ndarray | None = None,
    ctrl_mean: np.ndarray | None = None,
) -> None:
    if secondary_band_label is None:
        secondary_band_label = (
            f"controls d={int(offset_nm)} nm 2.5–97.5% (n={n_control_vesicles} vesicles)"
        )
    if secondary_median_label is None:
        secondary_median_label = f"controls d={int(offset_nm)} nm median"
    if primary_band_label is None:
        primary_band_label = f"fusion 2.5–97.5% (n={n_fusion_vesicles} vesicles)"
    if primary_median_label is None:
        primary_median_label = "fusion median (per-vesicle)"
    _plot_ripley_dual_envelope_panel(
        ax,
        r_vals,
        secondary_lo=ctrl_lo,
        secondary_med=ctrl_med,
        secondary_hi=ctrl_hi,
        secondary_band_label=secondary_band_label,
        secondary_median_label=secondary_median_label,
        primary_lo=fusion_lo,
        primary_med=fusion_med,
        primary_hi=fusion_hi,
        primary_band_label=primary_band_label,
        primary_median_label=primary_median_label,
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
    aunp_pick_star_pattern: str | None = None,
) -> np.ndarray:
    import starfile
    from .cleft import load_cleft_mapping
    from .aunps import _read_aunp_pick_star_dataframe

    aunps_dir = tomogram_path / alignment_dir / "STT_results" / "aunps"
    star_path = aunps_dir / "aunp_clusters.star"
    if not star_path.is_file():
        pick_pattern = normalize_aunp_pick_star_pattern(aunp_pick_star_pattern)
        aunps_pick_dir = tomogram_path / alignment_dir / "aunps"
        mapping = load_cleft_mapping(tomogram_path, alignment_dir)
        az_ids: list[int] | None = None
        if mapping:
            mapping = {int(k): v for k, v in mapping.items()}
            zone_az_ids = [idx for idx, zname in mapping.items() if zname == zone_name]
            if zone_az_ids:
                az_ids = zone_az_ids
        pick_files = discover_aunp_pick_star_files(
            aunps_pick_dir, az_ids, pattern=pick_pattern
        )
        if not pick_files:
            raise FileNotFoundError(
                f"No AuNP pick STAR files matching pattern {pick_pattern!r} "
                f"found under {aunps_pick_dir}"
            )
        pick_dfs = []
        for _az_id, pick_path in pick_files:
            pick_df = _read_aunp_pick_star_dataframe(pick_path)
            if pick_df is not None and not pick_df.empty:
                pick_dfs.append(pick_df)
        if not pick_dfs:
            raise FileNotFoundError(
                f"No AuNP coordinates in pick STAR files matching pattern {pick_pattern!r} "
                f"under {aunps_pick_dir}"
            )
        aunp_df = pd.concat(pick_dfs, ignore_index=True)
    else:
    star_data = starfile.read(star_path)
    if isinstance(star_data, dict):
        aunp_df = next(v for v in star_data.values() if isinstance(v, pd.DataFrame))
    else:
        aunp_df = star_data

    if "cleft" in aunp_df.columns:
        mapping = load_cleft_mapping(tomogram_path, alignment_dir)
        if mapping:
            mapping = {int(k): v for k, v in mapping.items()}
            az_ids = [idx for idx, zname in mapping.items() if zname == zone_name]
            if az_ids:
                aunp_df = aunp_df[aunp_df["cleft"].isin(az_ids)]

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


def _ripley_csv_row_meta(tomogram_path: Path, alignment_dir: str) -> dict[str, str]:
    return {
        "tomogram_name": tomogram_path.name,
        "alignment_dir": alignment_dir,
    }


def _stamp_tomogram_metadata(
    df: pd.DataFrame | None,
    *,
    tomogram_name: str,
    alignment_dir: str,
) -> pd.DataFrame | None:
    """Ensure combined tables identify their source tomogram."""
    if df is None or df.empty:
        return df
    out = df.copy()
    if "tomogram_name" not in out.columns:
        out["tomogram_name"] = tomogram_name
    if "alignment_dir" not in out.columns:
        out["alignment_dir"] = alignment_dir
    return out


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
    aunp_pick_star_pattern: str | None = None,
) -> pd.DataFrame | None:
    """
    Bivariate Ripley H on postsynaptic-projected fusion / control / AuNP positions.

    #1: fusion vs controls-at-d — per-vesicle 2.5–97.5% bands + median, per d.
    #2: label-permutation null on pooled fusion + AuNP projected positions.
    """
    zone_col = "nearest_scan_cleft_name"
    if zone_col not in df.columns:
        print("Skipping Ripley analysis: nearest_scan_cleft_name missing.")
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
    row_meta = _ripley_csv_row_meta(tomogram_path, alignment_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    all_result_rows: list[dict] = []
    offsets = sorted(d for d in ctrl["control_offset_nm"].unique() if d > 0)

    for zone_name in sorted(real[zone_col].dropna().unique()):
        try:
            post_surface = _load_postsynaptic_cleft_surface(tomogram_path, alignment_dir, str(zone_name))
        except FileNotFoundError as exc:
            print(f"Skipping Ripley for {zone_name}: {exc}")
            continue

        post_tree = cKDTree(post_surface)
        window_area = _estimate_planar_window_area_nm2(post_surface)

        try:
            aunp_xyz = _load_aunp_coordinates_for_zone(
                tomogram_path,
                alignment_dir,
                str(zone_name),
                post_tree,
                aunp_pick_star_pattern=aunp_pick_star_pattern,
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
        h12_fusion_mean = mean_h12_from_h_curves(fusion_vesicle_curves, r_vals)

        # --- #2 Label permutation (fusion vs AuNP) ---
        pool = np.vstack([fusion_post, aunp_post])
        h12_obs = ripley_h12_from_points(fusion_post, aunp_post, r_vals, window_area)
        perm_curves = _label_permutation_h12_curves(
            pool, len(fusion_post), r_vals, window_area, n_perm, rng
        )
        h12_null_lo, h12_null_med, h12_null_hi = _replicate_percentile_band(perm_curves)
        h12_obs_lo, h12_obs_med, h12_obs_hi = _replicate_percentile_band(fusion_vesicle_curves)
        p_label_two = _monte_carlo_p_two_sided(h12_obs, perm_curves)
        p_label_greater = (np.sum(perm_curves >= h12_obs, axis=0) + 1) / (n_perm + 1)
        p_label_less = (np.sum(perm_curves <= h12_obs, axis=0) + 1) / (n_perm + 1)

        _save_label_perm_ripley_figures(
            figures_dir,
            f"ripley_h12_label_permutation_{zone_name}",
            r_vals=r_vals,
            null_curves=perm_curves,
            obs_curves=fusion_vesicle_curves,
            n_perm=n_perm,
            n_obs_replicates=len(fusion_by_vesicle),
            ylabel="Ripley H₁₂(r) = √(K₁₂/π) − r",
            title_prefix=(
                f"Label-permutation null: fusion vs AuNP on postsynaptic AZ\n"
                f"{zone_name} (p-values: pooled observed vs null)"
            ),
            refline=0.0,
            refline_label="H₁₂ = 0",
            average_h12_on_k=True,
        )

        _plot_significance_single(
            r_vals,
            p_label_two,
            title=(
                f"Label-permutation significance (fusion vs AuNP)\n"
                f"{zone_name} | two-sided Monte Carlo p, n={n_perm}"
            ),
            output_path=figures_dir / f"ripley_h12_pvalues_label_permutation_{zone_name}.png",
            panel_note=(
                f"n_fusing_vesicles={len(fusion_by_vesicle)}, n_tomograms=1, "
                f"n_fusion={len(fusion_post)}, n_aunp={len(aunp_post)}"
            ),
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
        _apply_ripley_xlim(ax)
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
                    **row_meta,
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
        p_by_d: dict[str, np.ndarray] = {}
        panel_notes: dict[str, str] = {}
        fvc_by_offset: dict[float, tuple[np.ndarray, np.ndarray]] = {}
        fvc_panel_specs: list[dict[str, Any]] = []

        for offset_nm in offsets:
            sub_c = _dedupe_rows_by_xyz(
                ctrl[
                    (ctrl["probe_radius_nm"] == probe_radius_for_coords)
                    & (ctrl["control_offset_nm"] == offset_nm)
                    & (ctrl[zone_col] == zone_name)
                ],
                ("query_point_x_nm", "query_point_y_nm", "query_point_z_nm"),
            )
            if sub_c.empty:
                fvc_panel_specs.append(
                    {
                        "offset_nm": float(offset_nm),
                        "ctrl_curves": None,
                        "n_control_vesicles": 0,
                        "n_fusion_vesicles": len(fusion_by_vesicle),
                    }
                )
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
            h12_ctrl_mean = mean_h12_from_h_curves(ctrl_curves, r_vals)
            p_fusion_vs_ctrl = _vesicle_paired_pvalues(fusion_curves, ctrl_curves)
            fvc_by_offset[float(offset_nm)] = (fusion_curves, ctrl_curves)
            fvc_panel_specs.append(
                {
                    "offset_nm": float(offset_nm),
                    "ctrl_curves": ctrl_curves,
                    "n_control_vesicles": len(control_by_vesicle),
                    "n_fusion_vesicles": len(fusion_by_vesicle),
                }
            )

            p_by_d[f"d={int(offset_nm)} nm"] = p_fusion_vs_ctrl
            min_p_floor = _wilcoxon_min_achievable_p(len(paired_ids))
            panel_notes[f"d={int(offset_nm)} nm"] = (
                f"n_fusing_vesicles={len(fusion_by_vesicle)}, n_tomograms=1, "
                f"n_paired={len(paired_ids)}"
                + (
                    f"\nWilcoxon floor={min_p_floor:.3g}"
                    if np.isfinite(min_p_floor) and min_p_floor > 0.05
                    else ""
                )
            )

            for r, f_lo, f_med, f_hi, f_mean, c_lo, c_med, c_hi, c_mean, p_val in zip(
                r_vals,
                h12_fusion_lo,
                h12_fusion_med,
                h12_fusion_hi,
                h12_fusion_mean,
                h12_ctrl_lo,
                h12_ctrl_med,
                h12_ctrl_hi,
                h12_ctrl_mean,
                p_fusion_vs_ctrl,
            ):
                all_result_rows.append(
                    {
                        **row_meta,
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
                        "h12_control_mean": float(c_mean),
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

        _save_fusion_vs_control_by_d_figures(
            figures_dir,
            f"ripley_h12_fusion_vs_controls_by_d_{zone_name}",
            r_vals=r_vals,
            fusion_vesicle_curves=fusion_vesicle_curves,
            panel_specs=fvc_panel_specs,
            ylabel="H₁₂(r)",
            suptitle_base=(
                f"Ripley H₁₂ on postsynaptic AZ: fusion vs controls\n"
                f"{zone_name} | n_fusing_vesicles={len(fusion_by_vesicle)}, n_tomograms=1"
            ),
            refline=0.0,
            refline_label="H₁₂ = 0",
            average_h12_on_k=True,
        )

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
    aunp_pick_star_pattern: str | None = None,
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

    zone_col = "nearest_scan_cleft_name"
    if zone_col not in df.columns:
        print("Skipping Ripley's O analysis: nearest_scan_cleft_name missing.")
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
    row_meta = _ripley_csv_row_meta(tomogram_path, alignment_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    all_result_rows: list[dict] = []
    offsets = sorted(d for d in ctrl["control_offset_nm"].unique() if d > 0)

    for zone_name in sorted(real[zone_col].dropna().unique()):
        try:
            post_surface = _load_postsynaptic_cleft_surface(tomogram_path, alignment_dir, str(zone_name))
        except FileNotFoundError as exc:
            print(f"Skipping Ripley's O for {zone_name}: {exc}")
            continue

        post_tree = cKDTree(post_surface)

        try:
            aunp_xyz = _load_aunp_coordinates_for_zone(
                tomogram_path,
                alignment_dir,
                str(zone_name),
                post_tree,
                aunp_pick_star_pattern=aunp_pick_star_pattern,
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

        _save_label_perm_ripley_figures(
            figures_dir,
            f"ripley_o_label_permutation_{zone_name}",
            r_vals=r_vals,
            null_curves=perm_curves,
            obs_curves=fusion_vesicle_curves,
            n_perm=n_perm,
            n_obs_replicates=len(fusion_by_vesicle),
            ylabel="Ripley's O(r) [membrain-stats geodesic]",
            title_prefix=(
                f"Label-permutation null: Ripley's O fusion vs AuNP\n"
                f"{zone_name} (p-values: pooled observed vs null)"
            ),
            refline=1.0,
            refline_label="CSR (O=1)",
        )

        _plot_significance_single(
            r_vals,
            p_label_two,
            title=(
                f"Ripley's O label-permutation significance\n"
                f"{zone_name} | two-sided Monte Carlo p, n={n_perm}"
            ),
            output_path=figures_dir / f"ripley_o_pvalues_label_permutation_{zone_name}.png",
            panel_note=(
                f"n_fusing_vesicles={len(fusion_by_vesicle)}, n_tomograms=1, "
                f"n_fusion={len(fusion_post)}, n_aunp={len(aunp_post)}"
            ),
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
        _apply_ripley_xlim(ax)
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
                    **row_meta,
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
        p_by_d: dict[str, np.ndarray] = {}
        panel_notes: dict[str, str] = {}
        fvc_by_offset: dict[float, tuple[np.ndarray, np.ndarray]] = {}
        fvc_panel_specs: list[dict[str, Any]] = []

        for offset_nm in offsets:
            sub_c = _dedupe_rows_by_xyz(
                ctrl[
                    (ctrl["probe_radius_nm"] == probe_radius_for_coords)
                    & (ctrl["control_offset_nm"] == offset_nm)
                    & (ctrl[zone_col] == zone_name)
                ],
                ("query_point_x_nm", "query_point_y_nm", "query_point_z_nm"),
            )
            if sub_c.empty:
                fvc_panel_specs.append(
                    {
                        "offset_nm": float(offset_nm),
                        "ctrl_curves": None,
                        "n_control_vesicles": 0,
                        "n_fusion_vesicles": len(fusion_by_vesicle),
                    }
                )
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
            p_fusion_vs_ctrl = _vesicle_paired_pvalues(fusion_curves, ctrl_curves)
            fvc_by_offset[float(offset_nm)] = (fusion_curves, ctrl_curves)
            fvc_panel_specs.append(
                {
                    "offset_nm": float(offset_nm),
                    "ctrl_curves": ctrl_curves,
                    "n_control_vesicles": len(control_by_vesicle),
                    "n_fusion_vesicles": len(fusion_by_vesicle),
                }
            )

            p_by_d[f"d={int(offset_nm)} nm"] = p_fusion_vs_ctrl
            min_p_floor = _wilcoxon_min_achievable_p(len(paired_ids))
            panel_notes[f"d={int(offset_nm)} nm"] = (
                f"n_fusing_vesicles={len(fusion_by_vesicle)}, n_tomograms=1, "
                f"n_paired={len(paired_ids)}"
                + (
                    f"\nWilcoxon floor={min_p_floor:.3g}"
                    if np.isfinite(min_p_floor) and min_p_floor > 0.05
                    else ""
                )
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
                        **row_meta,
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

        _save_fusion_vs_control_by_d_figures(
            figures_dir,
            f"ripley_o_fusion_vs_controls_by_d_{zone_name}",
            r_vals=r_vals,
            fusion_vesicle_curves=fusion_vesicle_curves,
            panel_specs=fvc_panel_specs,
            ylabel="Ripley's O(r)",
            suptitle_base=(
                f"Ripley's O (membrain-stats geodesic): fusion vs controls\n"
                f"{zone_name} | n_fusing_vesicles={len(fusion_by_vesicle)}, n_tomograms=1"
            ),
            refline=1.0,
            refline_label="CSR (O=1)",
        )

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
    filename_tag: str = "",
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
    name_suffix = f"_{filename_tag}" if filename_tag else ""

    if "tomogram_name" in real.columns:
        vesicle_keys = real[["tomogram_name", "vesicle_id"]].drop_duplicates()
    else:
        vesicle_keys = real[["vesicle_id"]].drop_duplicates()
        vesicle_keys["tomogram_name"] = None

    sample_note = _fusing_vesicle_sample_note(real)
    if filename_tag == "pooled":
        title_suffix = f"\npooled | {sample_note}"
    elif filename_tag:
        title_suffix = f"\n{filename_tag} | {sample_note}"
    else:
        title_suffix = f"\n{sample_note}"

    # 1) Paired delta (real - control mean) per vesicle vs offset, one probe radius panel
    n_panels = len(probe_radii)
    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4), sharey=True)
    if n_panels == 1:
        axes = [axes]
    for ax, probe_radius in zip(axes, probe_radii):
        deltas = []
        offsets = []
        for _, key_row in vesicle_keys.iterrows():
            vesicle_id = key_row["vesicle_id"]
            tomogram_name = key_row.get("tomogram_name")
            r_mask = (real["vesicle_id"] == vesicle_id) & (real["probe_radius_nm"] == probe_radius)
            c_mask = (ctrl["vesicle_id"] == vesicle_id) & (ctrl["probe_radius_nm"] == probe_radius)
            if tomogram_name is not None and pd.notna(tomogram_name):
                r_mask &= real["tomogram_name"] == tomogram_name
                c_mask &= ctrl["tomogram_name"] == tomogram_name
            r_row = real[r_mask]
            if r_row.empty:
                continue
            real_val = float(r_row["packing_coefficient"].iloc[0])
            c_sub = ctrl[c_mask]
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
    fig.suptitle(f"Paired fusion minus control packing{title_suffix}", y=1.02)
    _add_figure_sample_note(fig, sample_note)
    fig.tight_layout()
    fig.savefig(output_dir / f"delta_packing_vs_offset{name_suffix}.png", dpi=150, bbox_inches="tight")
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
            float(fusion_vals.std(ddof=1) / np.sqrt(len(fusion_vals)))
            if len(fusion_vals) > 1
            else 0.0
        )
        fusion_line_label = "fusion mean" if fusion_sem <= 0 else "_nolegend_"
        ax.axhline(
            fusion_mean,
            color="C3",
            linestyle="--",
            linewidth=1.5,
            label=fusion_line_label,
        )
        if fusion_sem > 0:
            ax.axhspan(
                fusion_mean - fusion_sem,
                fusion_mean + fusion_sem,
                color="C3",
                alpha=0.2,
                label="fusion mean ± SEM",
            )
        ax.set_title(f"probe r = {int(probe_radius)} nm")
        ax.set_xlabel("Control offset d (nm)")
        ax.set_ylabel("AuNP density (per nm²)")
        ax.legend(fontsize=7, loc="best")
    for ax in axes_flat[n_panels:]:
        ax.set_visible(False)
    fig.suptitle(
        f"AuNP density at controls vs offset (mean ± SEM at each d){title_suffix}",
        y=1.02,
        fontsize=11,
    )
    _add_figure_sample_note(fig, sample_note)
    fig.tight_layout()
    fig.savefig(output_dir / f"aunp_density_vs_control_offset{name_suffix}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    if not ctrl.empty and tomogram_path is not None:
        try:
        _plot_fusion_vs_control_zonogram(
            real,
            ctrl,
            tomogram_path=tomogram_path,
            alignment_dir=alignment_dir,
            probe_radius_nm=mid_radius,
                output_path=output_dir / f"fusion_vs_control_zonogram{name_suffix}.png",
        )
        except Exception as exc:
            print(f"  Warning: zonogram overlay failed{f' for {filename_tag}' if filename_tag else ''}: {exc}")

    print(f"Saved figures to {output_dir}")


def presynaptic_membrane_name_for_zone(zone_name: str, zone_data: dict | None = None) -> str:
    """Map synaptic cleft name to presynaptic membrane key used by vesicle results."""
    if zone_data and zone_data.get("presynaptic_membrane_index") is not None:
        pre_idx = int(zone_data["presynaptic_membrane_index"])
    else:
        # cleft_pre1_post1 -> 1  (split("_")[2] is post1 — do not use that)
        match = re.search(r"pre(\d+)", str(zone_name))
        if not match:
            raise ValueError(
                f"Cannot parse presynaptic membrane index from zone name {zone_name!r}"
            )
        pre_idx = int(match.group(1))
    return f"presynapticmembranes_{pre_idx}"


def load_presynaptic_az_points_for_zone(
    tomogram_path: Path,
    alignment_dir: str,
    zone_name: str,
) -> np.ndarray:
    """Presynaptic outer + inner synaptic-cleft points for one zone."""
    az_dir = tomogram_path / alignment_dir / "STT_results" / "cleft"
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
    """Presynaptic membrane entry with zone-specific synaptic-cleft points for controls."""
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
            "cleft_points": az_xyz,
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
    aunp_pick_star_pattern: str | None = None,
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
            aunp_pick_star_pattern=aunp_pick_star_pattern,
        )
        scan_df.to_csv(cache_path, index=False)
        scan_by_radius[float(probe_radius)] = scan_df
    return scan_by_radius


def scan_by_radius_for_zone(
    scan_by_radius: dict[float, pd.DataFrame],
    zone_name: str,
) -> dict[float, pd.DataFrame]:
    """Restrict tomogram-wide scan tables to one synaptic cleft."""
    zone_scans: dict[float, pd.DataFrame] = {}
    for radius, scan_df in scan_by_radius.items():
        if scan_df.empty or "cleft_name" not in scan_df.columns:
            continue
        subset = scan_df[scan_df["cleft_name"] == zone_name]
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
        / "cleft_MIPs"
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
    aunp_pick_star_pattern: str | None = None,
) -> pd.DataFrame | None:
    """Run tangential-shuffle control analysis for one synaptic cleft."""
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
    df["cleft_name"] = zone_name
    df["tomogram_name"] = tomogram_path.name
    df["alignment_dir"] = alignment_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "fusion_point_vs_aunp_density.csv", index=False)

    meta = {
        "tomogram_name": tomogram_path.name,
        "alignment_dir": alignment_dir,
        "cleft_name": zone_name,
        "probe_radii_nm": sorted(scan_by_radius),
        "offset_distances_nm": list(offset_distances_nm),
        "n_directions": int(n_directions),
        "max_snap_distance_nm": float(max_snap_distance_nm),
        "seed": int(seed),
        "n_fusion_vesicles": len(zone_fusion_rows),
        "n_table_rows": len(df),
        "aunp_pick_star_pattern": normalize_aunp_pick_star_pattern(aunp_pick_star_pattern),
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
            aunp_pick_star_pattern=aunp_pick_star_pattern,
        )
    if not skip_ripley_o:
        run_ripley_o_membrain_postsynaptic_analysis(
            df,
            tomogram_path,
            alignment_dir,
            output_dir,
            seed=seed,
            probe_radius_for_coords=probe_for_coords,
            aunp_pick_star_pattern=aunp_pick_star_pattern,
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
    clefts_glb: dict,
    probe_radii: Sequence[float] = PACKING_DENSITY_PROBE_RADII_NM,
    receptor_crosssection: float = 122.0,
    aunps_per_receptor: float = 2.0,
    vertex_sampling_step: int = 50,
    offset_distances_nm: Sequence[float] = DEFAULT_OFFSET_DISTANCES_NM,
    n_directions: int = DEFAULT_N_DIRECTIONS,
    seed: int = DEFAULT_ANALYSIS_SEED,
    max_snap_distance_nm: float = 10.0,
    write_figures: bool = True,
    aunp_pick_star_pattern: str | None = None,
) -> list[pd.DataFrame]:
    """Run fusion-point vs AuNP density analysis for each synaptic cleft in one tomogram."""
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
        / "cleft_MIPs"
        / FUSION_POINT_VS_AUNP_DENSITY_SUBDIR
        / _scan_cache_subdir_name(aunp_pick_star_pattern)
    )
    scan_by_radius = load_scan_by_radius_for_tomogram(
        tomogram_path,
        alignment_dir,
        probe_radii=probe_radii,
        cache_dir=shared_cache,
        vertex_sampling_step=vertex_sampling_step,
        receptor_crosssection=receptor_crosssection,
        aunps_per_receptor=aunps_per_receptor,
        aunp_pick_star_pattern=aunp_pick_star_pattern,
    )

    membrane_az_pairs = import_presynaptic_membranes_and_clefts(
        tomogram_path, alignment_dir=alignment_dir
    )
    zone_frames: list[pd.DataFrame] = []
    for zone_name, zone_data in clefts_glb.get("clefts", {}).items():
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
            aunp_pick_star_pattern=aunp_pick_star_pattern,
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

    for tomo, _set_name, _cleft_ids, alignment_dir in tomo_paths:
        tomogram_path = Path(tomo)
        tomogram_name = tomogram_path.name
        base = (
            tomogram_path
            / alignment_dir
            / "STT_results"
            / "visualizations"
            / "cleft_MIPs"
            / FUSION_POINT_VS_AUNP_DENSITY_SUBDIR
        )
        if not base.is_dir():
            continue
        for zone_dir in sorted(p for p in base.iterdir() if p.is_dir()):
            fusion_csv = zone_dir / "fusion_point_vs_aunp_density.csv"
            df_f = _read_optional_csv(fusion_csv)
            if df_f is not None:
                fusion_frames.append(
                    _stamp_tomogram_metadata(
                        df_f,
                        tomogram_name=tomogram_name,
                        alignment_dir=alignment_dir,
                    )
                )
            df_h12 = _read_optional_csv(zone_dir / "ripley_h12_postsynaptic.csv")
            if df_h12 is not None:
                h12_frames.append(
                    _stamp_tomogram_metadata(
                        df_h12,
                        tomogram_name=tomogram_name,
                        alignment_dir=alignment_dir,
                    )
                )
            df_o = _read_optional_csv(zone_dir / "ripley_o_membrain_postsynaptic.csv")
            if df_o is not None:
                o_frames.append(
                    _stamp_tomogram_metadata(
                        df_o,
                        tomogram_name=tomogram_name,
                        alignment_dir=alignment_dir,
                    )
                )

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


def _collect_ripley_vesicle_artifacts(
    tomo_paths: Iterable[tuple[Any, Any, Any, str]],
    artifact_name: str,
) -> list[dict[str, np.ndarray | str]]:
    """Load all saved per-vesicle Ripley artifacts across tomograms and synaptic clefts."""
    artifacts: list[dict[str, np.ndarray | str]] = []
    for tomo, _set_name, _cleft_ids, alignment_dir in tomo_paths:
        base = (
            Path(tomo)
            / alignment_dir
            / "STT_results"
            / "visualizations"
            / "cleft_MIPs"
            / FUSION_POINT_VS_AUNP_DENSITY_SUBDIR
        )
        if not base.is_dir():
            continue
        for zone_dir in sorted(p for p in base.iterdir() if p.is_dir() and not p.name.startswith("_")):
            artifact_path = zone_dir / artifact_name
            if not artifact_path.is_file():
                continue
            artifacts.append(_load_ripley_vesicle_artifact(artifact_path))
    return artifacts


def _stack_nonempty_curves(parts: list[np.ndarray]) -> np.ndarray:
    valid = [p for p in parts if p.ndim == 2 and len(p) > 0]
    if not valid:
        return np.empty((0, 0))
    return np.vstack(valid)


def _ripley_artifact_pool_summary(
    artifacts: Sequence[dict[str, np.ndarray | str]],
) -> tuple[int, int, int]:
    """Return (n_fusion_vesicle_curves, n_tomograms, n_clefts) for pooled Ripley plots."""
    n_tomograms = len({str(a.get("tomogram_name", "")) for a in artifacts if a.get("tomogram_name")})
    n_zones = len(
        {
            (str(a.get("tomogram_name", "")), str(a.get("zone_name", "")))
            for a in artifacts
            if a.get("tomogram_name") and a.get("zone_name")
        }
    )
    fusion_parts = [_stack_nonempty_curves([np.asarray(a["fusion_vesicle_curves"])]) for a in artifacts]
    fusion_vesicle_curves = _stack_nonempty_curves(fusion_parts)
    return len(fusion_vesicle_curves), n_tomograms, n_zones


def _offset_keys_from_artifact(artifact: dict[str, np.ndarray | str]) -> list[float]:
    offsets: list[float] = []
    for key in artifact:
        if not isinstance(key, str) or not key.startswith("fusion_offset_"):
            continue
        tag = key.removeprefix("fusion_")
        nm = tag.removeprefix("offset_").removesuffix("nm")
        offsets.append(float(nm))
    return sorted(offsets)


def _offset_keys_from_artifacts(artifacts: Sequence[dict[str, np.ndarray | str]]) -> list[float]:
    offsets: set[float] = set()
    for artifact in artifacts:
        offsets.update(_offset_keys_from_artifact(artifact))
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
    """Stack saved per-vesicle H₁₂ curves from all tomograms and synaptic clefts."""
    artifacts = _collect_ripley_vesicle_artifacts(tomo_paths, RIPLEY_H12_VESICLE_CURVES_NPZ)
    if not artifacts:
        print("Skipping pooled Ripley H₁₂: no saved vesicle-curve artifacts found.")
        return None

    figures_dir = output_dir / "figures" / "pooled_ripley"
    figures_dir.mkdir(parents=True, exist_ok=True)
    all_result_rows: list[dict] = []
    zone_name = "all"

    r_vals = np.asarray(artifacts[0]["r_vals"], dtype=float)
    fusion_parts = [_stack_nonempty_curves([np.asarray(a["fusion_vesicle_curves"])]) for a in artifacts]
    null_parts = [_stack_nonempty_curves([np.asarray(a["label_perm_null_curves"])]) for a in artifacts]
    obs_parts = [np.asarray(a["h12_obs"], dtype=float) for a in artifacts]

    fusion_vesicle_curves = _stack_nonempty_curves(fusion_parts)
    perm_curves = _stack_nonempty_curves(null_parts)
    if fusion_vesicle_curves.size == 0 or perm_curves.size == 0:
        print("Skipping pooled Ripley H₁₂: empty vesicle curves.")
        return None

    n_fusion_vesicles, n_tomograms, n_clefts = _ripley_artifact_pool_summary(artifacts)
    pool_note = (
        f"n_fusion_vesicles={n_fusion_vesicles}, n_tomograms={n_tomograms}, "
        f"n_clefts={n_clefts}"
    )

    h12_obs = np.mean(np.vstack(obs_parts), axis=0)
    h12_fusion_lo, h12_fusion_med, h12_fusion_hi = _replicate_percentile_band(fusion_vesicle_curves)
    h12_null_lo, h12_null_med, h12_null_hi = _replicate_percentile_band(perm_curves)
    p_label_two = _monte_carlo_p_two_sided(h12_obs, perm_curves)
    n_null = perm_curves.shape[0]
    p_label_greater = (np.sum(perm_curves >= h12_obs, axis=0) + 1) / (n_null + 1)
    p_label_less = (np.sum(perm_curves <= h12_obs, axis=0) + 1) / (n_null + 1)

    _save_label_perm_ripley_figures(
        figures_dir,
        f"ripley_h12_label_permutation_{file_tag}",
        r_vals=r_vals,
        null_curves=perm_curves,
        obs_curves=fusion_vesicle_curves,
        n_perm=n_null,
        n_obs_replicates=len(fusion_vesicle_curves),
        ylabel="Ripley H₁₂(r) = √(K₁₂/π) − r",
        title_prefix=(
            "Pooled label-permutation null: fusion vs AuNP on postsynaptic AZ\n"
            f"all fusing vesicles | {pool_note}"
        ),
        refline=0.0,
        refline_label="H₁₂ = 0",
        sample_note=pool_note,
        average_h12_on_k=True,
    )

    _plot_significance_single(
        r_vals,
        p_label_two,
        title=(
            "Pooled label-permutation significance (fusion vs AuNP)\n"
            f"all fusing vesicles | {pool_note} | two-sided Monte Carlo p, n_null={n_null}"
        ),
        output_path=figures_dir / f"ripley_h12_pvalues_label_permutation_{file_tag}.png",
        panel_note=pool_note,
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
                "n_fusion_vesicles": n_fusion_vesicles,
                "n_tomograms": n_tomograms,
                "n_clefts": n_clefts,
            }
        )

    offsets = _offset_keys_from_artifacts(artifacts)
    p_by_d: dict[str, np.ndarray] = {}
    panel_notes: dict[str, str] = {}
    fvc_panel_specs: list[dict[str, Any]] = []

    for offset_nm in offsets:
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
            fvc_panel_specs.append(
                {
                    "offset_nm": float(offset_nm),
                    "ctrl_curves": None,
                    "n_control_vesicles": 0,
                    "n_fusion_vesicles": len(fusion_vesicle_curves),
                }
            )
            continue

        h12_ctrl_lo, h12_ctrl_med, h12_ctrl_hi = _replicate_percentile_band(ctrl_curves)
        p_fusion_vs_ctrl = _unpaired_curve_pvalues(fusion_curves, ctrl_curves)
        p_by_d[f"d={int(offset_nm)} nm"] = p_fusion_vs_ctrl
        panel_notes[f"d={int(offset_nm)} nm"] = (
            f"n_fusion_vesicles={len(fusion_curves)}, n_control_vesicles={len(ctrl_curves)}, "
            f"{pool_note}"
        )
        fvc_panel_specs.append(
            {
                "offset_nm": float(offset_nm),
                "ctrl_curves": ctrl_curves,
                "n_control_vesicles": len(ctrl_curves),
                "n_fusion_vesicles": len(fusion_curves),
            }
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
                    "n_tomograms": n_tomograms,
                    "n_clefts": n_clefts,
                }
            )

    _save_fusion_vs_control_by_d_figures(
        figures_dir,
        f"ripley_h12_fusion_vs_controls_by_d_{file_tag}",
        r_vals=r_vals,
        fusion_vesicle_curves=fusion_vesicle_curves,
        panel_specs=fvc_panel_specs,
        ylabel="H₁₂(r)",
        suptitle_base=f"Pooled Ripley H₁₂: fusion vs controls across all fusing vesicles\n{pool_note}",
        refline=0.0,
        refline_label="H₁₂ = 0",
        sample_note=pool_note,
        average_h12_on_k=True,
    )

    if p_by_d:
        _plot_significance_panels(
            r_vals,
            p_by_d,
            title=(
                "Pooled fusion vs controls significance (Mann–Whitney on per-vesicle H₁₂)\n"
                f"all fusing vesicles combined | {pool_note}"
            ),
            output_path=figures_dir / f"ripley_h12_pvalues_fusion_vs_control_{file_tag}.png",
            panel_notes=panel_notes,
        )

    print(f"  Pooled Ripley H₁₂ (all fusing vesicles): {pool_note}")

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
    """Stack saved per-vesicle Ripley's O curves from all tomograms and synaptic clefts."""
    artifacts = _collect_ripley_vesicle_artifacts(tomo_paths, RIPLEY_O_VESICLE_CURVES_NPZ)
    if not artifacts:
        print("Skipping pooled Ripley's O: no saved vesicle-curve artifacts found.")
        return None

    figures_dir = output_dir / "figures" / "pooled_ripley"
    figures_dir.mkdir(parents=True, exist_ok=True)
    all_result_rows: list[dict] = []
    zone_name = "all"

    r_vals = np.asarray(artifacts[0]["r_vals"], dtype=float)
    fusion_parts = [_stack_nonempty_curves([np.asarray(a["fusion_vesicle_curves"])]) for a in artifacts]
    null_parts = [_stack_nonempty_curves([np.asarray(a["label_perm_null_curves"])]) for a in artifacts]
    obs_parts = [np.asarray(a["o_obs"], dtype=float) for a in artifacts]

    fusion_vesicle_curves = _stack_nonempty_curves(fusion_parts)
    perm_curves = _stack_nonempty_curves(null_parts)
    if fusion_vesicle_curves.size == 0 or perm_curves.size == 0:
        print("Skipping pooled Ripley's O: empty vesicle curves.")
        return None

    n_fusion_vesicles, n_tomograms, n_clefts = _ripley_artifact_pool_summary(artifacts)
    pool_note = (
        f"n_fusion_vesicles={n_fusion_vesicles}, n_tomograms={n_tomograms}, "
        f"n_clefts={n_clefts}"
    )

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
                "n_fusion_vesicles": n_fusion_vesicles,
                "n_tomograms": n_tomograms,
                "n_clefts": n_clefts,
            }
        )

    _save_label_perm_ripley_figures(
        figures_dir,
        f"ripley_o_label_permutation_{file_tag}",
        r_vals=r_vals,
        null_curves=perm_curves,
        obs_curves=fusion_vesicle_curves,
        n_perm=n_null,
        n_obs_replicates=len(fusion_vesicle_curves),
        ylabel="Ripley's O(r) [membrain-stats geodesic]",
        title_prefix=(
            "Pooled label-permutation null: Ripley's O fusion vs AuNP\n"
            f"all fusing vesicles | {pool_note}"
        ),
        refline=1.0,
        refline_label="CSR (O=1)",
        sample_note=pool_note,
    )

    _plot_significance_single(
        r_vals,
        p_label_two,
        title=(
            "Pooled Ripley's O label-permutation significance\n"
            f"all fusing vesicles | {pool_note}"
        ),
        output_path=figures_dir / f"ripley_o_pvalues_label_permutation_{file_tag}.png",
        panel_note=f"{pool_note}, n_null={n_null}",
    )

    offsets = _offset_keys_from_artifacts(artifacts)
    p_by_d: dict[str, np.ndarray] = {}
    fvc_panel_specs: list[dict[str, Any]] = []

    for offset_nm in offsets:
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
            fvc_panel_specs.append(
                {
                    "offset_nm": float(offset_nm),
                    "ctrl_curves": None,
                    "n_control_vesicles": 0,
                    "n_fusion_vesicles": len(fusion_vesicle_curves),
                }
            )
            continue

        o_ctrl_lo, o_ctrl_med, o_ctrl_hi = _replicate_percentile_band(ctrl_curves)
        p_fusion_vs_ctrl = _unpaired_curve_pvalues(fusion_curves, ctrl_curves)
        p_by_d[f"d={int(offset_nm)} nm"] = p_fusion_vs_ctrl
        fvc_panel_specs.append(
            {
                "offset_nm": float(offset_nm),
                "ctrl_curves": ctrl_curves,
                "n_control_vesicles": len(ctrl_curves),
                "n_fusion_vesicles": len(fusion_curves),
            }
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
                    "n_tomograms": n_tomograms,
                    "n_clefts": n_clefts,
                }
            )

    _save_fusion_vs_control_by_d_figures(
        figures_dir,
        f"ripley_o_fusion_vs_controls_by_d_{file_tag}",
        r_vals=r_vals,
        fusion_vesicle_curves=fusion_vesicle_curves,
        panel_specs=fvc_panel_specs,
        ylabel="Ripley's O(r)",
        suptitle_base=f"Pooled Ripley's O: fusion vs controls across all fusing vesicles\n{pool_note}",
        refline=1.0,
        refline_label="CSR (O=1)",
        sample_note=pool_note,
    )

    if p_by_d:
        _plot_significance_panels(
            r_vals,
            p_by_d,
            title=(
                "Pooled Ripley's O fusion vs controls significance\n"
                f"all fusing vesicles combined | {pool_note}"
            ),
            output_path=figures_dir / f"ripley_o_pvalues_fusion_vs_control_{file_tag}.png",
        )

    print(f"  Pooled Ripley's O (all fusing vesicles): {pool_note}")

    if not all_result_rows:
        return None
    out_df = pd.DataFrame(all_result_rows)
    out_df.to_csv(output_dir / f"ripley_o_membrain_postsynaptic_{file_tag}.csv", index=False)
    return out_df


def aggregate_fusion_point_pooled_visualizations(
    tomo_paths: Iterable[tuple[Any, Any, Any, str]],
    *,
    results_dir: Path | str = COMBINED_RESULTS_DIR,
) -> pd.DataFrame | None:
    """
    First post-AuNPs batch step: combine CSVs and write pooled Ripley/packing figures.

    Called at the end of the AuNPs analysis step after all tomograms in the CSV
    have been processed. Uses per-tomogram fusion-point outputs only (no zonogram MRCs).
    """
    results_dir = Path(results_dir)
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    fusion_df, h12_df, o_df = collect_per_tomogram_fusion_point_vs_aunp_density_tables(tomo_paths)
    if fusion_df.empty:
        print("No fusion-point vs AuNP density tables found to aggregate.")
        return None

    fusion_df.to_csv(results_dir / "fusion_point_vs_aunp_density_combined.csv", index=False)
    print(
        f"Combined fusion-point vs AuNP density table: "
        f"{len(fusion_df)} rows -> {results_dir / 'fusion_point_vs_aunp_density_combined.csv'}"
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
    if not real.empty and not ctrl.empty:
        print("Generating pooled packing summary plots across all tomograms...")
    plot_results(
        fusion_df,
        figures_dir,
        tomogram_path=None,
            filename_tag="pooled",
        )

    print(f"Pooled fusion-point vs AuNP density figures -> {figures_dir}")
    return fusion_df


def aggregate_fusion_point_per_tomogram_visualizations(
    tomo_paths: Iterable[tuple[Any, Any, Any, str]],
    *,
    results_dir: Path | str = COMBINED_RESULTS_DIR,
    fusion_df: pd.DataFrame | None = None,
) -> None:
    """
    After per-tomogram active zonograms: cross-zone packing plots with zonogram overlays.

    Run once the visualization loop has written active zonogram MRCs.
    """
    results_dir = Path(results_dir)
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    if fusion_df is None:
        combined_csv = results_dir / "fusion_point_vs_aunp_density_combined.csv"
        if combined_csv.is_file():
            fusion_df = pd.read_csv(combined_csv)
        else:
            fusion_df, _, _ = collect_per_tomogram_fusion_point_vs_aunp_density_tables(tomo_paths)

    if fusion_df is None or fusion_df.empty:
        print("No fusion-point vs AuNP density tables for per-tomogram figures.")
        return

    real = fusion_df[fusion_df["point_type"] == "fusion"].copy()
    ctrl = fusion_df[fusion_df["point_type"] == "control"].copy()
    if real.empty or ctrl.empty:
        return

    print("Generating per-tomogram packing summary plots (with zonogram overlays)...")
    for tomo, _set_name, _cleft_ids, alignment_dir in tomo_paths:
        tomogram_path = Path(tomo)
        tomogram_name = tomogram_path.name
        sub_df = fusion_df[fusion_df["tomogram_name"] == tomogram_name]
        if sub_df.empty:
            continue
        plot_results(
            sub_df,
            figures_dir,
                tomogram_path=tomogram_path,
                alignment_dir=alignment_dir,
            filename_tag=tomogram_name,
        )

    print(f"Per-tomogram fusion-point vs AuNP density figures -> {figures_dir}")


def aggregate_fusion_point_vs_aunp_density_visualizations(
    tomo_paths: Iterable[tuple[Any, Any, Any, str]],
    *,
    results_dir: Path | str = COMBINED_RESULTS_DIR,
) -> None:
    """Run pooled then per-tomogram aggregation (single-call convenience wrapper)."""
    fusion_df = aggregate_fusion_point_pooled_visualizations(tomo_paths, results_dir=results_dir)
    aggregate_fusion_point_per_tomogram_visualizations(
        tomo_paths,
        results_dir=results_dir,
        fusion_df=fusion_df,
    )


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
        "--include-close",
        action="store_true",
        help="Include close vesicles as well as fusing (default: fusing only)",
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
    parser.add_argument(
        "--aunp-pick-star-pattern",
        type=str,
        default=None,
        help=(
            "Per-AZ AuNP pick STAR filename pattern with one '*' for the synaptic cleft index "
            f"(default: {DEFAULT_AUNP_PICK_STAR_PATTERN!r})"
        ),
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
        fusing_only=not args.include_close,
    )
    print(f"Fusion-point vesicles: {len(fusion_rows)}", end="")
    if not args.include_close:
        print(" (fusing only)", end="")
    else:
        print(" (fusing + close)", end="")
    print()
    if not fusion_rows:
        raise SystemExit("No fusion points found.")

    membrane_az_pairs = import_presynaptic_membranes_and_clefts(
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
                aunp_pick_star_pattern=args.aunp_pick_star_pattern,
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
        "aunp_pick_star_pattern": normalize_aunp_pick_star_pattern(args.aunp_pick_star_pattern),
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
            aunp_pick_star_pattern=args.aunp_pick_star_pattern,
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
            aunp_pick_star_pattern=args.aunp_pick_star_pattern,
        )


if __name__ == "__main__":
    main()
