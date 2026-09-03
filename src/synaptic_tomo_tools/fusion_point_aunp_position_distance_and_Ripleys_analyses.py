"""
3D fusion-site vs AuNP distance tables and bivariate Ripley K₁₂ / L₁₂ analysis.

Uses raw tomogram coordinates (no postsynaptic projection) except label-permutation
sites that land on an original AuNP pool index — those are snapped to the presynaptic
synaptic cleft. Tangential 40 nm presynaptic shifts use the same placement rules as the
legacy fusion-point control code.

Ripley window: ``synaptic_cleft_az_hull`` — convex hull of all presynaptic + postsynaptic
synaptic-cleft surface points, always additionally restricted to the angle in-betweenness
region (matching the AZ-center and monomer/dimer Ripley analyses) since fusion sites and
AuNP positions are only meaningful relative to the space between the two membranes. Edge
correction uses the deterministic grid quadrature method (``_isotropic_edge_factors_grid``)
throughout, never Monte Carlo.

Distance and Ripley partner sets are reported separately for monomer-only, dimer-only,
and combined monomer+dimer picks when both STAR files are present. If only one STAR
file exists, analyses run for that kind only. When a single general AuNP pick STAR
pool is used (``use_single_pick_pool``), analyses run for the ``all`` subset.

Vesicle–AuNP distance tables are also written in a Prism-friendly form with one column
per vesicle (or vesicle×simulation), distances listed down the column, for fusing,
close, 40 nm-shifted, and label-permutation sites. Per-zone wide distance CSVs and
cumulative-count histograms use ``__{subset}`` suffixes (``monomer``, ``dimer``, ``both``,
or ``all``). Pooled column and cumulative-histogram tables use the same suffix under
``results/aunps/``.
"""

from __future__ import annotations

import json
from dataclasses import replace as dataclasses_replace
from pathlib import Path
from typing import Literal, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .alignment_utils import require_alignment_dir
from .aunps import enumerate_close_vesicle_fusion_points
from .fusion_point_geometry import (
    FUSION_POINT_AZ_MAX_SNAP_DISTANCE_NM,
    FUSION_POINT_SHIFT_OFFSET_NM,
    filter_fusion_rows_for_zone,
    load_presynaptic_az_points_for_zone,
    presynaptic_membrane_name_for_zone,
    sample_tangential_control_on_az,
    zone_name_for_presynaptic_membrane,
)
from .ripley_library import (
    AUNP_SUBSETS,
    AunpSubset,
    DEFAULT_ANALYSIS_SEED,
    DEFAULT_RIPLEY_R_MAX_NM,
    DEFAULT_RIPLEY_R_STEP_NM,
    MAD_MIN_NULL_CURVES,
    MAD_R_RANGES,
    RIPLEY_PERCENTILE_HI,
    RIPLEY_PERCENTILE_LO,
    RipleyWindow3D,
    _isotropic_edge_factors_grid,
    _percentile_band,
    _points_inside_hull,
    _prism_sd_envelope_columns,
    _ripley_r_grid,
    available_aunp_subsets,
    build_ripley_window_3d,
    build_window_grid_points,
    cross_k12_3d_isotropic,
    cross_k12_curves_per_focus,
    curves_matrix_to_wide_dataframe,
    derive_symmetric_k_l_g_families,
    g_shell_reliability_mask,
    load_monomer_dimer_aunps_for_zone,
    load_synaptic_cleft_cleft_points,
    mad_result_to_curves_dataframe,
    mad_result_to_summary_row,
    mean_l12_from_averaged_k12,
    mean_l_from_k_curves,
    pair_correlation_from_k_diff,
    plot_ripley_window_geometry_diagnostic,
    prism_sd_envelope_columns_from_averaged_k12,
    ripley_l12,
    run_mad_tests_over_r_ranges,
    subset_aunps,
)
from .vesicles import import_presynaptic_membranes_and_clefts

WindowMode = Literal["synaptic_cleft_az_hull"]
RIPLEY_WINDOW_MODES: tuple[WindowMode, ...] = ("synaptic_cleft_az_hull",)
ControlKind = Literal["close", "shift_40nm", "label_permutation"]

DEFAULT_NULL_REPLICATES_N = 100

# Deterministic grid-quadrature spacing for isotropic edge correction (see
# ``_isotropic_edge_factors_grid``) -- matches the AZ-center and monomer/dimer analyses.
FUSION_POINT_EDGE_GRID_SPACING_NM = 2.0

ANALYSES_SUBDIR = "fusion_point_aunp_analyses"

POOLED_RIPLEY_CURVES_CSV = Path("results/aunps/fusion_point_aunp_ripley_l12_curves.csv")
POOLED_RIPLEY_PRISM_CSV = Path("results/aunps/fusion_point_aunp_ripley_l12_prism_envelopes_pooled.csv")
POOLED_RIPLEY_FIGURES_DIR = Path("results/aunps/figures/fusion_point_aunp_ripley_l12_pooled")

POOLED_G_CURVES_CSV = Path("results/aunps/fusion_point_aunp_ripley_g12_curves.csv")
POOLED_G_PRISM_CSV = Path("results/aunps/fusion_point_aunp_ripley_g12_prism_envelopes_pooled.csv")
POOLED_G_FIGURES_DIR = Path("results/aunps/figures/fusion_point_aunp_ripley_g12_pooled")

# Bidirectional K/L/g families (12: fusion-type-as-foci, 21: AuNPs-as-foci, combined:
# intensity-weighted) -- one aggregate curve per replicate (fusing/close have exactly one;
# shift/permutation have up to ``DEFAULT_NULL_REPLICATES_N``), computed IN ADDITION TO the
# per-vesicle L12/g12 curves above (which stay focused on individual-vesicle inspection).
L_BIDIR_FAMILIES: tuple[str, ...] = ("l12", "l21", "l_combined")
G_BIDIR_FAMILIES: tuple[str, ...] = ("g12", "g21", "g_combined")
ALL_BIDIR_FAMILIES: tuple[str, ...] = L_BIDIR_FAMILIES + G_BIDIR_FAMILIES

POOLED_BIDIR_CURVES_CSV = Path("results/aunps/fusion_point_aunp_ripley_bidirectional_curves.csv")
POOLED_BIDIR_FIGURES_DIR = Path("results/aunps/figures/fusion_point_aunp_ripley_bidirectional_pooled")

# Prism-style distance tables: one column per vesicle/site, distances listed down the column.
POOLED_DIST_FUSING_COLUMNS_CSV = Path(
    "results/aunps/fusion_point_aunp_distances_fusing_columns_pooled.csv"
)
POOLED_DIST_CLOSE_COLUMNS_CSV = Path(
    "results/aunps/fusion_point_aunp_distances_close_columns_pooled.csv"
)
POOLED_DIST_SHIFT_COLUMNS_CSV = Path(
    "results/aunps/fusion_point_aunp_distances_40nm_shift_columns_pooled.csv"
)
POOLED_DIST_PERM_COLUMNS_CSV = Path(
    "results/aunps/fusion_point_aunp_distances_label_permutation_columns_pooled.csv"
)
# Mean cumulative AuNP count vs distance (1 nm bins, centers 0.5, 1.5, …).
POOLED_DIST_SHIFT_CUMHIST_CSV = Path(
    "results/aunps/fusion_point_aunp_distances_40nm_shift_cumulative_hist_pooled.csv"
)
POOLED_DIST_LABEL_PERM_CUMHIST_CSV = Path(
    "results/aunps/fusion_point_aunp_distances_label_permutation_cumulative_hist_pooled.csv"
)
DISTANCE_HIST_BIN_WIDTH_NM = 1.0
DISTANCE_COLUMNS_ONLY_STEMS = {
    "fusing": "distances_fusing_columns_only",
    "close": "distances_close_columns_only",
    "shift_40nm": "distances_40nm_shift_columns_only",
    "label_permutation": "distances_label_permutation_columns_only",
    "shift_40nm_cumhist": "distances_40nm_shift_cumulative_hist",
    "label_permutation_cumhist": "distances_label_permutation_cumulative_hist",
}

CONTROL_COMPARISONS: tuple[ControlKind, ...] = ("close", "shift_40nm", "label_permutation")
CONTROL_CURVE_TYPE: dict[ControlKind, str] = {
    "close": "close_per_vesicle",
    "shift_40nm": "shift_40nm_replicate",
    "label_permutation": "label_permutation_replicate",
}
FUSING_CURVE_TYPE = "fusing_per_vesicle"


def output_dir_for_zone(tomogram_path: Path, alignment_dir: str, zone_name: str) -> Path:
    alignment_dir = require_alignment_dir(alignment_dir)
    return (
        Path(tomogram_path)
        / alignment_dir
        / "STT_results"
        / "aunps"
        / ANALYSES_SUBDIR
        / zone_name
    )


def _fusion_xyz_from_rows(rows: Sequence[dict]) -> np.ndarray:
    if not rows:
        return np.zeros((0, 3), dtype=float)
    return np.array(
        [[r["fusion_point_x_nm"], r["fusion_point_y_nm"], r["fusion_point_z_nm"]] for r in rows],
        dtype=float,
    )


def _shift_sites_by_replicate(
    fusion_rows: Sequence[dict],
    membrane_az_pairs: dict,
    *,
    offset_nm: float,
    n_shifts: int,
    rng: np.random.Generator,
    max_snap_distance_nm: float,
) -> dict[int, dict[int, np.ndarray]]:
    """shift_replicate_id -> vesicle_id -> xyz (successful placements only)."""
    by_replicate: dict[int, dict[int, np.ndarray]] = {k: {} for k in range(int(n_shifts))}
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
        vesicle_id = int(fp["vesicle_id"])
        shift_id = 0
        attempts = 0
        max_attempts = int(n_shifts) * 40
        while shift_id < int(n_shifts) and attempts < max_attempts:
            attempts += 1
            shifted, _ = sample_tangential_control_on_az(
                fusion_xyz,
                az_xyz,
                az_tree,
                float(offset_nm),
                rng,
                max_snap_distance_nm=max_snap_distance_nm,
            )
            if shifted is None:
                continue
            by_replicate[shift_id][vesicle_id] = shifted.astype(float)
            shift_id += 1
    return by_replicate


def _label_permutation_sites(
    fusion_rows: Sequence[dict],
    aunp_coords: np.ndarray,
    pre_surface: np.ndarray,
    *,
    n_perm: int,
    rng: np.random.Generator,
) -> tuple[dict[int, dict[int, np.ndarray]], dict[int, list[np.ndarray]]]:
    """Return (perm -> vesicle -> xyz, perm -> all fusion-labeled query points)."""
    fusion_world = _fusion_xyz_from_rows(fusion_rows)
    n_fusion = len(fusion_world)
    if n_fusion == 0:
        return {}, {}

    pool = np.vstack([fusion_world, np.atleast_2d(aunp_coords)])
    n_pool = len(pool)
    if n_pool < n_fusion + 1:
        return {}, {}

    pre_tree = cKDTree(pre_surface) if len(pre_surface) else None
    vesicle_ids = [int(fp["vesicle_id"]) for fp in fusion_rows]

    per_vesicle: dict[int, dict[int, np.ndarray]] = {}
    pooled_sites: dict[int, list[np.ndarray]] = {}

    for perm_id in range(int(n_perm)):
        labels = np.zeros(n_pool, dtype=bool)
        labels[rng.choice(n_pool, n_fusion, replace=False)] = True
        fusion_labeled = np.flatnonzero(labels)

        queries: list[np.ndarray] = []
        ves_map: dict[int, np.ndarray] = {}
        for pool_idx in fusion_labeled:
            p = int(pool_idx)
            source = pool[p]
            if p < n_fusion:
                query = source.astype(float)
                ves_map[vesicle_ids[p]] = query
            else:
                if pre_tree is not None:
                    _, snap_i = pre_tree.query(source, k=1)
                    query = pre_surface[int(snap_i)].astype(float)
                else:
                    query = source.astype(float)
            queries.append(query)
        per_vesicle[perm_id] = ves_map
        pooled_sites[perm_id] = queries

    return per_vesicle, pooled_sites


def build_distance_wide_csv(
    aunp_coords: np.ndarray,
    aunp_meta: pd.DataFrame,
    site_columns: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Rows = AuNPs; columns = meta + one column per query site with 3D distance."""
    aunp_coords = np.atleast_2d(np.asarray(aunp_coords, dtype=float))
    base = aunp_meta.reset_index(drop=True).copy()
    if not site_columns:
        return base
    dist_data = {
        col_name: np.linalg.norm(
            aunp_coords - np.asarray(site_xyz, dtype=float).reshape(3), axis=1
        )
        for col_name, site_xyz in site_columns.items()
    }
    return pd.concat([base, pd.DataFrame(dist_data)], axis=1)


def build_distance_columns_only_dataframe(
    aunp_coords: np.ndarray,
    site_columns: dict[str, np.ndarray],
) -> pd.DataFrame:
    """
    Prism-friendly wide table: one column per vesicle/site, AuNP distances listed down
    that column (no AuNP metadata columns).
    """
    aunp_coords = np.atleast_2d(np.asarray(aunp_coords, dtype=float))
    if not site_columns:
        return pd.DataFrame()
    dist_data = {
        col_name: np.linalg.norm(
            aunp_coords - np.asarray(site_xyz, dtype=float).reshape(3), axis=1
        )
        for col_name, site_xyz in site_columns.items()
    }
    return pd.DataFrame(dist_data)


def _site_column_name(vesicle_name: str, suffix: str) -> str:
    safe = str(vesicle_name).replace(" ", "_")
    return f"{safe}__{suffix}"


def _global_site_column_name(
    tomogram_name: str,
    alignment_dir: str,
    zone_name: str,
    vesicle_name: str,
    suffix: str,
) -> str:
    """Unique column id across tomograms/zones for pooled vesicle-distance tables."""
    parts = [
        str(tomogram_name).replace(" ", "_"),
        str(alignment_dir).replace(" ", "_"),
        str(zone_name).replace(" ", "_"),
        str(vesicle_name).replace(" ", "_"),
        str(suffix).replace(" ", "_"),
    ]
    return "__".join(parts)


def _tomogram_zone_column_name(
    tomogram_name: str,
    alignment_dir: str,
    zone_name: str,
    suffix: str,
) -> str:
    """Pooled column id for one tomogram×zone (no per-vesicle suffix)."""
    parts = [
        str(tomogram_name).replace(" ", "_"),
        str(alignment_dir).replace(" ", "_"),
        str(zone_name).replace(" ", "_"),
        str(suffix).replace(" ", "_"),
    ]
    return "__".join(parts)


def _distances_to_single_site(aunp_coords: np.ndarray, site_xyz: np.ndarray) -> np.ndarray:
    aunp_coords = np.atleast_2d(np.asarray(aunp_coords, dtype=float))
    site = np.asarray(site_xyz, dtype=float).reshape(3)
    return np.linalg.norm(aunp_coords - site, axis=1)


def _min_distances_to_query_sites(
    aunp_coords: np.ndarray,
    queries: Sequence[np.ndarray],
) -> np.ndarray:
    """Per-AuNP minimum distance to any query site in one label-permutation replicate."""
    aunp_coords = np.atleast_2d(np.asarray(aunp_coords, dtype=float))
    if not queries:
        return np.full(len(aunp_coords), np.nan)
    q = np.vstack([np.asarray(query, dtype=float).reshape(3) for query in queries])
    if len(q) == 0:
        return np.full(len(aunp_coords), np.nan)
    d = np.linalg.norm(aunp_coords[:, None, :] - q[None, :, :], axis=2)
    return np.min(d, axis=1)


def _distance_hist_bin_centers(
    r_max_nm: float = DEFAULT_RIPLEY_R_MAX_NM,
    *,
    bin_width_nm: float = DISTANCE_HIST_BIN_WIDTH_NM,
) -> np.ndarray:
    """Bin centers 0.5, 1.5, … for 1 nm bins [0, 1), [1, 2), … up to ``r_max_nm``."""
    coarse = float(bin_width_nm)
    r_max = float(r_max_nm)
    n_bins = int(np.floor(r_max / coarse))
    if n_bins < 1:
        n_bins = 1
    return (np.arange(n_bins, dtype=float) + 0.5) * coarse


def _cumulative_aunp_count_at_bin_centers(
    distances: np.ndarray,
    r_centers: np.ndarray,
) -> np.ndarray:
    """
    Cumulative AuNP count at each bin center.

    Center ``c`` denotes the bin [c − Δ/2, c + Δ/2) (e.g. 0.5 → [0, 1) nm);
    the value is the number of AuNPs with distance strictly below the upper edge
    (c + Δ/2).
    """
    d = np.asarray(distances, dtype=float)
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return np.zeros(len(r_centers), dtype=float)
    half = 0.5 * float(DISTANCE_HIST_BIN_WIDTH_NM)
    upper_edges = np.asarray(r_centers, dtype=float) + half
    return np.array([float(np.sum(d < edge)) for edge in upper_edges], dtype=float)


def _mean_cumulative_histogram(
    distances_per_replicate: Sequence[np.ndarray],
    r_centers: np.ndarray,
) -> np.ndarray:
    """Mean cumulative AuNP-count curve across null replicates."""
    curves = [
        _cumulative_aunp_count_at_bin_centers(d, r_centers)
        for d in distances_per_replicate
        if len(np.asarray(d, dtype=float).reshape(-1)) > 0
    ]
    if not curves:
        return np.full(len(r_centers), np.nan)
    return np.nanmean(np.vstack(curves), axis=0)


def build_shift_cumulative_histogram_dataframe(
    aunp_coords: np.ndarray,
    fusing_rows: Sequence[dict],
    shift_by_replicate: dict[int, dict[int, np.ndarray]],
    *,
    tomogram_name: str,
    alignment_dir: str,
    zone_name: str,
    r_max_nm: float = DEFAULT_RIPLEY_R_MAX_NM,
) -> pd.DataFrame:
    """Per-vesicle mean cumulative AuNP-count histogram (40 nm shift null)."""
    r_centers = _distance_hist_bin_centers(r_max_nm)
    columns: dict[str, np.ndarray] = {}
    for fp in fusing_rows:
        vesicle_id = int(fp["vesicle_id"])
        dists_by_rep = [
            _distances_to_single_site(aunp_coords, site)
            for rep_id in sorted(shift_by_replicate)
            for site in [shift_by_replicate.get(rep_id, {}).get(vesicle_id)]
            if site is not None
        ]
        col = _global_site_column_name(
            tomogram_name,
            alignment_dir,
            zone_name,
            fp["vesicle_name"],
            "shift_cumhist",
        )
        columns[col] = _mean_cumulative_histogram(dists_by_rep, r_centers)
    if not columns:
        return pd.DataFrame({"r_nm": r_centers})
    return pd.DataFrame({"r_nm": r_centers, **columns})


def build_label_permutation_cumulative_histogram_dataframe(
    aunp_coords: np.ndarray,
    label_pooled: dict[int, list[np.ndarray]],
    *,
    tomogram_name: str,
    alignment_dir: str,
    zone_name: str,
    r_max_nm: float = DEFAULT_RIPLEY_R_MAX_NM,
) -> pd.DataFrame:
    """Per tomogram×zone mean cumulative AuNP-count histogram (label-perm null)."""
    r_centers = _distance_hist_bin_centers(r_max_nm)
    dists_by_rep = [
        _min_distances_to_query_sites(aunp_coords, label_pooled[perm_id])
        for perm_id in sorted(label_pooled)
        if label_pooled[perm_id]
    ]
    col = _tomogram_zone_column_name(
        tomogram_name, alignment_dir, zone_name, "label_perm_cumhist"
    )
    return pd.DataFrame(
        {
            "r_nm": r_centers,
            col: _mean_cumulative_histogram(dists_by_rep, r_centers),
        }
    )


def _zone_column_prefix(tomogram_name: str, alignment_dir: str, zone_name: str) -> str:
    return "__".join(
        [
            str(tomogram_name).replace(" ", "_"),
            str(alignment_dir).replace(" ", "_"),
            str(zone_name).replace(" ", "_"),
            "",
        ]
    )


def merge_distance_column_dataframes(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Horizontally combine vesicle-distance tables, padding shorter columns with NaN."""
    usable = [f for f in frames if f is not None and not f.empty]
    if not usable:
        return pd.DataFrame()
    max_rows = max(len(f) for f in usable)
    out: dict[str, np.ndarray] = {}
    for frame in usable:
        for col in frame.columns:
            values = frame[col].to_numpy(dtype=float)
            padded = np.full(max_rows, np.nan, dtype=float)
            padded[: len(values)] = values
            out[str(col)] = padded
    return pd.DataFrame(out)


def _read_optional_distance_columns_csv(path: Path) -> pd.DataFrame:
    """Read a vesicle-column distance CSV, treating missing/empty/corrupt files as empty."""
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    return df if not df.empty else pd.DataFrame()


def upsert_pooled_distance_columns_csv(
    path: Path,
    new_df: pd.DataFrame,
    *,
    drop_column_prefix: str | None = None,
) -> Path:
    """
    Merge ``new_df`` into a pooled vesicle-column CSV.

    If ``drop_column_prefix`` is set, existing columns starting with that prefix are
    removed first (supports re-running a tomogram/zone without duplicating columns).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    if path.is_file():
        old = _read_optional_distance_columns_csv(path)
        if drop_column_prefix and not old.empty:
            keep = [c for c in old.columns if not str(c).startswith(drop_column_prefix)]
            old = old[keep] if keep else pd.DataFrame()
        if not old.empty:
            frames.append(old)
    if new_df is not None and not new_df.empty:
        frames.append(new_df)
    merged = merge_distance_column_dataframes(frames)
    if merged.empty:
        if path.is_file():
            path.unlink()
        return path
    merged.to_csv(path, index=False)
    return path


def merge_cumulative_histogram_dataframes(
    frames: Sequence[pd.DataFrame],
) -> pd.DataFrame:
    """Horizontally merge cumulative-histogram tables sharing the same ``r_nm`` grid."""
    usable = [
        f
        for f in frames
        if f is not None and not f.empty and "r_nm" in f.columns
    ]
    if not usable:
        return pd.DataFrame()
    r_nm = usable[0]["r_nm"].to_numpy(dtype=float)
    for frame in usable[1:]:
        other = frame["r_nm"].to_numpy(dtype=float)
        if len(other) != len(r_nm) or not np.allclose(other, r_nm):
            raise ValueError("r_nm grids must match when merging cumulative histograms")
    out: dict[str, np.ndarray] = {"r_nm": r_nm}
    for frame in usable:
        for col in frame.columns:
            if col == "r_nm":
                continue
            out[str(col)] = frame[col].to_numpy(dtype=float)
    return pd.DataFrame(out)


def upsert_pooled_cumulative_histogram_csv(
    path: Path,
    new_df: pd.DataFrame,
    *,
    drop_column_prefix: str | None = None,
) -> Path:
    """Merge ``new_df`` into a pooled cumulative-histogram CSV (rows = ``r_nm``)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    if path.is_file():
        old = _read_optional_distance_columns_csv(path)
        if drop_column_prefix and not old.empty:
            keep = ["r_nm"] + [
                c
                for c in old.columns
                if c != "r_nm" and not str(c).startswith(drop_column_prefix)
            ]
            old = old[keep] if keep else pd.DataFrame()
        if not old.empty:
            frames.append(old)
    if new_df is not None and not new_df.empty:
        frames.append(new_df)
    merged = merge_cumulative_histogram_dataframes(frames)
    if merged.empty:
        if path.is_file():
            path.unlink()
        return path
    merged.to_csv(path, index=False)
    return path


def write_pooled_fusion_point_aunp_distance_column_csvs(
    search_roots: Sequence[Path] | str | Path,
) -> list[Path]:
    """
    Rebuild the four pooled vesicle-column distance CSVs from per-zone outputs.

    Each pooled file has one column per vesicle (or per vesicle×simulation), with AuNP
    distances listed down the column. Shorter columns are NaN-padded.
    """
    if isinstance(search_roots, (str, Path)):
        roots = [Path(search_roots)]
    else:
        roots = [Path(p) for p in search_roots]

    stem_to_out = {
        DISTANCE_COLUMNS_ONLY_STEMS["fusing"]: POOLED_DIST_FUSING_COLUMNS_CSV,
        DISTANCE_COLUMNS_ONLY_STEMS["close"]: POOLED_DIST_CLOSE_COLUMNS_CSV,
        DISTANCE_COLUMNS_ONLY_STEMS["shift_40nm"]: POOLED_DIST_SHIFT_COLUMNS_CSV,
        DISTANCE_COLUMNS_ONLY_STEMS["label_permutation"]: POOLED_DIST_PERM_COLUMNS_CSV,
    }
    cumhist_stem_to_out = {
        DISTANCE_COLUMNS_ONLY_STEMS["shift_40nm_cumhist"]: POOLED_DIST_SHIFT_CUMHIST_CSV,
        DISTANCE_COLUMNS_ONLY_STEMS["label_permutation_cumhist"]: POOLED_DIST_LABEL_PERM_CUMHIST_CSV,
    }
    written: list[Path] = []
    subsets_for_pooling = AUNP_SUBSETS
    for stem, out_path_base in stem_to_out.items():
        for subset in subsets_for_pooling:
            out_path = _pooled_subset_csv(out_path_base, subset)
            found: list[Path] = []
            for root in roots:
                found.extend(
                    root.glob(f"**/{ANALYSES_SUBDIR}/*/{stem}__{subset}.csv")
                )
            frames = [
                _read_optional_distance_columns_csv(p)
                for p in sorted(set(found))
                if p.is_file()
            ]
            frames = [f for f in frames if not f.empty]
            merged = merge_distance_column_dataframes(frames)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if merged.empty:
                if out_path.is_file():
                    out_path.unlink()
                print(
                    f"Pooled vesicle-column distances ({stem}__{subset}: no data) "
                    f"-> skipped {out_path}"
                )
                continue
            merged.to_csv(out_path, index=False)
            print(
                f"Pooled vesicle-column distances ({stem}__{subset}: {merged.shape[1]} columns, "
                f"{merged.shape[0]} rows) -> {out_path}"
            )
            written.append(out_path)
    for stem, out_path_base in cumhist_stem_to_out.items():
        for subset in subsets_for_pooling:
            out_path = _pooled_subset_csv(out_path_base, subset)
            found: list[Path] = []
            for root in roots:
                found.extend(
                    root.glob(f"**/{ANALYSES_SUBDIR}/*/{stem}__{subset}.csv")
                )
            frames = [
                _read_optional_distance_columns_csv(p)
                for p in sorted(set(found))
                if p.is_file()
            ]
            frames = [f for f in frames if not f.empty]
            merged = merge_cumulative_histogram_dataframes(frames)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if merged.empty:
                if out_path.is_file():
                    out_path.unlink()
                print(
                    f"Pooled cumulative histogram ({stem}__{subset}: no data) "
                    f"-> skipped {out_path}"
                )
                continue
            merged.to_csv(out_path, index=False)
            n_value_cols = merged.shape[1] - 1
            print(
                f"Pooled cumulative histogram ({stem}__{subset}: {n_value_cols} columns, "
                f"{merged.shape[0]} r rows) -> {out_path}"
            )
            written.append(out_path)
    return written


def _distance_csv_name(stem: str, subset: AunpSubset) -> str:
    return f"{stem}__{subset}.csv"


def _pooled_subset_csv(path: Path, subset: AunpSubset) -> Path:
    """Per-subset pooled CSV path (``…_pooled__monomer.csv``)."""
    path = Path(path)
    return path.parent / f"{path.stem}__{subset}{path.suffix}"


def _fusing_mean_curve(obs_curves: np.ndarray, r_vals: np.ndarray, *, k_curves: np.ndarray | None = None) -> np.ndarray:
    """Mean L₁₂ via average-on-K then ``ripley_l12`` (inverts L if ``k_curves`` omitted)."""
    if k_curves is not None and len(k_curves):
        return mean_l_from_k_curves(k_curves, r_vals)
    if len(obs_curves) == 0:
        return np.full(len(r_vals), np.nan)
    return mean_l12_from_averaged_k12(obs_curves, r_vals)


def build_ripley_l12_prism_envelope_table(
    *,
    zone_name: str,
    aunp_subset: AunpSubset,
    window: RipleyWindow3D,
    r_vals: np.ndarray,
    obs_curves: np.ndarray,
    control_curves_by_comparison: dict[str, np.ndarray],
    n_aunp_partners: int,
    obs_k_curves: np.ndarray | None = None,
    control_k_curves_by_comparison: dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    """
    Pre-aggregated mean curves and control envelopes for graphing (e.g. Prism).

    Percentile envelopes use 2.5–97.5% across replicate L₁₂ curves. Mean ± SD/SEM use
    average-on-K then ``ripley_l12`` once (primary columns only).

    One row per (control_comparison, r_nm). ``fusing_L12_mean`` is identical within
    each comparison group (K-averaged→L across fusing-vesicle curves).
    """
    control_k_curves_by_comparison = control_k_curves_by_comparison or {}
    fusing_sd = prism_sd_envelope_columns_from_averaged_k12(
        obs_curves, r_vals, prefix="fusing_L12", k12_curves=obs_k_curves
    )
    rows: list[dict] = []
    for comparison, control_curves in control_curves_by_comparison.items():
        ctrl_k = control_k_curves_by_comparison.get(comparison)
        if len(control_curves):
            ctrl_lo, _, ctrl_hi = _percentile_band(control_curves)
            ctrl_sd = prism_sd_envelope_columns_from_averaged_k12(
                control_curves, r_vals, prefix="control_L12", k12_curves=ctrl_k
            )
            n_control = int(len(control_curves))
        else:
            ctrl_lo = ctrl_hi = np.full(len(r_vals), np.nan)
            ctrl_sd = prism_sd_envelope_columns_from_averaged_k12(
                np.empty((0, len(r_vals))), r_vals, prefix="control_L12", k12_curves=ctrl_k
            )
            n_control = 0
        for i, r_nm in enumerate(r_vals):
            rows.append(
                {
                    "cleft_name": zone_name,
                    "aunp_subset": aunp_subset,
                    "window_mode": window.defining_mode,
                    "control_comparison": comparison,
                    "r_nm": float(r_nm),
                    "fusing_L12_mean": float(fusing_sd["fusing_L12_mean"][i]),
                    "fusing_L12_sd": float(fusing_sd["fusing_L12_sd"][i]),
                    "fusing_L12_sd_envelope_lo": float(fusing_sd["fusing_L12_sd_envelope_lo"][i]),
                    "fusing_L12_sd_envelope_hi": float(fusing_sd["fusing_L12_sd_envelope_hi"][i]),
                    "fusing_L12_sem": float(fusing_sd["fusing_L12_sem"][i]),
                    "fusing_L12_sem_envelope_lo": float(fusing_sd["fusing_L12_sem_envelope_lo"][i]),
                    "fusing_L12_sem_envelope_hi": float(fusing_sd["fusing_L12_sem_envelope_hi"][i]),
                    "control_L12_mean": float(ctrl_sd["control_L12_mean"][i]),
                    "control_L12_sd": float(ctrl_sd["control_L12_sd"][i]),
                    "control_L12_envelope_lo": float(ctrl_lo[i]),
                    "control_L12_envelope_hi": float(ctrl_hi[i]),
                    "control_L12_sd_envelope_lo": float(ctrl_sd["control_L12_sd_envelope_lo"][i]),
                    "control_L12_sd_envelope_hi": float(ctrl_sd["control_L12_sd_envelope_hi"][i]),
                    "control_L12_sem": float(ctrl_sd["control_L12_sem"][i]),
                    "control_L12_sem_envelope_lo": float(ctrl_sd["control_L12_sem_envelope_lo"][i]),
                    "control_L12_sem_envelope_hi": float(ctrl_sd["control_L12_sem_envelope_hi"][i]),
                    "n_fusing_curves": int(len(obs_curves)),
                    "n_control_curves": n_control,
                    "n_aunp_partners": int(n_aunp_partners),
                    "envelope_percentile_lo": float(RIPLEY_PERCENTILE_LO),
                    "envelope_percentile_hi": float(RIPLEY_PERCENTILE_HI),
                    "window_volume_nm3": float(window.volume_nm3),
                }
            )
    return pd.DataFrame(rows)


def build_ripley_g12_prism_envelope_table(
    *,
    zone_name: str,
    aunp_subset: AunpSubset,
    window: RipleyWindow3D,
    r_vals: np.ndarray,
    obs_curves: np.ndarray,
    control_curves_by_comparison: dict[str, np.ndarray],
    n_aunp_partners: int,
) -> pd.DataFrame:
    """
    Pre-aggregated mean pair-correlation (g₁₂) curves and control envelopes for graphing.

    Unlike L₁₂, g₁₂ is a linear function of K₁₂ (a finite difference — see
    ``pair_correlation_from_k_diff``), so averaging g curves directly is unbiased; no
    K-space detour is needed.
    """
    fusing_sd = _prism_sd_envelope_columns(obs_curves, r_vals, prefix="fusing_G12")
    rows: list[dict] = []
    for comparison, control_curves in control_curves_by_comparison.items():
        if len(control_curves):
            ctrl_lo, ctrl_mean, ctrl_hi = _percentile_band(control_curves)
            ctrl_sd = _prism_sd_envelope_columns(control_curves, r_vals, prefix="control_G12")
            n_control = int(len(control_curves))
        else:
            ctrl_lo = ctrl_mean = ctrl_hi = np.full(len(r_vals), np.nan)
            ctrl_sd = _prism_sd_envelope_columns(np.empty((0, len(r_vals))), r_vals, prefix="control_G12")
            n_control = 0
        for i, r_nm in enumerate(r_vals):
            rows.append(
                {
                    "cleft_name": zone_name,
                    "aunp_subset": aunp_subset,
                    "window_mode": window.defining_mode,
                    "control_comparison": comparison,
                    "r_nm": float(r_nm),
                    "fusing_G12_mean": float(fusing_sd["fusing_G12_mean"][i]),
                    "fusing_G12_sd": float(fusing_sd["fusing_G12_sd"][i]),
                    "fusing_G12_sd_envelope_lo": float(fusing_sd["fusing_G12_sd_envelope_lo"][i]),
                    "fusing_G12_sd_envelope_hi": float(fusing_sd["fusing_G12_sd_envelope_hi"][i]),
                    "fusing_G12_sem": float(fusing_sd["fusing_G12_sem"][i]),
                    "fusing_G12_sem_envelope_lo": float(fusing_sd["fusing_G12_sem_envelope_lo"][i]),
                    "fusing_G12_sem_envelope_hi": float(fusing_sd["fusing_G12_sem_envelope_hi"][i]),
                    "control_G12_mean": float(ctrl_mean[i]),
                    "control_G12_sd": float(ctrl_sd["control_G12_sd"][i]),
                    "control_G12_envelope_lo": float(ctrl_lo[i]),
                    "control_G12_envelope_hi": float(ctrl_hi[i]),
                    "control_G12_sd_envelope_lo": float(ctrl_sd["control_G12_sd_envelope_lo"][i]),
                    "control_G12_sd_envelope_hi": float(ctrl_sd["control_G12_sd_envelope_hi"][i]),
                    "control_G12_sem": float(ctrl_sd["control_G12_sem"][i]),
                    "control_G12_sem_envelope_lo": float(ctrl_sd["control_G12_sem_envelope_lo"][i]),
                    "control_G12_sem_envelope_hi": float(ctrl_sd["control_G12_sem_envelope_hi"][i]),
                    "n_fusing_curves": int(len(obs_curves)),
                    "n_control_curves": n_control,
                    "n_aunp_partners": int(n_aunp_partners),
                    "envelope_percentile_lo": float(RIPLEY_PERCENTILE_LO),
                    "envelope_percentile_hi": float(RIPLEY_PERCENTILE_HI),
                    "window_volume_nm3": float(window.volume_nm3),
                }
            )
    return pd.DataFrame(rows)


def _plot_g_control_comparison(
    r_vals: np.ndarray,
    obs_curves: np.ndarray,
    control_curves: np.ndarray,
    *,
    output_path: Path,
    title: str,
    ylabel: str = "Pair correlation g₁₂(r)",
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    obs_mean = np.nanmean(obs_curves, axis=0) if len(obs_curves) else np.full(len(r_vals), np.nan)
    ax.plot(r_vals, obs_mean, color="C3", lw=2.2, label="Fusing mean", zorder=5)
    if len(control_curves):
        lo, ctrl_mean, hi = _percentile_band(control_curves)
        ax.fill_between(r_vals, lo, hi, color="0.75", alpha=0.55, label="Control 95% envelope", zorder=2)
        ax.plot(r_vals, ctrl_mean, color="0.45", lw=2.0, label="Control mean", zorder=3)
    ax.axhline(1.0, color="k", ls="--", lw=0.8, alpha=0.6)
    ax.set_xlim(0.0, float(r_vals[-1]) if len(r_vals) else DEFAULT_RIPLEY_R_MAX_NM)
    ax.set_xlabel("r (nm)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_ripley_control_comparison(
    r_vals: np.ndarray,
    obs_curves: np.ndarray,
    control_curves: np.ndarray,
    *,
    output_path: Path,
    title: str,
    ylabel: str = "Ripley L₁₂(r) = (3K₁₂/4π)^(1/3) − r",
    obs_k_curves: np.ndarray | None = None,
    control_k_curves: np.ndarray | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    obs_sd = prism_sd_envelope_columns_from_averaged_k12(
        obs_curves, r_vals, prefix="fusing_L12", k12_curves=obs_k_curves
    )
    ax.plot(
        r_vals,
        obs_sd["fusing_L12_mean"],
        color="C3",
        lw=2.2,
        label="Fusing mean",
        zorder=5,
    )
    if len(obs_curves) > 1:
        ax.plot(
            r_vals,
            obs_sd["fusing_L12_sd_envelope_lo"],
            color="C3",
            lw=0.9,
            ls=":",
            alpha=0.8,
            label="Fusing ±SD",
            zorder=4,
        )
        ax.plot(
            r_vals,
            obs_sd["fusing_L12_sd_envelope_hi"],
            color="C3",
            lw=0.9,
            ls=":",
            alpha=0.8,
            zorder=4,
        )

    if len(control_curves):
        lo, _, hi = _percentile_band(control_curves)
        ctrl_sd = prism_sd_envelope_columns_from_averaged_k12(
            control_curves, r_vals, prefix="control_L12", k12_curves=control_k_curves
        )
        ax.fill_between(r_vals, lo, hi, color="0.75", alpha=0.55, label="Control 95% envelope", zorder=2)
        ax.plot(
            r_vals,
            ctrl_sd["control_L12_mean"],
            color="0.45",
            lw=2.0,
            label="Control mean",
            zorder=3,
        )
        ax.plot(
            r_vals,
            ctrl_sd["control_L12_sd_envelope_lo"],
            color="0.45",
            lw=0.9,
            ls=":",
            alpha=0.8,
            label="Control ±SD",
            zorder=3,
        )
        ax.plot(
            r_vals,
            ctrl_sd["control_L12_sd_envelope_hi"],
            color="0.45",
            lw=0.9,
            ls=":",
            alpha=0.8,
            zorder=3,
        )

    ax.axhline(0.0, color="k", ls="--", lw=0.8, alpha=0.6)
    ax.set_xlim(0.0, float(r_vals[-1]) if len(r_vals) else DEFAULT_RIPLEY_R_MAX_NM)
    ax.set_xlabel("r (nm)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _k_curves_per_focus_grid(
    points: np.ndarray,
    aunp_coords: np.ndarray,
    r_vals: np.ndarray,
    window: RipleyWindow3D,
    grid_points: np.ndarray,
    edge_grid_spacing_nm: float,
) -> np.ndarray:
    """Per-focus K₁₂ curves (one row per point in ``points``) via grid edge correction."""
    points = np.atleast_2d(np.asarray(points, dtype=float))
    if len(points) == 0:
        return np.empty((0, len(r_vals)))
    if len(aunp_coords) == 0 or window.volume_nm3 <= 0:
        return np.full((len(points), len(r_vals)), np.nan)
    edge_factors = _isotropic_edge_factors_grid(points, r_vals, grid_points, edge_grid_spacing_nm)
    return cross_k12_curves_per_focus(points, aunp_coords, r_vals, window, edge_factors=edge_factors)


def _k_curve_pooled_grid(
    points: np.ndarray,
    aunp_coords: np.ndarray,
    r_vals: np.ndarray,
    window: RipleyWindow3D,
    grid_points: np.ndarray,
    edge_grid_spacing_nm: float,
) -> np.ndarray:
    """Single aggregate K₁₂ curve pooling every point in ``points`` as foci, via grid edge correction."""
    points = np.atleast_2d(np.asarray(points, dtype=float))
    edge_factors = _isotropic_edge_factors_grid(points, r_vals, grid_points, edge_grid_spacing_nm)
    return cross_k12_3d_isotropic(
        points, aunp_coords, r_vals, window, np.random.default_rng(0), edge_factors=edge_factors
    )


def _g_reliability_mask_for_foci(
    foci_groups: Sequence[np.ndarray],
    grid_points: np.ndarray,
    r_vals: np.ndarray,
    *,
    bin_width_nm: float,
) -> np.ndarray:
    """``g_shell_reliability_mask`` over every non-empty foci set actually used for a
    curve-type's K (e.g. every label-permutation replicate's pooled points), summed —
    same "sum across all foci used" principle as the AZ-center/monomer-dimer analyses."""
    parts = [np.atleast_2d(np.asarray(f, dtype=float)) for f in foci_groups if len(f)]
    if not parts:
        return np.ones(len(r_vals), dtype=bool)
    return g_shell_reliability_mask(
        np.vstack(parts), grid_points, r_vals, bin_width_nm=bin_width_nm
    )


def _g_from_k(k_curves: np.ndarray, r_vals: np.ndarray, *, bin_width_nm: float, unreliable: np.ndarray) -> np.ndarray:
    k_curves = np.atleast_2d(np.asarray(k_curves, dtype=float))
    if k_curves.size == 0:
        return np.empty((k_curves.shape[0], len(unreliable)))
    pcf = np.atleast_2d(pair_correlation_from_k_diff(k_curves, r_vals, bin_width_nm=bin_width_nm)["pcf"])
    return np.where(unreliable[None, :], np.nan, pcf)


def run_ripley_for_zone_window(
    *,
    zone_name: str,
    aunp_subset: AunpSubset,
    window: RipleyWindow3D,
    aunp_coords: np.ndarray,
    fusing_rows: Sequence[dict],
    close_rows: Sequence[dict],
    shift_by_replicate: dict[int, dict[int, np.ndarray]],
    label_perm_pooled: dict[int, list[np.ndarray]],
    r_vals: np.ndarray,
    grid_points: np.ndarray,
    edge_grid_spacing_nm: float,
    figures_dir: Path | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fusing_xyz = _fusion_xyz_from_rows(fusing_rows)
    close_xyz = _fusion_xyz_from_rows(close_rows)
    r_step_nm = float(r_vals[1] - r_vals[0]) if len(r_vals) > 1 else float(r_vals[0])

    def per_focus_k(points: np.ndarray) -> np.ndarray:
        return _k_curves_per_focus_grid(
            points, aunp_coords, r_vals, window, grid_points, edge_grid_spacing_nm
        )

    obs_k = per_focus_k(fusing_xyz)
    close_k = per_focus_k(close_xyz)

    shift_foci_all: list[np.ndarray] = []
    shift_k_list: list[np.ndarray] = []
    for rep_id in sorted(shift_by_replicate):
        pts = list(shift_by_replicate[rep_id].values())
        if not pts:
            continue
        pts_arr = np.vstack(pts)
        shift_foci_all.append(pts_arr)
        shift_k_list.append(
            _k_curve_pooled_grid(pts_arr, aunp_coords, r_vals, window, grid_points, edge_grid_spacing_nm)
        )
    shift_k = np.vstack(shift_k_list) if shift_k_list else np.empty((0, len(r_vals)))

    perm_foci_all: list[np.ndarray] = []
    perm_k_list: list[np.ndarray] = []
    for perm_id in sorted(label_perm_pooled):
        pts = label_perm_pooled[perm_id]
        if not pts:
            continue
        pts_arr = np.vstack(pts)
        perm_foci_all.append(pts_arr)
        perm_k_list.append(
            _k_curve_pooled_grid(pts_arr, aunp_coords, r_vals, window, grid_points, edge_grid_spacing_nm)
        )
    perm_k = np.vstack(perm_k_list) if perm_k_list else np.empty((0, len(r_vals)))

    obs_curves = ripley_l12(obs_k, r_vals)
    close_curves = ripley_l12(close_k, r_vals)
    shift_curves = ripley_l12(shift_k, r_vals)
    perm_curves = ripley_l12(perm_k, r_vals)

    # Pair correlation (g₁₂) as a finite difference of the same edge-corrected K curves,
    # NaN-masked wherever the window quadrature grid doesn't support that shell well
    # enough from the foci actually used (see ``g_shell_reliability_mask``).
    obs_g_unreliable = _g_reliability_mask_for_foci([fusing_xyz], grid_points, r_vals, bin_width_nm=r_step_nm)
    close_g_unreliable = _g_reliability_mask_for_foci([close_xyz], grid_points, r_vals, bin_width_nm=r_step_nm)
    shift_g_unreliable = _g_reliability_mask_for_foci(shift_foci_all, grid_points, r_vals, bin_width_nm=r_step_nm)
    perm_g_unreliable = _g_reliability_mask_for_foci(perm_foci_all, grid_points, r_vals, bin_width_nm=r_step_nm)

    obs_g = _g_from_k(obs_k, r_vals, bin_width_nm=r_step_nm, unreliable=obs_g_unreliable)
    close_g = _g_from_k(close_k, r_vals, bin_width_nm=r_step_nm, unreliable=close_g_unreliable)
    shift_g = _g_from_k(shift_k, r_vals, bin_width_nm=r_step_nm, unreliable=shift_g_unreliable)
    perm_g = _g_from_k(perm_k, r_vals, bin_width_nm=r_step_nm, unreliable=perm_g_unreliable)
    r_g_vals = pair_correlation_from_k_diff(
        np.zeros((1, len(r_vals))), r_vals, bin_width_nm=r_step_nm
    )["r_mid_nm"]

    mode_tag = window.defining_mode
    subset_tag = aunp_subset
    if figures_dir is not None:
        _plot_ripley_control_comparison(
            r_vals,
            obs_curves,
            close_curves,
            output_path=figures_dir / f"ripley_l12_{mode_tag}_{subset_tag}_vs_close.png",
            title=(
                f"{zone_name} | window={mode_tag} | AuNPs={subset_tag}\n"
                "Fusing vs close-vesicle fusion sites"
            ),
            obs_k_curves=obs_k,
            control_k_curves=close_k,
        )
        _plot_ripley_control_comparison(
            r_vals,
            obs_curves,
            shift_curves,
            output_path=figures_dir / f"ripley_l12_{mode_tag}_{subset_tag}_vs_40nm_shift.png",
            title=(
                f"{zone_name} | window={mode_tag} | AuNPs={subset_tag}\n"
                "Fusing vs 40 nm tangential shifts"
            ),
            obs_k_curves=obs_k,
            control_k_curves=shift_k,
        )
        _plot_ripley_control_comparison(
            r_vals,
            obs_curves,
            perm_curves,
            output_path=figures_dir / f"ripley_l12_{mode_tag}_{subset_tag}_vs_label_permutation.png",
            title=(
                f"{zone_name} | window={mode_tag} | AuNPs={subset_tag}\n"
                "Fusing vs label-permutation null"
            ),
            obs_k_curves=obs_k,
            control_k_curves=perm_k,
        )
        _plot_g_control_comparison(
            r_g_vals,
            obs_g,
            close_g,
            output_path=figures_dir / f"ripley_g12_{mode_tag}_{subset_tag}_vs_close.png",
            title=(
                f"{zone_name} | window={mode_tag} | AuNPs={subset_tag}\n"
                "Fusing vs close-vesicle fusion sites"
            ),
        )
        _plot_g_control_comparison(
            r_g_vals,
            obs_g,
            shift_g,
            output_path=figures_dir / f"ripley_g12_{mode_tag}_{subset_tag}_vs_40nm_shift.png",
            title=(
                f"{zone_name} | window={mode_tag} | AuNPs={subset_tag}\n"
                "Fusing vs 40 nm tangential shifts (100 replicates)"
            ),
        )
        _plot_g_control_comparison(
            r_g_vals,
            obs_g,
            perm_g,
            output_path=figures_dir / f"ripley_g12_{mode_tag}_{subset_tag}_vs_label_permutation.png",
            title=(
                f"{zone_name} | window={mode_tag} | AuNPs={subset_tag}\n"
                "Fusing vs label-permutation null (100 replicates)"
            ),
        )

    # MAD of zone-mean fusing L₁₂ vs each control null (only if that null has ≥1000 curves).
    # Reported for full r-grid and restricted 30–50 nm windows.
    obs_mean = (
        np.nanmean(obs_curves, axis=0)
        if len(obs_curves)
        else np.full(len(r_vals), np.nan)
    )
    mad_summary_rows: list[dict] = []
    mad_curve_frames: list[pd.DataFrame] = []
    mad_by_range: dict[str, list[dict]] = {label: [] for label, _, _ in MAD_R_RANGES}
    for null_name, null_curves in (
        ("close", close_curves),
        ("shift_40nm", shift_curves),
        ("label_permutation", perm_curves),
    ):
        for mad in run_mad_tests_over_r_ranges(
            obs_mean,
            null_curves,
            r_vals,
            null_name=null_name,
            min_null_curves=MAD_MIN_NULL_CURVES,
        ):
            mad_by_range.setdefault(str(mad["r_range"]), []).append(mad)
            mad_summary_rows.append(
                mad_result_to_summary_row(
                    mad,
                    extra_cols={
                        "cleft_name": zone_name,
                        "aunp_subset": aunp_subset,
                        "window_mode": mode_tag,
                    },
                )
            )
            mad_curves = mad_result_to_curves_dataframe(mad, r_vals, observed=obs_mean)
            mad_curves.insert(0, "cleft_name", zone_name)
            mad_curves.insert(1, "aunp_subset", aunp_subset)
            mad_curves.insert(2, "window_mode", mode_tag)
            mad_curve_frames.append(mad_curves)

    if figures_dir is not None:
        for r_range, mad_results_for_plot in mad_by_range.items():
            if not mad_results_for_plot:
                continue
            suffix = "" if r_range == "full" else f"_{r_range.replace('-', '_')}"
            n_panels = len(mad_results_for_plot)
            fig, axes = plt.subplots(2, n_panels, figsize=(4.2 * n_panels, 7.0), squeeze=False)
            for col, mad in enumerate(mad_results_for_plot):
                ax_raw = axes[0, col]
                ax_norm = axes[1, col]
                null_label = str(mad["null_name"])
                r_use = np.asarray(mad.get("r_vals", []), dtype=float)
                obs_use = np.asarray(mad.get("observed", []), dtype=float).reshape(-1)
                if len(r_use) and len(obs_use) == len(r_use):
                    ax_raw.plot(r_use, obs_use, color="C3", lw=2.0, label="Fusing mean L₁₂")
                if mad["status"] == "ok":
                    ax_raw.plot(r_use, mad["null_mean"], color="0.35", lw=1.5, label="Null mean")
                    ax_raw.fill_between(
                        r_use, mad["ce_lo"], mad["ce_hi"], color="0.75", alpha=0.55, label="99% CE"
                    )
                    ax_norm.plot(r_use, mad["normalized_obs"], color="C3", lw=2.0)
                    ax_norm.axhline(1.0, color="0.4", ls="--", lw=1.0)
                    ax_norm.axhline(-1.0, color="0.4", ls="--", lw=1.0)
                    reject_txt = "reject H0" if mad["rejects_null"] else "fail to reject H0"
                    ax_raw.set_title(
                        f"{null_label}\nT={mad['T_obs']:.3g}/Tcrit={mad['T_critical']:.3g} "
                        f"(p={mad['p_mad']:.3g}; {reject_txt})",
                        fontsize=9,
                    )
                else:
                    ax_raw.set_title(
                        f"{null_label}\nskipped (n={mad['n_null_curves']} < {mad['min_null_curves']})",
                        fontsize=9,
                    )
                ax_raw.axhline(0.0, color="0.5", ls="--", lw=0.8)
                ax_raw.set_xlabel("r (nm)")
                ax_raw.set_ylabel("L₁₂(r)")
                ax_raw.legend(fontsize=7, loc="best")
                ax_norm.set_xlabel("r (nm)")
                ax_norm.set_ylabel("Normalized L₁₂")
                if len(r_use):
                    ax_raw.set_xlim(float(r_use[0]), float(r_use[-1]))
                    ax_norm.set_xlim(float(r_use[0]), float(r_use[-1]))
            fig.suptitle(
                f"{zone_name} | {mode_tag} | {subset_tag} | MAD {r_range} "
                f"(n≥{MAD_MIN_NULL_CURVES})",
                fontsize=11,
            )
            fig.tight_layout()
            fig.savefig(
                figures_dir / f"ripley_l12_{mode_tag}_{subset_tag}_mad_vs_nulls{suffix}.png",
                dpi=150,
                bbox_inches="tight",
            )
            plt.close(fig)

    prism_df = build_ripley_l12_prism_envelope_table(
        zone_name=zone_name,
        aunp_subset=aunp_subset,
        window=window,
        r_vals=r_vals,
        obs_curves=obs_curves,
        control_curves_by_comparison={
            "close": close_curves,
            "shift_40nm": shift_curves,
            "label_permutation": perm_curves,
        },
        n_aunp_partners=len(aunp_coords),
        obs_k_curves=obs_k,
        control_k_curves_by_comparison={
            "close": close_k,
            "shift_40nm": shift_k,
            "label_permutation": perm_k,
        },
    )
    g_prism_df = build_ripley_g12_prism_envelope_table(
        zone_name=zone_name,
        aunp_subset=aunp_subset,
        window=window,
        r_vals=r_g_vals,
        obs_curves=obs_g,
        control_curves_by_comparison={
            "close": close_g,
            "shift_40nm": shift_g,
            "label_permutation": perm_g,
        },
        n_aunp_partners=len(aunp_coords),
    )

    rows: list[dict] = []
    for curve_type, curves in (
        ("fusing_per_vesicle", obs_curves),
        ("close_per_vesicle", close_curves),
        ("shift_40nm_replicate", shift_curves),
        ("label_permutation_replicate", perm_curves),
    ):
        for i, curve in enumerate(curves):
            for r_val, l_val in zip(r_vals, curve):
                rows.append(
                    {
                        "cleft_name": zone_name,
                        "aunp_subset": aunp_subset,
                        "window_mode": mode_tag,
                        "curve_type": curve_type,
                        "replicate_index": int(i),
                        "r_nm": float(r_val),
                        "ripley_l12": float(l_val),
                        "n_aunp_partners": int(len(aunp_coords)),
                        "window_volume_nm3": float(window.volume_nm3),
                    }
                )
    curves_df = pd.DataFrame(rows)

    g_rows: list[dict] = []
    for curve_type, curves in (
        ("fusing_per_vesicle", obs_g),
        ("close_per_vesicle", close_g),
        ("shift_40nm_replicate", shift_g),
        ("label_permutation_replicate", perm_g),
    ):
        for i, curve in enumerate(curves):
            for r_val, g_val in zip(r_g_vals, curve):
                g_rows.append(
                    {
                        "cleft_name": zone_name,
                        "aunp_subset": aunp_subset,
                        "window_mode": mode_tag,
                        "curve_type": curve_type,
                        "replicate_index": int(i),
                        "r_nm": float(r_val),
                        "ripley_g12": float(g_val),
                        "n_aunp_partners": int(len(aunp_coords)),
                        "window_volume_nm3": float(window.volume_nm3),
                    }
                )
    g_curves_df = pd.DataFrame(g_rows)

    mad_summary_df = pd.DataFrame(mad_summary_rows)
    mad_curves_df = (
        pd.concat(mad_curve_frames, ignore_index=True) if mad_curve_frames else pd.DataFrame()
    )
    return curves_df, prism_df, mad_summary_df, mad_curves_df, g_curves_df, g_prism_df


def _plot_bidirectional_family(
    r_vals: np.ndarray,
    family: str,
    obs_curves: np.ndarray,
    control_curves: np.ndarray,
    *,
    output_path: Path,
    title: str,
) -> None:
    if family.startswith("g"):
        _plot_g_control_comparison(
            r_vals, obs_curves, control_curves, output_path=output_path, title=title,
            ylabel=f"Pair correlation {family}(r)",
        )
    else:
        _plot_ripley_control_comparison(
            r_vals, obs_curves, control_curves, output_path=output_path, title=title,
            ylabel=f"Ripley {family}(r)",
        )


def run_bidirectional_ripley_for_zone_window(
    *,
    zone_name: str,
    aunp_subset: AunpSubset,
    window: RipleyWindow3D,
    aunp_coords: np.ndarray,
    fusing_rows: Sequence[dict],
    close_rows: Sequence[dict],
    shift_by_replicate: dict[int, dict[int, np.ndarray]],
    label_perm_pooled: dict[int, list[np.ndarray]],
    r_vals: np.ndarray,
    grid_points: np.ndarray,
    edge_grid_spacing_nm: float,
    figures_dir: Path | None,
) -> pd.DataFrame:
    """
    Bidirectional K/L/g families (``ALL_BIDIR_FAMILIES``: 12 = fusion-type-as-foci, matching
    ``run_ripley_for_zone_window``'s per-vesicle direction; 21 = AuNPs-as-foci; combined =
    intensity-weighted, Lotwick & Silverman 1982), one aggregate curve per replicate (all
    query points in that replicate pooled as a single focus set, rather than per-vesicle —
    the natural aggregate counterpart needed for a genuine two-directional comparison,
    matching the monomer/dimer and AZ-center Ripley analyses). Computed IN ADDITION to
    ``run_ripley_for_zone_window``'s per-vesicle L12/g12 output, not a replacement for it.

    Each replicate keeps its own point count (``n1``) since shift/permutation replicates can
    vary in how many points they contain (e.g. shift placement can fail for some vesicles in
    some replicates), so ``derive_symmetric_k_l_g_families`` is called once per replicate
    rather than batched.
    """
    fusing_xyz = _fusion_xyz_from_rows(fusing_rows)
    close_xyz = _fusion_xyz_from_rows(close_rows)
    r_step_nm = float(r_vals[1] - r_vals[0]) if len(r_vals) > 1 else float(r_vals[0])
    mode_tag = window.defining_mode
    subset_tag = aunp_subset
    n_aunp = len(aunp_coords)

    aunp_edge_factors = (
        _isotropic_edge_factors_grid(aunp_coords, r_vals, grid_points, edge_grid_spacing_nm)
        if n_aunp else None
    )
    aunp_g_unreliable = (
        g_shell_reliability_mask(aunp_coords, grid_points, r_vals, bin_width_nm=r_step_nm)
        if n_aunp else np.ones(len(r_vals), dtype=bool)
    )

    def _families_for_points(pts: np.ndarray) -> dict[str, np.ndarray] | None:
        pts = np.atleast_2d(np.asarray(pts, dtype=float))
        n1 = len(pts)
        if n1 == 0 or n_aunp == 0:
            return None
        k12 = _k_curve_pooled_grid(pts, aunp_coords, r_vals, window, grid_points, edge_grid_spacing_nm)
        k21 = cross_k12_3d_isotropic(
            aunp_coords, pts, r_vals, window, np.random.default_rng(0), edge_factors=aunp_edge_factors
        )
        fam = derive_symmetric_k_l_g_families(k12, k21, n1, n_aunp, r_vals, g_bin_width_nm=r_step_nm)
        pts_g_unreliable = g_shell_reliability_mask(pts, grid_points, r_vals, bin_width_nm=r_step_nm)
        combined_g_unreliable = pts_g_unreliable | aunp_g_unreliable
        fam["g12"] = np.where(pts_g_unreliable, np.nan, fam["g12"])
        fam["g21"] = np.where(aunp_g_unreliable, np.nan, fam["g21"])
        fam["g_combined"] = np.where(combined_g_unreliable, np.nan, fam["g_combined"])
        return fam

    curve_type_points: dict[str, list[np.ndarray]] = {
        "fusing": [fusing_xyz] if len(fusing_xyz) else [],
        "close": [close_xyz] if len(close_xyz) else [],
        "shift_40nm": [
            np.vstack(list(shift_by_replicate[rep].values()))
            for rep in sorted(shift_by_replicate)
            if shift_by_replicate[rep]
        ],
        "label_permutation": [
            np.vstack(label_perm_pooled[perm])
            for perm in sorted(label_perm_pooled)
            if label_perm_pooled[perm]
        ],
    }

    families_by_curve_type: dict[str, dict[str, np.ndarray]] = {}
    rows: list[dict] = []
    for curve_type, points_list in curve_type_points.items():
        fam_curves: dict[str, list[np.ndarray]] = {fam: [] for fam in ALL_BIDIR_FAMILIES}
        for rep_idx, pts in enumerate(points_list):
            fam = _families_for_points(pts)
            if fam is None:
                continue
            for fam_name in ALL_BIDIR_FAMILIES:
                fam_curves[fam_name].append(fam[fam_name])
                for r_val, val in zip(r_vals, fam[fam_name]):
                    rows.append(
                        {
                            "cleft_name": zone_name,
                            "aunp_subset": aunp_subset,
                            "window_mode": mode_tag,
                            "curve_type": curve_type,
                            "family": fam_name,
                            "replicate_index": int(rep_idx),
                            "r_nm": float(r_val),
                            "value": float(val),
                            "n_query_points": int(len(np.atleast_2d(pts))),
                            "n_aunp_partners": int(n_aunp),
                            "window_volume_nm3": float(window.volume_nm3),
                        }
                    )
        families_by_curve_type[curve_type] = {
            fam_name: (np.vstack(curves) if curves else np.empty((0, len(r_vals))))
            for fam_name, curves in fam_curves.items()
        }

    if figures_dir is not None:
        obs = families_by_curve_type.get("fusing", {})
        for fam_name in ALL_BIDIR_FAMILIES:
            obs_curves = obs.get(fam_name, np.empty((0, len(r_vals))))
            if obs_curves.shape[0] == 0:
                continue
            for control_type in ("close", "shift_40nm", "label_permutation"):
                control_curves = families_by_curve_type.get(control_type, {}).get(
                    fam_name, np.empty((0, len(r_vals)))
                )
                _plot_bidirectional_family(
                    r_vals, fam_name, obs_curves, control_curves,
                    output_path=figures_dir / f"ripley_bidir_{fam_name}_{mode_tag}_{subset_tag}_vs_{control_type}.png",
                    title=(
                        f"{zone_name} | window={mode_tag} | AuNPs={subset_tag} | {fam_name}\n"
                        f"Fusing vs {control_type}"
                    ),
                )

    return pd.DataFrame(rows)


def _extract_curves_matrix(
    df: pd.DataFrame,
    curve_type: str,
    *,
    value_col: str = "ripley_l12",
) -> tuple[np.ndarray, np.ndarray]:
    """Pivot long Ripley table to ``(r_vals, curves)`` with one row per replicate curve."""
    sub = df.loc[df["curve_type"] == curve_type]
    if sub.empty:
        r_vals = np.sort(df["r_nm"].unique()) if "r_nm" in df.columns and not df.empty else np.array([])
        return r_vals, np.empty((0, len(r_vals)))

    r_vals = np.sort(sub["r_nm"].unique())
    n_r = len(r_vals)
    id_cols = ["tomogram_name", "alignment_dir", "cleft_name", "replicate_index"]
    for col in id_cols:
        if col not in sub.columns:
            sub = sub.copy()
            sub[col] = ""
    curves: list[np.ndarray] = []
    for _, grp in sub.groupby(id_cols, sort=False):
        grp = grp.sort_values("r_nm")
        if len(grp) != n_r:
            continue
        if not np.allclose(grp["r_nm"].to_numpy(dtype=float), r_vals):
            continue
        curves.append(grp[value_col].to_numpy(dtype=float))
    if not curves:
        return r_vals, np.empty((0, n_r))
    return r_vals, np.vstack(curves)


def build_pooled_ripley_l12_prism_envelope_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pooled mean curves and control envelopes across all tomograms/zones.

    ``*_mean`` / ``*_sd_*`` / ``*_sem_*`` are mean ± SD/SEM of K₁₂ then mapped once through
    ``ripley_l12``. Percentile envelopes (``control_L12_envelope_*``) remain on L of each
    replicate curve (identical under the monotone K→L transform).
    """
    if df.empty or "tomogram_name" not in df.columns:
        return pd.DataFrame()

    rows: list[dict] = []
    for subset in AUNP_SUBSETS:
        for window_mode in RIPLEY_WINDOW_MODES:
            sub_df = df[
                (df["aunp_subset"] == subset) & (df["window_mode"] == window_mode)
            ]
            if sub_df.empty:
                continue

            r_vals, obs_curves = _extract_curves_matrix(sub_df, FUSING_CURVE_TYPE)
            if len(obs_curves) == 0:
                continue

            fusing_sd = prism_sd_envelope_columns_from_averaged_k12(
                obs_curves, r_vals, prefix="fusing_L12"
            )
            n_tomograms = int(sub_df["tomogram_name"].nunique())
            n_zones = int(
                sub_df[["tomogram_name", "alignment_dir", "cleft_name"]]
                .drop_duplicates()
                .shape[0]
            )

            for comparison in CONTROL_COMPARISONS:
                _, control_curves = _extract_curves_matrix(
                    sub_df, CONTROL_CURVE_TYPE[comparison]
                )
                if len(control_curves):
                    ctrl_lo, _, ctrl_hi = _percentile_band(control_curves)
                    ctrl_sd = prism_sd_envelope_columns_from_averaged_k12(
                        control_curves, r_vals, prefix="control_L12"
                    )
                    n_control = int(len(control_curves))
                else:
                    ctrl_lo = ctrl_hi = np.full(len(r_vals), np.nan)
                    ctrl_sd = prism_sd_envelope_columns_from_averaged_k12(
                        np.empty((0, len(r_vals))), r_vals, prefix="control_L12"
                    )
                    n_control = 0

                for i, r_nm in enumerate(r_vals):
                    rows.append(
                        {
                            "aunp_subset": subset,
                            "window_mode": window_mode,
                            "control_comparison": comparison,
                            "r_nm": float(r_nm),
                            "fusing_L12_mean": float(fusing_sd["fusing_L12_mean"][i]),
                            "fusing_L12_sd": float(fusing_sd["fusing_L12_sd"][i]),
                            "fusing_L12_sd_envelope_lo": float(
                                fusing_sd["fusing_L12_sd_envelope_lo"][i]
                            ),
                            "fusing_L12_sd_envelope_hi": float(
                                fusing_sd["fusing_L12_sd_envelope_hi"][i]
                            ),
                            "fusing_L12_sem": float(fusing_sd["fusing_L12_sem"][i]),
                            "fusing_L12_sem_envelope_lo": float(
                                fusing_sd["fusing_L12_sem_envelope_lo"][i]
                            ),
                            "fusing_L12_sem_envelope_hi": float(
                                fusing_sd["fusing_L12_sem_envelope_hi"][i]
                            ),
                            "control_L12_mean": float(ctrl_sd["control_L12_mean"][i]),
                            "control_L12_sd": float(ctrl_sd["control_L12_sd"][i]),
                            "control_L12_envelope_lo": float(ctrl_lo[i]),
                            "control_L12_envelope_hi": float(ctrl_hi[i]),
                            "control_L12_sd_envelope_lo": float(
                                ctrl_sd["control_L12_sd_envelope_lo"][i]
                            ),
                            "control_L12_sd_envelope_hi": float(
                                ctrl_sd["control_L12_sd_envelope_hi"][i]
                            ),
                            "control_L12_sem": float(ctrl_sd["control_L12_sem"][i]),
                            "control_L12_sem_envelope_lo": float(
                                ctrl_sd["control_L12_sem_envelope_lo"][i]
                            ),
                            "control_L12_sem_envelope_hi": float(
                                ctrl_sd["control_L12_sem_envelope_hi"][i]
                            ),
                            "n_fusing_curves": int(len(obs_curves)),
                            "n_control_curves": n_control,
                            "n_tomograms": n_tomograms,
                            "n_clefts": n_zones,
                            "envelope_percentile_lo": float(RIPLEY_PERCENTILE_LO),
                            "envelope_percentile_hi": float(RIPLEY_PERCENTILE_HI),
                        }
                    )
    return pd.DataFrame(rows)


def plot_pooled_fusion_point_aunp_ripley_l12_visualizations(
    curves_csv: Path | str = POOLED_RIPLEY_CURVES_CSV,
    output_dir: Path | str = POOLED_RIPLEY_FIGURES_DIR,
    prism_csv: Path | str = POOLED_RIPLEY_PRISM_CSV,
) -> list[Path]:
    """
    Pool individual Ripley curves across all tomograms/zones.

    Writes PNGs (fusing mean vs control mean + 95% envelope) and a Prism CSV.
    """
    curves_csv = Path(curves_csv)
    output_dir = Path(output_dir)
    prism_csv = Path(prism_csv)
    if not curves_csv.is_file():
        print(f"No pooled Ripley curves CSV at {curves_csv}; skipping pooled outputs.")
        return []

    df = pd.read_csv(curves_csv)
    if df.empty:
        print("Pooled Ripley curves CSV is empty; skipping pooled outputs.")
        return []

    if "tomogram_name" not in df.columns:
        print("Pooled Ripley curves CSV missing tomogram_name; skipping pooled outputs.")
        return []

    prism_long = build_pooled_ripley_l12_prism_envelope_table(df)
    if prism_long.empty:
        print("No pooled Ripley L₁₂ envelope rows generated; skipping pooled outputs.")
        return []

    prism_csv.parent.mkdir(parents=True, exist_ok=True)
    prism_long.to_csv(prism_csv, index=False)
    print(
        f"Pooled fusion-point/AuNP Ripley L₁₂ Prism table "
        f"({len(prism_long)} rows) -> {prism_csv}"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for (subset, window_mode, comparison), grp in prism_long.groupby(
        ["aunp_subset", "window_mode", "control_comparison"],
        sort=False,
    ):
        sub_df = df[
            (df["aunp_subset"] == subset)
            & (df["window_mode"] == window_mode)
        ]
        r_vals = np.sort(grp["r_nm"].unique())
        _, obs_curves = _extract_curves_matrix(sub_df, FUSING_CURVE_TYPE)
        _, control_curves = _extract_curves_matrix(sub_df, CONTROL_CURVE_TYPE[comparison])  # type: ignore[index]
        if len(obs_curves) == 0 or len(control_curves) == 0:
            continue

        meta_row = grp.iloc[0]
        out_path = output_dir / f"ripley_l12_{window_mode}_{subset}_vs_{comparison}_pooled.png"
        _plot_ripley_control_comparison(
            r_vals,
            obs_curves,
            control_curves,
            output_path=out_path,
            title=(
                f"Pooled | window={window_mode} | AuNPs={subset} | vs {comparison}\n"
                f"{int(meta_row['n_tomograms'])} tomogram(s), "
                f"{int(meta_row['n_clefts'])} zone(s) | "
                f"{int(meta_row['n_fusing_curves'])} fusing curves, "
                f"{int(meta_row['n_control_curves'])} control curves"
            ),
        )
        written.append(out_path)

    if written:
        print(f"Pooled fusion-point/AuNP Ripley L₁₂ figures ({len(written)}) -> {output_dir}")
    else:
        print("No pooled Ripley L₁₂ figures written (missing curve groups in CSV).")
    return written


def build_pooled_ripley_g12_prism_envelope_table(df: pd.DataFrame) -> pd.DataFrame:
    """Pooled mean pair-correlation (g₁₂) curves and control envelopes across all
    tomograms/zones. See ``build_ripley_g12_prism_envelope_table`` for why no K-space
    pooling variant is needed here."""
    if df.empty or "tomogram_name" not in df.columns:
        return pd.DataFrame()

    rows: list[dict] = []
    for subset in AUNP_SUBSETS:
        for window_mode in RIPLEY_WINDOW_MODES:
            sub_df = df[
                (df["aunp_subset"] == subset) & (df["window_mode"] == window_mode)
            ]
            if sub_df.empty:
                continue

            r_vals, obs_curves = _extract_curves_matrix(
                sub_df, FUSING_CURVE_TYPE, value_col="ripley_g12"
            )
            if len(obs_curves) == 0:
                continue

            fusing_sd = _prism_sd_envelope_columns(obs_curves, r_vals, prefix="fusing_G12")
            n_tomograms = int(sub_df["tomogram_name"].nunique())
            n_zones = int(
                sub_df[["tomogram_name", "alignment_dir", "cleft_name"]]
                .drop_duplicates()
                .shape[0]
            )

            for comparison in CONTROL_COMPARISONS:
                _, control_curves = _extract_curves_matrix(
                    sub_df, CONTROL_CURVE_TYPE[comparison], value_col="ripley_g12"
                )
                if len(control_curves):
                    ctrl_lo, ctrl_mean, ctrl_hi = _percentile_band(control_curves)
                    ctrl_sd = _prism_sd_envelope_columns(
                        control_curves, r_vals, prefix="control_G12"
                    )
                    n_control = int(len(control_curves))
                else:
                    ctrl_lo = ctrl_mean = ctrl_hi = np.full(len(r_vals), np.nan)
                    ctrl_sd = _prism_sd_envelope_columns(
                        np.empty((0, len(r_vals))), r_vals, prefix="control_G12"
                    )
                    n_control = 0

                for i, r_nm in enumerate(r_vals):
                    rows.append(
                        {
                            "aunp_subset": subset,
                            "window_mode": window_mode,
                            "control_comparison": comparison,
                            "r_nm": float(r_nm),
                            "fusing_G12_mean": float(fusing_sd["fusing_G12_mean"][i]),
                            "fusing_G12_sd": float(fusing_sd["fusing_G12_sd"][i]),
                            "fusing_G12_sd_envelope_lo": float(
                                fusing_sd["fusing_G12_sd_envelope_lo"][i]
                            ),
                            "fusing_G12_sd_envelope_hi": float(
                                fusing_sd["fusing_G12_sd_envelope_hi"][i]
                            ),
                            "fusing_G12_sem": float(fusing_sd["fusing_G12_sem"][i]),
                            "fusing_G12_sem_envelope_lo": float(
                                fusing_sd["fusing_G12_sem_envelope_lo"][i]
                            ),
                            "fusing_G12_sem_envelope_hi": float(
                                fusing_sd["fusing_G12_sem_envelope_hi"][i]
                            ),
                            "control_G12_mean": float(ctrl_mean[i]),
                            "control_G12_sd": float(ctrl_sd["control_G12_sd"][i]),
                            "control_G12_envelope_lo": float(ctrl_lo[i]),
                            "control_G12_envelope_hi": float(ctrl_hi[i]),
                            "control_G12_sd_envelope_lo": float(
                                ctrl_sd["control_G12_sd_envelope_lo"][i]
                            ),
                            "control_G12_sd_envelope_hi": float(
                                ctrl_sd["control_G12_sd_envelope_hi"][i]
                            ),
                            "control_G12_sem": float(ctrl_sd["control_G12_sem"][i]),
                            "control_G12_sem_envelope_lo": float(
                                ctrl_sd["control_G12_sem_envelope_lo"][i]
                            ),
                            "control_G12_sem_envelope_hi": float(
                                ctrl_sd["control_G12_sem_envelope_hi"][i]
                            ),
                            "n_fusing_curves": int(len(obs_curves)),
                            "n_control_curves": n_control,
                            "n_tomograms": n_tomograms,
                            "n_clefts": n_zones,
                            "envelope_percentile_lo": float(RIPLEY_PERCENTILE_LO),
                            "envelope_percentile_hi": float(RIPLEY_PERCENTILE_HI),
                        }
                    )
    return pd.DataFrame(rows)


def plot_pooled_fusion_point_aunp_ripley_g12_visualizations(
    curves_csv: Path | str = POOLED_G_CURVES_CSV,
    output_dir: Path | str = POOLED_G_FIGURES_DIR,
    prism_csv: Path | str = POOLED_G_PRISM_CSV,
) -> list[Path]:
    """Pool individual pair-correlation (g₁₂) curves across all tomograms/zones."""
    curves_csv = Path(curves_csv)
    output_dir = Path(output_dir)
    prism_csv = Path(prism_csv)
    if not curves_csv.is_file():
        print(f"No pooled g₁₂ curves CSV at {curves_csv}; skipping pooled outputs.")
        return []

    df = pd.read_csv(curves_csv)
    if df.empty:
        print("Pooled g₁₂ curves CSV is empty; skipping pooled outputs.")
        return []

    if "tomogram_name" not in df.columns:
        print("Pooled g₁₂ curves CSV missing tomogram_name; skipping pooled outputs.")
        return []

    prism_long = build_pooled_ripley_g12_prism_envelope_table(df)
    if prism_long.empty:
        print("No pooled g₁₂ envelope rows generated; skipping pooled outputs.")
        return []

    prism_csv.parent.mkdir(parents=True, exist_ok=True)
    prism_long.to_csv(prism_csv, index=False)
    print(
        f"Pooled fusion-point/AuNP pair-correlation (g₁₂) Prism table "
        f"({len(prism_long)} rows) -> {prism_csv}"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for (subset, window_mode, comparison), grp in prism_long.groupby(
        ["aunp_subset", "window_mode", "control_comparison"],
        sort=False,
    ):
        sub_df = df[
            (df["aunp_subset"] == subset)
            & (df["window_mode"] == window_mode)
        ]
        r_vals = np.sort(grp["r_nm"].unique())
        _, obs_curves = _extract_curves_matrix(sub_df, FUSING_CURVE_TYPE, value_col="ripley_g12")
        _, control_curves = _extract_curves_matrix(
            sub_df, CONTROL_CURVE_TYPE[comparison], value_col="ripley_g12"  # type: ignore[index]
        )
        if len(obs_curves) == 0 or len(control_curves) == 0:
            continue

        meta_row = grp.iloc[0]
        out_path = output_dir / f"ripley_g12_{window_mode}_{subset}_vs_{comparison}_pooled.png"
        _plot_g_control_comparison(
            r_vals,
            obs_curves,
            control_curves,
            output_path=out_path,
            title=(
                f"Pooled | window={window_mode} | AuNPs={subset} | vs {comparison}\n"
                f"{int(meta_row['n_tomograms'])} tomogram(s), "
                f"{int(meta_row['n_clefts'])} zone(s) | "
                f"{int(meta_row['n_fusing_curves'])} fusing curves, "
                f"{int(meta_row['n_control_curves'])} control curves"
            ),
        )
        written.append(out_path)

    if written:
        print(f"Pooled fusion-point/AuNP pair-correlation figures ({len(written)}) -> {output_dir}")
    else:
        print("No pooled g₁₂ figures written (missing curve groups in CSV).")
    return written


def _extract_bidir_curves_matrix(
    df: pd.DataFrame,
    curve_type: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Pivot the long bidirectional-family table to ``(r_vals, curves)``, one row per
    replicate curve (id'd by tomogram/zone/replicate_index, mirroring
    ``_extract_curves_matrix`` but reading the generic ``value`` column)."""
    sub = df.loc[df["curve_type"] == curve_type]
    if sub.empty:
        r_vals = np.sort(df["r_nm"].unique()) if "r_nm" in df.columns and not df.empty else np.array([])
        return r_vals, np.empty((0, len(r_vals)))

    r_vals = np.sort(sub["r_nm"].unique())
    n_r = len(r_vals)
    id_cols = ["tomogram_name", "alignment_dir", "cleft_name", "replicate_index"]
    for col in id_cols:
        if col not in sub.columns:
            sub = sub.copy()
            sub[col] = ""
    curves: list[np.ndarray] = []
    for _, grp in sub.groupby(id_cols, sort=False):
        grp = grp.sort_values("r_nm")
        if len(grp) != n_r or not np.allclose(grp["r_nm"].to_numpy(dtype=float), r_vals):
            continue
        curves.append(grp["value"].to_numpy(dtype=float))
    if not curves:
        return r_vals, np.empty((0, n_r))
    return r_vals, np.vstack(curves)


def build_pooled_bidirectional_prism_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pooled fusing vs control means and envelopes per (aunp_subset, window_mode,
    family, control_comparison, r_nm).

    Includes mean ± SEM/SD envelope columns for Prism paste (``fusing_sem_envelope_*``,
    ``control_sem_envelope_*``), plus the control 2.5–97.5% percentile band. SEM is
    across tomogram×zone curves on the stored L/g scale (not K→L transformed).
    """
    if df.empty or "tomogram_name" not in df.columns:
        return pd.DataFrame()

    rows: list[dict] = []
    for subset in AUNP_SUBSETS:
        for window_mode in RIPLEY_WINDOW_MODES:
            for family in ALL_BIDIR_FAMILIES:
                sub_df = df[
                    (df["aunp_subset"] == subset)
                    & (df["window_mode"] == window_mode)
                    & (df["family"] == family)
                ]
                if sub_df.empty:
                    continue
                r_vals, obs_curves = _extract_bidir_curves_matrix(sub_df, "fusing")
                if len(obs_curves) == 0:
                    continue
                fusing_sd = _prism_sd_envelope_columns(
                    obs_curves, r_vals, prefix="fusing"
                )
                n_tomograms = int(sub_df["tomogram_name"].nunique())
                n_zones = int(
                    sub_df[["tomogram_name", "alignment_dir", "cleft_name"]]
                    .drop_duplicates()
                    .shape[0]
                )
                for comparison in CONTROL_COMPARISONS:
                    _, control_curves = _extract_bidir_curves_matrix(sub_df, comparison)
                    if len(control_curves):
                        lo, _, hi = _percentile_band(control_curves)
                        n_control = int(len(control_curves))
                    else:
                        lo = hi = np.full(len(r_vals), np.nan)
                        n_control = 0
                    ctrl_sd = _prism_sd_envelope_columns(
                        control_curves, r_vals, prefix="control"
                    )
                    for i, r_nm in enumerate(r_vals):
                        rows.append(
                            {
                                "aunp_subset": subset,
                                "window_mode": window_mode,
                                "family": family,
                                "control_comparison": comparison,
                                "r_nm": float(r_nm),
                                "fusing_mean": float(fusing_sd["fusing_mean"][i]),
                                "fusing_sd": float(fusing_sd["fusing_sd"][i]),
                                "fusing_sd_envelope_lo": float(
                                    fusing_sd["fusing_sd_envelope_lo"][i]
                                ),
                                "fusing_sd_envelope_hi": float(
                                    fusing_sd["fusing_sd_envelope_hi"][i]
                                ),
                                "fusing_sem": float(fusing_sd["fusing_sem"][i]),
                                "fusing_sem_envelope_lo": float(
                                    fusing_sd["fusing_sem_envelope_lo"][i]
                                ),
                                "fusing_sem_envelope_hi": float(
                                    fusing_sd["fusing_sem_envelope_hi"][i]
                                ),
                                "control_mean": float(ctrl_sd["control_mean"][i]),
                                "control_sd": float(ctrl_sd["control_sd"][i]),
                                "control_envelope_lo": float(lo[i]),
                                "control_envelope_hi": float(hi[i]),
                                "control_sd_envelope_lo": float(
                                    ctrl_sd["control_sd_envelope_lo"][i]
                                ),
                                "control_sd_envelope_hi": float(
                                    ctrl_sd["control_sd_envelope_hi"][i]
                                ),
                                "control_sem": float(ctrl_sd["control_sem"][i]),
                                "control_sem_envelope_lo": float(
                                    ctrl_sd["control_sem_envelope_lo"][i]
                                ),
                                "control_sem_envelope_hi": float(
                                    ctrl_sd["control_sem_envelope_hi"][i]
                                ),
                                "n_fusing_curves": int(len(obs_curves)),
                                "n_control_curves": n_control,
                                "n_tomograms": n_tomograms,
                                "n_clefts": n_zones,
                            }
                        )
    return pd.DataFrame(rows)


def plot_pooled_fusion_point_aunp_ripley_bidirectional_visualizations(
    curves_csv: Path | str = POOLED_BIDIR_CURVES_CSV,
    output_dir: Path | str = POOLED_BIDIR_FIGURES_DIR,
    prism_csv: Path | str = None,
) -> list[Path]:
    """Pool the bidirectional (12/21/combined) K/L/g families across all tomograms/zones."""
    curves_csv = Path(curves_csv)
    output_dir = Path(output_dir)
    if prism_csv is None:
        prism_csv = output_dir.parent.parent / "fusion_point_aunp_ripley_bidirectional_prism_pooled.csv"
    prism_csv = Path(prism_csv)
    if not curves_csv.is_file():
        print(f"No pooled bidirectional curves CSV at {curves_csv}; skipping pooled outputs.")
        return []

    df = pd.read_csv(curves_csv)
    if df.empty or "tomogram_name" not in df.columns:
        print("Pooled bidirectional curves CSV is empty or missing tomogram_name; skipping.")
        return []

    prism_long = build_pooled_bidirectional_prism_table(df)
    if prism_long.empty:
        print("No pooled bidirectional envelope rows generated; skipping pooled outputs.")
        return []

    prism_csv.parent.mkdir(parents=True, exist_ok=True)
    prism_long.to_csv(prism_csv, index=False)
    print(f"Pooled fusion-point/AuNP bidirectional Prism table ({len(prism_long)} rows) -> {prism_csv}")

    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for (subset, window_mode, family, comparison), grp in prism_long.groupby(
        ["aunp_subset", "window_mode", "family", "control_comparison"], sort=False
    ):
        sub_df = df[
            (df["aunp_subset"] == subset) & (df["window_mode"] == window_mode) & (df["family"] == family)
        ]
        r_vals = np.sort(grp["r_nm"].unique())
        _, obs_curves = _extract_bidir_curves_matrix(sub_df, "fusing")
        _, control_curves = _extract_bidir_curves_matrix(sub_df, comparison)
        if len(obs_curves) == 0 or len(control_curves) == 0:
            continue
        meta_row = grp.iloc[0]
        out_path = output_dir / f"ripley_bidir_{family}_{window_mode}_{subset}_vs_{comparison}_pooled.png"
        _plot_bidirectional_family(
            r_vals, family, obs_curves, control_curves,
            output_path=out_path,
            title=(
                f"Pooled | window={window_mode} | AuNPs={subset} | {family} | vs {comparison}\n"
                f"{int(meta_row['n_tomograms'])} tomogram(s), {int(meta_row['n_clefts'])} zone(s) | "
                f"{int(meta_row['n_fusing_curves'])} fusing curves, {int(meta_row['n_control_curves'])} control curves"
            ),
        )
        written.append(out_path)

    if written:
        print(f"Pooled fusion-point/AuNP bidirectional figures ({len(written)}) -> {output_dir}")
    else:
        print("No pooled bidirectional figures written (missing curve groups in CSV).")
    return written


def run_fusion_point_aunp_analyses_for_zone(
    tomogram_path: Path,
    alignment_dir: str,
    zone_name: str,
    cleft_index: int,
    *,
    vesicle_distance_threshold: float = 20.0,
    fusion_point_threshold: float = 20.0,
    n_replicates: int = DEFAULT_NULL_REPLICATES_N,
    seed: int = DEFAULT_ANALYSIS_SEED,
    write_figures: bool = True,
    r_max_nm: float = DEFAULT_RIPLEY_R_MAX_NM,
    r_step_nm: float = DEFAULT_RIPLEY_R_STEP_NM,
    monomer_star_pattern: Optional[str] = None,
    dimer_star_pattern: Optional[str] = None,
    single_pick_star_pattern: Optional[str] = None,
    use_single_pick_pool: bool = False,
) -> dict[str, Path] | None:
    tomogram_path = Path(tomogram_path)
    alignment_dir = require_alignment_dir(alignment_dir)
    tomogram_name = tomogram_path.name

    membrane_name = presynaptic_membrane_name_for_zone(zone_name)
    membrane_az_pairs = import_presynaptic_membranes_and_clefts(tomogram_path, alignment_dir=alignment_dir)

    fusing_rows = filter_fusion_rows_for_zone(
        enumerate_close_vesicle_fusion_points(
            tomogram_path,
            alignment_dir=alignment_dir,
            vesicle_distance_threshold=vesicle_distance_threshold,
            fusion_point_threshold=fusion_point_threshold,
            fusing_only=True,
        ),
        membrane_name,
    )
    if not fusing_rows:
        print(f"  No fusing vesicles for {zone_name}, skipping fusion-point/AuNP analyses")
        return None

    close_rows = [
        row
        for row in filter_fusion_rows_for_zone(
            enumerate_close_vesicle_fusion_points(
                tomogram_path,
                alignment_dir=alignment_dir,
                vesicle_distance_threshold=vesicle_distance_threshold,
                fusion_point_threshold=fusion_point_threshold,
                fusing_only=False,
            ),
            membrane_name,
        )
        if row.get("vesicle_distance_class") == "close" or bool(row.get("is_close"))
    ]

    try:
        loaded = load_monomer_dimer_aunps_for_zone(
            tomogram_path,
            alignment_dir,
            cleft_index,
            monomer_star_pattern=monomer_star_pattern,
            dimer_star_pattern=dimer_star_pattern,
            single_pick_star_pattern=single_pick_star_pattern,
            use_single_pick_pool=use_single_pick_pool,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(
            f"  Skipping fusion-point/AuNP analyses for {zone_name}: {exc}"
        )
        return None

    aunp_coords_all = loaded.coords
    aunp_meta = loaded.meta
    subsets_to_run = available_aunp_subsets(loaded.kinds_loaded)
    if use_single_pick_pool or loaded.kinds_loaded == ("all",):
        print(
            f"  Single AuNP pick STAR pool for {zone_name}; "
            f"running {', '.join(subsets_to_run)} analyses."
        )
    elif len(loaded.kinds_loaded) == 1:
        print(
            f"  Only {loaded.kinds_loaded[0]} STAR found for {zone_name}; "
            f"running {', '.join(subsets_to_run)} analyses."
        )
    elif len(loaded.kinds_loaded) == 2:
        print(
            f"  Monomer and dimer STARs found for {zone_name}; "
            f"running {', '.join(subsets_to_run)} analyses."
        )

    pre_surface = load_presynaptic_az_points_for_zone(tomogram_path, alignment_dir, zone_name)

    out_dir = output_dir_for_zone(tomogram_path, alignment_dir, zone_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = out_dir / "figures"
    if write_figures:
        figures_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    r_vals = _ripley_r_grid(r_max_nm, r_step_nm)

    fusing_xyz = _fusion_xyz_from_rows(fusing_rows)
    close_xyz = _fusion_xyz_from_rows(close_rows)
    ripley_windows: dict[WindowMode, RipleyWindow3D | None] = {
        mode: None for mode in RIPLEY_WINDOW_MODES
    }

    from .cleft import import_cleft_segmentations

    az_segmentation = import_cleft_segmentations(
        tomogram_path, alignment_dir=alignment_dir
    ).get(zone_name)
    pre_membrane_coords = (
        az_segmentation.get("presynaptic_outer_coords") if az_segmentation is not None else None
    )
    post_membrane_coords = (
        az_segmentation.get("postsynaptic_outer_coords") if az_segmentation is not None else None
    )

    try:
        cleft_coords = load_synaptic_cleft_cleft_points(
            tomogram_path, alignment_dir, zone_name
        )
        ripley_windows["synaptic_cleft_az_hull"] = build_ripley_window_3d(
            cleft_coords,
            "synaptic_cleft_az_hull",
            pre_membrane_coords=pre_membrane_coords,
            post_membrane_coords=post_membrane_coords,
            use_angle_betweenness=True,
            rng=rng,
        )
    except (ValueError, FileNotFoundError) as exc:
        cleft_coords = None
        print(f"  Ripley synaptic-cleft AZ hull window skipped for {zone_name}: {exc}")

    # Deterministic grid-quadrature edge correction (replacing Monte Carlo): build the
    # window's quadrature grid once per window mode and reconcile window.volume_nm3 with
    # this grid's own volume estimate, so K's outer V/(n1*n2) scaling and the edge-factor
    # denominator agree on an identical V (same approach as the AZ-center and monomer/dimer
    # Ripley analyses).
    window_grid_points: dict[WindowMode, np.ndarray] = {}
    for mode, window in list(ripley_windows.items()):
        if window is None:
            continue
        grid_points = build_window_grid_points(window, FUSION_POINT_EDGE_GRID_SPACING_NM)
        grid_volume_nm3 = float(len(grid_points)) * (FUSION_POINT_EDGE_GRID_SPACING_NM ** 3)
        ripley_windows[mode] = dataclasses_replace(window, volume_nm3=grid_volume_nm3)
        window_grid_points[mode] = grid_points

    if write_figures and az_segmentation is not None:
        for mode, window in ripley_windows.items():
            if window is None:
                continue
            try:
                point_groups = [
                    {"coords": fusing_xyz, "label": "fusing sites", "color": "tab:red", "marker": "*", "size": 60},
                    {"coords": close_xyz, "label": "close sites", "color": "tab:orange", "marker": "s", "size": 22},
                ]
                dropped_parts: list[np.ndarray] = []
                title_bits: list[str] = [
                    f"{len(fusing_xyz)} fusing, {len(close_xyz)} close sites"
                ]
                if "all" in loaded.kinds_loaded:
                    all_coords, _ = subset_aunps(aunp_meta, subset="all")
                    all_inside = _points_inside_hull(all_coords, window.hull)
                    dropped_parts.append(all_coords[~all_inside])
                    point_groups.append(
                        {
                            "coords": all_coords[all_inside],
                            "label": "AuNPs",
                            "color": "tab:purple",
                            "marker": "o",
                            "size": 18,
                        }
                    )
                    title_bits.append(f"{int(all_inside.sum())} AuNPs")
                else:
                    monomer_coords_all, _ = subset_aunps(aunp_meta, subset="monomer")
                    dimer_coords_all, _ = subset_aunps(aunp_meta, subset="dimer")
                    monomer_inside = _points_inside_hull(monomer_coords_all, window.hull)
                    dimer_inside = _points_inside_hull(dimer_coords_all, window.hull)
                    dropped_parts.extend(
                        [
                            monomer_coords_all[~monomer_inside],
                            dimer_coords_all[~dimer_inside],
                        ]
                    )
                    point_groups.extend(
                        [
                            {
                                "coords": monomer_coords_all[monomer_inside],
                                "label": "monomer AuNPs",
                                "color": "tab:purple",
                                "marker": "o",
                                "size": 18,
                            },
                            {
                                "coords": dimer_coords_all[dimer_inside],
                                "label": "dimer AuNPs",
                                "color": "tab:green",
                                "marker": "^",
                                "size": 28,
                            },
                        ]
                    )
                    title_bits.append(
                        f"{int(monomer_inside.sum())} monomer + "
                        f"{int(dimer_inside.sum())} dimer AuNPs"
                    )
                dropped_coords = (
                    np.vstack(dropped_parts)
                    if any(len(p) for p in dropped_parts)
                    else np.zeros((0, 3))
                )
                plot_ripley_window_geometry_diagnostic(
                    tomogram_path,
                    alignment_dir,
                    zone_name,
                    point_groups=point_groups,
                    az_segmentation=az_segmentation,
                    window=window,
                    grid_points=window_grid_points[mode],
                    grid_spacing_nm=FUSION_POINT_EDGE_GRID_SPACING_NM,
                    output_path=figures_dir / "geometry_diagnostic.png",
                    dropped_coords=dropped_coords,
                    title_lines=[", ".join(title_bits)],
                    print_prefix="Fusion-point geometry diagnostic",
                )
            except Exception as diag_exc:
                print(f"  Skipping fusion-point geometry diagnostic ({mode}) for {zone_name}: {diag_exc}")

    # --- Distance wide CSVs (per AuNP subset) + vesicle-column tables (all AuNPs) ---
    original_cols: dict[str, np.ndarray] = {}
    original_cols_global: dict[str, np.ndarray] = {}
    for fp in fusing_rows:
        xyz = np.array(
            [fp["fusion_point_x_nm"], fp["fusion_point_y_nm"], fp["fusion_point_z_nm"]],
            dtype=float,
        )
        original_cols[_site_column_name(fp["vesicle_name"], "original")] = xyz
        original_cols_global[
            _global_site_column_name(
                tomogram_name, alignment_dir, zone_name, fp["vesicle_name"], "fusing"
            )
        ] = xyz

    close_cols: dict[str, np.ndarray] = {}
    close_cols_global: dict[str, np.ndarray] = {}
    for fp in close_rows:
        xyz = np.array(
            [fp["fusion_point_x_nm"], fp["fusion_point_y_nm"], fp["fusion_point_z_nm"]],
            dtype=float,
        )
        close_cols[_site_column_name(fp["vesicle_name"], "close")] = xyz
        close_cols_global[
            _global_site_column_name(
                tomogram_name, alignment_dir, zone_name, fp["vesicle_name"], "close"
            )
        ] = xyz

    shift_by_rep = _shift_sites_by_replicate(
        fusing_rows,
        membrane_az_pairs,
        offset_nm=FUSION_POINT_SHIFT_OFFSET_NM,
        n_shifts=n_replicates,
        rng=rng,
        max_snap_distance_nm=FUSION_POINT_AZ_MAX_SNAP_DISTANCE_NM,
    )
    shift_cols: dict[str, np.ndarray] = {}
    shift_cols_global: dict[str, np.ndarray] = {}
    for rep_id, ves_map in shift_by_rep.items():
        for fp in fusing_rows:
            vid = int(fp["vesicle_id"])
            if vid not in ves_map:
                continue
            xyz = ves_map[vid]
            shift_cols[_site_column_name(fp["vesicle_name"], f"shift_{rep_id:03d}")] = xyz
            shift_cols_global[
                _global_site_column_name(
                    tomogram_name,
                    alignment_dir,
                    zone_name,
                    fp["vesicle_name"],
                    f"shift_{rep_id:03d}",
                )
            ] = xyz

    # Label permutation uses the full monomer+dimer pool; same null sites for all subsets.
    label_per_ves, label_pooled = _label_permutation_sites(
        fusing_rows,
        aunp_coords_all,
        pre_surface,
        n_perm=n_replicates,
        rng=rng,
    )
    label_cols: dict[str, np.ndarray] = {}
    label_cols_global: dict[str, np.ndarray] = {}
    for perm_id, ves_map in label_per_ves.items():
        for fp in fusing_rows:
            vid = int(fp["vesicle_id"])
            if vid not in ves_map:
                continue
            xyz = ves_map[vid]
            label_cols[_site_column_name(fp["vesicle_name"], f"perm_{perm_id:03d}")] = xyz
            label_cols_global[
                _global_site_column_name(
                    tomogram_name,
                    alignment_dir,
                    zone_name,
                    fp["vesicle_name"],
                    f"perm_{perm_id:03d}",
                )
            ] = xyz

    distance_paths: dict[str, dict[str, Path]] = {subset: {} for subset in subsets_to_run}
    for subset in subsets_to_run:
        sub_coords, sub_meta = subset_aunps(aunp_meta, subset=subset)
        meta_out = sub_meta.copy()
        meta_out.insert(0, "aunp_subset", subset)
        meta_out.insert(1, "tomogram_name", tomogram_name)
        meta_out.insert(2, "alignment_dir", alignment_dir)
        meta_out.insert(3, "cleft_name", zone_name)

        df_orig = build_distance_wide_csv(sub_coords, meta_out, original_cols)
        df_close = build_distance_wide_csv(sub_coords, meta_out, close_cols)
        df_shift = build_distance_wide_csv(sub_coords, meta_out, shift_cols)
        df_perm = build_distance_wide_csv(sub_coords, meta_out, label_cols)

        p_orig = out_dir / _distance_csv_name("distances_original_fusing_wide", subset)
        p_close = out_dir / _distance_csv_name("distances_close_wide", subset)
        p_shift = out_dir / _distance_csv_name("distances_40nm_shift_wide", subset)
        p_perm = out_dir / _distance_csv_name("distances_label_permutation_wide", subset)
        df_orig.to_csv(p_orig, index=False)
        df_close.to_csv(p_close, index=False)
        df_shift.to_csv(p_shift, index=False)
        df_perm.to_csv(p_perm, index=False)
        distance_paths[subset] = {
            "original": p_orig,
            "close": p_close,
            "shift": p_shift,
            "permutation": p_perm,
        }

        if len(sub_coords) == 0:
            print(f"  No {subset} AuNPs for {zone_name}; distance CSVs written empty")

    zone_prefix = _zone_column_prefix(tomogram_name, alignment_dir, zone_name)
    distance_column_paths: dict[str, Path] = {}
    col_only_specs = [
        ("fusing", original_cols_global, POOLED_DIST_FUSING_COLUMNS_CSV),
        ("close", close_cols_global, POOLED_DIST_CLOSE_COLUMNS_CSV),
        ("shift_40nm", shift_cols_global, POOLED_DIST_SHIFT_COLUMNS_CSV),
        ("label_permutation", label_cols_global, POOLED_DIST_PERM_COLUMNS_CSV),
    ]
    cumhist_pooled_bases = {
        "shift_40nm_cumhist": POOLED_DIST_SHIFT_CUMHIST_CSV,
        "label_permutation_cumhist": POOLED_DIST_LABEL_PERM_CUMHIST_CSV,
    }

    for subset in subsets_to_run:
        sub_coords, _ = subset_aunps(aunp_meta, subset=subset)
        shift_cumhist_df = build_shift_cumulative_histogram_dataframe(
            sub_coords,
            fusing_rows,
            shift_by_rep,
            tomogram_name=tomogram_name,
            alignment_dir=alignment_dir,
            zone_name=zone_name,
        )
        label_perm_cumhist_df = build_label_permutation_cumulative_histogram_dataframe(
            sub_coords,
            label_pooled,
            tomogram_name=tomogram_name,
            alignment_dir=alignment_dir,
            zone_name=zone_name,
        )
        cumhist_by_kind = {
            "shift_40nm_cumhist": shift_cumhist_df,
            "label_permutation_cumhist": label_perm_cumhist_df,
        }

        for kind, cols_global, pooled_base in col_only_specs:
            df_cols = build_distance_columns_only_dataframe(sub_coords, cols_global)
            stem = DISTANCE_COLUMNS_ONLY_STEMS[kind]
            p_cols = out_dir / _distance_csv_name(stem, subset)
            df_cols.to_csv(p_cols, index=False)
            distance_column_paths[f"{kind}__{subset}"] = p_cols
            upsert_pooled_distance_columns_csv(
                _pooled_subset_csv(pooled_base, subset),
                df_cols,
                drop_column_prefix=zone_prefix,
            )

        for kind, df_hist in cumhist_by_kind.items():
            stem = DISTANCE_COLUMNS_ONLY_STEMS[kind]
            p_hist = out_dir / _distance_csv_name(stem, subset)
            df_hist.to_csv(p_hist, index=False)
            distance_column_paths[f"{kind}__{subset}"] = p_hist
            upsert_pooled_cumulative_histogram_csv(
                _pooled_subset_csv(cumhist_pooled_bases[kind], subset),
                df_hist,
                drop_column_prefix=zone_prefix,
            )
    # --- Ripley: both window modes × three AuNP partner subsets ---
    ripley_frames: list[pd.DataFrame] = []
    prism_frames: list[pd.DataFrame] = []
    mad_summary_frames: list[pd.DataFrame] = []
    mad_curve_frames: list[pd.DataFrame] = []
    g_ripley_frames: list[pd.DataFrame] = []
    g_prism_frames: list[pd.DataFrame] = []
    bidir_frames: list[pd.DataFrame] = []
    n_aunp_dropped_outside_hull: dict[str, int] = {}

    for window_mode in RIPLEY_WINDOW_MODES:
        window = ripley_windows.get(window_mode)
        if window is None:
            continue
        grid_points = window_grid_points[window_mode]
        for subset in subsets_to_run:
            sub_coords, _ = subset_aunps(aunp_meta, subset=subset)
            if len(sub_coords) == 0:
                print(f"  Skipping Ripley for {zone_name} ({subset}, {window_mode}): no partner AuNPs")
                continue
            # AuNPs outside this window's hull are always dropped before any Ripley
            # statistic is computed -- edge correction and volume normalization are only
            # valid for points observed inside the window.
            inside = _points_inside_hull(sub_coords, window.hull)
            n_dropped = int((~inside).sum())
            if n_dropped:
                print(
                    f"  Dropping {n_dropped} {subset} AuNP(s) outside the {window_mode} "
                    f"hull for {zone_name}"
                )
            n_aunp_dropped_outside_hull[f"{subset}__{window_mode}"] = n_dropped
            sub_coords = sub_coords[inside]
            if len(sub_coords) == 0:
                print(
                    f"  Skipping Ripley for {zone_name} ({subset}, {window_mode}): "
                    "no partner AuNPs inside the hull"
                )
                continue
            df_r, df_prism, df_mad_summary, df_mad_curves, df_g, df_g_prism = run_ripley_for_zone_window(
                zone_name=zone_name,
                aunp_subset=subset,
                window=window,
                aunp_coords=sub_coords,
                fusing_rows=fusing_rows,
                close_rows=close_rows,
                shift_by_replicate=shift_by_rep,
                label_perm_pooled=label_pooled,
                r_vals=r_vals,
                grid_points=grid_points,
                edge_grid_spacing_nm=FUSION_POINT_EDGE_GRID_SPACING_NM,
                figures_dir=figures_dir if write_figures else None,
            )
            ripley_frames.append(df_r)
            g_ripley_frames.append(df_g)
            df_bidir = run_bidirectional_ripley_for_zone_window(
                zone_name=zone_name,
                aunp_subset=subset,
                window=window,
                aunp_coords=sub_coords,
                fusing_rows=fusing_rows,
                close_rows=close_rows,
                shift_by_replicate=shift_by_rep,
                label_perm_pooled=label_pooled,
                r_vals=r_vals,
                grid_points=grid_points,
                edge_grid_spacing_nm=FUSION_POINT_EDGE_GRID_SPACING_NM,
                figures_dir=figures_dir if write_figures else None,
            )
            if not df_bidir.empty:
                bidir_frames.append(df_bidir)
            if not df_prism.empty:
                df_prism = df_prism.copy()
                df_prism.insert(0, "tomogram_name", tomogram_name)
                df_prism.insert(1, "alignment_dir", alignment_dir)
                prism_frames.append(df_prism)
            if not df_g_prism.empty:
                df_g_prism = df_g_prism.copy()
                df_g_prism.insert(0, "tomogram_name", tomogram_name)
                df_g_prism.insert(1, "alignment_dir", alignment_dir)
                g_prism_frames.append(df_g_prism)
            if not df_mad_summary.empty:
                df_mad_summary = df_mad_summary.copy()
                df_mad_summary.insert(0, "tomogram_name", tomogram_name)
                df_mad_summary.insert(1, "alignment_dir", alignment_dir)
                mad_summary_frames.append(df_mad_summary)
            if not df_mad_curves.empty:
                df_mad_curves = df_mad_curves.copy()
                df_mad_curves.insert(0, "tomogram_name", tomogram_name)
                df_mad_curves.insert(1, "alignment_dir", alignment_dir)
                mad_curve_frames.append(df_mad_curves)

    if not any(ripley_windows.values()):
        print(f"  Skipping Ripley for {zone_name}: no valid hull windows")

    def _write_individual_wide_tables(
        ripley_long: pd.DataFrame, *, value_col: str, file_prefix: str
    ) -> None:
        for (subset, window_mode, curve_type), grp in ripley_long.groupby(
            ["aunp_subset", "window_mode", "curve_type"], sort=False
        ):
            r_vals_g = np.sort(grp["r_nm"].unique())
            curves_list: list[np.ndarray] = []
            for _, rep in grp.groupby("replicate_index", sort=True):
                rep = rep.sort_values("r_nm")
                if len(rep) != len(r_vals_g):
                    continue
                curves_list.append(rep[value_col].to_numpy(dtype=float))
            if not curves_list:
                continue
            wide = curves_matrix_to_wide_dataframe(
                np.vstack(curves_list),
                r_vals_g,
                curve_type=str(curve_type),
            )
            wide_name = f"{file_prefix}_individual_{subset}_{window_mode}_{curve_type}_wide.csv"
            wide.to_csv(out_dir / wide_name, index=False)

    if ripley_frames:
        ripley_long = pd.concat(ripley_frames, ignore_index=True)
        ripley_long.insert(0, "tomogram_name", tomogram_name)
        ripley_long.insert(1, "alignment_dir", alignment_dir)
        ripley_long.to_csv(out_dir / "ripley_l12_curves.csv", index=False)
        # Explicit individual-curves filename for consistency with other Ripley analyses.
        ripley_long.to_csv(out_dir / "ripley_l12_individual_curves.csv", index=False)
        # Prism-friendly wide tables: one column per replicate curve.
        _write_individual_wide_tables(
            ripley_long, value_col="ripley_l12", file_prefix="ripley_l12"
        )
    if g_ripley_frames:
        g_ripley_long = pd.concat(g_ripley_frames, ignore_index=True)
        g_ripley_long.insert(0, "tomogram_name", tomogram_name)
        g_ripley_long.insert(1, "alignment_dir", alignment_dir)
        g_ripley_long.to_csv(out_dir / "ripley_g12_curves.csv", index=False)
        g_ripley_long.to_csv(out_dir / "ripley_g12_individual_curves.csv", index=False)
        _write_individual_wide_tables(
            g_ripley_long, value_col="ripley_g12", file_prefix="ripley_g12"
        )
    if mad_summary_frames:
        pd.concat(mad_summary_frames, ignore_index=True).to_csv(
            out_dir / "ripley_l12_mad_summary.csv", index=False
        )
    if mad_curve_frames:
        pd.concat(mad_curve_frames, ignore_index=True).to_csv(
            out_dir / "ripley_l12_mad_curves.csv", index=False
        )
    if prism_frames:
        prism_long = pd.concat(prism_frames, ignore_index=True)
        prism_long.to_csv(out_dir / "ripley_l12_prism_envelopes.csv", index=False)
    if g_prism_frames:
        g_prism_long = pd.concat(g_prism_frames, ignore_index=True)
        g_prism_long.to_csv(out_dir / "ripley_g12_prism_envelopes.csv", index=False)
    if bidir_frames:
        bidir_long = pd.concat(bidir_frames, ignore_index=True)
        bidir_long.insert(0, "tomogram_name", tomogram_name)
        bidir_long.insert(1, "alignment_dir", alignment_dir)
        bidir_long.to_csv(out_dir / "ripley_bidirectional_curves.csv", index=False)

    n_monomer = int((aunp_meta["aunp_kind"] == "monomer").sum())
    n_dimer = int((aunp_meta["aunp_kind"] == "dimer").sum())
    n_all = int((aunp_meta["aunp_kind"] == "all").sum())
    meta = {
        "tomogram_name": tomogram_name,
        "alignment_dir": alignment_dir,
        "cleft_name": zone_name,
        "cleft_index": int(cleft_index),
        "n_fusing_vesicles": len(fusing_rows),
        "n_close_vesicles": len(close_rows),
        "n_aunp_monomer": n_monomer,
        "n_aunp_dimer": n_dimer,
        "n_aunp_all": n_all,
        "n_aunp_monomer_dimer": len(aunp_coords_all),
        "use_single_pick_pool": bool(use_single_pick_pool),
        "aunp_kinds_loaded": list(loaded.kinds_loaded),
        "aunp_subsets_analyzed": list(subsets_to_run),
        "ripley_window_modes": list(RIPLEY_WINDOW_MODES),
        "ripley_windows": {
            mode: {
                "defining_from": "presynaptic_and_postsynaptic_cleft_surface_points",
                "volume_nm3": float(win.volume_nm3),
                "uses_angle_betweenness": bool(win.use_angle_betweenness),
                "n_defining_points": int(len(cleft_coords)) if cleft_coords is not None else None,
            }
            for mode, win in ripley_windows.items()
            if win is not None
        },
        "n_hull_fusing_fusion_sites": int(len(fusing_xyz)),
        "n_hull_close_fusion_sites": int(len(close_xyz)),
        "n_hull_aunp_monomer_dimer": int(len(aunp_coords_all)),
        "n_aunp_dropped_outside_hull": n_aunp_dropped_outside_hull,
        "n_shift_replicates": int(n_replicates),
        "n_label_permutations": int(n_replicates),
        "mad_min_null_curves": int(MAD_MIN_NULL_CURVES),
        "mad_nulls": ["close", "shift_40nm", "label_permutation"],
        "mad_r_ranges": [label for label, _, _ in MAD_R_RANGES],
        "curve_families": ["l12", "g12"],
        "bidirectional_curve_families": list(ALL_BIDIR_FAMILIES),
        "seed": int(seed),
        "ripley_edge_correction": "isotropic_3d_grid",
        "edge_correction_grid_spacing_nm": float(FUSION_POINT_EDGE_GRID_SPACING_NM),
        "g_estimator": "pair_correlation_from_k_diff",
        "window_volume_definition": "grid_quadrature_volume",
        "window_uses_angle_betweenness": True,
    }
    with open(out_dir / "run_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    if n_all:
        aunp_count_msg = f"{n_all} AuNPs (single pick pool)"
    else:
        aunp_count_msg = f"{n_monomer} monomer + {n_dimer} dimer AuNPs"
    print(
        f"  Fusion-point/AuNP 3D analyses ({zone_name}): "
        f"{len(fusing_rows)} fusing, {aunp_count_msg} -> {out_dir}"
    )
    return {
        "distance_paths": distance_paths,
        "distance_column_paths": distance_column_paths,
        "output_dir": out_dir,
    }


def build_fusion_null_query_point_dataframes_for_zonograms(
    tomogram_path: Path,
    alignment_dir: str,
    az_mapping: dict,
    *,
    vesicle_distance_threshold: float = 20.0,
    fusion_point_threshold: float = 20.0,
    n_replicates: int = DEFAULT_NULL_REPLICATES_N,
    seed: int = DEFAULT_ANALYSIS_SEED,
    monomer_star_pattern: Optional[str] = None,
    dimer_star_pattern: Optional[str] = None,
    single_pick_star_pattern: Optional[str] = None,
    use_single_pick_pool: bool = False,
) -> dict[str, pd.DataFrame]:
    """
    Long-form 40 nm shift and label-permutation query sites for zonogram overlays.

    Uses the same 3D geometry, AuNP pool, replicate count, and seed as
  ``run_fusion_point_aunp_analyses_for_zone``.
    """
    tomogram_path = Path(tomogram_path)
    alignment_dir = require_alignment_dir(alignment_dir)
    az_mapping = {int(k): v for k, v in (az_mapping or {}).items()}
    if not az_mapping:
        return {"40nm_shift": pd.DataFrame(), "label_permutation": pd.DataFrame()}

    membrane_az_pairs = import_presynaptic_membranes_and_clefts(
        tomogram_path, alignment_dir=alignment_dir
    )
    shift_rows: list[dict] = []
    perm_rows: list[dict] = []

    for az_idx in sorted(az_mapping):
        zone_name = az_mapping[az_idx]
        membrane_name = presynaptic_membrane_name_for_zone(zone_name)
        fusing_rows = filter_fusion_rows_for_zone(
            enumerate_close_vesicle_fusion_points(
                tomogram_path,
                alignment_dir=alignment_dir,
                vesicle_distance_threshold=vesicle_distance_threshold,
                fusion_point_threshold=fusion_point_threshold,
                fusing_only=True,
            ),
            membrane_name,
        )
        if not fusing_rows:
            continue

        try:
            loaded = load_monomer_dimer_aunps_for_zone(
                tomogram_path,
                alignment_dir,
                az_idx,
                monomer_star_pattern=monomer_star_pattern,
                dimer_star_pattern=dimer_star_pattern,
                single_pick_star_pattern=single_pick_star_pattern,
                use_single_pick_pool=use_single_pick_pool,
            )
        except (FileNotFoundError, ValueError) as exc:
            print(
                f"  Zonogram null overlays: skipping {zone_name} AuNPs ({exc})"
            )
            continue

        aunp_coords_all = loaded.coords

        pre_surface = load_presynaptic_az_points_for_zone(
            tomogram_path, alignment_dir, zone_name
        )
        rng = np.random.default_rng(seed)

        shift_by_rep = _shift_sites_by_replicate(
            fusing_rows,
            membrane_az_pairs,
            offset_nm=FUSION_POINT_SHIFT_OFFSET_NM,
            n_shifts=n_replicates,
            rng=rng,
            max_snap_distance_nm=FUSION_POINT_AZ_MAX_SNAP_DISTANCE_NM,
        )
        _, label_pooled = _label_permutation_sites(
            fusing_rows,
            aunp_coords_all,
            pre_surface,
            n_perm=n_replicates,
            rng=rng,
        )

        for rep_id, ves_map in shift_by_rep.items():
            for fp in fusing_rows:
                vid = int(fp["vesicle_id"])
                if vid not in ves_map:
                    continue
                query = ves_map[vid]
                shift_rows.append(
                    {
                        "cleft_name": zone_name,
                        "shift_replicate_id": int(rep_id),
                        "vesicle_id": vid,
                        "vesicle_name": fp.get("vesicle_name"),
                        "fusion_point_x_nm": float(fp["fusion_point_x_nm"]),
                        "fusion_point_y_nm": float(fp["fusion_point_y_nm"]),
                        "fusion_point_z_nm": float(fp["fusion_point_z_nm"]),
                        "query_point_x_nm": float(query[0]),
                        "query_point_y_nm": float(query[1]),
                        "query_point_z_nm": float(query[2]),
                        "control_offset_nm": float(FUSION_POINT_SHIFT_OFFSET_NM),
                    }
                )

        for perm_id, queries in label_pooled.items():
            for fusion_site_idx, query in enumerate(queries):
                q = np.asarray(query, dtype=float).reshape(3)
                perm_rows.append(
                    {
                        "cleft_name": zone_name,
                        "permutation_id": int(perm_id),
                        "fusion_site_index": int(fusion_site_idx),
                        "query_point_x_nm": float(q[0]),
                        "query_point_y_nm": float(q[1]),
                        "query_point_z_nm": float(q[2]),
                    }
                )

    return {
        "40nm_shift": pd.DataFrame(shift_rows),
        "label_permutation": pd.DataFrame(perm_rows),
    }


def run_fusion_point_aunp_analyses_for_tomogram(
    tomogram_path: Path,
    alignment_dir: str,
    *,
    cleft_indices: Sequence[int] | None = None,
    vesicle_distance_threshold: float = 20.0,
    fusion_point_threshold: float = 20.0,
    n_replicates: int = DEFAULT_NULL_REPLICATES_N,
    seed: int = DEFAULT_ANALYSIS_SEED,
    write_figures: bool = True,
    monomer_star_pattern: Optional[str] = None,
    dimer_star_pattern: Optional[str] = None,
    single_pick_star_pattern: Optional[str] = None,
    use_single_pick_pool: bool = False,
) -> tuple[list[pd.DataFrame], list[pd.DataFrame], list[pd.DataFrame], list[pd.DataFrame], list[pd.DataFrame]]:
    """Returns ``(ripley_l12_frames, l12_prism_frames, g12_frames, g12_prism_frames, bidirectional_frames)``."""
    from .cleft import load_cleft_mapping

    tomogram_path = Path(tomogram_path)
    alignment_dir = require_alignment_dir(alignment_dir)
    az_mapping = load_cleft_mapping(tomogram_path, alignment_dir) or {}
    if not az_mapping:
        print("No synaptic cleft mapping; skipping fusion-point/AuNP 3D analyses")
        return [], [], [], [], []

    az_mapping = {int(k): v for k, v in az_mapping.items()}
    indices = list(cleft_indices) if cleft_indices is not None else sorted(az_mapping)
    ripley_frames: list[pd.DataFrame] = []
    prism_frames: list[pd.DataFrame] = []
    g_ripley_frames: list[pd.DataFrame] = []
    g_prism_frames: list[pd.DataFrame] = []
    bidir_frames: list[pd.DataFrame] = []

    for az_idx in indices:
        if az_idx not in az_mapping:
            print(f"  Active zone index {az_idx} not in mapping, skipping")
            continue
        zone_name = az_mapping[az_idx]
        result = run_fusion_point_aunp_analyses_for_zone(
            tomogram_path,
            alignment_dir,
            zone_name,
            int(az_idx),
            vesicle_distance_threshold=vesicle_distance_threshold,
            fusion_point_threshold=fusion_point_threshold,
            n_replicates=n_replicates,
            seed=seed,
            write_figures=write_figures,
            monomer_star_pattern=monomer_star_pattern,
            dimer_star_pattern=dimer_star_pattern,
            single_pick_star_pattern=single_pick_star_pattern,
            use_single_pick_pool=use_single_pick_pool,
        )
        if result is None:
            continue
        out_dir = result["output_dir"]
        curves_path = out_dir / "ripley_l12_curves.csv"
        if curves_path.is_file():
            ripley_frames.append(pd.read_csv(curves_path))
        prism_path = out_dir / "ripley_l12_prism_envelopes.csv"
        if prism_path.is_file():
            prism_frames.append(pd.read_csv(prism_path))
        g_curves_path = out_dir / "ripley_g12_curves.csv"
        if g_curves_path.is_file():
            g_ripley_frames.append(pd.read_csv(g_curves_path))
        g_prism_path = out_dir / "ripley_g12_prism_envelopes.csv"
        if g_prism_path.is_file():
            g_prism_frames.append(pd.read_csv(g_prism_path))
        bidir_path = out_dir / "ripley_bidirectional_curves.csv"
        if bidir_path.is_file():
            bidir_frames.append(pd.read_csv(bidir_path))

    return ripley_frames, prism_frames, g_ripley_frames, g_prism_frames, bidir_frames
