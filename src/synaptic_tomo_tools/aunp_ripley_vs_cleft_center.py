"""
3D bivariate Ripley K₁₂ / L₁₂ of AuNP positions relative to the synaptic cleft center.

Type-1 foci: one synaptic cleft center per zone (mean of presynaptic + postsynaptic AZ points).
Type-2 partners: AuNP pick coordinates in that zone.

Reports three K/L families per zone (see ``L_CURVE_FAMILIES``): the direct K₁₂/L₁₂
(center-as-focus, the original statistic), the reversed K₂₁/L₂₁ (AuNPs-as-foci), and their
intensity-weighted combination K_combined/L_combined (Lotwick & Silverman 1982) — all three
pooled across zones the same way. Also reports g₁₂ (center-as-focus only), computed and
pooled at a fine 1nm shell width; individual-zone g₁₂ figures rebin to a coarser 10nm width
for display only (see ``AZ_CENTER_G12_DISPLAY_BIN_WIDTH_NM``) since a single zone's shells
are too noisy at 1nm to plot meaningfully.

Window: synaptic_cleft_az_hull (convex hull of presynaptic + postsynaptic AZ surface points).
No null-model controls — observed L₁₂ curves only, with pooled mean ± SD/SEM for Prism.
"""

from __future__ import annotations

import json
from dataclasses import replace as dataclasses_replace
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .alignment_utils import require_alignment_dir
from .ripley_library import (
    DEFAULT_ANALYSIS_SEED,
    DEFAULT_RIPLEY_R_STEP_NM,
    RipleyWindow3D,
    _angle_betweenness_mask,
    _intensity_weighted_combination,
    _isotropic_edge_factors_grid,
    _points_inside_hull,
    _prism_sd_envelope_columns,
    _ripley_r_grid,
    _safe_name,
    build_ripley_window_3d,
    build_window_grid_points,
    cross_k12_3d_isotropic,
    curves_matrix_to_long_dataframe,
    g_shell_reliability_mask,
    load_synaptic_cleft_cleft_points,
    pair_correlation_from_k_diff,
    plot_ripley_window_geometry_diagnostic,
    prism_sd_envelope_columns_from_averaged_k12,
    ripley_l12,
)

WINDOW_MODE = "synaptic_cleft_az_hull"
MIN_AUNP_PARTNERS = 3
# Keep aligned with ``ripley_library.DEFAULT_RIPLEY_R_MAX_NM`` (curves saved to this r_max).
AZ_CENTER_RIPLEY_R_MAX_NM = 500.0
AZ_CENTER_EDGE_GRID_SPACING_NM = 2.0
# Stored/pooled g bin width: fine enough (1nm, matching the L12 r-step) that pooling across
# many zones' shells still has real statistical power at each shell. g is computed as a
# finite difference of the already edge-corrected K curve (pair_correlation_from_k_diff),
# so — unlike an independent shell-count estimator — this doesn't need a separate
# reliability threshold; it inherits K's own edge correction automatically.
AZ_CENTER_G12_SHELL_WIDTH_NM = 1.0
# A single zone's g at the fine 1nm width is too noisy to plot on its own — the per-zone
# diagnostic figure recomputes g at this coarser width directly from the same K curve
# (pair_correlation_from_k_diff again, just with a larger bin_width_nm) for display only;
# stored/pooled data stays at 1nm.
AZ_CENTER_G12_DISPLAY_BIN_WIDTH_NM = 10.0
# Pooled Prism g exports rebinned to this width (finer 1 nm shells are hard to read in Prism).
AZ_CENTER_G12_PRISM_BIN_WIDTH_NM = 5.0

# (pooled-table column prefix, raw L-column name, raw K-column name) for each of the three
# K/L families: the direct K12/L12 (center-as-focus), the reversed K21/L21 (AuNPs-as-foci),
# and their intensity-weighted combination (see cross_k_bivariate_symmetric_3d_isotropic).
L_CURVE_FAMILIES: tuple[tuple[str, str, str], ...] = (
    ("center_L12", "l12", "k12"),
    ("center_L21", "l21", "k21"),
    ("center_L_combined", "l_combined", "k_combined"),
)

G_CURVE_FAMILIES: tuple[str, ...] = ("g12", "g21", "g_combined")

POOLED_CURVES_CSV = Path("results/aunps/aunp_vs_az_center_ripley_l12_curves.csv")
POOLED_PRISM_CSV = Path("results/aunps/aunp_vs_az_center_ripley_l12_prism_pooled.csv")
POOLED_FIGURES_DIR = Path("results/aunps/figures/aunp_vs_az_center_ripley_l12_pooled")
POOLED_G12_CSV = Path("results/aunps/aunp_vs_az_center_ripley_g12_shells_curves.csv")
POOLED_G12_POOLED_CSV = Path("results/aunps/aunp_vs_az_center_ripley_g12_pooled.csv")


def _normalize_pool_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure ``set_name`` / ``aunp_pick_label`` exist for pooled exports."""
    out = df.copy()
    if "set_name" not in out.columns:
        out["set_name"] = ""
    out["set_name"] = out["set_name"].fillna("").astype(str)
    if "aunp_pick_label" not in out.columns:
        out["aunp_pick_label"] = ""
    else:
        out["aunp_pick_label"] = out["aunp_pick_label"].fillna("").astype(str).str.strip()
    return out


def _pool_group_columns(df: pd.DataFrame) -> list[str]:
    """Group keys for pooled tables: ``set_name``, plus ``aunp_pick_label`` when labeled."""
    df = _normalize_pool_groups(df)
    cols = ["set_name"]
    labels = set(df["aunp_pick_label"].astype(str).str.strip())
    labels.discard("")
    if labels:
        cols.append("aunp_pick_label")
    return cols


def _pool_group_tags(set_name: str, pick_label: str) -> tuple[str, str]:
    set_tag = _safe_name(str(set_name)) or "all_sets"
    pick_tag = _safe_name(str(pick_label)) if str(pick_label).strip() else "all_picks"
    return set_tag, pick_tag


def _unpack_pool_group_key(key) -> tuple[str, str]:
    if isinstance(key, tuple):
        set_name = str(key[0])
        pick_label = str(key[1]) if len(key) > 1 else ""
    else:
        set_name = str(key)
        pick_label = ""
    return set_name, pick_label


def _curve_id_columns(df: pd.DataFrame) -> list[str]:
    id_cols = ["tomogram_name", "alignment_dir", "cleft_name"]
    df_norm = _normalize_pool_groups(df)
    labels = set(df_norm["aunp_pick_label"].astype(str).str.strip())
    labels.discard("")
    if labels:
        id_cols.append("aunp_pick_label")
    return id_cols


def _g_prism_bin_tag(bin_width_nm: float) -> str:
    if bin_width_nm == int(bin_width_nm):
        return _safe_name(f"{int(bin_width_nm)}nm")
    return _safe_name(f"{bin_width_nm}nm")


def _coarse_g_shell_edges(
    r_lo_fine: np.ndarray,
    r_hi_fine: np.ndarray,
    coarse_bin_width_nm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r_lo_fine = np.asarray(r_lo_fine, dtype=float)
    r_hi_fine = np.asarray(r_hi_fine, dtype=float)
    if len(r_lo_fine) == 0:
        empty = np.array([])
        return empty, empty, empty

    r_max = float(np.nanmax(r_hi_fine))
    coarse = float(coarse_bin_width_nm)
    if coarse <= 0 or r_max <= 0:
        return np.array([]), np.array([]), np.array([])

    n_bins = int(np.floor(r_max / coarse))
    if n_bins < 1:
        return np.array([]), np.array([]), np.array([])

    coarse_lo = np.arange(n_bins, dtype=float) * coarse
    coarse_hi = coarse_lo + coarse
    coarse_mid = 0.5 * (coarse_lo + coarse_hi)
    return coarse_lo, coarse_hi, coarse_mid


def _rebin_g_shell_values(
    r_lo_fine: np.ndarray,
    r_hi_fine: np.ndarray,
    g_fine: np.ndarray,
    coarse_bin_width_nm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    coarse_lo, coarse_hi, coarse_mid = _coarse_g_shell_edges(
        r_lo_fine, r_hi_fine, coarse_bin_width_nm
    )
    if len(coarse_lo) == 0:
        return coarse_mid, coarse_lo, coarse_hi, np.array([])

    g_fine = np.asarray(g_fine, dtype=float)
    rebinned = np.full(len(coarse_lo), np.nan, dtype=float)
    for i, (clo, chi) in enumerate(zip(coarse_lo, coarse_hi)):
        overlap = (r_lo_fine < chi) & (r_hi_fine > clo) & np.isfinite(g_fine)
        if not np.any(overlap):
            continue
        with np.errstate(invalid="ignore"):
            rebinned[i] = np.nanmean(g_fine[overlap])
    return coarse_mid, coarse_lo, coarse_hi, rebinned


def rebin_g_curves_table(
    df: pd.DataFrame,
    *,
    coarse_bin_width_nm: float,
) -> pd.DataFrame:
    """Rebin each tomogram×zone g curve to wider shells (display-oriented nanmean)."""
    if df.empty:
        return df.copy()

    id_cols = _curve_id_columns(df)
    for col in id_cols:
        if col not in df.columns:
            df = df.copy()
            df[col] = ""

    meta_cols = [
        c
        for c in df.columns
        if c not in id_cols + ["r_nm", "r_lo_nm", "r_hi_nm"] + list(G_CURVE_FAMILIES)
    ]
    rows: list[dict] = []

    for keys, grp in df.groupby(id_cols, sort=False):
        grp = grp.sort_values("r_nm")
        r_lo = grp["r_lo_nm"].to_numpy(dtype=float)
        r_hi = grp["r_hi_nm"].to_numpy(dtype=float)
        meta = {c: grp[c].iloc[0] for c in meta_cols if c in grp.columns}

        coarse_lo, coarse_hi, coarse_mid = _coarse_g_shell_edges(r_lo, r_hi, coarse_bin_width_nm)
        if len(coarse_mid) == 0:
            continue

        rebinned_g: dict[str, np.ndarray] = {}
        for stem in G_CURVE_FAMILIES:
            if stem not in grp.columns:
                continue
            _, _, _, rebinned_g[stem] = _rebin_g_shell_values(
                r_lo, r_hi, grp[stem].to_numpy(dtype=float), coarse_bin_width_nm
            )

        key_tuple = keys if isinstance(keys, tuple) else (keys,)
        for i in range(len(coarse_mid)):
            row = {**meta, "g_bin_width_nm": float(coarse_bin_width_nm)}
            for col, val in zip(id_cols, key_tuple):
                row[col] = val
            row.update(
                {
                    "r_nm": float(coarse_mid[i]),
                    "r_lo_nm": float(coarse_lo[i]),
                    "r_hi_nm": float(coarse_hi[i]),
                }
            )
            for stem, vals in rebinned_g.items():
                row[stem] = float(vals[i]) if i < len(vals) else np.nan
            rows.append(row)

    return pd.DataFrame(rows)


PRISM_AGG_STAT_SUFFIXES: tuple[str, ...] = (
    "_mean",
    "_sd",
    "_sd_envelope_lo",
    "_sd_envelope_hi",
    "_sem",
    "_sem_envelope_lo",
    "_sem_envelope_hi",
)


def _infer_r_shell_edges_from_r_nm(r_nm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Infer 1 nm shell edges when only distance centers are stored (``r_nm``)."""
    r_nm = np.asarray(r_nm, dtype=float)
    half = 0.5 * DEFAULT_RIPLEY_R_STEP_NM
    return r_nm - half, r_nm + half


def _prism_aggregated_stat_columns(df: pd.DataFrame) -> list[str]:
    stat_cols: list[str] = []
    for col in df.columns:
        if any(col.endswith(suffix) for suffix in PRISM_AGG_STAT_SUFFIXES):
            stat_cols.append(col)
    return stat_cols


def rebin_pooled_prism_aggregated_table(
    df: pd.DataFrame,
    *,
    coarse_bin_width_nm: float,
) -> pd.DataFrame:
    """
    Rebin an already-pooled Prism table (mean ± SD/SEM per ``r_nm``) to wider distance bins.

    Intended for converting combined multi-set exports such as
    ``aunp_vs_az_center_ripley_l12_prism_pooled.csv`` into coarser bins for Prism plotting.
    Stat columns are NaN-averaged within each coarse shell overlap region.
    """
    if df.empty or "r_nm" not in df.columns:
        return df.copy()

    r_lo, r_hi = _infer_r_shell_edges_from_r_nm(df["r_nm"].to_numpy(dtype=float))
    coarse_lo, coarse_hi, coarse_mid = _coarse_g_shell_edges(
        r_lo, r_hi, coarse_bin_width_nm
    )
    if len(coarse_mid) == 0:
        return pd.DataFrame()

    stat_cols = _prism_aggregated_stat_columns(df)
    meta_cols = [
        c
        for c in df.columns
        if c not in stat_cols + ["r_nm", "r_lo_nm", "r_hi_nm"]
    ]
    count_cols = [
        c
        for c in ("n_zone_curves", "n_zone_shells", "n_tomograms", "n_clefts")
        if c in df.columns
    ]

    rows: list[dict] = []
    for i, (clo, chi, cmid) in enumerate(zip(coarse_lo, coarse_hi, coarse_mid)):
        overlap = (r_lo < chi) & (r_hi > clo)
        if not np.any(overlap):
            continue
        chunk = df.loc[overlap]
        row = {c: chunk[c].iloc[0] for c in meta_cols if c in chunk.columns}
        row["r_nm"] = float(cmid)
        if "r_lo_nm" in df.columns:
            row["r_lo_nm"] = float(clo)
        if "r_hi_nm" in df.columns:
            row["r_hi_nm"] = float(chi)
        for c in count_cols:
            row[c] = float(np.nanmean(chunk[c].to_numpy(dtype=float)))
        for c in stat_cols:
            row[c] = float(np.nanmean(chunk[c].to_numpy(dtype=float)))
        rows.append(row)

    return pd.DataFrame(rows)


def write_pooled_az_center_l_prism_exports(
    prism_df: pd.DataFrame,
    *,
    out_dir: Path,
    coarse_bin_width_nm: float = AZ_CENTER_G12_PRISM_BIN_WIDTH_NM,
) -> list[Path]:
    """
    Write per-set (and per pick-label when present) L₁₂ Prism tables rebinned to
    ``coarse_bin_width_nm`` from an aggregated pooled Prism table.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bin_tag = _g_prism_bin_tag(coarse_bin_width_nm)
    written: list[Path] = []

    df = _normalize_pool_groups(prism_df)
    for key, sub in df.groupby(_pool_group_columns(df), sort=False):
        set_name, pick_label = _unpack_pool_group_key(key)
        set_tag, pick_tag = _pool_group_tags(set_name, pick_label)

        table = rebin_pooled_prism_aggregated_table(
            sub, coarse_bin_width_nm=coarse_bin_width_nm
        )
        if table.empty:
            continue

        pooled_path = out_dir / (
            f"aunp_vs_az_center_ripley_l12_prism_pooled_{set_tag}_{pick_tag}_{bin_tag}.csv"
        )
        table.to_csv(pooled_path, index=False)
        written.append(pooled_path)

    return written


def write_pooled_az_center_g12_aggregated_exports(
    pooled_df: pd.DataFrame,
    *,
    out_dir: Path,
    coarse_bin_width_nm: float = AZ_CENTER_G12_PRISM_BIN_WIDTH_NM,
) -> list[Path]:
    """
    Write per-set (and per pick-label when present) g₁₂ Prism tables rebinned to
    ``coarse_bin_width_nm`` from an aggregated pooled table such as
    ``aunp_vs_az_center_ripley_g12_pooled.csv``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bin_tag = _g_prism_bin_tag(coarse_bin_width_nm)
    written: list[Path] = []

    df = _normalize_pool_groups(pooled_df)
    for key, sub in df.groupby(_pool_group_columns(df), sort=False):
        set_name, pick_label = _unpack_pool_group_key(key)
        set_tag, pick_tag = _pool_group_tags(set_name, pick_label)

        table = rebin_pooled_prism_aggregated_table(
            sub, coarse_bin_width_nm=coarse_bin_width_nm
        )
        if table.empty:
            continue

        pooled_path = out_dir / (
            f"aunp_vs_az_center_ripley_g12_pooled_{set_tag}_{pick_tag}_{bin_tag}.csv"
        )
        table.to_csv(pooled_path, index=False)
        written.append(pooled_path)

    return written


def write_pooled_az_center_g_prism_exports(
    df: pd.DataFrame,
    *,
    out_dir: Path,
    coarse_bin_width_nm: float = AZ_CENTER_G12_PRISM_BIN_WIDTH_NM,
) -> list[Path]:
    """
    Write per-set (and per pick-label when present) Prism tables at ``coarse_bin_width_nm``.

    Each pool group gets its own pooled mean ± SEM CSV plus optional individual-wide g tables.
  """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bin_tag = _g_prism_bin_tag(coarse_bin_width_nm)
    written: list[Path] = []

    df = _normalize_pool_groups(df)
    for key, sub in df.groupby(_pool_group_columns(df), sort=False):
        set_name, pick_label = _unpack_pool_group_key(key)
        set_tag, pick_tag = _pool_group_tags(set_name, pick_label)

        table = build_pooled_aunp_vs_az_center_g12_table(sub)
        if table.empty:
            continue

        pooled_path = out_dir / (
            f"aunp_vs_az_center_ripley_g12_prism_pooled_{set_tag}_{pick_tag}_{bin_tag}.csv"
        )
        table.to_csv(pooled_path, index=False)
        written.append(pooled_path)

        for file_stem, value_col in [
            ("g12", "g12"),
            ("g21", "g21"),
            ("g_combined", "g_combined"),
        ]:
            if value_col not in sub.columns:
                continue
            wide = build_pooled_az_center_individual_wide_table(sub, value_col=value_col)
            if wide.shape[1] <= 1:
                continue
            wide_path = out_dir / (
                f"aunp_vs_az_center_ripley_{file_stem}_individual_wide_{set_tag}_{pick_tag}_{bin_tag}.csv"
            )
            wide.to_csv(wide_path, index=False)
            written.append(wide_path)

    return written


def build_pooled_az_center_individual_wide_table(
    sub: pd.DataFrame,
    *,
    value_col: str,
) -> pd.DataFrame:
    """Prism XY: one row per ``r_nm``, one column per tomogram+zone curve."""
    if sub.empty or value_col not in sub.columns:
        return pd.DataFrame()

    sub = sub.copy()
    r_vals = np.sort(sub["r_nm"].unique())
    n_r = len(r_vals)
    r_index = {round(float(r), 6): i for i, r in enumerate(r_vals)}
    id_cols = ["tomogram_name", "alignment_dir", "cleft_name"]
    for col in id_cols:
        if col not in sub.columns:
            sub[col] = ""

    data: dict[str, np.ndarray] = {"r_nm": r_vals}
    for keys, grp in sub.groupby(id_cols, sort=False):
        tomogram_name = str(keys[0])
        cleft_name = str(keys[2]) if len(keys) > 2 else str(keys[-1])
        col = _safe_name(f"{tomogram_name}_{cleft_name}")
        curve = np.full(n_r, np.nan)
        for r_nm, value in zip(
            grp["r_nm"].to_numpy(dtype=float), grp[value_col].to_numpy(dtype=float)
        ):
            idx = r_index.get(round(float(r_nm), 6))
            if idx is not None:
                curve[idx] = value
        data[col] = curve
    return pd.DataFrame(data)


def _write_pooled_az_center_individual_wide_exports(
    df: pd.DataFrame,
    *,
    out_dir: Path,
) -> list[Path]:
    """Write per-set/per-pick-label wide curve tables for Prism (l12, l21, l_combined)."""
    df = _normalize_pool_groups(df)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    family_cols = [("l12", "l12"), ("l21", "l21"), ("l_combined", "l_combined")]
    for key, sub in df.groupby(_pool_group_columns(df), sort=False):
        set_name, pick_label = _unpack_pool_group_key(key)
        set_tag, pick_tag = _pool_group_tags(set_name, pick_label)
        for file_stem, value_col in family_cols:
            if value_col not in sub.columns:
                continue
            wide = build_pooled_az_center_individual_wide_table(sub, value_col=value_col)
            if wide.shape[1] <= 1:
                continue
            path = out_dir / (
                f"aunp_vs_az_center_ripley_{file_stem}_individual_wide_{set_tag}_{pick_tag}.csv"
            )
            wide.to_csv(path, index=False)
            written.append(path)
    return written


_AZ_CENTER_MAX_ITER = 5
_AZ_CENTER_CONVERGENCE_TOL_NM = 1e-3


def compute_cleft_center_nm(az_segmentation: dict) -> np.ndarray:
    """
    Active zone center: the midpoint between the nearest presynaptic and postsynaptic
    active-zone (outer, cleft-facing) surface points to a running centroid estimate,
    refined by fixed-point iteration.

    A plain centroid of the two point clouds can drift outside the cleft (or hug one
    membrane) when the pre- and post-synaptic patches differ in shape or extent. Pairing
    the nearest pre/post points and taking their midpoint always lands on the segment
    joining them, so the result passes the same angle in-betweenness test used for the
    Ripley window (vectors from the center to its nearest pre- and post-membrane points
    point away from each other).
    """
    pre = az_segmentation.get("presynaptic_outer_coords")
    post = az_segmentation.get("postsynaptic_outer_coords")
    if pre is None or post is None or len(pre) == 0 or len(post) == 0:
        return np.full(3, np.nan)
    pre = np.atleast_2d(np.asarray(pre, dtype=float))
    post = np.atleast_2d(np.asarray(post, dtype=float))

    pre_tree = cKDTree(pre)
    post_tree = cKDTree(post)

    center = np.mean(np.vstack([pre, post]), axis=0)
    for _ in range(_AZ_CENTER_MAX_ITER):
        _, pre_idx = pre_tree.query(center)
        _, post_idx = post_tree.query(center)
        new_center = (pre[pre_idx] + post[post_idx]) / 2.0
        converged = np.linalg.norm(new_center - center) < _AZ_CENTER_CONVERGENCE_TOL_NM
        center = new_center
        if converged:
            break

    if not _angle_betweenness_mask(center.reshape(1, 3), pre, post)[0]:
        print(
            "  Warning: synaptic-cleft center failed the in-betweenness test after "
            f"{_AZ_CENTER_MAX_ITER} fixed-point iterations; using nearest pre/post "
            "midpoint anyway"
        )
    return center


def _extract_zone_curves_matrix(
    df: pd.DataFrame,
    value_col: str = "l12",
) -> tuple[np.ndarray, np.ndarray]:
    """Pivot long table to (r_vals, curves) with one curve per tomogram+zone.

    Zones that stop early (the r grid is truncated once it covers all AuNP partners) or
    that skip an unreliable shell contribute NaN at the r values they don't report, rather
    than being dropped outright — downstream aggregation uses NaN-aware statistics so each
    shell is summarized from however many zones actually reached it.
    """
    if df.empty or value_col not in df.columns:
        return np.array([]), np.empty((0, 0))

    sub = df.copy()
    r_vals = np.sort(sub["r_nm"].unique())
    n_r = len(r_vals)
    r_index = {round(float(r), 6): i for i, r in enumerate(r_vals)}
    id_cols = ["tomogram_name", "alignment_dir", "cleft_name"]
    for col in id_cols:
        if col not in sub.columns:
            sub[col] = ""

    curves: list[np.ndarray] = []
    for _, grp in sub.groupby(id_cols, sort=False):
        if grp.empty:
            continue
        curve = np.full(n_r, np.nan)
        for r_nm, value in zip(
            grp["r_nm"].to_numpy(dtype=float), grp[value_col].to_numpy(dtype=float)
        ):
            idx = r_index.get(round(float(r_nm), 6))
            if idx is not None:
                curve[idx] = value
        curves.append(curve)

    if not curves:
        return r_vals, np.empty((0, n_r))
    return r_vals, np.vstack(curves)


def _extract_zone_l12_curves_matrix(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Pivot long L₁₂ table to (r_vals, curves) with one curve per tomogram+zone."""
    return _extract_zone_curves_matrix(df, value_col="l12")


def build_aunp_vs_az_center_prism_table(
    *,
    zone_name: str,
    r_vals: np.ndarray,
    k12: np.ndarray,
    l12: np.ndarray,
    k21: np.ndarray,
    l21: np.ndarray,
    k_combined: np.ndarray,
    l_combined: np.ndarray,
    n_aunps: int,
    window_volume_nm3: float,
) -> pd.DataFrame:
    """Per-zone Prism table for all three K/L families (``L_CURVE_FAMILIES``): one observed
    curve each, so their SD/SEM/envelope columns are NaN for a single curve."""
    l_values = {"l12": l12, "l21": l21, "l_combined": l_combined}
    rows: list[dict] = []
    for i, r_nm in enumerate(r_vals):
        row = {
            "cleft_name": zone_name,
            "window_mode": WINDOW_MODE,
            "r_nm": float(r_nm),
            "k12": float(k12[i]),
            "k21": float(k21[i]),
            "k_combined": float(k_combined[i]),
        }
        for prefix, l_col, _ in L_CURVE_FAMILIES:
            l_val = float(l_values[l_col][i])
            row.update(
                {
                    prefix: l_val,
                    f"{prefix}_mean": l_val,
                    f"{prefix}_sd": np.nan,
                    f"{prefix}_sd_envelope_lo": np.nan,
                    f"{prefix}_sd_envelope_hi": np.nan,
                    f"{prefix}_sem": np.nan,
                    f"{prefix}_sem_envelope_lo": np.nan,
                    f"{prefix}_sem_envelope_hi": np.nan,
                }
            )
        row["n_aunp_partners"] = int(n_aunps)
        row["window_volume_nm3"] = float(window_volume_nm3)
        rows.append(row)
    return pd.DataFrame(rows)


def _pooled_l_family_columns(
    sub: pd.DataFrame,
    *,
    prefix: str,
    l_col: str,
    k_col: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Pooled K→L envelope columns for one L-family curve (a row of ``L_CURVE_FAMILIES``).

    Returns ``(r_vals, n_valid_per_r, from_k)``; ``from_k`` is empty if ``l_col`` isn't
    present in ``sub`` (e.g. older per-zone output written before that family existed).
    """
    r_vals, l_curves = _extract_zone_curves_matrix(sub, value_col=l_col)
    if len(l_curves) == 0:
        return r_vals, np.array([]), {}
    k_curves = None
    if k_col in sub.columns:
        _, k_curves_candidate = _extract_zone_curves_matrix(sub, value_col=k_col)
        if len(k_curves_candidate) == len(l_curves):
            k_curves = k_curves_candidate
    from_k = prism_sd_envelope_columns_from_averaged_k12(
        l_curves, r_vals, prefix=prefix, k12_curves=k_curves
    )
    n_valid_per_r = np.sum(~np.isnan(l_curves), axis=0)
    return r_vals, n_valid_per_r, from_k


def build_pooled_aunp_vs_az_center_prism_table(df: pd.DataFrame) -> pd.DataFrame:
    """Pooled mean ± SD/SEM of H₁₂ (Ripley L-transform) across tomogram-zone curves, per set,
    for each of the three K/L families (``L_CURVE_FAMILIES``): direct K₁₂/L₁₂
    (``center_L12_*``, center-as-focus — the original statistic), reversed K₂₁/L₂₁
    (``center_L21_*``, AuNPs-as-foci), and their intensity-weighted combination
    (``center_L_combined_*``, see ``cross_k_bivariate_symmetric_3d_isotropic``).

    Averaging is done on the K scale first, then the mean (and mean ± SD/SEM) is mapped
    through the H = (3K/4π)^(1/3) - r transform into the primary ``*_mean``/``*_sd``/``*_sem``
    columns (uses the family's stored K column when present). A family missing from ``df``
    (e.g. pooling older per-zone output written before it existed) is simply omitted from
    the row rather than filled with NaN placeholders.
    """
    if df.empty:
        return pd.DataFrame()

    df = _normalize_pool_groups(df)

    rows: list[dict] = []
    for key, sub in df.groupby(_pool_group_columns(df), sort=False):
        set_name, pick_label = _unpack_pool_group_key(key)
        anchor_r_vals: np.ndarray | None = None
        anchor_n_valid: np.ndarray | None = None
        family_envelopes: dict[str, dict[str, np.ndarray]] = {}

        for prefix, l_col, k_col in L_CURVE_FAMILIES:
            r_vals, n_valid_per_r, from_k = _pooled_l_family_columns(
                sub, prefix=prefix, l_col=l_col, k_col=k_col
            )
            if not from_k:
                continue
            family_envelopes[prefix] = from_k
            if anchor_r_vals is None:
                anchor_r_vals = r_vals
                anchor_n_valid = n_valid_per_r

        if anchor_r_vals is None:
            continue

        n_tomograms = int(sub["tomogram_name"].nunique()) if "tomogram_name" in sub.columns else 0
        n_zones = int(
            sub[["tomogram_name", "alignment_dir", "cleft_name"]].drop_duplicates().shape[0]
        )

        for i, r_nm in enumerate(anchor_r_vals):
            row = {
                "set_name": set_name,
                "aunp_pick_label": pick_label,
                "window_mode": WINDOW_MODE,
                "r_nm": float(r_nm),
                "n_zone_curves": int(anchor_n_valid[i]),
                "n_tomograms": n_tomograms,
                "n_clefts": n_zones,
            }
            for prefix, from_k in family_envelopes.items():
                row.update(
                    {
                        f"{prefix}_mean": float(from_k[f"{prefix}_mean"][i]),
                        f"{prefix}_sd": float(from_k[f"{prefix}_sd"][i]),
                        f"{prefix}_sd_envelope_lo": float(from_k[f"{prefix}_sd_envelope_lo"][i]),
                        f"{prefix}_sd_envelope_hi": float(from_k[f"{prefix}_sd_envelope_hi"][i]),
                        f"{prefix}_sem": float(from_k[f"{prefix}_sem"][i]),
                        f"{prefix}_sem_envelope_lo": float(from_k[f"{prefix}_sem_envelope_lo"][i]),
                        f"{prefix}_sem_envelope_hi": float(from_k[f"{prefix}_sem_envelope_hi"][i]),
                    }
                )
            rows.append(row)
    return pd.DataFrame(rows)


def build_pooled_aunp_vs_az_center_g12_table(df: pd.DataFrame) -> pd.DataFrame:
    """Pooled mean ± SD/SEM of g (local shell density) across tomogram-zone shells, per set,
    for each of ``G_CURVE_FAMILIES``: direct g₁₂ (center-as-focus, the original statistic),
    reversed g₂₁ (AuNPs-as-foci), and their intensity-weighted combination g_combined (see
    ``_intensity_weighted_combination``).

    Unlike L, g is a linear ratio (observed/expected shell count) rather than a nonlinear
    transform of K, so pooling directly on g carries no Jensen-inequality bias — straight
    NaN-aware mean/SD/SEM per shell is already the statistically sound aggregate for each
    family. A family missing from ``df`` (e.g. pooling older per-zone output written before
    g₂₁/g_combined existed) is simply omitted from the row rather than filled with NaN
    placeholders.
    """
    if df.empty:
        return pd.DataFrame()

    df = _normalize_pool_groups(df)

    rows: list[dict] = []
    for key, sub in df.groupby(_pool_group_columns(df), sort=False):
        set_name, pick_label = _unpack_pool_group_key(key)
        anchor_r_vals: np.ndarray | None = None
        anchor_n_valid: np.ndarray | None = None
        family_envelopes: dict[str, dict[str, np.ndarray]] = {}

        for stem in G_CURVE_FAMILIES:
            r_vals, curves = _extract_zone_curves_matrix(sub, value_col=stem)
            if len(curves) == 0:
                continue
            family_envelopes[stem] = _prism_sd_envelope_columns(curves, r_vals, prefix=stem)
            if anchor_r_vals is None:
                anchor_r_vals = r_vals
                anchor_n_valid = np.sum(~np.isnan(curves), axis=0)

        if anchor_r_vals is None:
            continue

        edges = _extract_zone_curves_matrix(sub, value_col="r_lo_nm")[1]
        r_lo = np.nanmean(edges, axis=0) if len(edges) else np.full(len(anchor_r_vals), np.nan)
        edges = _extract_zone_curves_matrix(sub, value_col="r_hi_nm")[1]
        r_hi = np.nanmean(edges, axis=0) if len(edges) else np.full(len(anchor_r_vals), np.nan)
        n_tomograms = int(sub["tomogram_name"].nunique()) if "tomogram_name" in sub.columns else 0
        n_zones = int(
            sub[["tomogram_name", "alignment_dir", "cleft_name"]].drop_duplicates().shape[0]
        )

        for i, r_nm in enumerate(anchor_r_vals):
            row = {
                "set_name": set_name,
                "aunp_pick_label": pick_label,
                "window_mode": WINDOW_MODE,
                "r_nm": float(r_nm),
                "r_lo_nm": float(r_lo[i]),
                "r_hi_nm": float(r_hi[i]),
                "n_zone_shells": int(anchor_n_valid[i]),
                "n_tomograms": n_tomograms,
                "n_clefts": n_zones,
            }
            for stem, env in family_envelopes.items():
                row.update(
                    {
                        f"{stem}_mean": float(env[f"{stem}_mean"][i]),
                        f"{stem}_sd": float(env[f"{stem}_sd"][i]),
                        f"{stem}_sd_envelope_lo": float(env[f"{stem}_sd_envelope_lo"][i]),
                        f"{stem}_sd_envelope_hi": float(env[f"{stem}_sd_envelope_hi"][i]),
                        f"{stem}_sem": float(env[f"{stem}_sem"][i]),
                        f"{stem}_sem_envelope_lo": float(env[f"{stem}_sem_envelope_lo"][i]),
                        f"{stem}_sem_envelope_hi": float(env[f"{stem}_sem_envelope_hi"][i]),
                    }
                )
            rows.append(row)
    return pd.DataFrame(rows)


def plot_cleft_center_diagnostic(
    tomogram_path: Path,
    alignment_dir: str,
    zone_name: str,
    *,
    aunp_coords: np.ndarray,
    az_segmentation: dict,
    window: RipleyWindow3D,
    grid_points: np.ndarray,
    grid_spacing_nm: float,
    output_path: Path,
    dropped_coords: np.ndarray | None = None,
    membrane_max_points: int = 4000,
    n_z_slices: int = 5,
) -> Path | None:
    """
    QC figure: AZ center, pre/post membrane surfaces, AuNP positions, the
    ``synaptic_cleft_az_hull`` convex hull used as the Ripley window, and the deterministic
    edge-correction grid points (from ``build_window_grid_points``) that are the actual
    sample set the analysis divides by — not a separate Monte-Carlo preview of it, but the
    literal points used, so this figure always matches what the computation really did.

    Thin wrapper around ``plot_ripley_window_geometry_diagnostic`` (shared with the
    monomer/dimer analysis) with a single "AuNPs" point group and the computed AZ/cleft
    center overlaid as a star marker, plus the 3D panel for directly judging whether the
    computed center sits correctly between the two membranes.

    ``dropped_coords`` (AuNPs outside the cleft hull, excluded from the Ripley analysis)
    are highlighted separately, distinct from the analyzed AuNPs.
    """
    aunp_coords = np.atleast_2d(np.asarray(aunp_coords, dtype=float))
    center = compute_cleft_center_nm(az_segmentation)
    if not np.all(np.isfinite(center)):
        print(f"  Skipping AZ-center diagnostic plot for {zone_name}: could not compute center")
        return None

    pre_outer = np.atleast_2d(np.asarray(az_segmentation.get("presynaptic_outer_coords", []), dtype=float))
    post_outer = np.atleast_2d(np.asarray(az_segmentation.get("postsynaptic_outer_coords", []), dtype=float))

    return plot_ripley_window_geometry_diagnostic(
        tomogram_path,
        alignment_dir,
        zone_name,
        point_groups=[{"coords": aunp_coords, "label": "AuNPs", "color": "tab:red", "size": 18}],
        az_segmentation=az_segmentation,
        window=window,
        grid_points=grid_points,
        grid_spacing_nm=grid_spacing_nm,
        output_path=output_path,
        dropped_coords=dropped_coords,
        center_point=center,
        center_label="AZ/cleft center",
        title_lines=[
            f"AZ center vs membranes & AuNPs ({len(aunp_coords)} AuNPs, "
            f"{len(pre_outer)}+{len(post_outer)} membrane pts)"
        ],
        membrane_max_points=membrane_max_points,
        n_z_slices=n_z_slices,
        include_3d_panel=True,
        print_prefix="AZ-center diagnostic plot",
    )


# Display label per g-family stem, used in per-zone diagnostic figure titles/axis labels.
G_FAMILY_DISPLAY_LABEL: dict[str, str] = {
    "g12": "g₁₂",
    "g21": "g₂₁",
    "g_combined": "g_combined",
}


def _plot_g_family_diagnostic(
    r_mid_nm: np.ndarray,
    g_vals: np.ndarray,
    *,
    stem: str,
    display_bin_width_nm: float,
    computed_shell_width_nm: float,
    n_aunps: int,
    tomogram_name: str,
    zone_name: str,
    figures_dir: Path,
    r_max_nm: float,
) -> Path:
    """One per-zone g-family diagnostic figure (already rebinned to ``display_bin_width_nm``
    by the caller — see ``rebin_g12_shells``); a single zone's shells at the fine stored
    resolution are too noisy to plot meaningfully."""
    label = G_FAMILY_DISPLAY_LABEL.get(stem, stem)
    n_reliable = int(np.sum(np.isfinite(g_vals)))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(
        r_mid_nm, g_vals, color="C1", lw=1.5, marker="o", ms=3,
        label=f"Observed {label} (shell density)",
    )
    ax.axhline(1.0, color="0.5", ls="--", lw=0.8, label="CSR (g = 1)")
    ax.set_xlabel("r (nm)")
    ax.set_ylabel(f"{label}(r) = observed / expected AuNP shell density")
    ax.set_title(
        f"{tomogram_name} | {zone_name}\n"
        f"AuNPs vs AZ center ({label}), {display_bin_width_nm:g}nm display bins "
        f"(computed at {computed_shell_width_nm:g}nm; {n_aunps} AuNPs, "
        f"{n_reliable}/{len(g_vals)} reliable bins)"
    )
    ax.set_xlim(
        0.0,
        float(r_mid_nm[-1] + 0.5 * display_bin_width_nm) if len(r_mid_nm) else r_max_nm,
    )
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out_path = figures_dir / f"pair_correlation_{stem}_observed.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def run_aunp_vs_az_center_ripley_for_zone(
    tomogram_path: Path,
    alignment_dir: str,
    zone_name: str,
    cleft_index: int,
    *,
    aunp_coords: np.ndarray,
    az_segmentation: dict,
    r_max_nm: float = AZ_CENTER_RIPLEY_R_MAX_NM,
    r_step_nm: float = DEFAULT_RIPLEY_R_STEP_NM,
    g12_shell_width_nm: float = AZ_CENTER_G12_SHELL_WIDTH_NM,
    seed: int = DEFAULT_ANALYSIS_SEED,
    write_figures: bool = True,
) -> dict[str, Path] | None:
    """Compute observed 3D L₁₂(center, AuNPs) for one synaptic cleft.

    The Ripley window is always restricted to the region of the synaptic-cleft hull that
    also sits "between" the pre- and post-synaptic membranes (angle in-betweenness test),
    since this analysis is specifically about the space between the two membranes. AuNPs
    outside the cleft hull are always dropped before computing K₁₂/L₁₂ — Ripley's edge
    correction and volume normalization are only valid for points observed inside the
    window, so out-of-hull points would bias the statistic rather than just look odd.
    """
    tomogram_path = Path(tomogram_path)
    alignment_dir = require_alignment_dir(alignment_dir)
    aunp_coords = np.atleast_2d(np.asarray(aunp_coords, dtype=float))
    if len(aunp_coords) < MIN_AUNP_PARTNERS:
        print(
            f"  Skipping AZ-center Ripley for {zone_name}: "
            f"only {len(aunp_coords)} AuNP(s) (need >= {MIN_AUNP_PARTNERS})"
        )
        return None

    center = compute_cleft_center_nm(az_segmentation)
    if not np.all(np.isfinite(center)):
        print(f"  Skipping AZ-center Ripley for {zone_name}: could not compute center")
        return None

    rng = np.random.default_rng(seed)
    try:
        cleft_coords = load_synaptic_cleft_cleft_points(
            tomogram_path, alignment_dir, zone_name
        )
        window = build_ripley_window_3d(
            cleft_coords,
            mode=WINDOW_MODE,
            pre_membrane_coords=az_segmentation.get("presynaptic_outer_coords"),
            post_membrane_coords=az_segmentation.get("postsynaptic_outer_coords"),
            use_angle_betweenness=True,
            rng=rng,
        )
    except Exception as exc:
        print(f"  Skipping AZ-center Ripley for {zone_name}: {exc}")
        return None

    inside_hull_mask = _points_inside_hull(aunp_coords, window.hull)
    dropped_coords = aunp_coords[~inside_hull_mask]
    aunp_coords = aunp_coords[inside_hull_mask]
    if len(dropped_coords):
        print(f"  Dropping {len(dropped_coords)} AuNP(s) outside the cleft hull for {zone_name}")
    if len(aunp_coords) < MIN_AUNP_PARTNERS:
        print(
            f"  Skipping AZ-center Ripley for {zone_name}: "
            f"only {len(aunp_coords)} AuNP(s) inside the cleft hull (need >= {MIN_AUNP_PARTNERS})"
        )
        return None

    r_vals = _ripley_r_grid(r_max_nm, r_step_nm)
    center_xyz = center.reshape(1, 3)

    grid_points = build_window_grid_points(window, AZ_CENTER_EDGE_GRID_SPACING_NM)
    edge_factors = _isotropic_edge_factors_grid(
        center_xyz, r_vals, grid_points, AZ_CENTER_EDGE_GRID_SPACING_NM
    )
    # cross_k12_3d_isotropic's outer scale factor is V/n2, where V = window.volume_nm3. The
    # edge_factors above (from _isotropic_edge_factors_grid) implicitly estimate that same V
    # as len(grid_points) * spacing^3 once r covers the whole window (see that function's
    # docstring for the derivation of why V cancels out algebraically in that regime). For
    # that cancellation to be *exact* rather than leaving a residual bias, both places must
    # use the identical V — so we replace the separately-estimated (200k-sample Monte Carlo)
    # window.volume_nm3 with this grid's own volume estimate before computing K12.
    grid_volume_nm3 = float(len(grid_points)) * (AZ_CENTER_EDGE_GRID_SPACING_NM ** 3)
    window = dataclasses_replace(window, volume_nm3=grid_volume_nm3)
    k12 = cross_k12_3d_isotropic(center_xyz, aunp_coords, r_vals, window, rng, edge_factors=edge_factors)
    l12 = ripley_l12(k12, r_vals)

    # Reversed direction (AuNPs-as-foci) and their intensity-weighted combination (see
    # cross_k_bivariate_symmetric_3d_isotropic / L_CURVE_FAMILIES). Edge correction is
    # recomputed for this direction since it depends on which set is the foci (now the
    # AuNPs, one grid query per AuNP instead of the single center point).
    edge_factors_aunps = _isotropic_edge_factors_grid(
        aunp_coords, r_vals, grid_points, AZ_CENTER_EDGE_GRID_SPACING_NM
    )
    k21 = cross_k12_3d_isotropic(
        aunp_coords, center_xyz, r_vals, window, rng, edge_factors=edge_factors_aunps
    )
    k_combined = _intensity_weighted_combination(k12, k21, 1, len(aunp_coords))
    l21 = ripley_l12(k21, r_vals)
    l_combined = ripley_l12(k_combined, r_vals)

    # g at the fine AZ_CENTER_G12_SHELL_WIDTH_NM resolution (1nm by default), computed as a
    # finite difference of the already edge-corrected k12/k21 curves above
    # (pair_correlation_from_k_diff) rather than an independent shell-count estimator — see
    # that function's docstring. g_combined is the intensity-weighted combination of the two
    # directions' *ratios* (not a pooling of raw counts — g12's and g21's underlying
    # intensity scales, λ_center=1/V vs λ_aunp=n_aunp/V, differ, so only the final ratios are
    # combinable, same as k_combined/l_combined above).
    #
    # Past the radius where a focus's ball has swallowed the entire window, every K value
    # feeding a shell's difference is forced onto the CSR reference curve by construction
    # (see _isotropic_edge_factors_grid's docstring) — L correctly reflects this as L→0, but
    # g's finite difference of two such saturated K values goes spuriously flat near 1
    # instead of NaN, since pair_correlation_from_k_diff has no visibility into how much
    # window is actually left. g_shell_reliability_mask/mask_unreliable_g_shells catch this
    # by NaN'ing shells with too few supporting window-quadrature grid points, mirroring the
    # reliability threshold the retired independent shell-count g estimator used.
    g12_result = pair_correlation_from_k_diff(k12, r_vals, bin_width_nm=g12_shell_width_nm)
    g21_result = pair_correlation_from_k_diff(k21, r_vals, bin_width_nm=g12_shell_width_nm)
    g12_r_nm = g12_result["r_mid_nm"]
    g12_unreliable = g_shell_reliability_mask(
        center_xyz, grid_points, r_vals, bin_width_nm=g12_shell_width_nm
    )
    g21_unreliable = g_shell_reliability_mask(
        aunp_coords, grid_points, r_vals, bin_width_nm=g12_shell_width_nm
    )
    g12 = np.where(g12_unreliable, np.nan, g12_result["pcf"])
    g21 = np.where(g21_unreliable, np.nan, g21_result["pcf"])
    g_combined = _intensity_weighted_combination(g12, g21, 1, len(aunp_coords))

    tomogram_name = tomogram_path.name
    out_dir = (
        tomogram_path
        / alignment_dir
        / "STT_results"
        / "aunps"
        / "aunp_vs_az_center_ripley"
        / zone_name
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = out_dir / "figures"
    if write_figures:
        figures_dir.mkdir(parents=True, exist_ok=True)
        try:
            plot_cleft_center_diagnostic(
                tomogram_path,
                alignment_dir,
                zone_name,
                aunp_coords=aunp_coords,
                az_segmentation=az_segmentation,
                window=window,
                grid_points=grid_points,
                grid_spacing_nm=AZ_CENTER_EDGE_GRID_SPACING_NM,
                dropped_coords=dropped_coords,
                output_path=figures_dir / "geometry_diagnostic.png",
            )
        except Exception as diag_exc:
            print(f"  Skipping AZ-center diagnostic plot for {zone_name}: {diag_exc}")

    curves_df = pd.DataFrame(
        {
            "cleft_name": zone_name,
            "cleft_index": int(cleft_index),
            "window_mode": WINDOW_MODE,
            "r_nm": r_vals,
            "k12": k12,
            "l12": l12,
            "k21": k21,
            "l21": l21,
            "k_combined": k_combined,
            "l_combined": l_combined,
            "n_aunp_partners": len(aunp_coords),
            "n_aunps_dropped_outside_hull": len(dropped_coords),
            "window_volume_nm3": float(window.volume_nm3),
            "center_x_nm": float(center[0]),
            "center_y_nm": float(center[1]),
            "center_z_nm": float(center[2]),
        }
    )
    curves_path = out_dir / "ripley_l12_curves.csv"
    curves_df.to_csv(curves_path, index=False)

    g12_df = pd.DataFrame(
        {
            "cleft_name": zone_name,
            "cleft_index": int(cleft_index),
            "window_mode": WINDOW_MODE,
            "r_nm": g12_r_nm,
            "r_lo_nm": g12_result["r_lo_nm"],
            "r_hi_nm": g12_result["r_hi_nm"],
            "g12": g12,
            "g21": g21,
            "g_combined": g_combined,
            "n_aunp_partners": len(aunp_coords),
            "window_volume_nm3": float(window.volume_nm3),
        }
    )
    g12_path = out_dir / "pair_correlation_g12_shells.csv"
    g12_df.to_csv(g12_path, index=False)

    individual_df = curves_matrix_to_long_dataframe(
        np.atleast_2d(l12),
        r_vals,
        curve_type="observed",
        extra_cols={
            "cleft_name": zone_name,
            "cleft_index": int(cleft_index),
            "window_mode": WINDOW_MODE,
            "n_aunp_partners": int(len(aunp_coords)),
            "n_aunps_dropped_outside_hull": int(len(dropped_coords)),
            "window_volume_nm3": float(window.volume_nm3),
        },
    )
    individual_path = out_dir / "ripley_l12_individual_curves.csv"
    individual_df.to_csv(individual_path, index=False)

    prism_df = build_aunp_vs_az_center_prism_table(
        zone_name=zone_name,
        r_vals=r_vals,
        k12=k12,
        l12=l12,
        k21=k21,
        l21=l21,
        k_combined=k_combined,
        l_combined=l_combined,
        n_aunps=len(aunp_coords),
        window_volume_nm3=float(window.volume_nm3),
    )
    prism_path = out_dir / "ripley_l12_prism.csv"
    prism_df.to_csv(prism_path, index=False)

    if write_figures:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(r_vals, l12, color="C0", lw=2, label="Observed L₁₂")
        ax.axhline(0.0, color="0.5", ls="--", lw=0.8)
        ax.set_xlabel("r (nm)")
        ax.set_ylabel("Ripley L₁₂(r) = (3K₁₂/4π)^(1/3) − r")
        ax.set_title(f"{tomogram_name} | {zone_name}\nAuNPs vs AZ center ({len(aunp_coords)} AuNPs)")
        ax.set_xlim(0.0, float(r_vals[-1]) if len(r_vals) else r_max_nm)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(figures_dir / "ripley_l12_observed.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        # Display-only: a single zone's g at the fine stored resolution (g12_shell_width_nm)
        # is too noisy to plot meaningfully — recompute directly from the same k12/k21 curves
        # at a coarser bin width (pair_correlation_from_k_diff again) rather than rebinning
        # the fine-resolution result, without touching the fine-resolution CSV written above.
        # g_combined_display is then the intensity-weighted combination of the two
        # *already-binned* ratios (see the g_combined computation above for why raw counts
        # can't be pooled across directions).
        g12_display = pair_correlation_from_k_diff(
            k12, r_vals, bin_width_nm=AZ_CENTER_G12_DISPLAY_BIN_WIDTH_NM
        )
        g21_display = pair_correlation_from_k_diff(
            k21, r_vals, bin_width_nm=AZ_CENTER_G12_DISPLAY_BIN_WIDTH_NM
        )
        g12_display_unreliable = g_shell_reliability_mask(
            center_xyz, grid_points, r_vals, bin_width_nm=AZ_CENTER_G12_DISPLAY_BIN_WIDTH_NM
        )
        g21_display_unreliable = g_shell_reliability_mask(
            aunp_coords, grid_points, r_vals, bin_width_nm=AZ_CENTER_G12_DISPLAY_BIN_WIDTH_NM
        )
        g12_display_vals = np.where(g12_display_unreliable, np.nan, g12_display["pcf"])
        g21_display_vals = np.where(g21_display_unreliable, np.nan, g21_display["pcf"])
        g_combined_display_vals = _intensity_weighted_combination(
            g12_display_vals, g21_display_vals, 1, len(aunp_coords)
        )
        for stem, r_mid_nm, g_vals in (
            ("g12", g12_display["r_mid_nm"], g12_display_vals),
            ("g21", g21_display["r_mid_nm"], g21_display_vals),
            ("g_combined", g12_display["r_mid_nm"], g_combined_display_vals),
        ):
            _plot_g_family_diagnostic(
                r_mid_nm,
                g_vals,
                stem=stem,
                display_bin_width_nm=AZ_CENTER_G12_DISPLAY_BIN_WIDTH_NM,
                computed_shell_width_nm=g12_shell_width_nm,
                n_aunps=len(aunp_coords),
                tomogram_name=tomogram_name,
                zone_name=zone_name,
                figures_dir=figures_dir,
                r_max_nm=r_max_nm,
            )

    meta = {
        "tomogram_name": tomogram_name,
        "alignment_dir": alignment_dir,
        "cleft_name": zone_name,
        "cleft_index": int(cleft_index),
        "window_mode": WINDOW_MODE,
        "n_aunp_partners": int(len(aunp_coords)),
        "n_aunps_dropped_outside_hull": int(len(dropped_coords)),
        "window_volume_nm3": float(window.volume_nm3),
        "center_definition": "nearest_pre_post_midpoint_betweenness_refined",
        "ripley_edge_correction": "isotropic_3d_mc",
        "window_uses_angle_betweenness": bool(window.use_angle_betweenness),
        "seed": int(seed),
        "g_estimator": "pair_correlation_from_k_diff",
        "g12_shell_width_nm": float(g12_shell_width_nm),
        "n_reliable_g12_shells": int(np.sum(np.isfinite(g12))),
        "g12_display_bin_width_nm": float(AZ_CENTER_G12_DISPLAY_BIN_WIDTH_NM),
    }
    with open(out_dir / "run_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(
        f"  AuNP vs AZ-center Ripley L₁₂ ({zone_name}): "
        f"{len(aunp_coords)} AuNPs -> {out_dir}"
    )
    return {
        "curves_path": curves_path,
        "individual_curves_path": individual_path,
        "prism_path": prism_path,
        "g12_shells_path": g12_path,
        "output_dir": out_dir,
    }


def run_aunp_vs_az_center_ripley_for_tomogram(
    tomogram_path: Path,
    alignment_dir: str,
    *,
    cleft_indices: Sequence[int] | None,
    df_valid: pd.DataFrame,
    az_segmentations: dict,
    r_max_nm: float = AZ_CENTER_RIPLEY_R_MAX_NM,
    r_step_nm: float = DEFAULT_RIPLEY_R_STEP_NM,
    g12_shell_width_nm: float = AZ_CENTER_G12_SHELL_WIDTH_NM,
    seed: int = DEFAULT_ANALYSIS_SEED,
    write_figures: bool = True,
) -> tuple[list[pd.DataFrame], list[pd.DataFrame], list[pd.DataFrame]]:
    """Run AZ-center Ripley for all mapped synaptic clefts in one tomogram."""
    from .cleft import load_cleft_mapping

    tomogram_path = Path(tomogram_path)
    alignment_dir = require_alignment_dir(alignment_dir)
    az_mapping = load_cleft_mapping(tomogram_path, alignment_dir) or {}
    if not az_mapping:
        print("No synaptic cleft mapping; skipping AuNP vs AZ-center Ripley analyses")
        return [], [], []

    az_mapping = {int(k): v for k, v in az_mapping.items()}
    indices = list(cleft_indices) if cleft_indices is not None else sorted(az_mapping)
    coord_cols = ["faCoordinateX", "faCoordinateY", "faCoordinateZ"]

    curve_frames: list[pd.DataFrame] = []
    prism_frames: list[pd.DataFrame] = []
    g12_frames: list[pd.DataFrame] = []

    for az_idx in indices:
        if az_idx not in az_mapping:
            print(f"  Active zone index {az_idx} not in mapping, skipping AZ-center Ripley")
            continue
        zone_name = az_mapping[az_idx]
        if zone_name not in az_segmentations:
            print(f"  No AZ segmentation for {zone_name}, skipping AZ-center Ripley")
            continue

        az_df = df_valid[df_valid["cleft"] == az_idx]
        if az_df.empty:
            print(f"  No AuNPs in synaptic cleft index {az_idx} ({zone_name}), skipping")
            continue
        aunp_coords = np.asarray(az_df[coord_cols], dtype=float)

        result = run_aunp_vs_az_center_ripley_for_zone(
            tomogram_path,
            alignment_dir,
            zone_name,
            int(az_idx),
            aunp_coords=aunp_coords,
            az_segmentation=az_segmentations[zone_name],
            r_max_nm=r_max_nm,
            r_step_nm=r_step_nm,
            g12_shell_width_nm=g12_shell_width_nm,
            seed=seed,
            write_figures=write_figures,
        )
        if result is None:
            continue
        curves_path = result["curves_path"]
        prism_path = result["prism_path"]
        g12_path = result["g12_shells_path"]
        if curves_path.is_file():
            curve_frames.append(pd.read_csv(curves_path))
        if prism_path.is_file():
            prism_frames.append(pd.read_csv(prism_path))
        if g12_path.is_file():
            g12_frames.append(pd.read_csv(g12_path))

    return curve_frames, prism_frames, g12_frames


# Display label per L_CURVE_FAMILIES prefix, used in figure titles/axis labels.
L_FAMILY_DISPLAY_LABEL: dict[str, str] = {
    "center_L12": "L₁₂",
    "center_L21": "L₂₁",
    "center_L_combined": "L_combined",
}


def _plot_pooled_l_family_figure(
    grp: pd.DataFrame,
    *,
    prefix: str,
    set_name: str,
    pick_label: str = "",
    output_dir: Path,
) -> Path | None:
    """One pooled mean ± SEM figure for one L-family curve (a row of ``L_CURVE_FAMILIES``).

    Returns ``None`` (writes nothing) if ``grp`` doesn't have this family's columns — e.g.
    pooling older per-zone output written before K₂₁/K_combined existed.
    """
    mean_col = f"{prefix}_mean"
    if mean_col not in grp.columns:
        return None

    r_vals = grp["r_nm"].to_numpy(dtype=float)
    mean = grp[mean_col].to_numpy(dtype=float)
    lo = grp[f"{prefix}_sem_envelope_lo"].to_numpy(dtype=float)
    hi = grp[f"{prefix}_sem_envelope_hi"].to_numpy(dtype=float)
    meta = grp.iloc[0]
    label = L_FAMILY_DISPLAY_LABEL.get(prefix, prefix)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(r_vals, mean, color="C0", lw=2, label=f"Mean {label}")
    ax.fill_between(r_vals, lo, hi, color="C0", alpha=0.25, label="±SEM")
    ax.axhline(0.0, color="0.5", ls="--", lw=0.8)
    ax.set_xlabel("r (nm)")
    ax.set_ylabel(f"Ripley {label}(r)")
    pick_title = f", pick: {pick_label}" if str(pick_label).strip() else ""
    ax.set_title(
        f"Pooled AuNP vs synaptic cleft center ({label}) — set: {set_name}{pick_title}\n"
        f"{int(meta['n_tomograms'])} tomogram(s), {int(meta['n_clefts'])} zone(s), "
        f"{int(meta['n_zone_curves'])} curves"
    )
    ax.set_xlim(0.0, float(r_vals[-1]) if len(r_vals) else AZ_CENTER_RIPLEY_R_MAX_NM)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    set_tag = _safe_name(str(set_name)) or "unspecified"
    pick_tag = _safe_name(str(pick_label)) if str(pick_label).strip() else ""
    stem = prefix.replace("center_", "").lower()
    pick_suffix = f"_{pick_tag}" if pick_tag else ""
    out_path = output_dir / f"ripley_{stem}_pooled_mean_sd_{set_tag}{pick_suffix}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Pooled AuNP vs AZ-center Ripley {label} figure (set {set_name}) -> {out_path}")
    return out_path


def _plot_pooled_g_family_figure(
    grp: pd.DataFrame,
    *,
    stem: str,
    set_name: str,
    pick_label: str = "",
    output_dir: Path,
) -> Path | None:
    """One pooled mean ± SEM figure for one g-family curve (a member of ``G_CURVE_FAMILIES``).

    Returns ``None`` (writes nothing) if ``grp`` doesn't have this family's columns — e.g.
    pooling older per-zone output written before g₂₁/g_combined existed.
    """
    mean_col = f"{stem}_mean"
    if mean_col not in grp.columns:
        return None

    r_vals = grp["r_nm"].to_numpy(dtype=float)
    mean = grp[mean_col].to_numpy(dtype=float)
    lo = grp[f"{stem}_sem_envelope_lo"].to_numpy(dtype=float)
    hi = grp[f"{stem}_sem_envelope_hi"].to_numpy(dtype=float)
    meta = grp.iloc[0]
    label = G_FAMILY_DISPLAY_LABEL.get(stem, stem)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(r_vals, mean, color="C1", lw=2, marker="o", ms=3, label=f"Mean {label}")
    ax.fill_between(r_vals, lo, hi, color="C1", alpha=0.25, label="±SEM")
    ax.axhline(1.0, color="0.5", ls="--", lw=0.8, label="CSR (g=1)")
    ax.set_xlabel("r (nm)")
    ax.set_ylabel(f"{label}(r) = observed / expected AuNP shell density")
    pick_title = f", pick: {pick_label}" if str(pick_label).strip() else ""
    ax.set_title(
        f"Pooled AuNP vs synaptic cleft center {label} — set: {set_name}{pick_title}\n"
        f"{int(meta['n_tomograms'])} tomogram(s), {int(meta['n_clefts'])} zone(s)"
    )
    ax.set_xlim(0.0, float(r_vals[-1]) if len(r_vals) else AZ_CENTER_RIPLEY_R_MAX_NM)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    set_tag = _safe_name(str(set_name)) or "unspecified"
    pick_tag = _safe_name(str(pick_label)) if str(pick_label).strip() else ""
    pick_suffix = f"_{pick_tag}" if pick_tag else ""
    out_path = output_dir / f"{stem}_pooled_mean_sd_{set_tag}{pick_suffix}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Pooled AuNP vs AZ-center {label} figure (set {set_name}) -> {out_path}")
    return out_path


def plot_pooled_aunp_vs_az_center_ripley_visualizations(
    curves_csv: Path | str = POOLED_CURVES_CSV,
    output_dir: Path | str = POOLED_FIGURES_DIR,
    prism_csv: Path | str = POOLED_PRISM_CSV,
    g12_csv: Path | str = POOLED_G12_CSV,
    g12_pooled_csv: Path | str = POOLED_G12_POOLED_CSV,
) -> list[Path]:
    """Build pooled Prism tables and mean ± SD L₁₂ and g₁₂ figures across all zones/tomograms."""
    curves_csv = Path(curves_csv)
    output_dir = Path(output_dir)
    prism_csv = Path(prism_csv)
    g12_csv = Path(g12_csv)
    g12_pooled_csv = Path(g12_pooled_csv)

    written: list[Path] = []
    prism_long = pd.DataFrame()

    if not curves_csv.is_file():
        print(f"No pooled AuNP vs AZ-center Ripley CSV at {curves_csv}; skipping L₁₂ pooled outputs.")
    else:
        df = pd.read_csv(curves_csv)
        if df.empty or "tomogram_name" not in df.columns:
            print("Pooled AuNP vs AZ-center Ripley CSV missing data; skipping L₁₂ pooled outputs.")
        else:
            prism_long = build_pooled_aunp_vs_az_center_prism_table(df)
            if prism_long.empty:
                print("No pooled AuNP vs AZ-center Ripley envelope rows generated.")
            else:
                prism_csv.parent.mkdir(parents=True, exist_ok=True)
                prism_long.to_csv(prism_csv, index=False)
                print(f"Pooled AuNP vs AZ-center Ripley Prism table ({len(prism_long)} rows) -> {prism_csv}")
                written.append(prism_csv)

                l_prism_written = write_pooled_az_center_l_prism_exports(
                    prism_long,
                    out_dir=prism_csv.parent,
                    coarse_bin_width_nm=AZ_CENTER_G12_PRISM_BIN_WIDTH_NM,
                )
                written += l_prism_written
                for path in l_prism_written:
                    print(f"Pooled AuNP vs AZ-center L₁₂ Prism export -> {path}")

            individual_written = _write_pooled_az_center_individual_wide_exports(
                df, out_dir=prism_csv.parent
            )
            written += individual_written
            for path in individual_written:
                print(f"Pooled AuNP vs AZ-center individual wide curves -> {path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    if not prism_long.empty:
        group_cols = ["set_name"]
        if "aunp_pick_label" in prism_long.columns and prism_long["aunp_pick_label"].astype(
            str
        ).str.strip().any():
            group_cols.append("aunp_pick_label")
        for key, grp in prism_long.groupby(group_cols, sort=False):
            set_name, pick_label = _unpack_pool_group_key(key)
            grp = grp.sort_values("r_nm")
            for prefix, _, _ in L_CURVE_FAMILIES:
                out_path = _plot_pooled_l_family_figure(
                    grp,
                    prefix=prefix,
                    set_name=set_name,
                    pick_label=pick_label,
                    output_dir=output_dir,
                )
                if out_path is not None:
                    written.append(out_path)

    if not g12_csv.is_file():
        print(f"No pooled AuNP vs AZ-center g₁₂ CSV at {g12_csv}; skipping g₁₂ pooled outputs.")
        return written

    g12_df = pd.read_csv(g12_csv)
    if g12_df.empty or "tomogram_name" not in g12_df.columns:
        print("Pooled AuNP vs AZ-center g₁₂ CSV missing data; skipping g₁₂ pooled outputs.")
        return written

    g12_pooled = build_pooled_aunp_vs_az_center_g12_table(g12_df)
    if g12_pooled.empty:
        print("No pooled AuNP vs AZ-center g₁₂ rows generated.")
        return written

    g12_pooled_csv.parent.mkdir(parents=True, exist_ok=True)
    g12_pooled.to_csv(g12_pooled_csv, index=False)
    print(f"Pooled AuNP vs AZ-center g₁₂ table (1 nm shells, {len(g12_pooled)} rows) -> {g12_pooled_csv}")
    written.append(g12_pooled_csv)

    g12_rebinned = rebin_g_curves_table(
        g12_df, coarse_bin_width_nm=AZ_CENTER_G12_PRISM_BIN_WIDTH_NM
    )
    if not g12_rebinned.empty:
        g_prism_written = write_pooled_az_center_g_prism_exports(
            g12_rebinned,
            out_dir=g12_pooled_csv.parent,
            coarse_bin_width_nm=AZ_CENTER_G12_PRISM_BIN_WIDTH_NM,
        )
        written += g_prism_written
        for path in g_prism_written:
            print(f"Pooled AuNP vs AZ-center g₁₂ Prism export -> {path}")

    for key, grp in g12_pooled.groupby(_pool_group_columns(g12_pooled), sort=False):
        set_name, pick_label = _unpack_pool_group_key(key)
        grp = grp.sort_values("r_nm")
        for stem in G_CURVE_FAMILIES:
            out_path = _plot_pooled_g_family_figure(
                grp,
                stem=stem,
                set_name=set_name,
                pick_label=pick_label,
                output_dir=output_dir,
            )
            if out_path is not None:
                written.append(out_path)

    return written
