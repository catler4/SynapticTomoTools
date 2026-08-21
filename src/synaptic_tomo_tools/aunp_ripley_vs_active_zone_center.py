"""
3D bivariate Ripley K₁₂ / L₁₂ of AuNP positions relative to the active zone center.

Type-1 foci: one active zone center per zone (mean of presynaptic + postsynaptic AZ points).
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
    _prism_long_to_wide,
    _prism_sd_envelope_columns,
    _ripley_r_grid,
    _safe_name,
    build_ripley_window_3d,
    build_window_grid_points,
    cross_k12_3d_isotropic,
    curves_matrix_to_long_dataframe,
    curves_matrix_to_wide_dataframe,
    g_shell_reliability_mask,
    load_synaptic_cleft_active_zone_points,
    pair_correlation_from_k_diff,
    plot_ripley_window_geometry_diagnostic,
    prism_sd_envelope_columns_from_averaged_k12,
    ripley_l12,
)

WINDOW_MODE = "synaptic_cleft_az_hull"
MIN_AUNP_PARTNERS = 3
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

# (pooled-table column prefix, raw L-column name, raw K-column name) for each of the three
# K/L families: the direct K12/L12 (center-as-focus), the reversed K21/L21 (AuNPs-as-foci),
# and their intensity-weighted combination (see cross_k_bivariate_symmetric_3d_isotropic).
L_CURVE_FAMILIES: tuple[tuple[str, str, str], ...] = (
    ("center_L12", "l12", "k12"),
    ("center_L21", "l21", "k21"),
    ("center_L_combined", "l_combined", "k_combined"),
)

POOLED_CURVES_CSV = Path("results/aunps/aunp_vs_az_center_ripley_l12_curves.csv")
POOLED_PRISM_CSV = Path("results/aunps/aunp_vs_az_center_ripley_l12_prism_pooled.csv")
POOLED_PRISM_WIDE_CSV = Path("results/aunps/aunp_vs_az_center_ripley_l12_prism_pooled_wide.csv")
POOLED_FIGURES_DIR = Path("results/aunps/figures/aunp_vs_az_center_ripley_l12_pooled")
POOLED_G12_CSV = Path("results/aunps/aunp_vs_az_center_ripley_g12_shells_curves.csv")
POOLED_G12_POOLED_CSV = Path("results/aunps/aunp_vs_az_center_ripley_g12_pooled.csv")


_AZ_CENTER_MAX_ITER = 5
_AZ_CENTER_CONVERGENCE_TOL_NM = 1e-3


def compute_active_zone_center_nm(az_segmentation: dict) -> np.ndarray:
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
            "  Warning: active-zone center failed the in-betweenness test after "
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
    id_cols = ["tomogram_name", "alignment_dir", "active_zone_name"]
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
            "active_zone_name": zone_name,
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
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Pooled K-scale and L-scale envelope columns for one L-family curve (a row of
    ``L_CURVE_FAMILIES``). Returns ``(r_vals, n_valid_per_r, from_k, from_l)``; ``from_k``/
    ``from_l`` are empty dicts if ``l_col`` isn't present in ``sub`` (e.g. older per-zone
    output written before that family existed)."""
    r_vals, l_curves = _extract_zone_curves_matrix(sub, value_col=l_col)
    if len(l_curves) == 0:
        return r_vals, np.array([]), {}, {}
    k_curves = None
    if k_col in sub.columns:
        _, k_curves_candidate = _extract_zone_curves_matrix(sub, value_col=k_col)
        if len(k_curves_candidate) == len(l_curves):
            k_curves = k_curves_candidate
    from_l = _prism_sd_envelope_columns(l_curves, r_vals, prefix=prefix)
    from_k = prism_sd_envelope_columns_from_averaged_k12(
        l_curves, r_vals, prefix=prefix, k12_curves=k_curves
    )
    n_valid_per_r = np.sum(~np.isnan(l_curves), axis=0)
    return r_vals, n_valid_per_r, from_k, from_l


def build_pooled_aunp_vs_az_center_prism_table(df: pd.DataFrame) -> pd.DataFrame:
    """Pooled mean ± SD/SEM of H₁₂ (Ripley L-transform) across tomogram-zone curves, per set,
    for each of the three K/L families (``L_CURVE_FAMILIES``): direct K₁₂/L₁₂
    (``center_L12_*``, center-as-focus — the original statistic), reversed K₂₁/L₂₁
    (``center_L21_*``, AuNPs-as-foci), and their intensity-weighted combination
    (``center_L_combined_*``, see ``cross_k_bivariate_symmetric_3d_isotropic``).

    For each family, averaging is done on the K scale first, then the mean (and mean ± SD/
    SEM) is mapped through the H = (3K/4π)^(1/3) - r transform (``*_mean``/``*_sd``/``*_sem``;
    uses the family's stored K column when present) — correct because H is a nonlinear
    function of K, so pooling must happen on the additive K scale, not after the transform.
    The mean ± SD/SEM computed directly on the per-curve H values is also reported
    (``*_mean_from_l`` etc.) for reference. A family missing from ``df`` (e.g. pooling older
    per-zone output written before it existed) is simply omitted from the row rather than
    filled with NaN placeholders.
    """
    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    if "set_name" not in df.columns:
        df["set_name"] = ""
    df["set_name"] = df["set_name"].fillna("").astype(str)

    rows: list[dict] = []
    for set_name, sub in df.groupby("set_name", sort=False):
        anchor_r_vals: np.ndarray | None = None
        anchor_n_valid: np.ndarray | None = None
        family_envelopes: dict[str, tuple[dict[str, np.ndarray], dict[str, np.ndarray]]] = {}

        for prefix, l_col, k_col in L_CURVE_FAMILIES:
            r_vals, n_valid_per_r, from_k, from_l = _pooled_l_family_columns(
                sub, prefix=prefix, l_col=l_col, k_col=k_col
            )
            if not from_k:
                continue
            family_envelopes[prefix] = (from_k, from_l)
            if anchor_r_vals is None:
                anchor_r_vals = r_vals
                anchor_n_valid = n_valid_per_r

        if anchor_r_vals is None:
            continue

        n_tomograms = int(sub["tomogram_name"].nunique()) if "tomogram_name" in sub.columns else 0
        n_zones = int(
            sub[["tomogram_name", "alignment_dir", "active_zone_name"]].drop_duplicates().shape[0]
        )

        for i, r_nm in enumerate(anchor_r_vals):
            row = {
                "set_name": set_name,
                "window_mode": WINDOW_MODE,
                "r_nm": float(r_nm),
                "n_zone_curves": int(anchor_n_valid[i]),
                "n_tomograms": n_tomograms,
                "n_active_zones": n_zones,
            }
            for prefix, (from_k, from_l) in family_envelopes.items():
                row.update(
                    {
                        f"{prefix}_mean": float(from_k[f"{prefix}_mean_from_k"][i]),
                        f"{prefix}_sd": float(from_k[f"{prefix}_sd_from_k"][i]),
                        f"{prefix}_sd_envelope_lo": float(from_k[f"{prefix}_sd_envelope_lo_from_k"][i]),
                        f"{prefix}_sd_envelope_hi": float(from_k[f"{prefix}_sd_envelope_hi_from_k"][i]),
                        f"{prefix}_sem": float(from_k[f"{prefix}_sem_from_k"][i]),
                        f"{prefix}_sem_envelope_lo": float(from_k[f"{prefix}_sem_envelope_lo_from_k"][i]),
                        f"{prefix}_sem_envelope_hi": float(from_k[f"{prefix}_sem_envelope_hi_from_k"][i]),
                        f"{prefix}_mean_from_l": float(from_l[f"{prefix}_mean"][i]),
                        f"{prefix}_sd_from_l": float(from_l[f"{prefix}_sd"][i]),
                        f"{prefix}_sd_envelope_lo_from_l": float(from_l[f"{prefix}_sd_envelope_lo"][i]),
                        f"{prefix}_sd_envelope_hi_from_l": float(from_l[f"{prefix}_sd_envelope_hi"][i]),
                        f"{prefix}_sem_from_l": float(from_l[f"{prefix}_sem"][i]),
                        f"{prefix}_sem_envelope_lo_from_l": float(from_l[f"{prefix}_sem_envelope_lo"][i]),
                        f"{prefix}_sem_envelope_hi_from_l": float(from_l[f"{prefix}_sem_envelope_hi"][i]),
                    }
                )
            rows.append(row)
    return pd.DataFrame(rows)


G_CURVE_FAMILIES: tuple[str, ...] = ("g12", "g21", "g_combined")


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

    df = df.copy()
    if "set_name" not in df.columns:
        df["set_name"] = ""
    df["set_name"] = df["set_name"].fillna("").astype(str)

    rows: list[dict] = []
    for set_name, sub in df.groupby("set_name", sort=False):
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
            sub[["tomogram_name", "alignment_dir", "active_zone_name"]].drop_duplicates().shape[0]
        )

        for i, r_nm in enumerate(anchor_r_vals):
            row = {
                "set_name": set_name,
                "window_mode": WINDOW_MODE,
                "r_nm": float(r_nm),
                "r_lo_nm": float(r_lo[i]),
                "r_hi_nm": float(r_hi[i]),
                "n_zone_shells": int(anchor_n_valid[i]),
                "n_tomograms": n_tomograms,
                "n_active_zones": n_zones,
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


def plot_active_zone_center_diagnostic(
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
    center = compute_active_zone_center_nm(az_segmentation)
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
    active_zone_index: int,
    *,
    aunp_coords: np.ndarray,
    az_segmentation: dict,
    r_max_nm: float = AZ_CENTER_RIPLEY_R_MAX_NM,
    r_step_nm: float = DEFAULT_RIPLEY_R_STEP_NM,
    g12_shell_width_nm: float = AZ_CENTER_G12_SHELL_WIDTH_NM,
    seed: int = DEFAULT_ANALYSIS_SEED,
    write_figures: bool = True,
) -> dict[str, Path] | None:
    """Compute observed 3D L₁₂(center, AuNPs) for one active zone.

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

    center = compute_active_zone_center_nm(az_segmentation)
    if not np.all(np.isfinite(center)):
        print(f"  Skipping AZ-center Ripley for {zone_name}: could not compute center")
        return None

    rng = np.random.default_rng(seed)
    try:
        cleft_coords = load_synaptic_cleft_active_zone_points(
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
            plot_active_zone_center_diagnostic(
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
            "active_zone_name": zone_name,
            "active_zone_index": int(active_zone_index),
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
            "active_zone_name": zone_name,
            "active_zone_index": int(active_zone_index),
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
            "active_zone_name": zone_name,
            "active_zone_index": int(active_zone_index),
            "window_mode": WINDOW_MODE,
            "n_aunp_partners": int(len(aunp_coords)),
            "n_aunps_dropped_outside_hull": int(len(dropped_coords)),
            "window_volume_nm3": float(window.volume_nm3),
        },
    )
    individual_path = out_dir / "ripley_l12_individual_curves.csv"
    individual_df.to_csv(individual_path, index=False)
    curves_matrix_to_wide_dataframe(
        np.atleast_2d(l12), r_vals, curve_type="observed"
    ).to_csv(out_dir / "ripley_l12_individual_observed_wide.csv", index=False)

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
    _prism_long_to_wide(prism_df, id_cols=["active_zone_name", "window_mode"]).to_csv(
        out_dir / "ripley_l12_prism_wide.csv", index=False
    )

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
        "active_zone_name": zone_name,
        "active_zone_index": int(active_zone_index),
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
    active_zone_indices: Sequence[int] | None,
    df_valid: pd.DataFrame,
    az_segmentations: dict,
    r_max_nm: float = AZ_CENTER_RIPLEY_R_MAX_NM,
    r_step_nm: float = DEFAULT_RIPLEY_R_STEP_NM,
    g12_shell_width_nm: float = AZ_CENTER_G12_SHELL_WIDTH_NM,
    seed: int = DEFAULT_ANALYSIS_SEED,
    write_figures: bool = True,
) -> tuple[list[pd.DataFrame], list[pd.DataFrame], list[pd.DataFrame]]:
    """Run AZ-center Ripley for all mapped active zones in one tomogram."""
    from .activezone import load_active_zone_mapping

    tomogram_path = Path(tomogram_path)
    alignment_dir = require_alignment_dir(alignment_dir)
    az_mapping = load_active_zone_mapping(tomogram_path, alignment_dir) or {}
    if not az_mapping:
        print("No active zone mapping; skipping AuNP vs AZ-center Ripley analyses")
        return [], [], []

    az_mapping = {int(k): v for k, v in az_mapping.items()}
    indices = list(active_zone_indices) if active_zone_indices is not None else sorted(az_mapping)
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

        az_df = df_valid[df_valid["active_zone"] == az_idx]
        if az_df.empty:
            print(f"  No AuNPs in active zone index {az_idx} ({zone_name}), skipping")
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
    ax.plot(r_vals, mean, color="C0", lw=2, label=f"Mean {label} (K→L)")
    from_l_col = f"{prefix}_mean_from_l"
    if from_l_col in grp.columns:
        ax.plot(
            r_vals,
            grp[from_l_col].to_numpy(dtype=float),
            color="C0",
            lw=1.5,
            ls="--",
            label=f"Mean {label} (of L)",
        )
    ax.fill_between(r_vals, lo, hi, color="C0", alpha=0.25, label="±SEM (K→L)")
    from_l_lo_col = f"{prefix}_sem_envelope_lo_from_l"
    from_l_hi_col = f"{prefix}_sem_envelope_hi_from_l"
    if from_l_lo_col in grp.columns:
        ax.fill_between(
            r_vals,
            grp[from_l_lo_col].to_numpy(dtype=float),
            grp[from_l_hi_col].to_numpy(dtype=float),
            color="C0",
            alpha=0.12,
            hatch="///",
            label="±SEM (of L)",
        )
    ax.axhline(0.0, color="0.5", ls="--", lw=0.8)
    ax.set_xlabel("r (nm)")
    ax.set_ylabel(f"Ripley {label}(r)")
    ax.set_title(
        f"Pooled AuNP vs active zone center ({label}) — set: {set_name}\n"
        f"{int(meta['n_tomograms'])} tomogram(s), {int(meta['n_active_zones'])} zone(s), "
        f"{int(meta['n_zone_curves'])} curves"
    )
    ax.set_xlim(0.0, float(r_vals[-1]) if len(r_vals) else AZ_CENTER_RIPLEY_R_MAX_NM)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    set_tag = _safe_name(str(set_name)) or "unspecified"
    stem = prefix.replace("center_", "").lower()
    out_path = output_dir / f"ripley_{stem}_pooled_mean_sd_{set_tag}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Pooled AuNP vs AZ-center Ripley {label} figure (set {set_name}) -> {out_path}")
    return out_path


def _plot_pooled_g_family_figure(
    grp: pd.DataFrame,
    *,
    stem: str,
    set_name: str,
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
    ax.set_title(
        f"Pooled AuNP vs active zone center {label} — set: {set_name}\n"
        f"{int(meta['n_tomograms'])} tomogram(s), {int(meta['n_active_zones'])} zone(s)"
    )
    ax.set_xlim(0.0, float(r_vals[-1]) if len(r_vals) else AZ_CENTER_RIPLEY_R_MAX_NM)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    set_tag = _safe_name(str(set_name)) or "unspecified"
    out_path = output_dir / f"{stem}_pooled_mean_sd_{set_tag}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Pooled AuNP vs AZ-center {label} figure (set {set_name}) -> {out_path}")
    return out_path


def plot_pooled_aunp_vs_az_center_ripley_visualizations(
    curves_csv: Path | str = POOLED_CURVES_CSV,
    output_dir: Path | str = POOLED_FIGURES_DIR,
    prism_csv: Path | str = POOLED_PRISM_CSV,
    prism_wide_csv: Path | str = POOLED_PRISM_WIDE_CSV,
    g12_csv: Path | str = POOLED_G12_CSV,
    g12_pooled_csv: Path | str = POOLED_G12_POOLED_CSV,
) -> list[Path]:
    """Build pooled Prism tables and mean ± SD L₁₂ and g₁₂ figures across all zones/tomograms."""
    curves_csv = Path(curves_csv)
    output_dir = Path(output_dir)
    prism_csv = Path(prism_csv)
    prism_wide_csv = Path(prism_wide_csv)
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
                _prism_long_to_wide(prism_long, id_cols=["set_name", "window_mode"]).to_csv(
                    prism_wide_csv, index=False
                )
                print(f"Pooled AuNP vs AZ-center Ripley Prism table ({len(prism_long)} rows) -> {prism_csv}")
                written += [prism_csv, prism_wide_csv]

    output_dir.mkdir(parents=True, exist_ok=True)

    for set_name, grp in (prism_long.groupby("set_name", sort=False) if not prism_long.empty else []):
        grp = grp.sort_values("r_nm")
        for prefix, _, _ in L_CURVE_FAMILIES:
            out_path = _plot_pooled_l_family_figure(
                grp, prefix=prefix, set_name=set_name, output_dir=output_dir
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
    print(f"Pooled AuNP vs AZ-center g₁₂ table ({len(g12_pooled)} rows) -> {g12_pooled_csv}")
    written.append(g12_pooled_csv)

    for set_name, grp in g12_pooled.groupby("set_name", sort=False):
        grp = grp.sort_values("r_nm")
        for stem in G_CURVE_FAMILIES:
            out_path = _plot_pooled_g_family_figure(
                grp, stem=stem, set_name=set_name, output_dir=output_dir
            )
            if out_path is not None:
                written.append(out_path)

    return written
