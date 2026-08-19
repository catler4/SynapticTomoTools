"""
3D bivariate Ripley K₁₂ / L₁₂ of monomer vs dimer AuNP positions (no fusion site).

Type-1 foci: monomer AuNP pick coordinates for a zone.
Type-2 partners: dimer AuNP pick coordinates for the same zone.

Reports three K/L families (see ``L_FAMILIES``): the direct K₁₂/L₁₂ (monomer-as-focus, the
original statistic), the reversed K₂₁/L₂₁ (dimer-as-focus), and their intensity-weighted
combination K_combined/L_combined (Lotwick & Silverman 1982). Also reports g₁₂/g₂₁/
g_combined (``G_FAMILIES``), each computed as a finite difference of the corresponding K
curve (``pair_correlation_from_k_diff``) rather than an independent shell-count estimator.
AuNPs outside the Ripley window (hull ∩ betweenness-region) are always dropped before any
of this is computed.

Window: synaptic_cleft_az_hull (convex hull of presynaptic + postsynaptic AZ surface points),
always additionally restricted to the angle in-betweenness region (matching the AZ-center
Ripley analysis) since monomer/dimer AuNP positions are only meaningful relative to the
space between the two membranes. Edge correction uses the deterministic grid quadrature
method (``_isotropic_edge_factors_grid``), matching the AZ-center Ripley analysis, rather
than Monte Carlo sampling.

Control: label permutation — pool all monomer + dimer points, then randomly reassign class
labels while preserving the per-zone monomer and dimer counts (1000 replicates by default).
Each replicate computes K in both directions (``label_permutation_k_bidirectional_curves``),
so every one of the six reported families gets its own null distribution.

Greedy segregation — same pooled points and class counts, but relabel by growing a compact
spatial cluster from a random seed. Each replicate randomly chooses whether that compact
class is monomers or dimers (then the remainder gets the other label). Replicate count
always matches the label-permutation count (``n_perm``, default 1000). Also bidirectional.

MAD tests (Rebola-style max absolute deviation vs 99% CE) are run for each of the six
families against both label-permutation and greedy segregation when that null has ≥1000
curves; otherwise they are skipped. Each MAD is reported for the full r-grid and for the
restricted 30–50 nm window.

Per zone, the first three label-permutation and greedy-segregation point sets are also
written as monomer/dimer STAR files under ``STT_results/.../simulated_null_stars/``
(same columns as the input pick STARs).

Pooled output is grouped per tomogram set (curves, individual curves, and MAD summaries).
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace as dataclasses_replace
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from scipy.spatial.distance import cdist

from .alignment_utils import require_alignment_dir
from .aunps import _read_aunp_pick_star_dataframe
from .ripley_library import (
    COORD_COLS,
    DEFAULT_ANALYSIS_SEED,
    DEFAULT_RIPLEY_R_MAX_NM,
    DEFAULT_RIPLEY_R_STEP_NM,
    MAD_MIN_NULL_CURVES,
    MAD_R_RANGES,
    RIPLEY_PERCENTILE_HI,
    RIPLEY_PERCENTILE_LO,
    _default_ripley_perm_workers,
    _find_monomer_dimer_star_path,
    _isotropic_edge_factors_grid,
    _percentile_band,
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
    derive_symmetric_k_l_g_families,
    g_shell_reliability_mask,
    label_permutation_k_bidirectional_curves,
    load_monomer_dimer_aunps_for_zone,
    load_synaptic_cleft_active_zone_points,
    mad_result_to_curves_dataframe,
    mad_result_to_summary_row,
    plot_ripley_window_geometry_diagnostic,
    prism_sd_envelope_columns_from_averaged_k12,
    run_mad_tests_over_r_ranges,
    subset_aunps,
    RipleyWindow3D,
)

WINDOW_MODE = "synaptic_cleft_az_hull"
MONOMER_DIMER_N_PERM = 1000
# Segregation always uses the same replicate count as label permutation (n_perm).
MIN_POINTS_PER_CLASS = 2
# Matches the AZ-center Ripley analysis's tuned grid spacing/accuracy tradeoff.
MONOMER_DIMER_EDGE_GRID_SPACING_NM = 2.0
# Example null point sets written as STAR files (label-perm + greedy each).
N_SIMULATED_NULL_STAR_EXAMPLES = 3
SIMULATED_STAR_COLS = (
    "faCoordinateX",
    "faCoordinateY",
    "faCoordinateZ",
    "active_zone",
    "postsynapse",
)

# The six reported statistics per zone: direct (monomer-as-focus), reversed
# (dimer-as-focus), and their intensity-weighted combination -- for both K's L-transform
# and the K-difference pair-correlation function (see derive_symmetric_k_l_g_families).
L_FAMILIES: tuple[str, ...] = ("l12", "l21", "l_combined")
G_FAMILIES: tuple[str, ...] = ("g12", "g21", "g_combined")
ALL_FAMILIES: tuple[str, ...] = L_FAMILIES + G_FAMILIES


def _family_tag(fam: str) -> str:
    """Column-prefix tag for a family key, e.g. ``'l12' -> 'L12'``, ``'g_combined' ->
    'G_combined'``."""
    return ("L" if fam.startswith("l") else "G") + fam[1:]


POOLED_CURVES_CSV = Path("results/aunps/aunp_monomer_dimer_ripley_l12_curves.csv")
POOLED_INDIVIDUAL_CURVES_CSV = Path(
    "results/aunps/aunp_monomer_dimer_ripley_l12_individual_curves.csv"
)
POOLED_MAD_SUMMARY_CSV = Path("results/aunps/aunp_monomer_dimer_ripley_l12_mad_summary.csv")
POOLED_PRISM_CSV = Path("results/aunps/aunp_monomer_dimer_ripley_l12_prism_pooled.csv")
POOLED_PRISM_WIDE_CSV = Path("results/aunps/aunp_monomer_dimer_ripley_l12_prism_pooled_wide.csv")
POOLED_FIGURES_DIR = Path("results/aunps/figures/aunp_monomer_dimer_ripley_l12_pooled")


def _extract_curves_matrix(df: pd.DataFrame, value_col: str) -> tuple[np.ndarray, np.ndarray]:
    """Pivot long table to (r_vals, curves) with one curve per tomogram+zone."""
    if df.empty or value_col not in df.columns:
        return np.array([]), np.empty((0, 0))

    sub = df.copy()
    r_vals = np.sort(sub["r_nm"].unique())
    n_r = len(r_vals)
    id_cols = ["tomogram_name", "alignment_dir", "active_zone_name"]
    for col in id_cols:
        if col not in sub.columns:
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


def _greedy_segregation_cluster_mask(
    pool: np.ndarray,
    n_cluster: int,
    seed_idx: int,
    *,
    pairwise_dist: np.ndarray | None = None,
) -> np.ndarray:
    """Greedy nearest-neighbor growth of a compact cluster from a seed point index."""
    pool = np.atleast_2d(np.asarray(pool, dtype=float))
    n_pool = len(pool)
    n_cluster = int(n_cluster)
    mask = np.zeros(n_pool, dtype=bool)
    if n_pool == 0 or n_cluster <= 0 or n_cluster > n_pool:
        return mask

    seed_idx = int(seed_idx) % n_pool
    dist = pairwise_dist if pairwise_dist is not None else cdist(pool, pool)
    mask[seed_idx] = True
    for _ in range(n_cluster - 1):
        # Distance of each point to the nearest already-clustered point.
        d_to_cluster = dist[:, mask].min(axis=1)
        d_to_cluster = d_to_cluster.copy()
        d_to_cluster[mask] = np.inf
        # argmin breaks ties by lowest index (matches prior set-based growth).
        best_idx = int(np.argmin(d_to_cluster))
        if not np.isfinite(d_to_cluster[best_idx]):
            break
        mask[best_idx] = True
    return mask


# Shared context for parallel greedy-segregation bidirectional-K workers.
_SEG_BIDIR_CTX: dict = {}


def _init_greedy_seg_bidir_worker(
    pool: np.ndarray,
    r_vals: np.ndarray,
    window,
    n_monomer: int,
    n_dimer: int,
    pairwise_dist: np.ndarray,
    pool_edge_factors: np.ndarray,
) -> None:
    _SEG_BIDIR_CTX["pool"] = pool
    _SEG_BIDIR_CTX["r_vals"] = r_vals
    _SEG_BIDIR_CTX["window"] = window
    _SEG_BIDIR_CTX["n_monomer"] = int(n_monomer)
    _SEG_BIDIR_CTX["n_dimer"] = int(n_dimer)
    _SEG_BIDIR_CTX["pairwise_dist"] = pairwise_dist
    _SEG_BIDIR_CTX["pool_edge_factors"] = pool_edge_factors


def _label_permutation_monomer_mask(
    n_pool: int,
    n_monomer: int,
    seed: int,
) -> np.ndarray:
    """Monomer mask for one label-permutation replicate (deterministic seed)."""
    rng = np.random.default_rng(int(seed))
    class1_idx = rng.choice(int(n_pool), int(n_monomer), replace=False)
    mask = np.zeros(int(n_pool), dtype=bool)
    mask[class1_idx] = True
    return mask


def _greedy_segregation_monomer_mask(
    pool: np.ndarray,
    n_monomer: int,
    n_dimer: int,
    pairwise_dist: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Monomer mask for one greedy-segregation replicate.

    Randomly chooses monomer or dimer as the compact class, grows that class from a
    random seed, and assigns the remainder to the other class.
    """
    pool = np.atleast_2d(np.asarray(pool, dtype=float))
    n_pool = len(pool)
    cluster_class = "dimer" if rng.random() < 0.5 else "monomer"
    n_cluster = int(n_dimer if cluster_class == "dimer" else n_monomer)
    seed_idx = int(rng.integers(0, n_pool))
    cluster_mask = _greedy_segregation_cluster_mask(
        pool, n_cluster, seed_idx, pairwise_dist=pairwise_dist
    )
    return ~cluster_mask if cluster_class == "dimer" else cluster_mask


def _greedy_seg_k_bidir_one_replicate(
    *,
    pool: np.ndarray,
    r_vals: np.ndarray,
    window,
    n_monomer: int,
    n_dimer: int,
    pairwise_dist: np.ndarray,
    pool_edge_factors: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    One greedy-segregation (K12, K21) curve pair: randomly choose monomer or dimer as the
    compact class, grow that class from a random seed, assign the remainder to the other
    class, then compute K in both directions.
    """
    monomer_mask = _greedy_segregation_monomer_mask(
        pool, n_monomer, n_dimer, pairwise_dist, rng
    )
    k12 = cross_k12_3d_isotropic(
        pool[monomer_mask],
        pool[~monomer_mask],
        r_vals,
        window,
        rng,
        edge_factors=pool_edge_factors[monomer_mask],
    )
    k21 = cross_k12_3d_isotropic(
        pool[~monomer_mask],
        pool[monomer_mask],
        r_vals,
        window,
        rng,
        edge_factors=pool_edge_factors[~monomer_mask],
    )
    return k12, k21


def _load_pool_star_dataframe_for_export(
    tomogram_path: Path,
    alignment_dir: str,
    active_zone_index: int,
    *,
    monomer_star_pattern: Optional[str] = None,
    dimer_star_pattern: Optional[str] = None,
) -> pd.DataFrame:
    """Load monomer then dimer STAR rows (same order as the Ripley pool)."""
    tomogram_path = Path(tomogram_path)
    alignment_dir = require_alignment_dir(alignment_dir)
    aunps_dir = tomogram_path / alignment_dir / "aunps"
    tomogram_name = tomogram_path.name
    frames: list[pd.DataFrame] = []
    for kind in ("monomer", "dimer"):
        path = _find_monomer_dimer_star_path(
            aunps_dir,
            tomogram_name,
            alignment_dir,
            int(active_zone_index),
            kind=kind,  # type: ignore[arg-type]
            pattern=monomer_star_pattern if kind == "monomer" else dimer_star_pattern,
        )
        if path is None:
            raise FileNotFoundError(f"Missing {kind} STAR for zone {active_zone_index}")
        df = _read_aunp_pick_star_dataframe(path)
        if df is None or df.empty:
            raise ValueError(f"Empty or unreadable {kind} STAR: {path}")
        missing = [c for c in COORD_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"{path} missing columns: {missing}")
        part = df.copy()
        if "active_zone" not in part.columns:
            part["active_zone"] = int(active_zone_index)
        if "postsynapse" not in part.columns:
            part["postsynapse"] = False
        frames.append(part)
    return pd.concat(frames, ignore_index=True)


def _star_export_dataframe(pool_df: pd.DataFrame, mask: np.ndarray) -> pd.DataFrame:
    """Subset pool rows to the example STAR column set."""
    out = pool_df.loc[np.asarray(mask, dtype=bool), list(SIMULATED_STAR_COLS)].copy()
    return out.reset_index(drop=True)


def _write_simulated_null_star_examples(
    *,
    out_dir: Path,
    pool: np.ndarray,
    pool_df: pd.DataFrame,
    n_monomer: int,
    n_dimer: int,
    analysis_seed: int,
    n_examples: int = N_SIMULATED_NULL_STAR_EXAMPLES,
) -> Path:
    """
    Write ``n_examples`` label-permutation and greedy-segregation point sets as STAR files.

    Uses the same deterministic per-replicate seeds as the parallel null workers so the
    first replicates match those null curves when process parallelism is used.
    """
    import starfile

    stars_dir = out_dir / "simulated_null_stars"
    stars_dir.mkdir(parents=True, exist_ok=True)
    pool = np.atleast_2d(np.asarray(pool, dtype=float))
    n_pool = len(pool)
    n_examples = int(min(max(0, n_examples), n_pool))
    if n_examples == 0 or n_monomer <= 0 or n_dimer <= 0 or n_monomer + n_dimer != n_pool:
        return stars_dir

    pairwise_dist = cdist(pool, pool)
    # Match label_permutation_l12_curves parallel seeds: seed + 1 + perm_id
    # Match _greedy_segregation_l12_curves parallel seeds: (seed+2e6) + 17 + rep_id
    greedy_base_seed = int(analysis_seed) + 2_000_000

    for rep in range(n_examples):
        rep_tag = f"{rep + 1:02d}"
        perm_mask = _label_permutation_monomer_mask(
            n_pool, n_monomer, int(analysis_seed) + 1 + rep
        )
        starfile.write(
            _star_export_dataframe(pool_df, perm_mask),
            stars_dir / f"label_permutation_rep{rep_tag}_monomer.star",
            overwrite=True,
        )
        starfile.write(
            _star_export_dataframe(pool_df, ~perm_mask),
            stars_dir / f"label_permutation_rep{rep_tag}_dimer.star",
            overwrite=True,
        )

        greedy_mask = _greedy_segregation_monomer_mask(
            pool,
            n_monomer,
            n_dimer,
            pairwise_dist,
            np.random.default_rng(greedy_base_seed + 17 + rep),
        )
        starfile.write(
            _star_export_dataframe(pool_df, greedy_mask),
            stars_dir / f"segregation_greedy_rep{rep_tag}_monomer.star",
            overwrite=True,
        )
        starfile.write(
            _star_export_dataframe(pool_df, ~greedy_mask),
            stars_dir / f"segregation_greedy_rep{rep_tag}_dimer.star",
            overwrite=True,
        )

    return stars_dir


def _greedy_seg_bidir_worker(task: tuple[int, int]) -> tuple[int, np.ndarray, np.ndarray]:
    """Run one greedy-segregation (K12, K21) curve pair (rep_id, seed)."""
    rep_id, seed = task
    k12, k21 = _greedy_seg_k_bidir_one_replicate(
        pool=_SEG_BIDIR_CTX["pool"],
        r_vals=_SEG_BIDIR_CTX["r_vals"],
        window=_SEG_BIDIR_CTX["window"],
        n_monomer=_SEG_BIDIR_CTX["n_monomer"],
        n_dimer=_SEG_BIDIR_CTX["n_dimer"],
        pairwise_dist=_SEG_BIDIR_CTX["pairwise_dist"],
        pool_edge_factors=_SEG_BIDIR_CTX["pool_edge_factors"],
        rng=np.random.default_rng(int(seed)),
    )
    return int(rep_id), k12, k21


def _greedy_segregation_k_bidirectional_curves(
    pool: np.ndarray,
    n_monomer: int,
    n_dimer: int,
    r_vals: np.ndarray,
    window,
    rng: np.random.Generator,
    *,
    n_rep: int,
    pool_edge_factors: np.ndarray,
    seed: int = DEFAULT_ANALYSIS_SEED,
    n_workers: int | None = None,
    pbar: Optional[tqdm] = None,
) -> dict[str, np.ndarray]:
    """
    Greedy segregation (K12, K21) curve pairs with a random clustered class per replicate.

    Each replicate independently chooses monomer or dimer as the compact class
    (equal probability), grows that class from a random seed via nearest-neighbor
    expansion to the observed class count, assigns the remaining points to the other
    class, then computes K in both directions (see ``derive_symmetric_k_l_g_families``
    for deriving L12/L21/L_combined/g12/g21/g_combined from the result).

    Uses the same edge-factor precompute + process-pool pattern as label permutation.
    ``pool_edge_factors`` (e.g. from the deterministic grid method) must already cover
    every pooled point as a potential focus.
    """
    pool = np.atleast_2d(np.asarray(pool, dtype=float))
    n_pool = len(pool)
    n_rep_int = int(n_rep)
    n_monomer = int(n_monomer)
    n_dimer = int(n_dimer)
    empty = {
        "k12": np.full((n_rep_int, len(r_vals)), np.nan, dtype=float),
        "k21": np.full((n_rep_int, len(r_vals)), np.nan, dtype=float),
    }
    if n_pool == 0 or n_rep_int == 0:
        return empty
    if n_monomer <= 0 or n_dimer <= 0 or n_monomer + n_dimer != n_pool:
        return empty
    if n_monomer > n_pool or n_dimer > n_pool:
        return empty

    pairwise_dist = cdist(pool, pool)
    pool_edge_factors = np.asarray(pool_edge_factors, dtype=float)
    if pool_edge_factors.shape != (n_pool, len(r_vals)):
        raise ValueError(
            f"pool_edge_factors shape {pool_edge_factors.shape} != "
            f"expected {(n_pool, len(r_vals))}"
        )

    if n_workers is None:
        n_workers = _default_ripley_perm_workers(n_rep_int)
    n_workers = max(1, min(int(n_workers), n_rep_int))

    k12_curves = np.full((n_rep_int, len(r_vals)), np.nan, dtype=float)
    k21_curves = np.full((n_rep_int, len(r_vals)), np.nan, dtype=float)

    if n_workers == 1:
        for rep_id in range(n_rep_int):
            k12, k21 = _greedy_seg_k_bidir_one_replicate(
                pool=pool,
                r_vals=r_vals,
                window=window,
                n_monomer=n_monomer,
                n_dimer=n_dimer,
                pairwise_dist=pairwise_dist,
                pool_edge_factors=pool_edge_factors,
                rng=rng,
            )
            k12_curves[rep_id] = k12
            k21_curves[rep_id] = k21
            if pbar is not None:
                pbar.set_postfix_str(f"greedy seg {rep_id + 1}/{n_rep_int}", refresh=False)
                pbar.update(1)
        return {"k12": k12_curves, "k21": k21_curves}

    # Deterministic per-replicate seeds (independent of call-order / worker scheduling).
    tasks = [(rep_id, int(seed) + 17 + rep_id) for rep_id in range(n_rep_int)]
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_greedy_seg_bidir_worker,
        initargs=(
            pool,
            r_vals,
            window,
            n_monomer,
            n_dimer,
            pairwise_dist,
            pool_edge_factors,
        ),
    ) as executor:
        futures = [executor.submit(_greedy_seg_bidir_worker, task) for task in tasks]
        for fut in as_completed(futures):
            rep_id, k12, k21 = fut.result()
            k12_curves[rep_id] = k12
            k21_curves[rep_id] = k21
            if pbar is not None:
                pbar.set_postfix_str(f"greedy seg {rep_id + 1}/{n_rep_int}", refresh=False)
                pbar.update(1)
    return {"k12": k12_curves, "k21": k21_curves}


def _plot_mad_panels(
    *,
    mad_results: list[dict],
    output_path: Path,
    title: str,
    observed_color: str = "C0",
    observed_label: str = "Observed L₁₂",
    value_label: str = "L₁₂(r)",
) -> None:
    """Plot raw + normalized MAD panels for each null (uses each result's own r-window)."""
    n_panels = max(1, len(mad_results))
    fig, axes = plt.subplots(2, n_panels, figsize=(4.2 * n_panels, 7.0), squeeze=False)
    for col, mad in enumerate(mad_results):
        ax_raw = axes[0, col]
        ax_norm = axes[1, col]
        null_label = str(mad["null_name"])
        r_use = np.asarray(mad.get("r_vals", []), dtype=float)
        obs_use = np.asarray(mad.get("observed", []), dtype=float).reshape(-1)
        if len(r_use) and len(obs_use) == len(r_use):
            ax_raw.plot(r_use, obs_use, color=observed_color, lw=2.0, label=observed_label)
        if mad["status"] == "ok":
            ax_raw.plot(r_use, mad["null_mean"], color="0.35", lw=1.5, label="Null mean")
            ax_raw.fill_between(
                r_use,
                mad["ce_lo"],
                mad["ce_hi"],
                color="0.75",
                alpha=0.55,
                label=f"{100 * float(mad['confidence']):.0f}% CE",
            )
            ax_raw.axvline(mad["r_at_max_nm"], color="C3", ls=":", lw=1.0, alpha=0.8)
            ax_norm.plot(r_use, mad["normalized_obs"], color=observed_color, lw=2.0, label="Normalized obs")
            ax_norm.axhline(1.0, color="0.4", ls="--", lw=1.0)
            ax_norm.axhline(-1.0, color="0.4", ls="--", lw=1.0)
            reject_txt = "reject H0" if mad["rejects_null"] else "fail to reject H0"
            ax_raw.set_title(
                f"{null_label}\n"
                f"T={mad['T_obs']:.3g} / Tcrit={mad['T_critical']:.3g} "
                f"(p={mad['p_mad']:.3g}; {reject_txt})",
                fontsize=9,
            )
        else:
            ax_raw.set_title(
                f"{null_label}\nskipped (n={mad['n_null_curves']} < {mad['min_null_curves']})",
                fontsize=9,
            )
            ax_norm.set_title("MAD not run", fontsize=9)
        ax_raw.axhline(0.0, color="0.5", ls="--", lw=0.8)
        ax_raw.set_xlabel("r (nm)")
        ax_raw.set_ylabel(value_label)
        ax_raw.legend(fontsize=7, loc="best")
        ax_norm.set_xlabel("r (nm)")
        ax_norm.set_ylabel(f"({value_label} − μ_null) / CE half-width")
        ax_norm.legend(fontsize=7, loc="best")
        if len(r_use):
            for ax in (ax_raw, ax_norm):
                ax.set_xlim(float(r_use[0]), float(r_use[-1]))
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _write_mad_outputs(
    *,
    out_dir: Path,
    figures_dir: Path | None,
    zone_name: str,
    r_vals: np.ndarray,
    observed_by_family: dict[str, np.ndarray],
    null_curves_by_family: dict[str, list[tuple[str, np.ndarray]]],
    write_figures: bool,
    figure_title_prefix: str,
) -> tuple[Path, Path]:
    """Run MAD for each family × null × r-range (≥1000 curves), for every family in
    ``observed_by_family`` (see ``ALL_FAMILIES``). Writes one combined summary/curves CSV
    (with a ``curve_family`` column) and one figure per family × r-range."""
    summary_rows: list[dict] = []
    curve_frames: list[pd.DataFrame] = []

    for fam, observed in observed_by_family.items():
        value_label = f"{_family_tag(fam)}(r)"
        mad_by_range: dict[str, list[dict]] = {label: [] for label, _, _ in MAD_R_RANGES}
        for null_name, null_curves in null_curves_by_family[fam]:
            for mad in run_mad_tests_over_r_ranges(
                observed,
                null_curves,
                r_vals,
                null_name=null_name,
                min_null_curves=MAD_MIN_NULL_CURVES,
            ):
                mad_by_range.setdefault(str(mad["r_range"]), []).append(mad)
                summary_rows.append(
                    mad_result_to_summary_row(
                        mad,
                        extra_cols={"active_zone_name": zone_name, "curve_family": fam},
                    )
                )
                curves_df = mad_result_to_curves_dataframe(mad, r_vals, observed=observed)
                curves_df.insert(0, "curve_family", fam)
                curves_df.insert(0, "active_zone_name", zone_name)
                curve_frames.append(curves_df)

        if write_figures and figures_dir is not None:
            for r_range, mad_results in mad_by_range.items():
                if not mad_results:
                    continue
                suffix = "" if r_range == "full" else f"_{r_range.replace('-', '_')}"
                _plot_mad_panels(
                    mad_results=mad_results,
                    output_path=figures_dir / f"ripley_{fam}_mad_vs_nulls{suffix}.png",
                    title=f"{figure_title_prefix} | {value_label} | r-range={r_range}",
                    observed_label=f"Observed {value_label}",
                    value_label=value_label,
                )

    summary_path = out_dir / "ripley_mad_summary.csv"
    curves_path = out_dir / "ripley_mad_curves.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    pd.concat(curve_frames, ignore_index=True).to_csv(curves_path, index=False)
    return summary_path, curves_path


def _segregation_band_columns(
    curves: np.ndarray,
    r_vals: np.ndarray,
    prefix: str,
) -> dict[str, np.ndarray]:
    lo, mean, hi = _percentile_band(curves)
    sd = _prism_sd_envelope_columns(curves, r_vals, prefix=prefix)
    n_r = len(r_vals)
    if len(lo) != n_r:
        nan = np.full(n_r, np.nan)
        lo = mean = hi = nan
    return {
        f"{prefix}_mean": sd[f"{prefix}_mean"],
        f"{prefix}_sd": sd[f"{prefix}_sd"],
        f"{prefix}_sd_envelope_lo": sd[f"{prefix}_sd_envelope_lo"],
        f"{prefix}_sd_envelope_hi": sd[f"{prefix}_sd_envelope_hi"],
        f"{prefix}_sem": sd[f"{prefix}_sem"],
        f"{prefix}_sem_envelope_lo": sd[f"{prefix}_sem_envelope_lo"],
        f"{prefix}_sem_envelope_hi": sd[f"{prefix}_sem_envelope_hi"],
        f"{prefix}_envelope_lo": lo,
        f"{prefix}_envelope_hi": hi,
        f"{prefix}_band_mean": mean,
    }


def build_monomer_dimer_individual_curves_table(
    *,
    zone_name: str,
    r_vals: np.ndarray,
    observed_by_family: dict[str, np.ndarray],
    perm_curves_by_family: dict[str, np.ndarray],
    seg_greedy_curves_by_family: dict[str, np.ndarray],
    n_monomer: int,
    n_dimer: int,
    window_volume_nm3: float,
) -> pd.DataFrame:
    """Long table of every individual curve (observed + all control replicates), for every
    family in ``ALL_FAMILIES`` (distinguished by a ``curve_family`` column)."""
    extras = {
        "active_zone_name": zone_name,
        "window_mode": WINDOW_MODE,
        "n_monomer": int(n_monomer),
        "n_dimer": int(n_dimer),
        "window_volume_nm3": float(window_volume_nm3),
    }
    frames: list[pd.DataFrame] = []
    for fam in observed_by_family:
        fam_extras = {**extras, "curve_family": fam}
        frames.append(
            curves_matrix_to_long_dataframe(
                np.atleast_2d(observed_by_family[fam]),
                r_vals,
                curve_type="observed",
                extra_cols=fam_extras,
            )
        )
        frames.append(
            curves_matrix_to_long_dataframe(
                perm_curves_by_family[fam],
                r_vals,
                curve_type="label_permutation",
                extra_cols=fam_extras,
            )
        )
        frames.append(
            curves_matrix_to_long_dataframe(
                seg_greedy_curves_by_family[fam],
                r_vals,
                curve_type="segregation_greedy",
                extra_cols=fam_extras,
            )
        )
    nonempty = [f for f in frames if not f.empty]
    if not nonempty:
        return pd.DataFrame()
    return pd.concat(nonempty, ignore_index=True)


def build_monomer_dimer_prism_table(
    *,
    zone_name: str,
    r_vals: np.ndarray,
    observed_by_family: dict[str, np.ndarray],
    perm_curves_by_family: dict[str, np.ndarray],
    seg_greedy_curves_by_family: dict[str, np.ndarray],
    n_monomer: int,
    n_dimer: int,
    n_perm: int,
    n_segregation: int,
    window_volume_nm3: float,
) -> pd.DataFrame:
    """Per-zone Prism table: observed value plus label-permutation and segregation
    controls, for every family in ``ALL_FAMILIES``. Column names for the ``l12`` family
    (``observed_L12``, ``control_L12_*``, ``segregation_greedy_L12_*``) are unchanged from
    before the six-family extension.
    """
    n_r = len(r_vals)
    per_family: dict[str, dict] = {}
    for fam in observed_by_family:
        tag = _family_tag(fam)
        perm_lo, perm_mean, perm_hi = _percentile_band(perm_curves_by_family[fam])
        perm_sd = _prism_sd_envelope_columns(
            perm_curves_by_family[fam], r_vals, prefix=f"control_{tag}"
        )
        if len(perm_lo) != n_r:
            nan = np.full(n_r, np.nan)
            perm_lo = perm_mean = perm_hi = nan
        seg = _segregation_band_columns(
            seg_greedy_curves_by_family[fam], r_vals, prefix=f"segregation_greedy_{tag}"
        )
        per_family[fam] = {"tag": tag, "perm_sd": perm_sd, "perm_lo": perm_lo, "perm_hi": perm_hi, "seg": seg}

    rows: list[dict] = []
    for i, r_nm in enumerate(r_vals):
        row: dict = {
            "active_zone_name": zone_name,
            "window_mode": WINDOW_MODE,
            "r_nm": float(r_nm),
        }
        for fam, data in per_family.items():
            tag = data["tag"]
            perm_sd = data["perm_sd"]
            seg = data["seg"]
            row.update(
                {
                    f"observed_{tag}": float(observed_by_family[fam][i]),
                    f"control_{tag}_mean": float(perm_sd[f"control_{tag}_mean"][i]),
                    f"control_{tag}_sd": float(perm_sd[f"control_{tag}_sd"][i]),
                    f"control_{tag}_sd_envelope_lo": float(perm_sd[f"control_{tag}_sd_envelope_lo"][i]),
                    f"control_{tag}_sd_envelope_hi": float(perm_sd[f"control_{tag}_sd_envelope_hi"][i]),
                    f"control_{tag}_sem": float(perm_sd[f"control_{tag}_sem"][i]),
                    f"control_{tag}_sem_envelope_lo": float(perm_sd[f"control_{tag}_sem_envelope_lo"][i]),
                    f"control_{tag}_sem_envelope_hi": float(perm_sd[f"control_{tag}_sem_envelope_hi"][i]),
                    f"control_{tag}_envelope_lo": float(data["perm_lo"][i]),
                    f"control_{tag}_envelope_hi": float(data["perm_hi"][i]),
                    f"segregation_greedy_{tag}_mean": float(seg[f"segregation_greedy_{tag}_mean"][i]),
                    f"segregation_greedy_{tag}_sd": float(seg[f"segregation_greedy_{tag}_sd"][i]),
                    f"segregation_greedy_{tag}_sd_envelope_lo": float(
                        seg[f"segregation_greedy_{tag}_sd_envelope_lo"][i]
                    ),
                    f"segregation_greedy_{tag}_sd_envelope_hi": float(
                        seg[f"segregation_greedy_{tag}_sd_envelope_hi"][i]
                    ),
                    f"segregation_greedy_{tag}_sem": float(seg[f"segregation_greedy_{tag}_sem"][i]),
                    f"segregation_greedy_{tag}_sem_envelope_lo": float(
                        seg[f"segregation_greedy_{tag}_sem_envelope_lo"][i]
                    ),
                    f"segregation_greedy_{tag}_sem_envelope_hi": float(
                        seg[f"segregation_greedy_{tag}_sem_envelope_hi"][i]
                    ),
                    f"segregation_greedy_{tag}_envelope_lo": float(
                        seg[f"segregation_greedy_{tag}_envelope_lo"][i]
                    ),
                    f"segregation_greedy_{tag}_envelope_hi": float(
                        seg[f"segregation_greedy_{tag}_envelope_hi"][i]
                    ),
                }
            )
        row.update(
            {
                "n_monomer": int(n_monomer),
                "n_dimer": int(n_dimer),
                "n_permutations": int(n_perm),
                "n_segregation_replicates": int(n_segregation),
                "envelope_percentile_lo": float(RIPLEY_PERCENTILE_LO),
                "envelope_percentile_hi": float(RIPLEY_PERCENTILE_HI),
                "window_volume_nm3": float(window_volume_nm3),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def build_pooled_monomer_dimer_prism_table(df: pd.DataFrame) -> pd.DataFrame:
    """Pooled mean ± SD of observed/control/segregation curves across zones, per tomogram
    set, for every family in ``ALL_FAMILIES``.

    For L-families (``l12``, ``l21``, ``l_combined``), reports both L-space mean±SD/SEM
    (``*_mean``, ``*_sd_*``) and K-space mean±SD/SEM mapped to L (``*_mean_from_k``,
    ``*_sd_*_from_k``, ``*_sem_*_from_k``) — the K-scale version is the statistically sound
    one (see ``ripley_library``'s K→L pooling docs); the L-space version is kept as a
    reference. G-families (``g12``, ``g21``, ``g_combined``) get only the direct treatment —
    g is a linear ratio, not a nonlinear transform of K, so no K-scale detour is needed.
    Column names for the ``l12`` family are unchanged from before the six-family extension.
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
        family_data: dict[str, dict] = {}
        for fam in ALL_FAMILIES:
            tag = _family_tag(fam)
            r_vals, obs_curves = _extract_curves_matrix(sub, fam)
            if len(obs_curves) == 0:
                continue
            _, ctrl_curves = _extract_curves_matrix(sub, f"perm_{fam}_mean")
            _, seg_curves = _extract_curves_matrix(sub, f"seg_greedy_{fam}_mean")

            is_l_family = fam in L_FAMILIES
            obs_sd = _prism_sd_envelope_columns(obs_curves, r_vals, prefix=f"observed_{tag}")
            ctrl_sd = _prism_sd_envelope_columns(ctrl_curves, r_vals, prefix=f"control_{tag}")
            seg_sd = _prism_sd_envelope_columns(
                seg_curves, r_vals, prefix=f"segregation_greedy_{tag}"
            )
            obs_from_k = ctrl_from_k = seg_from_k = None
            if is_l_family:
                obs_from_k = prism_sd_envelope_columns_from_averaged_k12(
                    obs_curves, r_vals, prefix=f"observed_{tag}"
                )
                ctrl_from_k = prism_sd_envelope_columns_from_averaged_k12(
                    ctrl_curves, r_vals, prefix=f"control_{tag}"
                )
                seg_from_k = prism_sd_envelope_columns_from_averaged_k12(
                    seg_curves, r_vals, prefix=f"segregation_greedy_{tag}"
                )

            family_data[fam] = {
                "tag": tag,
                "is_l_family": is_l_family,
                "obs_sd": obs_sd,
                "ctrl_sd": ctrl_sd,
                "seg_sd": seg_sd,
                "obs_from_k": obs_from_k,
                "ctrl_from_k": ctrl_from_k,
                "seg_from_k": seg_from_k,
                "n_zone_curves": len(obs_curves),
            }
            if anchor_r_vals is None:
                anchor_r_vals = r_vals

        if anchor_r_vals is None:
            continue

        n_tomograms = int(sub["tomogram_name"].nunique()) if "tomogram_name" in sub.columns else 0
        n_zones = int(
            sub[["tomogram_name", "alignment_dir", "active_zone_name"]].drop_duplicates().shape[0]
        )

        for i, r_nm in enumerate(anchor_r_vals):
            row: dict = {
                "set_name": set_name,
                "window_mode": WINDOW_MODE,
                "r_nm": float(r_nm),
                "n_tomograms": n_tomograms,
                "n_active_zones": n_zones,
            }
            for fam, data in family_data.items():
                tag = data["tag"]
                if fam == "l12":
                    row["n_zone_curves"] = int(data["n_zone_curves"])
                row[f"n_zone_curves_{tag}"] = int(data["n_zone_curves"])
                for role, sd in (
                    ("observed", data["obs_sd"]),
                    ("control", data["ctrl_sd"]),
                    ("segregation_greedy", data["seg_sd"]),
                ):
                    prefix = f"{role}_{tag}"
                    row[f"{prefix}_mean"] = float(sd[f"{prefix}_mean"][i])
                    row[f"{prefix}_sd"] = float(sd[f"{prefix}_sd"][i])
                    row[f"{prefix}_sd_envelope_lo"] = float(sd[f"{prefix}_sd_envelope_lo"][i])
                    row[f"{prefix}_sd_envelope_hi"] = float(sd[f"{prefix}_sd_envelope_hi"][i])
                    row[f"{prefix}_sem"] = float(sd[f"{prefix}_sem"][i])
                    row[f"{prefix}_sem_envelope_lo"] = float(sd[f"{prefix}_sem_envelope_lo"][i])
                    row[f"{prefix}_sem_envelope_hi"] = float(sd[f"{prefix}_sem_envelope_hi"][i])
                if data["is_l_family"]:
                    for role, from_k in (
                        ("observed", data["obs_from_k"]),
                        ("control", data["ctrl_from_k"]),
                        ("segregation_greedy", data["seg_from_k"]),
                    ):
                        prefix = f"{role}_{tag}"
                        row[f"{prefix}_mean_from_k"] = float(from_k[f"{prefix}_mean_from_k"][i])
                        row[f"{prefix}_sd_from_k"] = float(from_k[f"{prefix}_sd_from_k"][i])
                        row[f"{prefix}_sd_envelope_lo_from_k"] = float(
                            from_k[f"{prefix}_sd_envelope_lo_from_k"][i]
                        )
                        row[f"{prefix}_sd_envelope_hi_from_k"] = float(
                            from_k[f"{prefix}_sd_envelope_hi_from_k"][i]
                        )
                        row[f"{prefix}_sem_from_k"] = float(from_k[f"{prefix}_sem_from_k"][i])
                        row[f"{prefix}_sem_envelope_lo_from_k"] = float(
                            from_k[f"{prefix}_sem_envelope_lo_from_k"][i]
                        )
                        row[f"{prefix}_sem_envelope_hi_from_k"] = float(
                            from_k[f"{prefix}_sem_envelope_hi_from_k"][i]
                        )
            rows.append(row)
    return pd.DataFrame(rows)


def plot_monomer_dimer_diagnostic(
    tomogram_path: Path,
    alignment_dir: str,
    zone_name: str,
    *,
    monomer_coords: np.ndarray,
    dimer_coords: np.ndarray,
    az_segmentation: dict,
    window: RipleyWindow3D,
    grid_points: np.ndarray,
    grid_spacing_nm: float,
    output_path: Path,
    dropped_monomer_coords: np.ndarray | None = None,
    dropped_dimer_coords: np.ndarray | None = None,
    membrane_max_points: int = 4000,
    n_z_slices: int = 5,
) -> Path | None:
    """
    Thin wrapper around ``plot_ripley_window_geometry_diagnostic`` (shared with the
    AZ-center analysis): monomer AuNPs, dimer AuNPs, pre/post membrane surfaces, the
    ``synaptic_cleft_az_hull`` convex hull used as the Ripley window, and the deterministic
    edge-correction grid points (the literal points the analysis divides by, not a separate
    Monte-Carlo preview of the window).

    ``dropped_monomer_coords``/``dropped_dimer_coords`` (AuNPs outside the cleft hull, always
    dropped before the Ripley computation) are highlighted separately, distinct from the
    analyzed monomer/dimer AuNPs.
    """
    monomer_coords = np.atleast_2d(np.asarray(monomer_coords, dtype=float))
    dimer_coords = np.atleast_2d(np.asarray(dimer_coords, dtype=float))
    dropped_coords = np.vstack(
        [
            np.atleast_2d(np.asarray(dropped_monomer_coords, dtype=float))
            if dropped_monomer_coords is not None and len(dropped_monomer_coords)
            else np.zeros((0, 3)),
            np.atleast_2d(np.asarray(dropped_dimer_coords, dtype=float))
            if dropped_dimer_coords is not None and len(dropped_dimer_coords)
            else np.zeros((0, 3)),
        ]
    )
    pre_outer = np.atleast_2d(np.asarray(az_segmentation.get("presynaptic_outer_coords", []), dtype=float))
    post_outer = np.atleast_2d(np.asarray(az_segmentation.get("postsynaptic_outer_coords", []), dtype=float))

    return plot_ripley_window_geometry_diagnostic(
        tomogram_path,
        alignment_dir,
        zone_name,
        point_groups=[
            {"coords": monomer_coords, "label": "monomer AuNPs", "color": "tab:purple", "marker": "o", "size": 18},
            {"coords": dimer_coords, "label": "dimer AuNPs", "color": "tab:green", "marker": "^", "size": 28},
        ],
        az_segmentation=az_segmentation,
        window=window,
        grid_points=grid_points,
        grid_spacing_nm=grid_spacing_nm,
        output_path=output_path,
        dropped_coords=dropped_coords,
        title_lines=[
            f"Monomer ({len(monomer_coords)}) vs dimer ({len(dimer_coords)}) AuNPs "
            f"({len(pre_outer)}+{len(post_outer)} membrane pts)"
        ],
        membrane_max_points=membrane_max_points,
        n_z_slices=n_z_slices,
        include_3d_panel=True,
        print_prefix="Monomer/dimer diagnostic plot",
    )


def _plot_family_observed_vs_controls(
    *,
    r_vals: np.ndarray,
    observed: np.ndarray,
    perm_curves: np.ndarray,
    seg_greedy_curves: np.ndarray,
    fam: str,
    tomogram_name: str,
    zone_name: str,
    n_monomer: int,
    n_dimer: int,
    n_perm: int,
    n_segregation: int,
    r_max_nm: float,
    output_path: Path,
) -> None:
    """Per-zone observed-vs-null figure for one family (a member of ``ALL_FAMILIES``)."""
    label = _family_tag(fam)
    perm_lo, perm_band_mean, perm_hi = _percentile_band(perm_curves)
    seg_lo, seg_band_mean, seg_hi = _percentile_band(seg_greedy_curves)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(r_vals, observed, color="C0", lw=2, label=f"Observed monomer→dimer {label}")
    if len(perm_band_mean) == len(r_vals):
        ax.plot(r_vals, perm_band_mean, color="0.45", lw=1.5, label="Label-permutation mean")
        ax.fill_between(
            r_vals,
            perm_lo,
            perm_hi,
            color="0.8",
            alpha=0.8,
            label=f"Label-perm {RIPLEY_PERCENTILE_LO:g}–{RIPLEY_PERCENTILE_HI:g}%",
        )
    if len(seg_band_mean) == len(r_vals):
        ax.plot(r_vals, seg_band_mean, color="C3", lw=1.5, label="Greedy segregation mean")
        ax.fill_between(
            r_vals,
            seg_lo,
            seg_hi,
            color="C3",
            alpha=0.2,
            label=f"Greedy seg {RIPLEY_PERCENTILE_LO:g}–{RIPLEY_PERCENTILE_HI:g}%",
        )
    ax.axhline(1.0 if fam in G_FAMILIES else 0.0, color="0.5", ls="--", lw=0.8)
    ax.set_xlabel("r (nm)")
    ax.set_ylabel(f"{label}(r)")
    ax.set_title(
        f"{tomogram_name} | {zone_name}\n"
        f"monomer ({n_monomer}) vs dimer ({n_dimer}) | {label} | "
        f"{int(n_perm)} label perms, {int(n_segregation)} greedy seg reps"
    )
    ax.set_xlim(0.0, float(r_vals[-1]) if len(r_vals) else r_max_nm)
    ax.legend(loc="best", fontsize=7)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_monomer_dimer_ripley_for_zone(
    tomogram_path: Path,
    alignment_dir: str,
    zone_name: str,
    active_zone_index: int,
    *,
    monomer_star_pattern: Optional[str] = None,
    dimer_star_pattern: Optional[str] = None,
    n_perm: int = MONOMER_DIMER_N_PERM,
    r_max_nm: float = DEFAULT_RIPLEY_R_MAX_NM,
    r_step_nm: float = DEFAULT_RIPLEY_R_STEP_NM,
    seed: int = DEFAULT_ANALYSIS_SEED,
    write_figures: bool = True,
) -> dict[str, Path] | None:
    """Observed monomer/dimer K/L/g (see ``ALL_FAMILIES``: direct, reversed, and their
    intensity-weighted combination) with label-permutation and greedy-segregation controls
    for every family.

    Greedy segregation always uses the same replicate count as label permutation (``n_perm``).
    AuNPs outside the Ripley window are always dropped before any statistic is computed. The
    Ripley window is always restricted to the region of the synaptic-cleft hull that also
    sits "between" the pre- and post-synaptic membranes (angle in-betweenness test), matching
    the AZ-center Ripley analysis, since monomer/dimer AuNP positions are only meaningful
    relative to the space between the two membranes.
    """
    tomogram_path = Path(tomogram_path)
    alignment_dir = require_alignment_dir(alignment_dir)

    try:
        loaded = load_monomer_dimer_aunps_for_zone(
            tomogram_path,
            alignment_dir,
            int(active_zone_index),
            monomer_star_pattern=monomer_star_pattern,
            dimer_star_pattern=dimer_star_pattern,
        )
    except FileNotFoundError as exc:
        print(f"  Skipping monomer/dimer Ripley for {zone_name}: {exc}")
        return None

    if "monomer" not in loaded.kinds_loaded or "dimer" not in loaded.kinds_loaded:
        print(
            f"  Skipping monomer/dimer Ripley for {zone_name}: "
            f"need both monomer and dimer STARs (loaded: {loaded.kinds_loaded})"
        )
        return None

    monomer_coords, _ = subset_aunps(loaded.meta, subset="monomer")
    dimer_coords, _ = subset_aunps(loaded.meta, subset="dimer")

    from .activezone import import_active_zone_segmentations

    az_segmentation = import_active_zone_segmentations(
        tomogram_path, alignment_dir=alignment_dir
    ).get(zone_name)

    rng = np.random.default_rng(seed)
    try:
        cleft_coords = load_synaptic_cleft_active_zone_points(
            tomogram_path, alignment_dir, zone_name
        )
        window = build_ripley_window_3d(
            cleft_coords,
            mode=WINDOW_MODE,
            pre_membrane_coords=(
                az_segmentation.get("presynaptic_outer_coords") if az_segmentation is not None else None
            ),
            post_membrane_coords=(
                az_segmentation.get("postsynaptic_outer_coords") if az_segmentation is not None else None
            ),
            use_angle_betweenness=True,
            rng=rng,
        )
    except Exception as exc:
        print(f"  Skipping monomer/dimer Ripley for {zone_name}: {exc}")
        return None

    # AuNPs outside the Ripley window are always dropped -- Ripley's edge correction and
    # volume normalization are only valid for points observed inside the window, so
    # out-of-hull points would bias every statistic rather than just look odd.
    # ``pool_keep_mask`` (pool order: monomer then dimer) also keeps the disk-re-read
    # STAR export in sync below.
    monomer_inside = _points_inside_hull(monomer_coords, window.hull)
    dimer_inside = _points_inside_hull(dimer_coords, window.hull)
    dropped_monomer_coords = monomer_coords[~monomer_inside]
    dropped_dimer_coords = dimer_coords[~dimer_inside]
    monomer_coords = monomer_coords[monomer_inside]
    dimer_coords = dimer_coords[dimer_inside]
    pool_keep_mask = np.concatenate([monomer_inside, dimer_inside])
    if len(dropped_monomer_coords) or len(dropped_dimer_coords):
        print(
            f"  Dropping {len(dropped_monomer_coords)} monomer + "
            f"{len(dropped_dimer_coords)} dimer AuNP(s) outside the cleft hull for {zone_name}"
        )

    n_monomer = len(monomer_coords)
    n_dimer = len(dimer_coords)
    if n_monomer < MIN_POINTS_PER_CLASS or n_dimer < MIN_POINTS_PER_CLASS:
        print(
            f"  Skipping monomer/dimer Ripley for {zone_name}: "
            f"too few points (monomer={n_monomer}, dimer={n_dimer} inside the cleft hull; "
            f"need >= {MIN_POINTS_PER_CLASS} each)"
        )
        return None

    r_vals = _ripley_r_grid(r_max_nm, r_step_nm)
    pool = np.vstack([np.atleast_2d(monomer_coords), np.atleast_2d(dimer_coords)])
    n_perm_int = int(n_perm)
    # Segregation always matches label-permutation replicate count.
    n_segregation_int = n_perm_int
    n_perm_workers = _default_ripley_perm_workers(n_perm_int)
    progress_total = 1 + max(n_perm_int, 0) + max(n_segregation_int, 0)
    pbar = tqdm(
        total=progress_total,
        desc=f"{zone_name} monomer/dimer Ripley",
        unit="eval",
        file=sys.stdout,
        dynamic_ncols=True,
        leave=False,
    )
    try:
        # Deterministic grid-based edge correction (matching the AZ-center Ripley analysis
        # rather than Monte Carlo sampling) -- precomputed once for every pooled point as a
        # potential focus in either direction, and reused for observed + both null
        # generators. Also reconciles window.volume_nm3 with this grid's own volume estimate
        # (same trick as the AZ-center analysis) so K's outer V/(n1*n2) scaling and the
        # edge-factor denominator agree on an identical V.
        grid_points = build_window_grid_points(window, MONOMER_DIMER_EDGE_GRID_SPACING_NM)
        grid_volume_nm3 = float(len(grid_points)) * (MONOMER_DIMER_EDGE_GRID_SPACING_NM ** 3)
        window = dataclasses_replace(window, volume_nm3=grid_volume_nm3)
        pool_edge_factors = _isotropic_edge_factors_grid(
            pool, r_vals, grid_points, MONOMER_DIMER_EDGE_GRID_SPACING_NM
        )

        pbar.set_postfix_str("observed", refresh=False)
        k12_obs = cross_k12_3d_isotropic(
            monomer_coords, dimer_coords, r_vals, window, rng,
            edge_factors=pool_edge_factors[:n_monomer],
        )
        k21_obs = cross_k12_3d_isotropic(
            dimer_coords, monomer_coords, r_vals, window, rng,
            edge_factors=pool_edge_factors[n_monomer:],
        )
        observed = derive_symmetric_k_l_g_families(
            k12_obs, k21_obs, n_monomer, n_dimer, r_vals, g_bin_width_nm=r_step_nm
        )
        pbar.update(1)

        perm_k = label_permutation_k_bidirectional_curves(
            monomer_coords,
            dimer_coords,
            r_vals,
            window,
            n_perm=n_perm_int,
            seed=seed,
            rng=rng,
            n_workers=n_perm_workers,
            pbar=pbar if n_perm_int > 0 else None,
            pool_edge_factors=pool_edge_factors,
        )
        perm_all = derive_symmetric_k_l_g_families(
            perm_k["k12"], perm_k["k21"], n_monomer, n_dimer, r_vals, g_bin_width_nm=r_step_nm
        )

        seg_rng = np.random.default_rng(int(seed) + 1_000_000)
        seg_k = _greedy_segregation_k_bidirectional_curves(
            pool,
            n_monomer,
            n_dimer,
            r_vals,
            window,
            seg_rng,
            n_rep=n_segregation_int,
            pool_edge_factors=pool_edge_factors,
            seed=int(seed) + 2_000_000,
            n_workers=n_perm_workers,
            pbar=pbar if n_segregation_int > 0 else None,
        )
        seg_all = derive_symmetric_k_l_g_families(
            seg_k["k12"], seg_k["k21"], n_monomer, n_dimer, r_vals, g_bin_width_nm=r_step_nm
        )
    finally:
        pbar.close()

    # Past the radius where a focus's ball has swallowed the entire window, every K value
    # feeding a shell's difference is forced onto the CSR reference curve by construction
    # (see _isotropic_edge_factors_grid's docstring) — L correctly reflects this as L→0, but
    # g's finite difference of two such saturated K values goes spuriously flat near 1
    # instead of NaN, since pair_correlation_from_k_diff has no visibility into how much
    # window is actually left. NaN out those shells using the real (fixed) monomer/dimer
    # coordinates as the reliability yardstick for every curve, observed or null — a given
    # null replicate's random class1/class2 labeling changes which points play the K12/K21
    # role, but the window-visibility geometry is governed by where the real points sit, not
    # by which label they're wearing in a given replicate.
    g12_unreliable = g_shell_reliability_mask(
        monomer_coords, grid_points, r_vals, bin_width_nm=r_step_nm
    )
    g21_unreliable = g_shell_reliability_mask(
        dimer_coords, grid_points, r_vals, bin_width_nm=r_step_nm
    )
    g_combined_unreliable = g12_unreliable | g21_unreliable
    for results in (observed, perm_all, seg_all):
        results["g12"] = np.where(g12_unreliable, np.nan, results["g12"])
        results["g21"] = np.where(g21_unreliable, np.nan, results["g21"])
        results["g_combined"] = np.where(g_combined_unreliable, np.nan, results["g_combined"])

    perm_mean_by_family: dict[str, np.ndarray] = {}
    seg_mean_by_family: dict[str, np.ndarray] = {}
    for fam in ALL_FAMILIES:
        _, m, _ = _percentile_band(perm_all[fam])
        perm_mean_by_family[fam] = m if len(m) == len(r_vals) else np.full(len(r_vals), np.nan)
        _, m, _ = _percentile_band(seg_all[fam])
        seg_mean_by_family[fam] = m if len(m) == len(r_vals) else np.full(len(r_vals), np.nan)

    tomogram_name = tomogram_path.name
    out_dir = (
        tomogram_path
        / alignment_dir
        / "STT_results"
        / "aunps"
        / "aunp_monomer_dimer_ripley"
        / zone_name
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = out_dir / "figures"
    if write_figures:
        figures_dir.mkdir(parents=True, exist_ok=True)
        try:
            if az_segmentation is not None:
                plot_monomer_dimer_diagnostic(
                    tomogram_path,
                    alignment_dir,
                    zone_name,
                    monomer_coords=monomer_coords,
                    dimer_coords=dimer_coords,
                    az_segmentation=az_segmentation,
                    window=window,
                    grid_points=grid_points,
                    grid_spacing_nm=MONOMER_DIMER_EDGE_GRID_SPACING_NM,
                    dropped_monomer_coords=dropped_monomer_coords,
                    dropped_dimer_coords=dropped_dimer_coords,
                    output_path=figures_dir / "geometry_diagnostic.png",
                )
        except Exception as diag_exc:
            print(f"  Skipping monomer/dimer diagnostic plot for {zone_name}: {diag_exc}")

    simulated_stars_dir: Path | None = None
    try:
        pool_df = _load_pool_star_dataframe_for_export(
            tomogram_path,
            alignment_dir,
            int(active_zone_index),
            monomer_star_pattern=monomer_star_pattern,
            dimer_star_pattern=dimer_star_pattern,
        )
        if len(pool_df) != len(pool_keep_mask):
            raise ValueError(
                f"Pool STAR row count ({len(pool_df)}) != pre-drop pooled coords "
                f"({len(pool_keep_mask)})"
            )
        pool_df = pool_df.loc[pool_keep_mask].reset_index(drop=True)
        if len(pool_df) != len(pool):
            raise ValueError(
                f"Pool STAR row count ({len(pool_df)}) != pooled coords ({len(pool)})"
            )
        simulated_stars_dir = _write_simulated_null_star_examples(
            out_dir=out_dir,
            pool=pool,
            pool_df=pool_df,
            n_monomer=n_monomer,
            n_dimer=n_dimer,
            analysis_seed=int(seed),
        )
    except Exception as exc:
        print(f"  Warning: could not write simulated null STAR examples for {zone_name}: {exc}")

    curves_data: dict = {
        "active_zone_name": zone_name,
        "active_zone_index": int(active_zone_index),
        "window_mode": WINDOW_MODE,
        "r_nm": r_vals,
    }
    for fam in ALL_FAMILIES:
        curves_data[fam] = observed[fam]
        curves_data[f"perm_{fam}_mean"] = perm_mean_by_family[fam]
        curves_data[f"seg_greedy_{fam}_mean"] = seg_mean_by_family[fam]
    curves_data.update(
        {
            "n_monomer": n_monomer,
            "n_dimer": n_dimer,
            "n_permutations": int(n_perm_int),
            "n_segregation_replicates": int(n_segregation_int),
            "window_volume_nm3": float(window.volume_nm3),
        }
    )
    curves_df = pd.DataFrame(curves_data)
    curves_path = out_dir / "ripley_l12_curves.csv"
    curves_df.to_csv(curves_path, index=False)

    individual_df = build_monomer_dimer_individual_curves_table(
        zone_name=zone_name,
        r_vals=r_vals,
        observed_by_family={fam: observed[fam] for fam in ALL_FAMILIES},
        perm_curves_by_family={fam: perm_all[fam] for fam in ALL_FAMILIES},
        seg_greedy_curves_by_family={fam: seg_all[fam] for fam in ALL_FAMILIES},
        n_monomer=n_monomer,
        n_dimer=n_dimer,
        window_volume_nm3=float(window.volume_nm3),
    )
    individual_path = out_dir / "ripley_l12_individual_curves.csv"
    individual_df.to_csv(individual_path, index=False)
    # Prism-friendly wide tables (one file per curve family × role).
    for fam in ALL_FAMILIES:
        curves_matrix_to_wide_dataframe(
            np.atleast_2d(observed[fam]), r_vals, curve_type="observed"
        ).to_csv(out_dir / f"ripley_{fam}_individual_observed_wide.csv", index=False)
        curves_matrix_to_wide_dataframe(
            perm_all[fam], r_vals, curve_type="label_permutation"
        ).to_csv(out_dir / f"ripley_{fam}_individual_label_permutation_wide.csv", index=False)
        curves_matrix_to_wide_dataframe(
            seg_all[fam], r_vals, curve_type="segregation_greedy"
        ).to_csv(out_dir / f"ripley_{fam}_individual_segregation_greedy_wide.csv", index=False)

    prism_df = build_monomer_dimer_prism_table(
        zone_name=zone_name,
        r_vals=r_vals,
        observed_by_family={fam: observed[fam] for fam in ALL_FAMILIES},
        perm_curves_by_family={fam: perm_all[fam] for fam in ALL_FAMILIES},
        seg_greedy_curves_by_family={fam: seg_all[fam] for fam in ALL_FAMILIES},
        n_monomer=n_monomer,
        n_dimer=n_dimer,
        n_perm=n_perm_int,
        n_segregation=n_segregation_int,
        window_volume_nm3=float(window.volume_nm3),
    )
    prism_path = out_dir / "ripley_l12_prism.csv"
    prism_df.to_csv(prism_path, index=False)
    _prism_long_to_wide(prism_df, id_cols=["active_zone_name", "window_mode"]).to_csv(
        out_dir / "ripley_l12_prism_wide.csv", index=False
    )

    if write_figures:
        for fam in ALL_FAMILIES:
            _plot_family_observed_vs_controls(
                r_vals=r_vals,
                observed=observed[fam],
                perm_curves=perm_all[fam],
                seg_greedy_curves=seg_all[fam],
                fam=fam,
                tomogram_name=tomogram_name,
                zone_name=zone_name,
                n_monomer=n_monomer,
                n_dimer=n_dimer,
                n_perm=n_perm_int,
                n_segregation=n_segregation_int,
                r_max_nm=r_max_nm,
                output_path=figures_dir / f"ripley_{fam}_observed_vs_controls.png",
            )

    mad_summary_path, mad_curves_path = _write_mad_outputs(
        out_dir=out_dir,
        figures_dir=figures_dir if write_figures else None,
        zone_name=zone_name,
        r_vals=r_vals,
        observed_by_family={fam: observed[fam] for fam in ALL_FAMILIES},
        null_curves_by_family={
            fam: [
                ("label_permutation", perm_all[fam]),
                ("segregation_greedy", seg_all[fam]),
            ]
            for fam in ALL_FAMILIES
        },
        write_figures=write_figures,
        figure_title_prefix=(
            f"{tomogram_name} | {zone_name} | MAD vs nulls "
            f"(run only if n≥{MAD_MIN_NULL_CURVES})"
        ),
    )

    meta = {
        "tomogram_name": tomogram_name,
        "alignment_dir": alignment_dir,
        "active_zone_name": zone_name,
        "active_zone_index": int(active_zone_index),
        "window_mode": WINDOW_MODE,
        "curve_families": list(ALL_FAMILIES),
        "n_monomer": int(n_monomer),
        "n_dimer": int(n_dimer),
        "n_permutations": int(n_perm_int),
        "n_perm_workers": int(n_perm_workers),
        "n_segregation_replicates": int(n_segregation_int),
        "segregation_matches_n_perm": True,
        "segregation_modes": ["greedy_random_cluster_class"],
        "segregation_seed_strategy": "random_point",
        "segregation_cluster_class_choice": "random_per_replicate_monomer_or_dimer",
        "window_volume_nm3": float(window.volume_nm3),
        "window_uses_angle_betweenness": bool(window.use_angle_betweenness),
        "n_monomer_dropped_outside_hull": int(len(dropped_monomer_coords)),
        "n_dimer_dropped_outside_hull": int(len(dropped_dimer_coords)),
        "control_label_permutation": "label_permutation_preserving_class_counts",
        "control_segregation": "greedy_nearest_neighbor_cluster_random_class",
        "ripley_edge_correction": "isotropic_3d_grid",
        "edge_correction_grid_spacing_nm": float(MONOMER_DIMER_EDGE_GRID_SPACING_NM),
        "g_estimator": "pair_correlation_from_k_diff",
        "mad_min_null_curves": int(MAD_MIN_NULL_CURVES),
        "mad_nulls": [
            "label_permutation",
            "segregation_greedy",
        ],
        "mad_r_ranges": [label for label, _, _ in MAD_R_RANGES],
        "seed": int(seed),
        "n_simulated_null_star_examples": int(N_SIMULATED_NULL_STAR_EXAMPLES),
        "simulated_null_stars_dir": (
            str(simulated_stars_dir) if simulated_stars_dir is not None else None
        ),
    }
    with open(out_dir / "run_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(
        f"  Monomer/dimer Ripley ({zone_name}): "
        f"{n_monomer} monomer, {n_dimer} dimer, {int(n_perm_int)} perms, "
        f"{int(n_segregation_int)} greedy seg reps -> {out_dir}"
    )
    result = {
        "curves_path": curves_path,
        "individual_curves_path": individual_path,
        "prism_path": prism_path,
        "mad_summary_path": mad_summary_path,
        "mad_curves_path": mad_curves_path,
        "output_dir": out_dir,
    }
    if simulated_stars_dir is not None:
        result["simulated_null_stars_dir"] = simulated_stars_dir
    return result


def run_monomer_dimer_ripley_for_tomogram(
    tomogram_path: Path,
    alignment_dir: str,
    *,
    active_zone_indices: Sequence[int] | None,
    monomer_star_pattern: Optional[str] = None,
    dimer_star_pattern: Optional[str] = None,
    n_perm: int = MONOMER_DIMER_N_PERM,
    r_max_nm: float = DEFAULT_RIPLEY_R_MAX_NM,
    r_step_nm: float = DEFAULT_RIPLEY_R_STEP_NM,
    seed: int = DEFAULT_ANALYSIS_SEED,
    write_figures: bool = True,
) -> tuple[list[pd.DataFrame], list[pd.DataFrame], list[pd.DataFrame], list[pd.DataFrame]]:
    """Run monomer vs dimer Ripley for all mapped active zones in one tomogram.

    Returns ``(curve_frames, prism_frames, individual_frames, mad_summary_frames)``.
    Segregation replicate count always matches ``n_perm``.
    """
    from .activezone import load_active_zone_mapping

    if n_perm is None:
        n_perm = MONOMER_DIMER_N_PERM
    tomogram_path = Path(tomogram_path)
    alignment_dir = require_alignment_dir(alignment_dir)
    az_mapping = load_active_zone_mapping(tomogram_path, alignment_dir) or {}
    if not az_mapping:
        print("No active zone mapping; skipping monomer/dimer Ripley analyses")
        return [], [], [], []

    az_mapping = {int(k): v for k, v in az_mapping.items()}
    indices = list(active_zone_indices) if active_zone_indices is not None else sorted(az_mapping)

    curve_frames: list[pd.DataFrame] = []
    individual_frames: list[pd.DataFrame] = []
    prism_frames: list[pd.DataFrame] = []
    mad_summary_frames: list[pd.DataFrame] = []

    zone_tasks = [
        (int(az_idx), az_mapping[az_idx])
        for az_idx in indices
        if int(az_idx) in az_mapping
    ]
    for az_idx in indices:
        if int(az_idx) not in az_mapping:
            print(f"  Active zone index {az_idx} not in mapping, skipping monomer/dimer Ripley")
    for az_idx, zone_name in tqdm(
        zone_tasks,
        desc=f"{tomogram_path.name} monomer/dimer Ripley zones",
        unit="zone",
        file=sys.stdout,
        dynamic_ncols=True,
        leave=True,
    ):
        result = run_monomer_dimer_ripley_for_zone(
            tomogram_path,
            alignment_dir,
            zone_name,
            az_idx,
            monomer_star_pattern=monomer_star_pattern,
            dimer_star_pattern=dimer_star_pattern,
            n_perm=n_perm,
            r_max_nm=r_max_nm,
            r_step_nm=r_step_nm,
            seed=seed,
            write_figures=write_figures,
        )
        if result is None:
            continue
        curves_path = result["curves_path"]
        individual_path = result["individual_curves_path"]
        prism_path = result["prism_path"]
        mad_summary_path = result["mad_summary_path"]
        if curves_path.is_file():
            curve_frames.append(pd.read_csv(curves_path))
        if individual_path.is_file():
            individual_frames.append(pd.read_csv(individual_path))
        if prism_path.is_file():
            prism_frames.append(pd.read_csv(prism_path))
        if mad_summary_path.is_file():
            mad_df = pd.read_csv(mad_summary_path)
            mad_df.insert(0, "tomogram_name", tomogram_path.name)
            mad_df.insert(1, "alignment_dir", alignment_dir)
            mad_df["active_zone_index"] = int(az_idx)
            mad_df["n_permutations"] = int(n_perm) if n_perm is not None else MONOMER_DIMER_N_PERM
            mad_df["n_segregation_replicates"] = (
                int(n_perm) if n_perm is not None else MONOMER_DIMER_N_PERM
            )
            mad_df["seed"] = int(seed)
            mad_df["window_mode"] = WINDOW_MODE
            mad_summary_frames.append(mad_df)

    return curve_frames, prism_frames, individual_frames, mad_summary_frames


def _plot_pooled_family_figure(
    grp: pd.DataFrame,
    *,
    fam: str,
    set_name: str,
    output_dir: Path,
) -> Path | None:
    """One pooled observed-vs-null figure for one family (a member of ``ALL_FAMILIES``).

    Returns ``None`` (writes nothing) if ``grp`` doesn't have this family's columns — e.g.
    pooling older per-zone output written before the six-family extension.
    """
    tag = _family_tag(fam)
    obs_mean_col = f"observed_{tag}_mean"
    if obs_mean_col not in grp.columns:
        return None

    r_vals = grp["r_nm"].to_numpy(dtype=float)
    obs_mean = grp[obs_mean_col].to_numpy(dtype=float)
    obs_lo = grp[f"observed_{tag}_sd_envelope_lo"].to_numpy(dtype=float)
    obs_hi = grp[f"observed_{tag}_sd_envelope_hi"].to_numpy(dtype=float)
    nan_series = pd.Series(np.nan, index=grp.index)
    ctrl_mean = grp.get(f"control_{tag}_mean", nan_series).to_numpy(dtype=float)
    ctrl_lo = grp.get(f"control_{tag}_sd_envelope_lo", nan_series).to_numpy(dtype=float)
    ctrl_hi = grp.get(f"control_{tag}_sd_envelope_hi", nan_series).to_numpy(dtype=float)
    seg_mean = grp.get(f"segregation_greedy_{tag}_mean", nan_series).to_numpy(dtype=float)
    seg_lo = grp.get(f"segregation_greedy_{tag}_sd_envelope_lo", nan_series).to_numpy(dtype=float)
    seg_hi = grp.get(f"segregation_greedy_{tag}_sd_envelope_hi", nan_series).to_numpy(dtype=float)
    meta = grp.iloc[0]
    is_l_family = fam in L_FAMILIES
    direct_suffix = " (of L)" if is_l_family else ""
    from_k_suffix = " (K→L)" if is_l_family else ""

    set_tag = _safe_name(str(set_name)) or "unspecified"
    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    ax.plot(r_vals, obs_mean, color="C0", lw=2, label=f"Observed mean{direct_suffix}")
    from_k_col = f"observed_{tag}_mean_from_k"
    if from_k_col in grp.columns:
        ax.plot(
            r_vals, grp[from_k_col].to_numpy(dtype=float), color="C0", lw=1.5, ls="--",
            label=f"Observed mean{from_k_suffix}",
        )
    ax.fill_between(r_vals, obs_lo, obs_hi, color="C0", alpha=0.25, label=f"Observed ±SD{direct_suffix}")
    from_k_lo_col = f"observed_{tag}_sd_envelope_lo_from_k"
    if from_k_lo_col in grp.columns:
        ax.fill_between(
            r_vals,
            grp[from_k_lo_col].to_numpy(dtype=float),
            grp[f"observed_{tag}_sd_envelope_hi_from_k"].to_numpy(dtype=float),
            color="C0", alpha=0.12, hatch="///", label=f"Observed ±SD{from_k_suffix}",
        )

    if np.isfinite(ctrl_mean).any():
        ax.plot(r_vals, ctrl_mean, color="0.45", lw=1.5, label=f"Label-perm mean{direct_suffix}")
        from_k_col = f"control_{tag}_mean_from_k"
        if from_k_col in grp.columns:
            ax.plot(
                r_vals, grp[from_k_col].to_numpy(dtype=float), color="0.45", lw=1.3, ls="--",
                label=f"Label-perm mean{from_k_suffix}",
            )
        ax.fill_between(r_vals, ctrl_lo, ctrl_hi, color="0.7", alpha=0.4, label=f"Label-perm ±SD{direct_suffix}")
        from_k_lo_col = f"control_{tag}_sd_envelope_lo_from_k"
        if from_k_lo_col in grp.columns:
            ax.fill_between(
                r_vals,
                grp[from_k_lo_col].to_numpy(dtype=float),
                grp[f"control_{tag}_sd_envelope_hi_from_k"].to_numpy(dtype=float),
                color="0.45", alpha=0.12, hatch="///", label=f"Label-perm ±SD{from_k_suffix}",
            )

    if np.isfinite(seg_mean).any():
        ax.plot(r_vals, seg_mean, color="C3", lw=1.5, label=f"Greedy seg mean{direct_suffix}")
        from_k_col = f"segregation_greedy_{tag}_mean_from_k"
        if from_k_col in grp.columns:
            ax.plot(
                r_vals, grp[from_k_col].to_numpy(dtype=float), color="C3", lw=1.3, ls="--",
                label=f"Greedy seg mean{from_k_suffix}",
            )
        ax.fill_between(r_vals, seg_lo, seg_hi, color="C3", alpha=0.2, label=f"Greedy seg ±SD{direct_suffix}")
        from_k_lo_col = f"segregation_greedy_{tag}_sd_envelope_lo_from_k"
        if from_k_lo_col in grp.columns:
            ax.fill_between(
                r_vals,
                grp[from_k_lo_col].to_numpy(dtype=float),
                grp[f"segregation_greedy_{tag}_sd_envelope_hi_from_k"].to_numpy(dtype=float),
                color="C3", alpha=0.1, hatch="///", label=f"Greedy seg ±SD{from_k_suffix}",
            )

    ax.axhline(1.0 if fam in G_FAMILIES else 0.0, color="0.5", ls="--", lw=0.8)
    ax.set_xlabel("r (nm)")
    ax.set_ylabel(f"{tag}(r)")
    n_curves = int(meta[f"n_zone_curves_{tag}"]) if f"n_zone_curves_{tag}" in grp.columns else int(meta.get("n_zone_curves", 0))
    ax.set_title(
        f"Pooled monomer vs dimer ({tag}) — set: {set_name}\n"
        f"{int(meta['n_tomograms'])} tomogram(s), {int(meta['n_active_zones'])} zone(s), "
        f"{n_curves} curves"
    )
    ax.set_xlim(0.0, float(r_vals[-1]) if len(r_vals) else DEFAULT_RIPLEY_R_MAX_NM)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out_path = output_dir / f"ripley_{fam}_pooled_observed_vs_null_{set_tag}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Pooled monomer/dimer {tag} figure (set {set_name}) -> {out_path}")
    return out_path


def plot_pooled_monomer_dimer_ripley_visualizations(
    curves_csv: Path | str = POOLED_CURVES_CSV,
    output_dir: Path | str = POOLED_FIGURES_DIR,
    prism_csv: Path | str = POOLED_PRISM_CSV,
    prism_wide_csv: Path | str = POOLED_PRISM_WIDE_CSV,
) -> list[Path]:
    """Build pooled per-set Prism tables and observed-vs-null figures for every family in
    ``ALL_FAMILIES``."""
    curves_csv = Path(curves_csv)
    output_dir = Path(output_dir)
    prism_csv = Path(prism_csv)
    prism_wide_csv = Path(prism_wide_csv)

    if not curves_csv.is_file():
        print(f"No pooled monomer/dimer Ripley CSV at {curves_csv}; skipping pooled outputs.")
        return []

    df = pd.read_csv(curves_csv)
    if df.empty or "tomogram_name" not in df.columns:
        print("Pooled monomer/dimer Ripley CSV missing data; skipping pooled outputs.")
        return []

    prism_long = build_pooled_monomer_dimer_prism_table(df)
    if prism_long.empty:
        print("No pooled monomer/dimer Ripley envelope rows generated.")
        return []

    prism_csv.parent.mkdir(parents=True, exist_ok=True)
    prism_long.to_csv(prism_csv, index=False)
    _prism_long_to_wide(prism_long, id_cols=["set_name", "window_mode"]).to_csv(
        prism_wide_csv, index=False
    )
    print(f"Pooled monomer/dimer Ripley Prism table ({len(prism_long)} rows) -> {prism_csv}")

    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = [prism_csv, prism_wide_csv]

    for set_name, grp in prism_long.groupby("set_name", sort=False):
        grp = grp.sort_values("r_nm")
        for fam in ALL_FAMILIES:
            out_path = _plot_pooled_family_figure(
                grp, fam=fam, set_name=set_name, output_dir=output_dir
            )
            if out_path is not None:
                written.append(out_path)

    return written
