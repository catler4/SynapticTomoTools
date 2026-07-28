"""
3D bivariate Ripley K₁₂ / L₁₂ of monomer vs dimer AuNP positions (no fusion site).

Type-1 foci: monomer AuNP pick coordinates for a zone.
Type-2 partners: dimer AuNP pick coordinates for the same zone.

Window: synaptic_cleft_az_hull (convex hull of presynaptic + postsynaptic AZ surface points),
matching the vesicle fusion-site bivariate Ripley setup.

Control: label permutation — pool all monomer + dimer points, then randomly reassign class
labels while preserving the per-zone monomer and dimer counts (1000 replicates by default).

Greedy segregation — same pooled points and class counts, but relabel by growing a compact
spatial cluster from a random seed. Each replicate randomly chooses whether that compact
class is monomers or dimers (then the remainder gets the other label). Replicate count
always matches the label-permutation count (``n_perm``, default 1000).

MAD tests (Rebola-style max absolute deviation vs 99% CE) are run against label-permutation
and greedy segregation when that null has ≥1000 curves; otherwise they are skipped.
Each MAD is reported for the full r-grid and for the restricted 30–50 nm window.

Per zone, the first three label-permutation and greedy-segregation point sets are also
written as monomer/dimer STAR files under ``STT_results/.../simulated_null_stars/``
(same columns as the input pick STARs).

Pooled output is grouped per tomogram set (curves, individual curves, and MAD summaries).
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from scipy.spatial.distance import cdist

from .alignment_utils import require_alignment_dir
from .aunps import _read_aunp_pick_star_dataframe
from .fusion_point_aunp_position_distance_and_Ripleys_analyses import (
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
    _isotropic_edge_factors_for_foci,
    _percentile_band,
    _prism_sd_envelope_columns,
    _ripley_r_grid,
    build_ripley_window_3d,
    curves_matrix_to_long_dataframe,
    curves_matrix_to_wide_dataframe,
    label_permutation_l12_curves,
    load_monomer_dimer_aunps_for_zone,
    load_synaptic_cleft_active_zone_points,
    mad_result_to_curves_dataframe,
    mad_result_to_summary_row,
    prism_sd_envelope_columns_from_averaged_k12,
    ripley_l12_from_points,
    run_mad_tests_over_r_ranges,
    subset_aunps,
)

WINDOW_MODE = "synaptic_cleft_az_hull"
MONOMER_DIMER_N_PERM = 1000
# Segregation always uses the same replicate count as label permutation (n_perm).
MIN_POINTS_PER_CLASS = 2
# Example null point sets written as STAR files (label-perm + greedy each).
N_SIMULATED_NULL_STAR_EXAMPLES = 3
SIMULATED_STAR_COLS = (
    "faCoordinateX",
    "faCoordinateY",
    "faCoordinateZ",
    "active_zone",
    "postsynapse",
)

POOLED_CURVES_CSV = Path("results/aunps/aunp_monomer_dimer_ripley_l12_curves.csv")
POOLED_INDIVIDUAL_CURVES_CSV = Path(
    "results/aunps/aunp_monomer_dimer_ripley_l12_individual_curves.csv"
)
POOLED_MAD_SUMMARY_CSV = Path("results/aunps/aunp_monomer_dimer_ripley_l12_mad_summary.csv")
POOLED_PRISM_CSV = Path("results/aunps/aunp_monomer_dimer_ripley_l12_prism_pooled.csv")
POOLED_PRISM_WIDE_CSV = Path("results/aunps/aunp_monomer_dimer_ripley_l12_prism_pooled_wide.csv")
POOLED_FIGURES_DIR = Path("results/aunps/figures/aunp_monomer_dimer_ripley_l12_pooled")


def _safe_name(name: str) -> str:
    safe = str(name).strip().replace(" ", "_")
    for ch in '<>:"/\\|?*':
        safe = safe.replace(ch, "_")
    return safe


def _prism_long_to_wide(prism_long: pd.DataFrame, id_cols: Sequence[str]) -> pd.DataFrame:
    if prism_long.empty:
        return prism_long.copy()
    value_cols = [c for c in prism_long.columns if c not in id_cols and c != "r_nm"]
    return prism_long[list(id_cols) + ["r_nm"] + value_cols].copy()


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


# Shared context for parallel greedy-segregation L₁₂ workers.
_SEG_L12_CTX: dict = {}


def _init_greedy_seg_l12_worker(
    pool: np.ndarray,
    r_vals: np.ndarray,
    window,
    n_monomer: int,
    n_dimer: int,
    pairwise_dist: np.ndarray,
    pool_edge_factors: np.ndarray,
) -> None:
    _SEG_L12_CTX["pool"] = pool
    _SEG_L12_CTX["r_vals"] = r_vals
    _SEG_L12_CTX["window"] = window
    _SEG_L12_CTX["n_monomer"] = int(n_monomer)
    _SEG_L12_CTX["n_dimer"] = int(n_dimer)
    _SEG_L12_CTX["pairwise_dist"] = pairwise_dist
    _SEG_L12_CTX["pool_edge_factors"] = pool_edge_factors


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


def _greedy_seg_l12_one_replicate(
    *,
    pool: np.ndarray,
    r_vals: np.ndarray,
    window,
    n_monomer: int,
    n_dimer: int,
    pairwise_dist: np.ndarray,
    pool_edge_factors: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    One greedy-segregation L₁₂ curve: randomly choose monomer or dimer as the compact
    class, grow that class from a random seed, assign the remainder to the other class.
    """
    monomer_mask = _greedy_segregation_monomer_mask(
        pool, n_monomer, n_dimer, pairwise_dist, rng
    )
    return ripley_l12_from_points(
        pool[monomer_mask],
        pool[~monomer_mask],
        r_vals,
        window,
        rng,
        edge_factors=pool_edge_factors[monomer_mask],
    )


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


def _greedy_seg_l12_worker(task: tuple[int, int]) -> tuple[int, np.ndarray]:
    """Run one greedy-segregation L₁₂ curve (rep_id, seed)."""
    rep_id, seed = task
    curve = _greedy_seg_l12_one_replicate(
        pool=_SEG_L12_CTX["pool"],
        r_vals=_SEG_L12_CTX["r_vals"],
        window=_SEG_L12_CTX["window"],
        n_monomer=_SEG_L12_CTX["n_monomer"],
        n_dimer=_SEG_L12_CTX["n_dimer"],
        pairwise_dist=_SEG_L12_CTX["pairwise_dist"],
        pool_edge_factors=_SEG_L12_CTX["pool_edge_factors"],
        rng=np.random.default_rng(int(seed)),
    )
    return int(rep_id), curve


def _greedy_segregation_l12_curves(
    pool: np.ndarray,
    n_monomer: int,
    n_dimer: int,
    r_vals: np.ndarray,
    window,
    rng: np.random.Generator,
    *,
    n_rep: int,
    pool_edge_factors: np.ndarray | None = None,
    seed: int = DEFAULT_ANALYSIS_SEED,
    n_workers: int | None = None,
    pbar: Optional[tqdm] = None,
) -> np.ndarray:
    """
    Greedy segregation L₁₂ curves with a random clustered class per replicate.

    Each replicate independently chooses monomer or dimer as the compact class
    (equal probability), grows that class from a random seed via nearest-neighbor
    expansion to the observed class count, assigns the remaining points to the other
    class, then computes monomer→dimer L₁₂.

    Uses the same edge-factor precompute + process-pool pattern as label permutation.
    """
    pool = np.atleast_2d(np.asarray(pool, dtype=float))
    n_pool = len(pool)
    n_rep_int = int(n_rep)
    n_monomer = int(n_monomer)
    n_dimer = int(n_dimer)
    curves = np.full((n_rep_int, len(r_vals)), np.nan, dtype=float)
    if n_pool == 0 or n_rep_int == 0:
        return curves
    if n_monomer <= 0 or n_dimer <= 0 or n_monomer + n_dimer != n_pool:
        return curves
    if n_monomer > n_pool or n_dimer > n_pool:
        return curves

    pairwise_dist = cdist(pool, pool)
    if pool_edge_factors is None:
        pool_edge_factors = _isotropic_edge_factors_for_foci(pool, r_vals, window.hull, rng)
    else:
        pool_edge_factors = np.asarray(pool_edge_factors, dtype=float)
        if pool_edge_factors.shape != (n_pool, len(r_vals)):
            raise ValueError(
                f"pool_edge_factors shape {pool_edge_factors.shape} != "
                f"expected {(n_pool, len(r_vals))}"
            )

    if n_workers is None:
        n_workers = _default_ripley_perm_workers(n_rep_int)
    n_workers = max(1, min(int(n_workers), n_rep_int))

    if n_workers == 1:
        for rep_id in range(n_rep_int):
            curves[rep_id] = _greedy_seg_l12_one_replicate(
                pool=pool,
                r_vals=r_vals,
                window=window,
                n_monomer=n_monomer,
                n_dimer=n_dimer,
                pairwise_dist=pairwise_dist,
                pool_edge_factors=pool_edge_factors,
                rng=rng,
            )
            if pbar is not None:
                pbar.set_postfix_str(f"greedy seg {rep_id + 1}/{n_rep_int}", refresh=False)
                pbar.update(1)
        return curves

    # Deterministic per-replicate seeds (independent of call-order / worker scheduling).
    tasks = [(rep_id, int(seed) + 17 + rep_id) for rep_id in range(n_rep_int)]
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_greedy_seg_l12_worker,
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
        futures = [executor.submit(_greedy_seg_l12_worker, task) for task in tasks]
        for fut in as_completed(futures):
            rep_id, curve = fut.result()
            curves[rep_id] = curve
            if pbar is not None:
                pbar.set_postfix_str(f"greedy seg {rep_id + 1}/{n_rep_int}", refresh=False)
                pbar.update(1)
    return curves


def _plot_mad_panels(
    *,
    mad_results: list[dict],
    output_path: Path,
    title: str,
    observed_color: str = "C0",
    observed_label: str = "Observed L₁₂",
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
        ax_raw.set_ylabel("L₁₂(r)")
        ax_raw.legend(fontsize=7, loc="best")
        ax_norm.set_xlabel("r (nm)")
        ax_norm.set_ylabel("(L₁₂ − μ_null) / CE half-width")
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
    observed_l12: np.ndarray,
    null_named_curves: list[tuple[str, np.ndarray]],
    write_figures: bool,
    figure_title: str,
) -> tuple[Path, Path]:
    """Run MAD for each null × r-range (≥1000 curves), write summary/curves CSVs and figures."""
    summary_rows: list[dict] = []
    curve_frames: list[pd.DataFrame] = []
    mad_by_range: dict[str, list[dict]] = {label: [] for label, _, _ in MAD_R_RANGES}

    for null_name, null_curves in null_named_curves:
        for mad in run_mad_tests_over_r_ranges(
            observed_l12,
            null_curves,
            r_vals,
            null_name=null_name,
            min_null_curves=MAD_MIN_NULL_CURVES,
        ):
            mad_by_range.setdefault(str(mad["r_range"]), []).append(mad)
            summary_rows.append(
                mad_result_to_summary_row(
                    mad,
                    extra_cols={"active_zone_name": zone_name},
                )
            )
            curves_df = mad_result_to_curves_dataframe(mad, r_vals, observed=observed_l12)
            curves_df.insert(0, "active_zone_name", zone_name)
            curve_frames.append(curves_df)

    summary_path = out_dir / "ripley_l12_mad_summary.csv"
    curves_path = out_dir / "ripley_l12_mad_curves.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    pd.concat(curve_frames, ignore_index=True).to_csv(curves_path, index=False)

    if write_figures and figures_dir is not None:
        for r_range, mad_results in mad_by_range.items():
            if not mad_results:
                continue
            suffix = "" if r_range == "full" else f"_{r_range.replace('-', '_')}"
            _plot_mad_panels(
                mad_results=mad_results,
                output_path=figures_dir / f"ripley_l12_mad_vs_nulls{suffix}.png",
                title=f"{figure_title} | r-range={r_range}",
            )
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
    observed_l12: np.ndarray,
    perm_curves: np.ndarray,
    seg_greedy_curves: np.ndarray,
    n_monomer: int,
    n_dimer: int,
    window_volume_nm3: float,
) -> pd.DataFrame:
    """Long table of every individual L₁₂ curve (observed + all control replicates)."""
    extras = {
        "active_zone_name": zone_name,
        "window_mode": WINDOW_MODE,
        "n_monomer": int(n_monomer),
        "n_dimer": int(n_dimer),
        "window_volume_nm3": float(window_volume_nm3),
    }
    frames = [
        curves_matrix_to_long_dataframe(
            np.atleast_2d(observed_l12),
            r_vals,
            curve_type="observed",
            extra_cols=extras,
        ),
        curves_matrix_to_long_dataframe(
            perm_curves,
            r_vals,
            curve_type="label_permutation",
            extra_cols=extras,
        ),
        curves_matrix_to_long_dataframe(
            seg_greedy_curves,
            r_vals,
            curve_type="segregation_greedy",
            extra_cols=extras,
        ),
    ]
    nonempty = [f for f in frames if not f.empty]
    if not nonempty:
        return pd.DataFrame()
    return pd.concat(nonempty, ignore_index=True)


def build_monomer_dimer_prism_table(
    *,
    zone_name: str,
    r_vals: np.ndarray,
    observed_l12: np.ndarray,
    perm_curves: np.ndarray,
    seg_greedy_curves: np.ndarray,
    n_monomer: int,
    n_dimer: int,
    n_perm: int,
    n_segregation: int,
    window_volume_nm3: float,
) -> pd.DataFrame:
    """Per-zone Prism table: observed L₁₂ plus label-permutation and segregation controls."""
    perm_lo, perm_mean, perm_hi = _percentile_band(perm_curves)
    perm_sd = _prism_sd_envelope_columns(perm_curves, r_vals, prefix="control_L12")
    seg = _segregation_band_columns(
        seg_greedy_curves, r_vals, prefix="segregation_greedy_L12"
    )
    n_r = len(r_vals)
    if len(perm_lo) != n_r:
        nan = np.full(n_r, np.nan)
        perm_lo = perm_mean = perm_hi = nan

    rows: list[dict] = []
    for i, r_nm in enumerate(r_vals):
        rows.append(
            {
                "active_zone_name": zone_name,
                "window_mode": WINDOW_MODE,
                "r_nm": float(r_nm),
                "observed_L12": float(observed_l12[i]),
                "control_L12_mean": float(perm_sd["control_L12_mean"][i]),
                "control_L12_sd": float(perm_sd["control_L12_sd"][i]),
                "control_L12_sd_envelope_lo": float(perm_sd["control_L12_sd_envelope_lo"][i]),
                "control_L12_sd_envelope_hi": float(perm_sd["control_L12_sd_envelope_hi"][i]),
                "control_L12_sem": float(perm_sd["control_L12_sem"][i]),
                "control_L12_sem_envelope_lo": float(perm_sd["control_L12_sem_envelope_lo"][i]),
                "control_L12_sem_envelope_hi": float(perm_sd["control_L12_sem_envelope_hi"][i]),
                "control_L12_envelope_lo": float(perm_lo[i]),
                "control_L12_envelope_hi": float(perm_hi[i]),
                "segregation_greedy_L12_mean": float(seg["segregation_greedy_L12_mean"][i]),
                "segregation_greedy_L12_sd": float(seg["segregation_greedy_L12_sd"][i]),
                "segregation_greedy_L12_sd_envelope_lo": float(
                    seg["segregation_greedy_L12_sd_envelope_lo"][i]
                ),
                "segregation_greedy_L12_sd_envelope_hi": float(
                    seg["segregation_greedy_L12_sd_envelope_hi"][i]
                ),
                "segregation_greedy_L12_sem": float(seg["segregation_greedy_L12_sem"][i]),
                "segregation_greedy_L12_sem_envelope_lo": float(
                    seg["segregation_greedy_L12_sem_envelope_lo"][i]
                ),
                "segregation_greedy_L12_sem_envelope_hi": float(
                    seg["segregation_greedy_L12_sem_envelope_hi"][i]
                ),
                "segregation_greedy_L12_envelope_lo": float(
                    seg["segregation_greedy_L12_envelope_lo"][i]
                ),
                "segregation_greedy_L12_envelope_hi": float(
                    seg["segregation_greedy_L12_envelope_hi"][i]
                ),
                "n_monomer": int(n_monomer),
                "n_dimer": int(n_dimer),
                "n_permutations": int(n_perm),
                "n_segregation_replicates": int(n_segregation),
                "envelope_percentile_lo": float(RIPLEY_PERCENTILE_LO),
                "envelope_percentile_hi": float(RIPLEY_PERCENTILE_HI),
                "window_volume_nm3": float(window_volume_nm3),
            }
        )
    return pd.DataFrame(rows)


def build_pooled_monomer_dimer_prism_table(df: pd.DataFrame) -> pd.DataFrame:
    """Pooled mean ± SD of observed and control L₁₂ across zones, per tomogram set.

    Reports both L-space mean±SD/SEM (``*_mean``, ``*_sd_*``) and K-space mean±SD/SEM
    mapped to L (``*_mean_from_k``, ``*_sd_*_from_k``, ``*_sem_*_from_k``).
    """
    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    if "set_name" not in df.columns:
        df["set_name"] = ""
    df["set_name"] = df["set_name"].fillna("").astype(str)

    rows: list[dict] = []
    for set_name, sub in df.groupby("set_name", sort=False):
        r_vals, obs_curves = _extract_curves_matrix(sub, "l12")
        if len(obs_curves) == 0:
            continue
        _, ctrl_curves = _extract_curves_matrix(sub, "perm_l12_mean")
        _, seg_curves = _extract_curves_matrix(sub, "seg_greedy_l12_mean")

        obs_sd = _prism_sd_envelope_columns(obs_curves, r_vals, prefix="observed_L12")
        obs_from_k = prism_sd_envelope_columns_from_averaged_k12(
            obs_curves, r_vals, prefix="observed_L12"
        )
        if len(ctrl_curves):
            ctrl_sd = _prism_sd_envelope_columns(ctrl_curves, r_vals, prefix="control_L12")
            ctrl_from_k = prism_sd_envelope_columns_from_averaged_k12(
                ctrl_curves, r_vals, prefix="control_L12"
            )
        else:
            ctrl_sd = _prism_sd_envelope_columns(
                np.empty((0, len(r_vals))), r_vals, prefix="control_L12"
            )
            ctrl_from_k = prism_sd_envelope_columns_from_averaged_k12(
                np.empty((0, len(r_vals))), r_vals, prefix="control_L12"
            )
        if len(seg_curves):
            seg_sd = _prism_sd_envelope_columns(
                seg_curves, r_vals, prefix="segregation_greedy_L12"
            )
            seg_from_k = prism_sd_envelope_columns_from_averaged_k12(
                seg_curves, r_vals, prefix="segregation_greedy_L12"
            )
        else:
            seg_sd = _prism_sd_envelope_columns(
                np.empty((0, len(r_vals))), r_vals, prefix="segregation_greedy_L12"
            )
            seg_from_k = prism_sd_envelope_columns_from_averaged_k12(
                np.empty((0, len(r_vals))), r_vals, prefix="segregation_greedy_L12"
            )

        n_tomograms = int(sub["tomogram_name"].nunique()) if "tomogram_name" in sub.columns else 0
        n_zones = int(
            sub[["tomogram_name", "alignment_dir", "active_zone_name"]].drop_duplicates().shape[0]
        )

        for i, r_nm in enumerate(r_vals):
            rows.append(
                {
                    "set_name": set_name,
                    "window_mode": WINDOW_MODE,
                    "r_nm": float(r_nm),
                    "observed_L12_mean": float(obs_sd["observed_L12_mean"][i]),
                    "observed_L12_mean_from_k": float(obs_from_k["observed_L12_mean_from_k"][i]),
                    "observed_L12_sd": float(obs_sd["observed_L12_sd"][i]),
                    "observed_L12_sd_envelope_lo": float(obs_sd["observed_L12_sd_envelope_lo"][i]),
                    "observed_L12_sd_envelope_hi": float(obs_sd["observed_L12_sd_envelope_hi"][i]),
                    "observed_L12_sd_from_k": float(obs_from_k["observed_L12_sd_from_k"][i]),
                    "observed_L12_sd_envelope_lo_from_k": float(
                        obs_from_k["observed_L12_sd_envelope_lo_from_k"][i]
                    ),
                    "observed_L12_sd_envelope_hi_from_k": float(
                        obs_from_k["observed_L12_sd_envelope_hi_from_k"][i]
                    ),
                    "observed_L12_sem": float(obs_sd["observed_L12_sem"][i]),
                    "observed_L12_sem_envelope_lo": float(obs_sd["observed_L12_sem_envelope_lo"][i]),
                    "observed_L12_sem_envelope_hi": float(obs_sd["observed_L12_sem_envelope_hi"][i]),
                    "observed_L12_sem_from_k": float(obs_from_k["observed_L12_sem_from_k"][i]),
                    "observed_L12_sem_envelope_lo_from_k": float(
                        obs_from_k["observed_L12_sem_envelope_lo_from_k"][i]
                    ),
                    "observed_L12_sem_envelope_hi_from_k": float(
                        obs_from_k["observed_L12_sem_envelope_hi_from_k"][i]
                    ),
                    "control_L12_mean": float(ctrl_sd["control_L12_mean"][i]),
                    "control_L12_mean_from_k": float(ctrl_from_k["control_L12_mean_from_k"][i]),
                    "control_L12_sd": float(ctrl_sd["control_L12_sd"][i]),
                    "control_L12_sd_envelope_lo": float(ctrl_sd["control_L12_sd_envelope_lo"][i]),
                    "control_L12_sd_envelope_hi": float(ctrl_sd["control_L12_sd_envelope_hi"][i]),
                    "control_L12_sd_from_k": float(ctrl_from_k["control_L12_sd_from_k"][i]),
                    "control_L12_sd_envelope_lo_from_k": float(
                        ctrl_from_k["control_L12_sd_envelope_lo_from_k"][i]
                    ),
                    "control_L12_sd_envelope_hi_from_k": float(
                        ctrl_from_k["control_L12_sd_envelope_hi_from_k"][i]
                    ),
                    "control_L12_sem": float(ctrl_sd["control_L12_sem"][i]),
                    "control_L12_sem_envelope_lo": float(ctrl_sd["control_L12_sem_envelope_lo"][i]),
                    "control_L12_sem_envelope_hi": float(ctrl_sd["control_L12_sem_envelope_hi"][i]),
                    "control_L12_sem_from_k": float(ctrl_from_k["control_L12_sem_from_k"][i]),
                    "control_L12_sem_envelope_lo_from_k": float(
                        ctrl_from_k["control_L12_sem_envelope_lo_from_k"][i]
                    ),
                    "control_L12_sem_envelope_hi_from_k": float(
                        ctrl_from_k["control_L12_sem_envelope_hi_from_k"][i]
                    ),
                    "segregation_greedy_L12_mean": float(seg_sd["segregation_greedy_L12_mean"][i]),
                    "segregation_greedy_L12_mean_from_k": float(
                        seg_from_k["segregation_greedy_L12_mean_from_k"][i]
                    ),
                    "segregation_greedy_L12_sd": float(seg_sd["segregation_greedy_L12_sd"][i]),
                    "segregation_greedy_L12_sd_envelope_lo": float(
                        seg_sd["segregation_greedy_L12_sd_envelope_lo"][i]
                    ),
                    "segregation_greedy_L12_sd_envelope_hi": float(
                        seg_sd["segregation_greedy_L12_sd_envelope_hi"][i]
                    ),
                    "segregation_greedy_L12_sd_from_k": float(
                        seg_from_k["segregation_greedy_L12_sd_from_k"][i]
                    ),
                    "segregation_greedy_L12_sd_envelope_lo_from_k": float(
                        seg_from_k["segregation_greedy_L12_sd_envelope_lo_from_k"][i]
                    ),
                    "segregation_greedy_L12_sd_envelope_hi_from_k": float(
                        seg_from_k["segregation_greedy_L12_sd_envelope_hi_from_k"][i]
                    ),
                    "segregation_greedy_L12_sem": float(seg_sd["segregation_greedy_L12_sem"][i]),
                    "segregation_greedy_L12_sem_envelope_lo": float(
                        seg_sd["segregation_greedy_L12_sem_envelope_lo"][i]
                    ),
                    "segregation_greedy_L12_sem_envelope_hi": float(
                        seg_sd["segregation_greedy_L12_sem_envelope_hi"][i]
                    ),
                    "segregation_greedy_L12_sem_from_k": float(
                        seg_from_k["segregation_greedy_L12_sem_from_k"][i]
                    ),
                    "segregation_greedy_L12_sem_envelope_lo_from_k": float(
                        seg_from_k["segregation_greedy_L12_sem_envelope_lo_from_k"][i]
                    ),
                    "segregation_greedy_L12_sem_envelope_hi_from_k": float(
                        seg_from_k["segregation_greedy_L12_sem_envelope_hi_from_k"][i]
                    ),
                    "n_zone_curves": int(len(obs_curves)),
                    "n_tomograms": n_tomograms,
                    "n_active_zones": n_zones,
                }
            )
    return pd.DataFrame(rows)


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
    """Observed monomer→dimer L₁₂ with label-permutation and greedy-segregation controls.

    Greedy segregation always uses the same replicate count as label permutation (``n_perm``).
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
    n_monomer = len(monomer_coords)
    n_dimer = len(dimer_coords)
    if n_monomer < MIN_POINTS_PER_CLASS or n_dimer < MIN_POINTS_PER_CLASS:
        print(
            f"  Skipping monomer/dimer Ripley for {zone_name}: "
            f"too few points (monomer={n_monomer}, dimer={n_dimer}; "
            f"need >= {MIN_POINTS_PER_CLASS} each)"
        )
        return None

    try:
        cleft_coords = load_synaptic_cleft_active_zone_points(
            tomogram_path, alignment_dir, zone_name
        )
        window = build_ripley_window_3d(cleft_coords, mode=WINDOW_MODE)
    except Exception as exc:
        print(f"  Skipping monomer/dimer Ripley for {zone_name}: {exc}")
        return None

    r_vals = _ripley_r_grid(r_max_nm, r_step_nm)
    pool = np.vstack([np.atleast_2d(monomer_coords), np.atleast_2d(dimer_coords)])
    rng = np.random.default_rng(seed)
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
        # Edge factors depend only on fixed pooled coordinates — estimate once and reuse
        # for observed + greedy segregation (label-perm estimates its own pool factors).
        edge_rng = np.random.default_rng(int(seed) + 7919)
        pool_edge_factors = _isotropic_edge_factors_for_foci(
            pool, r_vals, window.hull, edge_rng
        )
        pbar.set_postfix_str("observed", refresh=False)
        observed_l12 = ripley_l12_from_points(
            monomer_coords,
            dimer_coords,
            r_vals,
            window,
            rng,
            edge_factors=pool_edge_factors[:n_monomer],
        )
        pbar.update(1)
        perm_curves = label_permutation_l12_curves(
            monomer_coords,
            dimer_coords,
            r_vals,
            window,
            n_perm=n_perm_int,
            seed=seed,
            rng=rng,
            pbar=pbar if n_perm_int > 0 else None,
        )
        seg_rng = np.random.default_rng(int(seed) + 1_000_000)
        seg_greedy_curves = _greedy_segregation_l12_curves(
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
    finally:
        pbar.close()
    _, perm_mean, _ = _percentile_band(perm_curves)
    if len(perm_mean) != len(r_vals):
        perm_mean = np.full(len(r_vals), np.nan)
    _, seg_greedy_mean, _ = _percentile_band(seg_greedy_curves)
    if len(seg_greedy_mean) != len(r_vals):
        seg_greedy_mean = np.full(len(r_vals), np.nan)

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

    simulated_stars_dir: Path | None = None
    try:
        pool_df = _load_pool_star_dataframe_for_export(
            tomogram_path,
            alignment_dir,
            int(active_zone_index),
            monomer_star_pattern=monomer_star_pattern,
            dimer_star_pattern=dimer_star_pattern,
        )
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

    curves_df = pd.DataFrame(
        {
            "active_zone_name": zone_name,
            "active_zone_index": int(active_zone_index),
            "window_mode": WINDOW_MODE,
            "r_nm": r_vals,
            "l12": observed_l12,
            "perm_l12_mean": perm_mean,
            "seg_greedy_l12_mean": seg_greedy_mean,
            "n_monomer": n_monomer,
            "n_dimer": n_dimer,
            "n_permutations": int(n_perm_int),
            "n_segregation_replicates": int(n_segregation_int),
            "window_volume_nm3": float(window.volume_nm3),
        }
    )
    curves_path = out_dir / "ripley_l12_curves.csv"
    curves_df.to_csv(curves_path, index=False)

    individual_df = build_monomer_dimer_individual_curves_table(
        zone_name=zone_name,
        r_vals=r_vals,
        observed_l12=observed_l12,
        perm_curves=perm_curves,
        seg_greedy_curves=seg_greedy_curves,
        n_monomer=n_monomer,
        n_dimer=n_dimer,
        window_volume_nm3=float(window.volume_nm3),
    )
    individual_path = out_dir / "ripley_l12_individual_curves.csv"
    individual_df.to_csv(individual_path, index=False)
    # Prism-friendly wide tables (one file per curve family).
    curves_matrix_to_wide_dataframe(
        np.atleast_2d(observed_l12), r_vals, curve_type="observed"
    ).to_csv(out_dir / "ripley_l12_individual_observed_wide.csv", index=False)
    curves_matrix_to_wide_dataframe(
        perm_curves, r_vals, curve_type="label_permutation"
    ).to_csv(out_dir / "ripley_l12_individual_label_permutation_wide.csv", index=False)
    curves_matrix_to_wide_dataframe(
        seg_greedy_curves, r_vals, curve_type="segregation_greedy"
    ).to_csv(out_dir / "ripley_l12_individual_segregation_greedy_wide.csv", index=False)

    prism_df = build_monomer_dimer_prism_table(
        zone_name=zone_name,
        r_vals=r_vals,
        observed_l12=observed_l12,
        perm_curves=perm_curves,
        seg_greedy_curves=seg_greedy_curves,
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
        perm_lo, perm_band_mean, perm_hi = _percentile_band(perm_curves)
        seg_lo, seg_band_mean, seg_hi = _percentile_band(seg_greedy_curves)
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        ax.plot(r_vals, observed_l12, color="C0", lw=2, label="Observed monomer→dimer L₁₂")
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
            ax.plot(
                r_vals,
                seg_band_mean,
                color="C3",
                lw=1.5,
                label="Greedy segregation mean",
            )
            ax.fill_between(
                r_vals,
                seg_lo,
                seg_hi,
                color="C3",
                alpha=0.2,
                label=f"Greedy seg {RIPLEY_PERCENTILE_LO:g}–{RIPLEY_PERCENTILE_HI:g}%",
            )
        ax.axhline(0.0, color="0.5", ls="--", lw=0.8)
        ax.set_xlabel("r (nm)")
        ax.set_ylabel("Ripley L₁₂(r) = (3K₁₂/4π)^(1/3) − r")
        ax.set_title(
            f"{tomogram_name} | {zone_name}\n"
            f"monomer ({n_monomer}) vs dimer ({n_dimer}) | "
            f"{int(n_perm_int)} label perms, {int(n_segregation_int)} greedy seg reps"
        )
        ax.set_xlim(0.0, float(r_vals[-1]) if len(r_vals) else r_max_nm)
        ax.legend(loc="best", fontsize=7)
        fig.tight_layout()
        fig.savefig(
            figures_dir / "ripley_l12_observed_vs_controls.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)

    mad_summary_path, mad_curves_path = _write_mad_outputs(
        out_dir=out_dir,
        figures_dir=figures_dir if write_figures else None,
        zone_name=zone_name,
        r_vals=r_vals,
        observed_l12=observed_l12,
        null_named_curves=[
            ("label_permutation", perm_curves),
            ("segregation_greedy", seg_greedy_curves),
        ],
        write_figures=write_figures,
        figure_title=(
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
        "control_label_permutation": "label_permutation_preserving_class_counts",
        "control_segregation": "greedy_nearest_neighbor_cluster_random_class",
        "ripley_edge_correction": "isotropic_3d_mc",
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
        f"  Monomer/dimer Ripley L₁₂ ({zone_name}): "
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


def plot_pooled_monomer_dimer_ripley_visualizations(
    curves_csv: Path | str = POOLED_CURVES_CSV,
    output_dir: Path | str = POOLED_FIGURES_DIR,
    prism_csv: Path | str = POOLED_PRISM_CSV,
    prism_wide_csv: Path | str = POOLED_PRISM_WIDE_CSV,
) -> list[Path]:
    """Build pooled per-set Prism tables and observed-vs-null L₁₂ figures."""
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
        r_vals = grp["r_nm"].to_numpy(dtype=float)
        obs_mean = grp["observed_L12_mean"].to_numpy(dtype=float)
        obs_lo = grp["observed_L12_sd_envelope_lo"].to_numpy(dtype=float)
        obs_hi = grp["observed_L12_sd_envelope_hi"].to_numpy(dtype=float)
        ctrl_mean = grp["control_L12_mean"].to_numpy(dtype=float)
        ctrl_lo = grp["control_L12_sd_envelope_lo"].to_numpy(dtype=float)
        ctrl_hi = grp["control_L12_sd_envelope_hi"].to_numpy(dtype=float)
        seg_mean = grp.get(
            "segregation_greedy_L12_mean", pd.Series(np.nan, index=grp.index)
        ).to_numpy(dtype=float)
        seg_lo = grp.get(
            "segregation_greedy_L12_sd_envelope_lo", pd.Series(np.nan, index=grp.index)
        ).to_numpy(dtype=float)
        seg_hi = grp.get(
            "segregation_greedy_L12_sd_envelope_hi", pd.Series(np.nan, index=grp.index)
        ).to_numpy(dtype=float)
        meta = grp.iloc[0]

        set_tag = _safe_name(str(set_name)) or "unspecified"
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        ax.plot(r_vals, obs_mean, color="C0", lw=2, label="Observed mean (of L)")
        if "observed_L12_mean_from_k" in grp.columns:
            ax.plot(
                r_vals,
                grp["observed_L12_mean_from_k"].to_numpy(dtype=float),
                color="C0",
                lw=1.5,
                ls="--",
                label="Observed mean (K→L)",
            )
        ax.fill_between(r_vals, obs_lo, obs_hi, color="C0", alpha=0.25, label="Observed ±SD (of L)")
        if "observed_L12_sd_envelope_lo_from_k" in grp.columns:
            ax.fill_between(
                r_vals,
                grp["observed_L12_sd_envelope_lo_from_k"].to_numpy(dtype=float),
                grp["observed_L12_sd_envelope_hi_from_k"].to_numpy(dtype=float),
                color="C0",
                alpha=0.12,
                hatch="///",
                label="Observed ±SD (K→L)",
            )
        ax.plot(r_vals, ctrl_mean, color="0.45", lw=1.5, label="Label-perm mean (of L)")
        if "control_L12_mean_from_k" in grp.columns:
            ax.plot(
                r_vals,
                grp["control_L12_mean_from_k"].to_numpy(dtype=float),
                color="0.45",
                lw=1.3,
                ls="--",
                label="Label-perm mean (K→L)",
            )
        ax.fill_between(r_vals, ctrl_lo, ctrl_hi, color="0.7", alpha=0.4, label="Label-perm ±SD (of L)")
        if "control_L12_sd_envelope_lo_from_k" in grp.columns:
            ax.fill_between(
                r_vals,
                grp["control_L12_sd_envelope_lo_from_k"].to_numpy(dtype=float),
                grp["control_L12_sd_envelope_hi_from_k"].to_numpy(dtype=float),
                color="0.45",
                alpha=0.12,
                hatch="///",
                label="Label-perm ±SD (K→L)",
            )
        if np.isfinite(seg_mean).any():
            ax.plot(r_vals, seg_mean, color="C3", lw=1.5, label="Greedy seg mean (of L)")
            if "segregation_greedy_L12_mean_from_k" in grp.columns:
                ax.plot(
                    r_vals,
                    grp["segregation_greedy_L12_mean_from_k"].to_numpy(dtype=float),
                    color="C3",
                    lw=1.3,
                    ls="--",
                    label="Greedy seg mean (K→L)",
                )
            ax.fill_between(
                r_vals, seg_lo, seg_hi, color="C3", alpha=0.2, label="Greedy seg ±SD (of L)"
            )
            if "segregation_greedy_L12_sd_envelope_lo_from_k" in grp.columns:
                ax.fill_between(
                    r_vals,
                    grp["segregation_greedy_L12_sd_envelope_lo_from_k"].to_numpy(dtype=float),
                    grp["segregation_greedy_L12_sd_envelope_hi_from_k"].to_numpy(dtype=float),
                    color="C3",
                    alpha=0.1,
                    hatch="///",
                    label="Greedy seg ±SD (K→L)",
                )
        ax.axhline(0.0, color="0.5", ls="--", lw=0.8)
        ax.set_xlabel("r (nm)")
        ax.set_ylabel("Ripley L₁₂(r) = (3K₁₂/4π)^(1/3) − r")
        ax.set_title(
            f"Pooled monomer vs dimer — set: {set_name}\n"
            f"{int(meta['n_tomograms'])} tomogram(s), {int(meta['n_active_zones'])} zone(s), "
            f"{int(meta['n_zone_curves'])} curves"
        )
        ax.set_xlim(0.0, float(r_vals[-1]) if len(r_vals) else DEFAULT_RIPLEY_R_MAX_NM)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        out_path = output_dir / f"ripley_l12_pooled_observed_vs_null_{set_tag}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Pooled monomer/dimer Ripley figure (set {set_name}) -> {out_path}")
        written.append(out_path)

    return written
