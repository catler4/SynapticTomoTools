"""
3D bivariate Ripley K₁₂ / L₁₂ of monomer vs dimer AuNP positions (no fusion site).

Type-1 foci: monomer AuNP pick coordinates for a zone.
Type-2 partners: dimer AuNP pick coordinates for the same zone.

Window: synaptic_cleft_az_hull (convex hull of presynaptic + postsynaptic AZ surface points),
matching the vesicle fusion-site bivariate Ripley setup.

Control: label permutation — pool all monomer + dimer points, then randomly reassign class
labels while preserving the per-zone monomer and dimer counts (1000 replicates by default).

Pooled output is grouped per tomogram set.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .alignment_utils import require_alignment_dir
from .fusion_point_aunp_position_distance_and_Ripleys_analyses import (
    DEFAULT_ANALYSIS_SEED,
    DEFAULT_RIPLEY_R_MAX_NM,
    DEFAULT_RIPLEY_R_STEP_NM,
    RIPLEY_PERCENTILE_HI,
    RIPLEY_PERCENTILE_LO,
    _percentile_band,
    _prism_sd_envelope_columns,
    _ripley_r_grid,
    build_ripley_window_3d,
    load_monomer_dimer_aunps_for_zone,
    load_synaptic_cleft_active_zone_points,
    ripley_l12_from_points,
    subset_aunps,
)

WINDOW_MODE = "synaptic_cleft_az_hull"
MONOMER_DIMER_N_PERM = 1000
MIN_POINTS_PER_CLASS = 2

POOLED_CURVES_CSV = Path("results/aunps/aunp_monomer_dimer_ripley_l12_curves.csv")
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


def _label_permutation_l12_curves(
    monomer_coords: np.ndarray,
    dimer_coords: np.ndarray,
    r_vals: np.ndarray,
    window,
    rng: np.random.Generator,
    *,
    n_perm: int,
) -> np.ndarray:
    """
    Label-permutation null L₁₂ curves.

    Pool all monomer + dimer points, then for each replicate randomly relabel exactly
    ``n_monomer`` points as class 1 (monomer) and the rest as class 2 (dimer), preserving
    the per-zone class counts. Returns an ``(n_perm, len(r_vals))`` array.
    """
    pool = np.vstack([np.atleast_2d(monomer_coords), np.atleast_2d(dimer_coords)])
    n_pool = len(pool)
    n_monomer = len(monomer_coords)
    curves = np.full((int(n_perm), len(r_vals)), np.nan, dtype=float)
    if n_pool == 0 or n_monomer == 0 or n_monomer >= n_pool:
        return curves

    for perm_id in range(int(n_perm)):
        class1_idx = rng.choice(n_pool, n_monomer, replace=False)
        mask = np.zeros(n_pool, dtype=bool)
        mask[class1_idx] = True
        curves[perm_id] = ripley_l12_from_points(pool[mask], pool[~mask], r_vals, window, rng)
    return curves


def build_monomer_dimer_prism_table(
    *,
    zone_name: str,
    r_vals: np.ndarray,
    observed_l12: np.ndarray,
    perm_curves: np.ndarray,
    n_monomer: int,
    n_dimer: int,
    n_perm: int,
    window_volume_nm3: float,
) -> pd.DataFrame:
    """Per-zone Prism table: observed L₁₂ plus label-permutation control envelope."""
    perm_lo, perm_mean, perm_hi = _percentile_band(perm_curves)
    perm_sd = _prism_sd_envelope_columns(perm_curves, r_vals, prefix="control_L12")
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
                "control_L12_envelope_lo": float(perm_lo[i]),
                "control_L12_envelope_hi": float(perm_hi[i]),
                "n_monomer": int(n_monomer),
                "n_dimer": int(n_dimer),
                "n_permutations": int(n_perm),
                "envelope_percentile_lo": float(RIPLEY_PERCENTILE_LO),
                "envelope_percentile_hi": float(RIPLEY_PERCENTILE_HI),
                "window_volume_nm3": float(window_volume_nm3),
            }
        )
    return pd.DataFrame(rows)


def build_pooled_monomer_dimer_prism_table(df: pd.DataFrame) -> pd.DataFrame:
    """Pooled mean ± SD of observed and control L₁₂ across zones, per tomogram set."""
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

        obs_sd = _prism_sd_envelope_columns(obs_curves, r_vals, prefix="observed_L12")
        if len(ctrl_curves):
            ctrl_sd = _prism_sd_envelope_columns(ctrl_curves, r_vals, prefix="control_L12")
        else:
            ctrl_sd = _prism_sd_envelope_columns(
                np.empty((0, len(r_vals))), r_vals, prefix="control_L12"
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
                    "observed_L12_sd": float(obs_sd["observed_L12_sd"][i]),
                    "observed_L12_sd_envelope_lo": float(obs_sd["observed_L12_sd_envelope_lo"][i]),
                    "observed_L12_sd_envelope_hi": float(obs_sd["observed_L12_sd_envelope_hi"][i]),
                    "control_L12_mean": float(ctrl_sd["control_L12_mean"][i]),
                    "control_L12_sd": float(ctrl_sd["control_L12_sd"][i]),
                    "control_L12_sd_envelope_lo": float(ctrl_sd["control_L12_sd_envelope_lo"][i]),
                    "control_L12_sd_envelope_hi": float(ctrl_sd["control_L12_sd_envelope_hi"][i]),
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
    """Observed monomer→dimer L₁₂ with label-permutation control for one active zone."""
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
    rng = np.random.default_rng(seed)
    observed_l12 = ripley_l12_from_points(monomer_coords, dimer_coords, r_vals, window, rng)
    perm_curves = _label_permutation_l12_curves(
        monomer_coords, dimer_coords, r_vals, window, rng, n_perm=n_perm
    )
    _, perm_mean, _ = _percentile_band(perm_curves)
    if len(perm_mean) != len(r_vals):
        perm_mean = np.full(len(r_vals), np.nan)

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

    curves_df = pd.DataFrame(
        {
            "active_zone_name": zone_name,
            "active_zone_index": int(active_zone_index),
            "window_mode": WINDOW_MODE,
            "r_nm": r_vals,
            "l12": observed_l12,
            "perm_l12_mean": perm_mean,
            "n_monomer": n_monomer,
            "n_dimer": n_dimer,
            "n_permutations": int(n_perm),
            "window_volume_nm3": float(window.volume_nm3),
        }
    )
    curves_path = out_dir / "ripley_l12_curves.csv"
    curves_df.to_csv(curves_path, index=False)

    prism_df = build_monomer_dimer_prism_table(
        zone_name=zone_name,
        r_vals=r_vals,
        observed_l12=observed_l12,
        perm_curves=perm_curves,
        n_monomer=n_monomer,
        n_dimer=n_dimer,
        n_perm=n_perm,
        window_volume_nm3=float(window.volume_nm3),
    )
    prism_path = out_dir / "ripley_l12_prism.csv"
    prism_df.to_csv(prism_path, index=False)
    _prism_long_to_wide(prism_df, id_cols=["active_zone_name", "window_mode"]).to_csv(
        out_dir / "ripley_l12_prism_wide.csv", index=False
    )

    if write_figures:
        perm_lo, perm_band_mean, perm_hi = _percentile_band(perm_curves)
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
                label=f"Null {RIPLEY_PERCENTILE_LO:g}–{RIPLEY_PERCENTILE_HI:g}%",
            )
        ax.axhline(0.0, color="0.5", ls="--", lw=0.8)
        ax.set_xlabel("r (nm)")
        ax.set_ylabel("Ripley L₁₂(r) = (3K₁₂/4π)^(1/3) − r")
        ax.set_title(
            f"{tomogram_name} | {zone_name}\n"
            f"monomer ({n_monomer}) vs dimer ({n_dimer}) | {int(n_perm)} label perms"
        )
        ax.set_xlim(0.0, float(r_vals[-1]) if len(r_vals) else r_max_nm)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(figures_dir / "ripley_l12_observed_vs_label_permutation.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    meta = {
        "tomogram_name": tomogram_name,
        "alignment_dir": alignment_dir,
        "active_zone_name": zone_name,
        "active_zone_index": int(active_zone_index),
        "window_mode": WINDOW_MODE,
        "n_monomer": int(n_monomer),
        "n_dimer": int(n_dimer),
        "n_permutations": int(n_perm),
        "window_volume_nm3": float(window.volume_nm3),
        "control": "label_permutation_preserving_class_counts",
        "ripley_edge_correction": "isotropic_3d_mc",
        "seed": int(seed),
    }
    with open(out_dir / "run_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(
        f"  Monomer/dimer Ripley L₁₂ ({zone_name}): "
        f"{n_monomer} monomer, {n_dimer} dimer, {int(n_perm)} perms -> {out_dir}"
    )
    return {
        "curves_path": curves_path,
        "prism_path": prism_path,
        "output_dir": out_dir,
    }


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
) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    """Run monomer vs dimer Ripley for all mapped active zones in one tomogram."""
    from .activezone import load_active_zone_mapping

    tomogram_path = Path(tomogram_path)
    alignment_dir = require_alignment_dir(alignment_dir)
    az_mapping = load_active_zone_mapping(tomogram_path, alignment_dir) or {}
    if not az_mapping:
        print("No active zone mapping; skipping monomer/dimer Ripley analyses")
        return [], []

    az_mapping = {int(k): v for k, v in az_mapping.items()}
    indices = list(active_zone_indices) if active_zone_indices is not None else sorted(az_mapping)

    curve_frames: list[pd.DataFrame] = []
    prism_frames: list[pd.DataFrame] = []

    for az_idx in indices:
        if az_idx not in az_mapping:
            print(f"  Active zone index {az_idx} not in mapping, skipping monomer/dimer Ripley")
            continue
        zone_name = az_mapping[az_idx]
        result = run_monomer_dimer_ripley_for_zone(
            tomogram_path,
            alignment_dir,
            zone_name,
            int(az_idx),
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
        prism_path = result["prism_path"]
        if curves_path.is_file():
            curve_frames.append(pd.read_csv(curves_path))
        if prism_path.is_file():
            prism_frames.append(pd.read_csv(prism_path))

    return curve_frames, prism_frames


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
        meta = grp.iloc[0]

        set_tag = _safe_name(str(set_name)) or "unspecified"
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        ax.plot(r_vals, obs_mean, color="C0", lw=2, label="Observed monomer→dimer (mean)")
        ax.fill_between(r_vals, obs_lo, obs_hi, color="C0", alpha=0.25, label="Observed mean ± SD")
        ax.plot(r_vals, ctrl_mean, color="0.45", lw=1.5, label="Label-permutation (mean)")
        ax.fill_between(r_vals, ctrl_lo, ctrl_hi, color="0.7", alpha=0.4, label="Null mean ± SD")
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
