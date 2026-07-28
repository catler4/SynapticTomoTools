"""
3D bivariate Ripley K₁₂ / L₁₂ of AuNP positions relative to the active zone center.

Type-1 foci: one active zone center per zone (mean of presynaptic + postsynaptic AZ points).
Type-2 partners: AuNP pick coordinates in that zone.

Window: synaptic_cleft_az_hull (convex hull of presynaptic + postsynaptic AZ surface points).
No null-model controls — observed L₁₂ curves only, with pooled mean ± SD/SEM for Prism.
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
    DEFAULT_RIPLEY_R_STEP_NM,
    _prism_sd_envelope_columns,
    _ripley_r_grid,
    build_ripley_window_3d,
    cross_k12_3d_isotropic,
    curves_matrix_to_long_dataframe,
    curves_matrix_to_wide_dataframe,
    load_synaptic_cleft_active_zone_points,
    mean_l12_from_averaged_k12,
    ripley_l12,
)


def _prism_long_to_wide(prism_long: pd.DataFrame, id_cols: Sequence[str]) -> pd.DataFrame:
    if prism_long.empty:
        return prism_long.copy()
    value_cols = [c for c in prism_long.columns if c not in id_cols and c != "r_nm"]
    return prism_long[list(id_cols) + ["r_nm"] + value_cols].copy()


def _safe_name(name: str) -> str:
    safe = str(name).strip().replace(" ", "_")
    for ch in '<>:"/\\|?*':
        safe = safe.replace(ch, "_")
    return safe

WINDOW_MODE = "synaptic_cleft_az_hull"
MIN_AUNP_PARTNERS = 3
AZ_CENTER_RIPLEY_R_MAX_NM = 500.0

POOLED_CURVES_CSV = Path("results/aunps/aunp_vs_az_center_ripley_l12_curves.csv")
POOLED_PRISM_CSV = Path("results/aunps/aunp_vs_az_center_ripley_l12_prism_pooled.csv")
POOLED_PRISM_WIDE_CSV = Path("results/aunps/aunp_vs_az_center_ripley_l12_prism_pooled_wide.csv")
POOLED_FIGURES_DIR = Path("results/aunps/figures/aunp_vs_az_center_ripley_l12_pooled")


def compute_active_zone_center_nm(az_segmentation: dict) -> np.ndarray:
    """
    Active zone center: mean of presynaptic and postsynaptic active-zone surface points.

    Matches the definition used in ``analyze_aunps`` for ``distance_to_active_zone_center``,
    applied per zone.
    """
    parts: list[np.ndarray] = []
    for key in ("presynaptic_coords", "postsynaptic_coords"):
        pts = az_segmentation.get(key)
        if pts is not None and len(pts) > 0:
            parts.append(np.atleast_2d(np.asarray(pts, dtype=float)))
    if not parts:
        return np.full(3, np.nan)
    return np.mean(np.vstack(parts), axis=0)


def _extract_zone_curves_matrix(
    df: pd.DataFrame,
    value_col: str = "l12",
) -> tuple[np.ndarray, np.ndarray]:
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


def _extract_zone_l12_curves_matrix(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Pivot long L₁₂ table to (r_vals, curves) with one curve per tomogram+zone."""
    return _extract_zone_curves_matrix(df, value_col="l12")


def build_aunp_vs_az_center_prism_table(
    *,
    zone_name: str,
    r_vals: np.ndarray,
    k12: np.ndarray,
    l12: np.ndarray,
    n_aunps: int,
    window_volume_nm3: float,
) -> pd.DataFrame:
    """Per-zone Prism table (one observed curve; SD/SEM columns are NaN for a single curve)."""
    rows: list[dict] = []
    for i, r_nm in enumerate(r_vals):
        rows.append(
            {
                "active_zone_name": zone_name,
                "window_mode": WINDOW_MODE,
                "r_nm": float(r_nm),
                "k12": float(k12[i]),
                "center_L12": float(l12[i]),
                "center_L12_mean": float(l12[i]),
                "center_L12_sd": np.nan,
                "center_L12_sd_envelope_lo": np.nan,
                "center_L12_sd_envelope_hi": np.nan,
                "center_L12_sem": np.nan,
                "center_L12_sem_envelope_lo": np.nan,
                "center_L12_sem_envelope_hi": np.nan,
                "n_aunp_partners": int(n_aunps),
                "window_volume_nm3": float(window_volume_nm3),
            }
        )
    return pd.DataFrame(rows)


def build_pooled_aunp_vs_az_center_prism_table(df: pd.DataFrame) -> pd.DataFrame:
    """Pooled mean ± SD/SEM of L₁₂ across tomogram-zone curves at each r, per tomogram set.

    Reports both mean-of-L (``center_L12_mean``) and mean-of-K then convert to L
    (``center_L12_mean_from_k``). Uses stored ``k12`` when present.
    """
    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    if "set_name" not in df.columns:
        df["set_name"] = ""
    df["set_name"] = df["set_name"].fillna("").astype(str)

    rows: list[dict] = []
    for set_name, sub in df.groupby("set_name", sort=False):
        r_vals, curves = _extract_zone_l12_curves_matrix(sub)
        if len(curves) == 0:
            continue

        sd = _prism_sd_envelope_columns(curves, r_vals, prefix="center_L12")
        if "k12" in sub.columns:
            _, k_curves = _extract_zone_curves_matrix(sub, value_col="k12")
            mean_from_k = mean_l12_from_averaged_k12(
                curves,
                r_vals,
                k12_curves=k_curves if len(k_curves) == len(curves) else None,
            )
        else:
            mean_from_k = mean_l12_from_averaged_k12(curves, r_vals)
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
                    "center_L12_mean": float(sd["center_L12_mean"][i]),
                    "center_L12_mean_from_k": float(mean_from_k[i]),
                    "center_L12_sd": float(sd["center_L12_sd"][i]),
                    "center_L12_sd_envelope_lo": float(sd["center_L12_sd_envelope_lo"][i]),
                    "center_L12_sd_envelope_hi": float(sd["center_L12_sd_envelope_hi"][i]),
                    "center_L12_sem": float(sd["center_L12_sem"][i]),
                    "center_L12_sem_envelope_lo": float(sd["center_L12_sem_envelope_lo"][i]),
                    "center_L12_sem_envelope_hi": float(sd["center_L12_sem_envelope_hi"][i]),
                    "n_zone_curves": int(len(curves)),
                    "n_tomograms": n_tomograms,
                    "n_active_zones": n_zones,
                }
            )
    return pd.DataFrame(rows)


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
    seed: int = DEFAULT_ANALYSIS_SEED,
    write_figures: bool = True,
) -> dict[str, Path] | None:
    """Compute observed 3D L₁₂(center, AuNPs) for one active zone."""
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

    try:
        cleft_coords = load_synaptic_cleft_active_zone_points(
            tomogram_path, alignment_dir, zone_name
        )
        window = build_ripley_window_3d(cleft_coords, mode=WINDOW_MODE)
    except Exception as exc:
        print(f"  Skipping AZ-center Ripley for {zone_name}: {exc}")
        return None

    r_vals = _ripley_r_grid(r_max_nm, r_step_nm)
    rng = np.random.default_rng(seed)
    center_xyz = center.reshape(1, 3)
    k12 = cross_k12_3d_isotropic(center_xyz, aunp_coords, r_vals, window, rng)
    l12 = ripley_l12(k12, r_vals)

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

    curves_df = pd.DataFrame(
        {
            "active_zone_name": zone_name,
            "active_zone_index": int(active_zone_index),
            "window_mode": WINDOW_MODE,
            "r_nm": r_vals,
            "k12": k12,
            "l12": l12,
            "n_aunp_partners": len(aunp_coords),
            "window_volume_nm3": float(window.volume_nm3),
            "center_x_nm": float(center[0]),
            "center_y_nm": float(center[1]),
            "center_z_nm": float(center[2]),
        }
    )
    curves_path = out_dir / "ripley_l12_curves.csv"
    curves_df.to_csv(curves_path, index=False)

    individual_df = curves_matrix_to_long_dataframe(
        np.atleast_2d(l12),
        r_vals,
        curve_type="observed",
        extra_cols={
            "active_zone_name": zone_name,
            "active_zone_index": int(active_zone_index),
            "window_mode": WINDOW_MODE,
            "n_aunp_partners": int(len(aunp_coords)),
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

    meta = {
        "tomogram_name": tomogram_name,
        "alignment_dir": alignment_dir,
        "active_zone_name": zone_name,
        "active_zone_index": int(active_zone_index),
        "window_mode": WINDOW_MODE,
        "n_aunp_partners": int(len(aunp_coords)),
        "window_volume_nm3": float(window.volume_nm3),
        "center_definition": "mean_of_presynaptic_and_postsynaptic_active_zone_surface_points",
        "ripley_edge_correction": "isotropic_3d_mc",
        "seed": int(seed),
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
    seed: int = DEFAULT_ANALYSIS_SEED,
    write_figures: bool = True,
) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    """Run AZ-center Ripley for all mapped active zones in one tomogram."""
    from .activezone import load_active_zone_mapping

    tomogram_path = Path(tomogram_path)
    alignment_dir = require_alignment_dir(alignment_dir)
    az_mapping = load_active_zone_mapping(tomogram_path, alignment_dir) or {}
    if not az_mapping:
        print("No active zone mapping; skipping AuNP vs AZ-center Ripley analyses")
        return [], []

    az_mapping = {int(k): v for k, v in az_mapping.items()}
    indices = list(active_zone_indices) if active_zone_indices is not None else sorted(az_mapping)
    coord_cols = ["faCoordinateX", "faCoordinateY", "faCoordinateZ"]

    curve_frames: list[pd.DataFrame] = []
    prism_frames: list[pd.DataFrame] = []

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


def plot_pooled_aunp_vs_az_center_ripley_visualizations(
    curves_csv: Path | str = POOLED_CURVES_CSV,
    output_dir: Path | str = POOLED_FIGURES_DIR,
    prism_csv: Path | str = POOLED_PRISM_CSV,
    prism_wide_csv: Path | str = POOLED_PRISM_WIDE_CSV,
) -> list[Path]:
    """Build pooled Prism tables and a mean ± SD L₁₂ figure across all zones/tomograms."""
    curves_csv = Path(curves_csv)
    output_dir = Path(output_dir)
    prism_csv = Path(prism_csv)
    prism_wide_csv = Path(prism_wide_csv)

    if not curves_csv.is_file():
        print(f"No pooled AuNP vs AZ-center Ripley CSV at {curves_csv}; skipping pooled outputs.")
        return []

    df = pd.read_csv(curves_csv)
    if df.empty or "tomogram_name" not in df.columns:
        print("Pooled AuNP vs AZ-center Ripley CSV missing data; skipping pooled outputs.")
        return []

    prism_long = build_pooled_aunp_vs_az_center_prism_table(df)
    if prism_long.empty:
        print("No pooled AuNP vs AZ-center Ripley envelope rows generated.")
        return []

    prism_csv.parent.mkdir(parents=True, exist_ok=True)
    prism_long.to_csv(prism_csv, index=False)
    _prism_long_to_wide(prism_long, id_cols=["set_name", "window_mode"]).to_csv(
        prism_wide_csv, index=False
    )
    print(f"Pooled AuNP vs AZ-center Ripley Prism table ({len(prism_long)} rows) -> {prism_csv}")

    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = [prism_csv, prism_wide_csv]

    for set_name, grp in prism_long.groupby("set_name", sort=False):
        grp = grp.sort_values("r_nm")
        r_vals = grp["r_nm"].to_numpy(dtype=float)
        mean = grp["center_L12_mean"].to_numpy(dtype=float)
        lo = grp["center_L12_sd_envelope_lo"].to_numpy(dtype=float)
        hi = grp["center_L12_sd_envelope_hi"].to_numpy(dtype=float)
        meta = grp.iloc[0]

        set_tag = _safe_name(str(set_name)) or "unspecified"
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        ax.plot(r_vals, mean, color="C0", lw=2, label="Mean L₁₂ (of L)")
        if "center_L12_mean_from_k" in grp.columns:
            ax.plot(
                r_vals,
                grp["center_L12_mean_from_k"].to_numpy(dtype=float),
                color="C0",
                lw=1.5,
                ls="--",
                label="Mean L₁₂ (K→L)",
            )
        ax.fill_between(r_vals, lo, hi, color="C0", alpha=0.25, label="Mean ± SD")
        ax.axhline(0.0, color="0.5", ls="--", lw=0.8)
        ax.set_xlabel("r (nm)")
        ax.set_ylabel("Ripley L₁₂(r) = (3K₁₂/4π)^(1/3) − r")
        ax.set_title(
            f"Pooled AuNP vs active zone center — set: {set_name}\n"
            f"{int(meta['n_tomograms'])} tomogram(s), {int(meta['n_active_zones'])} zone(s), "
            f"{int(meta['n_zone_curves'])} curves"
        )
        ax.set_xlim(0.0, float(r_vals[-1]) if len(r_vals) else AZ_CENTER_RIPLEY_R_MAX_NM)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        out_path = output_dir / f"ripley_l12_pooled_mean_sd_{set_tag}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Pooled AuNP vs AZ-center Ripley figure (set {set_name}) -> {out_path}")
        written.append(out_path)

    return written
