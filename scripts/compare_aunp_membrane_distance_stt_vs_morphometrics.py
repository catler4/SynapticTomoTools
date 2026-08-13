#!/usr/bin/env python3
"""
Compare AuNP→membrane distances: STT outer/inner mean vs Surface Morphometrics midplane PLY.

For each tomogram in an STT tomograms.csv, loads:
  - AuNP picks: aunp_tm_BP_active_zone_*_manual_refined.star (override with --aunp-pick-star-pattern)
  - STT AZ inner/outer point clouds from STT_results/activezone/
  - SM midplane meshes:
      <alignment_dir>/surface_morphometrics/<tomoname>_cleft_pre.ply
      <alignment_dir>/surface_morphometrics/<tomoname>_cleft_post.ply

STT "center" estimate (same as pipeline):
  distance = mean(nearest_to_outer, nearest_to_inner)

SM estimate:
  distance = nearest distance to the single midplane triangle mesh

Only **synaptic** AuNPs are analyzed by default (same STT rule):
  distance_to_postsynaptic_active_outer <= synaptic_designation_cutoff (default 30 nm).

How to run
----------
  PYTHONPATH=src python scripts/compare_aunp_membrane_distance_stt_vs_morphometrics.py \\
    --csv tomogram_csv_files/tomograms_15F1-H12Cys_FINAL.csv \\
    --data-dir data \\
    --output-dir results/aunp_membrane_distance_stt_vs_morphometrics

If PLY coordinates are in Angstroms (AuNPs are nm), pass ``--mesh-scale 0.1``.

Figures (unless ``--no-plot``):
  - overlaid distance histograms + Gaussian fits (pre and post)
  - Gaussian mean ± SD comparison (STT outer/inner mean vs SM midplane)
  - scatter STT vs SM with identity line
  - residual (SM−STT) histograms + Gaussian fits
  - Bland–Altman plots
  - per-tomogram Gaussian μ ± σ forest plot
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import trimesh
from scipy import stats
from scipy.spatial import KDTree

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from synaptic_tomo_tools.activezone import import_active_zone_segmentations
from synaptic_tomo_tools.alignment_utils import require_alignment_dir
from synaptic_tomo_tools.aunps import (
    DEFAULT_AUNP_PICK_STAR_PATTERN,
    load_aunp_pick_star_dataframes,
)

COORD_COLS = ("faCoordinateX", "faCoordinateY", "faCoordinateZ")

STT_COLOR = "#4C78A8"
SM_COLOR = "#F58518"


def default_data_root() -> Path:
    return Path(os.environ.get("TOMO_ROOT_BASE") or "data")


def parse_az_indices(value) -> list[int] | None:
    """Parse CSV aunp_active_zones; None means discover all matching STAR files."""
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return None
    indices: list[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        indices.append(int(float(part)))
    return indices or None


def load_csv_jobs(
    csv_path: Path,
    *,
    data_dir: Path,
    set_name: str | None = None,
) -> list[tuple[Path, str, str, list[int] | None]]:
    df = pd.read_csv(csv_path)
    required = {"tomoname", "set", "alignment_dir"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")

    if set_name:
        df = df[df["set"].astype(str) == str(set_name)]

    jobs: list[tuple[Path, str, str, list[int] | None]] = []
    for _, row in df.iterrows():
        tomoname = str(row["tomoname"]).strip()
        row_set = str(row["set"]).strip()
        alignment_dir = require_alignment_dir(
            row["alignment_dir"], context=f"tomogram {tomoname}"
        )
        if not tomoname or not row_set or tomoname.lower() == "nan":
            print(f"Skipping row with missing tomoname/set: {row.to_dict()}")
            continue
        az_indices = None
        if "aunp_active_zones" in df.columns:
            az_indices = parse_az_indices(row.get("aunp_active_zones"))
        tomo_path = Path(data_dir) / row_set / "TOP_TOMOS" / tomoname
        jobs.append((tomo_path, alignment_dir, row_set, az_indices))
    return jobs


def nearest_distances_to_cloud(points: np.ndarray, cloud: np.ndarray) -> np.ndarray:
    if cloud is None or len(np.atleast_2d(cloud)) == 0 or np.size(cloud) == 0:
        return np.full(len(points), np.nan)
    cloud = np.atleast_2d(np.asarray(cloud, dtype=float))
    if cloud.ndim != 2 or cloud.shape[1] != 3:
        return np.full(len(points), np.nan)
    tree = KDTree(cloud)
    dists, _ = tree.query(points)
    return np.asarray(dists, dtype=float)


def stack_clouds(clouds: list[np.ndarray]) -> np.ndarray:
    parts = []
    for cloud in clouds:
        arr = np.asarray(cloud, dtype=float)
        if arr.size == 0:
            continue
        arr = np.atleast_2d(arr)
        if arr.shape[1] == 3:
            parts.append(arr)
    return np.vstack(parts) if parts else np.zeros((0, 3), dtype=float)


def load_ply_mesh(path: Path, *, mesh_scale: float) -> trimesh.Trimesh:
    mesh = trimesh.load(path, process=False, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        geoms = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not geoms:
            raise ValueError(f"No triangle mesh found in {path}")
        mesh = trimesh.util.concatenate(geoms)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Expected a triangle mesh in {path}, got {type(mesh)}")
    if float(mesh_scale) != 1.0:
        mesh = mesh.copy()
        mesh.apply_scale(float(mesh_scale))
    return mesh


def nearest_distances_to_mesh(points: np.ndarray, mesh: trimesh.Trimesh) -> np.ndarray:
    """Unsigned nearest distance (nm) from points to triangle mesh surface."""
    points = np.asarray(points, dtype=float)
    if len(points) == 0:
        return np.zeros(0, dtype=float)
    try:
        _, distances, _ = mesh.nearest.on_surface(points)
        return np.asarray(distances, dtype=float)
    except Exception:
        # Fallback: vertex KDTree (closer to STT's point-cloud style)
        tree = KDTree(np.asarray(mesh.vertices, dtype=float))
        dists, _ = tree.query(points)
        return np.asarray(dists, dtype=float)


def stt_outer_inner_mean_distances(
    coords: np.ndarray,
    tomogram_path: Path,
    alignment_dir: str,
) -> dict[str, np.ndarray]:
    az_segmentations = import_active_zone_segmentations(
        tomogram_path, alignment_dir=alignment_dir
    )
    pre_outer = stack_clouds(
        [az.get("presynaptic_outer_coords", []) for az in az_segmentations.values()]
    )
    post_outer = stack_clouds(
        [az.get("postsynaptic_outer_coords", []) for az in az_segmentations.values()]
    )
    pre_inner = stack_clouds(
        [az.get("presynaptic_inner_coords", []) for az in az_segmentations.values()]
    )
    post_inner = stack_clouds(
        [az.get("postsynaptic_inner_coords", []) for az in az_segmentations.values()]
    )

    d_pre_outer = nearest_distances_to_cloud(coords, pre_outer)
    d_post_outer = nearest_distances_to_cloud(coords, post_outer)
    d_pre_inner = nearest_distances_to_cloud(coords, pre_inner)
    d_post_inner = nearest_distances_to_cloud(coords, post_inner)

    return {
        "distance_to_presynaptic_active_outer_nm": d_pre_outer,
        "distance_to_postsynaptic_active_outer_nm": d_post_outer,
        "distance_to_presynaptic_active_inner_nm": d_pre_inner,
        "distance_to_postsynaptic_active_inner_nm": d_post_inner,
        "distance_to_presynaptic_active_outer_inner_mean_nm": np.nanmean(
            np.vstack([d_pre_outer, d_pre_inner]), axis=0
        ),
        "distance_to_postsynaptic_active_outer_inner_mean_nm": np.nanmean(
            np.vstack([d_post_outer, d_post_inner]), axis=0
        ),
        "n_pre_outer_points": len(pre_outer),
        "n_post_outer_points": len(post_outer),
        "n_pre_inner_points": len(pre_inner),
        "n_post_inner_points": len(post_inner),
    }


def fit_gaussian(values: np.ndarray) -> tuple[float, float]:
    """Return (mu, sigma) from a 1D Gaussian MLE fit; NaNs if fewer than 2 points."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return (np.nan, np.nan)
    mu, sigma = stats.norm.fit(values)
    return (float(mu), float(sigma))


def summarize_comparison(df: pd.DataFrame, side: str) -> dict[str, float]:
    stt_col = f"distance_to_{side}synaptic_active_outer_inner_mean_nm"
    sm_col = f"distance_to_{side}synaptic_sm_midplane_nm"
    if stt_col not in df.columns or sm_col not in df.columns:
        return {}
    mask = np.isfinite(df[stt_col]) & np.isfinite(df[sm_col])
    if not np.any(mask):
        return {
            "n": 0.0,
            "mae_nm": np.nan,
            "bias_sm_minus_stt_nm": np.nan,
            "pearson_r": np.nan,
            "stt_mean_nm": np.nan,
            "sm_mean_nm": np.nan,
            "stt_gauss_mu_nm": np.nan,
            "stt_gauss_sigma_nm": np.nan,
            "sm_gauss_mu_nm": np.nan,
            "sm_gauss_sigma_nm": np.nan,
            "delta_gauss_mu_sm_minus_stt_nm": np.nan,
            "delta_gauss_sigma_sm_minus_stt_nm": np.nan,
        }
    stt = df.loc[mask, stt_col].to_numpy(dtype=float)
    sm = df.loc[mask, sm_col].to_numpy(dtype=float)
    delta = sm - stt
    if len(stt) >= 2 and np.std(stt) > 0 and np.std(sm) > 0:
        pearson_r = float(np.corrcoef(stt, sm)[0, 1])
    else:
        pearson_r = np.nan
    stt_mu, stt_sigma = fit_gaussian(stt)
    sm_mu, sm_sigma = fit_gaussian(sm)
    return {
        "n": float(len(stt)),
        "mae_nm": float(np.mean(np.abs(delta))),
        "bias_sm_minus_stt_nm": float(np.mean(delta)),
        "pearson_r": pearson_r,
        "stt_mean_nm": float(np.mean(stt)),
        "sm_mean_nm": float(np.mean(sm)),
        "stt_gauss_mu_nm": stt_mu,
        "stt_gauss_sigma_nm": stt_sigma,
        "sm_gauss_mu_nm": sm_mu,
        "sm_gauss_sigma_nm": sm_sigma,
        "delta_gauss_mu_sm_minus_stt_nm": sm_mu - stt_mu,
        "delta_gauss_sigma_sm_minus_stt_nm": sm_sigma - stt_sigma,
    }


def _hist_with_gaussian(
    ax,
    values: np.ndarray,
    *,
    color: str,
    label: str,
    bins: np.ndarray,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    mu, sigma = fit_gaussian(values)
    if len(values) == 0:
        return mu, sigma
    ax.hist(
        values,
        bins=bins,
        density=True,
        alpha=0.35,
        color=color,
        edgecolor=color,
        label=f"{label} (n={len(values)})",
    )
    if np.isfinite(mu) and np.isfinite(sigma) and sigma > 0:
        x = np.linspace(bins[0], bins[-1], 400)
        ax.plot(
            x,
            stats.norm.pdf(x, mu, sigma),
            color=color,
            lw=2.0,
            label=f"{label} Gauss μ={mu:.2f}, σ={sigma:.2f}",
        )
        ax.axvline(mu, color=color, ls="--", lw=1.2, alpha=0.9)
    return mu, sigma


def write_comparison_figures(
    df: pd.DataFrame,
    output_dir: Path,
    *,
    stem: str = "aunp_membrane_distance_stt_vs_sm",
) -> list[Path]:
    """Write histogram+Gaussian, mean±SD, and scatter comparison figures."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    # --- Histograms + Gaussian fits (pre / post) ---
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), sharey=False)
    gauss_rows: list[dict[str, float | str]] = []
    for ax, side, title in zip(
        axes,
        ("pre", "post"),
        ("Presynaptic AZ", "Postsynaptic AZ"),
    ):
        stt_col = f"distance_to_{side}synaptic_active_outer_inner_mean_nm"
        sm_col = f"distance_to_{side}synaptic_sm_midplane_nm"
        stt = df[stt_col].to_numpy(dtype=float) if stt_col in df.columns else np.array([])
        sm = df[sm_col].to_numpy(dtype=float) if sm_col in df.columns else np.array([])
        finite = np.concatenate(
            [stt[np.isfinite(stt)], sm[np.isfinite(sm)]]
        ) if (np.any(np.isfinite(stt)) or np.any(np.isfinite(sm))) else np.array([])
        if len(finite) == 0:
            ax.set_title(f"{title}\n(no data)")
            ax.set_xlabel("Distance (nm)")
            continue
        lo = float(np.nanpercentile(finite, 1))
        hi = float(np.nanpercentile(finite, 99))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = float(np.nanmin(finite)), float(np.nanmax(finite))
            if hi <= lo:
                hi = lo + 1.0
        bins = np.linspace(lo, hi, 40)
        stt_mu, stt_sigma = _hist_with_gaussian(
            ax, stt, color=STT_COLOR, label="STT outer/inner mean", bins=bins
        )
        sm_mu, sm_sigma = _hist_with_gaussian(
            ax, sm, color=SM_COLOR, label="SM midplane", bins=bins
        )
        ax.set_title(title)
        ax.set_xlabel("Distance to membrane (nm)")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8, loc="best")
        gauss_rows.append(
            {
                "side": side,
                "method": "STT_outer_inner_mean",
                "gauss_mu_nm": stt_mu,
                "gauss_sigma_nm": stt_sigma,
            }
        )
        gauss_rows.append(
            {
                "side": side,
                "method": "SM_midplane",
                "gauss_mu_nm": sm_mu,
                "gauss_sigma_nm": sm_sigma,
            }
        )
    fig.suptitle("AuNP–membrane distances: STT vs Surface Morphometrics", y=1.02)
    fig.tight_layout()
    hist_path = output_dir / f"{stem}_histograms_gaussian.png"
    fig.savefig(hist_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    written.append(hist_path)
    print(f"Wrote {hist_path}")

    # --- Mean ± SD comparison ---
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
    for ax, side, title in zip(
        axes,
        ("pre", "post"),
        ("Presynaptic AZ", "Postsynaptic AZ"),
    ):
        stats_side = summarize_comparison(df, side)
        methods = ["STT outer/inner mean", "SM midplane"]
        mus = [stats_side.get("stt_gauss_mu_nm", np.nan), stats_side.get("sm_gauss_mu_nm", np.nan)]
        sigmas = [
            stats_side.get("stt_gauss_sigma_nm", np.nan),
            stats_side.get("sm_gauss_sigma_nm", np.nan),
        ]
        x = np.arange(len(methods))
        colors = [STT_COLOR, SM_COLOR]
        ax.bar(x, mus, color=colors, alpha=0.75, edgecolor="0.2", width=0.55)
        ax.errorbar(
            x,
            mus,
            yerr=sigmas,
            fmt="none",
            ecolor="0.15",
            elinewidth=1.5,
            capsize=6,
            capthick=1.5,
        )
        for xi, mu, sigma in zip(x, mus, sigmas):
            if np.isfinite(mu) and np.isfinite(sigma):
                ax.text(
                    xi,
                    mu + sigma + 0.05 * max(1.0, abs(mu)),
                    f"μ={mu:.2f}\nσ={sigma:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
        delta_mu = stats_side.get("delta_gauss_mu_sm_minus_stt_nm", np.nan)
        delta_sigma = stats_side.get("delta_gauss_sigma_sm_minus_stt_nm", np.nan)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=15, ha="right")
        ax.set_ylabel("Gaussian fit (nm)")
        ax.set_title(
            f"{title}\nΔμ(SM−STT)={delta_mu:.2f} nm, Δσ(SM−STT)={delta_sigma:.2f} nm"
            if np.isfinite(delta_mu)
            else title
        )
    fig.suptitle("Gaussian fit mean ± SD", y=1.03)
    fig.tight_layout()
    mean_sd_path = output_dir / f"{stem}_gaussian_mean_sd.png"
    fig.savefig(mean_sd_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    written.append(mean_sd_path)
    print(f"Wrote {mean_sd_path}")

    # --- Scatter STT vs SM ---
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.6))
    for ax, side, title in zip(
        axes,
        ("pre", "post"),
        ("Presynaptic AZ", "Postsynaptic AZ"),
    ):
        stt_col = f"distance_to_{side}synaptic_active_outer_inner_mean_nm"
        sm_col = f"distance_to_{side}synaptic_sm_midplane_nm"
        if stt_col not in df.columns or sm_col not in df.columns:
            ax.set_title(f"{title}\n(no data)")
            continue
        mask = np.isfinite(df[stt_col]) & np.isfinite(df[sm_col])
        stt = df.loc[mask, stt_col].to_numpy(dtype=float)
        sm = df.loc[mask, sm_col].to_numpy(dtype=float)
        if len(stt) == 0:
            ax.set_title(f"{title}\n(no data)")
            continue
        ax.scatter(stt, sm, s=12, alpha=0.45, c="0.25", edgecolors="none")
        lo = float(np.nanmin([stt.min(), sm.min()]))
        hi = float(np.nanmax([stt.max(), sm.max()]))
        pad = 0.05 * (hi - lo) if hi > lo else 1.0
        lims = (lo - pad, hi + pad)
        ax.plot(lims, lims, color="0.5", ls="--", lw=1.2, label="y = x")
        stats_side = summarize_comparison(df, side)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("STT outer/inner mean (nm)")
        ax.set_ylabel("SM midplane (nm)")
        ax.set_title(
            f"{title}\nn={int(stats_side.get('n', 0))}, "
            f"r={stats_side.get('pearson_r', np.nan):.3f}"
        )
        ax.legend(fontsize=8, loc="best")
    fig.suptitle("Per-AuNP distance comparison", y=1.02)
    fig.tight_layout()
    scatter_path = output_dir / f"{stem}_scatter.png"
    fig.savefig(scatter_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    written.append(scatter_path)
    print(f"Wrote {scatter_path}")

    # --- Residual (SM − STT) histograms + Gaussian ---
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    for ax, side, title in zip(
        axes,
        ("pre", "post"),
        ("Presynaptic AZ", "Postsynaptic AZ"),
    ):
        stt_col = f"distance_to_{side}synaptic_active_outer_inner_mean_nm"
        sm_col = f"distance_to_{side}synaptic_sm_midplane_nm"
        if stt_col not in df.columns or sm_col not in df.columns:
            ax.set_title(f"{title}\n(no data)")
            continue
        mask = np.isfinite(df[stt_col]) & np.isfinite(df[sm_col])
        delta = (
            df.loc[mask, sm_col].to_numpy(dtype=float)
            - df.loc[mask, stt_col].to_numpy(dtype=float)
        )
        if len(delta) == 0:
            ax.set_title(f"{title}\n(no data)")
            continue
        lo = float(np.nanpercentile(delta, 1))
        hi = float(np.nanpercentile(delta, 99))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = float(np.nanmin(delta)), float(np.nanmax(delta))
            if hi <= lo:
                hi = lo + 1.0
        # Symmetric-ish bins around 0 when possible
        span = max(abs(lo), abs(hi))
        bins = np.linspace(-span, span, 41)
        mu, sigma = _hist_with_gaussian(
            ax, delta, color="#54A24B", label="SM − STT", bins=bins
        )
        ax.axvline(0.0, color="0.4", ls=":", lw=1.2, label="0")
        ax.set_xlabel("Δ distance (SM − STT) (nm)")
        ax.set_ylabel("Density")
        ax.set_title(
            f"{title}\nΔ Gauss μ={mu:.2f}, σ={sigma:.2f}"
            if np.isfinite(mu)
            else title
        )
        ax.legend(fontsize=8, loc="best")
        gauss_rows.append(
            {
                "side": side,
                "method": "delta_SM_minus_STT",
                "gauss_mu_nm": mu,
                "gauss_sigma_nm": sigma,
            }
        )
    fig.suptitle("Residual distances (SM − STT)", y=1.02)
    fig.tight_layout()
    delta_hist_path = output_dir / f"{stem}_delta_histograms_gaussian.png"
    fig.savefig(delta_hist_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    written.append(delta_hist_path)
    print(f"Wrote {delta_hist_path}")

    # --- Bland–Altman ---
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for ax, side, title in zip(
        axes,
        ("pre", "post"),
        ("Presynaptic AZ", "Postsynaptic AZ"),
    ):
        stt_col = f"distance_to_{side}synaptic_active_outer_inner_mean_nm"
        sm_col = f"distance_to_{side}synaptic_sm_midplane_nm"
        if stt_col not in df.columns or sm_col not in df.columns:
            ax.set_title(f"{title}\n(no data)")
            continue
        mask = np.isfinite(df[stt_col]) & np.isfinite(df[sm_col])
        stt = df.loc[mask, stt_col].to_numpy(dtype=float)
        sm = df.loc[mask, sm_col].to_numpy(dtype=float)
        if len(stt) == 0:
            ax.set_title(f"{title}\n(no data)")
            continue
        mean_methods = 0.5 * (stt + sm)
        diff = sm - stt
        bias = float(np.mean(diff))
        sd = float(np.std(diff, ddof=1)) if len(diff) > 1 else np.nan
        loa_hi = bias + 1.96 * sd if np.isfinite(sd) else np.nan
        loa_lo = bias - 1.96 * sd if np.isfinite(sd) else np.nan
        ax.scatter(mean_methods, diff, s=12, alpha=0.4, c="0.25", edgecolors="none")
        ax.axhline(bias, color=SM_COLOR, lw=1.8, label=f"bias={bias:.2f}")
        if np.isfinite(loa_hi):
            ax.axhline(loa_hi, color="0.45", ls="--", lw=1.2, label=f"+1.96σ={loa_hi:.2f}")
            ax.axhline(loa_lo, color="0.45", ls="--", lw=1.2, label=f"−1.96σ={loa_lo:.2f}")
        ax.axhline(0.0, color="0.6", ls=":", lw=1.0)
        ax.set_xlabel("Mean of STT and SM (nm)")
        ax.set_ylabel("SM − STT (nm)")
        ax.set_title(f"{title}\nn={len(diff)}")
        ax.legend(fontsize=8, loc="best")
    fig.suptitle("Bland–Altman: STT outer/inner mean vs SM midplane", y=1.02)
    fig.tight_layout()
    ba_path = output_dir / f"{stem}_bland_altman.png"
    fig.savefig(ba_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    written.append(ba_path)
    print(f"Wrote {ba_path}")

    # --- Per-tomogram μ ± σ forest plot ---
    if "tomoname" in df.columns and "alignment_dir" in df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(12.0, max(4.0, 0.35 * max(1, df.groupby(["tomoname", "alignment_dir"]).ngroups))))
        for ax, side, title in zip(
            axes,
            ("pre", "post"),
            ("Presynaptic AZ", "Postsynaptic AZ"),
        ):
            stt_col = f"distance_to_{side}synaptic_active_outer_inner_mean_nm"
            sm_col = f"distance_to_{side}synaptic_sm_midplane_nm"
            rows = []
            for (tomoname, alignment_dir), sub in df.groupby(
                ["tomoname", "alignment_dir"], sort=True
            ):
                label = f"{tomoname}__{alignment_dir}"
                stt_vals = (
                    sub[stt_col].to_numpy(dtype=float)
                    if stt_col in sub.columns
                    else np.array([])
                )
                sm_vals = (
                    sub[sm_col].to_numpy(dtype=float)
                    if sm_col in sub.columns
                    else np.array([])
                )
                stt_mu, stt_sigma = fit_gaussian(stt_vals)
                sm_mu, sm_sigma = fit_gaussian(sm_vals)
                rows.append(
                    {
                        "label": label,
                        "stt_mu": stt_mu,
                        "stt_sigma": stt_sigma,
                        "sm_mu": sm_mu,
                        "sm_sigma": sm_sigma,
                    }
                )
            if not rows:
                ax.set_title(f"{title}\n(no data)")
                continue
            # Stable order: by STT mu when available
            rows = sorted(
                rows,
                key=lambda r: (
                    1e9
                    if not np.isfinite(r["stt_mu"])
                    else float(r["stt_mu"])
                ),
            )
            y = np.arange(len(rows))
            stt_mu = np.array([r["stt_mu"] for r in rows], dtype=float)
            stt_sigma = np.array([r["stt_sigma"] for r in rows], dtype=float)
            sm_mu = np.array([r["sm_mu"] for r in rows], dtype=float)
            sm_sigma = np.array([r["sm_sigma"] for r in rows], dtype=float)
            labels = [r["label"] for r in rows]
            ax.errorbar(
                stt_mu,
                y - 0.12,
                xerr=stt_sigma,
                fmt="o",
                color=STT_COLOR,
                ecolor=STT_COLOR,
                elinewidth=1.2,
                capsize=3,
                markersize=5,
                label="STT μ±σ",
            )
            ax.errorbar(
                sm_mu,
                y + 0.12,
                xerr=sm_sigma,
                fmt="s",
                color=SM_COLOR,
                ecolor=SM_COLOR,
                elinewidth=1.2,
                capsize=3,
                markersize=5,
                label="SM μ±σ",
            )
            ax.set_yticks(y)
            ax.set_yticklabels(labels, fontsize=7)
            ax.set_xlabel("Gaussian fit distance (nm)")
            ax.set_title(title)
            ax.grid(axis="x", alpha=0.3)
            ax.legend(fontsize=8, loc="best")
        fig.suptitle("Per-tomogram Gaussian μ ± σ", y=1.01)
        fig.tight_layout()
        forest_path = output_dir / f"{stem}_per_tomogram_gaussian_mean_sd.png"
        fig.savefig(forest_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written.append(forest_path)
        print(f"Wrote {forest_path}")
    else:
        print("Skipping per-tomogram forest plot (tomoname/alignment_dir missing)")

    gauss_csv = output_dir / f"{stem}_gaussian_fit_params.csv"
    pd.DataFrame(gauss_rows).to_csv(gauss_csv, index=False)
    written.append(gauss_csv)
    print(f"Wrote {gauss_csv}")
    return written


def analyze_one(
    tomogram_path: Path,
    alignment_dir: str,
    *,
    set_name: str,
    az_indices: list[int] | None,
    star_pattern: str,
    mesh_scale: float,
    output_dir: Path,
    synaptic_designation_cutoff: float = 30.0,
    synaptic_only: bool = True,
    rerun: bool = False,
) -> pd.DataFrame | None:
    tomoname = tomogram_path.name
    alignment_dir = require_alignment_dir(alignment_dir)
    out_path = (
        output_dir
        / f"{tomoname}__{alignment_dir}_aunp_membrane_distance_stt_vs_sm.csv"
    )
    if not rerun and out_path.is_file() and out_path.stat().st_size > 0:
        print(f"  SKIP (already complete): {out_path}")
        try:
            return pd.read_csv(out_path)
        except Exception as exc:
            print(f"  WARNING: could not reload existing CSV ({exc}); recomputing")

    aunps_dir = tomogram_path / alignment_dir / "aunps"
    sm_dir = tomogram_path / alignment_dir / "surface_morphometrics"
    pre_ply = sm_dir / f"{tomoname}_cleft_pre.ply"
    post_ply = sm_dir / f"{tomoname}_cleft_post.ply"

    print(f"\n{'=' * 60}")
    print(f"{set_name} / {tomoname}__{alignment_dir}")
    print(f"{'=' * 60}")

    if not aunps_dir.is_dir():
        print(f"  SKIP: aunps dir missing: {aunps_dir}")
        return None

    star_dfs = load_aunp_pick_star_dataframes(
        aunps_dir, az_indices, pattern=star_pattern
    )
    if not star_dfs:
        print(
            f"  SKIP: no AuNP STAR files matching {star_pattern!r} "
            f"(az filter={az_indices})"
        )
        return None

    df = pd.concat(star_dfs, ignore_index=True)
    missing = [c for c in COORD_COLS if c not in df.columns]
    if missing:
        print(f"  SKIP: STAR missing coordinate columns {missing}")
        return None

    coords = df[list(COORD_COLS)].to_numpy(dtype=float)
    print(f"  AuNPs loaded: {len(coords):,} (zones={sorted(df['active_zone'].unique().tolist())})")

    # STT outer/inner mean
    try:
        stt = stt_outer_inner_mean_distances(coords, tomogram_path, alignment_dir)
    except Exception as exc:
        print(f"  SKIP: could not load STT active-zone segmentations: {exc}")
        return None
    print(
        f"  STT AZ points: pre outer/inner={stt['n_pre_outer_points']:,}/"
        f"{stt['n_pre_inner_points']:,}, post outer/inner="
        f"{stt['n_post_outer_points']:,}/{stt['n_post_inner_points']:,}"
    )

    out = df.copy()
    out.insert(0, "tomoname", tomoname)
    out.insert(1, "set", set_name)
    out.insert(2, "alignment_dir", alignment_dir)
    for key in (
        "distance_to_presynaptic_active_outer_nm",
        "distance_to_postsynaptic_active_outer_nm",
        "distance_to_presynaptic_active_inner_nm",
        "distance_to_postsynaptic_active_inner_nm",
        "distance_to_presynaptic_active_outer_inner_mean_nm",
        "distance_to_postsynaptic_active_outer_inner_mean_nm",
    ):
        out[key] = stt[key]

    # Same synaptic designation as STT analyze_aunps:
    # synaptic if distance to postsynaptic active outer <= cutoff (NaN -> extrasynaptic).
    post_outer = out["distance_to_postsynaptic_active_outer_nm"].to_numpy(dtype=float)
    synaptic_mask = np.isfinite(post_outer) & (
        post_outer <= float(synaptic_designation_cutoff)
    )
    out["synaptic_designation"] = np.where(synaptic_mask, "synaptic", "extrasynaptic")
    n_syn = int(synaptic_mask.sum())
    print(
        f"  Synaptic designation (post active outer ≤ {synaptic_designation_cutoff:g} nm): "
        f"{n_syn:,} synaptic / {len(out) - n_syn:,} extrasynaptic"
    )

    if synaptic_only:
        out = out.loc[synaptic_mask].copy().reset_index(drop=True)
        if out.empty:
            print("  SKIP: no synaptic AuNPs after designation filter")
            return None
        print(f"  Using synaptic AuNPs only: {len(out):,}")
        coords = out[list(COORD_COLS)].to_numpy(dtype=float)

    # Surface Morphometrics midplane meshes
    for side, ply_path, out_col in (
        ("pre", pre_ply, "distance_to_presynaptic_sm_midplane_nm"),
        ("post", post_ply, "distance_to_postsynaptic_sm_midplane_nm"),
    ):
        if not ply_path.is_file():
            print(f"  WARNING: missing {side} PLY: {ply_path}")
            out[out_col] = np.nan
            continue
        mesh = load_ply_mesh(ply_path, mesh_scale=mesh_scale)
        dists = nearest_distances_to_mesh(coords, mesh)
        out[out_col] = dists
        print(
            f"  SM {side} mesh: {ply_path.name} "
            f"({len(mesh.vertices):,} verts, {len(mesh.faces):,} faces); "
            f"mean dist={np.nanmean(dists):.2f} nm"
        )

    out["delta_pre_sm_minus_stt_mean_nm"] = (
        out["distance_to_presynaptic_sm_midplane_nm"]
        - out["distance_to_presynaptic_active_outer_inner_mean_nm"]
    )
    out["delta_post_sm_minus_stt_mean_nm"] = (
        out["distance_to_postsynaptic_sm_midplane_nm"]
        - out["distance_to_postsynaptic_active_outer_inner_mean_nm"]
    )

    for side in ("pre", "post"):
        stats = summarize_comparison(out, side)
        if stats.get("n", 0) > 0:
            print(
                f"  Compare {side}: n={int(stats['n'])}, "
                f"MAE={stats['mae_nm']:.2f} nm, "
                f"bias(SM-STT)={stats['bias_sm_minus_stt_nm']:.2f} nm, "
                f"r={stats['pearson_r']:.3f}"
            )

    out_path = (
        output_dir
        / f"{tomoname}__{alignment_dir}_aunp_membrane_distance_stt_vs_sm.csv"
    )
    out.to_csv(out_path, index=False)
    print(f"  Wrote {out_path}")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare AuNP membrane distances: STT outer/inner mean vs "
            "Surface Morphometrics midplane PLY."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--csv", type=Path, required=True, help="STT tomograms.csv")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Data root (default: $TOMO_ROOT_BASE or 'data')",
    )
    parser.add_argument(
        "--set",
        dest="set_name",
        default=None,
        help="Optional set filter",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/aunp_membrane_distance_stt_vs_morphometrics"),
        help="Directory for per-tomogram and pooled CSVs",
    )
    parser.add_argument(
        "--aunp-pick-star-pattern",
        type=str,
        default=DEFAULT_AUNP_PICK_STAR_PATTERN,
        help="AuNP STAR pattern (* = active-zone index)",
    )
    parser.add_argument(
        "--mesh-scale",
        type=float,
        default=1.0,
        help="Multiply PLY coordinates by this (use 0.1 if mesh is in Å and AuNPs in nm)",
    )
    parser.add_argument(
        "--synaptic-designation-cutoff",
        type=float,
        default=30.0,
        help=(
            "AuNP is synaptic if distance to postsynaptic active outer ≤ this (nm); "
            "same default as STT analyze_aunps"
        ),
    )
    parser.add_argument(
        "--include-extrasynaptic",
        action="store_true",
        help="Include extrasynaptic AuNPs (default: synaptic only, matching STT)",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Abort on first tomogram failure",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip PNG figures (histograms, Gaussian mean±SD, scatter)",
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Recompute even if per-tomogram comparison CSV already exists",
    )
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir) if args.data_dir else default_data_root()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    jobs = load_csv_jobs(
        Path(args.csv), data_dir=data_dir, set_name=args.set_name
    )
    if not jobs:
        print(f"No tomogram rows to process from {args.csv}", file=sys.stderr)
        return 1

    print(f"Jobs: {len(jobs)} from {args.csv} (data root: {data_dir})")
    print(f"AuNP STAR pattern: {args.aunp_pick_star_pattern}")
    print(f"Mesh scale: {args.mesh_scale}")
    print(
        f"Synaptic filter: "
        f"{'off (--include-extrasynaptic)' if args.include_extrasynaptic else 'on'} "
        f"(cutoff={args.synaptic_designation_cutoff:g} nm to post active outer)"
    )
    print(f"Output: {output_dir}")

    frames: list[pd.DataFrame] = []
    failed: list[str] = []
    for i, (tomo_path, alignment_dir, set_name, az_indices) in enumerate(jobs, start=1):
        label = f"{tomo_path.name}__{alignment_dir}"
        print(f"\n[{i}/{len(jobs)}] {label}")
        try:
            df = analyze_one(
                tomo_path,
                alignment_dir,
                set_name=set_name,
                az_indices=az_indices,
                star_pattern=args.aunp_pick_star_pattern,
                mesh_scale=float(args.mesh_scale),
                output_dir=output_dir,
                synaptic_designation_cutoff=float(args.synaptic_designation_cutoff),
                synaptic_only=not bool(args.include_extrasynaptic),
                rerun=bool(args.rerun),
            )
            if df is None:
                failed.append(label)
            else:
                frames.append(df)
        except Exception as exc:
            print(f"FAILED {label}: {exc}")
            failed.append(label)
            if args.stop_on_error:
                raise

    if frames:
        pooled = pd.concat(frames, ignore_index=True)
        pooled_path = output_dir / "aunp_membrane_distance_stt_vs_sm_pooled.csv"
        pooled.to_csv(pooled_path, index=False)
        print(f"\nWrote pooled table -> {pooled_path} ({len(pooled):,} AuNPs)")

        summary_rows = []
        for side in ("pre", "post"):
            stats_side = summarize_comparison(pooled, side)
            summary_rows.append({"side": side, **stats_side})
            if stats_side.get("n", 0) > 0:
                print(
                    f"Pooled {side}: n={int(stats_side['n'])}, "
                    f"MAE={stats_side['mae_nm']:.2f} nm, "
                    f"bias(SM-STT)={stats_side['bias_sm_minus_stt_nm']:.2f} nm, "
                    f"r={stats_side['pearson_r']:.3f}"
                )
                print(
                    f"         Gauss STT μ={stats_side['stt_gauss_mu_nm']:.2f} "
                    f"σ={stats_side['stt_gauss_sigma_nm']:.2f}; "
                    f"SM μ={stats_side['sm_gauss_mu_nm']:.2f} "
                    f"σ={stats_side['sm_gauss_sigma_nm']:.2f}; "
                    f"Δμ={stats_side['delta_gauss_mu_sm_minus_stt_nm']:.2f}, "
                    f"Δσ={stats_side['delta_gauss_sigma_sm_minus_stt_nm']:.2f}"
                )
        summary = pd.DataFrame(summary_rows)
        summary_path = output_dir / "aunp_membrane_distance_stt_vs_sm_summary.csv"
        summary.to_csv(summary_path, index=False)
        print(f"Wrote summary -> {summary_path}")

        if not args.no_plot:
            write_comparison_figures(pooled, output_dir)

    print(f"\nDone. Succeeded: {len(frames)}/{len(jobs)}")
    if failed:
        print(f"Failed/skipped ({len(failed)}):")
        for label in failed:
            print(f"  {label}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
