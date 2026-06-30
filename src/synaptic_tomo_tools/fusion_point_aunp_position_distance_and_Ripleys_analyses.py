"""
3D fusion-site vs AuNP distance tables and bivariate Ripley K₁₂ / L₁₂ analysis.

Uses raw tomogram coordinates (no postsynaptic projection) except label-permutation
sites that land on an original AuNP pool index — those are snapped to the presynaptic
active zone. Tangential 40 nm presynaptic shifts use the same placement rules as the
legacy fusion-point control code.

Ripley window options (both run for every zone):

1. ``fusion_aunp_hull`` — convex hull of fusing + close fusion sites and all monomer+dimer AuNPs.
2. ``synaptic_cleft_az_hull`` — convex hull of all presynaptic + postsynaptic active-zone surface
   points (entire synaptic cleft envelope for the zone).

Distance and Ripley partner sets are reported separately for monomer-only, dimer-only,
and combined monomer+dimer picks when both STAR files are present. If only one STAR
file exists, analyses run for that kind only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, cKDTree

from .alignment_utils import require_alignment_dir
from .aunps import (
    _read_aunp_pick_star_dataframe,
    enumerate_close_vesicle_fusion_points,
)
from .fusion_point_vs_aunp_density import (
    FUSION_POINT_AZ_MAX_SNAP_DISTANCE_NM,
    FUSION_POINT_SHIFT_OFFSET_NM,
    filter_fusion_rows_for_zone,
    load_presynaptic_az_points_for_zone,
    presynaptic_membrane_name_for_zone,
    sample_tangential_control_on_az,
    zone_name_for_presynaptic_membrane,
)
from .vesicles import import_presynaptic_membranes_and_active_zones

WindowMode = Literal["fusion_aunp_hull", "synaptic_cleft_az_hull"]
RIPLEY_WINDOW_MODES: tuple[WindowMode, ...] = ("fusion_aunp_hull", "synaptic_cleft_az_hull")
ControlKind = Literal["close", "shift_40nm", "label_permutation"]
AunpSubset = Literal["monomer", "dimer", "both"]

AUNP_SUBSETS: tuple[AunpSubset, ...] = ("monomer", "dimer", "both")

DEFAULT_NULL_REPLICATES_N = 100
DEFAULT_ANALYSIS_SEED = 42
DEFAULT_RIPLEY_R_MAX_NM = 100.0
DEFAULT_RIPLEY_R_STEP_NM = 1.0
RIPLEY_PERCENTILE_LO = 2.5
RIPLEY_PERCENTILE_HI = 97.5
EDGE_MC_SAMPLES = 384
EDGE_MIN_C = 1e-3

ANALYSES_SUBDIR = "fusion_point_aunp_analyses"
COORD_COLS = ("faCoordinateX", "faCoordinateY", "faCoordinateZ")

POOLED_RIPLEY_CURVES_CSV = Path("results/aunps/fusion_point_aunp_ripley_l12_curves.csv")
POOLED_RIPLEY_PRISM_CSV = Path("results/aunps/fusion_point_aunp_ripley_l12_prism_envelopes_pooled.csv")
POOLED_RIPLEY_PRISM_WIDE_CSV = Path("results/aunps/fusion_point_aunp_ripley_l12_prism_envelopes_pooled_wide.csv")
POOLED_RIPLEY_FIGURES_DIR = Path("results/aunps/figures/fusion_point_aunp_ripley_l12_pooled")

DEFAULT_MONOMER_STAR_PATTERN = "aunp_tm_BP_active_zone_*_manual_refined_monomer.star"
DEFAULT_DIMER_STAR_PATTERN = "aunp_tm_BP_active_zone_*_manual_refined_dimer.star"

CONTROL_COMPARISONS: tuple[ControlKind, ...] = ("close", "shift_40nm", "label_permutation")
CONTROL_CURVE_TYPE: dict[ControlKind, str] = {
    "close": "close_per_vesicle",
    "shift_40nm": "shift_40nm_replicate",
    "label_permutation": "label_permutation_replicate",
}
FUSING_CURVE_TYPE = "fusing_per_vesicle"
AunpKind = Literal["monomer", "dimer"]


@dataclass(frozen=True)
class ZoneAunpLoadResult:
    """Monomer and/or dimer pick coordinates loaded for one active zone."""

    coords: np.ndarray
    meta: pd.DataFrame
    kinds_loaded: tuple[AunpKind, ...]


def available_aunp_subsets(kinds_loaded: Sequence[AunpKind]) -> tuple[AunpSubset, ...]:
    """Analysis subsets to run given which STAR files were found."""
    kinds = tuple(kinds_loaded)
    out: list[AunpSubset] = []
    if "monomer" in kinds:
        out.append("monomer")
    if "dimer" in kinds:
        out.append("dimer")
    if "monomer" in kinds and "dimer" in kinds:
        out.append("both")
    return tuple(out)


@dataclass(frozen=True)
class RipleyWindow3D:
    """3D convex-hull window; ``volume_nm3`` is the full hull volume."""

    volume_nm3: float
    hull: ConvexHull
    defining_mode: WindowMode


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


def _normalize_monomer_dimer_star_pattern(
    pattern: Optional[str],
    *,
    default: str,
) -> str:
    """Return monomer/dimer STAR filename pattern (``*`` = active zone index)."""
    if pattern is None or not str(pattern).strip():
        return default
    pat = str(pattern).strip()
    if "*" not in pat or pat.count("*") != 1:
        raise ValueError(
            "Monomer/dimer STAR pattern must contain exactly one '*' for the active zone index "
            f"(e.g. {default!r})."
        )
    if not pat.endswith(".star"):
        raise ValueError("Monomer/dimer STAR pattern must end with '.star'.")
    return pat


def _monomer_dimer_star_filename(active_zone_index: int, pattern: str) -> str:
    return pattern.replace("*", str(int(active_zone_index)), 1)


def _resolve_monomer_dimer_star_paths(
    aunps_dir: Path,
    tomogram_name: str,
    alignment_dir: str,
    active_zone_index: int,
    *,
    kind: Literal["monomer", "dimer"],
    pattern: Optional[str] = None,
) -> Path:
    default = (
        DEFAULT_MONOMER_STAR_PATTERN if kind == "monomer" else DEFAULT_DIMER_STAR_PATTERN
    )
    pat = _normalize_monomer_dimer_star_pattern(pattern, default=default)
    filename = _monomer_dimer_star_filename(active_zone_index, pat)
    candidates = [
        aunps_dir / f"{tomogram_name}_{alignment_dir}_{filename}",
        aunps_dir / filename,
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"Required AuNP {kind} STAR not found for active zone {active_zone_index} "
        f"(pattern {pat!r}). Tried: {[str(p) for p in candidates]}"
    )


def _find_monomer_dimer_star_path(
    aunps_dir: Path,
    tomogram_name: str,
    alignment_dir: str,
    active_zone_index: int,
    *,
    kind: AunpKind,
    pattern: Optional[str] = None,
) -> Path | None:
    """Return monomer/dimer STAR path if present, else None."""
    try:
        return _resolve_monomer_dimer_star_paths(
            aunps_dir,
            tomogram_name,
            alignment_dir,
            active_zone_index,
            kind=kind,
            pattern=pattern,
        )
    except FileNotFoundError:
        return None


def _read_aunp_kind_star_frame(
    path: Path,
    *,
    kind: AunpKind,
    active_zone_index: int,
) -> pd.DataFrame:
    df = _read_aunp_pick_star_dataframe(path)
    if df is None or df.empty:
        raise ValueError(f"Empty or unreadable {kind} STAR: {path}")
    missing = [c for c in COORD_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    part = df[list(COORD_COLS)].copy()
    part["aunp_kind"] = kind
    part["source_star"] = path.name
    part["active_zone_index"] = int(active_zone_index)
    return part


def load_monomer_dimer_aunps_for_zone(
    tomogram_path: Path,
    alignment_dir: str,
    active_zone_index: int,
    *,
    monomer_star_pattern: Optional[str] = None,
    dimer_star_pattern: Optional[str] = None,
) -> ZoneAunpLoadResult:
    """Load monomer and/or dimer pick coordinates for one active zone index.

    Runs when at least one STAR file is present. Missing monomer or dimer files are
    skipped; ``kinds_loaded`` records which were found.
    """
    tomogram_path = Path(tomogram_path)
    alignment_dir = require_alignment_dir(alignment_dir)
    aunps_dir = tomogram_path / alignment_dir / "aunps"
    tomogram_name = tomogram_path.name

    frames: list[pd.DataFrame] = []
    kinds_loaded: list[AunpKind] = []

    for kind, pattern in (
        ("monomer", monomer_star_pattern),
        ("dimer", dimer_star_pattern),
    ):
        path = _find_monomer_dimer_star_path(
            aunps_dir,
            tomogram_name,
            alignment_dir,
            active_zone_index,
            kind=kind,
            pattern=pattern,
        )
        if path is None:
            continue
        frames.append(
            _read_aunp_kind_star_frame(
                path, kind=kind, active_zone_index=active_zone_index
            )
        )
        kinds_loaded.append(kind)

    if not frames:
        raise FileNotFoundError(
            f"No monomer or dimer AuNP STAR files found for active zone "
            f"{active_zone_index} in {aunps_dir} "
            f"(monomer pattern {monomer_star_pattern or DEFAULT_MONOMER_STAR_PATTERN!r}, "
            f"dimer pattern {dimer_star_pattern or DEFAULT_DIMER_STAR_PATTERN!r})"
        )

    meta = pd.concat(frames, ignore_index=True)
    meta["aunp_index"] = np.arange(len(meta), dtype=int)
    coords = meta[list(COORD_COLS)].to_numpy(dtype=float)
    return ZoneAunpLoadResult(
        coords=coords,
        meta=meta,
        kinds_loaded=tuple(kinds_loaded),
    )


def subset_aunps(
    meta: pd.DataFrame,
    *,
    subset: AunpSubset,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Return coordinates and metadata for one AuNP partner subset."""
    if subset == "both":
        out = meta.copy()
    else:
        out = meta.loc[meta["aunp_kind"] == subset].copy()
    out = out.reset_index(drop=True)
    coords = out[list(COORD_COLS)].to_numpy(dtype=float)
    return coords, out


def load_postsynaptic_active_zone_surface(
    tomogram_path: Path,
    alignment_dir: str,
    zone_name: str,
) -> np.ndarray:
    az_dir = Path(tomogram_path) / alignment_dir / "STT_results" / "activezone"
    parts: list[np.ndarray] = []
    for suffix in ("post_outer", "post_inner"):
        path = az_dir / f"{zone_name}_{suffix}.txt"
        if path.is_file():
            surf = np.atleast_2d(np.loadtxt(path, delimiter=None))
            if surf.size:
                parts.append(surf.astype(float))
    if not parts:
        raise FileNotFoundError(f"No postsynaptic AZ surfaces for {zone_name} in {az_dir}")
    return np.vstack(parts)


def load_synaptic_cleft_active_zone_points(
    tomogram_path: Path,
    alignment_dir: str,
    zone_name: str,
) -> np.ndarray:
    """Presynaptic + postsynaptic active-zone surface points for one zone."""
    pre = load_presynaptic_az_points_for_zone(tomogram_path, alignment_dir, zone_name)
    post = load_postsynaptic_active_zone_surface(tomogram_path, alignment_dir, zone_name)
    parts = [arr for arr in (pre, post) if len(arr)]
    if not parts:
        raise FileNotFoundError(
            f"No presynaptic or postsynaptic active-zone surface points for {zone_name}"
        )
    return np.vstack(parts)


def build_fusion_aunp_hull_defining_coords(
    fusing_xyz: np.ndarray,
    close_xyz: np.ndarray,
    aunp_coords_all: np.ndarray,
) -> np.ndarray:
    """
    Points used to define the Ripley window hull.

    Always includes all monomer+dimer AuNPs plus fusing and close-vesicle fusion sites.
    """
    parts: list[np.ndarray] = []
    for arr in (fusing_xyz, close_xyz, aunp_coords_all):
        arr = np.atleast_2d(np.asarray(arr, dtype=float))
        if len(arr):
            parts.append(arr)
    if not parts:
        return np.zeros((0, 3), dtype=float)
    return np.vstack(parts)


def build_ripley_window_3d(
    defining_coords: np.ndarray,
    mode: WindowMode,
) -> RipleyWindow3D:
    defining_coords = np.atleast_2d(np.asarray(defining_coords, dtype=float))
    if len(defining_coords) < 4:
        raise ValueError(f"Need at least 4 points to build a 3D convex hull ({mode}), got {len(defining_coords)}")
    hull = ConvexHull(defining_coords)
    if hull.volume <= 0:
        raise ValueError(f"Convex hull volume must be positive ({mode})")
    return RipleyWindow3D(volume_nm3=float(hull.volume), hull=hull, defining_mode=mode)


def _points_inside_hull(pts: np.ndarray, hull: ConvexHull, tol: float = 1e-6) -> np.ndarray:
    pts = np.atleast_2d(np.asarray(pts, dtype=float))
    if len(pts) == 0:
        return np.zeros(0, dtype=bool)
    normals = hull.equations[:, :-1]
    offsets = hull.equations[:, -1]
    return np.all(pts @ normals.T + offsets <= tol, axis=1)


def _isotropic_edge_factor_3d(
    center: np.ndarray,
    radius_nm: float,
    hull: ConvexHull,
    rng: np.random.Generator,
    *,
    n_samples: int = EDGE_MC_SAMPLES,
) -> float:
    if radius_nm <= 0:
        return 1.0
    center = np.asarray(center, dtype=float).reshape(3)
    dirs = rng.normal(size=(n_samples, 3))
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    dirs /= norms
    radii = rng.random(n_samples) ** (1.0 / 3.0) * float(radius_nm)
    samples = center + dirs * radii[:, None]
    inside = _points_inside_hull(samples, hull)
    frac = float(np.mean(inside))
    return max(frac, EDGE_MIN_C)


def cross_k12_3d_isotropic(
    x: np.ndarray,
    y: np.ndarray,
    r_vals: np.ndarray,
    window: RipleyWindow3D,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Edge-corrected bivariate cross-K in 3D.

    Type-1 foci ``x`` (fusion / controls); type-2 ``y`` (AuNPs).
    """
    x = np.atleast_2d(np.asarray(x, dtype=float))
    y = np.atleast_2d(np.asarray(y, dtype=float))
    r_vals = np.asarray(r_vals, dtype=float)
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0 or window.volume_nm3 <= 0:
        return np.full(len(r_vals), np.nan)

    tree = cKDTree(y)
    r_max = float(r_vals[-1])
    edge_factors = np.array(
        [_isotropic_edge_factor_3d(xi, r_max, window.hull, rng) for xi in x],
        dtype=float,
    )

    counts = np.zeros(len(r_vals), dtype=float)
    for i, xi in enumerate(x):
        neighbor_idx = tree.query_ball_point(xi, r=r_max)
        if not neighbor_idx:
            continue
        dists = np.linalg.norm(y[np.asarray(neighbor_idx, dtype=int)] - xi, axis=1)
        c_max = edge_factors[i]
        for k, r in enumerate(r_vals):
            if r >= r_max:
                c_r = c_max
            else:
                c_r = _isotropic_edge_factor_3d(xi, float(r), window.hull, rng)
            counts[k] += np.sum(dists < r) / c_r

    return (window.volume_nm3 / (n1 * n2)) * counts


def ripley_l12(k12: np.ndarray, r_vals: np.ndarray) -> np.ndarray:
    """3D standardized Ripley L₁₂: (3 K₁₂ / 4π)^(1/3) − r."""
    k12 = np.maximum(np.asarray(k12, dtype=float), 0.0)
    r_vals = np.asarray(r_vals, dtype=float)
    return np.cbrt(3.0 * k12 / (4.0 * np.pi)) - r_vals


def ripley_l12_from_points(
    x: np.ndarray,
    y: np.ndarray,
    r_vals: np.ndarray,
    window: RipleyWindow3D,
    rng: np.random.Generator,
) -> np.ndarray:
    return ripley_l12(cross_k12_3d_isotropic(x, y, r_vals, window, rng), r_vals)


def _ripley_r_grid(r_max_nm: float, r_step_nm: float) -> np.ndarray:
    n_steps = max(1, int(np.floor(r_max_nm / r_step_nm)))
    return np.arange(r_step_nm, r_max_nm + 0.5 * r_step_nm, r_step_nm, dtype=float)


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
        az_xyz = membrane_az_pairs[membrane]["active_zone_points"]
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
    base = aunp_meta.copy()
    for col_name, site_xyz in site_columns.items():
        site = np.asarray(site_xyz, dtype=float).reshape(3)
        base[col_name] = np.linalg.norm(aunp_coords - site, axis=1)
    return base


def _site_column_name(vesicle_name: str, suffix: str) -> str:
    safe = str(vesicle_name).replace(" ", "_")
    return f"{safe}__{suffix}"


def _percentile_band(curves: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if curves.size == 0:
        return np.array([]), np.array([]), np.array([])
    lo = np.nanpercentile(curves, RIPLEY_PERCENTILE_LO, axis=0)
    hi = np.nanpercentile(curves, RIPLEY_PERCENTILE_HI, axis=0)
    med = np.nanmean(curves, axis=0)
    return lo, med, hi


def _mean_sd_band(curves: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (mean − SD, mean, mean + SD, SD) across replicate curves at each r."""
    if curves.size == 0:
        empty = np.array([])
        return empty, empty, empty, empty
    mean = np.nanmean(curves, axis=0)
    if len(curves) > 1:
        sd = np.nanstd(curves, axis=0, ddof=1)
    else:
        sd = np.zeros_like(mean)
    return mean - sd, mean, mean + sd, sd


def _prism_sd_envelope_columns(
    curves: np.ndarray,
    r_vals: np.ndarray,
    *,
    prefix: str,
) -> dict[str, np.ndarray]:
    """SD summary columns for Prism tables (mean ± SD envelope)."""
    if len(curves):
        sd_lo, mean, sd_hi, sd = _mean_sd_band(curves)
    else:
        nan = np.full(len(r_vals), np.nan)
        sd_lo = mean = sd_hi = sd = nan
    return {
        f"{prefix}_mean": mean,
        f"{prefix}_sd": sd,
        f"{prefix}_sd_envelope_lo": sd_lo,
        f"{prefix}_sd_envelope_hi": sd_hi,
    }


def _fusing_mean_curve(obs_curves: np.ndarray, r_vals: np.ndarray) -> np.ndarray:
    if len(obs_curves) == 0:
        return np.full(len(r_vals), np.nan)
    return np.nanmean(obs_curves, axis=0)


def build_ripley_l12_prism_envelope_table(
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
    Pre-aggregated mean curves and control envelopes for graphing (e.g. Prism).

    Percentile envelopes use 2.5–97.5% across replicate curves. SD envelopes use
    mean ± 1 sample SD (per-vesicle curves for fusing/close; per-replicate for nulls).

    One row per (control_comparison, r_nm). ``fusing_L12_mean`` is identical within
    each comparison group (mean across fusing-vesicle curves).
    """
    fusing_sd = _prism_sd_envelope_columns(obs_curves, r_vals, prefix="fusing_L12")
    rows: list[dict] = []
    for comparison, control_curves in control_curves_by_comparison.items():
        if len(control_curves):
            ctrl_lo, ctrl_mean, ctrl_hi = _percentile_band(control_curves)
            ctrl_sd = _prism_sd_envelope_columns(control_curves, r_vals, prefix="control_L12")
            n_control = int(len(control_curves))
        else:
            ctrl_lo = ctrl_mean = ctrl_hi = np.full(len(r_vals), np.nan)
            ctrl_sd = _prism_sd_envelope_columns(np.empty((0, len(r_vals))), r_vals, prefix="control_L12")
            n_control = 0
        for i, r_nm in enumerate(r_vals):
            rows.append(
                {
                    "active_zone_name": zone_name,
                    "aunp_subset": aunp_subset,
                    "window_mode": window.defining_mode,
                    "control_comparison": comparison,
                    "r_nm": float(r_nm),
                    "fusing_L12_mean": float(fusing_sd["fusing_L12_mean"][i]),
                    "fusing_L12_sd": float(fusing_sd["fusing_L12_sd"][i]),
                    "fusing_L12_sd_envelope_lo": float(fusing_sd["fusing_L12_sd_envelope_lo"][i]),
                    "fusing_L12_sd_envelope_hi": float(fusing_sd["fusing_L12_sd_envelope_hi"][i]),
                    "control_L12_mean": float(ctrl_mean[i]),
                    "control_L12_sd": float(ctrl_sd["control_L12_sd"][i]),
                    "control_L12_envelope_lo": float(ctrl_lo[i]),
                    "control_L12_envelope_hi": float(ctrl_hi[i]),
                    "control_L12_sd_envelope_lo": float(ctrl_sd["control_L12_sd_envelope_lo"][i]),
                    "control_L12_sd_envelope_hi": float(ctrl_sd["control_L12_sd_envelope_hi"][i]),
                    "n_fusing_curves": int(len(obs_curves)),
                    "n_control_curves": n_control,
                    "n_aunp_partners": int(n_aunp_partners),
                    "envelope_percentile_lo": float(RIPLEY_PERCENTILE_LO),
                    "envelope_percentile_hi": float(RIPLEY_PERCENTILE_HI),
                    "window_volume_nm3": float(window.volume_nm3),
                }
            )
    return pd.DataFrame(rows)


def build_ripley_l12_prism_wide_table(
    prism_long: pd.DataFrame,
    *,
    id_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """
    Wide layout for Prism XY tables: one row per ``r_nm`` per window/comparison.

    Columns: r_nm, fusing/control means, percentile envelopes, mean ± SD envelopes (+ identifiers).
    """
    if prism_long.empty:
        return prism_long.copy()
    if id_cols is None:
        id_cols = ["active_zone_name", "aunp_subset", "window_mode", "control_comparison"]
    value_cols = [
        col
        for col in (
            "fusing_L12_mean",
            "fusing_L12_sd",
            "fusing_L12_sd_envelope_lo",
            "fusing_L12_sd_envelope_hi",
            "control_L12_mean",
            "control_L12_sd",
            "control_L12_envelope_lo",
            "control_L12_envelope_hi",
            "control_L12_sd_envelope_lo",
            "control_L12_sd_envelope_hi",
            "n_fusing_curves",
            "n_control_curves",
            "n_aunp_partners",
            "n_tomograms",
            "n_active_zones",
            "window_volume_nm3",
        )
        if col in prism_long.columns
    ]
    return prism_long[list(id_cols) + ["r_nm"] + value_cols].copy()


def _plot_ripley_control_comparison(
    r_vals: np.ndarray,
    obs_curves: np.ndarray,
    control_curves: np.ndarray,
    *,
    output_path: Path,
    title: str,
    ylabel: str = "Ripley L₁₂(r) = (3K₁₂/4π)^(1/3) − r",
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    obs_mean = np.nanmean(obs_curves, axis=0) if len(obs_curves) else np.full(len(r_vals), np.nan)
    ax.plot(r_vals, obs_mean, color="C3", lw=2.2, label="Fusing mean", zorder=5)

    if len(control_curves):
        lo, ctrl_mean, hi = _percentile_band(control_curves)
        ax.fill_between(r_vals, lo, hi, color="0.75", alpha=0.55, label="Control 95% envelope", zorder=2)
        ax.plot(r_vals, ctrl_mean, color="0.45", lw=2.0, label="Control mean", zorder=3)

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


def _per_point_l12_curves(
    points: np.ndarray,
    aunp_coords: np.ndarray,
    r_vals: np.ndarray,
    window: RipleyWindow3D,
    rng: np.random.Generator,
) -> np.ndarray:
    points = np.atleast_2d(np.asarray(points, dtype=float))
    if len(points) == 0:
        return np.empty((0, len(r_vals)))
    return np.vstack(
        [ripley_l12_from_points(pt.reshape(1, 3), aunp_coords, r_vals, window, rng) for pt in points]
    )


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
    rng: np.random.Generator,
    figures_dir: Path | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fusing_xyz = _fusion_xyz_from_rows(fusing_rows)
    close_xyz = _fusion_xyz_from_rows(close_rows)

    obs_curves = _per_point_l12_curves(fusing_xyz, aunp_coords, r_vals, window, rng)

    close_curves = _per_point_l12_curves(close_xyz, aunp_coords, r_vals, window, rng)

    shift_curves_list: list[np.ndarray] = []
    for rep_id in sorted(shift_by_replicate):
        pts = list(shift_by_replicate[rep_id].values())
        if not pts:
            continue
        shift_curves_list.append(
            ripley_l12_from_points(np.vstack(pts), aunp_coords, r_vals, window, rng)
        )
    shift_curves = np.vstack(shift_curves_list) if shift_curves_list else np.empty((0, len(r_vals)))

    perm_curves_list: list[np.ndarray] = []
    for perm_id in sorted(label_perm_pooled):
        pts = label_perm_pooled[perm_id]
        if not pts:
            continue
        perm_curves_list.append(
            ripley_l12_from_points(np.vstack(pts), aunp_coords, r_vals, window, rng)
        )
    perm_curves = np.vstack(perm_curves_list) if perm_curves_list else np.empty((0, len(r_vals)))

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
        )
        _plot_ripley_control_comparison(
            r_vals,
            obs_curves,
            shift_curves,
            output_path=figures_dir / f"ripley_l12_{mode_tag}_{subset_tag}_vs_40nm_shift.png",
            title=(
                f"{zone_name} | window={mode_tag} | AuNPs={subset_tag}\n"
                "Fusing vs 40 nm tangential shifts (100 replicates)"
            ),
        )
        _plot_ripley_control_comparison(
            r_vals,
            obs_curves,
            perm_curves,
            output_path=figures_dir / f"ripley_l12_{mode_tag}_{subset_tag}_vs_label_permutation.png",
            title=(
                f"{zone_name} | window={mode_tag} | AuNPs={subset_tag}\n"
                "Fusing vs label-permutation null (100 replicates)"
            ),
        )

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
                        "active_zone_name": zone_name,
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
    return pd.DataFrame(rows), prism_df


def _distance_csv_name(stem: str, subset: AunpSubset) -> str:
    return f"{stem}__{subset}.csv"


def _extract_curves_matrix(
    df: pd.DataFrame,
    curve_type: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Pivot long Ripley table to ``(r_vals, curves)`` with one row per replicate curve."""
    sub = df.loc[df["curve_type"] == curve_type]
    if sub.empty:
        r_vals = np.sort(df["r_nm"].unique()) if "r_nm" in df.columns and not df.empty else np.array([])
        return r_vals, np.empty((0, len(r_vals)))

    r_vals = np.sort(sub["r_nm"].unique())
    n_r = len(r_vals)
    id_cols = ["tomogram_name", "alignment_dir", "active_zone_name", "replicate_index"]
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
        curves.append(grp["ripley_l12"].to_numpy(dtype=float))
    if not curves:
        return r_vals, np.empty((0, n_r))
    return r_vals, np.vstack(curves)


def build_pooled_ripley_l12_prism_envelope_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pooled mean curves and control envelopes across all tomograms/zones.

    ``fusing_L12_mean`` is the mean across all pooled fusing-vesicle curves at each r.
    Percentile envelopes use 2.5–97.5%; SD envelopes use mean ± 1 sample SD.
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

            fusing_sd = _prism_sd_envelope_columns(obs_curves, r_vals, prefix="fusing_L12")
            n_tomograms = int(sub_df["tomogram_name"].nunique())
            n_zones = int(
                sub_df[["tomogram_name", "alignment_dir", "active_zone_name"]]
                .drop_duplicates()
                .shape[0]
            )

            for comparison in CONTROL_COMPARISONS:
                _, control_curves = _extract_curves_matrix(
                    sub_df, CONTROL_CURVE_TYPE[comparison]
                )
                if len(control_curves):
                    ctrl_lo, ctrl_mean, ctrl_hi = _percentile_band(control_curves)
                    ctrl_sd = _prism_sd_envelope_columns(
                        control_curves, r_vals, prefix="control_L12"
                    )
                    n_control = int(len(control_curves))
                else:
                    ctrl_lo = ctrl_mean = ctrl_hi = np.full(len(r_vals), np.nan)
                    ctrl_sd = _prism_sd_envelope_columns(
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
                            "control_L12_mean": float(ctrl_mean[i]),
                            "control_L12_sd": float(ctrl_sd["control_L12_sd"][i]),
                            "control_L12_envelope_lo": float(ctrl_lo[i]),
                            "control_L12_envelope_hi": float(ctrl_hi[i]),
                            "control_L12_sd_envelope_lo": float(
                                ctrl_sd["control_L12_sd_envelope_lo"][i]
                            ),
                            "control_L12_sd_envelope_hi": float(
                                ctrl_sd["control_L12_sd_envelope_hi"][i]
                            ),
                            "n_fusing_curves": int(len(obs_curves)),
                            "n_control_curves": n_control,
                            "n_tomograms": n_tomograms,
                            "n_active_zones": n_zones,
                            "envelope_percentile_lo": float(RIPLEY_PERCENTILE_LO),
                            "envelope_percentile_hi": float(RIPLEY_PERCENTILE_HI),
                        }
                    )
    return pd.DataFrame(rows)


def plot_pooled_fusion_point_aunp_ripley_l12_visualizations(
    curves_csv: Path | str = POOLED_RIPLEY_CURVES_CSV,
    output_dir: Path | str = POOLED_RIPLEY_FIGURES_DIR,
    prism_csv: Path | str = POOLED_RIPLEY_PRISM_CSV,
    prism_wide_csv: Path | str = POOLED_RIPLEY_PRISM_WIDE_CSV,
) -> list[Path]:
    """
    Pool individual Ripley curves across all tomograms/zones.

    Writes PNGs (fusing mean vs control mean + 95% envelope), a long Prism CSV,
    and a wide Prism CSV.
    """
    curves_csv = Path(curves_csv)
    output_dir = Path(output_dir)
    prism_csv = Path(prism_csv)
    prism_wide_csv = Path(prism_wide_csv)
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
    build_ripley_l12_prism_wide_table(
        prism_long,
        id_cols=["aunp_subset", "window_mode", "control_comparison"],
    ).to_csv(prism_wide_csv, index=False)
    print(
        f"Pooled fusion-point/AuNP Ripley L₁₂ Prism tables "
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
                f"{int(meta_row['n_active_zones'])} zone(s) | "
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


def run_fusion_point_aunp_analyses_for_zone(
    tomogram_path: Path,
    alignment_dir: str,
    zone_name: str,
    active_zone_index: int,
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
) -> dict[str, Path] | None:
    tomogram_path = Path(tomogram_path)
    alignment_dir = require_alignment_dir(alignment_dir)
    tomogram_name = tomogram_path.name

    membrane_name = presynaptic_membrane_name_for_zone(zone_name)
    membrane_az_pairs = import_presynaptic_membranes_and_active_zones(tomogram_path, alignment_dir=alignment_dir)

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
            active_zone_index,
            monomer_star_pattern=monomer_star_pattern,
            dimer_star_pattern=dimer_star_pattern,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(
            f"  Skipping fusion-point/AuNP analyses for {zone_name}: {exc}"
        )
        return None

    aunp_coords_all = loaded.coords
    aunp_meta = loaded.meta
    subsets_to_run = available_aunp_subsets(loaded.kinds_loaded)
    if len(loaded.kinds_loaded) == 1:
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
    hull_defining_coords = build_fusion_aunp_hull_defining_coords(
        fusing_xyz, close_xyz, aunp_coords_all
    )
    ripley_windows: dict[WindowMode, RipleyWindow3D | None] = {
        mode: None for mode in RIPLEY_WINDOW_MODES
    }

    try:
        ripley_windows["fusion_aunp_hull"] = build_ripley_window_3d(
            hull_defining_coords, "fusion_aunp_hull"
        )
    except ValueError as exc:
        print(f"  Ripley fusion+AuNP hull window skipped for {zone_name}: {exc}")

    try:
        cleft_coords = load_synaptic_cleft_active_zone_points(
            tomogram_path, alignment_dir, zone_name
        )
        ripley_windows["synaptic_cleft_az_hull"] = build_ripley_window_3d(
            cleft_coords, "synaptic_cleft_az_hull"
        )
    except (ValueError, FileNotFoundError) as exc:
        cleft_coords = None
        print(f"  Ripley synaptic-cleft AZ hull window skipped for {zone_name}: {exc}")

    # --- Distance wide CSVs (per AuNP subset) ---
    original_cols: dict[str, np.ndarray] = {}
    for fp in fusing_rows:
        original_cols[_site_column_name(fp["vesicle_name"], "original")] = np.array(
            [fp["fusion_point_x_nm"], fp["fusion_point_y_nm"], fp["fusion_point_z_nm"]],
            dtype=float,
        )

    shift_by_rep = _shift_sites_by_replicate(
        fusing_rows,
        membrane_az_pairs,
        offset_nm=FUSION_POINT_SHIFT_OFFSET_NM,
        n_shifts=n_replicates,
        rng=rng,
        max_snap_distance_nm=FUSION_POINT_AZ_MAX_SNAP_DISTANCE_NM,
    )
    shift_cols: dict[str, np.ndarray] = {}
    for rep_id, ves_map in shift_by_rep.items():
        for fp in fusing_rows:
            vid = int(fp["vesicle_id"])
            if vid not in ves_map:
                continue
            shift_cols[_site_column_name(fp["vesicle_name"], f"shift_{rep_id:03d}")] = ves_map[vid]

    # Label permutation uses the full monomer+dimer pool; same null sites for all subsets.
    label_per_ves, label_pooled = _label_permutation_sites(
        fusing_rows,
        aunp_coords_all,
        pre_surface,
        n_perm=n_replicates,
        rng=rng,
    )
    label_cols: dict[str, np.ndarray] = {}
    for perm_id, ves_map in label_per_ves.items():
        for fp in fusing_rows:
            vid = int(fp["vesicle_id"])
            if vid not in ves_map:
                continue
            label_cols[_site_column_name(fp["vesicle_name"], f"perm_{perm_id:03d}")] = ves_map[vid]

    distance_paths: dict[str, dict[str, Path]] = {subset: {} for subset in subsets_to_run}
    for subset in subsets_to_run:
        sub_coords, sub_meta = subset_aunps(aunp_meta, subset=subset)
        meta_out = sub_meta.copy()
        meta_out.insert(0, "aunp_subset", subset)
        meta_out.insert(1, "tomogram_name", tomogram_name)
        meta_out.insert(2, "alignment_dir", alignment_dir)
        meta_out.insert(3, "active_zone_name", zone_name)

        df_orig = build_distance_wide_csv(sub_coords, meta_out, original_cols)
        df_shift = build_distance_wide_csv(sub_coords, meta_out, shift_cols)
        df_perm = build_distance_wide_csv(sub_coords, meta_out, label_cols)

        p_orig = out_dir / _distance_csv_name("distances_original_fusing_wide", subset)
        p_shift = out_dir / _distance_csv_name("distances_40nm_shift_wide", subset)
        p_perm = out_dir / _distance_csv_name("distances_label_permutation_wide", subset)
        df_orig.to_csv(p_orig, index=False)
        df_shift.to_csv(p_shift, index=False)
        df_perm.to_csv(p_perm, index=False)
        distance_paths[subset] = {"original": p_orig, "shift": p_shift, "permutation": p_perm}

        if len(sub_coords) == 0:
            print(f"  No {subset} AuNPs for {zone_name}; distance CSVs written empty")

    # --- Ripley: both window modes × three AuNP partner subsets ---
    ripley_frames: list[pd.DataFrame] = []
    prism_frames: list[pd.DataFrame] = []

    for window_mode in RIPLEY_WINDOW_MODES:
        window = ripley_windows.get(window_mode)
        if window is None:
            continue
        for subset in subsets_to_run:
            sub_coords, _ = subset_aunps(aunp_meta, subset=subset)
            if len(sub_coords) == 0:
                print(f"  Skipping Ripley for {zone_name} ({subset}, {window_mode}): no partner AuNPs")
                continue
            df_r, df_prism = run_ripley_for_zone_window(
                zone_name=zone_name,
                aunp_subset=subset,
                window=window,
                aunp_coords=sub_coords,
                fusing_rows=fusing_rows,
                close_rows=close_rows,
                shift_by_replicate=shift_by_rep,
                label_perm_pooled=label_pooled,
                r_vals=r_vals,
                rng=rng,
                figures_dir=figures_dir if write_figures else None,
            )
            ripley_frames.append(df_r)
            if not df_prism.empty:
                df_prism = df_prism.copy()
                df_prism.insert(0, "tomogram_name", tomogram_name)
                df_prism.insert(1, "alignment_dir", alignment_dir)
                prism_frames.append(df_prism)

    if not any(ripley_windows.values()):
        print(f"  Skipping Ripley for {zone_name}: no valid hull windows")

    if ripley_frames:
        ripley_long = pd.concat(ripley_frames, ignore_index=True)
        ripley_long.insert(0, "tomogram_name", tomogram_name)
        ripley_long.insert(1, "alignment_dir", alignment_dir)
        ripley_long.to_csv(out_dir / "ripley_l12_curves.csv", index=False)
    if prism_frames:
        prism_long = pd.concat(prism_frames, ignore_index=True)
        prism_long.to_csv(out_dir / "ripley_l12_prism_envelopes.csv", index=False)
        build_ripley_l12_prism_wide_table(prism_long).to_csv(
            out_dir / "ripley_l12_prism_envelopes_wide.csv",
            index=False,
        )

    n_monomer = int((aunp_meta["aunp_kind"] == "monomer").sum())
    n_dimer = int((aunp_meta["aunp_kind"] == "dimer").sum())
    meta = {
        "tomogram_name": tomogram_name,
        "alignment_dir": alignment_dir,
        "active_zone_name": zone_name,
        "active_zone_index": int(active_zone_index),
        "n_fusing_vesicles": len(fusing_rows),
        "n_close_vesicles": len(close_rows),
        "n_aunp_monomer": n_monomer,
        "n_aunp_dimer": n_dimer,
        "n_aunp_monomer_dimer": len(aunp_coords_all),
        "aunp_kinds_loaded": list(loaded.kinds_loaded),
        "aunp_subsets_analyzed": list(subsets_to_run),
        "ripley_window_modes": list(RIPLEY_WINDOW_MODES),
        "ripley_windows": {
            mode: {
                "defining_from": (
                    "fusing_and_close_fusion_sites_plus_monomer_and_dimer_aunps"
                    if mode == "fusion_aunp_hull"
                    else "presynaptic_and_postsynaptic_active_zone_surface_points"
                ),
                "volume_nm3": float(win.volume_nm3),
                "n_defining_points": (
                    int(len(hull_defining_coords))
                    if mode == "fusion_aunp_hull"
                    else int(len(cleft_coords))
                    if cleft_coords is not None
                    else None
                ),
            }
            for mode, win in ripley_windows.items()
            if win is not None
        },
        "n_hull_fusing_fusion_sites": int(len(fusing_xyz)),
        "n_hull_close_fusion_sites": int(len(close_xyz)),
        "n_hull_aunp_monomer_dimer": int(len(aunp_coords_all)),
        "n_shift_replicates": int(n_replicates),
        "n_label_permutations": int(n_replicates),
        "seed": int(seed),
        "ripley_edge_correction": "isotropic_3d_mc",
        "window_volume_definition": "convex_hull_volume",
    }
    with open(out_dir / "run_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(
        f"  Fusion-point/AuNP 3D analyses ({zone_name}): "
        f"{len(fusing_rows)} fusing, {n_monomer} monomer + {n_dimer} dimer AuNPs -> {out_dir}"
    )
    return {
        "distance_paths": distance_paths,
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
) -> dict[str, pd.DataFrame]:
    """
    Long-form 40 nm shift and label-permutation query sites for zonogram overlays.

    Uses the same 3D geometry, monomer+dimer AuNP pool, replicate count, and seed as
  ``run_fusion_point_aunp_analyses_for_zone``.
    """
    tomogram_path = Path(tomogram_path)
    alignment_dir = require_alignment_dir(alignment_dir)
    az_mapping = {int(k): v for k, v in (az_mapping or {}).items()}
    if not az_mapping:
        return {"40nm_shift": pd.DataFrame(), "label_permutation": pd.DataFrame()}

    membrane_az_pairs = import_presynaptic_membranes_and_active_zones(
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
                        "active_zone_name": zone_name,
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
                        "active_zone_name": zone_name,
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
    active_zone_indices: Sequence[int] | None = None,
    vesicle_distance_threshold: float = 20.0,
    fusion_point_threshold: float = 20.0,
    n_replicates: int = DEFAULT_NULL_REPLICATES_N,
    seed: int = DEFAULT_ANALYSIS_SEED,
    write_figures: bool = True,
    monomer_star_pattern: Optional[str] = None,
    dimer_star_pattern: Optional[str] = None,
) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    from .activezone import load_active_zone_mapping

    tomogram_path = Path(tomogram_path)
    alignment_dir = require_alignment_dir(alignment_dir)
    az_mapping = load_active_zone_mapping(tomogram_path, alignment_dir) or {}
    if not az_mapping:
        print("No active zone mapping; skipping fusion-point/AuNP 3D analyses")
        return [], []

    az_mapping = {int(k): v for k, v in az_mapping.items()}
    indices = list(active_zone_indices) if active_zone_indices is not None else sorted(az_mapping)
    ripley_frames: list[pd.DataFrame] = []
    prism_frames: list[pd.DataFrame] = []

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

    return ripley_frames, prism_frames
