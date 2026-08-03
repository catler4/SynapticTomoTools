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

Vesicle–AuNP distance tables are also written in a Prism-friendly form with one column
per vesicle (or vesicle×simulation), distances listed down the column, for fusing,
close, 40 nm-shifted, and label-permutation sites. Per-zone files live under
``STT_results/aunps/fusion_point_aunp_analyses/<zone>/``; pooled files under
``results/aunps/fusion_point_aunp_distances_*_columns_pooled.csv``.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
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
WINDOW_VOLUME_MC_SAMPLES = 200_000

ANALYSES_SUBDIR = "fusion_point_aunp_analyses"
COORD_COLS = ("faCoordinateX", "faCoordinateY", "faCoordinateZ")

POOLED_RIPLEY_CURVES_CSV = Path("results/aunps/fusion_point_aunp_ripley_l12_curves.csv")
POOLED_RIPLEY_PRISM_CSV = Path("results/aunps/fusion_point_aunp_ripley_l12_prism_envelopes_pooled.csv")
POOLED_RIPLEY_PRISM_WIDE_CSV = Path("results/aunps/fusion_point_aunp_ripley_l12_prism_envelopes_pooled_wide.csv")
POOLED_RIPLEY_FIGURES_DIR = Path("results/aunps/figures/fusion_point_aunp_ripley_l12_pooled")

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
DISTANCE_COLUMNS_ONLY_STEMS = {
    "fusing": "distances_fusing_columns_only",
    "close": "distances_close_columns_only",
    "shift_40nm": "distances_40nm_shift_columns_only",
    "label_permutation": "distances_label_permutation_columns_only",
}

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
    """3D convex-hull window; ``volume_nm3`` is the hull volume (or hull ∩ betweenness-region
    volume when ``use_angle_betweenness`` is set)."""

    volume_nm3: float
    hull: ConvexHull
    defining_mode: WindowMode
    pre_membrane_coords: np.ndarray | None = None
    post_membrane_coords: np.ndarray | None = None
    use_angle_betweenness: bool = False


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
    *,
    pre_membrane_coords: np.ndarray | None = None,
    post_membrane_coords: np.ndarray | None = None,
    use_angle_betweenness: bool = False,
    rng: np.random.Generator | None = None,
) -> RipleyWindow3D:
    defining_coords = np.atleast_2d(np.asarray(defining_coords, dtype=float))
    if len(defining_coords) < 4:
        raise ValueError(f"Need at least 4 points to build a 3D convex hull ({mode}), got {len(defining_coords)}")
    hull = ConvexHull(defining_coords)
    if hull.volume <= 0:
        raise ValueError(f"Convex hull volume must be positive ({mode})")

    if use_angle_betweenness:
        pre_membrane_coords = np.atleast_2d(np.asarray(pre_membrane_coords, dtype=float)) if pre_membrane_coords is not None else np.zeros((0, 3))
        post_membrane_coords = np.atleast_2d(np.asarray(post_membrane_coords, dtype=float)) if post_membrane_coords is not None else np.zeros((0, 3))
        if len(pre_membrane_coords) == 0 or len(post_membrane_coords) == 0:
            raise ValueError(
                f"use_angle_betweenness requires non-empty pre/post membrane coords ({mode})"
            )
        volume = _hull_betweenness_volume_mc(
            hull, pre_membrane_coords, post_membrane_coords, rng or np.random.default_rng(0)
        )
        if volume <= 0:
            raise ValueError(f"Hull ∩ betweenness-region volume must be positive ({mode})")
    else:
        volume = float(hull.volume)

    return RipleyWindow3D(
        volume_nm3=volume,
        hull=hull,
        defining_mode=mode,
        pre_membrane_coords=pre_membrane_coords,
        post_membrane_coords=post_membrane_coords,
        use_angle_betweenness=use_angle_betweenness,
    )


def _points_inside_hull(pts: np.ndarray, hull: ConvexHull, tol: float = 1e-6) -> np.ndarray:
    pts = np.atleast_2d(np.asarray(pts, dtype=float))
    if len(pts) == 0:
        return np.zeros(0, dtype=bool)
    normals = hull.equations[:, :-1]
    offsets = hull.equations[:, -1]
    return np.all(pts @ normals.T + offsets <= tol, axis=1)


def _angle_betweenness_mask(
    points: np.ndarray,
    pre_membrane_coords: np.ndarray,
    post_membrane_coords: np.ndarray,
) -> np.ndarray:
    """
    True where a point sits "between" the two membranes: the vector from the point to
    its nearest presynaptic-membrane point and the vector to its nearest
    postsynaptic-membrane point face away from each other (angle > 90 deg, i.e. the
    two vectors have a negative dot product).

    Optional Ripley-window refinement: when ``RipleyWindow3D.use_angle_betweenness`` is
    set, this mask is ANDed with hull membership (see ``_window_contains``) to define the
    effective window used for volume normalization and edge correction.
    """
    points = np.atleast_2d(np.asarray(points, dtype=float))
    pre_membrane_coords = np.atleast_2d(np.asarray(pre_membrane_coords, dtype=float))
    post_membrane_coords = np.atleast_2d(np.asarray(post_membrane_coords, dtype=float))
    if len(points) == 0:
        return np.zeros(0, dtype=bool)
    if len(pre_membrane_coords) == 0 or len(post_membrane_coords) == 0:
        return np.zeros(len(points), dtype=bool)

    pre_tree = cKDTree(pre_membrane_coords)
    post_tree = cKDTree(post_membrane_coords)
    _, pre_idx = pre_tree.query(points)
    _, post_idx = post_tree.query(points)
    v_pre = pre_membrane_coords[pre_idx] - points
    v_post = post_membrane_coords[post_idx] - points
    dot = np.einsum("ij,ij->i", v_pre, v_post)
    return dot < 0.0


def _window_contains(window: RipleyWindow3D, pts: np.ndarray) -> np.ndarray:
    """Point membership in the effective Ripley window: inside the hull, additionally
    restricted to the angle-betweenness region when ``window.use_angle_betweenness``."""
    pts = np.atleast_2d(np.asarray(pts, dtype=float))
    inside = _points_inside_hull(pts, window.hull)
    if not window.use_angle_betweenness:
        return inside
    mask = inside.copy()
    if np.any(inside):
        mask[inside] = _angle_betweenness_mask(
            pts[inside], window.pre_membrane_coords, window.post_membrane_coords
        )
    return mask


def _hull_betweenness_volume_mc(
    hull: ConvexHull,
    pre_membrane_coords: np.ndarray,
    post_membrane_coords: np.ndarray,
    rng: np.random.Generator,
    *,
    n_samples: int = WINDOW_VOLUME_MC_SAMPLES,
) -> float:
    """Bounding-box Monte Carlo estimate of the hull ∩ betweenness-region volume."""
    mins = hull.points.min(axis=0)
    maxs = hull.points.max(axis=0)
    box_volume = float(np.prod(maxs - mins))
    if box_volume <= 0:
        return 0.0
    candidates = rng.uniform(mins, maxs, size=(int(n_samples), 3))
    inside_hull = _points_inside_hull(candidates, hull)
    mask = inside_hull.copy()
    if np.any(inside_hull):
        mask[inside_hull] = _angle_betweenness_mask(
            candidates[inside_hull], pre_membrane_coords, post_membrane_coords
        )
    return box_volume * float(np.mean(mask))


def _downsample_points(pts: np.ndarray, n_max: int, seed: int = 0) -> np.ndarray:
    pts = np.atleast_2d(np.asarray(pts, dtype=float))
    if len(pts) <= n_max:
        return pts
    idx = np.random.default_rng(seed).choice(len(pts), n_max, replace=False)
    return pts[idx]


def _sample_uniform_points_in_hull(
    hull: ConvexHull,
    n_points: int,
    rng: np.random.Generator,
    *,
    max_tries: int = 60,
) -> np.ndarray:
    """Rejection-sample points approximately uniformly inside a convex hull's volume.

    QC-plot utility only (visualizes what fraction of the hull volume a candidate
    edge-correction test would keep/reject) — not used by the production Ripley window.
    """
    mins = hull.points.min(axis=0)
    maxs = hull.points.max(axis=0)
    accepted: list[np.ndarray] = []
    n_accepted = 0
    tries = 0
    while n_accepted < n_points and tries < max_tries:
        batch = max(n_points * 4, 512)
        candidates = rng.uniform(mins, maxs, size=(batch, 3))
        inside = _points_inside_hull(candidates, hull)
        if np.any(inside):
            accepted.append(candidates[inside])
            n_accepted += int(np.sum(inside))
        tries += 1
    if not accepted:
        return np.zeros((0, 3), dtype=float)
    out = np.vstack(accepted)
    return out[:n_points]


def _unit_ball_offsets(
    rng: np.random.Generator,
    *,
    n_samples: int = EDGE_MC_SAMPLES,
) -> np.ndarray:
    """Uniform samples in the unit ball as (n_samples, 3) offsets."""
    dirs = rng.normal(size=(n_samples, 3))
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    dirs /= norms
    radii = rng.random(n_samples) ** (1.0 / 3.0)
    return dirs * radii[:, None]


def _isotropic_edge_factor_3d(
    center: np.ndarray,
    radius_nm: float,
    window: RipleyWindow3D,
    rng: np.random.Generator,
    *,
    n_samples: int = EDGE_MC_SAMPLES,
    unit_offsets: np.ndarray | None = None,
) -> float:
    if radius_nm <= 0:
        return 1.0
    center = np.asarray(center, dtype=float).reshape(3)
    if unit_offsets is None:
        unit_offsets = _unit_ball_offsets(rng, n_samples=n_samples)
    samples = center + unit_offsets * float(radius_nm)
    inside = _window_contains(window, samples)
    frac = float(np.mean(inside))
    return max(frac, EDGE_MIN_C)


def _isotropic_edge_factors_for_foci(
    centers: np.ndarray,
    r_vals: np.ndarray,
    window: RipleyWindow3D,
    rng: np.random.Generator,
    *,
    n_samples: int = EDGE_MC_SAMPLES,
) -> np.ndarray:
    """
    Vectorized isotropic edge factors for all foci × all radii.

    Returns array shaped ``(n_foci, n_r)``.
    """
    centers = np.atleast_2d(np.asarray(centers, dtype=float))
    r_vals = np.asarray(r_vals, dtype=float)
    n1 = len(centers)
    n_r = len(r_vals)
    if n1 == 0 or n_r == 0:
        return np.zeros((n1, n_r), dtype=float)

    unit_offsets = _unit_ball_offsets(rng, n_samples=n_samples)  # (M, 3)
    # Process per radius to keep memory bounded: (n1, M, 3)
    factors = np.empty((n1, n_r), dtype=float)
    for k, r in enumerate(r_vals):
        r = float(r)
        if r <= 0:
            factors[:, k] = 1.0
            continue
        samples = centers[:, None, :] + unit_offsets[None, :, :] * r
        flat = samples.reshape(-1, 3)
        inside = _window_contains(window, flat).reshape(n1, n_samples)
        frac = np.mean(inside, axis=1)
        factors[:, k] = np.maximum(frac, EDGE_MIN_C)
    return factors


def cross_k12_3d_isotropic(
    x: np.ndarray,
    y: np.ndarray,
    r_vals: np.ndarray,
    window: RipleyWindow3D,
    rng: np.random.Generator,
    *,
    edge_factors: np.ndarray | None = None,
) -> np.ndarray:
    """
    Edge-corrected bivariate cross-K in 3D.

    Type-1 foci ``x`` (fusion / controls); type-2 ``y`` (AuNPs).
    Neighbor counts are vectorized across radii; edge factors are batched over foci.

    ``edge_factors`` may supply a precomputed ``(n1, n_r)`` isotropic edge-correction
    matrix for the foci (skips the Monte-Carlo estimate); useful when the same physical
    points are reused across many label permutations.
    """
    x = np.atleast_2d(np.asarray(x, dtype=float))
    y = np.atleast_2d(np.asarray(y, dtype=float))
    r_vals = np.asarray(r_vals, dtype=float)
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0 or window.volume_nm3 <= 0:
        return np.full(len(r_vals), np.nan)

    tree = cKDTree(y)
    r_max = float(r_vals[-1])
    if edge_factors is None:
        edge_factors = _isotropic_edge_factors_for_foci(x, r_vals, window, rng)
    else:
        edge_factors = np.asarray(edge_factors, dtype=float)
        if edge_factors.shape != (n1, len(r_vals)):
            raise ValueError(
                f"edge_factors shape {edge_factors.shape} != expected {(n1, len(r_vals))}"
            )

    counts = np.zeros(len(r_vals), dtype=float)
    # Batch neighbor queries for all foci at r_max.
    neighbor_lists = tree.query_ball_point(x, r=r_max)
    for i, neighbor_idx in enumerate(neighbor_lists):
        if not neighbor_idx:
            continue
        dists = np.linalg.norm(y[np.asarray(neighbor_idx, dtype=int)] - x[i], axis=1)
        # Vectorized cumulative counts over the r grid.
        # (n_neighbors, n_r)
        within = dists[:, None] < r_vals[None, :]
        neighbor_counts = within.sum(axis=0).astype(float)
        counts += neighbor_counts / edge_factors[i]

    return (window.volume_nm3 / (n1 * n2)) * counts


def ripley_l12(k12: np.ndarray, r_vals: np.ndarray) -> np.ndarray:
    """3D standardized Ripley L₁₂: (3 K₁₂ / 4π)^(1/3) − r."""
    k12 = np.maximum(np.asarray(k12, dtype=float), 0.0)
    r_vals = np.asarray(r_vals, dtype=float)
    return np.cbrt(3.0 * k12 / (4.0 * np.pi)) - r_vals


def ripley_k12_from_l12(l12: np.ndarray, r_vals: np.ndarray) -> np.ndarray:
    """Invert L₁₂ → K₁₂: K = (4π/3)·(L+r)³ (non-negative radius argument)."""
    l12 = np.asarray(l12, dtype=float)
    r_vals = np.asarray(r_vals, dtype=float)
    return (4.0 * np.pi / 3.0) * np.maximum(l12 + r_vals, 0.0) ** 3


def _k12_curves_matrix(
    l12_curves: np.ndarray,
    r_vals: np.ndarray,
    *,
    k12_curves: np.ndarray | None = None,
) -> np.ndarray:
    """Return (n_curves, n_r) K₁₂ matrix, inverting from L₁₂ when needed."""
    r_vals = np.asarray(r_vals, dtype=float)
    if k12_curves is not None:
        return np.atleast_2d(np.asarray(k12_curves, dtype=float))
    l_mat = np.atleast_2d(np.asarray(l12_curves, dtype=float))
    if l_mat.size == 0 or l_mat.shape[0] == 0:
        return np.empty((0, len(r_vals)))
    return ripley_k12_from_l12(l_mat, r_vals[None, :])


def mean_l12_from_averaged_k12(
    l12_curves: np.ndarray,
    r_vals: np.ndarray,
    *,
    k12_curves: np.ndarray | None = None,
) -> np.ndarray:
    """
    Pool on the K₁₂ scale, then convert the mean K back to L₁₂.

    If ``k12_curves`` is provided it is averaged directly; otherwise each L₁₂ curve is
    inverted to K₁₂ first. Empty input yields an all-NaN L₁₂ vector.
    """
    return prism_sd_envelope_columns_from_averaged_k12(
        l12_curves, r_vals, prefix="tmp", k12_curves=k12_curves
    )["tmp_mean_from_k"]


def prism_sd_envelope_columns_from_averaged_k12(
    l12_curves: np.ndarray,
    r_vals: np.ndarray,
    *,
    prefix: str,
    k12_curves: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """
    Mean ± SD/SEM on the K₁₂ scale, then map those three curves back to L₁₂.

    Because L₁₂(K) is nonlinear, these envelopes differ from mean±SD computed on L₁₂.
    Percentile envelopes of L and of K→L are identical (monotone transform), so they are
    not duplicated here.

    Columns: ``{prefix}_mean_from_k``, ``{prefix}_sd_from_k`` (SD of K),
    ``{prefix}_sd_envelope_{lo,hi}_from_k``, ``{prefix}_sem_from_k`` (SEM of K),
    ``{prefix}_sem_envelope_{lo,hi}_from_k``.
    """
    r_vals = np.asarray(r_vals, dtype=float)
    nan = np.full(len(r_vals), np.nan)
    empty = {
        f"{prefix}_mean_from_k": nan.copy(),
        f"{prefix}_sd_from_k": nan.copy(),
        f"{prefix}_sd_envelope_lo_from_k": nan.copy(),
        f"{prefix}_sd_envelope_hi_from_k": nan.copy(),
        f"{prefix}_sem_from_k": nan.copy(),
        f"{prefix}_sem_envelope_lo_from_k": nan.copy(),
        f"{prefix}_sem_envelope_hi_from_k": nan.copy(),
    }
    k_mat = _k12_curves_matrix(l12_curves, r_vals, k12_curves=k12_curves)
    if k_mat.size == 0 or k_mat.shape[0] == 0:
        return empty

    mean_k = np.nanmean(k_mat, axis=0)
    if k_mat.shape[0] > 1:
        sd_k = np.nanstd(k_mat, axis=0, ddof=1)
        sem_k = sd_k / np.sqrt(float(k_mat.shape[0]))
    else:
        sd_k = np.zeros_like(mean_k)
        sem_k = np.zeros_like(mean_k)

    return {
        f"{prefix}_mean_from_k": ripley_l12(mean_k, r_vals),
        f"{prefix}_sd_from_k": sd_k,
        f"{prefix}_sd_envelope_lo_from_k": ripley_l12(mean_k - sd_k, r_vals),
        f"{prefix}_sd_envelope_hi_from_k": ripley_l12(mean_k + sd_k, r_vals),
        f"{prefix}_sem_from_k": sem_k,
        f"{prefix}_sem_envelope_lo_from_k": ripley_l12(mean_k - sem_k, r_vals),
        f"{prefix}_sem_envelope_hi_from_k": ripley_l12(mean_k + sem_k, r_vals),
    }


def ripley_l12_from_points(
    x: np.ndarray,
    y: np.ndarray,
    r_vals: np.ndarray,
    window: RipleyWindow3D,
    rng: np.random.Generator,
    *,
    edge_factors: np.ndarray | None = None,
) -> np.ndarray:
    return ripley_l12(
        cross_k12_3d_isotropic(x, y, r_vals, window, rng, edge_factors=edge_factors),
        r_vals,
    )


def ripley_l12_curves_per_focus(
    x: np.ndarray,
    y: np.ndarray,
    r_vals: np.ndarray,
    window: RipleyWindow3D,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Per-focus L₁₂ curves with shared edge-factor Monte Carlo and batched counts.

    Returns ``(n_foci, n_r)``. Each row is the univariate-focus L₁₂ using partners ``y``.
    """
    x = np.atleast_2d(np.asarray(x, dtype=float))
    y = np.atleast_2d(np.asarray(y, dtype=float))
    r_vals = np.asarray(r_vals, dtype=float)
    n1, n2 = len(x), len(y)
    if n1 == 0:
        return np.empty((0, len(r_vals)))
    if n2 == 0 or window.volume_nm3 <= 0:
        return np.full((n1, len(r_vals)), np.nan)

    tree = cKDTree(y)
    r_max = float(r_vals[-1])
    edge_factors = _isotropic_edge_factors_for_foci(x, r_vals, window, rng)
    neighbor_lists = tree.query_ball_point(x, r=r_max)
    k12 = np.zeros((n1, len(r_vals)), dtype=float)
    scale = window.volume_nm3 / float(n2)
    for i, neighbor_idx in enumerate(neighbor_lists):
        if not neighbor_idx:
            continue
        dists = np.linalg.norm(y[np.asarray(neighbor_idx, dtype=int)] - x[i], axis=1)
        neighbor_counts = (dists[:, None] < r_vals[None, :]).sum(axis=0).astype(float)
        k12[i] = scale * (neighbor_counts / edge_factors[i])
    return ripley_l12(k12, r_vals)


MAD_MIN_NULL_CURVES = 1000
MAD_CONFIDENCE = 0.99
# (label, r_min_nm or None, r_max_nm or None). None → use the full Ripley grid.
MAD_R_RANGES: tuple[tuple[str, float | None, float | None], ...] = (
    ("full", None, None),
    ("30-50nm", 30.0, 50.0),
)


def mad_test_from_curves(
    observed: np.ndarray,
    null_curves: np.ndarray,
    r_vals: np.ndarray,
    *,
    confidence: float = MAD_CONFIDENCE,
    min_null_curves: int = MAD_MIN_NULL_CURVES,
    null_name: str = "null",
    r_min_nm: float | None = None,
    r_max_nm: float | None = None,
    r_range: str = "full",
) -> dict:
    """
    Maximum Absolute Deviation (MAD) test vs a Monte Carlo null (Rebola / Diggle style).

    Skips (``status='skipped_insufficient_nulls'``) unless ``null_curves`` has at least
    ``min_null_curves`` replicates. Uses a two-sided ``confidence`` envelope (default 99%).

    The reference mean μ(r) is the pooled Diggle mean of the observed curve plus all
    null curves: μ = (L_obs + Σ L_s) / (N + 1). Both T_obs and each null MAD T_s are
    measured against that same μ (not leave-one-out). Pointwise CE percentiles remain
    null-only.

    Optional ``r_min_nm`` / ``r_max_nm`` restrict the max-|diff| search (inclusive) to that
    radius window; CE / pooled mean / normalized curves are reported on the same subset.
    """
    observed = np.asarray(observed, dtype=float).reshape(-1)
    null_curves = np.atleast_2d(np.asarray(null_curves, dtype=float))
    r_vals_full = np.asarray(r_vals, dtype=float)
    n_null = int(null_curves.shape[0]) if null_curves.size else 0
    alpha = 1.0 - float(confidence)
    lo_pct = 100.0 * (alpha / 2.0)
    hi_pct = 100.0 * (1.0 - alpha / 2.0)

    result = {
        "null_name": null_name,
        "r_range": str(r_range),
        "r_min_nm": float(r_min_nm) if r_min_nm is not None else np.nan,
        "r_max_nm": float(r_max_nm) if r_max_nm is not None else np.nan,
        "status": "ok",
        "n_null_curves": n_null,
        "confidence": float(confidence),
        "min_null_curves": int(min_null_curves),
        "T_obs": np.nan,
        "T_critical": np.nan,
        "p_mad": np.nan,
        "rejects_null": False,
        "r_at_max_nm": np.nan,
        "signed_diff_at_max": np.nan,
        "r_vals": r_vals_full,
        "null_mean": np.full(len(r_vals_full), np.nan),
        "ce_lo": np.full(len(r_vals_full), np.nan),
        "ce_hi": np.full(len(r_vals_full), np.nan),
        "abs_diff": np.full(len(r_vals_full), np.nan),
        "normalized_obs": np.full(len(r_vals_full), np.nan),
        "observed": observed,
    }

    mask = np.ones(len(r_vals_full), dtype=bool)
    if r_min_nm is not None:
        mask &= r_vals_full >= float(r_min_nm)
    if r_max_nm is not None:
        mask &= r_vals_full <= float(r_max_nm)
    if np.any(mask):
        result["r_vals"] = r_vals_full[mask]
        result["observed"] = observed[mask] if len(observed) == len(r_vals_full) else observed
        n_sub = int(mask.sum())
        result["null_mean"] = np.full(n_sub, np.nan)
        result["ce_lo"] = np.full(n_sub, np.nan)
        result["ce_hi"] = np.full(n_sub, np.nan)
        result["abs_diff"] = np.full(n_sub, np.nan)
        result["normalized_obs"] = np.full(n_sub, np.nan)

    if n_null < int(min_null_curves) or len(observed) != len(r_vals_full):
        result["status"] = "skipped_insufficient_nulls"
        return result
    if not np.any(mask):
        result["status"] = "skipped_empty_r_range"
        return result

    r_vals = r_vals_full[mask]
    observed_s = observed[mask]
    null_s = null_curves[:, mask]
    if not np.all(np.isfinite(observed_s)) or not np.any(np.isfinite(null_s)):
        result["status"] = "skipped_nonfinite"
        return result

    # Diggle pooled mean: μ(r) = (L_obs + Σ L_null) / (N + 1). Same μ for T_obs and all T_s.
    all_curves = np.vstack([observed_s[None, :], null_s])
    pooled_mean = np.nanmean(all_curves, axis=0)
    # Pointwise CE stays null-only (what the null band looks like).
    ce_lo = np.nanpercentile(null_s, lo_pct, axis=0)
    ce_hi = np.nanpercentile(null_s, hi_pct, axis=0)
    abs_diff = np.abs(observed_s - pooled_mean)
    T_obs = float(np.nanmax(abs_diff))
    r_at_max_idx = int(np.nanargmax(abs_diff))
    signed = float(observed_s[r_at_max_idx] - pooled_mean[r_at_max_idx])

    T_null = np.nanmax(np.abs(null_s - pooled_mean[None, :]), axis=1)
    T_critical = float(np.nanpercentile(T_null, 100.0 * confidence))
    p_mad = float((1 + np.sum(T_null >= T_obs)) / (1 + n_null))

    # Scale so the confidence envelope appears as ±1 (Rebola pooling convention).
    half = np.maximum(np.maximum(pooled_mean - ce_lo, ce_hi - pooled_mean), 1e-12)
    normalized = (observed_s - pooled_mean) / half

    result.update(
        {
            "T_obs": T_obs,
            "T_critical": T_critical,
            "p_mad": p_mad,
            "rejects_null": bool(T_obs > T_critical),
            "r_at_max_nm": float(r_vals[r_at_max_idx]),
            "signed_diff_at_max": signed,
            "r_vals": r_vals,
            # Kept as null_mean for downstream CSV/plot compatibility; value is pooled μ.
            "null_mean": pooled_mean,
            "ce_lo": ce_lo,
            "ce_hi": ce_hi,
            "abs_diff": abs_diff,
            "normalized_obs": normalized,
            "observed": observed_s,
        }
    )
    return result


def run_mad_tests_over_r_ranges(
    observed: np.ndarray,
    null_curves: np.ndarray,
    r_vals: np.ndarray,
    *,
    null_name: str,
    r_ranges: tuple[tuple[str, float | None, float | None], ...] = MAD_R_RANGES,
    min_null_curves: int = MAD_MIN_NULL_CURVES,
) -> list[dict]:
    """Run MAD for each configured radius window (full + 30–50 nm by default)."""
    return [
        mad_test_from_curves(
            observed,
            null_curves,
            r_vals,
            min_null_curves=min_null_curves,
            null_name=null_name,
            r_min_nm=r_min,
            r_max_nm=r_max,
            r_range=label,
        )
        for label, r_min, r_max in r_ranges
    ]


def mad_result_to_summary_row(
    mad: dict,
    *,
    extra_cols: dict | None = None,
) -> dict:
    """Flatten MAD scalar fields into one CSV/JSON-friendly row."""
    row = {
        "null_name": mad["null_name"],
        "r_range": mad.get("r_range", "full"),
        "r_min_nm": float(mad["r_min_nm"]) if np.isfinite(mad.get("r_min_nm", np.nan)) else np.nan,
        "r_max_nm": float(mad["r_max_nm"]) if np.isfinite(mad.get("r_max_nm", np.nan)) else np.nan,
        "status": mad["status"],
        "n_null_curves": int(mad["n_null_curves"]),
        "confidence": float(mad["confidence"]),
        "T_obs": float(mad["T_obs"]) if np.isfinite(mad["T_obs"]) else np.nan,
        "T_critical": float(mad["T_critical"]) if np.isfinite(mad["T_critical"]) else np.nan,
        "T_obs_over_T_critical": (
            float(mad["T_obs"] / mad["T_critical"])
            if np.isfinite(mad["T_obs"]) and np.isfinite(mad["T_critical"]) and mad["T_critical"] > 0
            else np.nan
        ),
        "p_mad": float(mad["p_mad"]) if np.isfinite(mad["p_mad"]) else np.nan,
        "rejects_null": bool(mad["rejects_null"]),
        "r_at_max_nm": float(mad["r_at_max_nm"]) if np.isfinite(mad["r_at_max_nm"]) else np.nan,
        "signed_diff_at_max": (
            float(mad["signed_diff_at_max"]) if np.isfinite(mad["signed_diff_at_max"]) else np.nan
        ),
    }
    if extra_cols:
        row.update(extra_cols)
    return row


def mad_result_to_curves_dataframe(
    mad: dict,
    r_vals: np.ndarray | None = None,
    *,
    observed: np.ndarray | None = None,
) -> pd.DataFrame:
    """Long table for MAD graphing on the MAD radius window (full or 30–50 nm)."""
    r_use = np.asarray(mad.get("r_vals", r_vals if r_vals is not None else []), dtype=float)
    if "observed" in mad and mad["observed"] is not None:
        observed_use = np.asarray(mad["observed"], dtype=float).reshape(-1)
    elif observed is not None:
        observed_use = np.asarray(observed, dtype=float).reshape(-1)
        if len(observed_use) != len(r_use) and r_vals is not None:
            r_full = np.asarray(r_vals, dtype=float)
            if len(observed_use) == len(r_full):
                observed_use = observed_use[np.isin(r_full, r_use)]
    else:
        observed_use = np.full(len(r_use), np.nan)
    return pd.DataFrame(
        {
            "r_nm": r_use,
            "observed_L12": observed_use,
            "null_mean_L12": mad["null_mean"],
            "ce_lo": mad["ce_lo"],
            "ce_hi": mad["ce_hi"],
            "abs_diff": mad["abs_diff"],
            "normalized_obs": mad["normalized_obs"],
            "null_name": mad["null_name"],
            "r_range": mad.get("r_range", "full"),
            "status": mad["status"],
        }
    )


def _ripley_r_grid(r_max_nm: float, r_step_nm: float) -> np.ndarray:
    n_steps = max(1, int(np.floor(r_max_nm / r_step_nm)))
    return np.arange(r_step_nm, r_max_nm + 0.5 * r_step_nm, r_step_nm, dtype=float)


# Shared context for parallel label-permutation L₁₂ workers (set via pool initializer).
_PERM_L12_CTX: dict = {}


def _default_ripley_perm_workers(n_perm: int) -> int:
    """Worker count for label-permutation Ripley evals (override via SYNAPTIC_RIPLEY_PERM_WORKERS)."""
    n_perm = int(n_perm)
    if n_perm <= 1:
        return 1
    env = os.environ.get("SYNAPTIC_RIPLEY_PERM_WORKERS")
    if env is not None and str(env).strip():
        try:
            return max(1, min(int(env), n_perm))
        except ValueError:
            pass
    n_cpu = os.cpu_count() or 1
    return max(1, min(n_cpu, n_perm))


def _init_label_perm_l12_worker(
    pool: np.ndarray,
    r_vals: np.ndarray,
    window: RipleyWindow3D,
    n_monomer: int,
    pool_edge_factors: np.ndarray | None = None,
) -> None:
    _PERM_L12_CTX["pool"] = pool
    _PERM_L12_CTX["r_vals"] = r_vals
    _PERM_L12_CTX["window"] = window
    _PERM_L12_CTX["n_monomer"] = int(n_monomer)
    _PERM_L12_CTX["pool_edge_factors"] = pool_edge_factors


def _label_perm_l12_worker(task: tuple[int, int]) -> tuple[int, np.ndarray]:
    """Run one label-permuted L₁₂ curve (perm_id, seed)."""
    perm_id, seed = task
    pool = _PERM_L12_CTX["pool"]
    r_vals = _PERM_L12_CTX["r_vals"]
    window = _PERM_L12_CTX["window"]
    n_monomer = _PERM_L12_CTX["n_monomer"]
    pool_edge_factors = _PERM_L12_CTX.get("pool_edge_factors")
    rng = np.random.default_rng(int(seed))
    class1_idx = rng.choice(len(pool), n_monomer, replace=False)
    mask = np.zeros(len(pool), dtype=bool)
    mask[class1_idx] = True
    focus_edge = pool_edge_factors[mask] if pool_edge_factors is not None else None
    curve = ripley_l12_from_points(
        pool[mask], pool[~mask], r_vals, window, rng, edge_factors=focus_edge
    )
    return int(perm_id), curve


def _label_permutation_l12_curves_sequential(
    pool: np.ndarray,
    n_monomer: int,
    r_vals: np.ndarray,
    window: RipleyWindow3D,
    rng: np.random.Generator,
    *,
    n_perm: int,
    pool_edge_factors: np.ndarray | None = None,
    pbar=None,
) -> np.ndarray:
    """Sequential label-permutation null L₁₂ curves."""
    n_pool = len(pool)
    curves = np.full((int(n_perm), len(r_vals)), np.nan, dtype=float)
    for perm_id in range(int(n_perm)):
        class1_idx = rng.choice(n_pool, n_monomer, replace=False)
        mask = np.zeros(n_pool, dtype=bool)
        mask[class1_idx] = True
        focus_edge = pool_edge_factors[mask] if pool_edge_factors is not None else None
        curves[perm_id] = ripley_l12_from_points(
            pool[mask], pool[~mask], r_vals, window, rng, edge_factors=focus_edge
        )
        if pbar is not None:
            pbar.set_postfix_str(f"perm {perm_id + 1}/{int(n_perm)}", refresh=False)
            pbar.update(1)
    return curves


def label_permutation_l12_curves(
    monomer_coords: np.ndarray,
    dimer_coords: np.ndarray,
    r_vals: np.ndarray,
    window: RipleyWindow3D,
    *,
    n_perm: int,
    seed: int = DEFAULT_ANALYSIS_SEED,
    rng: np.random.Generator | None = None,
    n_workers: int | None = None,
    pbar=None,
) -> np.ndarray:
    """
    Label-permutation null L₁₂ curves with optional process parallelism.

    Pool monomer + dimer points, then for each replicate randomly relabel exactly
    ``n_monomer`` points as class 1 (monomer) and the rest as class 2 (dimer).
    Returns an ``(n_perm, len(r_vals))`` array.

    Edge-correction factors depend only on the fixed pooled coordinates (not on the
    label assignment), so they are estimated once for the whole pool and indexed per
    replicate — this avoids ~``n_perm`` redundant Monte-Carlo edge computations.
    """
    pool = np.vstack([np.atleast_2d(monomer_coords), np.atleast_2d(dimer_coords)])
    n_pool = len(pool)
    n_monomer = len(np.atleast_2d(monomer_coords))
    n_perm_int = int(n_perm)
    curves = np.full((n_perm_int, len(r_vals)), np.nan, dtype=float)
    if n_pool == 0 or n_monomer == 0 or n_monomer >= n_pool or n_perm_int == 0:
        return curves

    if n_workers is None:
        n_workers = _default_ripley_perm_workers(n_perm_int)
    n_workers = max(1, min(int(n_workers), n_perm_int))

    # Precompute isotropic edge factors once for every pooled point (reused across perms).
    edge_rng = np.random.default_rng(int(seed) + 7919)
    pool_edge_factors = _isotropic_edge_factors_for_foci(pool, r_vals, window, edge_rng)

    if n_workers == 1:
        perm_rng = rng if rng is not None else np.random.default_rng(int(seed))
        return _label_permutation_l12_curves_sequential(
            pool,
            n_monomer,
            r_vals,
            window,
            perm_rng,
            n_perm=n_perm_int,
            pool_edge_factors=pool_edge_factors,
            pbar=pbar,
        )

    tasks = [(perm_id, int(seed) + 1 + perm_id) for perm_id in range(n_perm_int)]
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_label_perm_l12_worker,
        initargs=(pool, r_vals, window, n_monomer, pool_edge_factors),
    ) as executor:
        futures = [executor.submit(_label_perm_l12_worker, task) for task in tasks]
        for fut in as_completed(futures):
            perm_id, curve = fut.result()
            curves[perm_id] = curve
            if pbar is not None:
                pbar.set_postfix_str(f"perm {perm_id + 1}/{n_perm_int}", refresh=False)
                pbar.update(1)
    return curves


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
        old = pd.read_csv(path)
        if drop_column_prefix and not old.empty:
            keep = [c for c in old.columns if not str(c).startswith(drop_column_prefix)]
            old = old[keep] if keep else pd.DataFrame()
        if not old.empty:
            frames.append(old)
    if new_df is not None and not new_df.empty:
        frames.append(new_df)
    merged = merge_distance_column_dataframes(frames)
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
    written: list[Path] = []
    for stem, out_path in stem_to_out.items():
        found: list[Path] = []
        for root in roots:
            found.extend(root.glob(f"**/{ANALYSES_SUBDIR}/*/{stem}.csv"))
        frames = [pd.read_csv(p) for p in sorted(set(found)) if p.is_file()]
        merged = merge_distance_column_dataframes(frames)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(out_path, index=False)
        print(
            f"Pooled vesicle-column distances ({stem}: {merged.shape[1]} columns, "
            f"{merged.shape[0]} rows) -> {out_path}"
        )
        written.append(out_path)
    return written


def _distance_csv_name(stem: str, subset: AunpSubset) -> str:
    return f"{stem}__{subset}.csv"


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


def _mean_sem_band(curves: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (mean − SEM, mean, mean + SEM, SEM) across replicate curves at each r."""
    if curves.size == 0:
        empty = np.array([])
        return empty, empty, empty, empty
    mean = np.nanmean(curves, axis=0)
    n = int(np.shape(curves)[0])
    if n > 1:
        sd = np.nanstd(curves, axis=0, ddof=1)
        sem = sd / np.sqrt(float(n))
    else:
        sem = np.zeros_like(mean)
    return mean - sem, mean, mean + sem, sem


def _prism_sd_envelope_columns(
    curves: np.ndarray,
    r_vals: np.ndarray,
    *,
    prefix: str,
) -> dict[str, np.ndarray]:
    """SD + SEM summary columns for Prism tables (mean ± SD/SEM envelopes)."""
    if len(curves):
        sd_lo, mean, sd_hi, sd = _mean_sd_band(curves)
        sem_lo, _, sem_hi, sem = _mean_sem_band(curves)
    else:
        nan = np.full(len(r_vals), np.nan)
        sd_lo = mean = sd_hi = sd = nan
        sem_lo = sem_hi = sem = nan
    return {
        f"{prefix}_mean": mean,
        f"{prefix}_sd": sd,
        f"{prefix}_sd_envelope_lo": sd_lo,
        f"{prefix}_sd_envelope_hi": sd_hi,
        f"{prefix}_sem": sem,
        f"{prefix}_sem_envelope_lo": sem_lo,
        f"{prefix}_sem_envelope_hi": sem_hi,
    }


def curves_matrix_to_long_dataframe(
    curves: np.ndarray,
    r_vals: np.ndarray,
    *,
    curve_type: str,
    value_col: str = "ripley_l12",
    extra_cols: dict | None = None,
) -> pd.DataFrame:
    """Long table with one row per (replicate_index, r_nm) from an (n_curves, n_r) matrix."""
    curves = np.atleast_2d(np.asarray(curves, dtype=float))
    r_vals = np.asarray(r_vals, dtype=float)
    if curves.size == 0 or curves.shape[0] == 0:
        return pd.DataFrame(
            columns=["curve_type", "replicate_index", "r_nm", value_col, *(extra_cols or {})]
        )
    rows: list[dict] = []
    extras = extra_cols or {}
    for i, curve in enumerate(curves):
        for r_val, l_val in zip(r_vals, curve):
            rows.append(
                {
                    "curve_type": curve_type,
                    "replicate_index": int(i),
                    "r_nm": float(r_val),
                    value_col: float(l_val),
                    **extras,
                }
            )
    return pd.DataFrame(rows)


def curves_matrix_to_wide_dataframe(
    curves: np.ndarray,
    r_vals: np.ndarray,
    *,
    curve_type: str,
    column_prefix: str | None = None,
) -> pd.DataFrame:
    """Wide Prism-friendly table: one row per r_nm, one column per replicate curve."""
    curves = np.atleast_2d(np.asarray(curves, dtype=float))
    r_vals = np.asarray(r_vals, dtype=float)
    prefix = column_prefix or curve_type
    if curves.size == 0 or curves.shape[0] == 0:
        return pd.DataFrame({"r_nm": r_vals})
    data: dict[str, np.ndarray] = {"r_nm": r_vals}
    for i, curve in enumerate(curves):
        data[f"{prefix}_{i}"] = np.asarray(curve, dtype=float)
    return pd.DataFrame(data)


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
    mean ± 1 sample SD; SEM envelopes use mean ± SEM (SD / sqrt(n)).

    One row per (control_comparison, r_nm). ``fusing_L12_mean`` is identical within
    each comparison group (mean across fusing-vesicle curves).
    """
    fusing_sd = _prism_sd_envelope_columns(obs_curves, r_vals, prefix="fusing_L12")
    fusing_from_k = prism_sd_envelope_columns_from_averaged_k12(
        obs_curves, r_vals, prefix="fusing_L12"
    )
    rows: list[dict] = []
    for comparison, control_curves in control_curves_by_comparison.items():
        if len(control_curves):
            ctrl_lo, ctrl_mean, ctrl_hi = _percentile_band(control_curves)
            ctrl_sd = _prism_sd_envelope_columns(control_curves, r_vals, prefix="control_L12")
            ctrl_from_k = prism_sd_envelope_columns_from_averaged_k12(
                control_curves, r_vals, prefix="control_L12"
            )
            n_control = int(len(control_curves))
        else:
            ctrl_lo = ctrl_mean = ctrl_hi = np.full(len(r_vals), np.nan)
            ctrl_sd = _prism_sd_envelope_columns(np.empty((0, len(r_vals))), r_vals, prefix="control_L12")
            ctrl_from_k = prism_sd_envelope_columns_from_averaged_k12(
                np.empty((0, len(r_vals))), r_vals, prefix="control_L12"
            )
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
                    "fusing_L12_mean_from_k": float(fusing_from_k["fusing_L12_mean_from_k"][i]),
                    "fusing_L12_sd": float(fusing_sd["fusing_L12_sd"][i]),
                    "fusing_L12_sd_envelope_lo": float(fusing_sd["fusing_L12_sd_envelope_lo"][i]),
                    "fusing_L12_sd_envelope_hi": float(fusing_sd["fusing_L12_sd_envelope_hi"][i]),
                    "fusing_L12_sd_from_k": float(fusing_from_k["fusing_L12_sd_from_k"][i]),
                    "fusing_L12_sd_envelope_lo_from_k": float(
                        fusing_from_k["fusing_L12_sd_envelope_lo_from_k"][i]
                    ),
                    "fusing_L12_sd_envelope_hi_from_k": float(
                        fusing_from_k["fusing_L12_sd_envelope_hi_from_k"][i]
                    ),
                    "fusing_L12_sem": float(fusing_sd["fusing_L12_sem"][i]),
                    "fusing_L12_sem_envelope_lo": float(fusing_sd["fusing_L12_sem_envelope_lo"][i]),
                    "fusing_L12_sem_envelope_hi": float(fusing_sd["fusing_L12_sem_envelope_hi"][i]),
                    "fusing_L12_sem_from_k": float(fusing_from_k["fusing_L12_sem_from_k"][i]),
                    "fusing_L12_sem_envelope_lo_from_k": float(
                        fusing_from_k["fusing_L12_sem_envelope_lo_from_k"][i]
                    ),
                    "fusing_L12_sem_envelope_hi_from_k": float(
                        fusing_from_k["fusing_L12_sem_envelope_hi_from_k"][i]
                    ),
                    "control_L12_mean": float(ctrl_mean[i]),
                    "control_L12_mean_from_k": float(ctrl_from_k["control_L12_mean_from_k"][i]),
                    "control_L12_sd": float(ctrl_sd["control_L12_sd"][i]),
                    "control_L12_envelope_lo": float(ctrl_lo[i]),
                    "control_L12_envelope_hi": float(ctrl_hi[i]),
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
            "fusing_L12_mean_from_k",
            "fusing_L12_sd",
            "fusing_L12_sd_envelope_lo",
            "fusing_L12_sd_envelope_hi",
            "fusing_L12_sd_from_k",
            "fusing_L12_sd_envelope_lo_from_k",
            "fusing_L12_sd_envelope_hi_from_k",
            "control_L12_mean",
            "control_L12_mean_from_k",
            "control_L12_sd",
            "control_L12_envelope_lo",
            "control_L12_envelope_hi",
            "control_L12_sd_envelope_lo",
            "control_L12_sd_envelope_hi",
            "control_L12_sd_from_k",
            "control_L12_sd_envelope_lo_from_k",
            "control_L12_sd_envelope_hi_from_k",
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
    obs_from_k = prism_sd_envelope_columns_from_averaged_k12(
        obs_curves, r_vals, prefix="fusing_L12"
    )
    obs_mean = np.nanmean(obs_curves, axis=0) if len(obs_curves) else np.full(len(r_vals), np.nan)
    ax.plot(r_vals, obs_mean, color="C3", lw=2.2, label="Fusing mean (of L)", zorder=5)
    ax.plot(
        r_vals,
        obs_from_k["fusing_L12_mean_from_k"],
        color="C3",
        lw=1.6,
        ls="--",
        label="Fusing mean (K→L)",
        zorder=5,
    )
    if len(obs_curves) > 1:
        ax.plot(
            r_vals,
            obs_from_k["fusing_L12_sd_envelope_lo_from_k"],
            color="C3",
            lw=0.9,
            ls=":",
            alpha=0.8,
            label="Fusing ±SD (K→L)",
            zorder=4,
        )
        ax.plot(
            r_vals,
            obs_from_k["fusing_L12_sd_envelope_hi_from_k"],
            color="C3",
            lw=0.9,
            ls=":",
            alpha=0.8,
            zorder=4,
        )

    if len(control_curves):
        lo, ctrl_mean, hi = _percentile_band(control_curves)
        ctrl_from_k = prism_sd_envelope_columns_from_averaged_k12(
            control_curves, r_vals, prefix="control_L12"
        )
        ax.fill_between(r_vals, lo, hi, color="0.75", alpha=0.55, label="Control 95% envelope", zorder=2)
        ax.plot(r_vals, ctrl_mean, color="0.45", lw=2.0, label="Control mean (of L)", zorder=3)
        ax.plot(
            r_vals,
            ctrl_from_k["control_L12_mean_from_k"],
            color="0.45",
            lw=1.5,
            ls="--",
            label="Control mean (K→L)",
            zorder=3,
        )
        ax.plot(
            r_vals,
            ctrl_from_k["control_L12_sd_envelope_lo_from_k"],
            color="0.45",
            lw=0.9,
            ls=":",
            alpha=0.8,
            label="Control ±SD (K→L)",
            zorder=3,
        )
        ax.plot(
            r_vals,
            ctrl_from_k["control_L12_sd_envelope_hi_from_k"],
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
    return ripley_l12_curves_per_focus(points, aunp_coords, r_vals, window, rng)


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
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
                        "active_zone_name": zone_name,
                        "aunp_subset": aunp_subset,
                        "window_mode": mode_tag,
                    },
                )
            )
            mad_curves = mad_result_to_curves_dataframe(mad, r_vals, observed=obs_mean)
            mad_curves.insert(0, "active_zone_name", zone_name)
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
    curves_df = pd.DataFrame(rows)
    mad_summary_df = pd.DataFrame(mad_summary_rows)
    mad_curves_df = (
        pd.concat(mad_curve_frames, ignore_index=True) if mad_curve_frames else pd.DataFrame()
    )
    return curves_df, prism_df, mad_summary_df, mad_curves_df


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

    ``*_mean`` / ``*_sd_*`` are mean ± SD/SEM of L₁₂ curves.
    ``*_mean_from_k`` / ``*_sd_*_from_k`` / ``*_sem_*_from_k`` are mean ± SD/SEM of K₁₂,
    then mapped back to L₁₂ (differs from L-space SD/SEM because L is nonlinear).
    Percentile envelopes (``control_L12_envelope_*``) are unchanged — identical under K→L.
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
            fusing_from_k = prism_sd_envelope_columns_from_averaged_k12(
                obs_curves, r_vals, prefix="fusing_L12"
            )
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
                    ctrl_from_k = prism_sd_envelope_columns_from_averaged_k12(
                        control_curves, r_vals, prefix="control_L12"
                    )
                    n_control = int(len(control_curves))
                else:
                    ctrl_lo = ctrl_mean = ctrl_hi = np.full(len(r_vals), np.nan)
                    ctrl_sd = _prism_sd_envelope_columns(
                        np.empty((0, len(r_vals))), r_vals, prefix="control_L12"
                    )
                    ctrl_from_k = prism_sd_envelope_columns_from_averaged_k12(
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
                            "fusing_L12_mean_from_k": float(
                                fusing_from_k["fusing_L12_mean_from_k"][i]
                            ),
                            "fusing_L12_sd": float(fusing_sd["fusing_L12_sd"][i]),
                            "fusing_L12_sd_envelope_lo": float(
                                fusing_sd["fusing_L12_sd_envelope_lo"][i]
                            ),
                            "fusing_L12_sd_envelope_hi": float(
                                fusing_sd["fusing_L12_sd_envelope_hi"][i]
                            ),
                            "fusing_L12_sd_from_k": float(
                                fusing_from_k["fusing_L12_sd_from_k"][i]
                            ),
                            "fusing_L12_sd_envelope_lo_from_k": float(
                                fusing_from_k["fusing_L12_sd_envelope_lo_from_k"][i]
                            ),
                            "fusing_L12_sd_envelope_hi_from_k": float(
                                fusing_from_k["fusing_L12_sd_envelope_hi_from_k"][i]
                            ),
                            "fusing_L12_sem": float(fusing_sd["fusing_L12_sem"][i]),
                            "fusing_L12_sem_envelope_lo": float(
                                fusing_sd["fusing_L12_sem_envelope_lo"][i]
                            ),
                            "fusing_L12_sem_envelope_hi": float(
                                fusing_sd["fusing_L12_sem_envelope_hi"][i]
                            ),
                            "fusing_L12_sem_from_k": float(
                                fusing_from_k["fusing_L12_sem_from_k"][i]
                            ),
                            "fusing_L12_sem_envelope_lo_from_k": float(
                                fusing_from_k["fusing_L12_sem_envelope_lo_from_k"][i]
                            ),
                            "fusing_L12_sem_envelope_hi_from_k": float(
                                fusing_from_k["fusing_L12_sem_envelope_hi_from_k"][i]
                            ),
                            "control_L12_mean": float(ctrl_mean[i]),
                            "control_L12_mean_from_k": float(
                                ctrl_from_k["control_L12_mean_from_k"][i]
                            ),
                            "control_L12_sd": float(ctrl_sd["control_L12_sd"][i]),
                            "control_L12_envelope_lo": float(ctrl_lo[i]),
                            "control_L12_envelope_hi": float(ctrl_hi[i]),
                            "control_L12_sd_envelope_lo": float(
                                ctrl_sd["control_L12_sd_envelope_lo"][i]
                            ),
                            "control_L12_sd_envelope_hi": float(
                                ctrl_sd["control_L12_sd_envelope_hi"][i]
                            ),
                            "control_L12_sd_from_k": float(
                                ctrl_from_k["control_L12_sd_from_k"][i]
                            ),
                            "control_L12_sd_envelope_lo_from_k": float(
                                ctrl_from_k["control_L12_sd_envelope_lo_from_k"][i]
                            ),
                            "control_L12_sd_envelope_hi_from_k": float(
                                ctrl_from_k["control_L12_sd_envelope_hi_from_k"][i]
                            ),
                            "control_L12_sem": float(ctrl_sd["control_L12_sem"][i]),
                            "control_L12_sem_envelope_lo": float(
                                ctrl_sd["control_L12_sem_envelope_lo"][i]
                            ),
                            "control_L12_sem_envelope_hi": float(
                                ctrl_sd["control_L12_sem_envelope_hi"][i]
                            ),
                            "control_L12_sem_from_k": float(
                                ctrl_from_k["control_L12_sem_from_k"][i]
                            ),
                            "control_L12_sem_envelope_lo_from_k": float(
                                ctrl_from_k["control_L12_sem_envelope_lo_from_k"][i]
                            ),
                            "control_L12_sem_envelope_hi_from_k": float(
                                ctrl_from_k["control_L12_sem_envelope_hi_from_k"][i]
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
    use_angle_betweenness_window: bool = False,
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

    pre_membrane_coords = None
    post_membrane_coords = None
    if use_angle_betweenness_window:
        from .activezone import import_active_zone_segmentations

        az_segmentation = import_active_zone_segmentations(
            tomogram_path, alignment_dir=alignment_dir
        ).get(zone_name)
        if az_segmentation is not None:
            pre_membrane_coords = az_segmentation.get("presynaptic_outer_coords")
            post_membrane_coords = az_segmentation.get("postsynaptic_outer_coords")

    try:
        ripley_windows["fusion_aunp_hull"] = build_ripley_window_3d(
            hull_defining_coords,
            "fusion_aunp_hull",
            pre_membrane_coords=pre_membrane_coords,
            post_membrane_coords=post_membrane_coords,
            use_angle_betweenness=use_angle_betweenness_window,
            rng=rng,
        )
    except ValueError as exc:
        print(f"  Ripley fusion+AuNP hull window skipped for {zone_name}: {exc}")

    try:
        cleft_coords = load_synaptic_cleft_active_zone_points(
            tomogram_path, alignment_dir, zone_name
        )
        ripley_windows["synaptic_cleft_az_hull"] = build_ripley_window_3d(
            cleft_coords,
            "synaptic_cleft_az_hull",
            pre_membrane_coords=pre_membrane_coords,
            post_membrane_coords=post_membrane_coords,
            use_angle_betweenness=use_angle_betweenness_window,
            rng=rng,
        )
    except (ValueError, FileNotFoundError) as exc:
        cleft_coords = None
        print(f"  Ripley synaptic-cleft AZ hull window skipped for {zone_name}: {exc}")

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
        meta_out.insert(3, "active_zone_name", zone_name)

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

    # Vesicle-as-column tables using all loaded AuNPs for this zone (pooled across tomograms later).
    col_only_specs = [
        ("fusing", original_cols_global, POOLED_DIST_FUSING_COLUMNS_CSV),
        ("close", close_cols_global, POOLED_DIST_CLOSE_COLUMNS_CSV),
        ("shift_40nm", shift_cols_global, POOLED_DIST_SHIFT_COLUMNS_CSV),
        ("label_permutation", label_cols_global, POOLED_DIST_PERM_COLUMNS_CSV),
    ]
    zone_prefix = _zone_column_prefix(tomogram_name, alignment_dir, zone_name)
    distance_column_paths: dict[str, Path] = {}
    for kind, cols_global, pooled_path in col_only_specs:
        df_cols = build_distance_columns_only_dataframe(aunp_coords_all, cols_global)
        stem = DISTANCE_COLUMNS_ONLY_STEMS[kind]
        p_cols = out_dir / f"{stem}.csv"
        df_cols.to_csv(p_cols, index=False)
        distance_column_paths[kind] = p_cols
        upsert_pooled_distance_columns_csv(
            pooled_path, df_cols, drop_column_prefix=zone_prefix
        )
    # --- Ripley: both window modes × three AuNP partner subsets ---
    ripley_frames: list[pd.DataFrame] = []
    prism_frames: list[pd.DataFrame] = []
    mad_summary_frames: list[pd.DataFrame] = []
    mad_curve_frames: list[pd.DataFrame] = []

    for window_mode in RIPLEY_WINDOW_MODES:
        window = ripley_windows.get(window_mode)
        if window is None:
            continue
        for subset in subsets_to_run:
            sub_coords, _ = subset_aunps(aunp_meta, subset=subset)
            if len(sub_coords) == 0:
                print(f"  Skipping Ripley for {zone_name} ({subset}, {window_mode}): no partner AuNPs")
                continue
            df_r, df_prism, df_mad_summary, df_mad_curves = run_ripley_for_zone_window(
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

    if ripley_frames:
        ripley_long = pd.concat(ripley_frames, ignore_index=True)
        ripley_long.insert(0, "tomogram_name", tomogram_name)
        ripley_long.insert(1, "alignment_dir", alignment_dir)
        ripley_long.to_csv(out_dir / "ripley_l12_curves.csv", index=False)
        # Explicit individual-curves filename for consistency with other Ripley analyses.
        ripley_long.to_csv(out_dir / "ripley_l12_individual_curves.csv", index=False)
        # Prism-friendly wide tables: one column per replicate curve.
        for (subset, window_mode, curve_type), grp in ripley_long.groupby(
            ["aunp_subset", "window_mode", "curve_type"], sort=False
        ):
            r_vals_g = np.sort(grp["r_nm"].unique())
            curves_list: list[np.ndarray] = []
            for _, rep in grp.groupby("replicate_index", sort=True):
                rep = rep.sort_values("r_nm")
                if len(rep) != len(r_vals_g):
                    continue
                curves_list.append(rep["ripley_l12"].to_numpy(dtype=float))
            if not curves_list:
                continue
            wide = curves_matrix_to_wide_dataframe(
                np.vstack(curves_list),
                r_vals_g,
                curve_type=str(curve_type),
            )
            wide_name = (
                f"ripley_l12_individual_{subset}_{window_mode}_{curve_type}_wide.csv"
            )
            wide.to_csv(out_dir / wide_name, index=False)
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
                "uses_angle_betweenness": bool(win.use_angle_betweenness),
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
        "mad_min_null_curves": int(MAD_MIN_NULL_CURVES),
        "mad_nulls": ["close", "shift_40nm", "label_permutation"],
        "mad_r_ranges": [label for label, _, _ in MAD_R_RANGES],
        "seed": int(seed),
        "ripley_edge_correction": "isotropic_3d_mc",
        "window_volume_definition": "convex_hull_volume",
        "window_uses_angle_betweenness": bool(use_angle_betweenness_window),
    }
    with open(out_dir / "run_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(
        f"  Fusion-point/AuNP 3D analyses ({zone_name}): "
        f"{len(fusing_rows)} fusing, {n_monomer} monomer + {n_dimer} dimer AuNPs -> {out_dir}"
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
    use_angle_betweenness_window: bool = False,
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
            use_angle_betweenness_window=use_angle_betweenness_window,
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
