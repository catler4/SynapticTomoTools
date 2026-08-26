"""
Shared 3D bivariate Ripley K₁₂/L₁₂ infrastructure used by the AuNP Ripley analyses:
``aunp_monomer_dimer_ripley.py``, ``aunp_ripley_vs_cleft_center.py``, and
``fusion_point_aunp_position_distance_and_Ripleys_analyses.py``.

Covers: the convex-hull Ripley window and its membership tests, isotropic edge
correction (Monte Carlo and deterministic-grid variants), the K₁₂/L₁₂ estimator and
K↔L transforms, the pair-correlation function g as a finite difference of an
already-computed K curve (``pair_correlation_from_k_diff``), the symmetrized bivariate
K_(12,21) / L_(12,21) / g_(12,21) combinations (Lotwick & Silverman 1982),
label-permutation null curves, MAD (maximum absolute deviation) null-hypothesis tests,
curve-table/Prism-envelope builders, and the shared AuNP pick / synaptic-cleft surface
point loaders.

Analysis-specific logic (fusion-site placement, distance tables, window modes other
than ``synaptic_cleft_az_hull``, etc.) stays in the analysis modules that use it.
"""

from __future__ import annotations

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
from .aunps import _read_aunp_pick_star_dataframe
from .fusion_point_vs_aunp_density import load_presynaptic_az_points_for_zone

# ============================================================================
# Shared constants
# ============================================================================

DEFAULT_ANALYSIS_SEED = 42
DEFAULT_RIPLEY_R_MAX_NM = 100.0
DEFAULT_RIPLEY_R_STEP_NM = 1.0
RIPLEY_PERCENTILE_LO = 2.5
RIPLEY_PERCENTILE_HI = 97.5
EDGE_MC_SAMPLES = 384
EDGE_MIN_C = 1e-3
EDGE_MIN_HITS_MAX_SAMPLES = 50_000
WINDOW_VOLUME_MC_SAMPLES = 200_000
ANGLE_BETWEENNESS_MAX_MEMBRANE_DISTANCE_NM = 50.0
# Minimum window-quadrature grid points supporting a g shell (see g_shell_reliability_mask)
# before it's trusted rather than NaN'd out.
G12_MIN_GRID_POINTS_PER_SHELL = 8

COORD_COLS = ("faCoordinateX", "faCoordinateY", "faCoordinateZ")
DEFAULT_MONOMER_STAR_PATTERN = "aunp_tm_BP_active_zone_*_manual_refined_monomer.star"
DEFAULT_DIMER_STAR_PATTERN = "aunp_tm_BP_active_zone_*_manual_refined_dimer.star"

MAD_MIN_NULL_CURVES = 1000
MAD_CONFIDENCE = 0.99
# (label, r_min_nm or None, r_max_nm or None). None → use the full Ripley grid.
MAD_R_RANGES: tuple[tuple[str, float | None, float | None], ...] = (
    ("full", None, None),
    ("30-50nm", 30.0, 50.0),
)

AunpKind = Literal["monomer", "dimer", "all"]
AunpSubset = Literal["monomer", "dimer", "both", "all"]
AUNP_SUBSETS: tuple[AunpSubset, ...] = ("monomer", "dimer", "both", "all")


@dataclass(frozen=True)
class ZoneAunpLoadResult:
    """Monomer and/or dimer (or single-pool) pick coordinates for one synaptic cleft."""

    coords: np.ndarray
    meta: pd.DataFrame
    kinds_loaded: tuple[AunpKind, ...]


def available_aunp_subsets(kinds_loaded: Sequence[AunpKind]) -> tuple[AunpSubset, ...]:
    """Analysis subsets to run given which STAR files were found."""
    kinds = tuple(kinds_loaded)
    if kinds == ("all",) or (len(kinds) == 1 and kinds[0] == "all"):
        return ("all",)
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
    defining_mode: str
    pre_membrane_coords: np.ndarray | None = None
    post_membrane_coords: np.ndarray | None = None
    use_angle_betweenness: bool = False


# ============================================================================
# AuNP pick (monomer/dimer STAR) loading
# ============================================================================


def _normalize_monomer_dimer_star_pattern(
    pattern: Optional[str],
    *,
    default: str,
) -> str:
    """Return monomer/dimer STAR filename pattern (``*`` = synaptic cleft index)."""
    if pattern is None or not str(pattern).strip():
        return default
    pat = str(pattern).strip()
    if "*" not in pat or pat.count("*") != 1:
        raise ValueError(
            "Monomer/dimer STAR pattern must contain exactly one '*' for the synaptic cleft index "
            f"(e.g. {default!r})."
        )
    if not pat.endswith(".star"):
        raise ValueError("Monomer/dimer STAR pattern must end with '.star'.")
    return pat


def _monomer_dimer_star_filename(cleft_index: int, pattern: str) -> str:
    return pattern.replace("*", str(int(cleft_index)), 1)


def _resolve_monomer_dimer_star_paths(
    aunps_dir: Path,
    tomogram_name: str,
    alignment_dir: str,
    cleft_index: int,
    *,
    kind: Literal["monomer", "dimer"],
    pattern: Optional[str] = None,
) -> Path:
    default = (
        DEFAULT_MONOMER_STAR_PATTERN if kind == "monomer" else DEFAULT_DIMER_STAR_PATTERN
    )
    pat = _normalize_monomer_dimer_star_pattern(pattern, default=default)
    filename = _monomer_dimer_star_filename(cleft_index, pat)
    candidates = [
        aunps_dir / f"{tomogram_name}_{alignment_dir}_{filename}",
        aunps_dir / filename,
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"Required AuNP {kind} STAR not found for synaptic cleft {cleft_index} "
        f"(pattern {pat!r}). Tried: {[str(p) for p in candidates]}"
    )


def _find_monomer_dimer_star_path(
    aunps_dir: Path,
    tomogram_name: str,
    alignment_dir: str,
    cleft_index: int,
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
            cleft_index,
            kind=kind,
            pattern=pattern,
        )
    except FileNotFoundError:
        return None


def _read_aunp_kind_star_frame(
    path: Path,
    *,
    kind: AunpKind,
    cleft_index: int,
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
    part["cleft_index"] = int(cleft_index)
    return part


def load_monomer_dimer_aunps_for_zone(
    tomogram_path: Path,
    alignment_dir: str,
    cleft_index: int,
    *,
    monomer_star_pattern: Optional[str] = None,
    dimer_star_pattern: Optional[str] = None,
    single_pick_star_pattern: Optional[str] = None,
    use_single_pick_pool: bool = False,
) -> ZoneAunpLoadResult:
    """Load AuNP pick coordinates for one synaptic cleft index.

    When ``use_single_pick_pool`` is True, loads the general AuNP pick STAR
    (``single_pick_star_pattern``, or the default pick pattern) as kind ``all``.

    Otherwise loads monomer and/or dimer STAR files. Missing monomer or dimer files
    are skipped; ``kinds_loaded`` records which were found.
    """
    tomogram_path = Path(tomogram_path)
    alignment_dir = require_alignment_dir(alignment_dir)
    aunps_dir = tomogram_path / alignment_dir / "aunps"
    tomogram_name = tomogram_path.name

    frames: list[pd.DataFrame] = []
    kinds_loaded: list[AunpKind] = []

    if use_single_pick_pool:
        from .aunps import discover_aunp_pick_star_files, normalize_aunp_pick_star_pattern

        pat = normalize_aunp_pick_star_pattern(single_pick_star_pattern)
        found = discover_aunp_pick_star_files(
            aunps_dir, [int(cleft_index)], pattern=pat
        )
        if not found:
            raise FileNotFoundError(
                f"No AuNP pick STAR file found for synaptic cleft {cleft_index} "
                f"in {aunps_dir} (pattern {pat!r})"
            )
        path = found[0][1]
        frames.append(
            _read_aunp_kind_star_frame(path, kind="all", cleft_index=cleft_index)
        )
        kinds_loaded.append("all")
    else:
        for kind, pattern in (
            ("monomer", monomer_star_pattern),
            ("dimer", dimer_star_pattern),
        ):
            path = _find_monomer_dimer_star_path(
                aunps_dir,
                tomogram_name,
                alignment_dir,
                cleft_index,
                kind=kind,
                pattern=pattern,
            )
            if path is None:
                continue
            frames.append(
                _read_aunp_kind_star_frame(
                    path, kind=kind, cleft_index=cleft_index
                )
            )
            kinds_loaded.append(kind)

        if not frames:
            raise FileNotFoundError(
                f"No monomer or dimer AuNP STAR files found for synaptic cleft "
                f"{cleft_index} in {aunps_dir} "
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


# ============================================================================
# Active-zone surface point loading
# ============================================================================


def load_postsynaptic_cleft_surface(
    tomogram_path: Path,
    alignment_dir: str,
    zone_name: str,
) -> np.ndarray:
    az_dir = Path(tomogram_path) / alignment_dir / "STT_results" / "cleft"
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


def load_synaptic_cleft_cleft_points(
    tomogram_path: Path,
    alignment_dir: str,
    zone_name: str,
) -> np.ndarray:
    """Presynaptic + postsynaptic synaptic-cleft surface points for one zone."""
    pre = load_presynaptic_az_points_for_zone(tomogram_path, alignment_dir, zone_name)
    post = load_postsynaptic_cleft_surface(tomogram_path, alignment_dir, zone_name)
    parts = [arr for arr in (pre, post) if len(arr)]
    if not parts:
        raise FileNotFoundError(
            f"No presynaptic or postsynaptic synaptic-cleft surface points for {zone_name}"
        )
    return np.vstack(parts)


# ============================================================================
# Ripley window construction & membership tests
# ============================================================================


def build_ripley_window_3d(
    defining_coords: np.ndarray,
    mode: str,
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
    n = len(pts)
    if n == 0:
        return np.zeros(0, dtype=bool)
    normals = hull.equations[:, :-1]
    offsets = hull.equations[:, -1]
    n_facets = normals.shape[0]
    # `pts @ normals.T` materializes a dense (n_points, n_facets) matrix. Rough/noisy
    # synaptic-cleft surfaces can push n_facets into the thousands, and grid/MC point counts
    # (build_window_grid_points, _hull_betweenness_volume_mc) into the hundreds of
    # thousands, so the unchunked matrix can reach tens of GB for a single zone. Cap each
    # chunk's matrix at ~64 MB regardless of hull complexity or point count.
    chunk_size = max(1, (64 * 1024 * 1024) // (max(n_facets, 1) * 8))
    if n <= chunk_size:
        return np.all(pts @ normals.T + offsets <= tol, axis=1)
    out = np.empty(n, dtype=bool)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        out[start:end] = np.all(pts[start:end] @ normals.T + offsets <= tol, axis=1)
    return out


def _angle_betweenness_mask(
    points: np.ndarray,
    pre_membrane_coords: np.ndarray,
    post_membrane_coords: np.ndarray,
    *,
    max_membrane_distance_nm: float = ANGLE_BETWEENNESS_MAX_MEMBRANE_DISTANCE_NM,
) -> np.ndarray:
    """
    True where a point sits "between" the two membranes: the vector from the point to
    its nearest presynaptic-membrane point and the vector to its nearest
    postsynaptic-membrane point face away from each other (angle > 90 deg, i.e. the
    two vectors have a negative dot product), AND the point is within
    ``max_membrane_distance_nm`` of at least one of the two membranes (i.e.
    ``min(dist_to_nearest_pre, dist_to_nearest_post) <= max_membrane_distance_nm``).

    The angle test alone is satisfied by any point in the infinite slab between the two
    membrane planes, including far outside the lateral extent of the actual pre/post
    patches (e.g. off to the side of the synapse). The distance cap keeps the region
    anchored to the vicinity of the imaged membrane surfaces instead of that unbounded
    slab.

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
    pre_dist, pre_idx = pre_tree.query(points)
    post_dist, post_idx = post_tree.query(points)
    v_pre = pre_membrane_coords[pre_idx] - points
    v_post = post_membrane_coords[post_idx] - points
    dot = np.einsum("ij,ij->i", v_pre, v_post)
    within_distance = np.minimum(pre_dist, post_dist) <= float(max_membrane_distance_nm)
    return (dot < 0.0) & within_distance


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


# ============================================================================
# Isotropic edge correction
#
# Three interchangeable ways to estimate p(r) = Volume(Ball(focus, r) ∩ window) /
# Volume(Ball(focus, r)), the fraction of a ball's volume that survives the window —
# see each function's docstring for when it's actually used:
#
#   - ``_isotropic_edge_factors_for_foci`` (fixed-sample Monte Carlo, vectorized over
#     foci x radii): the default used by ``cross_k12_3d_isotropic`` and the fallback in
#     ``label_permutation_k_bidirectional_curves``/greedy-segregation controls when no
#     precomputed edge factors are supplied. Fast but has ~1/sqrt(N) sampling noise,
#     floored at ``EDGE_MIN_C``.
#   - ``_isotropic_edge_factors_grid`` (deterministic lattice quadrature): used by the
#     AZ-center, monomer/dimer, and fusion-point Ripley analyses instead of Monte Carlo —
#     no sampling noise, exactly reproducible, and (per its docstring) more accurate than
#     MC for a smooth 3D volume-ratio integral at a given point budget. Also backs
#     ``cross_k12_curves_per_focus`` (per-focus K₁₂ curves given precomputed edge factors).
#   - ``_isotropic_edge_factors_min_hits`` (exact-unbiased inverse-binomial MC) and
#     the scalar ``_isotropic_edge_factor_3d`` are not currently called from any
#     analysis (superseded by the grid method) — kept for reference/potential reuse.
# ============================================================================


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


def _isotropic_edge_factors_min_hits(
    centers: np.ndarray,
    r_vals: np.ndarray,
    window: RipleyWindow3D,
    rng: np.random.Generator,
    *,
    min_hits: int,
    batch_size: int = EDGE_MC_SAMPLES,
    max_samples: int = EDGE_MIN_HITS_MAX_SAMPLES,
) -> np.ndarray:
    """
    Isotropic edge factors via *exact* inverse-binomial MC sampling, for all foci x radii.

    ================================================================================
    WHAT THIS ESTIMATES
    ================================================================================
    For one focus x_i and one radius r, the "edge factor" p(r) is the fraction of the
    *volume* of the solid ball B(x_i, r) that lies inside the Ripley window W:

        p(r) = Volume(B(x_i, r) ∩ W) / Volume(B(x_i, r))

    This is the correct correction for THIS estimator's structure specifically because
    ``cross_k12_3d_isotropic`` counts, per focus, the number of observed type-2 points
    within the *cumulative* ball of radius r (not points at one exact pairwise distance).
    Under CSR with intensity λ = n2/V_window, the expected observed count for one focus is
    λ · Volume(B(x_i,r) ∩ W) — a volume, not a surface area — so dividing that observed
    count by p(r) recovers an unbiased estimate of λ · Volume(B(x_i,r)) = λ·(4/3)πr³, which
    is what makes the resulting K₁₂(r) match the CSR reference. (An earlier version of this
    docstring incorrectly suggested the classic *sphere-surface* Ripley correction applied
    here — that correction is for per-pair, exact-distance weighting, a different estimator
    structure; it does not apply to this cumulative-ball-count estimator.)

    ``p(r)`` itself is estimated by Monte Carlo: draw random points uniformly inside the
    solid ball (via ``_unit_ball_offsets``, which correctly samples ball *volume*, not
    surface) and check what fraction land inside ``window`` (via ``_window_contains``).

    ================================================================================
    WHY "STOP AT EXACTLY min_hits" GIVES AN EXACTLY UNBIASED 1/p(r) — NOT AN APPROXIMATION
    ================================================================================
    Downstream, this function's output is used as a *divisor*: ``cross_k12_3d_isotropic``
    computes ``neighbor_count / edge_factor``. So what actually needs to be unbiased is not
    p̂ itself, but 1/p̂ (the multiplicative correction applied to the observed count).

    If you draw a FIXED number of samples n and estimate p̂ = hits/n (binomial sampling),
    1/p̂ is a biased estimator of 1/p — because 1/x is convex, Jensen's inequality gives
    E[1/p̂] > 1/E[p̂] = 1/p whenever p̂ has any sampling variance. That bias grows the smaller
    n·p is, i.e. exactly in the regime this function exists to handle (small overlap
    fractions at large r).

    Instead, this function uses *inverse* (negative-binomial) sampling: draw Bernoulli(p)
    trials one at a time and stop the instant the target number of hits (``min_hits``) is
    reached, and use total_draws / min_hits as the estimate of 1/p. This is a completely
    different, and exactly unbiased, estimator — not an approximation — for the following
    reason. Let N be the number of draws needed to observe exactly k = ``min_hits``
    successes in an i.i.d. Bernoulli(p) sequence. N decomposes as N = X_1 + X_2 + ... + X_k,
    where X_j is the number of draws between the (j-1)-th and j-th success (X_1 counts up to
    the 1st success). By the memoryless property of independent Bernoulli trials, each X_j
    is itself Geometric(p) (support {1, 2, ...}), independent of the others, with
    E[X_j] = 1/p exactly. Therefore:

        E[N] = E[X_1] + ... + E[X_k] = k · (1/p)   =>   E[N/k] = 1/p   (exactly, no error term)

    This holds for *any* k >= 1 and *any* p in (0, 1] — it is an exact identity, not a
    large-sample approximation. So ``total_draws / min_hits`` is an exactly unbiased
    estimator of 1/p, with zero Jensen bias, PROVIDED the stopping point really is the
    exact draw that produced the min_hits-th success (no overshoot past it).

    ================================================================================
    WHY WE MUST TRUNCATE MID-BATCH (this is the actual code change from the prior version)
    ================================================================================
    Drawing one sample at a time in Python would be needlessly slow, so this function still
    draws in vectorized batches of ``batch_size`` samples. The earlier version of this
    function checked the hit count only *between* batches: if a batch pushed the cumulative
    hit count past ``min_hits``, it kept the *entire* batch (all ``batch_size`` draws), not
    just the draws up to and including the min_hits-th hit. That overshoot breaks the exact
    unbiasedness derived above — the actual stopping rule was "stop at the first batch
    boundary at or after min_hits hits", not "stop at the min_hits-th hit", so N was no
    longer the exact one-at-a-time inverse-sampling N.

    The fix: whenever a batch's cumulative hit count reaches or exceeds ``min_hits``, we
    look *inside* that batch (via ``np.flatnonzero`` on its boolean hit mask, which lists
    the 0-indexed positions of hits in the batch in the same left-to-right order the
    samples were drawn in — ``_window_contains`` preserves input row order) and find the
    exact position of the hit that completes the count to ``min_hits``. We then count only
    the draws up to and including that position, discarding the rest of the batch.

    This is statistically IDENTICAL to having drawn one sample at a time and stopped there,
    not merely a good approximation of it: a batch of n i.i.d. Bernoulli(p) draws generated
    all at once has exactly the same joint distribution as n draws generated one at a time,
    since each draw is independent of the others regardless of how many are pre-generated in
    one vectorized call. Truncating a pre-generated batch to its first m draws is therefore
    indistinguishable, in distribution, from having only drawn those first m one at a time
    and never having drawn the remainder at all. We only use the truncation to get exact,
    efficient stopping — it changes nothing about the underlying probability model.

    ================================================================================
    STATUS: not currently called from any analysis
    ================================================================================
    Superseded by ``_isotropic_edge_factors_grid`` (deterministic quadrature); kept here
    for reference and potential reuse rather than deleted.

    Returns an ``(n_foci, n_r)`` array; NaN at any (focus, r) where ``min_hits`` was not
    reached within ``max_samples`` draws.
    """
    centers = np.atleast_2d(np.asarray(centers, dtype=float))
    r_vals = np.asarray(r_vals, dtype=float)
    n1 = len(centers)
    n_r = len(r_vals)
    factors = np.full((n1, n_r), np.nan, dtype=float)
    for i in range(n1):
        for k, r in enumerate(r_vals):
            r = float(r)
            if r <= 0:
                factors[i, k] = 1.0
                continue
            hits = 0
            total = 0
            while hits < min_hits and total < max_samples:
                n = min(batch_size, max_samples - total)
                offsets = _unit_ball_offsets(rng, n_samples=n)
                samples = centers[i] + offsets * r
                inside = _window_contains(window, samples)  # bool, in draw order
                n_hits_in_batch = int(np.sum(inside))

                if hits + n_hits_in_batch >= min_hits:
                    # This batch contains the draw that completes the count to min_hits.
                    # Truncate to exactly that draw, discarding the rest of the batch, so
                    # `total` ends up equal to N in the exact-unbiasedness derivation above
                    # (draws-to-reach-min_hits), not draws-to-reach-the-batch-boundary.
                    needed = min_hits - hits  # how many more hits this batch must supply
                    hit_positions = np.flatnonzero(inside)  # 0-indexed, in draw order
                    cutoff_idx = hit_positions[needed - 1]  # index of the completing hit
                    total += cutoff_idx + 1  # draws consumed: 0..cutoff_idx inclusive
                    hits = min_hits
                else:
                    # Not enough hits yet even using the whole batch: consume all of it
                    # and draw another batch.
                    hits += n_hits_in_batch
                    total += n
            if hits >= min_hits:
                # hits == min_hits exactly whenever we broke out via the truncation branch
                # above; total/hits is then the exact-unbiased estimate of 1/p derived above.
                factors[i, k] = hits / total
    return factors


def build_window_grid_points(window: RipleyWindow3D, spacing_nm: float) -> np.ndarray:
    """
    Precompute a fixed, evenly-spaced 3D lattice of points filling ``window``, at
    ``spacing_nm`` resolution — the deterministic quadrature grid consumed by
    ``_isotropic_edge_factors_grid``.

    Build this ONCE per zone/window (it does not depend on r or on the focus point) and
    reuse it for every r value and every focus; see ``_isotropic_edge_factors_grid`` for how
    it's used.

    Candidate points are laid out on a regular lattice covering ``window.hull``'s
    axis-aligned bounding box, then filtered to those actually inside the window (the hull,
    additionally restricted to the angle-betweenness region when
    ``window.use_angle_betweenness``) via ``_window_contains`` — the same membership test
    used everywhere else in this module, so the grid is guaranteed consistent with the
    window definition the rest of the estimator uses.
    """
    mins = window.hull.points.min(axis=0)
    maxs = window.hull.points.max(axis=0)
    axes = [np.arange(mins[d], maxs[d] + spacing_nm, spacing_nm) for d in range(3)]
    mesh = np.meshgrid(*axes, indexing="ij")
    candidates = np.stack([m.ravel() for m in mesh], axis=1)
    inside = _window_contains(window, candidates)
    return candidates[inside]


def _isotropic_edge_factors_grid(
    centers: np.ndarray,
    r_vals: np.ndarray,
    grid_points: np.ndarray,
    spacing_nm: float,
) -> np.ndarray:
    """
    Deterministic isotropic edge factors via a fixed evenly-spaced grid (Riemann-sum
    quadrature over the window), replacing Monte Carlo sampling entirely.

    ================================================================================
    WHAT THIS ESTIMATES, AND WHY THE CONVERSION FACTOR IS h³ / ((4/3)πr³)
    ================================================================================
    As in ``_isotropic_edge_factors_min_hits``, the quantity needed is

        p(r) = Volume(Ball(x_i, r) ∩ W) / Volume(Ball(x_i, r))

    ``grid_points`` (from ``build_window_grid_points``) is a fixed lattice covering W at
    spacing ``spacing_nm`` — each point represents one grid cell of volume h³ =
    ``spacing_nm``³. For a focus x_i, ``count(r)`` = number of grid points within r of x_i
    approximates

        count(r) ≈ Volume(Ball(x_i, r) ∩ W) / h³

    (a Riemann-sum / midpoint-rule estimate of that intersection volume — the grid points
    are a fixed sample of W, and counting how many fall in the ball estimates how much of
    W's volume the ball covers). Rearranging for p(r):

        p(r) = count(r) · h³ / Volume(Ball(x_i, r)) = count(r) · h³ / ((4/3)π r³)

    ================================================================================
    WHY THIS AUTOMATICALLY REPRODUCES THE EXACT r >= R_full BEHAVIOR, WITH NO SPECIAL CASE
    ================================================================================
    Once r is large enough that Ball(x_i, r) contains the entire window (r >= R_full, the
    max distance from x_i to any point of W), every grid point is within r, so ``count(r)``
    saturates at M = ``len(grid_points)``, the total grid point count, for all larger r.
    Then p(r) = M·h³/((4/3)πr³) ≈ V/((4/3)πr³) (since M·h³ is itself a Riemann-sum estimate
    of V = Volume(W)) — exactly the closed-form result derived for that regime (where K₁₂(r)
    collapses to (4/3)πr³ and L₁₂(r) to 0 identically), falling out of this same formula
    automatically rather than needing a separate branch.

    ================================================================================
    WHY THIS SHOULD BE MORE ACCURATE THAN MONTE CARLO, NOT JUST FASTER
    ================================================================================
    ``_isotropic_edge_factors_min_hits`` estimates p(r) by random sampling, with sampling
    error that shrinks only as 1/sqrt(min_hits) — an irreducible property of stochastic
    estimation. A fixed grid is a deterministic quadrature rule instead: for a smooth,
    low-dimensional (3D) integral like this one, deterministic quadrature error typically
    shrinks faster with point count than random Monte Carlo error does (Monte Carlo's
    dimension-independent 1/sqrt(N) rate is what makes it attractive in *high* dimensions,
    not low ones — it is not the best tool for a 3D volume-ratio problem). It is also exactly
    reproducible: the same window and focus always give the same p(r), so the L₁₂(r) curve is
    smooth in r by construction rather than carrying independent per-r sampling noise (which
    was part of why the L₁₂(r) curves looked jagged before).

    ================================================================================
    THE r -> 0 EDGE CASE
    ================================================================================
    At very small r (smaller than the grid spacing), ``count(r)`` can legitimately be 0,
    which would make p(r) = 0 and blow up the division in ``cross_k12_3d_isotropic``. We
    fall back to p(r) = 1.0 in that case (matching the existing r <= 0 convention already
    used elsewhere). This is not a hack: at such small r the true observed
    ``neighbor_count(r)`` is also always 0 for real point data (AuNPs cannot be that close to
    an idealized center point), so K₁₂ = (V/n2)·(0/1) = 0 either way, correctly recovering
    the deterministic L₁₂ = -r floor without any NaN or divide-by-zero.

    We also clip p(r) to at most 1.0: the true p(r) can never exceed 1 (Ball ∩ W ⊆ Ball), so
    any grid-discretization overshoot above 1 (possible at small r, where only a handful of
    grid cells are involved) is capped rather than allowed to under-correct the count.

    ================================================================================
    WHERE THIS IS CALLED FROM
    ================================================================================
    Called once per zone from ``run_aunp_vs_az_center_ripley_for_zone`` in
    aunp_ripley_vs_cleft_center.py, immediately after ``build_window_grid_points``
    builds the grid for that zone's window (also once per zone, independent of r). Its
    output replaces ``_isotropic_edge_factors_min_hits``'s output as the ``edge_factors``
    argument to ``cross_k12_3d_isotropic``.

    Returns an ``(n_foci, n_r)`` array; no NaNs (unlike the Monte Carlo version, this method
    always returns a usable value — see the r -> 0 case above for the one situation where
    the raw computation would otherwise be ill-defined).
    """
    centers = np.atleast_2d(np.asarray(centers, dtype=float))
    r_vals = np.asarray(r_vals, dtype=float)
    grid_points = np.atleast_2d(np.asarray(grid_points, dtype=float))
    n1 = len(centers)
    cell_volume = float(spacing_nm) ** 3
    with np.errstate(divide="ignore", invalid="ignore"):
        ball_volumes = (4.0 / 3.0) * np.pi * r_vals**3

    factors = np.empty((n1, len(r_vals)), dtype=float)
    for i in range(n1):
        dists = np.sort(np.linalg.norm(grid_points - centers[i], axis=1))
        counts = np.searchsorted(dists, r_vals, side="left")
        with np.errstate(divide="ignore", invalid="ignore"):
            p = counts * cell_volume / ball_volumes
        p = np.where((r_vals <= 0) | (counts == 0), 1.0, p)
        factors[i] = np.minimum(p, 1.0)
    return factors


# ============================================================================
# Shared geometry diagnostic figure
# ============================================================================


def plot_ripley_window_geometry_diagnostic(
    tomogram_path: Path,
    alignment_dir: str,
    zone_name: str,
    *,
    point_groups: Sequence[dict],
    az_segmentation: dict,
    window: RipleyWindow3D,
    grid_points: np.ndarray,
    grid_spacing_nm: float,
    output_path: Path,
    dropped_coords: np.ndarray | None = None,
    center_point: np.ndarray | None = None,
    center_label: str = "center",
    title_lines: Sequence[str] = (),
    membrane_max_points: int = 4000,
    n_z_slices: int = 5,
    include_3d_panel: bool = True,
    print_prefix: str = "Geometry diagnostic",
) -> Path | None:
    """
    Shared QC figure for the AuNP Ripley analyses: one or more point groups (e.g. monomer +
    dimer AuNPs, or a single AZ-center AuNP set), pre/post membrane surfaces, the convex
    hull used as the Ripley window, and the deterministic edge-correction grid points (from
    ``build_window_grid_points``) that are the actual sample set the analysis divides by —
    not a separate Monte-Carlo preview of it, but the literal points used, so this figure
    always matches what the computation really did (grid points are already restricted to
    the effective window: inside the hull, additionally restricted to the angle
    in-betweenness region when ``window.use_angle_betweenness`` is set).

    ``point_groups``: sequence of dicts, each ``{"coords", "label", "color"}`` plus optional
    ``"marker"`` (default ``"o"``) and ``"size"`` (default ``18``) — one scatter series per
    point set being visualized (e.g. monomer + dimer AuNPs get two groups; AZ-center gets
    one "AuNPs" group).

    ``dropped_coords`` (points outside the window's hull, excluded from the Ripley
    analysis) are highlighted separately. ``center_point`` (optional) overlays a single star
    marker (e.g. a computed AZ/cleft center) in both the 3D panel and whichever z-band panel
    it falls in — omit for analyses with no single "center" concept (e.g. monomer/dimer).

    XY-projection z-sliced panels, optionally preceded by a 3D overview panel (see
    ``include_3d_panel``); ``title_lines`` are joined after the standard
    tomogram/zone header line.
    """
    tomogram_path = Path(tomogram_path)
    alignment_dir = require_alignment_dir(alignment_dir)
    groups = [
        {
            "coords": np.atleast_2d(np.asarray(g["coords"], dtype=float)),
            "label": str(g["label"]),
            "color": g.get("color", "tab:red"),
            "marker": g.get("marker", "o"),
            "size": g.get("size", 18),
        }
        for g in point_groups
    ]
    dropped_coords = (
        np.atleast_2d(np.asarray(dropped_coords, dtype=float))
        if dropped_coords is not None and len(dropped_coords)
        else np.zeros((0, 3))
    )
    center_point = (
        np.asarray(center_point, dtype=float).reshape(3)
        if center_point is not None and np.all(np.isfinite(np.asarray(center_point, dtype=float)))
        else None
    )

    pre_outer = np.atleast_2d(np.asarray(az_segmentation.get("presynaptic_outer_coords", []), dtype=float))
    post_outer = np.atleast_2d(np.asarray(az_segmentation.get("postsynaptic_outer_coords", []), dtype=float))

    hull = window.hull
    hull_pts = hull.points

    pre_plot = _downsample_points(pre_outer, membrane_max_points, seed=1)
    post_plot = _downsample_points(post_outer, membrane_max_points, seed=2)

    grid_points = np.atleast_2d(np.asarray(grid_points, dtype=float)) if len(grid_points) else np.zeros((0, 3))
    grid_plot = _downsample_points(grid_points, 20_000, seed=4)  # figures stay light even for big grids
    grid_volume_nm3 = float(len(grid_points)) * (float(grid_spacing_nm) ** 3)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def _scatter2d(ax, pts, *, z_lo, z_hi, **kwargs):
        pts = pts[(pts[:, 2] >= z_lo) & (pts[:, 2] <= z_hi)] if len(pts) else pts
        if len(pts) == 0:
            return
        ax.scatter(pts[:, 0], pts[:, 1], **kwargs)

    n_z_slices = max(1, int(n_z_slices))
    z_source_arrays = [hull_pts, pre_outer, post_outer, dropped_coords] + [g["coords"] for g in groups]
    if center_point is not None:
        z_source_arrays.append(center_point.reshape(1, 3))
    z_values = np.concatenate([arr[:, 2] for arr in z_source_arrays if len(arr)])
    z_min, z_max = float(z_values.min()), float(z_values.max())
    if z_max <= z_min:
        z_max = z_min + 1.0
    z_edges = np.linspace(z_min, z_max, n_z_slices + 1)

    if include_3d_panel:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d projection)
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        fig = plt.figure(figsize=(4.0 * max(n_z_slices, 3), 9.0))
        gs = fig.add_gridspec(2, n_z_slices, height_ratios=[1.3, 1.0])
        ax3d = fig.add_subplot(gs[0, :], projection="3d")
        axes = [fig.add_subplot(gs[1, k]) for k in range(n_z_slices)]

        def _scatter3d(pts, **kwargs):
            if len(pts) == 0:
                return
            ax3d.scatter(pts[:, 0], pts[:, 1], pts[:, 2], **kwargs)

        _scatter3d(grid_plot, s=2, c="mediumseagreen", alpha=0.15, label="edge-correction grid")
        _scatter3d(pre_plot, s=2, c="tab:blue", alpha=0.15, label="presynaptic membrane")
        _scatter3d(post_plot, s=2, c="tab:orange", alpha=0.15, label="postsynaptic membrane")
        for g in groups:
            _scatter3d(
                g["coords"], s=g["size"] + 7, c=g["color"], marker=g["marker"], label=g["label"],
                edgecolors="k", linewidths=0.3,
            )
        _scatter3d(
            dropped_coords, s=40, c="black", marker="x", label="dropped (outside hull)", linewidths=1.5,
        )
        if center_point is not None:
            ax3d.scatter(*center_point, s=220, c="black", marker="*", label=center_label, zorder=6)
        hull_faces = Poly3DCollection(
            hull_pts[hull.simplices], alpha=0.06, facecolor="tab:green", edgecolor="tab:green", linewidths=0.3
        )
        ax3d.add_collection3d(hull_faces)
        ax3d.set_xlabel("x (nm)")
        ax3d.set_ylabel("y (nm)")
        ax3d.set_zlabel("z (nm)")
        ax3d.set_title("3D view")
        ax3d.legend(loc="upper left", fontsize=6, markerscale=1.5)
        top_rect = (0.0, 0.06, 1.0, 0.90)
    else:
        fig, axes = plt.subplots(1, n_z_slices, figsize=(4.0 * n_z_slices, 4.5), squeeze=False)
        axes = list(axes[0])
        top_rect = (0.0, 0.06, 1.0, 0.88)

    for k, ax in enumerate(axes):
        z_lo, z_hi = float(z_edges[k]), float(z_edges[k + 1])
        # Hull footprint restricted to this z-band: the 2D hull of only the (densely
        # sampled) defining points whose z falls in this band, so the outline tracks the
        # true local cross-section instead of the whole hull's full-depth XY shadow.
        band_hull_pts = hull_pts[(hull_pts[:, 2] >= z_lo) & (hull_pts[:, 2] <= z_hi)]
        if len(band_hull_pts) >= 3:
            try:
                band_hull_2d = ConvexHull(band_hull_pts[:, [0, 1]])
                band_hull_vertices = band_hull_2d.points[band_hull_2d.vertices]
                poly = plt.Polygon(
                    band_hull_vertices, closed=True, fill=True, facecolor="tab:green",
                    edgecolor="tab:green", alpha=0.08, linewidth=1.0,
                    label="cleft hull (XY footprint, this z-band)",
                )
                ax.add_patch(poly)
            except Exception:
                pass
        _scatter2d(
            ax, grid_plot, z_lo=z_lo, z_hi=z_hi, s=4, c="mediumseagreen", alpha=0.3,
            label="edge-correction grid", zorder=1,
        )
        _scatter2d(ax, pre_plot, z_lo=z_lo, z_hi=z_hi, s=2, c="tab:blue", alpha=0.12, label="presynaptic membrane")
        _scatter2d(ax, post_plot, z_lo=z_lo, z_hi=z_hi, s=2, c="tab:orange", alpha=0.12, label="postsynaptic membrane")
        for g in groups:
            _scatter2d(
                ax, g["coords"], z_lo=z_lo, z_hi=z_hi, s=g["size"], c=g["color"], marker=g["marker"],
                label=g["label"], edgecolors="k", linewidths=0.3, zorder=5,
            )
        _scatter2d(
            ax, dropped_coords, z_lo=z_lo, z_hi=z_hi, s=40, c="black", marker="x",
            label="dropped (outside hull)", linewidths=1.5, zorder=6,
        )
        if center_point is not None and z_lo <= center_point[2] <= z_hi:
            ax.scatter(center_point[0], center_point[1], s=180, c="black", marker="*", label=center_label, zorder=6)
        ax.set_xlabel("x (nm)")
        if k == 0:
            ax.set_ylabel("y (nm)")
        ax.set_title(f"z: {z_lo:.0f}–{z_hi:.0f} nm")
        ax.set_aspect("equal", adjustable="datalim")

    legend_by_label: dict[str, object] = {}
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            legend_by_label.setdefault(label, handle)
    fig.legend(
        list(legend_by_label.values()), list(legend_by_label.keys()),
        loc="lower center", ncol=len(legend_by_label), fontsize=7,
    )

    dropped_line = f"\nDropped outside hull: {len(dropped_coords)} point(s)" if len(dropped_coords) else ""
    header = f"{tomogram_path.name} | {zone_name}"
    body = "\n".join(
        [
            *title_lines,
            f"Edge-correction grid: {len(grid_points)} points @ {grid_spacing_nm:g}nm spacing "
            f"(~{grid_volume_nm3:.3e} nm³)",
        ]
    )
    fig.suptitle(f"{header}\n{body}{dropped_line}")
    fig.tight_layout(rect=top_rect)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {print_prefix} ({zone_name}) -> {output_path}")
    return output_path


# ============================================================================
# Pair-correlation function (from K difference) and the K/L/g symmetric combination
# ============================================================================


def _intensity_weighted_combination(
    value_kl: np.ndarray,
    value_lk: np.ndarray,
    n_k: int,
    n_l: int,
) -> np.ndarray:
    """
    Symmetrized combination of a type-k-focused and a type-l-focused bivariate estimate:

        value_(kl,lk) = (λ_k·value_kl + λ_l·value_lk) / (λ_k + λ_l)

    with λ_k = n_k/V, λ_l = n_l/V for the shared window volume V — V cancels in the ratio,
    leaving weights n_k, n_l. This is the standard symmetrized bivariate-K combination
    (Lotwick & Silverman 1982), applied here to any per-r curve pair (K, L, or g) computed
    in both directions.
    """
    value_kl = np.asarray(value_kl, dtype=float)
    value_lk = np.asarray(value_lk, dtype=float)
    denom = float(n_k) + float(n_l)
    if denom <= 0:
        return np.full(np.broadcast_shapes(value_kl.shape, value_lk.shape), np.nan)
    return (float(n_k) * value_kl + float(n_l) * value_lk) / denom


def _uniform_shell_bin_edges(
    r_vals: np.ndarray,
    bin_width_nm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Shared bin-edge construction for every uniform-shell estimator in this module
    (``pair_correlation_from_k_diff``, ``count_grid_points_per_shell``, and the local
    kind-composition ratio shell counts): ``r_vals`` is padded with an implicit 0 so the
    innermost bin is ``[0, bin_width_nm)``, then grouped into bins of ``bin_width_nm``
    (rounded to the nearest whole multiple of ``r_vals``'s step; any trailing remainder
    that doesn't fill a complete bin is dropped).

    Returns ``(r_lo_nm, r_hi_nm, idx_lo, idx_hi)`` — ``idx_lo``/``idx_hi`` index into the
    0-padded array (i.e. into ``np.concatenate([[0.0], r_vals])``, or an equally-padded
    curve/count array), one pair per bin.
    """
    r_vals = np.asarray(r_vals, dtype=float)
    if len(r_vals) < 1:
        empty = np.array([])
        return empty, empty, np.array([], dtype=int), np.array([], dtype=int)
    r_step = float(r_vals[0]) if len(r_vals) == 1 else float(r_vals[1] - r_vals[0])
    stride = max(1, int(round(float(bin_width_nm) / r_step))) if r_step > 0 else 1
    r_padded = np.concatenate([[0.0], r_vals])
    n_bins = (len(r_padded) - 1) // stride
    idx_lo = np.arange(n_bins) * stride
    idx_hi = idx_lo + stride
    return r_padded[idx_lo], r_padded[idx_hi], idx_lo, idx_hi


def pair_correlation_from_k_diff(
    k_vals: np.ndarray,
    r_vals: np.ndarray,
    *,
    bin_width_nm: float,
) -> dict[str, np.ndarray]:
    """
    Pair-correlation function as a finite difference of an already-computed, edge-corrected
    K(r) curve — the exact discretization of g(r) = K'(r)/(4πr²) via the true shell volume
    rather than the small-shell approximation 4πr²dr:

        g(d) = [K(d + dr) - K(d)] / ((4/3)π[(d + dr)³ - d³])

    Because K already carries whatever edge correction/window normalization it was computed
    with, g inherits it automatically — unlike an independent shell-count-ratio estimator,
    no separate "expected count under CSR" computation (grid quadrature, reliability
    threshold, etc.) is needed here at all.

    ``r_vals`` must be the uniform grid ``k_vals`` was evaluated on (e.g. ``_ripley_r_grid``'s
    output, starting at one step past 0); K(0) = 0 is used implicitly so the innermost bin is
    [0, bin_width_nm). ``bin_width_nm`` may be any positive multiple of ``r_vals``'s step —
    rounded to the nearest whole multiple, so passing the step itself gives the finest
    possible resolution and a larger value gives a coarser one computed directly from the
    same K curve (not by re-binning an intermediate result). Any trailing remainder of
    ``r_vals`` that doesn't fill a complete bin is dropped rather than raising, so a
    unfavorable r_max/bin_width combination on one zone doesn't abort a batch run.

    ``k_vals`` may be a single curve, shape ``(n_r,)`` (e.g. one zone's observed K), or a
    batch of curves, shape ``(n_curves, n_r)`` (e.g. a whole label-permutation/greedy-
    segregation null ensemble) — every curve in a batch shares the same ``r_vals``/bin edges,
    so the batched form is a single vectorized computation, not a Python loop.

    Returns ``r_lo_nm``, ``r_hi_nm``, ``r_mid_nm`` (each shape ``(n_bins,)``) and ``pcf``
    (shape ``(n_bins,)`` for a single input curve, ``(n_curves, n_bins)`` for a batch).
    """
    r_vals = np.asarray(r_vals, dtype=float)
    k_vals = np.asarray(k_vals, dtype=float)
    is_1d = k_vals.ndim == 1
    k_mat = np.atleast_2d(k_vals)
    if len(r_vals) < 1:
        empty_pcf = np.array([]) if is_1d else np.empty((k_mat.shape[0], 0))
        return {
            "r_lo_nm": np.array([]),
            "r_hi_nm": np.array([]),
            "r_mid_nm": np.array([]),
            "pcf": empty_pcf,
        }

    # Prepend the implicit K(0) = 0 point so the innermost bin is [0, bin_width_nm).
    k_padded = np.concatenate([np.zeros((k_mat.shape[0], 1)), k_mat], axis=1)
    r_lo, r_hi, idx_lo, idx_hi = _uniform_shell_bin_edges(r_vals, bin_width_nm)
    shell_volume = (4.0 / 3.0) * np.pi * (r_hi**3 - r_lo**3)
    with np.errstate(divide="ignore", invalid="ignore"):
        pcf_mat = (k_padded[:, idx_hi] - k_padded[:, idx_lo]) / shell_volume[None, :]

    return {
        "r_lo_nm": r_lo,
        "r_hi_nm": r_hi,
        "r_mid_nm": 0.5 * (r_lo + r_hi),
        "pcf": pcf_mat[0] if is_1d else pcf_mat,
    }


def count_grid_points_per_shell(
    centers: np.ndarray,
    grid_points: np.ndarray,
    r_vals: np.ndarray,
    *,
    bin_width_nm: float,
) -> dict[str, np.ndarray]:
    """
    For each focus in ``centers``, count how many of ``grid_points`` fall within each shell
    — on the exact same bin edges ``pair_correlation_from_k_diff`` uses for that ``r_vals``/
    ``bin_width_nm`` (uniform grid, K(0) = 0 implicit so the innermost bin is
    [0, bin_width_nm)).

    This is how much of the window's quadrature grid is actually available to support a
    shell's volume estimate as seen from each focus — the same quantity the retired
    independent shell-count g estimator used for its reliability threshold.
    ``pair_correlation_from_k_diff`` can't see this on its own (it only has the
    already-integrated K curve), so it's computed here separately and combined by the caller
    (see ``g_shell_reliability_mask``) to catch the situation where a focus's ball has
    already grown past the point where the entire window is inside it: every K value
    entering that shell's difference is then forced onto the CSR reference curve by
    construction (see ``_isotropic_edge_factors_grid``'s docstring), so the resulting g isn't
    measuring anything, even though ``pair_correlation_from_k_diff`` has no way to know that
    on its own.

    Returns ``r_lo_nm``, ``r_hi_nm``, ``r_mid_nm`` (shape ``(n_bins,)``) and ``counts``
    (shape ``(n_foci, n_bins)``).
    """
    centers = np.atleast_2d(np.asarray(centers, dtype=float))
    grid_points = np.atleast_2d(np.asarray(grid_points, dtype=float))
    r_vals = np.asarray(r_vals, dtype=float)
    if len(r_vals) < 1:
        return {
            "r_lo_nm": np.array([]),
            "r_hi_nm": np.array([]),
            "r_mid_nm": np.array([]),
            "counts": np.empty((len(centers), 0), dtype=int),
        }

    r_lo, r_hi, idx_lo, idx_hi = _uniform_shell_bin_edges(r_vals, bin_width_nm)
    n_bins = len(idx_lo)
    r_edges = np.concatenate([r_lo, r_hi[-1:]]) if n_bins else np.array([0.0])

    counts = np.zeros((len(centers), n_bins), dtype=int)
    for i, c in enumerate(centers):
        dists = (
            np.sort(np.linalg.norm(grid_points - c, axis=1)) if len(grid_points) else np.array([])
        )
        cum = np.searchsorted(dists, r_edges, side="left")
        counts[i] = np.diff(cum)

    return {
        "r_lo_nm": r_lo,
        "r_hi_nm": r_hi,
        "r_mid_nm": 0.5 * (r_lo + r_hi),
        "counts": counts,
    }


def g_shell_reliability_mask(
    centers: np.ndarray,
    grid_points: np.ndarray,
    r_vals: np.ndarray,
    *,
    bin_width_nm: float,
    min_grid_points_per_shell: int = G12_MIN_GRID_POINTS_PER_SHELL,
) -> np.ndarray:
    """
    Boolean mask, shape ``(n_bins,)``, True where a shell (same ``r_vals``/``bin_width_nm``
    as ``pair_correlation_from_k_diff``) is NOT well supported by the window quadrature grid
    as seen from ``centers`` — summed across all of them if there's more than one (the same
    sum-before-ratio principle used elsewhere in this module: a single low-support focus
    shouldn't dominate a many-foci direction's reliability signal any more than it dominates
    that direction's K value).
    """
    counts = count_grid_points_per_shell(
        centers, grid_points, r_vals, bin_width_nm=bin_width_nm
    )["counts"]
    total = counts.sum(axis=0)
    return total < min_grid_points_per_shell


# ============================================================================
# K₁₂ / L₁₂ estimator and transforms
# ============================================================================


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


def mean_l_from_k_curves(k_curves: np.ndarray, r_vals: np.ndarray) -> np.ndarray:
    """``L(nanmean(K, axis=0), r)`` — average replicate K curves, convert once with ``ripley_l12``."""
    r_vals = np.asarray(r_vals, dtype=float)
    k_mat = np.atleast_2d(np.asarray(k_curves, dtype=float))
    if k_mat.size == 0 or k_mat.shape[0] == 0:
        return np.full(len(r_vals), np.nan)
    return ripley_l12(np.nanmean(k_mat, axis=0), r_vals)


def mean_l12_from_averaged_k12(
    l12_curves: np.ndarray | None,
    r_vals: np.ndarray,
    *,
    k12_curves: np.ndarray | None = None,
) -> np.ndarray:
    """
    Pool on the K₁₂ scale, then convert the mean K back to L₁₂.

    Prefer ``k12_curves`` when available. If only L₁₂ curves are passed, each is inverted
    to K₁₂ first (valid for invertible single-curve-per-row summaries; do not use L→K
    inversion as a substitute for averaging raw L replicates when K is available).
    Empty input yields an all-NaN L₁₂ vector.
    """
    return prism_sd_envelope_columns_from_averaged_k12(
        l12_curves, r_vals, prefix="tmp", k12_curves=k12_curves
    )["tmp_mean"]


def prism_sd_envelope_columns_from_averaged_k12(
    l12_curves: np.ndarray | None,
    r_vals: np.ndarray,
    *,
    prefix: str,
    k12_curves: np.ndarray | None = None,
    weights: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """
    Mean ± SD/SEM on the K₁₂ scale, then map mean and mean±SD/SEM through ``ripley_l12``.

    Prefer ``k12_curves`` when available. Passing only ``l12_curves`` inverts L→K first —
    acceptable for invertible single-curve-per-row zone summaries, not as a substitute for
    averaging raw L replicates when stored K is available.

    Because L₁₂(K) is nonlinear, these envelopes differ from mean±SD computed on L₁₂.
    Percentile envelopes of L and of K→L are identical (monotone transform), so they are
    not duplicated here.

    ``weights`` (e.g. AuNP partner count per curve): when given, the mean and the
    variance it's centered on are weighted (see ``_weighted_nanmean``) instead of treating
    every curve as equally informative — so a curve backed by very little data can't
    dominate the average the way an unweighted mean would let it.

    Columns (primary / only K→L reporting): ``{prefix}_mean``, ``{prefix}_sd`` (SD of K),
    ``{prefix}_sd_envelope_{lo,hi}``, ``{prefix}_sem`` (SEM of K),
    ``{prefix}_sem_envelope_{lo,hi}``.
    """
    r_vals = np.asarray(r_vals, dtype=float)
    nan = np.full(len(r_vals), np.nan)
    empty = {
        f"{prefix}_mean": nan.copy(),
        f"{prefix}_sd": nan.copy(),
        f"{prefix}_sd_envelope_lo": nan.copy(),
        f"{prefix}_sd_envelope_hi": nan.copy(),
        f"{prefix}_sem": nan.copy(),
        f"{prefix}_sem_envelope_lo": nan.copy(),
        f"{prefix}_sem_envelope_hi": nan.copy(),
    }
    if k12_curves is None and l12_curves is None:
        return empty
    k_mat = _k12_curves_matrix(
        np.empty((0, len(r_vals))) if l12_curves is None else l12_curves,
        r_vals,
        k12_curves=k12_curves,
    )
    if k_mat.size == 0 or k_mat.shape[0] == 0:
        return empty

    mean_k = _weighted_nanmean(k_mat, weights)
    n_valid = np.sum(~np.isnan(k_mat), axis=0)
    if weights is None:
        with np.errstate(invalid="ignore"):
            sd_k = np.nanstd(k_mat, axis=0, ddof=1)
            sem_k = sd_k / np.sqrt(np.maximum(n_valid, 1))
    else:
        w = np.asarray(weights, dtype=float).reshape(-1, 1)
        valid = ~np.isnan(k_mat)
        w_masked = np.where(valid, w, 0.0)
        wsum = w_masked.sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            var_k = np.nansum(w_masked * (k_mat - mean_k) ** 2, axis=0) / wsum
        sd_k = np.sqrt(var_k)
        sem_k = sd_k / np.sqrt(np.maximum(n_valid, 1))
    sd_k = np.where(n_valid > 1, sd_k, 0.0)
    sem_k = np.where(n_valid > 1, sem_k, 0.0)

    return {
        f"{prefix}_mean": ripley_l12(mean_k, r_vals),
        f"{prefix}_sd": sd_k,
        f"{prefix}_sd_envelope_lo": ripley_l12(mean_k - sd_k, r_vals),
        f"{prefix}_sd_envelope_hi": ripley_l12(mean_k + sd_k, r_vals),
        f"{prefix}_sem": sem_k,
        f"{prefix}_sem_envelope_lo": ripley_l12(mean_k - sem_k, r_vals),
        f"{prefix}_sem_envelope_hi": ripley_l12(mean_k + sem_k, r_vals),
    }


def cross_k12_curves_per_focus(
    x: np.ndarray,
    y: np.ndarray,
    r_vals: np.ndarray,
    window: RipleyWindow3D,
    *,
    edge_factors: np.ndarray,
) -> np.ndarray:
    """
    Per-focus K₁₂ curves (one row per element of ``x``), given precomputed isotropic edge
    factors (e.g. from ``_isotropic_edge_factors_grid``) for each focus.

    Unlike ``cross_k12_3d_isotropic`` (which aggregates all of ``x`` into a single pooled
    curve), this keeps one K₁₂(r) curve per individual focus — used where each query point
    (e.g. one fusion site) needs its own curve rather than a zone-wide aggregate.

    Returns ``(n_foci, n_r)``.
    """
    x = np.atleast_2d(np.asarray(x, dtype=float))
    y = np.atleast_2d(np.asarray(y, dtype=float))
    r_vals = np.asarray(r_vals, dtype=float)
    n1, n2 = len(x), len(y)
    if n1 == 0:
        return np.empty((0, len(r_vals)))
    if n2 == 0 or window.volume_nm3 <= 0:
        return np.full((n1, len(r_vals)), np.nan)

    edge_factors = np.asarray(edge_factors, dtype=float)
    if edge_factors.shape != (n1, len(r_vals)):
        raise ValueError(
            f"edge_factors shape {edge_factors.shape} != expected {(n1, len(r_vals))}"
        )

    tree = cKDTree(y)
    r_max = float(r_vals[-1])
    neighbor_lists = tree.query_ball_point(x, r=r_max)
    k12 = np.zeros((n1, len(r_vals)), dtype=float)
    scale = window.volume_nm3 / float(n2)
    for i, neighbor_idx in enumerate(neighbor_lists):
        if not neighbor_idx:
            continue
        dists = np.linalg.norm(y[np.asarray(neighbor_idx, dtype=int)] - x[i], axis=1)
        neighbor_counts = (dists[:, None] < r_vals[None, :]).sum(axis=0).astype(float)
        k12[i] = scale * (neighbor_counts / edge_factors[i])
    return k12


def cross_k_bivariate_symmetric_3d_isotropic(
    x: np.ndarray,
    y: np.ndarray,
    r_vals: np.ndarray,
    window: RipleyWindow3D,
    rng: np.random.Generator,
    *,
    edge_factors_xy: np.ndarray | None = None,
    edge_factors_yx: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """
    Symmetrized bivariate K: K_(12,21)(r) = (λ1·K12(r) + λ2·K21(r)) / (λ1 + λ2) — the
    standard combination for a two-type point process (Lotwick & Silverman 1982). K12 and
    K21 both estimate the same theoretical cross-K under a stationary bivariate process, but
    have different variances driven by each direction's point counts, so this pools them
    into one curve weighted by each type's intensity (see
    ``_intensity_weighted_combination`` for the λ_k = n_k/V cancellation).

    K12 uses ``x`` as foci / ``y`` as partners; K21 swaps their roles. Edge correction is
    recomputed independently for each direction (it depends on which set is the foci) unless
    the corresponding precomputed ``edge_factors_xy``/``edge_factors_yx`` is supplied.

    Returns ``k12``, ``k21``, and ``k_combined`` (each shape ``(n_r,)``).
    """
    x = np.atleast_2d(np.asarray(x, dtype=float))
    y = np.atleast_2d(np.asarray(y, dtype=float))
    n1, n2 = len(x), len(y)
    k12 = cross_k12_3d_isotropic(x, y, r_vals, window, rng, edge_factors=edge_factors_xy)
    k21 = cross_k12_3d_isotropic(y, x, r_vals, window, rng, edge_factors=edge_factors_yx)
    k_combined = _intensity_weighted_combination(k12, k21, n1, n2)
    return {"k12": k12, "k21": k21, "k_combined": k_combined}


def ripley_l_bivariate_symmetric_3d_isotropic(
    x: np.ndarray,
    y: np.ndarray,
    r_vals: np.ndarray,
    window: RipleyWindow3D,
    rng: np.random.Generator,
    *,
    edge_factors_xy: np.ndarray | None = None,
    edge_factors_yx: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """
    L-transform (``ripley_l12``) of each of ``cross_k_bivariate_symmetric_3d_isotropic``'s
    three K curves. ``l_combined`` transforms ``k_combined`` directly rather than averaging
    ``l12``/``l21`` — pooling on the K scale first and transforming once, consistent with
    ``prism_sd_envelope_columns_from_averaged_k12``'s reasoning: L is a concave function of
    K, so averaging L directly is biased low relative to averaging K first and transforming.

    Returns ``k12``, ``k21``, ``k_combined``, ``l12``, ``l21``, ``l_combined``.
    """
    k_result = cross_k_bivariate_symmetric_3d_isotropic(
        x,
        y,
        r_vals,
        window,
        rng,
        edge_factors_xy=edge_factors_xy,
        edge_factors_yx=edge_factors_yx,
    )
    r_vals = np.asarray(r_vals, dtype=float)
    return {
        **k_result,
        "l12": ripley_l12(k_result["k12"], r_vals),
        "l21": ripley_l12(k_result["k21"], r_vals),
        "l_combined": ripley_l12(k_result["k_combined"], r_vals),
    }


def _ripley_r_grid(r_max_nm: float, r_step_nm: float) -> np.ndarray:
    n_steps = max(1, int(np.floor(r_max_nm / r_step_nm)))
    return np.arange(r_step_nm, r_max_nm + 0.5 * r_step_nm, r_step_nm, dtype=float)


# ============================================================================
# MAD (maximum absolute deviation) null-hypothesis tests
# ============================================================================


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


# ============================================================================
# Label-permutation null curves (parallelizable)
# ============================================================================

# Shared context for parallel label-permutation bidirectional-K workers (pool initializer).
_PERM_BIDIR_CTX: dict = {}


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


def _init_label_perm_bidir_worker(
    pool: np.ndarray,
    r_vals: np.ndarray,
    window: RipleyWindow3D,
    n_monomer: int,
    pool_edge_factors: np.ndarray,
) -> None:
    _PERM_BIDIR_CTX["pool"] = pool
    _PERM_BIDIR_CTX["r_vals"] = r_vals
    _PERM_BIDIR_CTX["window"] = window
    _PERM_BIDIR_CTX["n_monomer"] = int(n_monomer)
    _PERM_BIDIR_CTX["pool_edge_factors"] = pool_edge_factors


def _label_perm_bidir_worker(task: tuple[int, int]) -> tuple[int, np.ndarray, np.ndarray]:
    """One label-permuted (K12, K21) pair (perm_id, seed)."""
    perm_id, seed = task
    pool = _PERM_BIDIR_CTX["pool"]
    r_vals = _PERM_BIDIR_CTX["r_vals"]
    window = _PERM_BIDIR_CTX["window"]
    n_monomer = _PERM_BIDIR_CTX["n_monomer"]
    pool_edge_factors = _PERM_BIDIR_CTX["pool_edge_factors"]
    rng = np.random.default_rng(int(seed))
    class1_idx = rng.choice(len(pool), n_monomer, replace=False)
    mask = np.zeros(len(pool), dtype=bool)
    mask[class1_idx] = True
    k12 = cross_k12_3d_isotropic(
        pool[mask], pool[~mask], r_vals, window, rng, edge_factors=pool_edge_factors[mask]
    )
    k21 = cross_k12_3d_isotropic(
        pool[~mask], pool[mask], r_vals, window, rng, edge_factors=pool_edge_factors[~mask]
    )
    return int(perm_id), k12, k21


def _label_permutation_k_bidirectional_curves_sequential(
    pool: np.ndarray,
    n_monomer: int,
    r_vals: np.ndarray,
    window: RipleyWindow3D,
    rng: np.random.Generator,
    *,
    n_perm: int,
    pool_edge_factors: np.ndarray,
    pbar=None,
) -> dict[str, np.ndarray]:
    """Sequential label-permutation null (K12, K21) curve pairs."""
    n_pool = len(pool)
    k12_curves = np.full((int(n_perm), len(r_vals)), np.nan, dtype=float)
    k21_curves = np.full((int(n_perm), len(r_vals)), np.nan, dtype=float)
    for perm_id in range(int(n_perm)):
        class1_idx = rng.choice(n_pool, n_monomer, replace=False)
        mask = np.zeros(n_pool, dtype=bool)
        mask[class1_idx] = True
        k12_curves[perm_id] = cross_k12_3d_isotropic(
            pool[mask], pool[~mask], r_vals, window, rng, edge_factors=pool_edge_factors[mask]
        )
        k21_curves[perm_id] = cross_k12_3d_isotropic(
            pool[~mask], pool[mask], r_vals, window, rng, edge_factors=pool_edge_factors[~mask]
        )
        if pbar is not None:
            pbar.set_postfix_str(f"perm {perm_id + 1}/{int(n_perm)}", refresh=False)
            pbar.update(1)
    return {"k12": k12_curves, "k21": k21_curves}


def label_permutation_k_bidirectional_curves(
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
    pool_edge_factors: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """
    Label-permutation null K curves in BOTH directions per replicate, with optional process
    parallelism: K12 (the size-n_monomer relabeled class as foci) and K21 (the size-n_dimer
    relabeled class as foci). Kept on the K scale (not transformed to L) so callers can
    derive L12/L21/L_combined and g12/g21/g_combined (via ``derive_symmetric_k_l_g_families``)
    from the same null draw, rather than needing a separate null generator per statistic.

    Pool monomer + dimer points, then for each replicate randomly relabel exactly
    ``n_monomer`` points as class 1 and the rest as class 2. Returns
    ``{"k12": (n_perm, n_r), "k21": (n_perm, n_r)}``.

    ``pool_edge_factors`` may supply a precomputed ``(n_pool, n_r)`` edge-correction matrix
    for every pooled point treated as a potential focus (e.g. via the deterministic grid
    method, ``_isotropic_edge_factors_grid``) — reused for BOTH directions of every replicate
    (``pool_edge_factors[mask]`` / ``[~mask]``) since it already covers every pooled point.
    Falls back to the isotropic Monte Carlo estimate if not supplied.
    """
    pool = np.vstack([np.atleast_2d(monomer_coords), np.atleast_2d(dimer_coords)])
    n_pool = len(pool)
    n_monomer = len(np.atleast_2d(monomer_coords))
    n_perm_int = int(n_perm)
    empty = {
        "k12": np.full((n_perm_int, len(r_vals)), np.nan, dtype=float),
        "k21": np.full((n_perm_int, len(r_vals)), np.nan, dtype=float),
    }
    if n_pool == 0 or n_monomer == 0 or n_monomer >= n_pool or n_perm_int == 0:
        return empty

    if n_workers is None:
        n_workers = _default_ripley_perm_workers(n_perm_int)
    n_workers = max(1, min(int(n_workers), n_perm_int))

    if pool_edge_factors is None:
        # Precompute isotropic edge factors once for every pooled point (reused across perms).
        edge_rng = np.random.default_rng(int(seed) + 7919)
        pool_edge_factors = _isotropic_edge_factors_for_foci(pool, r_vals, window, edge_rng)
    else:
        pool_edge_factors = np.asarray(pool_edge_factors, dtype=float)
        if pool_edge_factors.shape != (n_pool, len(r_vals)):
            raise ValueError(
                f"pool_edge_factors shape {pool_edge_factors.shape} != "
                f"expected {(n_pool, len(r_vals))}"
            )

    if n_workers == 1:
        perm_rng = rng if rng is not None else np.random.default_rng(int(seed))
        return _label_permutation_k_bidirectional_curves_sequential(
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
    k12_curves = np.full((n_perm_int, len(r_vals)), np.nan, dtype=float)
    k21_curves = np.full((n_perm_int, len(r_vals)), np.nan, dtype=float)
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_label_perm_bidir_worker,
        initargs=(pool, r_vals, window, n_monomer, pool_edge_factors),
    ) as executor:
        futures = [executor.submit(_label_perm_bidir_worker, task) for task in tasks]
        for fut in as_completed(futures):
            perm_id, k12, k21 = fut.result()
            k12_curves[perm_id] = k12
            k21_curves[perm_id] = k21
            if pbar is not None:
                pbar.set_postfix_str(f"perm {perm_id + 1}/{n_perm_int}", refresh=False)
                pbar.update(1)
    return {"k12": k12_curves, "k21": k21_curves}


def derive_symmetric_k_l_g_families(
    k12: np.ndarray,
    k21: np.ndarray,
    n1: int,
    n2: int,
    r_vals: np.ndarray,
    *,
    g_bin_width_nm: float,
) -> dict[str, np.ndarray]:
    """
    From type-1-focused K12 and type-2-focused K21 (each either a single curve, shape
    ``(n_r,)``, or a batch of curves, shape ``(n_curves, n_r)`` — e.g. a null ensemble),
    derive all six reported statistics: ``l12``, ``l21``, ``l_combined`` (via ``ripley_l12``
    and ``_intensity_weighted_combination``) and ``g12``, ``g21``, ``g_combined`` (via
    ``pair_correlation_from_k_diff``, itself batch-aware, and the same combination rule
    applied to the resulting ratios rather than to raw counts — see
    ``cross_k_bivariate_symmetric_3d_isotropic`` for why only the final K/ratio values, not
    intermediate counts, are combinable across directions).

    Returns ``k12``, ``k21``, ``k_combined``, ``l12``, ``l21``, ``l_combined``, ``g12``,
    ``g21``, ``g_combined`` — each the same shape as the input ``k12``/``k21``.
    """
    k_combined = _intensity_weighted_combination(k12, k21, n1, n2)
    l12 = ripley_l12(k12, r_vals)
    l21 = ripley_l12(k21, r_vals)
    l_combined = ripley_l12(k_combined, r_vals)
    g12 = pair_correlation_from_k_diff(k12, r_vals, bin_width_nm=g_bin_width_nm)["pcf"]
    g21 = pair_correlation_from_k_diff(k21, r_vals, bin_width_nm=g_bin_width_nm)["pcf"]
    g_combined = _intensity_weighted_combination(g12, g21, n1, n2)
    return {
        "k12": k12,
        "k21": k21,
        "k_combined": k_combined,
        "l12": l12,
        "l21": l21,
        "l_combined": l_combined,
        "g12": g12,
        "g21": g21,
        "g_combined": g_combined,
    }


# ============================================================================
# Curve statistics bands & curve-table builders
# ============================================================================


def _weighted_nanmean(curves: np.ndarray, weights: np.ndarray | None) -> np.ndarray:
    """
    Column-wise mean across replicate curves, each curve weighted by ``weights`` (e.g. the
    number of AuNP partners that curve's K/L/g value was computed from) instead of counted
    equally — so a curve backed by very little data (a handful of AuNPs) can't dominate a
    pooled mean the way an unweighted average would let it. ``weights=None`` reproduces a
    plain ``np.nanmean``.

    NaN entries (e.g. reliability-masked g shells) are excluded from both the weighted sum
    and that column's weight total, per column.
    """
    curves = np.asarray(curves, dtype=float)
    if curves.size == 0:
        return np.array([])
    if weights is None:
        return np.nanmean(curves, axis=0)
    weights = np.asarray(weights, dtype=float).reshape(-1, 1)
    valid = ~np.isnan(curves)
    w = np.where(valid, weights, 0.0)
    wsum = w.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.nansum(curves * w, axis=0) / wsum
    # Columns with zero total weight (all-NaN, or all zero-weight curves) fall back to an
    # unweighted mean rather than silently returning NaN/0.
    zero_weight = wsum <= 0
    if np.any(zero_weight):
        mean[zero_weight] = np.nanmean(curves[:, zero_weight], axis=0)
    return mean


def _percentile_band(
    curves: np.ndarray, *, weights: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    ``(lo, mean, hi)`` percentile envelope across replicate curves. The percentile bounds
    (``lo``/``hi``) are always unweighted (they describe the spread of individual curves,
    not a point estimate); only the returned ``mean`` uses ``weights`` when given — see
    ``_weighted_nanmean``.
    """
    if curves.size == 0:
        return np.array([]), np.array([]), np.array([])
    lo = np.nanpercentile(curves, RIPLEY_PERCENTILE_LO, axis=0)
    hi = np.nanpercentile(curves, RIPLEY_PERCENTILE_HI, axis=0)
    med = _weighted_nanmean(curves, weights)
    return lo, med, hi


def _mean_sd_band(curves: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (mean − SD, mean, mean + SD, SD) across replicate curves at each r.

    Uses the per-column count of non-NaN curves (not the total row count) so shells where
    some curves are missing/NaN (e.g. a curve stopped early) are still summarized correctly
    from however many curves actually reached that shell.
    """
    if curves.size == 0:
        empty = np.array([])
        return empty, empty, empty, empty
    mean = np.nanmean(curves, axis=0)
    n_valid = np.sum(~np.isnan(curves), axis=0)
    with np.errstate(invalid="ignore"):
        sd = np.nanstd(curves, axis=0, ddof=1)
    sd = np.where(n_valid > 1, sd, 0.0)
    return mean - sd, mean, mean + sd, sd


def _mean_sem_band(curves: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (mean − SEM, mean, mean + SEM, SEM) across replicate curves at each r.

    Uses the per-column count of non-NaN curves (not the total row count) — see
    ``_mean_sd_band``.
    """
    if curves.size == 0:
        empty = np.array([])
        return empty, empty, empty, empty
    mean = np.nanmean(curves, axis=0)
    n_valid = np.sum(~np.isnan(curves), axis=0)
    with np.errstate(invalid="ignore"):
        sd = np.nanstd(curves, axis=0, ddof=1)
        sem = sd / np.sqrt(np.maximum(n_valid, 1))
    sem = np.where(n_valid > 1, sem, 0.0)
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


# ============================================================================
# Small shared naming / table-shape helpers
# ============================================================================


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
