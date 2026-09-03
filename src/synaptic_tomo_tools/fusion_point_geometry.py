"""
Shared fusion-point geometry helpers (presynaptic AZ tangential controls, zone mapping).

Used by 3D fusion-point vs AuNP distance/Ripley analyses and visualization overlays.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.spatial import cKDTree

FUSION_POINT_SHIFT_OFFSET_NM = 40.0
FUSION_POINT_AZ_MAX_SNAP_DISTANCE_NM = 5.0


def _random_tangent_direction(
    center: np.ndarray,
    neighbors: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Unit tangent direction in the local AZ plane (PCA: two largest-variance axes)."""
    if len(neighbors) < 3:
        direction = rng.normal(size=3)
        direction[2] *= 0.2
        return direction / np.linalg.norm(direction)
    centered = neighbors - center
    cov = centered.T @ centered / len(centered)
    _, evecs = np.linalg.eigh(cov)
    t1 = evecs[:, 2]
    t2 = evecs[:, 1]
    t1 /= np.linalg.norm(t1)
    t2 /= np.linalg.norm(t2)
    theta = rng.uniform(0.0, 2.0 * np.pi)
    direction = np.cos(theta) * t1 + np.sin(theta) * t2
    return direction / np.linalg.norm(direction)


def sample_tangential_control_on_az(
    fusion_xyz: np.ndarray,
    az_xyz: np.ndarray,
    az_tree: cKDTree,
    offset_nm: float,
    rng: np.random.Generator,
    *,
    neighbor_radius_nm: float = 30.0,
    min_separation_from_fusion_nm: float = 5.0,
    max_snap_distance_nm: float = 10.0,
    max_attempts: int = 40,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Move fusion_xyz by offset_nm in a random tangent direction on the presynaptic AZ,
    then snap to the nearest presynaptic synaptic-cleft point.

    Retries up to max_attempts when the snapped point is too close to the fusion point
    or the candidate lies farther than max_snap_distance_nm from the AZ surface.
    Returns None if no valid control is found within max_attempts.
    """
    neighbor_idx = az_tree.query_ball_point(fusion_xyz, r=neighbor_radius_nm)
    if len(neighbor_idx) < 5:
        neighbor_idx = az_tree.query_ball_point(fusion_xyz, r=neighbor_radius_nm * 2.0)
    neighbors = az_xyz[np.asarray(neighbor_idx, dtype=int)]

    for _ in range(max_attempts):
        direction = _random_tangent_direction(fusion_xyz, neighbors, rng)
        candidate = fusion_xyz + offset_nm * direction
        snap_dist_nm, snap_idx = az_tree.query(candidate, k=1)
        if snap_dist_nm > max_snap_distance_nm:
            continue
        snapped = az_xyz[int(snap_idx)]
        if np.linalg.norm(snapped - fusion_xyz) >= min_separation_from_fusion_nm:
            return snapped, direction
    return None, None


def zone_name_for_presynaptic_membrane(membrane_name: str | None) -> str | None:
    """Map ``presynapticmembranes_N`` key to ``cleft_preN_postN`` zone name."""
    if not membrane_name or not str(membrane_name).startswith("presynapticmembranes_"):
        return None
    try:
        idx = int(str(membrane_name).removeprefix("presynapticmembranes_"))
    except ValueError:
        return None
    return f"cleft_pre{idx}_post{idx}"


def presynaptic_membrane_name_for_zone(zone_name: str, zone_data: dict | None = None) -> str:
    """Map synaptic cleft name to presynaptic membrane key used by vesicle results."""
    if zone_data and zone_data.get("presynaptic_membrane_index") is not None:
        pre_idx = int(zone_data["presynaptic_membrane_index"])
    else:
        match = re.search(r"pre(\d+)", str(zone_name))
        if not match:
            raise ValueError(
                f"Cannot parse presynaptic membrane index from zone name {zone_name!r}"
            )
        pre_idx = int(match.group(1))
    return f"presynapticmembranes_{pre_idx}"


def load_presynaptic_az_points_for_zone(
    tomogram_path: Path,
    alignment_dir: str,
    zone_name: str,
) -> np.ndarray:
    """Presynaptic outer + inner synaptic-cleft points for one zone."""
    az_dir = tomogram_path / alignment_dir / "STT_results" / "cleft"
    parts: list[np.ndarray] = []
    for suffix in ("pre_outer", "pre_inner"):
        path = az_dir / f"{zone_name}_{suffix}.txt"
        if path.is_file():
            surf = np.atleast_2d(np.loadtxt(path, delimiter=None))
            if surf.size:
                parts.append(surf.astype(float))
    if not parts:
        return np.zeros((0, 3), dtype=float)
    return np.vstack(parts)


def filter_fusion_rows_for_zone(
    fusion_rows: Sequence[dict],
    membrane_name: str,
) -> list[dict]:
    return [row for row in fusion_rows if row.get("closest_membrane") == membrane_name]
