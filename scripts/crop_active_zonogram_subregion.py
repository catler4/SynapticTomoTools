#!/usr/bin/env python3
"""
Crop a subregion from an existing active zonogram MRC and save a smaller zonogram
PNG (three-panel XY / YZ / XZ layout) matching the full-pipeline format at
1 pixel per voxel, plus optional stereo and membrane-overlay PNGs (pre/post
inner/outer on YZ and XZ panels, same colors as slice overlays).

Run after the main pipeline when you want a zoomed-in active zonogram for a
subset of the synapse.

Interactive mode (default): click two opposite corners on the XY max-projection,
then click top/bottom Z on the XZ view (X/Y come from step 1). Close each window
to continue.

Non-interactive mode: pass --crop x0,y0,x1,y1 and optional --z-min / --z-max.
Use --drag for rectangle drag selection instead of click-corners (less reliable on macOS).

Example (interactive crop region):
    python scripts/crop_active_zonogram_subregion.py \
        --set 15F1 \
        --tomogram 20240111_WaffleHipp_116 \
        --alignment-dir best_alignment \
        --zone-suffix active_zone_pre1_post1_az0

Example (manual crop region)
    python scripts/crop_active_zonogram_subregion.py \
        --mrc /path/to/{tomogram}_active_zonogram_{zone}.mrc \
        --crop 180,90,380,210 \
        --output-dir results/visualizations/20240111_WaffleHipp_116/best_alignment/active_zonograms/full

Tomogram MRC discovery uses the same layout as the main pipeline:
    {tomo_root}/{set}/TOP_TOMOS/{tomogram}/{alignment_dir}/STT_results/visualizations/active_zonograms/
Default tomo_root: TOMO_ROOT_BASE env var, or the goliath ProcessingCJS path.
Use --set 15F1 to search only that set; omit --set to search all sets under tomo-root.
Override tomo-root with --tomo-root (e.g. --tomo-root data for the local repo copy).
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import mrcfile
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

DEFAULT_TOMO_ROOT_BASE = Path(
    os.environ.get(
        "TOMO_ROOT_BASE",
        "/goliath/processing/Gouaux/CJS/BestTomo/ProcessingCJS/tomograms",
    )
)
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results" / "visualizations"

MRC_SEARCH_SUBDIRS = (
    "STT_results/visualizations/active_zonograms",
    "active_zonograms",
    "STT_results/visualizations/active_zonograms/full",
)

# Match visualization.py active-zone slice overlays
MEMBRANE_STYLE = {
    "pre_outer": {"color": (255, 0, 0), "alpha": 0.12},
    "post_outer": {"color": (0, 255, 0), "alpha": 0.12},
    "pre_inner": {"color": (255, 133, 133), "alpha": 0.06},
    "post_inner": {"color": (133, 255, 133), "alpha": 0.06},
}

MONOMER_STAR_PATTERN = "aunp_tm_BP_active_zone_*_manual_refined_monomer.star"
EACH_DIMER_STAR_PATTERN = "aunp_tm_BP_active_zone_*_manual_refined_each_dimer.star"
COORD_COLS = ("faCoordinateX", "faCoordinateY", "faCoordinateZ")
AUNP_COLORS = {
    "monomer": (161, 113, 177),  # #A171B1
    "dimer": (60, 84, 164),  # #3C54A4
}


def _imshow_limits(vol: torch.Tensor) -> tuple[float, float]:
    return -20 * float(vol.std()), 0.0


def load_zonogram_volume(mrc_path: Path) -> torch.Tensor:
    with mrcfile.open(mrc_path, mode="r") as mrc:
        data = np.asarray(mrc.data, dtype=np.float32)
    return torch.tensor(data)


def normalize_bounds(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    z0: float,
    z1: float,
    shape: tuple[int, int, int],
) -> tuple[int, int, int, int, int, int]:
    """Clip selection to volume indices. Shape is (Z, Y, X). Returns x0,x1,y0,y1,z0,z1."""
    z_size, y_size, x_size = shape
    xs = sorted((x0, x1))
    ys = sorted((y0, y1))
    zs = sorted((z0, z1))

    xi0 = int(np.floor(xs[0]))
    xi1 = int(np.ceil(xs[1]))
    yi0 = int(np.floor(ys[0]))
    yi1 = int(np.ceil(ys[1]))
    zi0 = int(np.floor(zs[0]))
    zi1 = int(np.ceil(zs[1]))

    xi0 = max(0, min(x_size - 1, xi0))
    yi0 = max(0, min(y_size - 1, yi0))
    zi0 = max(0, min(z_size - 1, zi0))
    xi1 = max(xi0 + 1, min(x_size, xi1))
    yi1 = max(yi0 + 1, min(y_size, yi1))
    zi1 = max(zi0 + 1, min(z_size, zi1))
    return xi0, xi1, yi0, yi1, zi0, zi1


def crop_volume(
    vol: torch.Tensor,
    x0: int,
    x1: int,
    y0: int,
    y1: int,
    z0: int,
    z1: int,
) -> torch.Tensor:
    return vol[z0:z1, y0:y1, x0:x1].contiguous()



def _rotate_volume_around_y(vol: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate (Z, Y, X) volume around the Y axis."""
    from scipy.ndimage import affine_transform

    angle = np.deg2rad(angle_deg)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rot = np.array([[cos_a, 0.0, sin_a], [0.0, 1.0, 0.0], [-sin_a, 0.0, cos_a]])
    center = (np.array(vol.shape, dtype=float) - 1.0) / 2.0
    offset = center - rot @ center
    return affine_transform(vol, rot, offset=offset, order=1, mode="nearest")


def _normalize_to_uint8(img: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    scaled = (img.astype(np.float32) - vmin) / (vmax - vmin)
    scaled = np.clip(scaled, 0.0, 1.0)
    return (scaled * 255).astype(np.uint8)


def _float_zonogram_canvas(vol: torch.Tensor) -> tuple[np.ndarray, float, float, int, int, int]:
    z_size, y_size, x_size = vol.shape
    vmin, vmax = _imshow_limits(vol)
    xy = torch.min(vol, dim=0).values.numpy()
    yz = torch.min(vol, dim=2).values.T.numpy()
    xz = torch.min(vol, dim=1).values.numpy()
    canvas = np.full((y_size + z_size, x_size + z_size), vmin, dtype=np.float32)
    canvas[0:y_size, 0:x_size] = xy
    canvas[0:y_size, x_size : x_size + z_size] = yz
    canvas[y_size : y_size + z_size, 0:x_size] = xz
    return canvas, vmin, vmax, y_size, z_size, x_size


def composite_native_zonogram(vol: torch.Tensor) -> np.ndarray:
    """
    Build the 3-panel zonogram layout at 1 pixel per voxel (no matplotlib resampling).
    Volume shape is (Z, Y, X). Output shape is (Y + Z, X + Z).
    """
    canvas, vmin, vmax, _, _, _ = _float_zonogram_canvas(vol)
    return np.flipud(_normalize_to_uint8(canvas, vmin, vmax))


def _gray_to_rgb(gray: np.ndarray) -> np.ndarray:
    return np.stack([gray, gray, gray], axis=-1)


def _blend_points(
    rgb: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    color: tuple[int, int, int],
    alpha: float,
) -> None:
    if rows.size == 0:
        return
    h, w = rgb.shape[:2]
    rows = np.clip(np.round(rows).astype(int), 0, h - 1)
    cols = np.clip(np.round(cols).astype(int), 0, w - 1)
    underlying = rgb[rows, cols].astype(np.float32)
    tint = np.array(color, dtype=np.float32)
    rgb[rows, cols] = np.clip(underlying * (1.0 - alpha) + tint * alpha, 0, 255).astype(np.uint8)


def _parse_az_index(zone_suffix: str) -> int | None:
    match = re.search(r"_az(\d+)$", zone_suffix)
    return int(match.group(1)) if match else None


def _parse_zone_suffix(zone_suffix: str) -> str:
    match = re.match(r"^(.*)_az\d+$", zone_suffix)
    return match.group(1) if match else zone_suffix


def _zone_suffix_from_mrc(mrc_path: Path) -> str | None:
    match = re.search(r"_active_zonogram_(.+)\.mrc$", mrc_path.name)
    return match.group(1) if match else None


def _resolve_tomogram_from_mrc(mrc_path: Path) -> tuple[Path, str]:
    path = mrc_path.resolve()
    for parent in path.parents:
        if parent.name == "STT_results":
            alignment_dir = parent.parent.name
            tomogram_path = parent.parent.parent
            return tomogram_path, alignment_dir
    raise ValueError(f"Could not infer tomogram path from MRC location: {mrc_path}")


def _load_az_surface_txt(path: Path) -> np.ndarray:
    if not path.exists():
        return np.zeros((0, 3))
    data = np.atleast_2d(np.loadtxt(path))
    if data.size == 0 or data.shape[1] < 3:
        return np.zeros((0, 3))
    return data


def _load_membrane_surfaces(tomogram_path: Path, alignment_dir: str, zone_name: str) -> dict[str, np.ndarray]:
    az_dir = tomogram_path / alignment_dir / "STT_results" / "activezone"
    return {
        "pre_outer": _load_az_surface_txt(az_dir / f"{zone_name}_pre_outer.txt"),
        "post_outer": _load_az_surface_txt(az_dir / f"{zone_name}_post_outer.txt"),
        "pre_inner": _load_az_surface_txt(az_dir / f"{zone_name}_pre_inner.txt"),
        "post_inner": _load_az_surface_txt(az_dir / f"{zone_name}_post_inner.txt"),
    }


def _load_zonogram_zone_data(tomogram_path: Path, alignment_dir: str, zone_name: str) -> dict:
    from synaptic_tomo_tools.activezone import (
        define_active_zonogram,
        find_active_zones_from_glb,
        import_membrane_segmentations_from_glb,
    )

    np.random.seed(42)
    membrane_data = import_membrane_segmentations_from_glb(str(tomogram_path), alignment_dir=alignment_dir)
    active_zones_data = find_active_zones_from_glb(membrane_data, distance_range=(10.0, 40.0))
    zonogram_results = define_active_zonogram(active_zones_data)
    zonogram_data = zonogram_results.get("zonogram_data", {})
    if zone_name not in zonogram_data:
        available = ", ".join(sorted(zonogram_data))
        raise KeyError(f"Zone '{zone_name}' not in zonogram data (available: {available})")
    return zonogram_data[zone_name]


def _transform_membrane_to_crop(
    coords: np.ndarray,
    *,
    full_vol: torch.Tensor,
    zone_data: dict,
    bounds: tuple[int, int, int, int, int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from synaptic_tomo_tools.visualization import transform_positions_to_zonogram_coords

    if coords.size == 0:
        empty = np.zeros(0)
        return empty, empty, empty

    zonogram_findingampa = (np.eye(3), np.zeros(3), full_vol, ())
    pts = transform_positions_to_zonogram_coords(coords, zonogram_findingampa, zone_data)
    x0, x1, y0, y1, z0, z1 = bounds
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    inside = (x >= x0) & (x < x1) & (y >= y0) & (y < y1) & (z >= z0) & (z < z1)
    return x[inside] - x0, y[inside] - y0, z[inside] - z0


def _transform_world_to_crop(
    coords: np.ndarray,
    *,
    full_vol: torch.Tensor,
    zone_data: dict,
    bounds: tuple[int, int, int, int, int, int],
) -> np.ndarray:
    x_c, y_c, z_c = _transform_membrane_to_crop(
        coords, full_vol=full_vol, zone_data=zone_data, bounds=bounds
    )
    if x_c.size == 0:
        return np.zeros((0, 3), dtype=float)
    return np.column_stack([x_c, y_c, z_c])


def _resolve_aunp_star(aunps_dir: Path, active_zone: int, pattern: str) -> Path | None:
    from synaptic_tomo_tools.aunps import aunp_pick_star_filename

    direct = aunps_dir / aunp_pick_star_filename(active_zone, pattern)
    if direct.is_file():
        return direct
    glob_pat = pattern.replace("*", f"{active_zone}")
    matches = sorted(aunps_dir.glob(glob_pat))
    if matches:
        return matches[0]
    # Last resort: any file ending with the pattern suffix after the zone index
    suffix = pattern.split("*", 1)[1]
    for path in sorted(aunps_dir.glob(f"*{suffix}")):
        if f"active_zone_{active_zone}" in path.name:
            return path
    return None


def _read_aunp_pick_coordinates(star_path: Path) -> np.ndarray:
    import pandas as pd
    import starfile

    from synaptic_tomo_tools.aunps import _read_aunp_pick_star_dataframe

    df = _read_aunp_pick_star_dataframe(star_path)
    if df is None or df.empty:
        return np.zeros((0, 3))
    missing = [c for c in COORD_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{star_path} missing columns: {missing}")
    coords = df[list(COORD_COLS)].apply(pd.to_numeric, errors="coerce").dropna(how="any")
    if coords.empty:
        return np.zeros((0, 3))
    return coords.to_numpy(dtype=float)


def _load_monomer_dimer_pick_coords(
    tomogram_path: Path, alignment_dir: str, active_zone: int
) -> tuple[np.ndarray, np.ndarray]:
    aunps_dir = tomogram_path / alignment_dir / "aunps"
    monomer_path = _resolve_aunp_star(aunps_dir, active_zone, MONOMER_STAR_PATTERN)
    dimer_path = _resolve_aunp_star(aunps_dir, active_zone, EACH_DIMER_STAR_PATTERN)

    monomer = (
        _read_aunp_pick_coordinates(monomer_path)
        if monomer_path is not None
        else np.zeros((0, 3))
    )
    dimer = (
        _read_aunp_pick_coordinates(dimer_path)
        if dimer_path is not None
        else np.zeros((0, 3))
    )
    if monomer.size == 0 and dimer.size == 0:
        raise FileNotFoundError(
            f"No monomer/dimer STAR picks under {aunps_dir} for active zone {active_zone}"
        )
    return monomer, dimer


def _rotate_points_y(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    angle_deg: float,
    center_x: float,
    center_z: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    angle = np.deg2rad(angle_deg)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    x0 = x - center_x
    z0 = z - center_z
    xr = x0 * cos_a + z0 * sin_a + center_x
    zr = -x0 * sin_a + z0 * cos_a + center_z
    return xr, y, zr


def _stamp_stereo_aunps(
    rgb: np.ndarray,
    *,
    monomer_xyz: np.ndarray,
    dimer_xyz: np.ndarray,
    angle_deg: float,
    y_size: int,
    x_size: int,
    z_size: int,
) -> None:
    center_x = (x_size - 1) / 2.0
    center_z = (z_size - 1) / 2.0
    for pts, label in ((monomer_xyz, "monomer"), (dimer_xyz, "dimer")):
        if pts.size == 0:
            continue
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        color = AUNP_COLORS[label]
        for angle, col_offset in ((-angle_deg, 0), (angle_deg, x_size)):
            xr, yr, _ = _rotate_points_y(x, y, z, angle, center_x, center_z)
            cols = np.round(xr).astype(int) + col_offset
            rows = np.round(yr).astype(int)
            valid = (
                (rows >= 0)
                & (rows < y_size)
                & (cols >= col_offset)
                & (cols < col_offset + x_size)
            )
            _blend_points(rgb, rows[valid], cols[valid], color, alpha=1.0)


def composite_native_stereo_with_aunps(
    vol: torch.Tensor,
    angle_deg: float,
    *,
    monomer_xyz: np.ndarray,
    dimer_xyz: np.ndarray,
) -> np.ndarray:
    """Stereo min-projection pair with monomer (purple) and dimer (blue) pick centers."""
    vmin, vmax = _imshow_limits(vol)
    vol_np = vol.numpy()
    left = np.min(_rotate_volume_around_y(vol_np, -angle_deg), axis=0)
    right = np.min(_rotate_volume_around_y(vol_np, angle_deg), axis=0)
    z_size, y_size, x_size = vol.shape

    canvas = np.full((y_size, x_size * 2), vmin, dtype=np.float32)
    canvas[:, :x_size] = left
    canvas[:, x_size:] = right
    gray = _normalize_to_uint8(canvas, vmin, vmax)
    rgb = _gray_to_rgb(gray)
    _stamp_stereo_aunps(
        rgb,
        monomer_xyz=monomer_xyz,
        dimer_xyz=dimer_xyz,
        angle_deg=angle_deg,
        y_size=y_size,
        x_size=x_size,
        z_size=z_size,
    )
    return np.flipud(rgb)


def _stamp_membrane_on_side_panels(
    rgb: np.ndarray,
    *,
    x_c: np.ndarray,
    y_c: np.ndarray,
    z_c: np.ndarray,
    y_size: int,
    x_size: int,
    layer: str,
) -> None:
    style = MEMBRANE_STYLE[layer]
    # YZ panel (top right): rows=Y, cols=Z
    _blend_points(rgb, y_c, x_size + z_c, style["color"], style["alpha"])
    # XZ panel (bottom left): rows=Z, cols=X
    _blend_points(rgb, y_size + z_c, x_c, style["color"], style["alpha"])


def composite_native_zonogram_with_membranes(
    cropped: torch.Tensor,
    *,
    full_vol: torch.Tensor,
    zone_data: dict,
    membrane_surfaces: dict[str, np.ndarray],
    bounds: tuple[int, int, int, int, int, int],
) -> np.ndarray:
    canvas, vmin, vmax, y_size, z_size, x_size = _float_zonogram_canvas(cropped)
    gray = _normalize_to_uint8(canvas, vmin, vmax)
    rgb = _gray_to_rgb(gray)

    for layer in ("pre_inner", "post_inner", "pre_outer", "post_outer"):
        x_c, y_c, z_c = _transform_membrane_to_crop(
            membrane_surfaces[layer],
            full_vol=full_vol,
            zone_data=zone_data,
            bounds=bounds,
        )
        _stamp_membrane_on_side_panels(
            rgb, x_c=x_c, y_c=y_c, z_c=z_c, y_size=y_size, x_size=x_size, layer=layer
        )

    return np.flipud(rgb)


def _stamp_aunps_on_zonogram_panels(
    rgb: np.ndarray,
    *,
    monomer_xyz: np.ndarray,
    dimer_xyz: np.ndarray,
    y_size: int,
    x_size: int,
) -> None:
    for pts, label in ((monomer_xyz, "monomer"), (dimer_xyz, "dimer")):
        if pts.size == 0:
            continue
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        color = AUNP_COLORS[label]
        _blend_points(rgb, y, x, color, alpha=1.0)
        _blend_points(rgb, y, x_size + z, color, alpha=1.0)
        _blend_points(rgb, y_size + z, x, color, alpha=1.0)


def composite_native_zonogram_with_membranes_and_aunps(
    cropped: torch.Tensor,
    *,
    full_vol: torch.Tensor,
    zone_data: dict,
    membrane_surfaces: dict[str, np.ndarray],
    bounds: tuple[int, int, int, int, int, int],
    monomer_xyz: np.ndarray,
    dimer_xyz: np.ndarray,
) -> np.ndarray:
    canvas, vmin, vmax, y_size, z_size, x_size = _float_zonogram_canvas(cropped)
    gray = _normalize_to_uint8(canvas, vmin, vmax)
    rgb = _gray_to_rgb(gray)

    for layer in ("pre_inner", "post_inner", "pre_outer", "post_outer"):
        x_c, y_c, z_c = _transform_membrane_to_crop(
            membrane_surfaces[layer],
            full_vol=full_vol,
            zone_data=zone_data,
            bounds=bounds,
        )
        _stamp_membrane_on_side_panels(
            rgb, x_c=x_c, y_c=y_c, z_c=z_c, y_size=y_size, x_size=x_size, layer=layer
        )

    _stamp_aunps_on_zonogram_panels(
        rgb,
        monomer_xyz=monomer_xyz,
        dimer_xyz=dimer_xyz,
        y_size=y_size,
        x_size=x_size,
    )
    return np.flipud(rgb)


def composite_native_stereo(vol: torch.Tensor, angle_deg: float) -> np.ndarray:
    """Side-by-side stereo min-projection at 1 pixel per voxel. Output shape (Y, 2*X)."""
    vmin, vmax = _imshow_limits(vol)
    vol_np = vol.numpy()
    left = np.min(_rotate_volume_around_y(vol_np, -angle_deg), axis=0)
    right = np.min(_rotate_volume_around_y(vol_np, angle_deg), axis=0)
    y_size, x_size = left.shape

    canvas = np.full((y_size, x_size * 2), vmin, dtype=np.float32)
    canvas[:, :x_size] = left
    canvas[:, x_size:] = right
    return np.flipud(_normalize_to_uint8(canvas, vmin, vmax))


def save_subregion(
    cropped: torch.Tensor,
    output_png: Path,
    *,
    output_mrc: Path | None = None,
    output_stereo_png: Path | None = None,
    output_stereo_aunps_png: Path | None = None,
    output_membrane_png: Path | None = None,
    output_aunps_membranes_png: Path | None = None,
    full_vol: torch.Tensor | None = None,
    zone_data: dict | None = None,
    membrane_surfaces: dict[str, np.ndarray] | None = None,
    monomer_xyz: np.ndarray | None = None,
    dimer_xyz: np.ndarray | None = None,
    bounds: tuple[int, int, int, int, int, int] | None = None,
    stereo_angle_deg: float = 6.0,
) -> tuple[tuple[int, int], tuple[int, int] | None]:
    from PIL import Image

    output_png.parent.mkdir(parents=True, exist_ok=True)

    zonogram = composite_native_zonogram(cropped)
    Image.fromarray(zonogram).save(output_png)
    png_shape = (zonogram.shape[0], zonogram.shape[1])

    if output_membrane_png is not None:
        if full_vol is None or zone_data is None or membrane_surfaces is None or bounds is None:
            raise ValueError("Membrane PNG requires full_vol, zone_data, membrane_surfaces, and bounds")
        membrane_rgb = composite_native_zonogram_with_membranes(
            cropped,
            full_vol=full_vol,
            zone_data=zone_data,
            membrane_surfaces=membrane_surfaces,
            bounds=bounds,
        )
        Image.fromarray(membrane_rgb).save(output_membrane_png)

    if output_aunps_membranes_png is not None:
        if (
            full_vol is None
            or zone_data is None
            or membrane_surfaces is None
            or bounds is None
            or monomer_xyz is None
            or dimer_xyz is None
        ):
            raise ValueError(
                "AuNP+membrane PNG requires full_vol, zone_data, membrane_surfaces, "
                "bounds, monomer_xyz, and dimer_xyz"
            )
        combined_rgb = composite_native_zonogram_with_membranes_and_aunps(
            cropped,
            full_vol=full_vol,
            zone_data=zone_data,
            membrane_surfaces=membrane_surfaces,
            bounds=bounds,
            monomer_xyz=monomer_xyz,
            dimer_xyz=dimer_xyz,
        )
        Image.fromarray(combined_rgb).save(output_aunps_membranes_png)

    stereo_shape = None
    if output_stereo_png is not None:
        stereo = composite_native_stereo(cropped, stereo_angle_deg)
        Image.fromarray(stereo).save(output_stereo_png)
        stereo_shape = (stereo.shape[0], stereo.shape[1])

    if output_stereo_aunps_png is not None:
        if monomer_xyz is None or dimer_xyz is None:
            raise ValueError("Stereo AuNP PNG requires monomer_xyz and dimer_xyz")
        stereo_aunps = composite_native_stereo_with_aunps(
            cropped,
            stereo_angle_deg,
            monomer_xyz=monomer_xyz,
            dimer_xyz=dimer_xyz,
        )
        Image.fromarray(stereo_aunps).save(output_stereo_aunps_png)

    if output_mrc is not None:
        mrcfile.write(output_mrc, cropped.numpy(), overwrite=True)

    return png_shape, stereo_shape


def _parse_crop(s: str) -> tuple[float, float, float, float]:
    parts = [float(p.strip()) for p in s.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("--crop expects four comma-separated values: x0,y0,x1,y1")
    return parts[0], parts[1], parts[2], parts[3]


def _ensure_interactive_backend() -> None:
    import matplotlib

    backend = matplotlib.get_backend().lower()
    if "agg" in backend:
        for candidate in ("macosx", "TkAgg", "Qt5Agg"):
            try:
                matplotlib.use(candidate, force=True)
                print(f"Using matplotlib backend: {candidate}")
                return
            except Exception:
                continue
        raise SystemExit(
            "No interactive matplotlib backend available. Use --crop for non-interactive mode."
        )


def _draw_selection_rect(ax, x0: float, x1: float, y0: float, y1: float):
    from matplotlib.patches import Rectangle

    rect = Rectangle(
        (x0, y0),
        x1 - x0,
        y1 - y0,
        fill=False,
        edgecolor="yellow",
        linewidth=2,
    )
    ax.add_patch(rect)
    return rect


def _click_corner_selector(
    ax,
    vol_slice: np.ndarray,
    title: str,
    *,
    y_label: str = "Y (voxels)",
) -> tuple[float, float, float, float]:
    """Click two opposite corners; close the window to confirm."""
    import matplotlib.pyplot as plt

    vmin, vmax = _imshow_limits(torch.tensor(vol_slice))
    ax.imshow(vol_slice, cmap="gray", interpolation="mitchell", vmin=vmin, vmax=vmax, origin="lower")
    ax.set_title(
        f"{title}\nClick two opposite corners on the image, then close this window."
    )
    ax.set_xlabel("X (voxels)")
    ax.set_ylabel(y_label)

    print(f"\n{title}")
    print("  1. Click the first corner of the crop box on the image.")
    print("  2. Click the opposite corner.")
    print("  3. Close the window to continue.\n")

    fig = ax.figure
    pts = plt.ginput(2, timeout=0)
    if len(pts) < 2:
        raise SystemExit("Need two corner clicks on the image before closing the window.")

    xs = sorted((pts[0][0], pts[1][0]))
    ys = sorted((pts[0][1], pts[1][1]))
    _draw_selection_rect(ax, xs[0], xs[1], ys[0], ys[1])
    fig.canvas.draw_idle()
    plt.show()

    return xs[0], ys[0], xs[1], ys[1]


def _drag_rectangle_selector(
    ax,
    vol_slice: np.ndarray,
    title: str,
    *,
    y_label: str = "Y (voxels)",
) -> tuple[float, float, float, float]:
    """Drag a rectangle; close the window to confirm."""
    import matplotlib.pyplot as plt
    from matplotlib.widgets import RectangleSelector

    vmin, vmax = _imshow_limits(torch.tensor(vol_slice))
    ax.imshow(vol_slice, cmap="gray", interpolation="mitchell", vmin=vmin, vmax=vmax, origin="lower")
    ax.set_title(f"{title}\nDrag a rectangle on the image, then close this window.")
    ax.set_xlabel("X (voxels)")
    ax.set_ylabel(y_label)

    selection: dict[str, float | None] = {"x0": None, "y0": None, "x1": None, "y1": None}

    def onselect(eclick, erelease):
        xs = sorted((eclick.xdata, erelease.xdata))
        ys = sorted((eclick.ydata, erelease.ydata))
        selection["x0"], selection["x1"] = xs[0], xs[1]
        selection["y0"], selection["y1"] = ys[0], ys[1]

    RectangleSelector(
        ax,
        onselect,
        useblit=False,
        button=None,
        minspanx=2,
        minspany=2,
        spancoords="data",
        interactive=True,
        props=dict(facecolor="yellow", edgecolor="yellow", alpha=0.25, fill=True),
    )
    plt.tight_layout()
    plt.show()

    if any(selection[k] is None for k in ("x0", "y0", "x1", "y1")):
        raise SystemExit("No rectangle selected. Drag on the image, then close the window.")
    return selection["x0"], selection["y0"], selection["x1"], selection["y1"]  # type: ignore[return-value]


def _select_region(
    ax,
    vol_slice: np.ndarray,
    title: str,
    *,
    y_label: str = "Y (voxels)",
    use_drag: bool = False,
) -> tuple[float, float, float, float]:
    if use_drag:
        return _drag_rectangle_selector(ax, vol_slice, title, y_label=y_label)
    return _click_corner_selector(ax, vol_slice, title, y_label=y_label)


def _draw_z_band(ax, z0: float, z1: float, x_lim: tuple[float, float]):
    from matplotlib.patches import Rectangle

    rect = Rectangle(
        (x_lim[0], z0),
        x_lim[1] - x_lim[0],
        z1 - z0,
        fill=False,
        edgecolor="yellow",
        linewidth=2,
    )
    ax.add_patch(rect)
    return rect


def _select_z_range(
    ax,
    xz_slice: np.ndarray,
    x0: float,
    x1: float,
    *,
    use_drag: bool = False,
) -> tuple[float, float]:
    """Pick Z only from an XZ projection; X/Y are already fixed."""
    import matplotlib.pyplot as plt
    from matplotlib.widgets import SpanSelector

    title = "Step 2/2: choose Z range on XZ (vertical axis). X/Y already set from step 1."
    vmin, vmax = _imshow_limits(torch.tensor(xz_slice))
    ax.imshow(xz_slice, cmap="gray", interpolation="mitchell", vmin=vmin, vmax=vmax, origin="lower")
    ax.axvline(x0, color="cyan", linewidth=1, linestyle="--", alpha=0.8)
    ax.axvline(x1, color="cyan", linewidth=1, linestyle="--", alpha=0.8)
    ax.set_title(f"{title}\nClick top and bottom Z, then close this window.")
    ax.set_xlabel("X (voxels)")
    ax.set_ylabel("Z (voxels)")

    print(f"\n{title}")
    if use_drag:
        print("  Drag vertically on the image to set Z range, then close the window.\n")
    else:
        print("  1. Click the top Z you want to keep.")
        print("  2. Click the bottom Z.")
        print("  3. Close the window to continue.\n")

    fig = ax.figure
    x_lim = (float(ax.get_xlim()[0]), float(ax.get_xlim()[1]))
    z_bounds: dict[str, float | None] = {"z0": None, "z1": None}

    if use_drag:
        def onselect(zmin, zmax):
            z_bounds["z0"], z_bounds["z1"] = zmin, zmax

        SpanSelector(
            ax,
            onselect,
            direction="vertical",
            useblit=False,
            props=dict(facecolor="yellow", edgecolor="yellow", alpha=0.25, fill=True),
            interactive=True,
            drag_from_anywhere=True,
        )
        plt.tight_layout()
        plt.show()
        if z_bounds["z0"] is None or z_bounds["z1"] is None:
            raise SystemExit("No Z range selected. Drag vertically on the image, then close the window.")
        return z_bounds["z0"], z_bounds["z1"]  # type: ignore[return-value]

    pts = plt.ginput(2, timeout=0)
    if len(pts) < 2:
        raise SystemExit("Need two clicks on the image to set Z before closing the window.")

    zs = sorted((pts[0][1], pts[1][1]))
    _draw_z_band(ax, zs[0], zs[1], x_lim)
    fig.canvas.draw_idle()
    plt.show()
    return zs[0], zs[1]


def interactive_select_bounds(vol: torch.Tensor, *, use_drag: bool = False) -> tuple[int, int, int, int, int, int]:
    _ensure_interactive_backend()
    import matplotlib.pyplot as plt

    z_size, y_size, x_size = vol.shape

    fig1, ax1 = plt.subplots(figsize=(10, 6))
    xy = torch.min(vol, dim=0).values.numpy()
    x0, y0, x1, y1 = _select_region(
        ax1,
        xy,
        "Step 1/2: select XY region.",
        use_drag=use_drag,
    )
    plt.close(fig1)

    xz_crop_y = slice(int(np.floor(y0)), int(np.ceil(y1)))
    xz = torch.min(vol[:, xz_crop_y, :], dim=1).values.numpy()
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    zz0, zz1 = _select_z_range(ax2, xz, x0, x1, use_drag=use_drag)
    plt.close(fig2)

    z0, z1 = sorted((zz0, zz1))
    return normalize_bounds(x0, y0, x1, y1, z0, z1, (z_size, y_size, x_size))


def _set_dirs(tomo_root: Path, set_name: str | None) -> list[Path]:
    """Return set directories to search under tomo_root."""
    tomo_root = tomo_root.expanduser().resolve()
    if set_name:
        set_dir = tomo_root / set_name
        if not set_dir.is_dir():
            raise FileNotFoundError(f"Set directory not found: {set_dir}")
        return [set_dir]
    return sorted(p for p in tomo_root.iterdir() if p.is_dir())


def _mrc_candidates_for_sets(
    *,
    tomogram: str,
    alignment_dir: str,
    zone_suffix: str,
    tomo_root: Path,
    results_root: Path,
    set_name: str | None,
) -> list[Path]:
    filename = f"{tomogram}_active_zonogram_{zone_suffix}.mrc"
    candidates: list[Path] = []

    results_png_dir = results_root / tomogram / alignment_dir / "active_zonograms" / "full"
    candidates.append(results_png_dir / filename)

    for set_dir in _set_dirs(tomo_root, set_name):
        base = set_dir / "TOP_TOMOS" / tomogram / alignment_dir
        for sub in MRC_SEARCH_SUBDIRS:
            candidates.append(base / sub / filename)
    return candidates


def find_mrc_path(
    *,
    mrc: Path | None,
    tomogram: str | None,
    alignment_dir: str | None,
    zone_suffix: str | None,
    tomo_root: Path,
    results_root: Path,
    set_name: str | None = None,
) -> Path:
    if mrc is not None:
        mrc = mrc.expanduser().resolve()
        if not mrc.exists():
            raise FileNotFoundError(f"MRC not found: {mrc}")
        return mrc

    if not tomogram or not alignment_dir or not zone_suffix:
        raise SystemExit(
            "Provide --mrc or all of --tomogram, --alignment-dir, and --zone-suffix."
        )

    filename = f"{tomogram}_active_zonogram_{zone_suffix}.mrc"
    candidates = _mrc_candidates_for_sets(
        tomogram=tomogram,
        alignment_dir=alignment_dir,
        zone_suffix=zone_suffix,
        tomo_root=tomo_root,
        results_root=results_root,
        set_name=set_name,
    )

    for path in candidates:
        if path.exists():
            return path.resolve()

    tried = "\n  ".join(str(p) for p in candidates[:8])
    extra = f"\n  ... and {len(candidates) - 8} more" if len(candidates) > 8 else ""
    raise FileNotFoundError(f"Could not find {filename}. Tried:\n  {tried}{extra}")


def default_output_dir(
    mrc_path: Path,
    tomogram: str | None,
    alignment_dir: str | None,
    results_root: Path,
) -> Path:
    if tomogram and alignment_dir:
        results_dir = results_root / tomogram / alignment_dir / "active_zonograms" / "full"
        if results_dir.exists():
            return results_dir
    return mrc_path.parent


def output_stem(source_mrc: Path, suffix: str, bounds: tuple[int, int, int, int, int, int]) -> str:
    base = source_mrc.stem
    x0, x1, y0, y1, z0, z1 = bounds
    region_tag = f"x{x0}-{x1}_y{y0}-{y1}_z{z0}-{z1}"
    return f"{base}_{suffix}_{region_tag}"


def list_available_zonograms(
    tomogram: str,
    alignment_dir: str,
    tomo_root: Path,
    results_root: Path,
    *,
    set_name: str | None = None,
) -> None:
    scope = f"{tomo_root}/{set_name}" if set_name else str(tomo_root)
    print(f"Zonogram MRC files for {tomogram} / {alignment_dir} (searching under {scope}):")
    found = False
    for set_dir in _set_dirs(tomo_root, set_name):
        az_dir = set_dir / "TOP_TOMOS" / tomogram / alignment_dir / "STT_results" / "visualizations" / "active_zonograms"
        if az_dir.is_dir():
            for mrc in sorted(az_dir.glob(f"{tomogram}_active_zonogram_*.mrc")):
                print(f"  {mrc}")
                found = True
    res_dir = results_root / tomogram / alignment_dir / "active_zonograms" / "full"
    if res_dir.is_dir():
        for png in sorted(res_dir.glob(f"{tomogram}_active_zonogram_*.png")):
            if "_subregion_" not in png.name:
                print(f"  [PNG only] {png}")
                found = True
    if not found:
        print("  (none found)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crop a subregion from an active zonogram and save a three-panel PNG (+ MRC)."
    )
    parser.add_argument("--mrc", type=Path, help="Path to source active zonogram .mrc")
    parser.add_argument("--tomogram", help="Tomogram name (used with --alignment-dir and --zone-suffix)")
    parser.add_argument("--alignment-dir", help="Alignment subdirectory, e.g. best_alignment")
    parser.add_argument(
        "--zone-suffix",
        help="Zone part of filename after _active_zonogram_, e.g. active_zone_pre1_post1_az0",
    )
    parser.add_argument(
        "--set",
        dest="set_name",
        help="Experimental set name (e.g. 15F1). If omitted, search all sets under --tomo-root.",
    )
    parser.add_argument(
        "--tomo-root",
        type=Path,
        default=DEFAULT_TOMO_ROOT_BASE,
        help=(
            "Tomogram root containing {set}/TOP_TOMOS/ trees "
            "(default: TOMO_ROOT_BASE env or goliath ProcessingCJS path)"
        ),
    )
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument(
        "--crop",
        type=_parse_crop,
        help="Non-interactive XY crop: x0,y0,x1,y1 in zonogram voxel coordinates (full Z unless --z-min/--z-max)",
    )
    parser.add_argument("--z-min", type=int, help="Z index start (inclusive)")
    parser.add_argument("--z-max", type=int, help="Z index end (exclusive)")
    parser.add_argument("--output-dir", type=Path, help="Directory for cropped PNG/MRC")
    parser.add_argument(
        "--output-suffix",
        default="subregion",
        help="Token inserted in output filename before coordinate tag (default: subregion)",
    )
    parser.add_argument("--no-mrc", action="store_true", help="Save PNG only, not cropped .mrc")
    parser.add_argument("--no-stereo", action="store_true", help="Skip stereo side-by-side PNG")
    parser.add_argument(
        "--no-membranes",
        action="store_true",
        help="Skip membrane-overlay PNG (YZ/XZ panels with pre/post inner/outer)",
    )
    parser.add_argument(
        "--no-stereo-aunps",
        action="store_true",
        help="Skip stereo PNG with monomer (purple) and dimer (blue) AuNP picks",
    )
    parser.add_argument(
        "--no-aunps-membranes",
        action="store_true",
        help="Skip 3-panel PNG with membranes and AuNP picks overlaid",
    )
    parser.add_argument(
        "--stereo-angle",
        type=float,
        default=6.0,
        help="Stereo tilt angle in degrees for each eye (default: 6)",
    )
    parser.add_argument(
        "--drag",
        action="store_true",
        help="Use click-drag rectangle instead of two-corner clicks (default: click corners)",
    )
    parser.add_argument("--list", action="store_true", help="List zonogram files for --tomogram and exit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.list:
        if not args.tomogram or not args.alignment_dir:
            raise SystemExit("--list requires --tomogram and --alignment-dir")
        list_available_zonograms(
            args.tomogram,
            args.alignment_dir,
            args.tomo_root,
            args.results_root,
            set_name=args.set_name,
        )
        return

    mrc_path = find_mrc_path(
        mrc=args.mrc,
        tomogram=args.tomogram,
        alignment_dir=args.alignment_dir,
        zone_suffix=args.zone_suffix,
        tomo_root=args.tomo_root,
        results_root=args.results_root,
        set_name=args.set_name,
    )
    print(f"Loading {mrc_path}")
    vol = load_zonogram_volume(mrc_path)
    z_size, y_size, x_size = vol.shape
    print(f"Volume shape (Z, Y, X): ({z_size}, {y_size}, {x_size})")

    if args.crop is not None:
        x0, y0, x1, y1 = args.crop
        z0 = args.z_min if args.z_min is not None else 0
        z1 = args.z_max if args.z_max is not None else z_size
        bounds = normalize_bounds(x0, y0, x1, y1, z0, z1, vol.shape)
    else:
        print("Interactive mode: select XY, then top/bottom Z on XZ. Close each window when done.")
        bounds = interactive_select_bounds(vol, use_drag=args.drag)

    x0, x1, y0, y1, z0, z1 = bounds
    print(f"Crop indices — X:[{x0},{x1}) Y:[{y0},{y1}) Z:[{z0},{z1})")
    cropped = crop_volume(vol, x0, x1, y0, y1, z0, z1)
    print(f"Cropped shape (Z, Y, X): {tuple(cropped.shape)}")

    out_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else default_output_dir(mrc_path, args.tomogram, args.alignment_dir, args.results_root)
    )
    stem = output_stem(mrc_path, args.output_suffix, bounds)
    png_path = out_dir / f"{stem}.png"
    mrc_out = None if args.no_mrc else out_dir / f"{stem}.mrc"
    stereo_out = None if args.no_stereo else out_dir / f"{stem}_stereo.png"
    membrane_out = None if args.no_membranes else out_dir / f"{stem}_membranes.png"
    stereo_aunps_out = None if args.no_stereo or args.no_stereo_aunps else out_dir / f"{stem}_stereo_aunps.png"
    aunps_membranes_out = (
        None
        if args.no_aunps_membranes or args.no_membranes
        else out_dir / f"{stem}_aunps_membranes.png"
    )

    zone_data = None
    membrane_surfaces = None
    monomer_xyz = None
    dimer_xyz = None
    zone_suffix = args.zone_suffix or _zone_suffix_from_mrc(mrc_path)
    tomogram_path = None
    alignment_dir = None
    active_zone = _parse_az_index(zone_suffix) if zone_suffix else None

    needs_membranes = membrane_out is not None or aunps_membranes_out is not None
    needs_aunps = stereo_aunps_out is not None or aunps_membranes_out is not None

    if needs_membranes or needs_aunps:
        if not zone_suffix or active_zone is None:
            print("Warning: could not determine active zone from zone suffix; skipping overlay PNGs.")
            membrane_out = None
            stereo_aunps_out = None
            aunps_membranes_out = None
        else:
            zone_name = _parse_zone_suffix(zone_suffix)
            tomogram_path, alignment_dir = _resolve_tomogram_from_mrc(mrc_path)
            if needs_membranes:
                try:
                    membrane_surfaces = _load_membrane_surfaces(tomogram_path, alignment_dir, zone_name)
                    zone_data = _load_zonogram_zone_data(tomogram_path, alignment_dir, zone_name)
                except Exception as exc:
                    print(f"Warning: could not load membrane overlay data ({exc}); skipping membrane PNGs.")
                    membrane_out = None
                    aunps_membranes_out = None
            if needs_aunps:
                try:
                    if zone_data is None:
                        zone_data = _load_zonogram_zone_data(tomogram_path, alignment_dir, zone_name)
                    monomer_world, dimer_world = _load_monomer_dimer_pick_coords(
                        tomogram_path, alignment_dir, active_zone
                    )
                    monomer_xyz = _transform_world_to_crop(
                        monomer_world, full_vol=vol, zone_data=zone_data, bounds=bounds
                    )
                    dimer_xyz = _transform_world_to_crop(
                        dimer_world, full_vol=vol, zone_data=zone_data, bounds=bounds
                    )
                except Exception as exc:
                    print(f"Warning: could not load AuNP picks ({exc}); skipping AuNP overlay PNGs.")
                    stereo_aunps_out = None
                    aunps_membranes_out = None

    save_subregion(
        cropped,
        png_path,
        output_mrc=mrc_out,
        output_stereo_png=stereo_out,
        output_stereo_aunps_png=stereo_aunps_out,
        output_membrane_png=membrane_out,
        output_aunps_membranes_png=aunps_membranes_out,
        full_vol=vol,
        zone_data=zone_data,
        membrane_surfaces=membrane_surfaces,
        monomer_xyz=monomer_xyz,
        dimer_xyz=dimer_xyz,
        bounds=bounds,
        stereo_angle_deg=args.stereo_angle,
    )
    cz, cy, cx = cropped.shape
    print(f"Saved {png_path} ({cy + cz} x {cx + cz} px, 1 px/voxel per panel)")
    if membrane_out is not None:
        print(f"Saved {membrane_out} (membrane overlay on YZ/XZ panels)")
    if aunps_membranes_out is not None:
        n_mono = len(monomer_xyz) if monomer_xyz is not None else 0
        n_dimer = len(dimer_xyz) if dimer_xyz is not None else 0
        print(
            f"Saved {aunps_membranes_out} "
            f"(membranes + {n_mono} monomer + {n_dimer} dimer picks on 3-panel view)"
        )
    if stereo_aunps_out is not None:
        n_mono = len(monomer_xyz) if monomer_xyz is not None else 0
        n_dimer = len(dimer_xyz) if dimer_xyz is not None else 0
        print(f"Saved {stereo_aunps_out} (stereo with {n_mono} monomer + {n_dimer} dimer picks)")
    if stereo_out is not None:
        print(f"Saved {stereo_out} ({cy} x {cx * 2} px, 1 px/voxel)")
    if mrc_out is not None:
        print(f"Saved {mrc_out}")


if __name__ == "__main__":
    main()
