#!/usr/bin/env python3
"""
Crop a subregion from an existing active zonogram MRC and save a smaller zonogram
PNG (three-panel XY / YZ / XZ layout) matching the full-pipeline format, plus a
side-by-side stereo min-projection pair for cross-eyed / parallel viewing.

Run after the main pipeline when you want a zoomed-in active zonogram for a
subset of the synapse.

Interactive mode (default): click two opposite corners on the XY max-projection,
then click top/bottom Z on the XZ view (X/Y come from step 1). Close each window
to continue.

Non-interactive mode: pass --crop x0,y0,x1,y1 and optional --z-min / --z-max.
Use --drag for rectangle drag selection instead of click-corners (less reliable on macOS).

Example (interactive crop region):
    python scripts/crop_active_zonogram_subregion.py \\
        --set 15F1 \\
        --tomogram 20240111_WaffleHipp_116 \\
        --alignment-dir best_alignment \\
        --zone-suffix active_zone_pre1_post1_az0

Example (manual crop region)
    python scripts/crop_active_zonogram_subregion.py \\
        --mrc /path/to/{tomogram}_active_zonogram_{zone}.mrc \\
        --crop 180,90,380,210 \\
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


def zonogram_findingampa_tuple(vol: torch.Tensor):
    return (np.eye(3), np.zeros(3), vol, ())


def _rotate_volume_around_y(vol: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate (Z, Y, X) volume around the Y axis."""
    from scipy.ndimage import affine_transform

    angle = np.deg2rad(angle_deg)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rot = np.array([[cos_a, 0.0, sin_a], [0.0, 1.0, 0.0], [-sin_a, 0.0, cos_a]])
    center = (np.array(vol.shape, dtype=float) - 1.0) / 2.0
    offset = center - rot @ center
    return affine_transform(vol, rot, offset=offset, order=1, mode="nearest")


def render_stereo_view(vol: torch.Tensor, angle_deg: float = 6.0):
    """Side-by-side stereo min projection pair (±angle around Y)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vmin, vmax = _imshow_limits(vol)
    vol_np = vol.numpy()
    left = np.min(_rotate_volume_around_y(vol_np, -angle_deg), axis=0)
    right = np.min(_rotate_volume_around_y(vol_np, angle_deg), axis=0)

    fig_w = (vol.shape[2] * 2) / 50
    fig_h = vol.shape[1] / 50
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(max(fig_w, 4), max(fig_h, 3)))
    for ax, img, label in (
        (ax_left, left, f"Left (-{angle_deg:g}°)"),
        (ax_right, right, f"Right (+{angle_deg:g}°)"),
    ):
        ax.imshow(img, cmap="gray", interpolation="mitchell", vmin=vmin, vmax=vmax, origin="lower")
        ax.axis("off")
        ax.set_title(label, fontsize=8)
    plt.tight_layout()
    return fig


def save_subregion(
    cropped: torch.Tensor,
    output_png: Path,
    *,
    output_mrc: Path | None = None,
    output_stereo_png: Path | None = None,
    stereo_angle_deg: float = 6.0,
    dpi: int = 100,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from synaptic_tomo_tools.visualization import render_active_zonograms_findingampa_style

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig = render_active_zonograms_findingampa_style(zonogram_findingampa_tuple(cropped))
    fig.savefig(output_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    if output_stereo_png is not None:
        fig_stereo = render_stereo_view(cropped, angle_deg=stereo_angle_deg)
        fig_stereo.savefig(output_stereo_png, dpi=dpi, bbox_inches="tight")
        plt.close(fig_stereo)
    if output_mrc is not None:
        mrcfile.write(output_mrc, cropped.numpy(), overwrite=True)


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

    save_subregion(
        cropped,
        png_path,
        output_mrc=mrc_out,
        output_stereo_png=stereo_out,
        stereo_angle_deg=args.stereo_angle,
    )
    print(f"Saved {png_path}")
    if stereo_out is not None:
        print(f"Saved {stereo_out}")
    if mrc_out is not None:
        print(f"Saved {mrc_out}")


if __name__ == "__main__":
    main()
