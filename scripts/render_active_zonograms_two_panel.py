#!/usr/bin/env python3
"""
Re-render findingampa-style active zonogram PNGs with only two panels
(main XY + bottom XZ; no right YZ panel).

Reads a tomogram CSV (``set``, ``tomoname``/``tomogram``, ``alignment_dir``),
switches into each ``{data_dir}/{set}/TOP_TOMOS/{tomoname}/{alignment_dir}/``,
and re-renders PNGs from existing ``active_zonograms/active_zonogram_{i}.mrc``
volumes (the active-zone region from a prior ``finding_ampa render-active-zonograms``
run is reused; it is not recomputed).

Default writes ``active_zonogram_{i}_two_panel.png`` next to the originals
(does not overwrite ``active_zonogram_{i}.png``).

Example
-------
python scripts/render_active_zonograms_two_panel.py \\
    --csv tomogram_csv_files/tomograms_15F1_FINAL.csv \\
    --data-dir /path/to/data
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
import mrcfile
import numpy as np
import torch

_AZ_MRC_RE = re.compile(r"^active_zonogram_(\d+)\.mrc$")


def require_alignment_dir(alignment_dir: object, *, context: str = "") -> str:
    if alignment_dir is None:
        msg = "alignment_dir is required and cannot be None."
        if context:
            msg = f"{msg} ({context})"
        raise ValueError(msg)
    text = str(alignment_dir).strip()
    if not text or text.lower() in ("nan", "none"):
        msg = "alignment_dir must be a non-empty string from the tomogram CSV."
        if context:
            msg = f"{msg} ({context})"
        raise ValueError(msg)
    return text


def render_active_zonograms_two_panel(cleft_data):
    """
    Findingampa-style active zonogram with only main (XY) and bottom (XZ) panels.

    Drops the right (YZ) panel from findingampa's ``render_active_zonograms``.
    ``cleft_data[2]`` may be a torch tensor or a numpy ndarray.
    """
    res_ddw = cleft_data[2]
    if not torch.is_tensor(res_ddw):
        res_ddw = torch.as_tensor(res_ddw)

    width = res_ddw.shape[2] / 50
    height = (res_ddw.shape[1] + res_ddw.shape[0]) / 50
    fig = plt.figure(figsize=(width, height))
    gs = gridspec.GridSpec(
        2,
        1,
        height_ratios=[res_ddw.shape[1], res_ddw.shape[0]],
    )

    axxy = plt.subplot(gs[0, 0])
    axxz = plt.subplot(gs[1, 0], sharex=axxy)

    vmin = float(-20 * res_ddw.std())
    axxy.imshow(
        torch.min(res_ddw, axis=0).values,
        cmap="gray",
        interpolation="mitchell",
        vmax=-0.0,
        vmin=vmin,
        origin="lower",
    )
    axxy.quiver(
        0, 0, 0, 50, color="g", angles="xy", scale_units="xy", units="xy", width=1, scale=1, clip_on=False
    )
    axxy.quiver(
        0, 0, 50, 0, color="r", angles="xy", scale_units="xy", units="xy", width=1, scale=1, clip_on=False
    )

    axxz.imshow(
        torch.min(res_ddw, axis=1).values,
        cmap="gray",
        interpolation="mitchell",
        vmax=-0.0,
        vmin=vmin,
        origin="lower",
    )
    axxz.quiver(
        0, 0, 0, 50, color="b", angles="xy", scale_units="xy", units="xy", width=1, scale=1, clip_on=False
    )
    axxz.quiver(
        0, 0, 50, 0, color="r", angles="xy", scale_units="xy", units="xy", width=1, scale=1, clip_on=False
    )

    axxy.axis("off")
    axxz.axis("off")
    plt.tight_layout()
    return fig


def discover_active_zonogram_mrcs(active_dir: Path) -> list[tuple[int, Path]]:
    found: list[tuple[int, Path]] = []
    for path in sorted(active_dir.glob("active_zonogram_*.mrc")):
        m = _AZ_MRC_RE.match(path.name)
        if m:
            found.append((int(m.group(1)), path))
    return found


def resolve_active_dir(alignment_dir: Path) -> Path:
    for name in ("active_zonograms", "active_zonogram"):
        candidate = alignment_dir / name
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"No active_zonograms/ folder under {alignment_dir}. "
        "Run finding_ampa render-active-zonograms first."
    )


def resolve_alignment_path(data_dir: Path, set_name: str, tomoname: str, alignment: str) -> Path:
    primary = data_dir / set_name / "TOP_TOMOS" / tomoname / alignment
    if primary.is_dir():
        return primary
    alt = data_dir / set_name / tomoname / alignment
    if alt.is_dir():
        return alt
    return primary


def render_png_from_mrc(mrc_path: Path, output_png: Path, *, dpi: float = 150.0) -> Path:
    with mrcfile.open(mrc_path, mode="r") as mrc:
        volume = np.asarray(mrc.data, dtype=np.float32).copy()
    fig = render_active_zonograms_two_panel((None, None, torch.as_tensor(volume)))
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return output_png


def process_alignment_cwd(
    *,
    suffix: str = "",
    dpi: float = 150.0,
    dry_run: bool = False,
) -> list[Path]:
    """Process ``active_zonograms/`` relative to the current working directory."""
    active_dir = resolve_active_dir(Path.cwd())
    mrcs = discover_active_zonogram_mrcs(active_dir)
    if not mrcs:
        raise FileNotFoundError(
            f"No active_zonogram_*.mrc files in {active_dir}. "
            "Run finding_ampa render-active-zonograms first."
        )

    written: list[Path] = []
    for az_id, mrc_path in mrcs:
        out_path = active_dir / f"active_zonogram_{az_id}{suffix}.png"
        if dry_run:
            print(f"  Would render {mrc_path.name} -> {out_path}")
            written.append(out_path)
            continue
        render_png_from_mrc(mrc_path, out_path, dpi=dpi)
        print(f"  Rendered: {out_path}")
        written.append(out_path)
    return written


def iter_csv_rows(csv_path: Path):
    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            raise ValueError(f"{csv_path} has no header")
        fields = {f.strip().lower(): f for f in reader.fieldnames}
        set_col = fields.get("set")
        tomo_col = fields.get("tomoname") or fields.get("tomogram")
        align_col = fields.get("alignment_dir")
        if not set_col or not tomo_col or not align_col:
            raise ValueError(
                f"{csv_path} must include set, tomoname/tomogram, and alignment_dir columns"
            )
        for row in reader:
            set_name = (row.get(set_col) or "").strip()
            tomoname = (row.get(tomo_col) or "").strip()
            alignment = (row.get(align_col) or "").strip()
            if not set_name or not tomoname or set_name.startswith("#"):
                continue
            yield set_name, tomoname, require_alignment_dir(alignment, context=tomoname)


def process_csv(
    csv_path: Path,
    data_dir: Path,
    *,
    suffix: str = "",
    dpi: float = 150.0,
    dry_run: bool = False,
) -> list[Path]:
    written: list[Path] = []
    start_cwd = Path.cwd()
    n_ok = 0
    n_skip = 0

    for set_name, tomoname, alignment in iter_csv_rows(csv_path):
        alignment_path = resolve_alignment_path(data_dir, set_name, tomoname, alignment)
        label = f"{set_name}/{tomoname}/{alignment}"
        if not alignment_path.is_dir():
            print(f"Skipping missing directory: {alignment_path}")
            n_skip += 1
            continue

        print(f"Entering {alignment_path}")
        os.chdir(alignment_path)
        try:
            batch = process_alignment_cwd(suffix=suffix, dpi=dpi, dry_run=dry_run)
            written.extend(batch)
            n_ok += 1
            print(f"Completed {label} ({len(batch)} PNG(s))")
        except FileNotFoundError as exc:
            print(f"Skipping {label}: {exc}")
            n_skip += 1
        finally:
            os.chdir(start_cwd)

    print(f"Finished CSV: {n_ok} alignment(s) processed, {n_skip} skipped")
    return written


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Re-render active zonogram PNGs (main XY + bottom XZ only) for every "
            "tomogram/alignment row in a CSV, switching into each alignment directory."
        )
    )
    p.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="Tomogram CSV with set, tomoname/tomogram, and alignment_dir columns",
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Root data directory containing {set}/TOP_TOMOS/{tomoname}/{alignment_dir}/",
    )
    p.add_argument(
        "--suffix",
        type=str,
        default="_two_panel",
        help=(
            "Filename suffix before .png (default: _two_panel -> "
            "active_zonogram_0_two_panel.png). Does not overwrite the original "
            "active_zonogram_{i}.png."
        ),
    )
    p.add_argument("--dpi", type=float, default=150.0, help="PNG DPI (default: 150)")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="List MRC -> PNG paths without writing files",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    suffix = args.suffix or ""
    if suffix and not suffix.startswith("_") and not suffix.startswith("-"):
        suffix = f"_{suffix}"

    if not args.csv.is_file():
        print(f"Error: CSV not found: {args.csv}", file=sys.stderr)
        return 1
    if not args.data_dir.is_dir():
        print(f"Error: data directory not found: {args.data_dir}", file=sys.stderr)
        return 1

    written = process_csv(
        args.csv,
        args.data_dir,
        suffix=suffix,
        dpi=args.dpi,
        dry_run=args.dry_run,
    )
    print(f"Done: {len(written)} PNG(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
