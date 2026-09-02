#!/usr/bin/env python3
"""
Build a supplementary-figure PDF (and optionally copy source assets) from
``data/supplementary_fig_list.txt``.

For each tomogram (grouped by set):
  - active zonogram position + matching zonogram PNG pair(s)
  - center Z slice from ``{tomoname}_full_rec_BP_3DCTF_BIN4_ddw.mrc`` (100 nm scale bar)
  - labels: tomogram name, cleft / active zone id, tissue quality

Optional per-tomogram overrides via CSV (see ``data/supplementary_fig_overrides.example.csv``).
"""
from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import mrcfile
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from synaptic_tomo_tools.alignment_utils import require_alignment_dir

DEFAULT_LIST = Path("data/supplementary_fig_list.txt")
DEFAULT_TOMOCSV = Path("tomogram_csv_files/tomograms_full_set_FINAL.csv")
DEFAULT_DATA_DIR = Path("data")
DEFAULT_OUTPUT_PDF = Path("results/supplementary_figure.pdf")

_TISSUE_DEFAULT = "tissue"
_SET_HEADER_RE = re.compile(r"^>\s*(.+?)\s*$")
_ENTRY_RE = re.compile(r"^(.+?)\s*\(([^)]+)\)\s*$")
_CLEFT_ID_RE = re.compile(r"active_zonogram_(\d+)_position\.png$")
_CLEFT_ID_HYPHEN_RE = re.compile(r"active-zonogram_(\d+)_position\.png$")

DEFAULT_SCALE_BAR_NM = 100.0
_DEFAULT_VOXEL_SIZE_NM = 1.0


def read_voxel_size_nm(mrc) -> tuple[float, float, float]:
    """Read in-plane and Z voxel size in nm from an open MRC (fallback 1 nm)."""
    vs = mrc.voxel_size
    vx, vy, vz = float(vs.x), float(vs.y), float(vs.z)
    if vx > 0 and vy > 0 and vz > 0:
        return (vx / 10.0, vy / 10.0, vz / 10.0)
    fallback = _DEFAULT_VOXEL_SIZE_NM
    return (fallback, fallback, fallback)


def add_scale_bar_to_grayscale_image(
    gray: np.ndarray,
    *,
    bar_length_nm: float = DEFAULT_SCALE_BAR_NM,
    voxel_size_nm_x: float = _DEFAULT_VOXEL_SIZE_NM,
    label: str | None = None,
) -> Image.Image:
    """Draw a horizontal scale bar on the bottom-right of a grayscale slice."""
    img = Image.fromarray(gray).convert("RGB")
    draw = ImageDraw.Draw(img)
    width, height = img.size
    bar_px = int(round(bar_length_nm / float(voxel_size_nm_x)))
    bar_px = max(8, min(bar_px, max(width // 3, 8)))
    margin = max(10, width // 40)
    thickness = max(3, height // 80)
    x2 = width - margin
    x1 = x2 - bar_px
    y2 = height - margin
    y1 = y2 - thickness
    outline = max(1, thickness // 3)
    for offset in range(outline, 0, -1):
        draw.rectangle(
            [x1 - offset, y1 - offset, x2 + offset, y2 + offset],
            fill="black",
        )
    draw.rectangle([x1, y1, x2, y2], fill="white")
    text = label if label is not None else f"{int(bar_length_nm)} nm"
    font = ImageFont.load_default()
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]
    text_x = x1 + (bar_px - text_w) // 2
    text_y = y1 - text_h - 4
    if text_y < 2:
        text_y = y2 + 4
    pad = 2
    draw.rectangle(
        [
            text_x - pad,
            text_y - pad,
            text_x + text_w + pad,
            text_y + text_h + pad,
        ],
        fill="black",
    )
    draw.text((text_x, text_y), text, fill="white", font=font)
    return img


@dataclass
class SupplementaryEntry:
    set_name: str
    tomoname: str
    tissue_quality: str = _TISSUE_DEFAULT


@dataclass
class TomogramOverride:
    tissue_quality: str | None = None
    alignment_dir: str | None = None
    cleft_ids: list[int] | None = None
    position_png: Path | None = None
    zonogram_png: Path | None = None
    mrc_path: Path | None = None
    slice_z: int | None = None
    tomogram_slice_png: Path | None = None


@dataclass
class CleftImagePair:
    cleft_id: int
    position_png: Path
    zonogram_png: Path


@dataclass
class ResolvedTomogramAssets:
    entry: SupplementaryEntry
    alignment_dir: str
    cleft_ids_display: str
    tissue_quality: str
    pairs: list[CleftImagePair] = field(default_factory=list)
    mrc_path: Path | None = None
    tomogram_slice_png: Path | None = None
    tomogram_root: Path | None = None
    warnings: list[str] = field(default_factory=list)


def parse_supplementary_list(path: Path) -> list[SupplementaryEntry]:
    entries: list[SupplementaryEntry] = []
    current_set: str | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        set_match = _SET_HEADER_RE.match(line)
        if set_match:
            current_set = set_match.group(1).strip()
            continue
        if current_set is None:
            raise ValueError(f"Tomogram entry before any set header: {line}")
        entry_match = _ENTRY_RE.match(line)
        if entry_match:
            tomoname = entry_match.group(1).strip()
            tissue = entry_match.group(2).strip()
        else:
            tomoname = line.strip()
            tissue = _TISSUE_DEFAULT
        entries.append(SupplementaryEntry(current_set, tomoname, tissue))
    return entries


def parse_cleft_ids_cell(cell: str | None) -> list[int] | None:
    if cell is None:
        return None
    text = str(cell).strip()
    if not text or text.lower() == "nan":
        return None
    out: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if part.isdigit():
            out.append(int(part))
        elif part.replace(".", "", 1).isdigit():
            out.append(int(float(part)))
    return out or None


def load_tomogram_csv_index(csv_path: Path) -> dict[tuple[str, str], list[dict]]:
    """Index CSV rows by (set, tomoname). Multiple alignment rows may exist."""
    index: dict[tuple[str, str], list[dict]] = {}
    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames or "alignment_dir" not in reader.fieldnames:
            raise ValueError(f"{csv_path} must include alignment_dir column.")
        for row in reader:
            tomoname = row["tomoname"].strip()
            set_name = row["set"].strip()
            alignment_dir = require_alignment_dir(
                row.get("alignment_dir"), context=f"tomogram {tomoname}"
            )
            cleft_ids = parse_cleft_ids_cell(row.get("cleft_IDs"))
            index.setdefault((set_name, tomoname), []).append(
                {
                    "alignment_dir": alignment_dir,
                    "cleft_ids": cleft_ids,
                }
            )
    return index


def load_overrides_csv(path: Path | None) -> dict[tuple[str, str], TomogramOverride]:
    if path is None or not path.is_file():
        return {}
    overrides: dict[tuple[str, str], TomogramOverride] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            set_name = (row.get("set") or "").strip()
            tomoname = (row.get("tomoname") or "").strip()
            if not set_name or not tomoname or set_name.startswith("#"):
                continue
            key = (set_name, tomoname)
            prev = overrides.get(key, TomogramOverride())
            tissue = (row.get("tissue_quality") or "").strip()
            alignment = (row.get("alignment_dir") or "").strip()
            cleft_cell = (row.get("cleft_id") or row.get("cleft_ids") or "").strip()
            position = (row.get("position_png") or "").strip()
            zonogram = (row.get("zonogram_png") or "").strip()
            mrc = (row.get("mrc_path") or "").strip()
            slice_png = (row.get("tomogram_slice_png") or "").strip()
            slice_z_cell = (row.get("slice_z") or "").strip()

            overrides[key] = TomogramOverride(
                tissue_quality=tissue or prev.tissue_quality,
                alignment_dir=alignment or prev.alignment_dir,
                cleft_ids=parse_cleft_ids_cell(cleft_cell) or prev.cleft_ids,
                position_png=Path(position) if position else prev.position_png,
                zonogram_png=Path(zonogram) if zonogram else prev.zonogram_png,
                mrc_path=Path(mrc) if mrc else prev.mrc_path,
                tomogram_slice_png=Path(slice_png) if slice_png else prev.tomogram_slice_png,
                slice_z=int(slice_z_cell) if slice_z_cell.isdigit() else prev.slice_z,
            )
    return overrides


def choose_csv_row(rows: list[dict], override: TomogramOverride | None) -> dict:
    if override and override.alignment_dir:
        for row in rows:
            if row["alignment_dir"] == override.alignment_dir:
                return row
        raise ValueError(
            f"No CSV row with alignment_dir={override.alignment_dir!r} "
            f"(override requested but not in tomogram CSV)."
        )
    for row in rows:
        if row["alignment_dir"] == "best_alignment":
            return row
    return rows[0]


def tomogram_path(data_dir: Path, set_name: str, tomoname: str) -> Path:
    return data_dir / set_name / "TOP_TOMOS" / tomoname


def discover_active_zonogram_dirs(alignment_path: Path) -> list[Path]:
    candidates = [
        alignment_path / "active_zonograms",
        alignment_path / "active_zonogram",
    ]
    return [p for p in candidates if p.is_dir()]


def default_position_zonogram_paths(
    alignment_path: Path,
    cleft_id: int,
) -> tuple[Path | None, Path | None]:
    for active_dir in discover_active_zonogram_dirs(alignment_path):
        underscore_pos = active_dir / f"active_zonogram_{cleft_id}_position.png"
        underscore_zono = active_dir / f"active_zonogram_{cleft_id}.png"
        hyphen_pos = active_dir / f"active-zonogram_{cleft_id}_position.png"
        hyphen_zono = active_dir / f"active-zonogram_{cleft_id}.png"
        if underscore_pos.is_file() and underscore_zono.is_file():
            return underscore_pos, underscore_zono
        if hyphen_pos.is_file() and hyphen_zono.is_file():
            return hyphen_pos, hyphen_zono
    return None, None


def discover_cleft_ids_from_pngs(alignment_path: Path) -> list[int]:
    found: set[int] = set()
    for active_dir in discover_active_zonogram_dirs(alignment_path):
        for png in active_dir.glob("*_position.png"):
            m = _CLEFT_ID_RE.match(png.name) or _CLEFT_ID_HYPHEN_RE.match(png.name)
            if m:
                found.add(int(m.group(1)))
    return sorted(found)


def default_mrc_path(alignment_path: Path, tomoname: str) -> Path | None:
    exact = alignment_path / f"{tomoname}_full_rec_BP_3DCTF_BIN4_ddw.mrc"
    if exact.is_file():
        return exact
    matches = sorted(alignment_path.glob("*ddw.mrc"))
    return matches[0] if matches else None


def render_center_slice_png(
    mrc_path: Path,
    output_png: Path,
    *,
    slice_z: int | None = None,
    scale_bar_nm: float = DEFAULT_SCALE_BAR_NM,
) -> Path:
    with mrcfile.open(mrc_path, mode="r") as mrc:
        data = np.asarray(mrc.data, dtype=np.float32)
        voxel_size_nm = read_voxel_size_nm(mrc)
    if data.ndim != 3:
        raise ValueError(f"Expected 3D MRC at {mrc_path}, got shape {data.shape}")
    z_index = slice_z if slice_z is not None else data.shape[0] // 2
    if z_index < 0 or z_index >= data.shape[0]:
        raise ValueError(f"slice_z={z_index} out of range for shape {data.shape}")
    slice2d = data[z_index]
    finite = slice2d[np.isfinite(slice2d)]
    if finite.size == 0:
        gray = np.zeros(slice2d.shape, dtype=np.uint8)
    else:
        lo, hi = np.percentile(finite, (1, 99))
        if hi <= lo:
            lo, hi = float(finite.min()), float(finite.max())
        if hi <= lo:
            scaled = np.zeros_like(slice2d, dtype=np.float32)
        else:
            scaled = np.clip((slice2d - lo) / (hi - lo), 0.0, 1.0)
        gray = (scaled * 255).astype(np.uint8)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    slice_img = add_scale_bar_to_grayscale_image(
        gray,
        bar_length_nm=scale_bar_nm,
        voxel_size_nm_x=voxel_size_nm[0],
    )
    slice_img.save(output_png)
    return output_png


def format_cleft_display(cleft_ids: Sequence[int]) -> str:
    if not cleft_ids:
        return "unknown"
    return ", ".join(str(i) for i in cleft_ids)


def resolve_tomogram_assets(
    entry: SupplementaryEntry,
    csv_index: dict[tuple[str, str], list[dict]],
    data_dir: Path,
    override: TomogramOverride | None,
    *,
    work_dir: Path | None = None,
    scale_bar_nm: float = DEFAULT_SCALE_BAR_NM,
) -> ResolvedTomogramAssets:
    key = (entry.set_name, entry.tomoname)
    rows = csv_index.get(key)
    warnings: list[str] = []
    if not rows:
        raise FileNotFoundError(
            f"No CSV row for set={entry.set_name!r}, tomoname={entry.tomoname!r}"
        )
    if len(rows) > 1 and not (override and override.alignment_dir):
        warnings.append(
            f"Multiple CSV alignment rows; using {choose_csv_row(rows, override)['alignment_dir']}"
        )
    csv_row = choose_csv_row(rows, override)
    alignment_dir = override.alignment_dir if override and override.alignment_dir else csv_row["alignment_dir"]
    alignment_dir = require_alignment_dir(alignment_dir, context=entry.tomoname)
    tissue = (
        override.tissue_quality
        if override and override.tissue_quality
        else entry.tissue_quality
    )
    cleft_ids = (
        override.cleft_ids
        if override and override.cleft_ids
        else csv_row.get("cleft_ids")
    )
    root = tomogram_path(data_dir, entry.set_name, entry.tomoname)
    alignment_path = root / alignment_dir
    if not alignment_path.is_dir():
        raise FileNotFoundError(f"Alignment directory not found: {alignment_path}")

    pairs: list[CleftImagePair] = []
    if override and override.position_png and override.zonogram_png:
        if not override.position_png.is_file():
            raise FileNotFoundError(f"Override position_png missing: {override.position_png}")
        if not override.zonogram_png.is_file():
            raise FileNotFoundError(f"Override zonogram_png missing: {override.zonogram_png}")
        cid = cleft_ids[0] if cleft_ids else 0
        pairs.append(
            CleftImagePair(cid, override.position_png, override.zonogram_png)
        )
    else:
        if cleft_ids is None:
            cleft_ids = discover_cleft_ids_from_pngs(alignment_path)
            if not cleft_ids:
                cleft_ids = [0]
                warnings.append("No cleft_IDs in CSV and no position PNGs found; trying cleft 0")
        for cid in cleft_ids:
            pos, zono = default_position_zonogram_paths(alignment_path, cid)
            if pos is None or zono is None:
                warnings.append(f"Missing active zonogram pair for cleft {cid}")
                continue
            pairs.append(CleftImagePair(cid, pos, zono))

    mrc_path = None
    if override and override.mrc_path:
        mrc_path = override.mrc_path
    else:
        mrc_path = default_mrc_path(alignment_path, entry.tomoname)
    if mrc_path is None or not mrc_path.is_file():
        warnings.append(f"No ddw MRC found under {alignment_path}")

    slice_png: Path | None = None
    if override and override.tomogram_slice_png and override.tomogram_slice_png.is_file():
        slice_png = override.tomogram_slice_png
    elif mrc_path is not None:
        target_dir = work_dir or Path(tempfile.gettempdir())
        target_dir.mkdir(parents=True, exist_ok=True)
        slice_png = target_dir / f"{entry.tomoname}_center_slice_z.png"
        render_center_slice_png(
            mrc_path,
            slice_png,
            slice_z=override.slice_z if override else None,
            scale_bar_nm=scale_bar_nm,
        )

    return ResolvedTomogramAssets(
        entry=entry,
        alignment_dir=alignment_dir,
        cleft_ids_display=format_cleft_display(
            cleft_ids if cleft_ids is not None else [p.cleft_id for p in pairs]
        ),
        tissue_quality=tissue,
        pairs=pairs,
        mrc_path=mrc_path,
        tomogram_slice_png=slice_png,
        tomogram_root=root,
        warnings=warnings,
    )


def copy_assets(
    assets: ResolvedTomogramAssets,
    copy_root: Path,
) -> list[Path]:
    dest_dir = copy_root / assets.entry.set_name / assets.entry.tomoname
    dest_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for pair in assets.pairs:
        for src in (pair.position_png, pair.zonogram_png):
            dst = dest_dir / src.name
            shutil.copy2(src, dst)
            copied.append(dst)
    if assets.mrc_path and assets.mrc_path.is_file():
        dst = dest_dir / assets.mrc_path.name
        shutil.copy2(assets.mrc_path, dst)
        copied.append(dst)
    if assets.tomogram_slice_png and assets.tomogram_slice_png.is_file():
        dst = dest_dir / "tomogram_center_slice.png"
        shutil.copy2(assets.tomogram_slice_png, dst)
        copied.append(dst)
    meta_path = dest_dir / "metadata.txt"
    meta_path.write_text(
        "\n".join(
            [
                f"tomoname: {assets.entry.tomoname}",
                f"set: {assets.entry.set_name}",
                f"alignment_dir: {assets.alignment_dir}",
                f"cleft_ids: {assets.cleft_ids_display}",
                f"tissue_quality: {assets.tissue_quality}",
            ]
        ),
        encoding="utf-8",
    )
    copied.append(meta_path)
    return copied


def _draw_image(
    c: canvas.Canvas,
    img_path: Path,
    x: float,
    y: float,
    max_width: float,
    max_height: float,
) -> float:
    if not img_path.is_file():
        return 0.0
    img = Image.open(img_path)
    iw, ih = img.size
    scale = min(max_width / iw, max_height / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    c.drawImage(ImageReader(img), x, y, width=nw, height=nh)
    return float(nh)


def build_pdf(
    grouped_assets: list[tuple[str, list[ResolvedTomogramAssets]]],
    output_pdf: Path,
) -> None:
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(output_pdf), pagesize=letter)
    width, height = letter
    margin = 36
    gap = 12

    for set_name, asset_list in grouped_assets:
        c.setFont("Helvetica-Bold", 20)
        c.drawString(margin, height - margin - 20, f"Set: {set_name}")
        c.showPage()

        for assets in asset_list:
            y = height - margin
            title_h = 34
            c.setFillColor(HexColor("#cccccc"))
            c.rect(margin - 6, y - title_h, width - 2 * margin + 12, title_h, fill=1, stroke=0)
            c.setFillColor("black")
            c.setFont("Helvetica-Bold", 16)
            c.drawString(margin, y - 22, assets.entry.tomoname)
            y -= title_h

            info_h = 50
            c.setFillColor(HexColor("#eeeeee"))
            c.rect(margin - 6, y - info_h, width - 2 * margin + 12, info_h, fill=1, stroke=0)
            c.setFillColor("black")
            c.setFont("Helvetica", 11)
            c.drawString(margin, y - 14, f"Cleft / active zone: {assets.cleft_ids_display}")
            c.drawString(margin, y - 28, f"Tissue quality: {assets.tissue_quality}")
            c.drawString(margin, y - 42, f"Alignment: {assets.alignment_dir}")
            y -= info_h + gap

            usable_width = width - 2 * margin
            pair_height = 220
            for pair in assets.pairs:
                side_w = (usable_width - gap) / 2
                c.setFont("Helvetica-Bold", 11)
                c.drawString(margin, y, f"Position (cleft {pair.cleft_id})")
                c.drawString(margin + side_w + gap, y, f"Active zonogram (cleft {pair.cleft_id})")
                y -= 12
                h1 = _draw_image(c, pair.position_png, margin, y - pair_height, side_w, pair_height)
                h2 = _draw_image(
                    c,
                    pair.zonogram_png,
                    margin + side_w + gap,
                    y - pair_height,
                    side_w,
                    pair_height,
                )
                y -= max(h1, h2) + gap

            if assets.tomogram_slice_png and assets.tomogram_slice_png.is_file():
                slice_height = 220
                c.setFont("Helvetica-Bold", 11)
                c.drawString(margin, y, "Tomogram center slice")
                y -= 12
                nh = _draw_image(
                    c,
                    assets.tomogram_slice_png,
                    margin,
                    y - slice_height,
                    usable_width,
                    slice_height,
                )
                y -= nh + gap

            if assets.warnings:
                c.setFont("Helvetica", 9)
                for warn in assets.warnings:
                    c.drawString(margin, y, f"Warning: {warn}")
                    y -= 11

            c.showPage()

    c.save()


def group_assets_by_set(
    entries: list[SupplementaryEntry],
    resolved: dict[tuple[str, str], ResolvedTomogramAssets],
) -> list[tuple[str, list[ResolvedTomogramAssets]]]:
    grouped: dict[str, list[ResolvedTomogramAssets]] = {}
    order: list[str] = []
    for entry in entries:
        key = (entry.set_name, entry.tomoname)
        if key not in resolved:
            continue
        if entry.set_name not in grouped:
            grouped[entry.set_name] = []
            order.append(entry.set_name)
        grouped[entry.set_name].append(resolved[key])
    return [(name, grouped[name]) for name in order]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepare supplementary figure PDF from supplementary_fig_list.txt"
    )
    parser.add_argument("--list", type=Path, default=DEFAULT_LIST, help="Supplementary list file")
    parser.add_argument("--tomocsv", type=Path, default=DEFAULT_TOMOCSV, help="Tomogram CSV")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Data root")
    parser.add_argument("--overrides", type=Path, default=None, help="Optional overrides CSV")
    parser.add_argument(
        "--output-pdf",
        type=Path,
        default=DEFAULT_OUTPUT_PDF,
        help="Output PDF path (use --no-pdf to skip)",
    )
    parser.add_argument(
        "--copy-assets-dir",
        type=Path,
        default=None,
        help="Copy all source images (and metadata) into this directory",
    )
    parser.add_argument("--no-pdf", action="store_true", help="Skip PDF generation")
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Directory for generated tomogram slice PNGs (default: temp)",
    )
    parser.add_argument(
        "--scale-bar-nm",
        type=float,
        default=DEFAULT_SCALE_BAR_NM,
        help="Scale bar length on extracted tomogram slices (default: 100 nm)",
    )
    args = parser.parse_args(argv)

    entries = parse_supplementary_list(args.list)
    csv_index = load_tomogram_csv_index(args.tomocsv)
    overrides = load_overrides_csv(args.overrides)

    work_dir = args.work_dir
    if work_dir is None and not args.no_pdf:
        work_dir = Path(tempfile.mkdtemp(prefix="supp_fig_slices_"))

    resolved: dict[tuple[str, str], ResolvedTomogramAssets] = {}
    errors: list[str] = []

    for entry in entries:
        key = (entry.set_name, entry.tomoname)
        override = overrides.get(key)
        try:
            assets = resolve_tomogram_assets(
                entry,
                csv_index,
                args.data_dir,
                override,
                work_dir=work_dir,
                scale_bar_nm=args.scale_bar_nm,
            )
            if not assets.pairs:
                raise FileNotFoundError("No active zonogram image pairs resolved")
            resolved[key] = assets
            for warn in assets.warnings:
                print(f"Warning [{entry.tomoname}]: {warn}")
            print(f"Resolved {entry.set_name}/{entry.tomoname} ({assets.alignment_dir})")
        except Exception as exc:
            msg = f"Failed {entry.set_name}/{entry.tomoname}: {exc}"
            print(msg)
            errors.append(msg)

    if not resolved:
        print("No tomograms resolved; nothing to do.")
        return 1

    if args.copy_assets_dir is not None:
        args.copy_assets_dir.mkdir(parents=True, exist_ok=True)
        n_copied = 0
        for assets in resolved.values():
            copied = copy_assets(assets, args.copy_assets_dir)
            n_copied += len(copied)
        print(f"Copied {n_copied} files under {args.copy_assets_dir}")

    if not args.no_pdf:
        grouped = group_assets_by_set(entries, resolved)
        build_pdf(grouped, args.output_pdf)
        print(f"PDF written: {args.output_pdf}")

    if errors:
        print("\nErrors:")
        for err in errors:
            print(f"  - {err}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
