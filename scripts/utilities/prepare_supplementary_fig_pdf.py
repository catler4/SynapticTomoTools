#!/usr/bin/env python3
"""
Build a supplementary-figure PDF (and optionally copy source assets) from
``data/supplementary_fig_list.txt``.

For each tomogram (grouped by set):
  - active zonogram position + matching zonogram PNG pair(s)
  - ``active_zonograms/slicer000.jpg`` (cleft 0) or ``slicer000_{cleft_id}.jpg``
    (later clefts) when present, else center Z slice from
    ``{tomoname}_full_rec_BP_3DCTF_BIN4_ddw.mrc`` (100 nm scale bar)
  - labels: tomogram name, cleft / active zone id, tissue quality

Joins ``tomograms_full_set_FINAL.csv``: each CSV alignment row and each cleft/active
zone gets its own PDF page (and copy subdirectory).

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
from tqdm import tqdm
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
_SET_HEADER_RE = re.compile(r"^>\s*(\S+)(?:\s*\((.+)\))?\s*$")
_ENTRY_RE = re.compile(r"^(.+?)\s*\(([^)]+)\)\s*$")
_CLEFT_ID_RE = re.compile(r"active_zonogram_(\d+)_position(?:_cropped)?\.png$")

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
    draw.text((text_x, text_y), text, fill="white", font=font)
    return img


@dataclass
class SupplementaryEntry:
    set_name: str
    tomoname: str
    tissue_quality: str = _TISSUE_DEFAULT
    set_display_name: str = ""

    def __post_init__(self) -> None:
        if not self.set_display_name:
            self.set_display_name = self.set_name


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
    cleft_id: int
    tissue_quality: str
    pair: CleftImagePair
    mrc_path: Path | None = None
    tomogram_slice_png: Path | None = None
    tomogram_root: Path | None = None
    warnings: list[str] = field(default_factory=list)


def parse_supplementary_list(path: Path) -> list[SupplementaryEntry]:
    entries: list[SupplementaryEntry] = []
    current_set: str | None = None
    current_set_display: str = ""
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        set_match = _SET_HEADER_RE.match(line)
        if set_match:
            current_set = set_match.group(1).strip()
            display = (set_match.group(2) or "").strip()
            current_set_display = display or current_set
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
        entries.append(
            SupplementaryEntry(
                current_set,
                tomoname,
                tissue,
                set_display_name=current_set_display,
            )
        )
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


def select_csv_rows(
    rows: list[dict],
    override: TomogramOverride | None,
) -> list[dict]:
    """Return CSV rows to render for one supplementary-list entry."""
    if override and override.alignment_dir:
        for row in rows:
            if row["alignment_dir"] == override.alignment_dir:
                return [row]
        raise ValueError(
            f"No CSV row with alignment_dir={override.alignment_dir!r} "
            f"(override requested but not in tomogram CSV)."
        )
    return list(rows)


def row_override(
    override: TomogramOverride | None,
    csv_row: dict,
) -> TomogramOverride | None:
    """Per-alignment override: tissue applies globally; file paths only when pinned."""
    if override is None:
        return None
    alignment_dir = csv_row["alignment_dir"]
    if override.alignment_dir and override.alignment_dir != alignment_dir:
        return None
    use_file_overrides = override.alignment_dir is not None or not any(
        [
            override.position_png,
            override.zonogram_png,
            override.mrc_path,
            override.tomogram_slice_png,
            override.slice_z is not None,
        ]
    )
    return TomogramOverride(
        tissue_quality=override.tissue_quality,
        alignment_dir=alignment_dir,
        cleft_ids=override.cleft_ids,
        position_png=override.position_png if use_file_overrides else None,
        zonogram_png=override.zonogram_png if use_file_overrides else None,
        mrc_path=override.mrc_path if use_file_overrides else None,
        slice_z=override.slice_z if use_file_overrides else None,
        tomogram_slice_png=override.tomogram_slice_png if use_file_overrides else None,
    )


def tomogram_path(data_dir: Path, set_name: str, tomoname: str) -> Path:
    return data_dir / set_name / "TOP_TOMOS" / tomoname


def discover_active_zonogram_dirs(alignment_path: Path) -> list[Path]:
    candidates = [
        alignment_path / "active_zonograms",
        alignment_path / "active_zonogram",
    ]
    return [p for p in candidates if p.is_dir()]


def _cleft_mip_png_candidates(active_dir: Path, cleft_id: int, set_name: str) -> list[Path]:
    """Cleft MIP image for the bottom panel (two-panel render only)."""
    return [active_dir / f"active_zonogram_{cleft_id}_two_panel.png"]


def default_position_zonogram_paths(
    alignment_path: Path,
    cleft_id: int,
    *,
    set_name: str,
) -> tuple[Path | None, Path | None]:
    for active_dir in discover_active_zonogram_dirs(alignment_path):
        position_candidates = [
            active_dir / f"active_zonogram_{cleft_id}_position_cropped.png",
            active_dir / f"active_zonogram_{cleft_id}_position.png",
        ]
        for pos in position_candidates:
            if not pos.is_file():
                continue
            for zono in _cleft_mip_png_candidates(active_dir, cleft_id, set_name):
                if zono.is_file():
                    return pos, zono
    return None, None


def discover_cleft_ids_from_pngs(alignment_path: Path) -> list[int]:
    found: set[int] = set()
    for active_dir in discover_active_zonogram_dirs(alignment_path):
        for png in active_dir.glob("*_position*.png"):
            m = _CLEFT_ID_RE.match(png.name)
            if m:
                found.add(int(m.group(1)))
    return sorted(found)


def default_slicer_jpg_path(alignment_path: Path, cleft_id: int) -> Path | None:
    """Per-cleft denoised slice, then shared ``slicer000.jpg`` if only one exists."""
    cleft_id = int(cleft_id)
    names = ["slicer000.jpg"] if cleft_id == 0 else [f"slicer000_{cleft_id}.jpg", "slicer000.jpg"]
    for active_dir in discover_active_zonogram_dirs(alignment_path):
        for filename in names:
            jpg = active_dir / filename
            if jpg.is_file():
                return jpg
    return None


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



def cleft_ids_for_row(
    alignment_path: Path,
    csv_row: dict,
    override: TomogramOverride | None,
    warnings: list[str],
) -> list[int]:
    if override and override.cleft_ids:
        return list(override.cleft_ids)
    cleft_ids = csv_row.get("cleft_ids")
    if cleft_ids is not None:
        return list(cleft_ids)
    discovered = discover_cleft_ids_from_pngs(alignment_path)
    if discovered:
        return discovered
    warnings.append("No cleft_IDs in CSV and no position PNGs found; trying cleft 0")
    return [0]


def count_planned_pages(
    entry: SupplementaryEntry,
    csv_index: dict[tuple[str, str], list[dict]],
    overrides: dict[tuple[str, str], TomogramOverride],
    data_dir: Path,
) -> int:
    """Estimate PDF pages for one supplementary-list entry (alignment × zone)."""
    key = (entry.set_name, entry.tomoname)
    rows = csv_index.get(key)
    if not rows:
        return 0
    try:
        selected_rows = select_csv_rows(rows, overrides.get(key))
    except ValueError:
        return 0
    override = overrides.get(key)
    n_pages = 0
    for csv_row in selected_rows:
        per_row_override = row_override(override, csv_row)
        if per_row_override and per_row_override.position_png and per_row_override.zonogram_png:
            n_pages += 1
            continue
        alignment_dir = require_alignment_dir(csv_row["alignment_dir"], context=entry.tomoname)
        alignment_path = tomogram_path(data_dir, entry.set_name, entry.tomoname) / alignment_dir
        if not alignment_path.is_dir():
            n_pages += 1
            continue
        cleft_ids = cleft_ids_for_row(alignment_path, csv_row, per_row_override, [])
        n_pages += len(cleft_ids)
    return n_pages


def resolve_tomogram_assets_for_row(
    entry: SupplementaryEntry,
    csv_row: dict,
    data_dir: Path,
    override: TomogramOverride | None,
    *,
    work_dir: Path | None = None,
    scale_bar_nm: float = DEFAULT_SCALE_BAR_NM,
) -> list[ResolvedTomogramAssets]:
    """One ``ResolvedTomogramAssets`` per cleft/active zone for this CSV alignment row."""
    warnings: list[str] = []
    alignment_dir = require_alignment_dir(csv_row["alignment_dir"], context=entry.tomoname)
    tissue = (
        override.tissue_quality
        if override and override.tissue_quality
        else entry.tissue_quality
    )
    root = tomogram_path(data_dir, entry.set_name, entry.tomoname)
    alignment_path = root / alignment_dir
    if not alignment_path.is_dir():
        raise FileNotFoundError(f"Alignment directory not found: {alignment_path}")

    zone_pairs: list[CleftImagePair] = []
    if override and override.position_png and override.zonogram_png:
        if not override.position_png.is_file():
            raise FileNotFoundError(f"Override position_png missing: {override.position_png}")
        if not override.zonogram_png.is_file():
            raise FileNotFoundError(f"Override zonogram_png missing: {override.zonogram_png}")
        cleft_ids = cleft_ids_for_row(alignment_path, csv_row, override, warnings)
        cid = cleft_ids[0]
        zone_pairs.append(
            CleftImagePair(cid, override.position_png, override.zonogram_png)
        )
    else:
        cleft_ids = cleft_ids_for_row(alignment_path, csv_row, override, warnings)
        for cid in cleft_ids:
            pos, zono = default_position_zonogram_paths(
                alignment_path, cid, set_name=entry.set_name
            )
            if pos is None or zono is None:
                warnings.append(f"Missing active zonogram pair for cleft {cid}")
                continue
            zone_pairs.append(CleftImagePair(cid, pos, zono))

    mrc_path = None
    if override and override.mrc_path:
        mrc_path = override.mrc_path
    else:
        mrc_path = default_mrc_path(alignment_path, entry.tomoname)
    if mrc_path is None or not mrc_path.is_file():
        warnings.append(f"No ddw MRC found under {alignment_path}")

    override_slice = (
        override.tomogram_slice_png
        if override and override.tomogram_slice_png and override.tomogram_slice_png.is_file()
        else None
    )
    mrc_slice_png: Path | None = None

    resolved: list[ResolvedTomogramAssets] = []
    for pair in zone_pairs:
        slice_png: Path | None = override_slice
        if slice_png is None:
            slicer_jpg = default_slicer_jpg_path(alignment_path, pair.cleft_id)
            if slicer_jpg is not None:
                slice_png = slicer_jpg
            elif mrc_path is not None:
                if mrc_slice_png is None:
                    target_dir = work_dir or Path(tempfile.gettempdir())
                    target_dir.mkdir(parents=True, exist_ok=True)
                    mrc_slice_png = (
                        target_dir / f"{entry.tomoname}_{alignment_dir}_center_slice_z.png"
                    )
                    render_center_slice_png(
                        mrc_path,
                        mrc_slice_png,
                        slice_z=override.slice_z if override else None,
                        scale_bar_nm=scale_bar_nm,
                    )
                slice_png = mrc_slice_png
        resolved.append(
            ResolvedTomogramAssets(
                entry=entry,
                alignment_dir=alignment_dir,
                cleft_id=pair.cleft_id,
                tissue_quality=tissue,
                pair=pair,
                mrc_path=mrc_path,
                tomogram_slice_png=slice_png,
                tomogram_root=root,
                warnings=list(warnings),
            )
        )
    return resolved


def resolve_all_assets_for_entry(
    entry: SupplementaryEntry,
    csv_index: dict[tuple[str, str], list[dict]],
    data_dir: Path,
    override: TomogramOverride | None,
    *,
    work_dir: Path | None = None,
    scale_bar_nm: float = DEFAULT_SCALE_BAR_NM,
) -> list[ResolvedTomogramAssets]:
    """Resolve one supplementary-list entry; multiple alignments and zones → multiple pages."""
    key = (entry.set_name, entry.tomoname)
    rows = csv_index.get(key)
    if not rows:
        raise FileNotFoundError(
            f"No CSV row for set={entry.set_name!r}, tomoname={entry.tomoname!r}"
        )
    selected_rows = select_csv_rows(rows, override)
    assets_list: list[ResolvedTomogramAssets] = []
    for csv_row in selected_rows:
        per_row_override = row_override(override, csv_row)
        batch = resolve_tomogram_assets_for_row(
            entry,
            csv_row,
            data_dir,
            per_row_override,
            work_dir=work_dir,
            scale_bar_nm=scale_bar_nm,
        )
        if not batch:
            raise FileNotFoundError(
                f"No active zonogram image pairs for alignment {csv_row['alignment_dir']}"
            )
        assets_list.extend(batch)
    return assets_list


def copy_assets(
    assets: ResolvedTomogramAssets,
    copy_root: Path,
) -> list[Path]:
    dest_dir = (
        copy_root
        / assets.entry.set_name
        / assets.entry.tomoname
        / assets.alignment_dir
        / f"cleft_{assets.cleft_id}"
    )
    dest_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for src in (assets.pair.position_png, assets.pair.zonogram_png):
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
                f"cleft_id: {assets.cleft_id}",
                f"tissue_designation: {assets.tissue_quality}",
            ]
        ),
        encoding="utf-8",
    )
    copied.append(meta_path)
    return copied


def _max_fit_height(iw: int, ih: int, max_width: float, max_height: float) -> float:
    if iw <= 0 or ih <= 0:
        return 0.0
    return min(max_height, max_width * ih / iw)


def _common_fit_height(
    img_paths: list[Path],
    max_width: float,
    max_height: float,
) -> float:
    heights: list[float] = []
    for img_path in img_paths:
        if not img_path.is_file():
            continue
        with Image.open(img_path) as img:
            iw, ih = img.size
        heights.append(_max_fit_height(iw, ih, max_width, max_height))
    if not heights:
        return 0.0
    return min(heights)


def _image_fit_size(iw: int, ih: int, max_width: float, max_height: float) -> tuple[int, int]:
    if iw <= 0 or ih <= 0:
        return 0, 0
    scale = min(max_width / iw, max_height / ih)
    return max(1, int(iw * scale)), max(1, int(ih * scale))


def _image_fit_size_fill_width(
    iw: int,
    ih: int,
    max_width: float,
    max_height: float,
) -> tuple[int, int]:
    """Scale to ``max_width`` when height allows; otherwise limit by height."""
    if iw <= 0 or ih <= 0:
        return 0, 0
    scale = max_width / iw
    if ih * scale > max_height:
        scale = max_height / ih
    return max(1, int(iw * scale)), max(1, int(ih * scale))


def _draw_image_top_aligned(
    c: canvas.Canvas,
    img_path: Path,
    x: float,
    y_top: float,
    max_width: float,
    max_height: float,
    *,
    target_height: float | None = None,
) -> float:
    """Draw image with its top edge at ``y_top``; return drawn height."""
    if not img_path.is_file():
        return 0.0
    img = Image.open(img_path)
    iw, ih = img.size
    if target_height is not None and target_height > 0:
        nh = max(1, int(round(target_height)))
        nw = max(1, int(round(iw * nh / ih)))
        if nw > max_width:
            nw = max(1, int(max_width))
    else:
        nw, nh = _image_fit_size(iw, ih, max_width, max_height)
    if nh == 0:
        return 0.0
    y_bottom = y_top - nh
    x_draw = x + (max_width - nw) / 2.0
    c.drawImage(ImageReader(img), x_draw, y_bottom, width=nw, height=nh)
    return float(nh)


def _draw_image_top_aligned_fill_width(
    c: canvas.Canvas,
    img_path: Path,
    x: float,
    y_top: float,
    max_width: float,
    max_height: float,
) -> float:
    """Draw image at full available width when possible (for Cleft MIP)."""
    if not img_path.is_file():
        return 0.0
    img = Image.open(img_path)
    iw, ih = img.size
    nw, nh = _image_fit_size_fill_width(iw, ih, max_width, max_height)
    if nh == 0:
        return 0.0
    y_bottom = y_top - nh
    x_draw = x + (max_width - nw) / 2.0
    c.drawImage(ImageReader(img), x_draw, y_bottom, width=nw, height=nh)
    return float(nh)


def build_pdf(
    grouped_assets: list[tuple[str, list[ResolvedTomogramAssets]]],
    output_pdf: Path,
    *,
    page_progress: tqdm | None = None,
) -> None:
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(output_pdf), pagesize=letter)
    width, height = letter
    margin = 36
    gap = 12
    label_h = 14
    top_row_image_frac = 0.54

    def _draw_page_header(assets: ResolvedTomogramAssets) -> float:
        """Draw title + info blocks; return ``y_top`` below header."""
        y_top = height - margin
        title_row_h = 26
        c.setFillColor(HexColor("#cccccc"))
        c.rect(
            margin - 6,
            y_top - title_row_h,
            width - 2 * margin + 12,
            title_row_h,
            fill=1,
            stroke=0,
        )
        c.setFillColor("black")
        c.setFont("Helvetica-Bold", 11)
        c.drawString(margin, y_top - 16, f"Tomogram ID: {assets.entry.tomoname}")
        set_text = f"SET: {assets.entry.set_display_name}"
        set_text_width = c.stringWidth(set_text, "Helvetica-Bold", 11)
        c.drawString(width - margin - set_text_width, y_top - 16, set_text)
        y_top -= title_row_h

        info_row_h = 26
        c.setFillColor(HexColor("#eeeeee"))
        c.rect(
            margin - 6,
            y_top - info_row_h,
            width - 2 * margin + 12,
            info_row_h,
            fill=1,
            stroke=0,
        )
        c.setFillColor("black")
        c.setFont("Helvetica", 11)
        info_y = y_top - 16
        c.drawString(margin, info_y, f"Cleft ID: {assets.cleft_id}")
        tissue_text = f"Type: {assets.tissue_quality}"
        tissue_text_width = c.stringWidth(tissue_text, "Helvetica", 11)
        c.drawString(width - margin - tissue_text_width, info_y, tissue_text)
        return y_top - info_row_h - gap

    for _set_name, asset_list in grouped_assets:
        for assets in asset_list:
            usable_width = width - 2 * margin
            side_w = (usable_width - gap) / 2
            has_slice = (
                assets.tomogram_slice_png is not None and assets.tomogram_slice_png.is_file()
            )
            pair = assets.pair

            y_top = _draw_page_header(assets)

            labels_and_gaps = label_h + gap + label_h + gap
            image_area = y_top - margin - labels_and_gaps
            top_row_cap = max(100.0, image_area * top_row_image_frac)
            position_label = "Membrane Segmentation & Cleft Position"

            c.setFont("Helvetica-Bold", 11)
            if has_slice:
                c.drawString(margin, y_top - 10, "Denoised Tomogram Slice")
                c.drawString(
                    margin + side_w + gap,
                    y_top - 10,
                    position_label,
                )
            else:
                c.drawString(margin, y_top - 10, position_label)
            y_top -= label_h

            top_row_h = 0.0
            if has_slice:
                top_row_h = _common_fit_height(
                    [assets.tomogram_slice_png, pair.position_png],
                    side_w,
                    top_row_cap,
                )
            h_slice = 0.0
            if has_slice:
                h_slice = _draw_image_top_aligned(
                    c,
                    assets.tomogram_slice_png,
                    margin,
                    y_top,
                    side_w,
                    top_row_cap,
                    target_height=top_row_h,
                )
            position_w = side_w if has_slice else usable_width
            position_x = margin + side_w + gap if has_slice else margin
            h_position = _draw_image_top_aligned(
                c,
                pair.position_png,
                position_x,
                y_top,
                position_w,
                top_row_cap,
                target_height=top_row_h if has_slice else None,
            )
            y_top -= (top_row_h if has_slice else max(h_position, h_slice)) + gap

            mip_cap = y_top - margin - label_h
            if mip_cap < 80:
                c.showPage()
                y_top = height - margin
                c.setFont("Helvetica-Bold", 12)
                c.drawString(
                    margin,
                    y_top - 12,
                    f"Tomogram ID: {assets.entry.tomoname} — Cleft MIP",
                )
                y_top -= label_h + gap
                mip_cap = y_top - margin
            else:
                c.setFont("Helvetica-Bold", 11)
                c.drawString(margin, y_top - 10, "Cleft MIP")
                y_top -= label_h
            mip_h = _draw_image_top_aligned_fill_width(
                c,
                pair.zonogram_png,
                margin,
                y_top,
                usable_width,
                mip_cap,
            )
            y_top -= mip_h + gap

            if assets.warnings:
                c.setFont("Helvetica", 9)
                for warn in assets.warnings:
                    if y_top < margin + 10:
                        break
                    c.drawString(margin, y_top - 8, f"Warning: {warn}")
                    y_top -= 11

            c.showPage()
            if page_progress is not None:
                page_progress.update(1)

    c.save()


def _require_existing_file(path: Path, arg_name: str) -> Path:
    path = Path(path)
    if path.is_dir():
        raise ValueError(
            f"{arg_name} must be a file, but {path} is a directory."
        )
    if not path.is_file():
        raise FileNotFoundError(f"{arg_name} not found: {path}")
    return path


def _require_output_pdf_path(path: Path) -> Path:
    path = Path(path)
    if path.is_dir():
        suggested = path / "supplementary_figure.pdf"
        raise ValueError(
            f"--output-pdf must be a PDF file path, but {path} is a directory. "
            f"Use e.g. {suggested} for the PDF and --copy-assets-dir {path} "
            "if you want assets copied into that folder."
        )
    if path.suffix.lower() != ".pdf":
        raise ValueError(
            f"--output-pdf should end with .pdf (got {path}). "
            "If you meant an output folder, use --copy-assets-dir instead."
        )
    return path


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
    parser.add_argument(
        "--test",
        action="store_true",
        help="Process only the first 3 tomograms from the list (for troubleshooting)",
    )
    args = parser.parse_args(argv)

    args.list = _require_existing_file(args.list, "--list")
    args.tomocsv = _require_existing_file(args.tomocsv, "--tomocsv")
    if args.overrides is not None:
        args.overrides = _require_existing_file(args.overrides, "--overrides")
    if not args.no_pdf:
        args.output_pdf = _require_output_pdf_path(args.output_pdf)

    entries = parse_supplementary_list(args.list)
    if args.test:
        entries = entries[:3]
        tqdm.write(f"Test mode: processing first {len(entries)} tomogram(s) from the list")
    csv_index = load_tomogram_csv_index(args.tomocsv)
    overrides = load_overrides_csv(args.overrides)

    work_dir = args.work_dir
    if work_dir is None and not args.no_pdf:
        work_dir = Path(tempfile.mkdtemp(prefix="supp_fig_slices_"))

    resolved_ordered: list[ResolvedTomogramAssets] = []
    grouped: dict[str, list[ResolvedTomogramAssets]] = {}
    set_order: list[str] = []
    errors: list[str] = []

    total_pages = sum(
        count_planned_pages(entry, csv_index, overrides, args.data_dir) for entry in entries
    )
    resolve_bar = tqdm(total=total_pages, desc="Resolving pages", unit="page")

    for entry in entries:
        key = (entry.set_name, entry.tomoname)
        override = overrides.get(key)
        try:
            assets_batch = resolve_all_assets_for_entry(
                entry,
                csv_index,
                args.data_dir,
                override,
                work_dir=work_dir,
                scale_bar_nm=args.scale_bar_nm,
            )
            for assets in assets_batch:
                for warn in assets.warnings:
                    resolve_bar.write(
                        f"Warning [{entry.tomoname}/{assets.alignment_dir}/"
                        f"cleft {assets.cleft_id}]: {warn}"
                    )
                resolve_bar.set_postfix_str(
                    f"{entry.tomoname} {assets.alignment_dir} cleft{assets.cleft_id}",
                    refresh=False,
                )
                resolve_bar.update(1)
            resolved_ordered.extend(assets_batch)
            if entry.set_name not in grouped:
                grouped[entry.set_name] = []
                set_order.append(entry.set_name)
            grouped[entry.set_name].extend(assets_batch)
        except Exception as exc:
            msg = f"Failed {entry.set_name}/{entry.tomoname}: {exc}"
            resolve_bar.write(msg)
            errors.append(msg)

    resolve_bar.close()

    if not resolved_ordered:
        print("No tomograms resolved; nothing to do.")
        return 1

    if args.copy_assets_dir is not None:
        args.copy_assets_dir.mkdir(parents=True, exist_ok=True)
        n_copied = 0
        copy_bar = tqdm(resolved_ordered, desc="Copying assets", unit="page")
        for assets in copy_bar:
            copy_bar.set_postfix_str(
                f"{assets.entry.tomoname} {assets.alignment_dir} cleft{assets.cleft_id}",
                refresh=False,
            )
            copied = copy_assets(assets, args.copy_assets_dir)
            n_copied += len(copied)
        copy_bar.close()
        tqdm.write(f"Copied {n_copied} files under {args.copy_assets_dir}")

    if not args.no_pdf:
        pdf_bar = tqdm(total=len(resolved_ordered), desc="Writing PDF", unit="page")
        build_pdf(
            [(name, grouped[name]) for name in set_order],
            args.output_pdf,
            page_progress=pdf_bar,
        )
        pdf_bar.close()
        tqdm.write(f"PDF written: {args.output_pdf}")

    if errors:
        tqdm.write("\nErrors:")
        for err in errors:
            tqdm.write(f"  - {err}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
