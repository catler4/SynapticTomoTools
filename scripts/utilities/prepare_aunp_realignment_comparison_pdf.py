#!/usr/bin/env python3
"""
Build a PDF comparing active zonograms between ``best_alignment`` and a *liza*
alignment directory for the same tomogram/cleft.

Only CSV rows whose ``alignment_dir`` contains ``liza`` (case-insensitive) are used.
Each page shows that cleft's active zonogram from ``best_alignment`` (left) and from
the indicated liza alignment (right).
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from tqdm import tqdm
from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

_SCRIPTS_UTIL = Path(__file__).resolve().parent
_SRC = Path(__file__).resolve().parent.parent.parent / "src"
if str(_SCRIPTS_UTIL) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_UTIL))
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from synaptic_tomo_tools.alignment_utils import require_alignment_dir  # noqa: E402

from prepare_supplementary_fig_pdf import (  # noqa: E402
    DEFAULT_DATA_DIR,
    _draw_image_top_aligned,
    _common_fit_height,
    discover_active_zonogram_dirs,
    discover_cleft_ids_from_pngs,
    tomogram_path,
)

DEFAULT_TOMOCSV = Path("tomogram_csv_files/tomograms_set_best_realign_examples.csv")
DEFAULT_OUTPUT_PDF = Path("results/aunp_realignment_comparison.pdf")
BEST_ALIGNMENT = "best_alignment"
_LIZA_RE = re.compile(r"liza", re.IGNORECASE)
_INT_RE = re.compile(r"\d+")


@dataclass
class ComparisonPage:
    set_name: str
    tomoname: str
    liza_alignment_dir: str
    cleft_id: int
    best_zonogram: Path
    liza_zonogram: Path
    warnings: list[str] = field(default_factory=list)


def parse_cleft_ids_cell(cell: str | None) -> list[int] | None:
    """Parse cleft IDs; tolerate trailing notes like ``0 **membrane also improves``."""
    if cell is None:
        return None
    text = str(cell).strip()
    if not text or text.lower() == "nan" or text.startswith("---"):
        return None
    out: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        m = _INT_RE.search(part)
        if m:
            out.append(int(m.group(0)))
    return out or None


def load_liza_csv_rows(csv_path: Path) -> list[dict]:
    """Load CSV rows whose alignment_dir matches *liza*."""
    rows: list[dict] = []
    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            raise ValueError(f"{csv_path} has no header")
        required = {"tomoname", "set", "alignment_dir"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"{csv_path} missing columns: {sorted(missing)}")
        for row in reader:
            tomoname = (row.get("tomoname") or "").strip()
            set_name = (row.get("set") or "").strip()
            if not tomoname or not set_name or tomoname.startswith("---"):
                continue
            try:
                alignment_dir = require_alignment_dir(
                    row.get("alignment_dir"), context=f"tomogram {tomoname}"
                )
            except ValueError:
                continue
            if not _LIZA_RE.search(alignment_dir):
                continue
            rows.append(
                {
                    "tomoname": tomoname,
                    "set": set_name,
                    "alignment_dir": alignment_dir,
                    "cleft_ids": parse_cleft_ids_cell(row.get("cleft_IDs")),
                }
            )
    return rows


def resolve_zonogram_png(
    data_dir: Path,
    set_name: str,
    tomoname: str,
    alignment_dir: str,
    cleft_id: int,
) -> Path | None:
    """Return the active-zonogram MIP PNG for one alignment/cleft, if present.

    Prefers ``active_zonogram_{id}_two_panel.png`` (same as the supplementary
    figure), then falls back to ``active_zonogram_{id}.png``.
    """
    alignment_path = tomogram_path(data_dir, set_name, tomoname) / alignment_dir
    if not alignment_path.is_dir():
        return None
    candidates: list[Path] = []
    for active_dir in discover_active_zonogram_dirs(alignment_path):
        candidates.extend(
            [
                active_dir / f"active_zonogram_{cleft_id}_two_panel.png",
                active_dir / f"active_zonogram_{cleft_id}.png",
            ]
        )
    for path in candidates:
        if path.is_file():
            return path
    return None


def cleft_ids_for_comparison(
    data_dir: Path,
    row: dict,
) -> list[int]:
    if row.get("cleft_ids"):
        return list(row["cleft_ids"])
    # Prefer clefts discovered under the liza alignment; fall back to best_alignment.
    for align in (row["alignment_dir"], BEST_ALIGNMENT):
        alignment_path = tomogram_path(data_dir, row["set"], row["tomoname"]) / align
        if alignment_path.is_dir():
            found = discover_cleft_ids_from_pngs(alignment_path)
            if found:
                return found
    return [0]


def resolve_comparison_pages(
    rows: list[dict],
    data_dir: Path,
) -> tuple[list[ComparisonPage], list[str]]:
    pages: list[ComparisonPage] = []
    errors: list[str] = []
    for row in rows:
        set_name = row["set"]
        tomoname = row["tomoname"]
        liza_dir = row["alignment_dir"]
        try:
            cleft_ids = cleft_ids_for_comparison(data_dir, row)
        except Exception as exc:
            errors.append(f"{set_name}/{tomoname}: {exc}")
            continue
        for cid in cleft_ids:
            warnings: list[str] = []
            best_png = resolve_zonogram_png(
                data_dir, set_name, tomoname, BEST_ALIGNMENT, cid
            )
            liza_png = resolve_zonogram_png(
                data_dir, set_name, tomoname, liza_dir, cid
            )
            if best_png is None:
                errors.append(
                    f"{set_name}/{tomoname}/cleft {cid}: missing zonogram under "
                    f"{BEST_ALIGNMENT}"
                )
                continue
            if liza_png is None:
                errors.append(
                    f"{set_name}/{tomoname}/cleft {cid}: missing zonogram under "
                    f"{liza_dir}"
                )
                continue
            pages.append(
                ComparisonPage(
                    set_name=set_name,
                    tomoname=tomoname,
                    liza_alignment_dir=liza_dir,
                    cleft_id=cid,
                    best_zonogram=best_png,
                    liza_zonogram=liza_png,
                    warnings=warnings,
                )
            )
    return pages, errors


def build_comparison_pdf(
    pages: list[ComparisonPage],
    output_pdf: Path,
    *,
    page_progress: tqdm | None = None,
) -> None:
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(output_pdf), pagesize=letter)
    width, height = letter
    margin = 36
    gap = 14
    label_h = 14

    for page in pages:
        usable_width = width - 2 * margin
        side_w = (usable_width - gap) / 2.0
        y_top = height - margin

        # Header
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
        c.drawString(margin, y_top - 16, f"Tomogram ID: {page.tomoname}")
        set_text = f"SET: {page.set_name}"
        set_w = c.stringWidth(set_text, "Helvetica-Bold", 11)
        c.drawString(width - margin - set_w, y_top - 16, set_text)
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
        c.drawString(margin, info_y, f"Cleft ID: {page.cleft_id}")
        compare_text = f"{BEST_ALIGNMENT}  vs  {page.liza_alignment_dir}"
        compare_w = c.stringWidth(compare_text, "Helvetica", 11)
        c.drawString(width - margin - compare_w, info_y, compare_text)
        y_top -= info_row_h + gap

        # Column labels
        c.setFont("Helvetica-Bold", 11)
        c.drawString(margin, y_top - 10, BEST_ALIGNMENT)
        c.drawString(margin + side_w + gap, y_top - 10, page.liza_alignment_dir)
        y_top -= label_h + 4

        img_cap = y_top - margin
        common_h = _common_fit_height(
            [page.best_zonogram, page.liza_zonogram],
            side_w,
            img_cap,
        )
        _draw_image_top_aligned(
            c,
            page.best_zonogram,
            margin,
            y_top,
            side_w,
            img_cap,
            target_height=common_h,
        )
        _draw_image_top_aligned(
            c,
            page.liza_zonogram,
            margin + side_w + gap,
            y_top,
            side_w,
            img_cap,
            target_height=common_h,
        )

        if page.warnings:
            c.setFont("Helvetica", 9)
            warn_y = margin + 8
            for warn in page.warnings:
                c.drawString(margin, warn_y, f"Warning: {warn}")
                warn_y += 11

        c.showPage()
        if page_progress is not None:
            page_progress.update(1)

    c.save()


def _require_existing_file(path: Path, arg_name: str) -> Path:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{arg_name} not found: {path}")
    return path


def _require_output_pdf_path(path: Path) -> Path:
    path = Path(path)
    if path.is_dir():
        raise ValueError(
            f"--output-pdf must be a PDF file path, not a directory ({path})."
        )
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"--output-pdf should end with .pdf (got {path}).")
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare active zonograms: best_alignment vs *liza* alignment dirs "
            "from a tomogram CSV."
        )
    )
    parser.add_argument(
        "--tomocsv",
        type=Path,
        default=DEFAULT_TOMOCSV,
        help=f"Tomogram CSV (default: {DEFAULT_TOMOCSV})",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Data root containing <set>/TOP_TOMOS/<tomo>/...",
    )
    parser.add_argument(
        "--output-pdf",
        type=Path,
        default=DEFAULT_OUTPUT_PDF,
        help=f"Output PDF path (default: {DEFAULT_OUTPUT_PDF})",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Process only the first 3 *liza* CSV rows (troubleshooting)",
    )
    args = parser.parse_args(argv)

    args.tomocsv = _require_existing_file(args.tomocsv, "--tomocsv")
    args.output_pdf = _require_output_pdf_path(args.output_pdf)

    liza_rows = load_liza_csv_rows(args.tomocsv)
    if args.test:
        liza_rows = liza_rows[:3]
        tqdm.write(f"Test mode: first {len(liza_rows)} liza CSV row(s)")
    if not liza_rows:
        print(f"No *liza* alignment rows found in {args.tomocsv}")
        return 1

    tqdm.write(f"Found {len(liza_rows)} *liza* CSV row(s)")
    pages, errors = resolve_comparison_pages(liza_rows, args.data_dir)
    if not pages:
        print("No comparison pages resolved; nothing to write.")
        for err in errors:
            print(f"  - {err}")
        return 1

    pdf_bar = tqdm(total=len(pages), desc="Writing PDF", unit="page")
    build_comparison_pdf(pages, args.output_pdf, page_progress=pdf_bar)
    pdf_bar.close()
    tqdm.write(f"PDF written: {args.output_pdf} ({len(pages)} page(s))")

    if errors:
        tqdm.write("\nSkipped / errors:")
        for err in errors:
            tqdm.write(f"  - {err}")
        # Still succeed if we wrote some pages
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
