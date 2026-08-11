#!/usr/bin/env python3
"""
Create ``tomograms/`` and ``segmentations/`` directories with per-tomogram .mrc
symlinks from an STT tomograms.csv list.

How to run
----------
From the SynapticTomoTools repo (or anywhere), with the CSV and data root set:

  python scripts/symlink_tomograms_and_segmentations.py \\
    --csv tomogram_csv_files/tomograms_15F1-H12Cys_quis_FINAL.csv \\
    --data-dir data \\
    --output-dir /path/to/project

Optional:
  --force     overwrite existing links with the same name

What it creates under ``--output-dir`` (default: current directory)
-------------------------------------------------------------------
  tomograms/<tomoname>.mrc
    -> <data>/<set>/TOP_TOMOS/<tomoname>/<alignment_dir>/<tomoname>_full_rec_BP_3DCTF_BIN4.mrc
       (falls back to *_BIN4_ddw.mrc if the non-ddw file is missing)

  segmentations/<tomoname>.mrc
    -> <data>/<set>/TOP_TOMOS/<tomoname>/<alignment_dir>/STT_results/membranes_labeled/<tomoname>_membranes_relabeled.mrc

CSV must include columns: tomoname, set, alignment_dir.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def require_alignment_dir(alignment_dir, *, context: str = "") -> str:
    if alignment_dir is None:
        msg = "alignment_dir is required and cannot be None."
        if context:
            msg = f"{msg} ({context})"
        raise ValueError(msg)
    s = str(alignment_dir).strip()
    if not s or s.lower() in ("nan", "none"):
        raise ValueError("alignment_dir must be a non-empty string.")
    return s


def tomogram_root(data_dir: Path, set_name: str, tomoname: str) -> Path:
    return Path(data_dir) / set_name / "TOP_TOMOS" / tomoname


def find_bp_tomogram_mrc(alignment_dir: Path, tomoname: str) -> Path | None:
    """Prefer exact BIN4 BP MRC; fall back to common _ddw variant if needed."""
    exact = alignment_dir / f"{tomoname}_full_rec_BP_3DCTF_BIN4.mrc"
    if exact.is_file():
        return exact
    ddw = alignment_dir / f"{tomoname}_full_rec_BP_3DCTF_BIN4_ddw.mrc"
    if ddw.is_file():
        return ddw
    return None


def find_labeled_segmentation_mrc(alignment_dir: Path, tomoname: str) -> Path | None:
    preferred = (
        alignment_dir
        / "STT_results"
        / "membranes_labeled"
        / f"{tomoname}_membranes_relabeled.mrc"
    )
    if preferred.is_file():
        return preferred
    legacy = alignment_dir / "membranes_labeled" / f"{tomoname}_membranes_relabeled.mrc"
    if legacy.is_file():
        return legacy
    return None


def make_symlink(link_path: Path, target: Path, *, force: bool) -> None:
    link_path = Path(link_path)
    target = Path(target).resolve()
    if link_path.is_symlink() or link_path.exists():
        if not force:
            raise FileExistsError(f"Refusing to overwrite existing path: {link_path}")
        link_path.unlink()
    link_path.symlink_to(target)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Create tomograms/ and segmentations/ with per-tomogram .mrc symlinks "
            "from an STT tomograms.csv."
        )
    )
    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="STT tomograms.csv (columns: tomoname, set, alignment_dir)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Data root containing <set>/TOP_TOMOS/<tomoname>/...",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to create tomograms/ and segmentations/ (default: current directory)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing symlinks/files with the same name",
    )
    args = parser.parse_args(argv)

    csv_path = Path(args.csv)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) if args.output_dir else Path.cwd()
    if not csv_path.is_file():
        print(f"CSV not found: {csv_path}", file=sys.stderr)
        return 1
    if not data_dir.is_dir():
        print(f"Data directory not found: {data_dir}", file=sys.stderr)
        return 1

    df = pd.read_csv(csv_path)
    required = {"tomoname", "set", "alignment_dir"}
    missing = required - set(df.columns)
    if missing:
        print(
            f"CSV missing required columns: {sorted(missing)}",
            file=sys.stderr,
        )
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    out_tomo = output_dir / "tomograms"
    out_seg = output_dir / "segmentations"
    out_tomo.mkdir(parents=True, exist_ok=True)
    out_seg.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir.resolve()}")

    ok_tomo = 0
    ok_seg = 0
    failed = 0

    for _, row in df.iterrows():
        tomoname = str(row["tomoname"]).strip()
        set_name = str(row["set"]).strip()
        try:
            alignment_dir_name = require_alignment_dir(
                row["alignment_dir"], context=f"tomogram {tomoname}"
            )
        except ValueError as exc:
            print(f"SKIP {tomoname}: {exc}")
            failed += 1
            continue
        if not tomoname or not set_name or tomoname.lower() == "nan":
            print(f"SKIP row with missing tomoname/set: {row.to_dict()}")
            failed += 1
            continue

        align_path = tomogram_root(data_dir, set_name, tomoname) / alignment_dir_name
        link_name = f"{tomoname}.mrc"

        bp = find_bp_tomogram_mrc(align_path, tomoname)
        if bp is None:
            print(
                f"MISSING tomogram MRC for {tomoname} under {align_path} "
                f"(expected {tomoname}_full_rec_BP_3DCTF_BIN4.mrc)"
            )
            failed += 1
        else:
            try:
                make_symlink(out_tomo / link_name, bp, force=args.force)
                print(f"tomograms/{link_name} -> {bp}")
                ok_tomo += 1
            except OSError as exc:
                print(f"FAILED tomograms/{link_name}: {exc}")
                failed += 1

        seg = find_labeled_segmentation_mrc(align_path, tomoname)
        if seg is None:
            print(
                f"MISSING labeled segmentation for {tomoname} under {align_path} "
                f"(expected STT_results/membranes_labeled/{tomoname}_membranes_relabeled.mrc)"
            )
            failed += 1
        else:
            try:
                make_symlink(out_seg / link_name, seg, force=args.force)
                print(f"segmentations/{link_name} -> {seg}")
                ok_seg += 1
            except OSError as exc:
                print(f"FAILED segmentations/{link_name}: {exc}")
                failed += 1

    print(
        f"\nDone. tomogram links: {ok_tomo}, segmentation links: {ok_seg}, "
        f"missing/failed: {failed}"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
