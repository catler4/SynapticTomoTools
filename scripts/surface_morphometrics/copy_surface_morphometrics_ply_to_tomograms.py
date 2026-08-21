#!/usr/bin/env python3
"""
Copy Surface Morphometrics cleft PLY meshes into each tomogram's alignment dir.

For each row in an STT tomograms.csv, looks in ``--source-dir`` for:

  <tomoname>_cleft_pre.ply
  <tomoname>_cleft_post.ply

and copies them to:

  <data>/<set>/TOP_TOMOS/<tomoname>/<alignment_dir>/surface_morphometrics/

How to run
----------
  python scripts/surface_morphometrics/copy_surface_morphometrics_ply_to_tomograms.py \\
    --csv tomogram_csv_files/tomograms_15F1-H12Cys_FINAL.csv \\
    --data-dir data \\
    --source-dir /path/to/ply_folder

Optional:
  --dry-run   print actions without copying
  --force     overwrite existing destination files
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import pandas as pd

PLY_SUFFIXES = ("_cleft_pre.ply", "_cleft_post.ply")


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


def default_data_root() -> Path:
    return Path(os.environ.get("TOMO_ROOT_BASE") or "data")


def load_csv_jobs(
    csv_path: Path,
    *,
    data_dir: Path,
    set_name: str | None = None,
) -> list[tuple[str, str, Path]]:
    """Return list of (tomoname, alignment_dir, dest_surface_morphometrics_dir)."""
    df = pd.read_csv(csv_path)
    required = {"tomoname", "set", "alignment_dir"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")

    if set_name:
        df = df[df["set"].astype(str) == str(set_name)]

    jobs: list[tuple[str, str, Path]] = []
    for _, row in df.iterrows():
        tomoname = str(row["tomoname"]).strip()
        row_set = str(row["set"]).strip()
        try:
            alignment_dir = require_alignment_dir(
                row["alignment_dir"], context=f"tomogram {tomoname}"
            )
        except ValueError as exc:
            print(f"Skipping {tomoname}: {exc}")
            continue
        if not tomoname or not row_set or tomoname.lower() == "nan":
            print(f"Skipping row with missing tomoname/set: {row.to_dict()}")
            continue
        dest_dir = (
            Path(data_dir)
            / row_set
            / "TOP_TOMOS"
            / tomoname
            / alignment_dir
            / "surface_morphometrics"
        )
        jobs.append((tomoname, alignment_dir, dest_dir))
    return jobs


def find_source_ply(source_dir: Path, tomoname: str, suffix: str) -> Path | None:
    """
    Find ``<tomoname><suffix>`` under source_dir.

    Also accepts a one-level nested match if the exact basename is unique among
    recursive hits (e.g. source_dir/some_subdir/<tomoname>_cleft_pre.ply).
    """
    exact = source_dir / f"{tomoname}{suffix}"
    if exact.is_file():
        return exact

    matches = sorted(source_dir.rglob(f"{tomoname}{suffix}"))
    files = [p for p in matches if p.is_file()]
    if len(files) == 1:
        return files[0]
    if len(files) > 1:
        print(
            f"  WARNING: multiple matches for {tomoname}{suffix}; "
            f"using first: {files[0]}"
        )
        for extra in files[1:]:
            print(f"           also found: {extra}")
        return files[0]
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Copy <tomoname>_cleft_pre/post.ply files from a source folder into "
            "each tomogram's <alignment_dir>/surface_morphometrics/."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--csv", type=Path, required=True, help="STT tomograms.csv")
    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="Directory containing (or nesting) the PLY files to copy",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Data root (default: $TOMO_ROOT_BASE or 'data')",
    )
    parser.add_argument(
        "--set",
        dest="set_name",
        default=None,
        help="Optional set filter",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be copied without writing files",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing destination PLY files",
    )
    args = parser.parse_args(argv)

    csv_path = Path(args.csv)
    source_dir = Path(args.source_dir)
    data_dir = Path(args.data_dir) if args.data_dir else default_data_root()

    if not csv_path.is_file():
        print(f"CSV not found: {csv_path}", file=sys.stderr)
        return 1
    if not source_dir.is_dir():
        print(f"Source directory not found: {source_dir}", file=sys.stderr)
        return 1
    if not data_dir.is_dir():
        print(f"Data directory not found: {data_dir}", file=sys.stderr)
        return 1

    jobs = load_csv_jobs(csv_path, data_dir=data_dir, set_name=args.set_name)
    if not jobs:
        print(f"No tomogram rows to process from {csv_path}", file=sys.stderr)
        return 1

    print(f"CSV:        {csv_path}")
    print(f"Source:     {source_dir.resolve()}")
    print(f"Data root:  {data_dir.resolve()}")
    print(f"Tomograms:  {len(jobs)}")
    if args.dry_run:
        print("Dry run: no files will be copied")

    copied = 0
    skipped_exists = 0
    missing = 0
    failed = 0

    for tomoname, alignment_dir, dest_dir in jobs:
        print(f"\n{tomoname}__{alignment_dir}")
        if not args.dry_run:
            dest_dir.mkdir(parents=True, exist_ok=True)

        for suffix in PLY_SUFFIXES:
            src = find_source_ply(source_dir, tomoname, suffix)
            dest = dest_dir / f"{tomoname}{suffix}"
            if src is None:
                print(f"  MISSING {tomoname}{suffix} under {source_dir}")
                missing += 1
                continue
            if dest.exists() and not args.force:
                print(f"  SKIP exists (use --force): {dest}")
                skipped_exists += 1
                continue
            print(f"  {'WOULD COPY' if args.dry_run else 'COPY'} {src} -> {dest}")
            if args.dry_run:
                copied += 1
                continue
            try:
                shutil.copy2(src, dest)
                copied += 1
            except OSError as exc:
                print(f"  FAILED {dest}: {exc}")
                failed += 1

    print(
        f"\nDone. copied/would-copy={copied}, missing={missing}, "
        f"skipped-exists={skipped_exists}, failed={failed}"
    )
    return 1 if (missing or failed) else 0


if __name__ == "__main__":
    raise SystemExit(main())
