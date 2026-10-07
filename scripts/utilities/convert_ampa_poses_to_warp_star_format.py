#!/usr/bin/env python3
"""
Split a combined AMPA-poses STAR (e.g. all_ampa_poses_ilp_*.star) into per-tomogram
RELION particle STAR files in Warp-style format (same columns as *predicted_particles.star):

  _rlnCoordinateX/Y/Z, _rlnAngleRot/Tilt/Psi, _rlnMicrographName

Each row's _rlnMicrographName is set to ``<rlnTomoName>.tomostar``. Output files:

  <tomogram_name>_particles.star

Usage:
  python scripts/convert_ampa_poses_to_warp_star_format.py \\
      star_files/all_ampa_poses_ilp_aunp5.0-9.0nm_mem15.0-22.0nm_steric5.0nm.star

  python scripts/convert_ampa_poses_to_warp_star_format.py input.star -o out_dir/
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
import starfile

OUTPUT_COLUMNS = [
    "rlnCoordinateX",
    "rlnCoordinateY",
    "rlnCoordinateZ",
    "rlnAngleRot",
    "rlnAngleTilt",
    "rlnAnglePsi",
    "rlnMicrographName",
]

REQUIRED_INPUT = [
    "rlnTomoName",
    "rlnCoordinateX",
    "rlnCoordinateY",
    "rlnCoordinateZ",
    "rlnAngleRot",
    "rlnAngleTilt",
    "rlnAnglePsi",
]


def _safe_filename_stem(name: str) -> str:
    """Tomogram names for filesystem (Windows-safe subset)."""
    s = str(name).strip()
    s = re.sub(r'[<>:"/\\|?*]', "_", s)
    s = s.replace("\0", "")
    return s or "unknown_tomogram"


def _load_particles_df(path: Path) -> pd.DataFrame:
    data = starfile.read(path)
    if isinstance(data, dict):
        if "particles" not in data:
            raise ValueError(f"No 'particles' block in {path}")
        return data["particles"]
    if isinstance(data, pd.DataFrame):
        return data
    raise ValueError(f"Unexpected STAR content type: {type(data)}")


def convert(input_star: Path, output_dir: Path, *, dry_run: bool = False) -> list[Path]:
    input_star = input_star.resolve()
    if not input_star.is_file():
        raise FileNotFoundError(input_star)

    df = _load_particles_df(input_star)
    missing = [c for c in REQUIRED_INPUT if c not in df.columns]
    if missing:
        raise ValueError(
            f"Input STAR missing required columns: {missing}. "
            f"Found: {list(df.columns)}"
        )

    output_dir = output_dir.resolve()
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for tomo, group in df.groupby("rlnTomoName", sort=False):
        tomo_str = str(tomo).strip()
        micrograph = f"{tomo_str}.tomostar"
        out_df = pd.DataFrame(
            {
                "rlnCoordinateX": group["rlnCoordinateX"].astype(float),
                "rlnCoordinateY": group["rlnCoordinateY"].astype(float),
                "rlnCoordinateZ": group["rlnCoordinateZ"].astype(float),
                "rlnAngleRot": group["rlnAngleRot"].astype(float),
                "rlnAngleTilt": group["rlnAngleTilt"].astype(float),
                "rlnAnglePsi": group["rlnAnglePsi"].astype(float),
                "rlnMicrographName": micrograph,
            }
        )
        out_df = out_df[OUTPUT_COLUMNS]
        stem = _safe_filename_stem(tomo_str)
        out_path = output_dir / f"{stem}_particles.star"
        if not dry_run:
            starfile.write({"particles": out_df}, out_path)
        written.append(out_path)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Convert combined AMPA poses STAR into per-tomogram *_particles.star "
            "(Warp / predicted_particles column layout)."
        )
    )
    parser.add_argument(
        "input_star",
        type=Path,
        help="Combined STAR (e.g. all_ampa_poses_ilp_*.star)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for output STAR files (default: same directory as input)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print tomogram names and paths only; do not write files",
    )
    args = parser.parse_args()

    out_dir = args.output_dir if args.output_dir is not None else args.input_star.parent

    try:
        paths = convert(args.input_star, out_dir, dry_run=args.dry_run)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print(f"Input:  {args.input_star.resolve()}")
    print(f"Output: {out_dir.resolve()}")
    print(f"Tomograms: {len(paths)}")
    for p in paths:
        print(f"  {'(dry-run) ' if args.dry_run else ''}{p.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
