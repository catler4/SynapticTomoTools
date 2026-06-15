#!/usr/bin/env bash
#
# Copy AuNP pick STAR files for tomograms listed in a CSV into the current directory.
#
# Path layout (matches synaptic_tomo_tools.cli.load_tomograms):
#   {BASE_DIR}/{set}/TOP_TOMOS/{tomoname}/{alignment_dir}/aunps/
#
# Usage:
#   ./scripts/copy_aunp_pick_stars_from_csv.sh tomogram_csv_files/my_subset.csv
#   ./scripts/copy_aunp_pick_stars_from_csv.sh --base-dir /other/root my_subset.csv
#
set -euo pipefail

DEFAULT_BASE_DIR="/goliath/processing/Gouaux/CJS/BestTomo/ProcessingCJS/tomograms"

usage() {
    cat <<'EOF'
Usage: copy_aunp_pick_stars_from_csv.sh [OPTIONS] TOMOGRAM_CSV

Copy per-active-zone AuNP pick STAR files for tomograms in TOMOGRAM_CSV into
the current working directory. Only active zones listed in the aunp_active_zones
column are copied (comma-separated indices, e.g. 0 or 0,1).

Copied files are prefixed with {tomoname}_{alignment_dir}_.

Options:
  -b, --base-dir DIR   Tomogram root directory (default: goliath ProcessingCJS path)
  -h, --help           Show this help

STAR patterns copied (per designated active zone, if present):
  aunp_tm_BP_active_zone_<N>_manual_refined.star
  aunp_tm_BP_active_zone_<N>_manual_refined_dimer.star
  aunp_tm_BP_active_zone_<N>_manual_refined_monomer.star
  aunp_tm_BP_active_zone_<N>_manual_refined_each_dimer.star
EOF
}

BASE_DIR="$DEFAULT_BASE_DIR"
CSV_PATH=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -b|--base-dir)
            [[ $# -ge 2 ]] || { echo "Error: --base-dir requires a path" >&2; exit 1; }
            BASE_DIR="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            break
            ;;
        -*)
            echo "Error: unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
        *)
            if [[ -z "$CSV_PATH" ]]; then
                CSV_PATH="$1"
            else
                echo "Error: unexpected extra argument: $1" >&2
                usage >&2
                exit 1
            fi
            shift
            ;;
    esac
done

if [[ -z "$CSV_PATH" ]]; then
    echo "Error: TOMOGRAM_CSV is required" >&2
    usage >&2
    exit 1
fi

if [[ ! -f "$CSV_PATH" ]]; then
    echo "Error: CSV not found: $CSV_PATH" >&2
    exit 1
fi

if [[ ! -d "$BASE_DIR" ]]; then
    echo "Error: base tomogram directory not found: $BASE_DIR" >&2
    exit 1
fi

export COPY_AUNP_CSV_PATH="$CSV_PATH"
export COPY_AUNP_BASE_DIR="$BASE_DIR"

python3 <<'PY'
import csv
import os
import shutil
import sys
from pathlib import Path

csv_path = Path(os.environ["COPY_AUNP_CSV_PATH"])
base_dir = Path(os.environ["COPY_AUNP_BASE_DIR"])
dest_dir = Path.cwd()

patterns = [
    "aunp_tm_BP_active_zone_{az}_manual_refined.star",
    "aunp_tm_BP_active_zone_{az}_manual_refined_dimer.star",
    "aunp_tm_BP_active_zone_{az}_manual_refined_monomer.star",
    "aunp_tm_BP_active_zone_{az}_manual_refined_each_dimer.star",
]

required_cols = {"tomoname", "set", "alignment_dir"}
copied = 0
missing = 0
skipped_rows = 0

def parse_az_indices(value) -> list[int]:
  if value is None:
    return []
  s = str(value).strip()
  if not s or s.lower() == "nan":
    return []
  indices: list[int] = []
  for part in s.split(","):
    part = part.strip()
    if not part:
      continue
    try:
      indices.append(int(float(part)))
    except ValueError:
      print(f"Warning: skipping invalid active zone index '{part}'", file=sys.stderr)
  return indices

with csv_path.open(newline="") as f:
    reader = csv.DictReader(f)
    if reader.fieldnames is None:
        print(f"Error: empty CSV: {csv_path}", file=sys.stderr)
        sys.exit(1)
    missing_cols = required_cols - set(reader.fieldnames)
    if missing_cols:
        print(
            f"Error: CSV missing required columns: {', '.join(sorted(missing_cols))}",
            file=sys.stderr,
        )
        sys.exit(1)
    if "aunp_active_zones" not in reader.fieldnames:
        print("Error: CSV missing column 'aunp_active_zones'", file=sys.stderr)
        sys.exit(1)

    for row in reader:
        tomoname = str(row["tomoname"]).strip()
        set_name = str(row["set"]).strip()
        alignment_dir = str(row["alignment_dir"]).strip()
        if not tomoname or not set_name or not alignment_dir or alignment_dir.lower() == "nan":
            print(f"Warning: skipping row with missing tomoname/set/alignment_dir: {row}", file=sys.stderr)
            skipped_rows += 1
            continue

        az_indices = parse_az_indices(row.get("aunp_active_zones"))
        if not az_indices:
            print(
                f"Warning: no aunp_active_zones for {tomoname} ({alignment_dir}); skipping",
                file=sys.stderr,
            )
            skipped_rows += 1
            continue

        aunps_dir = base_dir / set_name / "TOP_TOMOS" / tomoname / alignment_dir / "aunps"
        if not aunps_dir.is_dir():
            print(f"Warning: aunps directory not found: {aunps_dir}", file=sys.stderr)
            skipped_rows += 1
            continue

        prefix = f"{tomoname}_{alignment_dir}_"
        for az in az_indices:
            for pattern in patterns:
                fname = pattern.format(az=az)
                src = aunps_dir / fname
                if not src.is_file():
                    missing += 1
                    continue
                dest = dest_dir / f"{prefix}{fname}"
                shutil.copy2(src, dest)
                print(f"Copied: {dest.name}")
                copied += 1

print(
    f"\nDone. Copied {copied} file(s); "
    f"{missing} expected file(s) not found; "
    f"{skipped_rows} row(s) skipped."
)
PY
