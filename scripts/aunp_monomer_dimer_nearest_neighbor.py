#!/usr/bin/env python3
"""
Nearest-neighbor distances for monomer + individual dimer AuNP picks.

Loads monomer picks (*_manual_refined_monomer.star) and per-dimer picks
(*_manual_refined_each_dimer.star) for a tomogram / alignment / active zone,
pools them, and computes each pick's nearest neighbor among the combined set.

Expects prefixed STAR files as produced by copy_aunp_pick_stars_from_csv.sh, e.g.:
  {tomoname}_{alignment_dir}_aunp_tm_BP_active_zone_{az}_manual_refined_monomer.star
  {tomoname}_{alignment_dir}_aunp_tm_BP_active_zone_{az}_manual_refined_each_dimer.star
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import starfile
from scipy.spatial import KDTree

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

DEFAULT_PICKS_DIR = _REPO_ROOT / "data" / "all_aunp_picks"
DEFAULT_OUTPUT_DIR = _REPO_ROOT / "results" / "aunp_monomer_dimer_nn"

COORD_COLS = ("faCoordinateX", "faCoordinateY", "faCoordinateZ")


def parse_az_indices(value) -> List[int]:
    if value is None:
        return []
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return []
    indices: List[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            indices.append(int(float(part)))
        except ValueError as exc:
            raise ValueError(f"Invalid active zone index: {part!r}") from exc
    return indices


def pick_star_paths(
    picks_dir: Path,
    tomoname: str,
    alignment_dir: str,
    active_zone: int,
) -> Tuple[Path, Path]:
    prefix = f"{tomoname}_{alignment_dir}_"
    monomer = (
        picks_dir
        / f"{prefix}aunp_tm_BP_active_zone_{active_zone}_manual_refined_monomer.star"
    )
    each_dimer = (
        picks_dir
        / f"{prefix}aunp_tm_BP_active_zone_{active_zone}_manual_refined_each_dimer.star"
    )
    return monomer, each_dimer


def read_star_coordinates(star_path: Path) -> pd.DataFrame:
    if not star_path.is_file():
        raise FileNotFoundError(star_path)

    star_data = starfile.read(star_path)
    if isinstance(star_data, dict):
        df = None
        for value in star_data.values():
            if isinstance(value, pd.DataFrame):
                df = value
                break
        if df is None:
            raise ValueError(f"No DataFrame found in STAR file: {star_path}")
    elif isinstance(star_data, pd.DataFrame):
        df = star_data
    else:
        raise ValueError(f"Unexpected STAR content in {star_path}: {type(star_data)}")

    missing = [c for c in COORD_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{star_path} missing coordinate columns: {missing}")

    out = df[list(COORD_COLS)].copy()
    out = out.apply(pd.to_numeric, errors="coerce")
    out = out.dropna(how="any")
    return out.reset_index(drop=True)


def load_monomer_dimer_picks(
    picks_dir: Path,
    tomoname: str,
    alignment_dir: str,
    active_zone: int,
) -> pd.DataFrame:
    monomer_path, each_dimer_path = pick_star_paths(
        picks_dir, tomoname, alignment_dir, active_zone
    )

    frames: List[pd.DataFrame] = []
    for classification, path in (
        ("monomer", monomer_path),
        ("dimer", each_dimer_path),
    ):
        if not path.is_file():
            print(f"  Warning: missing {classification} picks: {path.name}")
            continue
        coords = read_star_coordinates(path)
        if coords.empty:
            print(f"  Warning: no coordinates in {path.name}")
            continue
        coords = coords.copy()
        coords["classification"] = classification
        coords["source_star"] = path.name
        coords["source_pick_index"] = np.arange(len(coords), dtype=int)
        frames.append(coords)

    if not frames:
        raise FileNotFoundError(
            f"No monomer or each_dimer STAR files found for "
            f"{tomoname} ({alignment_dir}, active zone {active_zone}) in {picks_dir}"
        )

    combined = pd.concat(frames, ignore_index=True)
    combined["tomogram_name"] = tomoname
    combined["alignment_dir"] = alignment_dir
    combined["active_zone"] = int(active_zone)
    combined["aunp_index"] = np.arange(len(combined), dtype=int)
    return combined


def add_nearest_neighbor_columns(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) == 0:
        return df.copy()

    coords = df[list(COORD_COLS)].to_numpy(dtype=float)
    if len(coords) == 1:
        out = df.copy()
        out["nearest_neighbor_distance_nm"] = np.nan
        out["nearest_neighbor_aunp_index"] = np.nan
        out["nearest_neighbor_classification"] = pd.NA
        return out

    tree = KDTree(coords)
    dists, idxs = tree.query(coords, k=2)
    out = df.copy()
    out["nearest_neighbor_distance_nm"] = dists[:, 1]
    out["nearest_neighbor_aunp_index"] = idxs[:, 1].astype(int)
    out["nearest_neighbor_classification"] = out.loc[
        idxs[:, 1], "classification"
    ].to_numpy()
    return out


def output_basename(tomoname: str, alignment_dir: str, active_zone: int) -> str:
    return f"{tomoname}__{alignment_dir}__az{active_zone}"


def write_results_table(df: pd.DataFrame, output_csv: Path) -> None:
    cols = [
        "tomogram_name",
        "alignment_dir",
        "active_zone",
        "aunp_index",
        "classification",
        "source_star",
        "source_pick_index",
        "faCoordinateX",
        "faCoordinateY",
        "faCoordinateZ",
        "nearest_neighbor_distance_nm",
        "nearest_neighbor_aunp_index",
        "nearest_neighbor_classification",
    ]
    out = df[cols].copy()
    out = out.rename(
        columns={
            "faCoordinateX": "faCoordinateX_nm",
            "faCoordinateY": "faCoordinateY_nm",
            "faCoordinateZ": "faCoordinateZ_nm",
        }
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    print(f"  Wrote table: {output_csv}")


def write_histogram(
    df: pd.DataFrame,
    output_png: Path,
    title: str,
    *,
    xlim: Optional[Tuple[float, float]] = None,
    n_bins: int = 30,
) -> None:
    import matplotlib.pyplot as plt

    output_png.parent.mkdir(parents=True, exist_ok=True)
    distances = df["nearest_neighbor_distance_nm"].dropna().to_numpy()
    if len(distances) == 0:
        print(f"  Skipping histogram (no finite nearest-neighbor distances): {output_png.name}")
        return

    if xlim is not None:
        bins = np.linspace(xlim[0], xlim[1], n_bins + 1)
        hist_range = xlim
    else:
        bins = np.histogram_bin_edges(distances, bins="auto")
        hist_range = None

    fig, ax = plt.subplots(figsize=(7, 5))
    for classification, color, label in (
        ("monomer", "#9467BD", "Monomer"),
        ("dimer", "#4C72B0", "Dimer (each_dimer)"),
    ):
        subset = df.loc[
            df["classification"] == classification, "nearest_neighbor_distance_nm"
        ].dropna()
        if len(subset) == 0:
            continue
        ax.hist(
            subset,
            bins=bins,
            range=hist_range,
            alpha=0.55,
            color=color,
            label=f"{label} (n={len(subset)})",
            edgecolor="white",
            linewidth=0.5,
        )

    ax.hist(
        distances,
        bins=bins,
        range=hist_range,
        histtype="step",
        color="black",
        linewidth=1.5,
        label=f"All picks (n={len(distances)})",
    )
    ax.set_xlabel("Nearest neighbor distance (nm)")
    ax.set_ylabel("AuNP count")
    ax.set_title(title)
    if xlim is not None:
        ax.set_xlim(xlim)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_png, dpi=150)
    plt.close(fig)
    print(f"  Wrote histogram: {output_png}")


def write_combined_outputs(
    frames: List[pd.DataFrame],
    output_dir: Path,
    label: str,
    write_plot: bool,
) -> None:
    if not frames:
        return

    combined = pd.concat(frames, ignore_index=True)
    stem = f"{label}_combined" if label else "combined"
    combined_csv = output_dir / f"{stem}_nn_distances.csv"
    write_results_table(combined, combined_csv)
    for classification in ("monomer", "dimer"):
        subset = combined[combined["classification"] == classification]
        if subset.empty:
            print(f"  Warning: no {classification} picks in combined table; skipping split CSV")
            continue
        write_results_table(subset, output_dir / f"{stem}_nn_distances_{classification}.csv")
    if write_plot:
        n_tomos = combined[["tomogram_name", "alignment_dir", "active_zone"]].drop_duplicates().shape[0]
        title = f"All tomograms (n={n_tomos})\nNearest neighbor distances"
        write_histogram(combined, output_dir / f"{stem}_nn_histogram.png", title)
        write_histogram(
            combined,
            output_dir / f"{stem}_nn_histogram_0_15nm.png",
            title + " (0–15 nm)",
            xlim=(0.0, 15.0),
        )


def analyze_one(
    picks_dir: Path,
    output_dir: Path,
    tomoname: str,
    alignment_dir: str,
    active_zone: int,
    write_plot: bool,
) -> Optional[pd.DataFrame]:
    print(f"Processing {tomoname} ({alignment_dir}, active zone {active_zone})")
    df = load_monomer_dimer_picks(picks_dir, tomoname, alignment_dir, active_zone)
    df = add_nearest_neighbor_columns(df)

    base = output_basename(tomoname, alignment_dir, active_zone)
    write_results_table(df, output_dir / f"{base}_nn_distances.csv")
    if write_plot:
        title = f"{tomoname}\n{alignment_dir}, active zone {active_zone}"
        write_histogram(df, output_dir / f"{base}_nn_histogram.png", title)
    return df


def iter_csv_jobs(csv_path: Path) -> Iterable[Tuple[str, str, int]]:
    df = pd.read_csv(csv_path)
    required = {"tomoname", "alignment_dir", "aunp_active_zones"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")

    for _, row in df.iterrows():
        tomoname = str(row["tomoname"]).strip()
        alignment_dir = str(row["alignment_dir"]).strip()
        if not tomoname or not alignment_dir or alignment_dir.lower() == "nan":
            print(f"Skipping row with missing tomoname/alignment_dir: {row.to_dict()}")
            continue
        try:
            az_indices = parse_az_indices(row["aunp_active_zones"])
        except ValueError as exc:
            print(f"Skipping {tomoname}: {exc}")
            continue
        if not az_indices:
            print(f"Skipping {tomoname}: no aunp_active_zones")
            continue
        for az in az_indices:
            yield tomoname, alignment_dir, az


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute nearest-neighbor distances for monomer + each_dimer AuNP picks "
            "from data/all_aunp_picks."
        )
    )
    parser.add_argument(
        "--picks-dir",
        type=Path,
        default=DEFAULT_PICKS_DIR,
        help=f"Directory with prefixed STAR files (default: {DEFAULT_PICKS_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for CSV tables and histograms (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help="Tomogram CSV (uses tomoname, alignment_dir, aunp_active_zones per row)",
    )
    parser.add_argument("--tomogram", help="Single tomogram name")
    parser.add_argument("--alignment-dir", help="Alignment directory for single-tomogram mode")
    parser.add_argument(
        "--active-zone",
        type=int,
        action="append",
        dest="active_zones",
        help="Active zone index (repeat for multiple zones in single-tomogram mode)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip histogram PNG output",
    )
    parser.add_argument(
        "--combine-only",
        action="store_true",
        help="Only merge existing per-tomogram *_nn_distances.csv files in --output-dir",
    )
    return parser


def load_existing_per_tomogram_tables(output_dir: Path) -> List[pd.DataFrame]:
    frames: List[pd.DataFrame] = []
    for path in sorted(output_dir.glob("*_nn_distances.csv")):
        if "_combined_" in path.name:
            continue
        df = pd.read_csv(path)
        rename_back = {
            "faCoordinateX_nm": "faCoordinateX",
            "faCoordinateY_nm": "faCoordinateY",
            "faCoordinateZ_nm": "faCoordinateZ",
        }
        df = df.rename(columns={k: v for k, v in rename_back.items() if k in df.columns})
        frames.append(df)
    return frames


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    picks_dir = args.picks_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    write_plot = not args.no_plot

    if not picks_dir.is_dir():
        print(f"Error: picks directory not found: {picks_dir}", file=sys.stderr)
        return 1

    jobs: List[Tuple[str, str, int]] = []
    if args.csv:
        jobs.extend(iter_csv_jobs(args.csv.expanduser().resolve()))
    elif args.tomogram and args.alignment_dir and args.active_zones:
        for az in args.active_zones:
            jobs.append((args.tomogram, args.alignment_dir, int(az)))
    else:
        parser.error("Provide --csv or (--tomogram, --alignment-dir, and --active-zone)")

    if not jobs:
        print("No tomogram/active-zone jobs to process.", file=sys.stderr)
        return 1

    if args.combine_only:
        all_results = load_existing_per_tomogram_tables(output_dir)
        if not all_results:
            print(f"Error: no per-tomogram *_nn_distances.csv files in {output_dir}", file=sys.stderr)
            return 1
        label = args.csv.stem if args.csv else "combined"
        print(f"Combining {len(all_results)} existing tables...")
        write_combined_outputs(all_results, output_dir, label, write_plot=write_plot)
        print("Done.")
        return 0

    ok = 0
    failed = 0
    all_results: List[pd.DataFrame] = []
    for tomoname, alignment_dir, active_zone in jobs:
        try:
            df = analyze_one(
                picks_dir,
                output_dir,
                tomoname,
                alignment_dir,
                active_zone,
                write_plot=write_plot,
            )
            if df is not None:
                all_results.append(df)
            ok += 1
        except Exception as exc:
            failed += 1
            print(f"  Error: {exc}", file=sys.stderr)

    if len(all_results) > 1 or (len(all_results) == 1 and args.csv):
        label = args.csv.stem if args.csv else "combined"
        print(f"\nWriting combined outputs ({len(all_results)} tomogram/active-zone jobs)...")
        write_combined_outputs(all_results, output_dir, label, write_plot=write_plot)

    print(f"\nFinished: {ok} succeeded, {failed} failed.")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
