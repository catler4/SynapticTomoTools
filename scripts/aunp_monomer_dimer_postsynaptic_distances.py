#!/usr/bin/env python3
"""
Postsynaptic active-zone center distances for monomer and individual dimer AuNP picks.

For each tomogram / alignment / active zone in a CSV:
  - Load picks from standard tomogram layout:
      {TOMO_ROOT}/{set}/TOP_TOMOS/{tomoname}/{alignment_dir}/aunps/
        aunp_tm_BP_active_zone_{N}_manual_refined_monomer.star
        aunp_tm_BP_active_zone_{N}_manual_refined_each_dimer.star
  - Load active-zone inner/outer postsynaptic membranes from prior activezone analysis:
      {tomoname}/{alignment_dir}/STT_results/activezone/
  - Compute distance to postsynaptic active-zone center as mean(outer, inner) KD-tree
    distances (same strategy as analyze_aunps in synaptic_tomo_tools).
  - Write a curated dimer_closest table: one pick per dimer pair (closest to postsynaptic).

Requires active_zone_mapping.json from a prior activezone run when multiple zones exist.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import KDTree

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from synaptic_tomo_tools.activezone import (  # noqa: E402
    import_active_zone_segmentations,
    load_active_zone_mapping,
)
from synaptic_tomo_tools.aunps import (  # noqa: E402
    _read_aunp_pick_star_dataframe,
    aunp_pick_star_filename,
)

DEFAULT_OUTPUT_DIR = _REPO_ROOT / "results" / "aunp_monomer_dimer_postsynaptic_distances"
DEFAULT_TOMO_ROOT_BASE = os.environ.get(
    "TOMO_ROOT_BASE",
    "/goliath/processing/Gouaux/CJS/BestTomo/ProcessingCJS/tomograms",
)

MONOMER_PATTERN = "aunp_tm_BP_active_zone_*_manual_refined_monomer.star"
EACH_DIMER_PATTERN = "aunp_tm_BP_active_zone_*_manual_refined_each_dimer.star"
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


def tomogram_path(tomo_root_base: Path, set_name: str, tomoname: str) -> Path:
    return tomo_root_base / set_name / "TOP_TOMOS" / tomoname


def resolve_active_zone_name(
    tomogram_dir: Path,
    alignment_dir: str,
    active_zone_idx: int,
    az_segmentations: dict,
) -> str:
    mapping_raw = load_active_zone_mapping(tomogram_dir, alignment_dir)
    if mapping_raw:
        mapping = {int(k): v for k, v in mapping_raw.items()}
        if active_zone_idx in mapping:
            zone_name = mapping[active_zone_idx]
            if zone_name in az_segmentations:
                return zone_name
            raise KeyError(
                f"Mapped zone '{zone_name}' for active zone {active_zone_idx} "
                f"not found in STT_results/activezone segmentations"
            )

    if len(az_segmentations) == 1:
        return next(iter(az_segmentations))

    raise ValueError(
        f"No active_zone_mapping.json for {tomogram_dir.name} ({alignment_dir}) "
        f"and {len(az_segmentations)} zones present; cannot map active zone index "
        f"{active_zone_idx}. Run activezone analysis first."
    )


def nearest_distances_to_cloud(points: np.ndarray, cloud: np.ndarray) -> np.ndarray:
    if cloud is None or len(cloud) == 0:
        return np.full(points.shape[0], np.nan)
    tree = KDTree(np.atleast_2d(cloud).astype(float))
    dists, _ = tree.query(np.atleast_2d(points).astype(float))
    return np.asarray(dists, dtype=float)


def postsynaptic_center_distances(
    coords: np.ndarray,
    post_outer: np.ndarray,
    post_inner: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    dist_outer = nearest_distances_to_cloud(coords, post_outer)
    dist_inner = nearest_distances_to_cloud(coords, post_inner)
    dist_mean = np.nanmean(np.vstack([dist_outer, dist_inner]), axis=0)
    return dist_outer, dist_inner, dist_mean


def load_pick_coordinates(
    aunps_dir: Path,
    active_zone: int,
    classification: str,
    pattern: str,
) -> Tuple[pd.DataFrame, Path]:
    star_path = aunps_dir / aunp_pick_star_filename(active_zone, pattern)
    if not star_path.is_file():
        raise FileNotFoundError(star_path)
    df = _read_aunp_pick_star_dataframe(star_path)
    if df is None or df.empty:
        raise ValueError(f"No coordinates in {star_path}")
    missing = [c for c in COORD_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{star_path} missing columns: {missing}")
    out = df[list(COORD_COLS)].apply(pd.to_numeric, errors="coerce").dropna(how="any")
    if out.empty:
        raise ValueError(f"No valid coordinates in {star_path}")
    out = out.reset_index(drop=True)
    out["classification"] = classification
    out["source_star"] = star_path.name
    out["source_pick_index"] = np.arange(len(out), dtype=int)
    return out, star_path


def load_monomer_dimer_picks(aunps_dir: Path, active_zone: int) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for classification, pattern in (
        ("monomer", MONOMER_PATTERN),
        ("dimer", EACH_DIMER_PATTERN),
    ):
        star_path = aunps_dir / aunp_pick_star_filename(active_zone, pattern)
        if not star_path.is_file():
            print(f"  Warning: missing {classification} picks: {star_path}")
            continue
        try:
            df, _ = load_pick_coordinates(aunps_dir, active_zone, classification, pattern)
            frames.append(df)
        except ValueError as exc:
            print(f"  Warning: {exc}")
    if not frames:
        raise FileNotFoundError(
            f"No monomer or each_dimer STAR files with coordinates under {aunps_dir} "
            f"for active zone {active_zone}"
        )
    return pd.concat(frames, ignore_index=True)


def analyze_one(
    tomo_root_base: Path,
    output_dir: Path,
    tomoname: str,
    set_name: str,
    alignment_dir: str,
    active_zone: int,
) -> pd.DataFrame:
    tomogram_dir = tomogram_path(tomo_root_base, set_name, tomoname)
    aunps_dir = tomogram_dir / alignment_dir / "aunps"
    print(f"Processing {tomoname} ({set_name}, {alignment_dir}, active zone {active_zone})")

    az_segmentations = import_active_zone_segmentations(tomogram_dir, alignment_dir=alignment_dir)
    if not az_segmentations:
        raise FileNotFoundError(
            f"No active zone segmentations in {tomogram_dir / alignment_dir / 'STT_results' / 'activezone'}"
        )

    zone_name = resolve_active_zone_name(
        tomogram_dir, alignment_dir, active_zone, az_segmentations
    )
    zone_data = az_segmentations[zone_name]
    post_outer = np.atleast_2d(np.asarray(zone_data.get("postsynaptic_outer_coords", [])))
    post_inner = np.atleast_2d(np.asarray(zone_data.get("postsynaptic_inner_coords", [])))
    if post_outer.size == 0 and post_inner.size == 0:
        raise ValueError(f"No postsynaptic inner/outer points for zone '{zone_name}'")

    df = load_monomer_dimer_picks(aunps_dir, active_zone)
    coords = df[list(COORD_COLS)].to_numpy(dtype=float)
    dist_outer, dist_inner, dist_mean = postsynaptic_center_distances(
        coords, post_outer, post_inner
    )

    df["tomogram_name"] = tomoname
    df["set_name"] = set_name
    df["alignment_dir"] = alignment_dir
    df["active_zone"] = int(active_zone)
    df["active_zone_name"] = zone_name
    df["distance_to_postsynaptic_active_outer_nm"] = dist_outer
    df["distance_to_postsynaptic_active_inner_nm"] = dist_inner
    df["distance_to_postsynaptic_active_outer_inner_mean_nm"] = dist_mean
    df["aunp_index"] = np.arange(len(df), dtype=int)

    base = f"{tomoname}__{alignment_dir}__az{active_zone}"
    write_results_table(df, output_dir / f"{base}_postsynaptic_distances.csv")
    dimer_closest = select_closest_dimer_picks_per_pair(df)
    if not dimer_closest.empty:
        write_results_table(
            dimer_closest,
            output_dir / f"{base}_postsynaptic_distances_dimer_closest.csv",
        )
    return df


def select_closest_dimer_picks_per_pair(df: pd.DataFrame) -> pd.DataFrame:
    """
    From each_dimer picks, keep one AuNP per dimer pair: the closest to postsynaptic
    active-zone center (lowest outer/inner mean distance).

    Pairs are consecutive rows in each_dimer.star (source_pick_index // 2), matching
    the two individual AuNPs per dimer.
    """
    dimers = df[df["classification"] == "dimer"].copy()
    if dimers.empty:
        return dimers

    dimers["dimer_pair_index"] = dimers["source_pick_index"] // 2
    group_cols = ["tomogram_name", "alignment_dir", "active_zone", "dimer_pair_index"]
    pair_sizes = dimers.groupby(group_cols, dropna=False).size()
    odd_pairs = pair_sizes[pair_sizes != 2]
    if not odd_pairs.empty:
        print(
            f"  Warning: {len(odd_pairs)} dimer pair(s) do not have exactly 2 picks; "
            "still selecting closest pick per group"
        )

    dist_col = "distance_to_postsynaptic_active_outer_inner_mean_nm"
    pick_idx = dimers.groupby(group_cols, dropna=False)[dist_col].idxmin()
    return dimers.loc[pick_idx].reset_index(drop=True)


def write_results_table(df: pd.DataFrame, output_csv: Path) -> None:
    cols = [
        "tomogram_name",
        "set_name",
        "alignment_dir",
        "active_zone",
        "active_zone_name",
        "aunp_index",
        "classification",
        "source_star",
        "source_pick_index",
        "faCoordinateX",
        "faCoordinateY",
        "faCoordinateZ",
        "distance_to_postsynaptic_active_outer_nm",
        "distance_to_postsynaptic_active_inner_nm",
        "distance_to_postsynaptic_active_outer_inner_mean_nm",
    ]
    if "dimer_pair_index" in df.columns:
        cols.insert(cols.index("source_pick_index") + 1, "dimer_pair_index")
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


def write_combined_outputs(frames: List[pd.DataFrame], output_dir: Path, label: str) -> None:
    if not frames:
        return
    combined = pd.concat(frames, ignore_index=True)
    stem = f"{label}_combined"
    write_results_table(combined, output_dir / f"{stem}_postsynaptic_distances.csv")
    for classification in ("monomer", "dimer"):
        subset = combined[combined["classification"] == classification]
        if subset.empty:
            print(f"  Warning: no {classification} picks in combined table; skipping split CSV")
            continue
        write_results_table(
            subset,
            output_dir / f"{stem}_postsynaptic_distances_{classification}.csv",
        )
    dimer_closest = select_closest_dimer_picks_per_pair(combined)
    if not dimer_closest.empty:
        write_results_table(
            dimer_closest,
            output_dir / f"{stem}_postsynaptic_distances_dimer_closest.csv",
        )


def iter_csv_jobs(csv_path: Path) -> Iterable[Tuple[str, str, str, int]]:
    df = pd.read_csv(csv_path)
    required = {"tomoname", "set", "alignment_dir", "aunp_active_zones"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")

    for _, row in df.iterrows():
        tomoname = str(row["tomoname"]).strip()
        set_name = str(row["set"]).strip()
        alignment_dir = str(row["alignment_dir"]).strip()
        if not tomoname or not set_name or not alignment_dir or alignment_dir.lower() == "nan":
            print(f"Skipping row with missing tomoname/set/alignment_dir: {row.to_dict()}")
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
            yield tomoname, set_name, alignment_dir, az


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute postsynaptic active-zone center distances (inner/outer mean) "
            "for monomer and each_dimer AuNP picks using STT_results activezone segmentations."
        )
    )
    parser.add_argument("--csv", type=Path, required=True, help="Tomogram CSV")
    parser.add_argument(
        "--tomo-root-base",
        type=Path,
        default=Path(DEFAULT_TOMO_ROOT_BASE),
        help="Root directory containing {set}/TOP_TOMOS/ (default: TOMO_ROOT_BASE env or goliath path)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    csv_path = args.csv.expanduser().resolve()
    tomo_root_base = args.tomo_root_base.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not csv_path.is_file():
        print(f"Error: CSV not found: {csv_path}", file=sys.stderr)
        return 1
    if not tomo_root_base.is_dir():
        print(f"Error: tomogram root not found: {tomo_root_base}", file=sys.stderr)
        return 1

    jobs = list(iter_csv_jobs(csv_path))
    if not jobs:
        print("No jobs to process.", file=sys.stderr)
        return 1

    ok = 0
    failed = 0
    all_results: List[pd.DataFrame] = []
    for tomoname, set_name, alignment_dir, active_zone in jobs:
        try:
            df = analyze_one(
                tomo_root_base,
                output_dir,
                tomoname,
                set_name,
                alignment_dir,
                active_zone,
            )
            all_results.append(df)
            ok += 1
        except Exception as exc:
            failed += 1
            print(f"  Error: {exc}", file=sys.stderr)

    if all_results:
        print(f"\nWriting combined outputs ({len(all_results)} jobs)...")
        write_combined_outputs(all_results, output_dir, csv_path.stem)

    print(f"\nFinished: {ok} succeeded, {failed} failed.")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
