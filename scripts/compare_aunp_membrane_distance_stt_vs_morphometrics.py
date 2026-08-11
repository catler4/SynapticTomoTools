#!/usr/bin/env python3
"""
Compare AuNP→membrane distances: STT outer/inner mean vs Surface Morphometrics midplane PLY.

For each tomogram in an STT tomograms.csv, loads:
  - AuNP picks: aunp_tm_BP_active_zone_*_manual_refined.star (override with --aunp-pick-star-pattern)
  - STT AZ inner/outer point clouds from STT_results/activezone/
  - SM midplane meshes:
      <alignment_dir>/surface_morphometrics/<tomoname>_cleft_pre.ply
      <alignment_dir>/surface_morphometrics/<tomoname>_cleft_post.ply

STT "center" estimate (same as pipeline):
  distance = mean(nearest_to_outer, nearest_to_inner)

SM estimate:
  distance = nearest distance to the single midplane triangle mesh

How to run
----------
  PYTHONPATH=src python scripts/compare_aunp_membrane_distance_stt_vs_morphometrics.py \\
    --csv tomogram_csv_files/tomograms_15F1-H12Cys_FINAL.csv \\
    --data-dir data \\
    --output-dir results/aunp_membrane_distance_stt_vs_morphometrics

If PLY coordinates are in Angstroms (AuNPs are nm), pass ``--mesh-scale 0.1``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import trimesh
from scipy.spatial import KDTree

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from synaptic_tomo_tools.activezone import import_active_zone_segmentations
from synaptic_tomo_tools.alignment_utils import require_alignment_dir
from synaptic_tomo_tools.aunps import (
    DEFAULT_AUNP_PICK_STAR_PATTERN,
    load_aunp_pick_star_dataframes,
)

COORD_COLS = ("faCoordinateX", "faCoordinateY", "faCoordinateZ")


def default_data_root() -> Path:
    return Path(os.environ.get("TOMO_ROOT_BASE") or "data")


def parse_az_indices(value) -> list[int] | None:
    """Parse CSV aunp_active_zones; None means discover all matching STAR files."""
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return None
    indices: list[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        indices.append(int(float(part)))
    return indices or None


def load_csv_jobs(
    csv_path: Path,
    *,
    data_dir: Path,
    set_name: str | None = None,
) -> list[tuple[Path, str, str, list[int] | None]]:
    df = pd.read_csv(csv_path)
    required = {"tomoname", "set", "alignment_dir"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")

    if set_name:
        df = df[df["set"].astype(str) == str(set_name)]

    jobs: list[tuple[Path, str, str, list[int] | None]] = []
    for _, row in df.iterrows():
        tomoname = str(row["tomoname"]).strip()
        row_set = str(row["set"]).strip()
        alignment_dir = require_alignment_dir(
            row["alignment_dir"], context=f"tomogram {tomoname}"
        )
        if not tomoname or not row_set or tomoname.lower() == "nan":
            print(f"Skipping row with missing tomoname/set: {row.to_dict()}")
            continue
        az_indices = None
        if "aunp_active_zones" in df.columns:
            az_indices = parse_az_indices(row.get("aunp_active_zones"))
        tomo_path = Path(data_dir) / row_set / "TOP_TOMOS" / tomoname
        jobs.append((tomo_path, alignment_dir, row_set, az_indices))
    return jobs


def nearest_distances_to_cloud(points: np.ndarray, cloud: np.ndarray) -> np.ndarray:
    if cloud is None or len(np.atleast_2d(cloud)) == 0 or np.size(cloud) == 0:
        return np.full(len(points), np.nan)
    cloud = np.atleast_2d(np.asarray(cloud, dtype=float))
    if cloud.ndim != 2 or cloud.shape[1] != 3:
        return np.full(len(points), np.nan)
    tree = KDTree(cloud)
    dists, _ = tree.query(points)
    return np.asarray(dists, dtype=float)


def stack_clouds(clouds: list[np.ndarray]) -> np.ndarray:
    parts = []
    for cloud in clouds:
        arr = np.asarray(cloud, dtype=float)
        if arr.size == 0:
            continue
        arr = np.atleast_2d(arr)
        if arr.shape[1] == 3:
            parts.append(arr)
    return np.vstack(parts) if parts else np.zeros((0, 3), dtype=float)


def load_ply_mesh(path: Path, *, mesh_scale: float) -> trimesh.Trimesh:
    mesh = trimesh.load(path, process=False, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        geoms = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not geoms:
            raise ValueError(f"No triangle mesh found in {path}")
        mesh = trimesh.util.concatenate(geoms)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Expected a triangle mesh in {path}, got {type(mesh)}")
    if float(mesh_scale) != 1.0:
        mesh = mesh.copy()
        mesh.apply_scale(float(mesh_scale))
    return mesh


def nearest_distances_to_mesh(points: np.ndarray, mesh: trimesh.Trimesh) -> np.ndarray:
    """Unsigned nearest distance (nm) from points to triangle mesh surface."""
    points = np.asarray(points, dtype=float)
    if len(points) == 0:
        return np.zeros(0, dtype=float)
    try:
        _, distances, _ = mesh.nearest.on_surface(points)
        return np.asarray(distances, dtype=float)
    except Exception:
        # Fallback: vertex KDTree (closer to STT's point-cloud style)
        tree = KDTree(np.asarray(mesh.vertices, dtype=float))
        dists, _ = tree.query(points)
        return np.asarray(dists, dtype=float)


def stt_outer_inner_mean_distances(
    coords: np.ndarray,
    tomogram_path: Path,
    alignment_dir: str,
) -> dict[str, np.ndarray]:
    az_segmentations = import_active_zone_segmentations(
        tomogram_path, alignment_dir=alignment_dir
    )
    pre_outer = stack_clouds(
        [az.get("presynaptic_outer_coords", []) for az in az_segmentations.values()]
    )
    post_outer = stack_clouds(
        [az.get("postsynaptic_outer_coords", []) for az in az_segmentations.values()]
    )
    pre_inner = stack_clouds(
        [az.get("presynaptic_inner_coords", []) for az in az_segmentations.values()]
    )
    post_inner = stack_clouds(
        [az.get("postsynaptic_inner_coords", []) for az in az_segmentations.values()]
    )

    d_pre_outer = nearest_distances_to_cloud(coords, pre_outer)
    d_post_outer = nearest_distances_to_cloud(coords, post_outer)
    d_pre_inner = nearest_distances_to_cloud(coords, pre_inner)
    d_post_inner = nearest_distances_to_cloud(coords, post_inner)

    return {
        "distance_to_presynaptic_active_outer_nm": d_pre_outer,
        "distance_to_postsynaptic_active_outer_nm": d_post_outer,
        "distance_to_presynaptic_active_inner_nm": d_pre_inner,
        "distance_to_postsynaptic_active_inner_nm": d_post_inner,
        "distance_to_presynaptic_active_outer_inner_mean_nm": np.nanmean(
            np.vstack([d_pre_outer, d_pre_inner]), axis=0
        ),
        "distance_to_postsynaptic_active_outer_inner_mean_nm": np.nanmean(
            np.vstack([d_post_outer, d_post_inner]), axis=0
        ),
        "n_pre_outer_points": len(pre_outer),
        "n_post_outer_points": len(post_outer),
        "n_pre_inner_points": len(pre_inner),
        "n_post_inner_points": len(post_inner),
    }


def summarize_comparison(df: pd.DataFrame, side: str) -> dict[str, float]:
    stt_col = f"distance_to_{side}synaptic_active_outer_inner_mean_nm"
    sm_col = f"distance_to_{side}synaptic_sm_midplane_nm"
    delta_col = f"delta_{side}_sm_minus_stt_mean_nm"
    if stt_col not in df.columns or sm_col not in df.columns:
        return {}
    mask = np.isfinite(df[stt_col]) & np.isfinite(df[sm_col])
    if not np.any(mask):
        return {
            "n": 0.0,
            "mae_nm": np.nan,
            "bias_sm_minus_stt_nm": np.nan,
            "pearson_r": np.nan,
            "stt_mean_nm": np.nan,
            "sm_mean_nm": np.nan,
        }
    stt = df.loc[mask, stt_col].to_numpy(dtype=float)
    sm = df.loc[mask, sm_col].to_numpy(dtype=float)
    delta = sm - stt
    if delta_col in df.columns:
        pass
    if len(stt) >= 2 and np.std(stt) > 0 and np.std(sm) > 0:
        pearson_r = float(np.corrcoef(stt, sm)[0, 1])
    else:
        pearson_r = np.nan
    return {
        "n": float(len(stt)),
        "mae_nm": float(np.mean(np.abs(delta))),
        "bias_sm_minus_stt_nm": float(np.mean(delta)),
        "pearson_r": pearson_r,
        "stt_mean_nm": float(np.mean(stt)),
        "sm_mean_nm": float(np.mean(sm)),
    }


def analyze_one(
    tomogram_path: Path,
    alignment_dir: str,
    *,
    set_name: str,
    az_indices: list[int] | None,
    star_pattern: str,
    mesh_scale: float,
    output_dir: Path,
) -> pd.DataFrame | None:
    tomoname = tomogram_path.name
    alignment_dir = require_alignment_dir(alignment_dir)
    aunps_dir = tomogram_path / alignment_dir / "aunps"
    sm_dir = tomogram_path / alignment_dir / "surface_morphometrics"
    pre_ply = sm_dir / f"{tomoname}_cleft_pre.ply"
    post_ply = sm_dir / f"{tomoname}_cleft_post.ply"

    print(f"\n{'=' * 60}")
    print(f"{set_name} / {tomoname}__{alignment_dir}")
    print(f"{'=' * 60}")

    if not aunps_dir.is_dir():
        print(f"  SKIP: aunps dir missing: {aunps_dir}")
        return None

    star_dfs = load_aunp_pick_star_dataframes(
        aunps_dir, az_indices, pattern=star_pattern
    )
    if not star_dfs:
        print(
            f"  SKIP: no AuNP STAR files matching {star_pattern!r} "
            f"(az filter={az_indices})"
        )
        return None

    df = pd.concat(star_dfs, ignore_index=True)
    missing = [c for c in COORD_COLS if c not in df.columns]
    if missing:
        print(f"  SKIP: STAR missing coordinate columns {missing}")
        return None

    coords = df[list(COORD_COLS)].to_numpy(dtype=float)
    print(f"  AuNPs: {len(coords):,} (zones={sorted(df['active_zone'].unique().tolist())})")

    # STT outer/inner mean
    try:
        stt = stt_outer_inner_mean_distances(coords, tomogram_path, alignment_dir)
    except Exception as exc:
        print(f"  SKIP: could not load STT active-zone segmentations: {exc}")
        return None
    print(
        f"  STT AZ points: pre outer/inner={stt['n_pre_outer_points']:,}/"
        f"{stt['n_pre_inner_points']:,}, post outer/inner="
        f"{stt['n_post_outer_points']:,}/{stt['n_post_inner_points']:,}"
    )

    out = df.copy()
    out.insert(0, "tomoname", tomoname)
    out.insert(1, "set", set_name)
    out.insert(2, "alignment_dir", alignment_dir)
    for key in (
        "distance_to_presynaptic_active_outer_nm",
        "distance_to_postsynaptic_active_outer_nm",
        "distance_to_presynaptic_active_inner_nm",
        "distance_to_postsynaptic_active_inner_nm",
        "distance_to_presynaptic_active_outer_inner_mean_nm",
        "distance_to_postsynaptic_active_outer_inner_mean_nm",
    ):
        out[key] = stt[key]

    # Surface Morphometrics midplane meshes
    for side, ply_path, out_col in (
        ("pre", pre_ply, "distance_to_presynaptic_sm_midplane_nm"),
        ("post", post_ply, "distance_to_postsynaptic_sm_midplane_nm"),
    ):
        if not ply_path.is_file():
            print(f"  WARNING: missing {side} PLY: {ply_path}")
            out[out_col] = np.nan
            continue
        mesh = load_ply_mesh(ply_path, mesh_scale=mesh_scale)
        dists = nearest_distances_to_mesh(coords, mesh)
        out[out_col] = dists
        print(
            f"  SM {side} mesh: {ply_path.name} "
            f"({len(mesh.vertices):,} verts, {len(mesh.faces):,} faces); "
            f"mean dist={np.nanmean(dists):.2f} nm"
        )

    out["delta_pre_sm_minus_stt_mean_nm"] = (
        out["distance_to_presynaptic_sm_midplane_nm"]
        - out["distance_to_presynaptic_active_outer_inner_mean_nm"]
    )
    out["delta_post_sm_minus_stt_mean_nm"] = (
        out["distance_to_postsynaptic_sm_midplane_nm"]
        - out["distance_to_postsynaptic_active_outer_inner_mean_nm"]
    )

    for side in ("pre", "post"):
        stats = summarize_comparison(out, side)
        if stats.get("n", 0) > 0:
            print(
                f"  Compare {side}: n={int(stats['n'])}, "
                f"MAE={stats['mae_nm']:.2f} nm, "
                f"bias(SM-STT)={stats['bias_sm_minus_stt_nm']:.2f} nm, "
                f"r={stats['pearson_r']:.3f}"
            )

    out_path = (
        output_dir
        / f"{tomoname}__{alignment_dir}_aunp_membrane_distance_stt_vs_sm.csv"
    )
    out.to_csv(out_path, index=False)
    print(f"  Wrote {out_path}")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare AuNP membrane distances: STT outer/inner mean vs "
            "Surface Morphometrics midplane PLY."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--csv", type=Path, required=True, help="STT tomograms.csv")
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
        "--output-dir",
        type=Path,
        default=Path("results/aunp_membrane_distance_stt_vs_morphometrics"),
        help="Directory for per-tomogram and pooled CSVs",
    )
    parser.add_argument(
        "--aunp-pick-star-pattern",
        type=str,
        default=DEFAULT_AUNP_PICK_STAR_PATTERN,
        help="AuNP STAR pattern (* = active-zone index)",
    )
    parser.add_argument(
        "--mesh-scale",
        type=float,
        default=1.0,
        help="Multiply PLY coordinates by this (use 0.1 if mesh is in Å and AuNPs in nm)",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Abort on first tomogram failure",
    )
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir) if args.data_dir else default_data_root()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    jobs = load_csv_jobs(
        Path(args.csv), data_dir=data_dir, set_name=args.set_name
    )
    if not jobs:
        print(f"No tomogram rows to process from {args.csv}", file=sys.stderr)
        return 1

    print(f"Jobs: {len(jobs)} from {args.csv} (data root: {data_dir})")
    print(f"AuNP STAR pattern: {args.aunp_pick_star_pattern}")
    print(f"Mesh scale: {args.mesh_scale}")
    print(f"Output: {output_dir}")

    frames: list[pd.DataFrame] = []
    failed: list[str] = []
    for i, (tomo_path, alignment_dir, set_name, az_indices) in enumerate(jobs, start=1):
        label = f"{tomo_path.name}__{alignment_dir}"
        print(f"\n[{i}/{len(jobs)}] {label}")
        try:
            df = analyze_one(
                tomo_path,
                alignment_dir,
                set_name=set_name,
                az_indices=az_indices,
                star_pattern=args.aunp_pick_star_pattern,
                mesh_scale=float(args.mesh_scale),
                output_dir=output_dir,
            )
            if df is None:
                failed.append(label)
            else:
                frames.append(df)
        except Exception as exc:
            print(f"FAILED {label}: {exc}")
            failed.append(label)
            if args.stop_on_error:
                raise

    if frames:
        pooled = pd.concat(frames, ignore_index=True)
        pooled_path = output_dir / "aunp_membrane_distance_stt_vs_sm_pooled.csv"
        pooled.to_csv(pooled_path, index=False)
        print(f"\nWrote pooled table -> {pooled_path} ({len(pooled):,} AuNPs)")

        summary_rows = []
        for side in ("pre", "post"):
            stats = summarize_comparison(pooled, side)
            summary_rows.append({"side": side, **stats})
            if stats.get("n", 0) > 0:
                print(
                    f"Pooled {side}: n={int(stats['n'])}, "
                    f"MAE={stats['mae_nm']:.2f} nm, "
                    f"bias(SM-STT)={stats['bias_sm_minus_stt_nm']:.2f} nm, "
                    f"r={stats['pearson_r']:.3f}"
                )
        summary = pd.DataFrame(summary_rows)
        summary_path = output_dir / "aunp_membrane_distance_stt_vs_sm_summary.csv"
        summary.to_csv(summary_path, index=False)
        print(f"Wrote summary -> {summary_path}")

    print(f"\nDone. Succeeded: {len(frames)}/{len(jobs)}")
    if failed:
        print(f"Failed/skipped ({len(failed)}):")
        for label in failed:
            print(f"  {label}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
