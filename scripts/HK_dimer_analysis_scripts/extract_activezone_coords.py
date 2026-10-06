#!/usr/bin/env python3
import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

import numpy as np
import mrcfile

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec


def load_xyz_txt(file_path: Path) -> np.ndarray:
    coords: List[List[float]] = []
    with file_path.open("r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            if len(parts) < 3:
                continue
            try:
                x, y, z = map(float, parts[:3])
            except Exception:
                continue
            coords.append([x, y, z])
    if coords:
        return np.asarray(coords, dtype=float)
    return np.zeros((0, 3), dtype=float)


def write_xyz_txt(coords: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for pt in coords:
            f.write(f"{pt[0]:.3f} {pt[1]:.3f} {pt[2]:.3f}\n")


def parse_star_coords(star_path: Path) -> Dict[str, np.ndarray]:
    monomer: List[List[float]] = []
    dimer: List[List[float]] = []

    with star_path.open("r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("#") or s.lower().startswith("data_") or s.lower().startswith("loop_") or s.startswith("_"):
                continue

            parts = re.split(r"\s+", s)
            if len(parts) < 4:
                continue

            try:
                z = float(parts[0])
                y = float(parts[1])
                x = float(parts[2])
                t = parts[3].strip().lower()
            except Exception:
                continue

            xyz = [x, y, z]

            if t == "monomer":
                monomer.append(xyz)
            elif t == "dimer":
                dimer.append(xyz)

    return {
        "monomer": np.asarray(monomer, dtype=float) if monomer else np.zeros((0, 3), dtype=float),
        "dimer": np.asarray(dimer, dtype=float) if dimer else np.zeros((0, 3), dtype=float),
    }


def write_activezone_star_from_filtered_txt(filtered_m_txt: Path, filtered_d_txt: Path, out_star: Path) -> None:
    m_xyz = load_xyz_txt(filtered_m_txt)
    d_xyz = load_xyz_txt(filtered_d_txt)

    out_star.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now().strftime("%H:%M:%S on %d/%m/%Y")

    with out_star.open("w") as f:
        f.write(f"# Created by activezone extraction script at {now}\n\n")
        f.write("data_\n\n")
        f.write("loop_\n")
        f.write("_faCoordinateX #1\n")
        f.write("_faCoordinateY #2\n")
        f.write("_faCoordinateZ #3\n")
        f.write("_type #4\n")

        # write as Z Y X type, same as your input convention
        for x, y, z in m_xyz:
            f.write(f"{z:.6f}\t{y:.6f}\t{x:.6f}\tmonomer\n")
        for x, y, z in d_xyz:
            f.write(f"{z:.6f}\t{y:.6f}\t{x:.6f}\tdimer\n")


def _load_npy_center_cs(npy_file: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(npy_file, allow_pickle=True)
    if isinstance(data, np.ndarray) and data.dtype == object and data.shape == ():
        data = data.item()
    if not isinstance(data, dict) or "center" not in data or "cs" not in data:
        raise ValueError(f"{npy_file} must contain dict with keys 'center' and 'cs'")
    return np.array(data["center"], dtype=float), np.array(data["cs"], dtype=float)


def _mrc_size_xyz(ext_mrc: Path) -> np.ndarray:
    with mrcfile.open(ext_mrc, permissive=True) as m:
        vol = m.data
    return np.array([vol.shape[2], vol.shape[1], vol.shape[0]], dtype=float)


def transform_coordinates(coords_xyz: np.ndarray, center: np.ndarray, cs: np.ndarray, size_xyz: np.ndarray) -> np.ndarray:
    if coords_xyz.size == 0:
        return np.zeros((0, 3), dtype=float)
    delta = coords_xyz - center
    rotated = delta.dot(cs.T)
    return rotated + size_xyz / 2.0


def mask_in_bounds(coords_xyz: np.ndarray, size_xyz: np.ndarray, margin: float = 0.0) -> np.ndarray:
    if coords_xyz.size == 0:
        return np.zeros((0,), dtype=bool)

    nx, ny, nz = size_xyz.astype(float)
    x = coords_xyz[:, 0]
    y = coords_xyz[:, 1]
    z = coords_xyz[:, 2]

    return (
        (x >= margin) & (x < nx - margin) &
        (y >= margin) & (y < ny - margin) &
        (z >= margin) & (z < nz - margin)
    )


def mask_extracted_coords_by_zero_patch(
    coords_xyz: np.ndarray,
    mrc_file: Path,
    patch_radius: int = 1,
    rounding: str = "round",
    verbose: bool = True,
) -> np.ndarray:
    if coords_xyz.size == 0:
        return np.zeros((0,), dtype=bool)

    with mrcfile.open(mrc_file, permissive=True) as m:
        vol = m.data

    nz, ny, nx = vol.shape

    if rounding == "round":
        ijk = np.rint(coords_xyz).astype(int)
    elif rounding == "floor":
        ijk = np.floor(coords_xyz).astype(int)
    elif rounding == "ceil":
        ijk = np.ceil(coords_xyz).astype(int)
    else:
        raise ValueError(f"Unknown rounding={rounding}")

    x = ijk[:, 0]
    y = ijk[:, 1]
    z = ijk[:, 2]

    keep_mask = np.ones((len(coords_xyz),), dtype=bool)

    for i in range(len(coords_xyz)):
        xi, yi, zi = x[i], y[i], z[i]

        if not (0 <= xi < nx and 0 <= yi < ny and 0 <= zi < nz):
            keep_mask[i] = False
            continue

        x0 = xi - patch_radius
        x1 = xi + patch_radius
        y0 = yi - patch_radius
        y1 = yi + patch_radius

        if x0 < 0 or y0 < 0 or x1 >= nx or y1 >= ny:
            keep_mask[i] = False
            continue

        center_val = vol[zi, yi, xi]
        if center_val != 0:
            continue

        patch = vol[zi, y0:y1 + 1, x0:x1 + 1]
        if np.all(patch == 0):
            keep_mask[i] = False

    if verbose:
        dropped = len(coords_xyz) - int(np.sum(keep_mask))
        print(f"[INFO] Zero-patch mask ({mrc_file.name}): kept {int(np.sum(keep_mask))}/{len(coords_xyz)} dropped {dropped}")

    return keep_mask


def find_active_zonogram_pairs(az_dir: Path, verbose: bool = True) -> List[Tuple[int, int, Path, Path]]:
    npy_pat = re.compile(r"^active_zonogram_(\d+)\.npy$", re.IGNORECASE)
    mrc_pat = re.compile(r"^active_zonogram_(\d+)\.mrc$", re.IGNORECASE)

    npys: Dict[int, Path] = {}
    mrcs: Dict[int, Path] = {}

    for p in az_dir.iterdir():
        if not p.is_file():
            continue

        m = npy_pat.match(p.name)
        if m:
            npys[int(m.group(1))] = p
            continue

        m = mrc_pat.match(p.name)
        if m:
            mrcs[int(m.group(1))] = p
            continue

    common = sorted(set(npys.keys()) & set(mrcs.keys()))
    pairs: List[Tuple[int, int, Path, Path]] = []

    for zon_idx in common:
        k = zon_idx + 1
        pairs.append((k, zon_idx, npys[zon_idx], mrcs[zon_idx]))

    if verbose:
        print(f"[INFO] Detected active_zonogram indices: {common if common else 'none'}")
        for k, zon_idx, npy_p, mrc_p in pairs:
            print(f"       k={k}: zon_idx={zon_idx} -> {npy_p.name}, {mrc_p.name}")

    return pairs


def load_coordinates_or_none(coord_path: Path):
    if coord_path is None or not coord_path.exists():
        return None

    coords = []
    with coord_path.open("r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            coords.append(list(map(float, parts)))

    return np.array(coords) if len(coords) > 0 else None


def render_png_from_mrc(
    mrc_path: Path,
    output_path: Path,
    monomer_coords: np.ndarray = None,
    dimer_coords: np.ndarray = None,
):
    with mrcfile.open(mrc_path, permissive=True) as mrc:
        vol_np = mrc.data.copy()
        vol = torch.tensor(vol_np)

    if vol.ndim != 3 or 0 in vol.shape:
        raise ValueError(f"Invalid MRC volume shape: {vol.shape}")

    width = (vol.shape[2] + vol.shape[0]) / 50
    height = (vol.shape[1] + vol.shape[0]) / 50

    fig = plt.figure(figsize=(width, height))
    gs = gridspec.GridSpec(
        2, 2,
        width_ratios=[vol.shape[2], vol.shape[0]],
        height_ratios=[vol.shape[1], vol.shape[0]],
    )

    axxy = plt.subplot(gs[0, 0])
    axyz = plt.subplot(gs[0, 1], sharey=axxy)
    axxz = plt.subplot(gs[1, 0], sharex=axxy)

    axxy.imshow(torch.min(vol, axis=0).values, cmap="gray", interpolation="mitchell",
                vmax=0, vmin=-20 * vol.std(), origin="lower")
    axxy.axis("off")

    axyz.imshow(torch.min(vol, axis=2).values.T, cmap="gray", interpolation="mitchell",
                vmax=0, vmin=-20 * vol.std(), origin="lower")
    axyz.axis("off")

    axxz.imshow(torch.min(vol, axis=1).values, cmap="gray", interpolation="mitchell",
                vmax=0, vmin=-20 * vol.std(), origin="lower")
    axxz.axis("off")

    if monomer_coords is not None and monomer_coords.ndim == 2 and monomer_coords.shape[1] == 3:
        axxy.scatter(monomer_coords[:, 0], monomer_coords[:, 1], s=100, facecolors="none", edgecolors="red", linewidths=1.5)
        axyz.scatter(monomer_coords[:, 2], monomer_coords[:, 1], s=100, facecolors="none", edgecolors="red", linewidths=1.5)
        axxz.scatter(monomer_coords[:, 0], monomer_coords[:, 2], s=100, facecolors="none", edgecolors="red", linewidths=1.5)

    if dimer_coords is not None and dimer_coords.ndim == 2 and dimer_coords.shape[1] == 3:
        axxy.scatter(dimer_coords[:, 0], dimer_coords[:, 1], s=100, facecolors="none", edgecolors="blue", linewidths=1.5)
        axyz.scatter(dimer_coords[:, 2], dimer_coords[:, 1], s=100, facecolors="none", edgecolors="blue", linewidths=1.5)
        axxz.scatter(dimer_coords[:, 0], dimer_coords[:, 2], s=100, facecolors="none", edgecolors="blue", linewidths=1.5)

    if monomer_coords is not None:
        fig.text(0.02, 0.98, f"Monomer: {monomer_coords.shape[0]}", ha="left", va="top", fontsize=10, color="red")
    if dimer_coords is not None:
        fig.text(0.02, 0.93, f"Dimer: {dimer_coords.shape[0]}", ha="left", va="top", fontsize=10, color="blue")

    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def generate_pngs_for_pair(
    az_mrc: Path,
    extracted_m: Path,
    extracted_d: Path,
    png_dir: Path,
    zon_idx: int,
    verbose: bool = True,
):
    png_dir.mkdir(parents=True, exist_ok=True)

    base_png = png_dir / f"active_zonogram_{zon_idx}.png"
    annotated_png = png_dir / f"active_zonogram_{zon_idx}_annotated.png"

    render_png_from_mrc(az_mrc, base_png)
    if verbose:
        print(f"[OK] Unannotated PNG saved: {base_png}")

    monomer_coords = load_coordinates_or_none(extracted_m)
    dimer_coords = load_coordinates_or_none(extracted_d)

    if monomer_coords is not None or dimer_coords is not None:
        render_png_from_mrc(
            az_mrc,
            annotated_png,
            monomer_coords=monomer_coords,
            dimer_coords=dimer_coords,
        )
        if verbose:
            print(f"[OK] Annotated PNG saved  : {annotated_png}")
    else:
        if verbose:
            print(f"[INFO] No coords for annotation for zon_idx={zon_idx}")


def resolve_path(path_str: str, root: Path) -> Path:
    p = Path(path_str).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (root / p).resolve()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "global STAR -> active zonogram extracted coords. "
            "--star and --out can be absolute paths or relative to --dir."
        )
    )

    parser.add_argument("--dir", required=True, type=str, help="Root directory containing active_zonograms/")
    parser.add_argument("--star", required=True, type=str, help="STAR file: absolute path or relative to --dir")
    parser.add_argument("--out", default="active_zonograms/dual", type=str, help="Output directory: absolute path or relative to --dir")

    parser.add_argument("--margin", type=float, default=0.0)
    parser.add_argument("--no_png", action="store_true")
    parser.add_argument("--quiet", action="store_true")

    parser.add_argument("--zero_patch_radius", type=int, default=1)
    parser.add_argument("--zero_patch_round", choices=["round", "floor", "ceil"], default="round")

    args = parser.parse_args()
    verbose = not args.quiet

    root = Path(args.dir).expanduser().resolve()
    star_path = resolve_path(args.star, root)
    out_dir = resolve_path(args.out, root)

    az_dir = root / "active_zonograms"
    png_dir = out_dir / "png"

    if not star_path.exists():
        raise FileNotFoundError(f"STAR not found: {star_path}")
    if not az_dir.exists():
        raise FileNotFoundError(f"active_zonograms dir not found: {az_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"[INFO] root        : {root}")
        print(f"[INFO] star        : {star_path}")
        print(f"[INFO] az_dir      : {az_dir}")
        print(f"[INFO] out_dir     : {out_dir}")
        print(f"[INFO] margin      : {args.margin}")
        print(f"[INFO] zero_radius : {args.zero_patch_radius}")
        print(f"[INFO] zero_round  : {args.zero_patch_round}")

    parsed = parse_star_coords(star_path)
    monomer_global = parsed["monomer"]
    dimer_global = parsed["dimer"]

    write_xyz_txt(monomer_global, out_dir / "monomer.txt")
    write_xyz_txt(dimer_global, out_dir / "dimer.txt")

    pairs = find_active_zonogram_pairs(az_dir, verbose=verbose)
    if not pairs:
        print(f"[WARN] No active_zonogram pairs found in: {az_dir}")
        return

    for k, zon_idx, npy_file, mrc_file in pairs:
        center, cs = _load_npy_center_cs(npy_file)
        size_xyz = _mrc_size_xyz(mrc_file)

        monomer_extracted_all = transform_coordinates(monomer_global, center, cs, size_xyz)
        monomer_inb = mask_in_bounds(monomer_extracted_all, size_xyz, margin=args.margin)
        monomer_zp = mask_extracted_coords_by_zero_patch(
            monomer_extracted_all,
            mrc_file,
            patch_radius=args.zero_patch_radius,
            rounding=args.zero_patch_round,
            verbose=verbose,
        )
        monomer_keep = monomer_inb & monomer_zp

        monomer_filtered_global = monomer_global[monomer_keep]
        monomer_extracted_final = monomer_extracted_all[monomer_keep]

        dimer_extracted_all = transform_coordinates(dimer_global, center, cs, size_xyz)
        dimer_inb = mask_in_bounds(dimer_extracted_all, size_xyz, margin=args.margin)
        dimer_zp = mask_extracted_coords_by_zero_patch(
            dimer_extracted_all,
            mrc_file,
            patch_radius=args.zero_patch_radius,
            rounding=args.zero_patch_round,
            verbose=verbose,
        )
        dimer_keep = dimer_inb & dimer_zp

        dimer_filtered_global = dimer_global[dimer_keep]
        dimer_extracted_final = dimer_extracted_all[dimer_keep]

        filtered_m_path = out_dir / f"filtered_monomer_{k}.txt"
        filtered_d_path = out_dir / f"filtered_dimer_{k}.txt"
        extracted_m_path = out_dir / f"extracted_m_{k}.txt"
        extracted_d_path = out_dir / f"extracted_d_{k}.txt"

        write_xyz_txt(monomer_filtered_global, filtered_m_path)
        write_xyz_txt(dimer_filtered_global, filtered_d_path)

        np.savetxt(extracted_m_path, monomer_extracted_final, fmt="%.3f")
        np.savetxt(extracted_d_path, dimer_extracted_final, fmt="%.3f")

        activezone_star_path = out_dir / f"activezone_{k}.star"
        write_activezone_star_from_filtered_txt(
            filtered_m_path,
            filtered_d_path,
            activezone_star_path,
        )

        print(
            f"[DONE] k={k} <-> zon_idx={zon_idx} | "
            f"monomer {len(monomer_filtered_global)} | "
            f"dimer {len(dimer_filtered_global)} | "
            f"STAR={activezone_star_path}"
        )

        if not args.no_png:
            generate_pngs_for_pair(
                az_mrc=mrc_file,
                extracted_m=extracted_m_path,
                extracted_d=extracted_d_path,
                png_dir=png_dir,
                zon_idx=zon_idx,
                verbose=verbose,
            )

    print(f"[ALL DONE] Outputs in: {out_dir}")


if __name__ == "__main__":
    main()