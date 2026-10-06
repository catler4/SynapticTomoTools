#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import mrcfile
import starfile

from scipy.optimize import least_squares


BIN_SCALE = 2.0

# BIN2 dimer search
DIMER_SEARCH_RADIUS_BIN2 = 10
LOCAL_MIN_RADIUS = 2
MIN_SEPARATION_BIN2 = 4

# Mask monomer positions in BIN2 before finding dimer AuNPs
MONOMER_MASK_RADIUS_BIN2 = 4

# Duplicate proofreading
DUPLICATE_DISTANCE_BIN4 = 2.0
MAX_DUPLICATE_FIX_ITER = 5

# BIN4 Gaussian fitting
GAUSS_RADIUS_BIN4 = 1


# ===================== IO =====================

def load_mrc(path):
    with mrcfile.open(path, permissive=True) as m:
        return m.data.astype(np.float32)


def load_star_as_zyx_type(path):
    obj = starfile.read(path)
    if isinstance(obj, dict):
        df = next(iter(obj.values()))
    else:
        df = obj

    df = df.iloc[:, :4].copy()
    df.columns = ["z", "y", "x", "type"]

    df["z"] = df["z"].astype(float)
    df["y"] = df["y"].astype(float)
    df["x"] = df["x"].astype(float)
    df["type"] = df["type"].astype(str)

    return df


def save_xyz_txt(coords_xyz, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    coords_xyz = np.asarray(coords_xyz, dtype=float)

    if coords_xyz.size == 0:
        path.write_text("")
        return

    coords_xyz = coords_xyz.reshape(-1, 3)
    np.savetxt(path, coords_xyz, fmt="%.6f", delimiter="\t")


def load_xyz_txt(path):
    path = Path(path)

    if not path.exists() or path.stat().st_size == 0:
        return np.zeros((0, 3), dtype=float)

    try:
        arr = np.loadtxt(path)
    except Exception:
        return np.zeros((0, 3), dtype=float)

    arr = np.asarray(arr, dtype=float)

    if arr.size == 0:
        return np.zeros((0, 3), dtype=float)

    if arr.ndim == 1:
        if arr.size < 3:
            return np.zeros((0, 3), dtype=float)
        arr = arr[:3].reshape(1, 3)

    if arr.shape[1] < 3:
        return np.zeros((0, 3), dtype=float)

    return arr[:, :3].astype(float)


# ===================== PATCH =====================

def extract_patch(vol, center_zyx, radius):
    zc, yc, xc = np.round(center_zyx).astype(int)

    z0 = max(0, zc - radius)
    z1 = min(vol.shape[0], zc + radius + 1)

    y0 = max(0, yc - radius)
    y1 = min(vol.shape[1], yc + radius + 1)

    x0 = max(0, xc - radius)
    x1 = min(vol.shape[2], xc + radius + 1)

    patch = vol[z0:z1, y0:y1, x0:x1]
    origin_zyx = np.array([z0, y0, x0], dtype=float)

    return patch, origin_zyx


# ===================== MASKING =====================

def mask_positions_in_patch(
    patch,
    origin_zyx_bin2,
    positions_zyx_bin2,
    radius,
):
    """
    Mask known positions in a BIN2 patch.

    AuNP density is dark, so masked regions are replaced
    with a very bright value to prevent them from being selected
    as local minima.
    """
    if patch.size == 0:
        return patch

    if positions_zyx_bin2 is None or np.asarray(positions_zyx_bin2).size == 0:
        return patch

    positions_zyx_bin2 = np.asarray(positions_zyx_bin2, dtype=float).reshape(-1, 3)

    masked = patch.copy()

    Z, Y, X = masked.shape

    patch_min = origin_zyx_bin2
    patch_max = origin_zyx_bin2 + np.array([Z, Y, X], dtype=float)

    bright_value = float(
        np.max(masked) + abs(np.max(masked) - np.min(masked)) + 100.0
    )

    for pos_zyx in positions_zyx_bin2:
        if np.any(pos_zyx < patch_min - radius):
            continue
        if np.any(pos_zyx >= patch_max + radius):
            continue

        local_zyx = pos_zyx - origin_zyx_bin2
        zc, yc, xc = np.round(local_zyx).astype(int)

        z0 = max(0, zc - radius)
        z1 = min(Z, zc + radius + 1)

        y0 = max(0, yc - radius)
        y1 = min(Y, yc + radius + 1)

        x0 = max(0, xc - radius)
        x1 = min(X, xc + radius + 1)

        if z0 >= z1 or y0 >= y1 or x0 >= x1:
            continue

        zz, yy, xx = np.meshgrid(
            np.arange(z0, z1),
            np.arange(y0, y1),
            np.arange(x0, x1),
            indexing="ij"
        )

        dist = np.sqrt(
            (zz - zc) ** 2 +
            (yy - yc) ** 2 +
            (xx - xc) ** 2
        )

        mask = dist <= radius

        sub = masked[z0:z1, y0:y1, x0:x1]
        sub[mask] = bright_value
        masked[z0:z1, y0:y1, x0:x1] = sub

    return masked


# ===================== LOCAL MINIMA =====================

def find_local_minima_sorted(patch):
    if patch.size == 0:
        return []

    Z, Y, X = patch.shape
    r = LOCAL_MIN_RADIUS
    needed = 2 * r + 1

    if Z < needed or Y < needed or X < needed:
        return []

    minima = []

    for z in range(r, Z - r):
        for y in range(r, Y - r):
            for x in range(r, X - r):
                val = patch[z, y, x]
                neigh = patch[z-r:z+r+1, y-r:y+r+1, x-r:x+r+1]

                if np.all(val <= neigh):
                    minima.append((z, y, x))

    minima_sorted = sorted(minima, key=lambda c: patch[c])
    return minima_sorted


def find_two_darkest_local_minima(patch, origin_zyx):
    minima_sorted = find_local_minima_sorted(patch)

    if len(minima_sorted) < 2:
        return None

    p1 = np.array(minima_sorted[0], dtype=float)

    p2 = None
    for p in minima_sorted[1:]:
        p = np.array(p, dtype=float)
        dist = np.linalg.norm(p - p1)
        if dist >= MIN_SEPARATION_BIN2:
            p2 = p
            break

    if p2 is None:
        return None

    g1_zyx = origin_zyx + p1
    g2_zyx = origin_zyx + p2

    return g1_zyx, g2_zyx


def find_one_new_partner_minimum(patch, origin_zyx, fixed_partner_zyx_bin2):
    """
    Find one new AuNP local minimum for a dimer while keeping the other
    already-valid AuNP as fixed_partner.
    """
    minima_sorted = find_local_minima_sorted(patch)

    if len(minima_sorted) == 0:
        return None

    fixed_local = fixed_partner_zyx_bin2 - origin_zyx

    for p in minima_sorted:
        p = np.array(p, dtype=float)
        dist = np.linalg.norm(p - fixed_local)

        if dist >= MIN_SEPARATION_BIN2:
            return origin_zyx + p

    return None


# ===================== DIMER DETECTION =====================

def detect_eachdimer_pairs_from_bin2(vol_bin2, dimer_zyx_bin4, monomer_zyx_bin4):
    pairs_zyx_bin4 = []

    if dimer_zyx_bin4.size == 0:
        return np.zeros((0, 2, 3), dtype=float)

    if monomer_zyx_bin4 is None or monomer_zyx_bin4.size == 0:
        monomer_zyx_bin2 = np.zeros((0, 3), dtype=float)
    else:
        monomer_zyx_bin2 = np.asarray(monomer_zyx_bin4, dtype=float) * BIN_SCALE

    for seed_zyx_bin4 in dimer_zyx_bin4:
        seed_zyx_bin2 = seed_zyx_bin4 * BIN_SCALE

        patch, origin = extract_patch(
            vol_bin2,
            seed_zyx_bin2,
            DIMER_SEARCH_RADIUS_BIN2
        )

        patch_masked = mask_positions_in_patch(
            patch,
            origin,
            monomer_zyx_bin2,
            radius=MONOMER_MASK_RADIUS_BIN2
        )

        result = find_two_darkest_local_minima(patch_masked, origin)

        if result is None:
            continue

        c1_bin2, c2_bin2 = result

        c1_bin4 = c1_bin2 / BIN_SCALE
        c2_bin4 = c2_bin2 / BIN_SCALE

        pairs_zyx_bin4.append([c1_bin4, c2_bin4])

    if len(pairs_zyx_bin4) == 0:
        return np.zeros((0, 2, 3), dtype=float)

    return np.asarray(pairs_zyx_bin4, dtype=float)


# ===================== DUPLICATE PROOFREADING =====================

def flatten_pairs_with_indices(pairs_zyx_bin4):
    records = []

    for pair_idx in range(len(pairs_zyx_bin4)):
        for side_idx in range(2):
            coord = pairs_zyx_bin4[pair_idx, side_idx]
            partner = pairs_zyx_bin4[pair_idx, 1 - side_idx]
            dist_to_partner = np.linalg.norm(coord - partner)

            records.append({
                "pair_idx": pair_idx,
                "side_idx": side_idx,
                "coord": coord,
                "partner": partner,
                "dist_to_partner": dist_to_partner,
            })

    return records


def find_duplicate_groups(records, threshold=DUPLICATE_DISTANCE_BIN4):
    n = len(records)
    visited = set()
    groups = []

    for i in range(n):
        if i in visited:
            continue

        group = [i]
        visited.add(i)

        changed = True
        while changed:
            changed = False
            for j in range(n):
                if j in visited:
                    continue

                for k in group:
                    d = np.linalg.norm(records[j]["coord"] - records[k]["coord"])
                    if d <= threshold:
                        group.append(j)
                        visited.add(j)
                        changed = True
                        break

        if len(group) > 1:
            groups.append(group)

    return groups


def redetect_one_side_with_masks(
    vol_bin2,
    seed_zyx_bin4,
    fixed_partner_zyx_bin4,
    monomer_zyx_bin4,
    occupied_zyx_bin4,
):
    """
    Re-detect only one AuNP of a dimer.
    The fixed partner is kept.
    Monomers and already occupied eachdimer positions are masked.
    """
    seed_zyx_bin2 = seed_zyx_bin4 * BIN_SCALE
    fixed_partner_zyx_bin2 = fixed_partner_zyx_bin4 * BIN_SCALE

    if monomer_zyx_bin4 is None or monomer_zyx_bin4.size == 0:
        monomer_zyx_bin2 = np.zeros((0, 3), dtype=float)
    else:
        monomer_zyx_bin2 = np.asarray(monomer_zyx_bin4, dtype=float) * BIN_SCALE

    if occupied_zyx_bin4 is None or occupied_zyx_bin4.size == 0:
        occupied_zyx_bin2 = np.zeros((0, 3), dtype=float)
    else:
        occupied_zyx_bin2 = np.asarray(occupied_zyx_bin4, dtype=float) * BIN_SCALE

    all_mask_zyx_bin2 = np.vstack([
        monomer_zyx_bin2.reshape(-1, 3),
        occupied_zyx_bin2.reshape(-1, 3),
    ])

    patch, origin = extract_patch(
        vol_bin2,
        seed_zyx_bin2,
        DIMER_SEARCH_RADIUS_BIN2
    )

    patch_masked = mask_positions_in_patch(
        patch,
        origin,
        all_mask_zyx_bin2,
        radius=MONOMER_MASK_RADIUS_BIN2
    )

    new_coord_bin2 = find_one_new_partner_minimum(
        patch_masked,
        origin,
        fixed_partner_zyx_bin2
    )

    if new_coord_bin2 is None:
        return None

    return new_coord_bin2 / BIN_SCALE


def proofread_duplicate_eachdimer_pairs(
    pairs_zyx_bin4,
    dimer_zyx_bin4,
    monomer_zyx_bin4,
    vol_bin2,
):
    """
    If duplicate eachdimer coordinates are detected:
      1. Compare each duplicate coordinate distance to its own pair partner.
      2. The coordinate with shorter partner distance is treated as real.
      3. The loser coordinate is removed and re-detected.
      4. During re-detection, monomers and already-used eachdimer coordinates
         are masked.
    """
    if pairs_zyx_bin4.size == 0:
        return pairs_zyx_bin4

    pairs = pairs_zyx_bin4.copy()

    for iteration in range(MAX_DUPLICATE_FIX_ITER):
        records = flatten_pairs_with_indices(pairs)
        duplicate_groups = find_duplicate_groups(
            records,
            threshold=DUPLICATE_DISTANCE_BIN4
        )

        if len(duplicate_groups) == 0:
            print(f"No duplicate eachdimer coordinates detected after iteration {iteration}.")
            break

        print(f"[Duplicate proofreading] Iteration {iteration + 1}")
        print(f"Duplicate groups found: {len(duplicate_groups)}")

        fixed_any = False

        for group in duplicate_groups:
            group_records = [records[i] for i in group]

            # Smaller distance to its own partner is considered more realistic.
            group_records_sorted = sorted(
                group_records,
                key=lambda r: r["dist_to_partner"]
            )

            winner = group_records_sorted[0]
            losers = group_records_sorted[1:]

            print(
                f"  Winner: pair {winner['pair_idx']} side {winner['side_idx']} "
                f"partner_dist={winner['dist_to_partner']:.2f}"
            )

            for loser in losers:
                pair_idx = loser["pair_idx"]
                side_idx = loser["side_idx"]
                fixed_partner = pairs[pair_idx, 1 - side_idx]
                seed = dimer_zyx_bin4[pair_idx]

                occupied = []

                for pi in range(len(pairs)):
                    for si in range(2):
                        if pi == pair_idx and si == side_idx:
                            continue

                        occupied.append(pairs[pi, si])

                occupied = np.asarray(occupied, dtype=float)

                new_coord = redetect_one_side_with_masks(
                    vol_bin2=vol_bin2,
                    seed_zyx_bin4=seed,
                    fixed_partner_zyx_bin4=fixed_partner,
                    monomer_zyx_bin4=monomer_zyx_bin4,
                    occupied_zyx_bin4=occupied,
                )

                if new_coord is None:
                    print(
                        f"  [WARNING] Could not re-detect pair {pair_idx} side {side_idx}. "
                        f"Keeping original coordinate."
                    )
                    continue

                old_coord = pairs[pair_idx, side_idx].copy()
                pairs[pair_idx, side_idx] = new_coord
                fixed_any = True

                print(
                    f"  Re-detected pair {pair_idx} side {side_idx}: "
                    f"old zyx={old_coord} -> new zyx={new_coord}"
                )

        if not fixed_any:
            print("[WARNING] Duplicate proofreading could not fix any coordinate.")
            break

    return pairs


# ===================== GAUSSIAN FITTING =====================

def gaussian_3d_negative(params, zz, yy, xx):
    z0, y0, x0, amp, sigma_z, sigma_y, sigma_x, offset = params

    exp_term = (
        ((zz - z0) ** 2) / (2 * sigma_z ** 2) +
        ((yy - y0) ** 2) / (2 * sigma_y ** 2) +
        ((xx - x0) ** 2) / (2 * sigma_x ** 2)
    )

    return offset - amp * np.exp(-exp_term)


def fit_one_gaussian_center(vol, center_zyx, radius=GAUSS_RADIUS_BIN4):
    patch, origin = extract_patch(vol, center_zyx, radius)

    if patch.size == 0:
        return center_zyx

    Z, Y, X = patch.shape

    zz, yy, xx = np.meshgrid(
        np.arange(Z),
        np.arange(Y),
        np.arange(X),
        indexing="ij"
    )

    data = patch.astype(float)

    min_idx = np.unravel_index(np.argmin(data), data.shape)

    z0_init, y0_init, x0_init = min_idx
    offset_init = np.median(data)
    amp_init = max(offset_init - np.min(data), 1e-6)

    p0 = np.array([
        z0_init,
        y0_init,
        x0_init,
        amp_init,
        2.0,
        2.0,
        2.0,
        offset_init
    ], dtype=float)

    lower = np.array([
        0, 0, 0,
        0,
        0.5, 0.5, 0.5,
        np.min(data) - abs(np.min(data)) - 10
    ], dtype=float)

    upper = np.array([
        Z - 1, Y - 1, X - 1,
        np.inf,
        4.0, 4.0, 4.0,
        np.max(data) + abs(np.max(data)) + 10
    ], dtype=float)

    def residual(params):
        model = gaussian_3d_negative(params, zz, yy, xx)
        return (model - data).ravel()

    try:
        res = least_squares(
            residual,
            p0,
            bounds=(lower, upper),
            max_nfev=300
        )

        z_fit, y_fit, x_fit = res.x[:3]
        refined_zyx = origin + np.array([z_fit, y_fit, x_fit], dtype=float)

        return refined_zyx

    except Exception:
        return center_zyx


def gaussian_refine_xyz_file(vol_bin4, input_txt, output_txt):
    coords_xyz = load_xyz_txt(input_txt)

    if coords_xyz.size == 0:
        empty = np.zeros((0, 3), dtype=float)
        save_xyz_txt(empty, output_txt)
        return empty

    refined_xyz = []

    for xyz in coords_xyz:
        x, y, z = xyz
        center_zyx = np.array([z, y, x], dtype=float)

        refined_zyx = fit_one_gaussian_center(vol_bin4, center_zyx)

        rz, ry, rx = refined_zyx
        refined_xyz.append([rx, ry, rz])

    refined_xyz = np.asarray(refined_xyz, dtype=float)
    save_xyz_txt(refined_xyz, output_txt)

    return refined_xyz


# ===================== MIDPOINT =====================

def make_midpoints_from_eachdimer_pairs(eachdimer_xyz):
    if eachdimer_xyz.size == 0:
        return np.zeros((0, 3), dtype=float)

    midpoints = []
    n = len(eachdimer_xyz)

    for i in range(0, n - 1, 2):
        p1 = eachdimer_xyz[i]
        p2 = eachdimer_xyz[i + 1]
        mid = (p1 + p2) / 2.0
        midpoints.append(mid)

    if len(midpoints) == 0:
        return np.zeros((0, 3), dtype=float)

    return np.asarray(midpoints, dtype=float)


# ===================== MAIN PIPELINE =====================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Split STAR into monomer/dimer TXT, mask monomers in BIN2, "
            "detect eachdimer, proofread duplicated eachdimer coordinates, "
            "Gaussian refine in BIN4, and generate midpoint coordinates."
        )
    )

    parser.add_argument("--star", required=True, help="Input STAR file. First 4 columns are treated as z y x type.")
    parser.add_argument("--bin2", required=True, help="BIN2 tomogram MRC file.")
    parser.add_argument("--bin4", required=True, help="BIN4 tomogram MRC file.")
    parser.add_argument("--dir", required=True, help="Output directory.")

    args = parser.parse_args()

    star_path = Path(args.star)
    out_dir = Path(args.dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prefix = star_path.stem

    monomer_txt = out_dir / f"{prefix}_monomer.txt"
    dimer_txt = out_dir / f"{prefix}_dimer.txt"
    eachdimer_txt = out_dir / f"{prefix}_eachdimer.txt"

    monomer_gau_txt = out_dir / f"{prefix}_monomer_gau.txt"
    eachdimer_gau_txt = out_dir / f"{prefix}_eachdimer_gau.txt"
    midpoint_txt = out_dir / f"{prefix}_midpoint_gau.txt"

    print("Loading STAR...")
    df = load_star_as_zyx_type(star_path)

    monomer_df = df[df["type"] == "monomer"].copy()
    dimer_df = df[df["type"] == "dimer"].copy()

    monomer_xyz = monomer_df[["x", "y", "z"]].to_numpy(dtype=float)
    dimer_xyz = dimer_df[["x", "y", "z"]].to_numpy(dtype=float)

    save_xyz_txt(monomer_xyz, monomer_txt)
    save_xyz_txt(dimer_xyz, dimer_txt)

    print(f"Saved monomer TXT: {monomer_txt}")
    print(f"Saved dimer TXT:   {dimer_txt}")
    print(f"Monomer count: {len(monomer_xyz)}")
    print(f"Dimer seed count: {len(dimer_xyz)}")

    print("Loading BIN2 tomogram...")
    vol_bin2 = load_mrc(args.bin2)

    monomer_zyx_bin4 = monomer_df[["z", "y", "x"]].to_numpy(dtype=float)
    dimer_zyx_bin4 = dimer_df[["z", "y", "x"]].to_numpy(dtype=float)

    print("Detecting eachdimer AuNPs from BIN2 with monomer masking...")
    print(f"Monomer mask radius in BIN2 voxels: {MONOMER_MASK_RADIUS_BIN2}")

    eachdimer_pairs_zyx_bin4 = detect_eachdimer_pairs_from_bin2(
        vol_bin2,
        dimer_zyx_bin4,
        monomer_zyx_bin4
    )

    print(f"Initial detected dimer pairs: {len(eachdimer_pairs_zyx_bin4)}")
    print(f"Initial eachdimer count: {len(eachdimer_pairs_zyx_bin4) * 2}")

    print("Proofreading duplicate eachdimer coordinates...")
    print(f"Duplicate threshold in BIN4 voxels: {DUPLICATE_DISTANCE_BIN4}")

    eachdimer_pairs_zyx_bin4 = proofread_duplicate_eachdimer_pairs(
        pairs_zyx_bin4=eachdimer_pairs_zyx_bin4,
        dimer_zyx_bin4=dimer_zyx_bin4,
        monomer_zyx_bin4=monomer_zyx_bin4,
        vol_bin2=vol_bin2,
    )

    if eachdimer_pairs_zyx_bin4.size == 0:
        eachdimer_zyx_bin4 = np.zeros((0, 3), dtype=float)
    else:
        eachdimer_zyx_bin4 = eachdimer_pairs_zyx_bin4.reshape(-1, 3)

    eachdimer_xyz_bin4 = eachdimer_zyx_bin4[:, [2, 1, 0]]
    save_xyz_txt(eachdimer_xyz_bin4, eachdimer_txt)

    print(f"Saved eachdimer TXT: {eachdimer_txt}")
    print(f"Eachdimer count after proofreading: {len(eachdimer_xyz_bin4)}")

    print("Loading BIN4 tomogram...")
    vol_bin4 = load_mrc(args.bin4)

    print("Gaussian refining monomer coordinates in BIN4...")
    monomer_gau_xyz = gaussian_refine_xyz_file(
        vol_bin4,
        monomer_txt,
        monomer_gau_txt
    )

    print(f"Saved Gaussian monomer TXT: {monomer_gau_txt}")
    print(f"Gaussian monomer count: {len(monomer_gau_xyz)}")

    print("Gaussian refining eachdimer coordinates in BIN4...")
    eachdimer_gau_xyz = gaussian_refine_xyz_file(
        vol_bin4,
        eachdimer_txt,
        eachdimer_gau_txt
    )

    print(f"Saved Gaussian eachdimer TXT: {eachdimer_gau_txt}")
    print(f"Gaussian eachdimer count: {len(eachdimer_gau_xyz)}")

    print("Generating midpoint coordinates from eachdimer_gau...")
    midpoint_xyz = make_midpoints_from_eachdimer_pairs(eachdimer_gau_xyz)
    save_xyz_txt(midpoint_xyz, midpoint_txt)

    print(f"Saved midpoint TXT: {midpoint_txt}")
    print(f"Midpoint count: {len(midpoint_xyz)}")

    print("Done.")


if __name__ == "__main__":
    main()