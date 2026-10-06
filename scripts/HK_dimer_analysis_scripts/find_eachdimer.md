# AuNP Monomer/Dimer Coordinate Refinement

Detects the individual gold nanoparticles (AuNPs) that make up each dimer, refines all AuNP coordinates to sub-voxel precision with 3D Gaussian fitting, and computes the midpoint of each dimer.

---

## Overview

Given a STAR file of manually or automatically picked **monomer** and **dimer** seed points, the script:

1. Splits the seeds into monomers and dimers.
2. For each dimer seed, finds the two AuNPs in the **BIN2** tomogram (the two darkest, well-separated local minima around the seed), while masking known monomer positions.
3. Proofreads duplicated AuNP assignments (the same AuNP claimed by two different dimer seeds) and re-detects the losing side.
4. Refines monomer and dimer-AuNP coordinates by fitting a 3D Gaussian in the **BIN4** tomogram.
5. Computes the midpoint of each AuNP pair.

AuNPs appear **dark** (high electron density) in the tomograms, so the script searches for local **minima**.

---

## Requirements

- Python 3.8+
- `numpy`
- `pandas`
- `scipy`
- `mrcfile`
- `starfile`

```bash
pip install numpy pandas scipy mrcfile starfile
```

---

## Usage

```bash
python find_eachdimer.py \
    --star tomo01.star \
    --bin2 tomo01_bin2.mrc \
    --bin4 tomo01_bin4.mrc \
    --dir  output/
```

### Arguments

| Argument | Required | Description |
|---|---|---|
| `--star` | yes | Input STAR file (see format below). |
| `--bin2` | yes | BIN2 tomogram (`.mrc`), used for dimer AuNP detection. |
| `--bin4` | yes | BIN4 tomogram (`.mrc`), used for Gaussian refinement. |
| `--dir`  | yes | Output directory (created if it does not exist). |

---

## Input

### STAR file

The **first four columns** are read by position (column names are ignored) and interpreted as:

| Column | Meaning |
|---|---|
| 1 | `z` coordinate |
| 2 | `y` coordinate |
| 3 | `x` coordinate |
| 4 | `type`, either `monomer` or `dimer` |

- Coordinates must be in **BIN4 voxel units**.
- Each `dimer` row is a single seed point near a dimer (one seed per dimer).
- Rows with any other `type` value are ignored.

### Tomograms

- `--bin2`: tomogram binned by 2. Seed coordinates are multiplied by `BIN_SCALE = 2.0` to map from BIN4 to BIN2.
- `--bin4`: tomogram binned by 4, in the same coordinate frame as the STAR coordinates.

---

## Output

All files are written to `--dir`, prefixed with the STAR file name (`{prefix}` = STAR file stem). Every file is tab-separated with six decimals, in **x, y, z** order (note: reversed relative to the STAR input, which is z, y, x). Coordinates are in BIN4 voxel units.

| File | Content | Refinement |
|---|---|---|
| `{prefix}_monomer.txt` | Monomer coordinates from the STAR file | none |
| `{prefix}_dimer.txt` | Dimer seed coordinates from the STAR file (one per dimer) | none |
| `{prefix}_eachdimer.txt` | The two AuNPs detected per dimer, after duplicate proofreading | BIN2 local minimum |
| `{prefix}_monomer_gau.txt` | Monomer coordinates after Gaussian fitting | sub-voxel |
| `{prefix}_eachdimer_gau.txt` | Dimer AuNP coordinates after Gaussian fitting | sub-voxel |
| `{prefix}_midpoint_gau.txt` | Midpoint of each AuNP pair | sub-voxel |

### Row correspondence

- For *N* dimer seeds, `eachdimer*.txt` contains up to **2N** rows. **Consecutive rows (1–2, 3–4, ...) form one dimer pair.**
- `midpoint_gau.txt` contains one row per pair (up to *N*).
- If two sufficiently separated minima cannot be found around a seed, that dimer is skipped, so the actual counts may be lower than *N*.

---

## Algorithm Details

### 1. Dimer AuNP detection (BIN2)

- Extract a cubic patch of radius `DIMER_SEARCH_RADIUS_BIN2` (10 voxels) around each dimer seed.
- Mask monomer positions (spheres of radius `MONOMER_MASK_RADIUS_BIN2`, 4 voxels) by replacing them with a very bright value so they cannot be selected as minima.
- Find all local minima (a voxel that is less than or equal to every voxel in a `(2r+1)^3` neighborhood, `r = LOCAL_MIN_RADIUS = 2`) and sort them by intensity.
- The darkest minimum is the first AuNP. The second AuNP is the darkest remaining minimum at least `MIN_SEPARATION_BIN2` (4 voxels) away from the first.

### 2. Duplicate proofreading

Two different dimer seeds can end up selecting the same AuNP. The script:

1. Groups all detected AuNP coordinates that lie within `DUPLICATE_DISTANCE_BIN4` (2.0 BIN4 voxels) of each other.
2. Within each group, the coordinate whose distance to its own pair partner is **shortest** is treated as the true assignment (winner).
3. Each loser is removed and re-detected: its partner is kept fixed, and monomers plus all other already-used AuNP positions are masked.
4. Steps 1–3 repeat for up to `MAX_DUPLICATE_FIX_ITER` (5) iterations or until no duplicates remain.

If re-detection fails for a coordinate, the original coordinate is kept and a warning is printed.

### 3. Gaussian refinement (BIN4)

Each coordinate is refined by fitting an inverted 3D Gaussian

```
I(z, y, x) = offset - amp * exp( -[ (z-z0)^2/(2σz^2) + (y-y0)^2/(2σy^2) + (x-x0)^2/(2σx^2) ] )
```

to a patch of radius `GAUSS_RADIUS_BIN4` (1 voxel) around the input coordinate, using `scipy.optimize.least_squares` (bounded, max 300 evaluations). The fitted center `(z0, y0, x0)` replaces the input coordinate. If the fit fails, the input coordinate is returned unchanged.

### 4. Midpoints

For each consecutive pair of rows in `eachdimer_gau`, the midpoint is the mean of the two coordinates.

---

## Parameters

Defined as constants at the top of the script:

| Constant | Default | Description |
|---|---|---|
| `BIN_SCALE` | `2.0` | Scale factor from BIN4 to BIN2 coordinates |
| `DIMER_SEARCH_RADIUS_BIN2` | `10` | Patch radius around each dimer seed (BIN2 voxels) |
| `LOCAL_MIN_RADIUS` | `2` | Neighborhood radius for local minimum detection |
| `MIN_SEPARATION_BIN2` | `4` | Minimum distance between the two AuNPs of a dimer (BIN2 voxels) |
| `MONOMER_MASK_RADIUS_BIN2` | `4` | Radius of the mask applied to known positions (BIN2 voxels) |
| `DUPLICATE_DISTANCE_BIN4` | `2.0` | Distance below which two AuNP coordinates are duplicates (BIN4 voxels) |
| `MAX_DUPLICATE_FIX_ITER` | `5` | Maximum proofreading iterations |
| `GAUSS_RADIUS_BIN4` | `1` | Patch radius for Gaussian fitting (BIN4 voxels) |

Parameters may need to be adjusted for different pixel sizes, particle sizes, or binning factors.

---

## Notes and Limitations

- Coordinate convention: **input STAR is z, y, x; output TXT is x, y, z.**
- The script assumes the BIN4 tomogram and STAR coordinates share the same coordinate frame, and that BIN2 coordinates are exactly 2x the BIN4 coordinates.
- Dimer detection is seed-based: the quality of the result depends on the seed being close to the dimer (within the search radius).
- Pair order matters: the midpoint calculation relies on consecutive rows of `eachdimer_gau.txt` being a pair.
- Local minimum search is implemented with pure Python loops and can be slow for large numbers of seeds.
