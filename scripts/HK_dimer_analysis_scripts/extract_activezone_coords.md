# Extract Active Zone Coordinates

Maps monomer/dimer particle coordinates from the tomogram coordinate system into each **active zonogram** (the unfolded view of the active zone), keeps only the particles that fall on valid zonogram data, and writes per-active-zone coordinate files, STAR files, and overview PNGs.

---

## Overview

Given one STAR file with all monomer/dimer coordinates of a tomogram and a set of active zonograms (`.mrc` + `.npy` pairs), the script:

1. Reads monomer and dimer coordinates from the STAR file.
2. Transforms the coordinates into each zonogram's coordinate space using the zonogram's `center` and rotation matrix `cs`.
3. Removes particles that fall outside the zonogram volume or on empty (zero-valued) regions of the zonogram.
4. Saves the remaining particles in both the original tomogram coordinates and the zonogram coordinates.
5. Writes a STAR file per active zone and, optionally, PNGs of the zonogram with the particles marked.

---

## Requirements

- Python 3.8+
- `numpy`
- `mrcfile`
- `torch`
- `matplotlib`

```bash
pip install numpy mrcfile torch matplotlib
```

---

## Usage

```bash
python extract_activezone_coords.py \
    --dir  /path/to/root \
    --star /path/to/global_particles.star
```

With options:

```bash
python extract_activezone_coords.py \
    --dir  /path/to/root \
    --star global_particles.star \
    --out  active_zonograms/dual \
    --margin 2 \
    --zero_patch_radius 1 \
    --zero_patch_round round \
    --no_png
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--dir` | required | Root directory. Must contain an `active_zonograms/` folder. |
| `--star` | required | Input STAR file. Absolute path, or relative to `--dir`. |
| `--out` | `active_zonograms/dual` | Output directory. Absolute path, or relative to `--dir`. |
| `--margin` | `0.0` | Particles closer than this many voxels to the zonogram boundary are removed. |
| `--zero_patch_radius` | `1` | Half-size (voxels) of the XY patch used by the zero-patch filter (`1` = 3x3 patch). |
| `--zero_patch_round` | `round` | How transformed coordinates are converted to voxel indices: `round`, `floor`, or `ceil`. |
| `--no_png` | off | Skip PNG generation. |
| `--quiet` | off | Suppress progress messages. |

---

## Input

### Directory layout

```
root/
├── active_zonograms/
│   ├── active_zonogram_0.mrc
│   ├── active_zonogram_0.npy
│   ├── active_zonogram_1.mrc
│   ├── active_zonogram_1.npy
│   └── ...
└── global_particles.star
```

Only indices for which **both** the `.mrc` and the `.npy` exist are processed.

### STAR file

Rows are read by position (header, comment, and `loop_` lines are skipped):

| Column | Meaning |
|---|---|
| 1 | `z` coordinate |
| 2 | `y` coordinate |
| 3 | `x` coordinate |
| 4 | `type`, either `monomer` or `dimer` (case-insensitive) |

Rows with any other type are ignored. Coordinates are in the voxel units of the tomogram that the zonograms were generated from.

### Active zonogram files

- `active_zonogram_{i}.mrc`: the zonogram volume. Its shape defines the valid region, and zero-valued voxels are treated as "no data".
- `active_zonogram_{i}.npy`: a pickled dictionary with two keys:
  - `center`: the center of the zonogram in tomogram coordinates (x, y, z)
  - `cs`: the 3x3 rotation matrix

---

## Output

All files are written to `--out`. Text coordinate files are in **x y z** order (three decimals). STAR files are written in **z y x type** order, the same convention as the input.

The numbering uses `k = i + 1`, where `i` is the index in `active_zonogram_{i}`. For example, `active_zonogram_0` produces files ending in `_1`.

| File | Content |
|---|---|
| `monomer.txt`, `dimer.txt` | All monomer / dimer coordinates from the STAR file (x y z), unfiltered. |
| `filtered_monomer_{k}.txt`, `filtered_dimer_{k}.txt` | Particles kept for this zonogram, in **original tomogram coordinates**. |
| `extracted_m_{k}.txt`, `extracted_d_{k}.txt` | The same particles in **zonogram coordinates**. Row order matches the filtered files. |
| `activezone_{k}.star` | The kept particles as a STAR file (`z y x type`), monomers first and then dimers. |
| `png/active_zonogram_{i}.png` | Zonogram projections without annotation. |
| `png/active_zonogram_{i}_annotated.png` | Zonogram projections with monomers (red circles) and dimers (blue circles) and their counts. |

The `activezone_{k}.star` files can be used directly as input for downstream comparison and analysis scripts.

---

## Algorithm Details

### Coordinate transform

For each zonogram:

```
zonogram_xyz = (tomogram_xyz - center) · csᵀ + size_xyz / 2
```

where `size_xyz` is the zonogram volume size in (x, y, z) voxels.

### Filtering

A particle is kept only if it passes both filters.

1. **Bounds filter.** The transformed coordinate must be inside the volume: `margin <= coordinate < size - margin` on every axis.
2. **Zero-patch filter.** The transformed coordinate is converted to a voxel index (`--zero_patch_round`). The particle is removed if:
   - the voxel (or its XY patch) falls outside the volume, or
   - the center voxel is exactly `0` and the whole `(2r+1) x (2r+1)` XY patch in the same Z slice is also `0` (`r = --zero_patch_radius`).

   This removes particles that map onto regions of the zonogram that contain no data.

### PNGs

Each PNG shows three minimum-intensity projections of the zonogram (XY, ZY, XZ) with a gray colormap, as used for display of dark-density features. Particle positions are overlaid as open circles on all three views.

---

## Notes and Limitations

- A particle can appear in several zonograms if their regions overlap; each zonogram gets its own output files.
- Coordinates written to the STAR files come from the 3-decimal text files, so their effective precision is three decimals.
- The zero-patch filter only inspects a 2D patch in a single Z slice and only removes particles whose center voxel is exactly zero.
- The zonogram `.mrc` is read several times per zonogram, which can be slow for large volumes.
- Index convention: zonogram files are numbered from 0, whereas output files (`activezone_{k}`) are numbered from 1.
