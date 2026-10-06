# Post-to-Pre Membrane Distance Map (MRC)

Builds a 3D distance-map volume from pre- and post-synaptic membrane point clouds. For every post-membrane point, the nearest 3D distance to the pre-membrane is computed, and the values are written into an MRC volume on the **same grid** (shape, voxel size, origin) as a reference tomogram.

The output can be opened in ChimeraX together with a membrane surface model, and the distances can be painted onto the surface with `color sample`.

---

## Overview

1. Load pre- and post-membrane point clouds (voxel coordinates).
2. Optionally correct the coordinate convention (centered coordinates, axis flips, manual shifts).
3. Compute the nearest 3D Euclidean distance from every post point to the pre point cloud.
4. Optionally drop post points that are too far from the pre-membrane (`--max_dist`) and small disconnected fragments (cluster filtering).
5. Splat the distance values onto a 3D grid and apply a 3D Gaussian blur (blurred average), so voxels near the post-membrane receive smoothly interpolated values.
6. Save the volume as an MRC file that inherits the reference tomogram's voxel size and origin.

---

## Requirements

- Python 3.8+
- `numpy`
- `scipy`
- `mrcfile`

```bash
pip install numpy scipy mrcfile
```

---

## Usage

Minimal example:

```bash
python make_distance_mrc.py \
    --pre  pre_membrane.txt \
    --post post_membrane.txt \
    --reference-mrc tomogram_bin4.mrc \
    --out-mrc distance_map.mrc
```

With coordinate correction and fragment removal:

```bash
python make_distance_mrc.py \
    --pre  pre_membrane.txt \
    --post post_membrane.txt \
    --reference-mrc tomogram_bin4.mrc \
    --out-mrc distance_map.mrc \
    --sigma 3.0 \
    --max_dist 40 \
    --coords_are_centered \
    --flip_z \
    --keep_largest_only
```

---

## Input

| Argument | Required | Description |
|---|---|---|
| `--pre` | yes | Pre-membrane coordinates, TXT, one point per line: `x y z` (voxel units). |
| `--post` | yes | Post-membrane coordinates, TXT, one point per line: `x y z` (voxel units). |
| `--reference-mrc` | yes | Tomogram MRC whose shape, voxel size and origin are copied to the output. |
| `--out-mrc` | yes | Output distance-map MRC path. |

TXT files are whitespace-separated. Empty lines, lines with fewer than three columns, and non-numeric lines are skipped; only the first three columns are used.

Coordinates are assumed to be in the **voxel units of the reference MRC**, so the reference tomogram must have the same binning as the coordinates.

---

## Output

A single `float32` MRC file with the same shape `(nz, ny, nx)` as the reference tomogram.

- Voxels near the post-membrane contain the (blurred) nearest distance to the pre-membrane, in **voxels**.
- Voxels with no nearby post-membrane coverage are set to `--empty_value` (default `NaN`).
- Voxel size, origin, and start indices are copied from the reference MRC.
- Header statistics (`dmin`, `dmax`, `dmean`) are computed manually from finite voxels only, because the default `update_header_stats()` does not handle NaN correctly and can make ChimeraX choose a meaningless display range.

To convert distances to nanometers, multiply by the voxel size (in nm) of the reference tomogram.

---

## Options

### Volume construction

| Option | Default | Description |
|---|---|---|
| `--sigma` | `3.0` | 3D Gaussian blur sigma (voxels). Larger values give smoother, more filled-in maps. |
| `--percentile_clip` | `99.0` | Clip high distance values at this percentile. Use a negative value to disable. |
| `--empty_value` | `NaN` | Value for voxels without coverage. Pass a finite number (e.g. `-1`) if a finite sentinel is needed. |

### Distance filtering

| Option | Default | Description |
|---|---|---|
| `--max_dist` | `40.0` | Keep only post points whose nearest pre distance is at most this value (voxels). Farther points are removed before splatting. Use a negative value to disable. |

### Fragment removal

| Option | Default | Description |
|---|---|---|
| `--cluster_radius` | `5.0` | Two post points closer than this (voxels) are considered connected. Clusters are the connected components of this graph. |
| `--min_cluster_size` | `0` | Drop clusters with fewer points than this. `0` disables the filter. Values of 20 to 50 are a reasonable starting point. |
| `--keep_largest_only` | off | Keep only the single largest cluster and drop all others. Overrides `--min_cluster_size`. |

The number of clusters and their size range are printed to the log, which helps in choosing `--cluster_radius` and `--min_cluster_size`.

### Coordinate convention correction

These options are applied in the order listed below, and are applied identically to the pre and post points, so distances are unchanged and only the position within the volume changes.

| Option | Description |
|---|---|
| `--coords_are_centered` | Use if coordinates are relative to the tomogram center (`(0,0,0)` = volume center, values may be negative). Shifts coordinates by `+size/2`. |
| `--flip_x` / `--flip_y` / `--flip_z` | Mirror an axis (`c -> n-1-c`). Useful when the coordinates and the MRC disagree on axis direction or Z-flip convention. |
| `--shift_x` / `--shift_y` / `--shift_z` | Add a manual translation (voxels, may be negative) to all coordinates. |

If the distance map does not line up with the membrane model in ChimeraX, try these options one at a time.

---

## Algorithm Details

### Distance computation

A `cKDTree` is built from the pre-membrane points, and the nearest neighbor of every post point is queried. Every post point is used (there is no subsampling).

### Splatting and blurred average

1. Post points are rounded to integer voxel indices. Points outside the volume or with non-finite distance are discarded.
2. Two volumes are accumulated: the sum of distance values and the point count per voxel.
3. Both are blurred with a 3D Gaussian (`scipy.ndimage.gaussian_filter`).
4. The output is `sum_blur / count_blur`, a smooth weighted average of the nearby distances.
5. Voxels where the blurred count is effectively zero are set to `--empty_value`.

### Cluster filtering

Post points are linked when they are within `--cluster_radius` of each other (`cKDTree.query_pairs`). Connected components of this graph (`scipy.sparse.csgraph.connected_components`) are the clusters. Small or non-largest clusters are removed, which eliminates floating mesh fragments.

---

## ChimeraX Usage

```
open tomogram_bin4.mrc
open distance_map.mrc
# Open or build the membrane surface model, e.g. as #1, with the distance map as #3
color sample #1 map #3 palette blue:white:red range 0,30
```

`color sample` reads the map value at each surface vertex position and colors the vertex accordingly. Vertices that fall on `NaN` voxels are typically left uncolored.

---

## Notes and Limitations

- The filled region is a shell around the post-membrane, not a thin sheet. With `gaussian_filter`'s default truncation (4 sigma), voxels within roughly `4 * sigma` of a post point receive a value. Surface vertices farther than this from any post point will sample `NaN`.
- Blurring averages the distances of nearby points, so regions where the membrane is strongly curved or folded can mix different distances.
- Input coordinates are treated as 0-based array indices. Any start-index offset in the reference MRC header is copied to the output but is not applied to the input coordinates.
- Memory usage scales with the volume size (several full-size float32 arrays are held at once). Use a binned tomogram as the reference if memory is limited.
- Distances are in voxels, not physical units.
