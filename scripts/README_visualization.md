# Synaptic Tomogram Visualization Script

This script generates 2D visualizations for analyzed synaptic tomograms.

## Features

- **Two Image Versions**: Generates two separate visualizations per tomogram:
  - **Vesicles and Active Zones**: Shows vesicles, active zones, and fusion sites
  - **Vesicles and AuNPs**: Shows vesicles and gold nanoparticles

- **Vesicles**: 
  - All vesicles with any point in the slice (pink circles)
  - Vesicles within 10nm of active zone highlighted (aqua circles)

- **Active Zones**:
  - Presynaptic active zone (transparent red)
  - Postsynaptic active zone (transparent green)
  - Only segments within ±1 pixel of slice center

- **Fusion Sites**: 
  - Putative fusion sites for vesicles within 10nm (orange stars)
  - Only shown in vesicles and active zones image

- **AuNPs**: 
  - Gold nanoparticles used for analyses (active_zone != -1)
  - Only those within ±5 pixels of slice center
  - Only shown in vesicles and AuNPs image

- **Contextual Filtering**: Only objects near the shown tomogram slice are displayed

## Usage

### Basic Usage
```bash
python visualize_synaptic_tomo_results.py
```

### Advanced Usage
```bash
# Specify custom data and output directories
python visualize_synaptic_tomo_results.py --data-dir /path/to/data --output-dir /path/to/output

# Process only a specific tomogram
python visualize_synaptic_tomo_results.py --tomo-name 20231026_HippAu_26

# Combine options
python visualize_synaptic_tomo_results.py --data-dir ../data/ --output-dir ./figures --tomo-name 20231026_HippAu_26
```

## Command Line Options

- `--data-dir`: Path to data directory (default: `../data/`)
- `--output-dir`: Output directory for figures (default: `./visualization_output`)
- `--tomo-name`: Process only specific tomogram (optional)

## Output

The script generates two PNG files per tomogram:
- `{tomogram_name}_vesicles_active_zones.png`: Vesicles, active zones, and fusion sites
- `{tomogram_name}_vesicles_aunps.png`: Vesicles and AuNPs

## Filtering Details

- **Z-coordinate filtering**: 
  - Vesicles and AuNPs: ±5 pixels from slice center
  - Active zones: ±1 pixel from slice center
- **AuNP filtering**: Only those with `active_zone != -1` (used for analyses)
- **Fusion sites**: Computed for vesicles within 10nm of active zone

## Requirements

- numpy
- pandas
- matplotlib
- pathlib
- scipy (for fusion point computation)
- mrcfile (optional, for tomogram slice loading)
- starfile (optional, for AuNP loading)

## Notes

- The script automatically finds all analyzed tomograms in the data directory
- Figures are saved at 300 DPI for high quality
- The script handles missing data gracefully and continues processing other tomograms
- NumPy version warnings can be ignored (they don't affect functionality)
- 3D visualization has been removed; only 2D overlays are produced
- Membrane segmentations are no longer displayed to reduce visual clutter 