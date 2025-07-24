"""
Multi-Step Extracellular Space Classification

This script uses a three-step approach to classify extracellular vs intracellular space:
1. Identify vesicles and organelles (small, enclosed regions)
2. Use remaining membrane structure to define cell boundaries  
3. Classify remaining space as intracellular vs extracellular

Requirements:
    pip install mrcfile scipy scikit-image numpy matplotlib

Usage:
    python scripts/watershed_extracellular.py <tomogram_dir>
"""

import mrcfile
import numpy as np
from scipy import ndimage
from skimage import morphology, segmentation, measure
from pathlib import Path
import argparse
import sys
import os
import matplotlib.pyplot as plt

def load_membrane_segmentation_central_slice(membrain_dir):
    """Load the central z-slice of the membrane segmentation from membrain/ directory."""
    membrain_path = Path(membrain_dir)
    mrc_files = list(membrain_path.glob("*.mrc"))
    if not mrc_files:
        raise FileNotFoundError(f"No .mrc files found in {membrain_dir}")
    mrc_file = mrc_files[0]
    print(f"Loading membrane segmentation from: {mrc_file}")
    with mrcfile.open(mrc_file, permissive=True) as mrc:
        membrane_data = mrc.data
    print(f"Original membrane segmentation shape: {membrane_data.shape}")
    z_size = membrane_data.shape[0]
    z_start = int(z_size * 0.4)
    z_end = int(z_size * 0.6)
    central_idx = (z_start + z_end) // 2
    membrane_slice = membrane_data[central_idx]
    print(f"Extracted central slice index: {central_idx} (z range: {z_start} to {z_end})")
    return membrane_slice, central_idx

def load_tomogram_central_slice(tomogram_dir, central_idx):
    tomo_dir = Path(tomogram_dir) / "best_alignment"
    mrc_files = list(tomo_dir.glob("*.mrc"))
    if not mrc_files:
        print(f"No tomogram .mrc file found in {tomo_dir}, skipping overlay image.")
        return None
    tomo_file = mrc_files[0]
    print(f"Loading tomogram for overlay: {tomo_file}")
    with mrcfile.open(tomo_file, permissive=True) as mrc:
        tomo_data = mrc.data
    if central_idx >= tomo_data.shape[0]:
        print(f"Central index {central_idx} out of bounds for tomogram shape {tomo_data.shape}")
        return None
    return tomo_data[central_idx]

# Update all steps to operate in 2D
from scipy.ndimage import distance_transform_edt

def step1_identify_vesicles_organelles_2d(membrane_slice, min_area=800, max_area=2000):
    print("Step 1 (2D): Identifying vesicles and organelles...")
    membrane_binary = membrane_slice > 0
    distance = distance_transform_edt(~membrane_binary)
    local_maxima = morphology.local_maxima(distance)
    labeled_maxima = measure.label(local_maxima)
    props = measure.regionprops(labeled_maxima)
    vesicle_mask = np.zeros_like(membrane_binary, dtype=bool)
    for prop in props:
        if min_area <= prop.area <= max_area:
            vesicle_mask[labeled_maxima == prop.label] = True
    remaining_membranes = membrane_binary & ~vesicle_mask
    print(f"Identified {vesicle_mask.sum()} vesicle/organelle pixels")
    print(f"Remaining membrane pixels: {remaining_membranes.sum()}")
    return vesicle_mask, remaining_membranes

def step2_define_cell_boundaries_2d(remaining_membranes, min_cell_area=5000):
    print("Step 2 (2D): Defining cell boundaries...")
    distance = distance_transform_edt(~remaining_membranes)
    local_maxima = morphology.local_maxima(distance)
    filtered_maxima = local_maxima & (distance > 5)
    labeled_cells = measure.label(filtered_maxima)
    props = measure.regionprops(labeled_cells)
    cell_mask = np.zeros_like(remaining_membranes, dtype=bool)
    for prop in props:
        if prop.area >= min_cell_area:
            cell_mask[labeled_cells == prop.label] = True
    extracellular_mask = ~(cell_mask | remaining_membranes)
    print(f"Identified {cell_mask.sum()} intracellular pixels")
    print(f"Identified {extracellular_mask.sum()} extracellular pixels")
    return cell_mask, extracellular_mask

def step3_refine_classification_2d(cell_mask, extracellular_mask, remaining_membranes, 
                                   intracellular_distance=10, extracellular_distance=20):
    print("Step 3 (2D): Refining classification based on distance...")
    distance = distance_transform_edt(~remaining_membranes)
    refined_intracellular = (distance <= intracellular_distance) & ~remaining_membranes
    refined_extracellular = (distance > extracellular_distance) & ~refined_intracellular
    unclassified = ~(refined_intracellular | refined_extracellular | remaining_membranes)
    print(f"Refined intracellular pixels: {refined_intracellular.sum()}")
    print(f"Refined extracellular pixels: {refined_extracellular.sum()}")
    print(f"Unclassified pixels: {unclassified.sum()}")
    return refined_intracellular, refined_extracellular

def save_results_2d(vesicle_mask, intracellular_mask, extracellular_mask, output_dir):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    from imageio import imwrite
    imwrite(str(output_path / "vesicles_organelles_mask.png"), vesicle_mask.astype(np.uint8)*255)
    imwrite(str(output_path / "intracellular_mask.png"), intracellular_mask.astype(np.uint8)*255)
    imwrite(str(output_path / "extracellular_mask.png"), extracellular_mask.astype(np.uint8)*255)
    stats_file = output_path / "classification_stats_2d.txt"
    total_pixels = vesicle_mask.size
    with open(stats_file, 'w') as f:
        f.write(f"Total pixels: {total_pixels}\n")
        f.write(f"Vesicles/organelles: {vesicle_mask.sum()} pixels ({vesicle_mask.sum()/total_pixels*100:.2f}%)\n")
        f.write(f"Intracellular: {intracellular_mask.sum()} pixels ({intracellular_mask.sum()/total_pixels*100:.2f}%)\n")
        f.write(f"Extracellular: {extracellular_mask.sum()} pixels ({extracellular_mask.sum()/total_pixels*100:.2f}%)\n")
        f.write(f"Membranes: {(vesicle_mask | intracellular_mask | extracellular_mask).sum()} pixels\n")
    print(f"Saved 2D statistics to: {stats_file}")

def save_overlay_image_2d(vesicle_mask, intracellular_mask, extracellular_mask, output_dir, tomogram_slice):
    vmin, vmax = np.percentile(tomogram_slice, [2, 98])
    # Normalize grayscale to 0-255
    norm = np.clip((tomogram_slice - vmin) / (vmax - vmin), 0, 1)
    gray_img = (norm * 255).astype(np.uint8)
    # Make RGB image
    rgb_img = np.stack([gray_img]*3, axis=-1)
    # Apply masks with solid colors
    # Vesicles/organelle: green
    rgb_img[vesicle_mask] = [0, 255, 0]
    # Intracellular: blue
    rgb_img[intracellular_mask] = [0, 0, 255]
    # Extracellular: red
    rgb_img[extracellular_mask] = [255, 0, 0]
    from imageio import imwrite
    overlay_file = Path(output_dir) / "classification_overlay_2d.png"
    imwrite(overlay_file, rgb_img)
    print(f"Saved 2D overlay image to: {overlay_file}")

def main():
    parser = argparse.ArgumentParser(description="2D Multi-step extracellular space classification (central slice)")
    parser.add_argument("tomogram_dir", help="Path to tomogram directory")
    parser.add_argument("--output-dir", type=str, default="results/extracellular_segmentation_2d", help="Output directory")
    args = parser.parse_args()
    tomogram_dir = Path(args.tomogram_dir)
    membrain_dir = tomogram_dir / "best_alignment" / "membrain"
    if not membrain_dir.exists():
        print(f"Error: membrain directory not found at {membrain_dir}")
        sys.exit(1)
    try:
        membrane_slice, central_idx = load_membrane_segmentation_central_slice(membrain_dir)
        vesicle_mask, remaining_membranes = step1_identify_vesicles_organelles_2d(membrane_slice)
        cell_mask, extracellular_mask = step2_define_cell_boundaries_2d(remaining_membranes)
        refined_intracellular, refined_extracellular = step3_refine_classification_2d(
            cell_mask, extracellular_mask, remaining_membranes)
        tomogram_slice = load_tomogram_central_slice(tomogram_dir, central_idx)
        save_results_2d(vesicle_mask, refined_intracellular, refined_extracellular, args.output_dir)
        if tomogram_slice is not None:
            save_overlay_image_2d(vesicle_mask, refined_intracellular, refined_extracellular, args.output_dir, tomogram_slice)
        print("2D multi-step classification complete!")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 