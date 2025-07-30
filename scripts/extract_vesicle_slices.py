#!/usr/bin/env python3
"""
Vesicle Slice Extraction Script

This script extracts slices from vesicles that are within 10nm of the presynaptic active zone membrane.
It saves PNG images of each vesicle slice with proper orientation and contrast adjustment.

Usage:
    python scripts/extract_vesicle_slices.py --csv data/tomograms.csv --output-dir results/vesicle_slices
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
import os
import mrcfile
from scipy.spatial.distance import cdist
from scipy.ndimage import rotate
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import warnings
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Image, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import sys

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from synaptic_tomo_tools.vesicles import import_presynaptic_membranes_and_active_zones
warnings.filterwarnings('ignore')

def load_tomogram_data(tomogram_path):
    """Load tomogram data from the best alignment directory."""
    tomogram_path = Path(tomogram_path)
    
    # Look for ddw.mrc files in the best_alignment directory
    ddw_files = list(tomogram_path.glob("best_alignment/*_ddw.mrc"))
    if not ddw_files:
        print(f"No ddw.mrc files found in {tomogram_path}")
        return None
    
    # Use the first ddw file found
    ddw_file = ddw_files[0]
    print(f"Loading tomogram: {ddw_file}")
    
    try:
        with mrcfile.open(ddw_file) as mrc:
            data = mrc.data
        return data
    except Exception as e:
        print(f"Error loading tomogram {ddw_file}: {e}")
        return None

def load_vesicle_data(tomogram_path):
    """Load vesicle data from the STT results."""
    vesicle_file = Path(tomogram_path) / "best_alignment" / "STT_results" / "vesicles" / "vesicle_results.json"
    
    if not vesicle_file.exists():
        print(f"No vesicle results found: {vesicle_file}")
        return []
    
    try:
        with open(vesicle_file, 'r') as f:
            vesicle_data = json.load(f)
        return vesicle_data.get('vesicles', [])
    except Exception as e:
        print(f"Error loading vesicle data: {e}")
        return []

def load_active_zone_data(tomogram_path):
    """Load active zone data from the STT results."""
    az_dir = Path(tomogram_path) / "best_alignment" / "STT_results" / "active_zones"
    
    if not az_dir.exists():
        print(f"No active zones directory found: {az_dir}")
        return []
    
    # Load all active zone pre files
    az_files = list(az_dir.glob("*_pre.txt"))
    active_zone_points = []
    
    for az_file in az_files:
        try:
            points = np.loadtxt(az_file, delimiter=None)
            if points.ndim == 2 and points.shape[1] == 3:
                active_zone_points.extend(points)
        except Exception as e:
            print(f"Error loading active zone file {az_file}: {e}")
    
    return np.array(active_zone_points) if active_zone_points else np.array([])

def find_closest_active_zone_point(vesicle_center, active_zone_points):
    """Find the closest active zone point to the vesicle center."""
    if len(active_zone_points) == 0:
        return None, float('inf')
    
    distances = cdist([vesicle_center], active_zone_points)[0]
    min_idx = np.argmin(distances)
    min_distance = distances[min_idx]
    closest_point = active_zone_points[min_idx]
    
    return closest_point, min_distance

def extract_vesicle_slice(tomogram_data, vesicle_center, closest_az_point, slice_size=120):
    """
    Extract a slice from the tomogram centered on the vesicle.
    
    Args:
        tomogram_data: 3D tomogram data
        vesicle_center: (x, y, z) coordinates of vesicle center
        closest_az_point: (x, y, z) coordinates of closest active zone point
        slice_size: Size of the extracted slice in pixels
    
    Returns:
        slice_data: 2D slice data
        orientation_info: Information about the slice orientation
    """
    # Debug: Print coordinate information
    print(f"  Tomogram shape: {tomogram_data.shape}")
    print(f"  Vesicle center: {vesicle_center}")
    print(f"  Closest AZ point: {closest_az_point}")
    
    # Convert coordinates to integer indices
    center_x, center_y, center_z = map(int, vesicle_center)
    print(f"  Integer coordinates: ({center_x}, {center_y}, {center_z})")
    
    # Check if coordinates are within bounds
    if (center_x < 0 or center_x >= tomogram_data.shape[0] or
        center_y < 0 or center_y >= tomogram_data.shape[1] or
        center_z < 0 or center_z >= tomogram_data.shape[2]):
        print(f"  WARNING: Coordinates out of bounds!")
        return None, None
    
    # Calculate the direction vector from vesicle center to closest AZ point
    direction = np.array(closest_az_point) - np.array(vesicle_center)
    direction = direction / np.linalg.norm(direction)
    
    # Calculate rotation angles to align the direction with the z-axis (pointing down)
    # First, rotate around the x-axis to get the direction into the x-z plane
    angle_x = np.arctan2(direction[1], direction[2])
    
    # Then rotate around the y-axis to align with z-axis
    direction_rotated = np.array([
        direction[0],
        direction[1] * np.cos(angle_x) - direction[2] * np.sin(angle_x),
        direction[1] * np.sin(angle_x) + direction[2] * np.cos(angle_x)
    ])
    angle_y = np.arctan2(direction_rotated[0], direction_rotated[2])
    
    # Extract a larger region to allow for rotation
    margin = int(slice_size * 0.7)  # Extra margin for rotation
    
    # Calculate bounds with safety checks
    start_x = max(0, center_x - margin)
    end_x = min(tomogram_data.shape[0], center_x + margin)
    start_y = max(0, center_y - margin)
    end_y = min(tomogram_data.shape[1], center_y + margin)
    start_z = max(0, center_z - margin)
    end_z = min(tomogram_data.shape[2], center_z + margin)
    
    # Check if the region is too small
    if (end_x - start_x) < slice_size or (end_y - start_y) < slice_size or (end_z - start_z) < slice_size:
        print(f"  WARNING: Region too small for rotation, using simple extraction")
        # Fall back to simple extraction
        return extract_simple_slice(tomogram_data, vesicle_center, slice_size), {
            'method': 'simple_fallback',
            'vesicle_center': vesicle_center,
            'az_point': closest_az_point
        }
    
    # Extract the region
    region = tomogram_data[start_x:end_x, start_y:end_y, start_z:end_z]
    
    # Calculate the center of the extracted region
    region_center = np.array([
        center_x - start_x,
        center_y - start_y,
        center_z - start_z
    ])
    
    # Apply rotations
    # First rotate around x-axis
    region_rotated = rotate(region, np.degrees(angle_x), axes=(1, 2), reshape=False)
    
    # Then rotate around y-axis
    region_rotated = rotate(region_rotated, np.degrees(angle_y), axes=(0, 2), reshape=False)
    
    # Extract the center slice (XY plane at the vesicle center Z)
    slice_center = int(region_rotated.shape[2] // 2)
    slice_data = region_rotated[:, :, slice_center]
    
    # Debug: Print slice information
    print(f"  Extracted XY slice at Z={slice_center}, shape: {slice_data.shape}")
    
    # Crop to the desired size
    center_y, center_x = slice_data.shape[0] // 2, slice_data.shape[1] // 2
    half_size = slice_size // 2
    
    start_y = max(0, center_y - half_size)
    end_y = min(slice_data.shape[0], center_y + half_size)
    start_x = max(0, center_x - half_size)
    end_x = min(slice_data.shape[1], center_x + half_size)
    
    slice_data = slice_data[start_y:end_y, start_x:end_x]
    
    # Pad if necessary to reach the desired size
    if slice_data.shape[0] < slice_size:
        pad_y = slice_size - slice_data.shape[0]
        slice_data = np.pad(slice_data, ((0, pad_y), (0, 0)), mode='constant')
    
    if slice_data.shape[1] < slice_size:
        pad_x = slice_size - slice_data.shape[1]
        slice_data = np.pad(slice_data, ((0, 0), (0, pad_x)), mode='constant')
    
    return slice_data, {
        'angle_x': np.degrees(angle_x),
        'angle_y': np.degrees(angle_y),
        'direction': direction.tolist(),
        'vesicle_center': vesicle_center,
        'az_point': closest_az_point
    }

def extract_simple_slice(tomogram_data, vesicle_center, slice_size=120):
    """Extract a simple slice without rotation for debugging."""
    center_x, center_y, center_z = map(int, vesicle_center)
    
    # Extract the same type of slice as visualization.py
    # The tomogram data is [Z, Y, X], so we extract data[z_center] to get XY slice
    z_center = center_z
    
    # Extract the XY slice at the vesicle's Z position
    if 0 <= z_center < tomogram_data.shape[0]:
        slice_data = tomogram_data[z_center, :, :]
    else:
        print(f"  WARNING: Z coordinate {z_center} out of bounds [0, {tomogram_data.shape[0]})")
        return None
    
    # Crop around the vesicle center
    half_size = slice_size // 2
    
    start_y = max(0, center_y - half_size)
    end_y = min(slice_data.shape[0], center_y + half_size)
    start_x = max(0, center_x - half_size)
    end_x = min(slice_data.shape[1], center_x + half_size)
    
    # Extract the region
    slice_data = slice_data[start_y:end_y, start_x:end_x]
    
    # Pad if necessary
    if slice_data.shape[0] < slice_size:
        pad_y = slice_size - slice_data.shape[0]
        slice_data = np.pad(slice_data, ((0, pad_y), (0, 0)), mode='constant')
    
    if slice_data.shape[1] < slice_size:
        pad_x = slice_size - slice_data.shape[1]
        slice_data = np.pad(slice_data, ((0, 0), (0, pad_x)), mode='constant')
    
    return slice_data

def extract_large_slice_for_rotation(tomogram_data, vesicle_center, final_size=120):
    """Extract a larger slice for rotation to avoid blank corners."""
    center_x, center_y, center_z = map(int, vesicle_center)
    
    # Extract the same type of slice as visualization.py
    z_center = center_z
    
    # Extract the XY slice at the vesicle's Z position
    if 0 <= z_center < tomogram_data.shape[0]:
        slice_data = tomogram_data[z_center, :, :]
    else:
        print(f"  WARNING: Z coordinate {z_center} out of bounds [0, {tomogram_data.shape[0]})")
        return None
    
    # Extract a much larger region to accommodate rotation
    # For a 45-degree rotation, we need about 1.4x the final size
    # For safety, we'll use 2x the final size
    large_size = final_size * 2
    half_large_size = large_size // 2
    
    start_y = max(0, center_y - half_large_size)
    end_y = min(slice_data.shape[0], center_y + half_large_size)
    start_x = max(0, center_x - half_large_size)
    end_x = min(slice_data.shape[1], center_x + half_large_size)
    
    # Extract the large region
    large_slice = slice_data[start_y:end_y, start_x:end_x]
    
    # Pad if necessary to ensure we have a square region
    max_dim = max(large_slice.shape[0], large_slice.shape[1])
    if large_slice.shape[0] < max_dim:
        pad_y = max_dim - large_slice.shape[0]
        large_slice = np.pad(large_slice, ((0, pad_y), (0, 0)), mode='constant')
    
    if large_slice.shape[1] < max_dim:
        pad_x = max_dim - large_slice.shape[1]
        large_slice = np.pad(large_slice, ((0, 0), (0, pad_x)), mode='constant')
    
    return large_slice

def rotate_slice_to_az_direction(large_slice, vesicle_center, closest_az_point, final_size=120):
    """Rotate the slice so that the point closest to active zone points down."""
    from scipy.ndimage import rotate
    
    # Calculate the vector from vesicle center to closest AZ point
    az_vector = np.array(closest_az_point) - np.array(vesicle_center)
    
    # Calculate the angle to rotate so this vector points down (negative Y direction)
    # The angle is the angle between the AZ vector and the negative Y axis
    angle_rad = np.arctan2(az_vector[0], -az_vector[1])  # atan2(x, -y) for pointing down
    angle_deg = np.degrees(angle_rad)
    
    # Rotate the large slice
    rotated_slice = rotate(large_slice, angle_deg, reshape=False)
    
    # Crop to the final size from the center
    center_y, center_x = rotated_slice.shape[0] // 2, rotated_slice.shape[1] // 2
    half_size = final_size // 2
    
    start_y = center_y - half_size
    end_y = center_y + half_size
    start_x = center_x - half_size
    end_x = center_x + half_size
    
    # Crop to final size
    cropped_slice = rotated_slice[start_y:end_y, start_x:end_x]
    
    return cropped_slice

def extract_minip_for_rotation(tomogram_data, vesicle_center, final_size=120):
    """Extract a large MinIP slice for rotation to avoid blank corners."""
    center_x, center_y, center_z = map(int, vesicle_center)
    
    # Extract multiple Z-slices around the vesicle and create MinIP
    # Use 20 slices total: 10 on each side of the vesicle center
    z_range = 10  # Extract 10 slices on each side of the vesicle center (20 total)
    z_start = max(0, center_z - z_range)
    z_end = min(tomogram_data.shape[0], center_z + z_range)
    
    # Extract XY slices at different Z positions
    slices = []
    for z in range(z_start, z_end):
        slice_data = tomogram_data[z, :, :]
        slices.append(slice_data)
    
    # Create minimum intensity projection
    if slices:
        minip_data = np.min(slices, axis=0)
    else:
        print(f"  WARNING: No slices extracted for MinIP")
        return None
    
    # Extract a much larger region to accommodate rotation
    # For a 45-degree rotation, we need about 1.4x the final size
    # For safety, we'll use 2x the final size
    large_size = final_size * 2
    half_large_size = large_size // 2
    
    start_y = max(0, center_y - half_large_size)
    end_y = min(minip_data.shape[0], center_y + half_large_size)
    start_x = max(0, center_x - half_large_size)
    end_x = min(minip_data.shape[1], center_x + half_large_size)
    
    # Extract the large region
    large_minip = minip_data[start_y:end_y, start_x:end_x]
    
    # Pad if necessary to ensure we have a square region
    max_dim = max(large_minip.shape[0], large_minip.shape[1])
    if large_minip.shape[0] < max_dim:
        pad_y = max_dim - large_minip.shape[0]
        large_minip = np.pad(large_minip, ((0, pad_y), (0, 0)), mode='constant')
    
    if large_minip.shape[1] < max_dim:
        pad_x = max_dim - large_minip.shape[1]
        large_minip = np.pad(large_minip, ((0, 0), (0, pad_x)), mode='constant')
    
    return large_minip

def extract_thick_slice_for_rotation(tomogram_data, vesicle_center, final_size=120, thickness_nm=20):
    """Extract a thick slice (20 nm) for rotation to avoid blank corners."""
    center_x, center_y, center_z = map(int, vesicle_center)
    
    # Extract multiple Z-slices around the vesicle and create thick slice
    # Assuming 1 nm per pixel, 10 nm = 10 slices
    z_range = 5  # Extract 5 slices on each side of the vesicle center (10 total)
    z_start = max(0, center_z - z_range)
    z_end = min(tomogram_data.shape[0], center_z + z_range)
    
    # Extract XY slices at different Z positions
    slices = []
    for z in range(z_start, z_end):
        slice_data = tomogram_data[z, :, :]
        slices.append(slice_data)
    
    # Create average projection (thick slice)
    if slices:
        thick_data = np.mean(slices, axis=0)
    else:
        print(f"  WARNING: No slices extracted for thick slice")
        return None
    
    # Extract a much larger region to accommodate rotation
    # For a 45-degree rotation, we need about 1.4x the final size
    # For safety, we'll use 2x the final size
    large_size = final_size * 2
    half_large_size = large_size // 2
    
    start_y = max(0, center_y - half_large_size)
    end_y = min(thick_data.shape[0], center_y + half_large_size)
    start_x = max(0, center_x - half_large_size)
    end_x = min(thick_data.shape[1], center_x + half_large_size)
    
    # Extract the large region
    large_thick = thick_data[start_y:end_y, start_x:end_x]
    
    # Pad if necessary to ensure we have a square region
    max_dim = max(large_thick.shape[0], large_thick.shape[1])
    if large_thick.shape[0] < max_dim:
        pad_y = max_dim - large_thick.shape[0]
        large_thick = np.pad(large_thick, ((0, pad_y), (0, 0)), mode='constant')
    
    if large_thick.shape[1] < max_dim:
        pad_x = max_dim - large_thick.shape[1]
        large_thick = np.pad(large_thick, ((0, 0), (0, pad_x)), mode='constant')
    
    return large_thick

def extract_mip_slice(tomogram_data, vesicle_center, slice_size=120):
    """Extract a maximum intensity projection through the vesicle."""
    center_x, center_y, center_z = map(int, vesicle_center)
    
    # Extract multiple Z-slices around the vesicle and create MIP
    z_range = 10  # Extract 10 slices around the vesicle center
    z_start = max(0, center_z - z_range)
    z_end = min(tomogram_data.shape[0], center_z + z_range)
    
    # Extract XY slices at different Z positions
    slices = []
    for z in range(z_start, z_end):
        slice_data = tomogram_data[z, :, :]
        slices.append(slice_data)
    
    # Create maximum intensity projection
    if slices:
        slice_data = np.max(slices, axis=0)
    else:
        print(f"  WARNING: No slices extracted for MIP")
        return None
    
    # Crop around the vesicle center
    half_size = slice_size // 2
    
    start_y = max(0, center_y - half_size)
    end_y = min(slice_data.shape[0], center_y + half_size)
    start_x = max(0, center_x - half_size)
    end_x = min(slice_data.shape[1], center_x + half_size)
    
    # Extract the region
    slice_data = slice_data[start_y:end_y, start_x:end_x]
    
    # Pad if necessary
    if slice_data.shape[0] < slice_size:
        pad_y = slice_size - slice_data.shape[0]
        slice_data = np.pad(slice_data, ((0, pad_y), (0, 0)), mode='constant')
    
    if slice_data.shape[1] < slice_size:
        pad_x = slice_size - slice_data.shape[1]
        slice_data = np.pad(slice_data, ((0, 0), (0, pad_x)), mode='constant')
    
    return slice_data



def save_slice_as_png(slice_data, output_path, vesicle_info, contrast_percentile=99):
    """
    Save the slice as a PNG image with contrast adjustment.
    
    Args:
        slice_data: 2D slice data
        output_path: Path to save the PNG file
        vesicle_info: Information about the vesicle
        contrast_percentile: Percentile for contrast adjustment
    """
    # Apply contrast adjustment similar to visualization.py
    vmin = np.percentile(slice_data, 100 - contrast_percentile)
    vmax = np.percentile(slice_data, contrast_percentile)
    
    # Normalize the data
    normalized_data = np.clip((slice_data - vmin) / (vmax - vmin), 0, 1)
    
    # Create the figure with fixed aspect ratio
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    # Display the slice with proper aspect ratio
    im = ax.imshow(normalized_data, cmap='gray', origin='lower', aspect='equal')
    
    # Add image type and thickness information on the image
    note = vesicle_info.get('note', '')
    if 'MinIP' in note:
        image_type = "MinIP (20 pixels thick)"
    elif 'thick' in note:
        image_type = "Slice (20 pixels thick)"
    elif 'az_volume' in note.lower():
        image_type = "AZ Volume (10 nm thick)"
    else:
        image_type = "Slice (1 pixel thick)"
    
    # Add text in bottom-left corner of the image itself
    ax.text(5, 5, image_type, 
            color='white', fontsize=14, weight='bold', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
    
    # Add title with vesicle information
    title = f"Vesicle {vesicle_info.get('id', 'unknown')}\n"
    title += f"Distance to AZ: {vesicle_info.get('distance_to_az', 'unknown'):.1f} nm\n"
    title += f"Volume: {vesicle_info.get('volume', 'unknown'):.0f} nm³\n"
    title += f"Diameter: {vesicle_info.get('diameter', 'unknown'):.0f} nm"
    ax.set_title(title, fontsize=10)
    
    # Remove axes
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add a scale bar (assuming 1 nm per pixel)
    scale_bar_length = 20  # 20 nm
    bar_y = 5  # Further down
    bar_x = slice_data.shape[1] - scale_bar_length - 5  # Further right
    ax.plot([bar_x, bar_x + scale_bar_length], [bar_y, bar_y], 
            'white', linewidth=5)  # Thicker scale bar
    ax.text(bar_x + scale_bar_length/2, bar_y + 3, f'{scale_bar_length} nm', 
            color='white', fontsize=12, weight='bold', ha='center', va='bottom')  # Text closer to bar
    
    # Set the plot limits to ensure the full image is visible
    ax.set_xlim(0, slice_data.shape[1])
    ax.set_ylim(0, slice_data.shape[0])
    
    # Save the image without tight_layout to prevent cropping
    plt.savefig(output_path, dpi=150, bbox_inches=None, pad_inches=0.1)
    plt.close()

def process_tomogram(tomogram_path, output_dir):
    """Process a single tomogram to extract vesicle slices."""
    tomogram_name = Path(tomogram_path).name
    print(f"\nProcessing tomogram: {tomogram_name}")
    
    # Load data
    tomogram_data = load_tomogram_data(tomogram_path)
    if tomogram_data is None:
        return 0
    
    vesicles = load_vesicle_data(tomogram_path)
    if not vesicles:
        print("No vesicles found")
        return 0
    
    # Load membrane_active_zone_pairs for fusion point calculation
    membrane_active_zone_pairs = import_presynaptic_membranes_and_active_zones(tomogram_path)
    if not membrane_active_zone_pairs:
        print("No membrane-active zone pairs found")
        return 0
    
    # Create output directory for this tomogram
    tomogram_output_dir = output_dir / tomogram_name
    tomogram_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process vesicles within 10nm of active zone
    extracted_count = 0
    for i, vesicle in enumerate(vesicles):
        # Check if vesicle is within 10nm of active zone
        distance_to_az = vesicle.get('distance_to_az', float('inf'))
        if distance_to_az > 10.0:
            continue
        
        # Get vesicle center and size information
        vesicle_coords = np.array(vesicle['coordinates'])
        vesicle_center = vesicle_coords.mean(axis=0)
        
        # Calculate vesicle diameter (approximate)
        if len(vesicle_coords) > 1:
            distances = cdist([vesicle_center], vesicle_coords)[0]
            vesicle_diameter = np.max(distances) * 2
            print(f"  Vesicle diameter (approximate): {vesicle_diameter:.1f} pixels")
        else:
            vesicle_diameter = 0
            print(f"  Vesicle diameter: unknown (single point)")
        
        # Get vesicle diameter from volume spherical fit if available
        vesicle_diameter_nm = vesicle.get('diameter', 0)  # This should be the diameter from spherical fit
        
        # Calculate fusion point for this vesicle
        fusion_point = calculate_fusion_point_for_vesicle(vesicle, membrane_active_zone_pairs)
        if fusion_point is None:
            print(f"  Skipping vesicle {i} - no fusion point found")
            continue
        
        # Extract slice
        try:
            # Extract the same slice as simple version, then rotate it
            simple_slice = extract_simple_slice(tomogram_data, vesicle_center)
            
            if simple_slice is None:
                print(f"  Skipping vesicle {i} - coordinate issues")
                continue
            
            # Extract a larger slice for rotation to avoid blank corners
            large_slice = extract_large_slice_for_rotation(tomogram_data, vesicle_center)
            if large_slice is not None:
                # Apply rotation to orient the slice so fusion point direction points down
                rotated_slice = rotate_slice_to_az_direction(large_slice, vesicle_center, fusion_point)
                
                # Save oriented slice
                output_path = tomogram_output_dir / f"vesicle_{i:04d}_slice.png"
                vesicle_info = {
                    'id': i,
                    'distance_to_az': distance_to_az,
                    'volume': vesicle.get('volume', 0),
                    'diameter': vesicle_diameter_nm,
                    'note': 'Rotated slice (fusion point direction down)'
                }
                save_slice_as_png(rotated_slice, output_path, vesicle_info)
            
            # Also create rotated MinIP slice
            large_minip = extract_minip_for_rotation(tomogram_data, vesicle_center)
            if large_minip is not None:
                # Apply rotation to orient the MinIP slice so fusion point direction points down
                rotated_minip = rotate_slice_to_az_direction(large_minip, vesicle_center, fusion_point)
                
                # Save oriented MinIP slice
                minip_output_path = tomogram_output_dir / f"vesicle_{i:04d}_slice_minip.png"
                minip_vesicle_info = {
                    'id': i,
                    'distance_to_az': distance_to_az,
                    'volume': vesicle.get('volume', 0),
                    'diameter': vesicle_diameter_nm,
                    'note': 'Rotated MinIP slice (fusion point direction down)'
                }
                save_slice_as_png(rotated_minip, minip_output_path, minip_vesicle_info)
            
            # Also create rotated thick slice (20 nm)
            large_thick = extract_thick_slice_for_rotation(tomogram_data, vesicle_center)
            if large_thick is not None:
                # Apply rotation to orient the thick slice so fusion point direction points down
                rotated_thick = rotate_slice_to_az_direction(large_thick, vesicle_center, fusion_point)
                
                # Save oriented thick slice
                thick_output_path = tomogram_output_dir / f"vesicle_{i:04d}_slice_thick.png"
                thick_vesicle_info = {
                    'id': i,
                    'distance_to_az': distance_to_az,
                    'volume': vesicle.get('volume', 0),
                    'diameter': vesicle_diameter_nm,
                    'note': 'Rotated thick slice (20 nm, fusion point direction down)'
                }
                save_slice_as_png(rotated_thick, thick_output_path, thick_vesicle_info)
            

            
            extracted_count += 1
            print(f"  Extracted slice for vesicle {i} (distance: {distance_to_az:.1f} nm)")
            
        except Exception as e:
            print(f"  Error extracting slice for vesicle {i}: {e}")
    
    print(f"Extracted {extracted_count} vesicle slices for {tomogram_name}")
    return extracted_count

def create_vesicle_summary_pdf(output_dir):
    """Create a PDF summary of all vesicle slices."""
    output_dir = Path(output_dir)
    
    if not output_dir.exists():
        print(f"Output directory {output_dir} does not exist")
        return
    
    # Find all tomogram directories
    tomogram_dirs = [d for d in output_dir.iterdir() if d.is_dir()]
    
    if not tomogram_dirs:
        print(f"No tomogram directories found in {output_dir}")
        return
    
    # Create PDF file
    pdf_path = output_dir / "vesicle_slices_summary.pdf"
    doc = SimpleDocTemplate(str(pdf_path), pagesize=A4)
    story = []
    
    # Get styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=16,
        spaceAfter=20,
        alignment=1  # Center alignment
    )
    
    # Add title
    title = Paragraph("Vesicle Slices Summary", title_style)
    story.append(title)
    story.append(Spacer(1, 20))
    
    # Process each tomogram
    for tomogram_dir in sorted(tomogram_dirs):
        tomogram_name = tomogram_dir.name
        
        # Add tomogram title
        tomogram_title = Paragraph(f"Tomogram: {tomogram_name}", styles['Heading2'])
        story.append(tomogram_title)
        story.append(Spacer(1, 10))
        
        # Find all vesicle slice files
        slice_files = list(tomogram_dir.glob("vesicle_*_slice.png"))
        minip_files = list(tomogram_dir.glob("vesicle_*_slice_minip.png"))
        thick_files = list(tomogram_dir.glob("vesicle_*_slice_thick.png"))
        
        # Group files by vesicle ID
        vesicle_groups = {}
        for slice_file in slice_files:
            # Extract vesicle ID from filename (e.g., vesicle_0017_slice.png -> 0017)
            vesicle_id = slice_file.stem.split('_')[1]
            if vesicle_id not in vesicle_groups:
                vesicle_groups[vesicle_id] = {}
            
            # Find corresponding minip and thick files
            minip_file = tomogram_dir / f"vesicle_{vesicle_id}_slice_minip.png"
            thick_file = tomogram_dir / f"vesicle_{vesicle_id}_slice_thick.png"
            
            vesicle_groups[vesicle_id] = {
                'slice': slice_file if slice_file.exists() else None,
                'minip': minip_file if minip_file.exists() else None,
                'thick': thick_file if thick_file.exists() else None
            }
        
        # Create table for this tomogram
        if vesicle_groups:
            # Create table data
            table_data = []
            
            # Add header
            table_data.append(['Vesicle ID', 'Slice', 'Thick Slice', 'MinIP'])
            
            # Add vesicle rows
            for vesicle_id in sorted(vesicle_groups.keys()):
                group = vesicle_groups[vesicle_id]
                
                # Create image objects for the table
                slice_img = Image(str(group['slice']), width=1.5*inch, height=1.5*inch) if group['slice'] else Paragraph("N/A", styles['Normal'])
                thick_img = Image(str(group['thick']), width=1.5*inch, height=1.5*inch) if group['thick'] else Paragraph("N/A", styles['Normal'])
                minip_img = Image(str(group['minip']), width=1.5*inch, height=1.5*inch) if group['minip'] else Paragraph("N/A", styles['Normal'])
                
                table_data.append([f"Vesicle {vesicle_id}", slice_img, thick_img, minip_img])
            
            # Create table
            table = Table(table_data, colWidths=[1*inch, 1.5*inch, 1.5*inch, 1.5*inch])
            
            # Style the table
            table_style = TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ])
            table.setStyle(table_style)
            
            story.append(table)
            story.append(Spacer(1, 20))
    
    # Build PDF
    try:
        doc.build(story)
        print(f"Vesicle summary PDF created: {pdf_path}")
    except Exception as e:
        print(f"Error creating PDF: {e}")

def create_close_vesicle_summary_pdf(output_dir, csv_path, data_dir):
    """Create a PDF summary of vesicles within 4nm of active zone, pooled from all tomograms."""
    output_dir = Path(output_dir)
    
    if not output_dir.exists():
        print(f"Output directory {output_dir} does not exist")
        return
    
    # Load tomogram information
    df = pd.read_csv(csv_path)
    
    # Create PDF file with portrait orientation (table is now compact enough)
    from reportlab.lib.pagesizes import A4
    pdf_path = output_dir / "close_vesicles_summary.pdf"
    doc = SimpleDocTemplate(str(pdf_path), pagesize=A4)
    story = []
    
    # Get styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=16,
        spaceAfter=20,
        alignment=1  # Center alignment
    )
    
    # Add title
    title = Paragraph("Close Vesicles Summary (≤4nm from Active Zone)", title_style)
    story.append(title)
    story.append(Spacer(1, 20))
    
    # Collect all vesicles within 4nm from all tomograms
    all_close_vesicles = []
    
    for _, row in df.iterrows():
        tomogram_name = row['tomoname']
        set_name = row['set']
        
        # Construct tomogram path
        if data_dir:
            tomogram_path = Path(data_dir) / set_name / "TOP_TOMOS" / tomogram_name
        else:
            tomogram_path = Path("data") / set_name / "TOP_TOMOS" / tomogram_name
        
        if not tomogram_path.exists():
            continue
        
        # Load vesicle data for this tomogram
        vesicles = load_vesicle_data(tomogram_path)
        if not vesicles:
            continue
        
        # Find vesicles within 4nm
        for i, vesicle in enumerate(vesicles):
            distance_to_az = vesicle.get('distance_to_az', float('inf'))
            if distance_to_az <= 4.0:
                # Add tomogram info and index to vesicle data
                vesicle['tomogram'] = tomogram_name
                vesicle['set'] = set_name
                vesicle['index'] = i  # Use the index for file matching
                all_close_vesicles.append(vesicle)
    
    if not all_close_vesicles:
        print("No vesicles found within 4nm of active zone")
        return
    
    # Sort by distance to active zone (closest to farthest, 0 to 4nm)
    all_close_vesicles.sort(key=lambda x: x.get('distance_to_az', float('inf')))
    
    # Create table data
    table_data = []
    
    # Add header with proper Paragraph objects
    header_style = ParagraphStyle(
        'HeaderStyle',
        parent=styles['Normal'],
        fontSize=10,
        fontName='Helvetica-Bold',
        alignment=1  # Center alignment
    )
    table_data.append([
        Paragraph('Tomogram', header_style),
        Paragraph('Distance to AZ (nm)', header_style),
        Paragraph('Volume (nm³)', header_style),
        Paragraph('Diameter (nm)', header_style),
        Paragraph('Slice', header_style),
        Paragraph('Thick Slice', header_style),
        Paragraph('MinIP', header_style)
    ])
    
    # Add vesicle rows
    for vesicle in all_close_vesicles:
        tomogram_name = vesicle['tomogram']
        vesicle_index = vesicle.get('index', 0)  # Use index for file matching
        vesicle_id = vesicle.get('id', 'unknown')  # Keep ID for display
        distance_to_az = vesicle.get('distance_to_az', 0)
        volume = vesicle.get('volume', 0)
        diameter = vesicle.get('diameter', 0)
        
        # Look for corresponding image files using the index
        tomogram_output_dir = output_dir / tomogram_name
        
        # Use the index to construct filenames (same as in main processing)
        slice_file = tomogram_output_dir / f"vesicle_{vesicle_index:04d}_slice.png"
        thick_file = tomogram_output_dir / f"vesicle_{vesicle_index:04d}_slice_thick.png"
        minip_file = tomogram_output_dir / f"vesicle_{vesicle_index:04d}_slice_minip.png"
        
        # Debug: Print what we're looking for and what we found
        print(f"  Looking for vesicle index {vesicle_index} (ID: {vesicle_id}) in {tomogram_output_dir}")
        print(f"    Looking for: vesicle_{vesicle_index:04d}_slice.png")
        print(f"    Found slice: {slice_file.exists() if slice_file else False}")
        print(f"    Found thick: {thick_file.exists() if thick_file else False}")
        print(f"    Found minip: {minip_file.exists() if minip_file else False}")
        
        # Create image objects for the table (larger size for landscape orientation)
        slice_img = Image(str(slice_file), width=1.0*inch, height=1.0*inch) if slice_file and slice_file.exists() else Paragraph("N/A", styles['Normal'])
        thick_img = Image(str(thick_file), width=1.0*inch, height=1.0*inch) if thick_file and thick_file.exists() else Paragraph("N/A", styles['Normal'])
        minip_img = Image(str(minip_file), width=1.0*inch, height=1.0*inch) if minip_file and minip_file.exists() else Paragraph("N/A", styles['Normal'])
        
        # Create text cells with proper wrapping
        tomogram_cell = Paragraph(tomogram_name, styles['Normal'])
        distance_cell = Paragraph(f"{distance_to_az:.1f}", styles['Normal'])
        volume_cell = Paragraph(f"{volume:.0f}", styles['Normal'])
        diameter_cell = Paragraph(f"{diameter:.0f}", styles['Normal'])
        
        table_data.append([
            tomogram_cell,
            distance_cell,
            volume_cell,
            diameter_cell,
            slice_img,
            thick_img,
            minip_img
        ])
    
    # Calculate optimal column widths based on content
    # Get the maximum width needed for each column
    col_widths = []
    
    # Column 0: Tomogram name (text) - extract text from Paragraph objects
    max_tomogram_width = max(len(row[0].text) for row in table_data[1:]) if len(table_data) > 1 else 10
    col_widths.append(min(max_tomogram_width * 0.08 * inch, 1.5 * inch))  # Cap at 1.5 inch
    
    # Column 1: Distance (short number)
    col_widths.append(0.8 * inch)
    
    # Column 2: Volume (number)
    col_widths.append(0.8 * inch)
    
    # Column 3: Diameter (number)
    col_widths.append(0.8 * inch)
    
    # Columns 4-6: Images (fixed size)
    col_widths.extend([1.1 * inch, 1.1 * inch, 1.1 * inch])
    
    # Create table with calculated column widths
    table = Table(table_data, colWidths=col_widths)
    
    # Style the table
    table_style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('WRAP', (0, 0), (-1, -1), True),
    ])
    table.setStyle(table_style)
    
    story.append(table)
    story.append(Spacer(1, 20))
    
    # Add summary statistics
    total_vesicles = len(all_close_vesicles)
    avg_distance = sum(v.get('distance_to_az', 0) for v in all_close_vesicles) / total_vesicles
    avg_volume = sum(v.get('volume', 0) for v in all_close_vesicles) / total_vesicles
    avg_diameter = sum(v.get('diameter', 0) for v in all_close_vesicles) / total_vesicles
    
    summary_text = f"""
    Summary Statistics:
    - Total vesicles within 4nm: {total_vesicles}
    - Average distance to AZ: {avg_distance:.2f} nm
    - Average volume: {avg_volume:.0f} nm³
    - Average diameter: {avg_diameter:.1f} nm
    """
    
    summary_para = Paragraph(summary_text, styles['Normal'])
    story.append(summary_para)
    
    # Build PDF
    try:
        doc.build(story)
        print(f"Close vesicles summary PDF created: {pdf_path}")
    except Exception as e:
        print(f"Error creating PDF: {e}")

def calculate_fusion_point_for_vesicle(vesicle, membrane_active_zone_pairs, fusion_point_threshold=10.0):
    """
    Calculate the putative fusion point for a vesicle.
    
    Args:
        vesicle: Vesicle dictionary with coordinates and distance_to_az
        membrane_active_zone_pairs: Dictionary of membrane-active zone pairs
        fusion_point_threshold: Distance threshold for fusion point calculation
        
    Returns:
        Fusion point coordinates (np.ndarray) or None if not found
    """
    # Only consider vesicles within 10 nm of the presynaptic active zone
    if vesicle.get('distance_to_az', 0.0) > 10.0:
        return None
    
    vesicle_points = np.array(vesicle['coordinates'])
    membrane_name = vesicle.get('closest_membrane', None)
    
    if not membrane_name or membrane_name not in membrane_active_zone_pairs:
        return None
    
    active_zone_points = membrane_active_zone_pairs[membrane_name]['active_zone_points']
    if active_zone_points is None or len(active_zone_points) == 0:
        return None
    
    # For each vesicle point, find all active zone points within fusion_point_threshold
    from scipy.spatial import KDTree
    tree = KDTree(active_zone_points)
    close_points = []
    
    for pt in vesicle_points:
        idxs = tree.query_ball_point(pt, r=fusion_point_threshold)
        if idxs:
            close_points.extend(active_zone_points[idxs])
    
    if close_points:
        fusion_point = np.mean(np.vstack(close_points), axis=0)
        return fusion_point
    
    return None

def main():
    parser = argparse.ArgumentParser(description='Extract vesicle slices from tomograms')
    parser.add_argument('--csv', default='data/tomograms.csv', help='CSV file with tomogram information')
    parser.add_argument('--output-dir', default='results/vesicle_slices', help='Output directory for slices')
    parser.add_argument('--data-dir', default='data', help='Base data directory')
    parser.add_argument('--set', help='Filter by set name')
    parser.add_argument('--start-from', help='Start from specific tomogram')
    args = parser.parse_args()
    
    # Load tomogram information
    df = pd.read_csv(args.csv)
    
    # Filter by set if specified
    if args.set:
        df = df[df['set'] == args.set]
    
    # Filter by starting tomogram if specified
    if args.start_from:
        start_idx = df[df['tomoname'] == args.start_from].index
        if len(start_idx) > 0:
            df = df.iloc[start_idx[0]:]
            print(f"Starting from tomogram: {args.start_from}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each tomogram
    total_extracted = 0
    for _, row in df.iterrows():
        tomogram_name = row['tomoname']
        set_name = row['set']
        
        # Construct tomogram path
        if args.data_dir:
            tomogram_path = Path(args.data_dir) / set_name / "TOP_TOMOS" / tomogram_name
        else:
            tomogram_path = Path("data") / set_name / "TOP_TOMOS" / tomogram_name
        
        if tomogram_path.exists():
            extracted = process_tomogram(tomogram_path, output_dir)
            total_extracted += extracted
        else:
            print(f"Tomogram path not found: {tomogram_path}")
    
    print(f"\nTotal vesicle slices extracted: {total_extracted}")
    print(f"Output directory: {output_dir.absolute()}")
    
    # Create PDF summary
    print("\nCreating vesicle slices summary PDF...")
    create_vesicle_summary_pdf(output_dir)
    
    # Create close vesicles summary PDF
    print("\nCreating close vesicles summary PDF (≤4nm from active zone)...")
    create_close_vesicle_summary_pdf(output_dir, args.csv, args.data_dir)

if __name__ == "__main__":
    main() 