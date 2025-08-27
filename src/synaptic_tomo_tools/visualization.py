#!/usr/bin/env python3
"""
Synaptic Tomogram Results Visualization Script

This script generates visualizations for analyzed synaptic tomograms, including overlays
of membranes, vesicles, active zones, and AuNPs. It processes all available tomograms
and saves the figures as output files.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import json
import argparse
from datetime import datetime

# Import from the same package

# Try to import mrcfile, but handle gracefully if not available
try:
    import mrcfile
except ImportError:
    print("Warning: mrcfile not available. Tomogram slice loading will be disabled.")
    mrcfile = None

# Helper to find analyzed tomograms
def find_analyzed_tomograms(base_dir="../data/"):
    """Find all tomograms that have been analyzed and have results."""
    tomos = []
    for root, dirs, files in os.walk(base_dir):
        if 'best_alignment' in dirs:
            tomo_path = Path(root)
            vesicle_json = tomo_path / 'best_alignment' / 'STT_results' / 'vesicles' / 'vesicle_results.json'
            if vesicle_json.exists():
                tomos.append(str(tomo_path))
    return sorted(tomos)

def load_tomogram_slice(tomo_path, z_center=None):
    """Load a 2D slice from the tomogram."""
    if mrcfile is None:
        print("mrcfile not available, skipping tomogram slice loading")
        return None, None
    
    mrcs = list((Path(tomo_path) / 'best_alignment').glob('*ddw.mrc'))
    if not mrcs:
        return None, None
    with mrcfile.open(mrcs[0], 'r') as mrc:
        data = mrc.data
    if z_center is None:
        z_center = data.shape[0] // 2
    return data[z_center], z_center

def load_membrane_coords(tomo_path, kind='presynaptic'):
    """Load membrane coordinates from text files."""
    aunps_dir = Path(tomo_path) / 'best_alignment' / 'aunps'
    files = sorted(aunps_dir.glob(f'{kind}membranes_*.txt'))
    coords = [np.loadtxt(f) for f in files if f.exists()]
    return coords

def load_active_zone_coords(tomo_path):
    """Load active zone coordinates."""
    az_dir = Path(tomo_path) / 'best_alignment' / 'STT_results' / 'active_zones'
    files = sorted(az_dir.glob('active_zone_pre*_post*_pre.txt'))
    coords = [np.loadtxt(f) for f in files if f.exists()]
    return coords

def load_vesicles(tomo_path):
    """Load vesicle data from JSON file."""
    ves_file = Path(tomo_path) / 'best_alignment' / 'STT_results' / 'vesicles' / 'vesicle_results.json'
    with open(ves_file) as f:
        data = json.load(f)
    return data['vesicles']

def load_aunps(tomo_path, active_zone_indices=None):
    """Load AuNP coordinates from STAR file(s), optionally filtered by active_zone_indices."""
    aunps_dir = Path(tomo_path) / 'best_alignment' / 'aunps'
    import starfile
    import glob
    import pandas as pd
    star_dfs = []
    if active_zone_indices is not None:
        for idx in active_zone_indices:
            star_file = aunps_dir / f"aunp_tm_BP_active_zone_{idx}.star"
            print("[viz] Trying to load:", star_file)
            if star_file.exists():
                star_data = starfile.read(star_file)
                if isinstance(star_data, dict):
                    for v in star_data.values():
                        if isinstance(v, pd.DataFrame):
                            star_dfs.append(v)
                            break
                elif isinstance(star_data, pd.DataFrame):
                    star_dfs.append(star_data)
    else:
        print("[viz] active_zone_indices is None, loading all aunp_tm_BP_active_zone_*.star files")
        import re
        pattern = str(aunps_dir / "aunp_tm_BP_active_zone_*.star")
        for file in glob.glob(pattern):
            fname = Path(file).name
            m = re.match(r"aunp_tm_BP_active_zone_(\d+)\.star", fname)
            if m:
                star_data = starfile.read(Path(file))
                if isinstance(star_data, dict):
                    for v in star_data.values():
                        if isinstance(v, pd.DataFrame):
                            star_dfs.append(v)
                            break
                elif isinstance(star_data, pd.DataFrame):
                    star_dfs.append(star_data)
        if not star_dfs:
            print("[viz] No numeric aunp_tm_BP_active_zone_*.star files found and _all.star fallback is disabled.")
            return None
    return pd.concat(star_dfs, ignore_index=True)

def load_fusion_points(tomo_path):
    """Load fusion points for vesicles within 10nm of active zone."""
    try:
        from scipy.spatial import KDTree
        from .aunps import compute_fusion_points
        
        print(f"Computing fusion points for {Path(tomo_path).name}...")
        fusion_points = compute_fusion_points(tomo_path)
        print(f"Computed {len(fusion_points)} fusion points")
        if len(fusion_points) > 0:
            print(f"Fusion points shape: {fusion_points.shape}")
            print(f"Fusion points z-range: {fusion_points[:, 2].min():.1f} to {fusion_points[:, 2].max():.1f}")
        return fusion_points
    except Exception as e:
        print(f'Could not load fusion points: {e}')
        import traceback
        traceback.print_exc()
        return None

def filter_near_slice(coords_list, z_center, z_thresh):
    """Filter a list of Nx3 arrays to only those with mean z within z_thresh of z_center."""
    filtered = []
    for arr in coords_list:
        if arr.shape[1] < 3:
            continue
        if np.abs(np.mean(arr[:,2]) - z_center) <= z_thresh:
            filtered.append(arr)
    return filtered

def filter_vesicles_near_slice(vesicles, z_center, z_thresh):
    """Filter vesicles whose center z is within z_thresh of z_center."""
    return [v for v in vesicles if abs(v['center'][2] - z_center) <= z_thresh]

def filter_aunps_near_slice(aunps, z_center, z_thresh):
    """Filter AuNPs with active_zone != -1 and z within z_thresh of z_center."""
    if aunps is None or aunps.empty:
        return None
    if 'active_zone' not in aunps.columns:
        return None
    mask = (aunps['active_zone'] != -1) & (np.abs(aunps['faCoordinateZ'] - z_center) <= z_thresh)
    filtered = aunps[mask]
    return filtered if not filtered.empty else None

def filter_vesicles_in_slice(vesicles, z_center, z_thresh):
    """Filter vesicles that have any point within z_thresh of z_center."""
    filtered = []
    for v in vesicles:
        vesicle_points = np.array(v.get('coordinates', []))
        if len(vesicle_points) == 0:
            continue
        # Check if any point of the vesicle is within the slice range
        z_coords = vesicle_points[:, 2]
        if np.any(np.abs(z_coords - z_center) <= z_thresh):
            filtered.append(v)
    return filtered

def filter_coords_in_slice(coords_list, z_center, z_thresh, max_segment_size=None):
    """Filter coordinates to only those within the slice range and below size threshold."""
    filtered = []
    for coords in coords_list:
        if coords.shape[1] < 3:
            continue
        # Only include points within the slice range
        mask = np.abs(coords[:, 2] - z_center) <= z_thresh
        if np.any(mask):
            segment_coords = coords[mask]
            
            # Filter by segment size if threshold is provided
            if max_segment_size is not None:
                # Calculate the maximum distance between any two points in the segment
                if len(segment_coords) > 1:
                    from scipy.spatial.distance import pdist
                    max_dist = np.max(pdist(segment_coords))
                    if max_dist <= max_segment_size:
                        filtered.append(segment_coords)
                else:
                    # Single point segments are always included
                    filtered.append(segment_coords)
            else:
                filtered.append(segment_coords)
    return filtered

def load_postsynaptic_active_zone_coords(tomo_path):
    """Load postsynaptic active zone coordinates."""
    az_dir = Path(tomo_path) / 'best_alignment' / 'STT_results' / 'active_zones'
    files = sorted(az_dir.glob('active_zone_pre*_post*_post.txt'))
    coords = [np.loadtxt(f) for f in files if f.exists()]
    return coords

def plot_tomogram_overlays(tomo_path, output_dir, aunp_active_zone_indices=None, rerun=False):
    """Generate 2D overlay plot and save to file. Optionally filter AuNPs by active zone indices."""
    vesicles = load_vesicles(tomo_path)
    pre_mem = load_membrane_coords(tomo_path, 'presynatptic')
    post_mem = load_membrane_coords(tomo_path, 'postsynaptic')
    azs_pre = load_active_zone_coords(tomo_path)
    azs_post = load_postsynaptic_active_zone_coords(tomo_path)
    aunps = load_aunps(tomo_path, aunp_active_zone_indices)
    fusion_points = load_fusion_points(tomo_path)
    
    # Debug: Check what was loaded
    print(f"Loaded {len(pre_mem)} presynaptic membrane files")
    print(f"Loaded {len(post_mem)} postsynaptic membrane files")
    print(f"Loaded {len(azs_pre)} presynaptic active zone files")
    print(f"Loaded {len(azs_post)} postsynaptic active zone files")
    print(f"Loaded {len(fusion_points) if fusion_points is not None else 0} fusion points")
    
    # Find z center of first active zone
    z_center = int(np.mean(azs_pre[0][:,2])) if azs_pre else None
    print(f"Using z_center: {z_center}")
    slice2d, zc = load_tomogram_slice(tomo_path, z_center)

    if slice2d is None:
        print(f"Could not load tomogram slice for {tomo_path}")
        return

    # Contrast adjustment: use 2nd and 98th percentiles for vmin/vmax
    vmin, vmax = np.percentile(slice2d, [2, 98])
    
    # Filter objects for the slice
    z_thresh = 5  # Increased from 2 to 5 pixels
    z_thresh_az = 1  # Stricter threshold for active zones
    z_thresh_aunps_fusion = 10  # 10 nm threshold for AuNPs and fusion sites
    z_thresh_vesicles = 1  # 1 pixel threshold - vesicle must intersect with slice
    vesicles_in_slice = filter_vesicles_in_slice(vesicles, z_center, z_thresh_vesicles)
    azs_pre_in_slice = filter_coords_in_slice(azs_pre, z_center, z_thresh_az, None)
    azs_post_in_slice = filter_coords_in_slice(azs_post, z_center, z_thresh_az, None)
    aunps_near = filter_aunps_near_slice(aunps, z_center, z_thresh_aunps_fusion)
    
    # Filter fusion points near the slice
    fusion_points_near = None
    if fusion_points is not None and len(fusion_points) > 0:
        mask = np.abs(fusion_points[:, 2] - z_center) <= z_thresh_aunps_fusion
        if np.any(mask):
            fusion_points_near = fusion_points[mask]
    
    # Debug output
    print(f"Found {len(azs_pre_in_slice)} presynaptic active zone segments in slice")
    print(f"Found {len(azs_post_in_slice)} postsynaptic active zone segments in slice")
    print(f"Found {len(vesicles_in_slice)} vesicles in slice")
    print(f"Found {len(fusion_points_near) if fusion_points_near is not None else 0} fusion points in slice")
    
    tomo_name = Path(tomo_path).name
    
    # Version 1: Vesicles and Active Zones
    output_file1 = output_dir / f"{tomo_name}_vesicles_active_zones.png"
    if output_file1.exists() and not rerun:
        print(f"Skipping {output_file1}, already exists.")
    else:
        fig1, ax1 = plt.subplots(figsize=(12, 12))
        ax1.imshow(slice2d, cmap='gray', vmin=vmin, vmax=vmax)
    
    # Overlay vesicles with transparency
    for v in vesicles_in_slice:
        c = np.array(v['center'])
        r = v['radius']
        circ = Circle((c[0], c[1]), r, color='pink', fill=False, lw=1.5, alpha=0.7, 
                     label='Vesicle' if 'Vesicle' not in [l.get_label() for l in ax1.get_legend_handles_labels()[0]] else '')
        ax1.add_patch(circ)
    
    # Highlight vesicles within 10 nm with transparency
    for v in vesicles_in_slice:
        if v.get('distance_to_az', 99) <= 10:
            c = np.array(v['center'])
            r = v['radius']
            circ = Circle((c[0], c[1]), r, color='aqua', fill=False, lw=2, alpha=0.8, 
                         label='<=10nm' if '<=10nm' not in [l.get_label() for l in ax1.get_legend_handles_labels()[0]] else '')
            ax1.add_patch(circ)
    
    # Overlay presynaptic active zone with transparent red
    for coords in azs_pre_in_slice:
        ax1.scatter(coords[:,0], coords[:,1], color='red', s=3, alpha=0.1, 
                label='Presynaptic Active Zone' if 'Presynaptic Active Zone' not in [l.get_label() for l in ax1.get_legend_handles_labels()[0]] else '')
    
    # Overlay postsynaptic active zone with transparent green
    for coords in azs_post_in_slice:
        ax1.scatter(coords[:,0], coords[:,1], color='green', s=3, alpha=0.1, 
                label='Postsynaptic Active Zone' if 'Postsynaptic Active Zone' not in [l.get_label() for l in ax1.get_legend_handles_labels()[0]] else '')
    
    # Add note about distance filtering to legend
    legend_elements = [
        Line2D([0], [0], color='pink', lw=1.5, label='Vesicles (intersecting slice)'),
        Line2D([0], [0], color='aqua', lw=2, label='Vesicles <10 nm from AZ'),
        Line2D([0], [0], color='red', lw=1.5, label='Presynaptic Active Zone'),
        Line2D([0], [0], color='green', lw=1.5, label='Postsynaptic Active Zone')
    ]
    ax1.legend(handles=legend_elements)
    ax1.set_title(f'Vesicles and Active Zones - {tomo_name}')
    ax1.set_xlabel('X (pixels)')
    ax1.set_ylabel('Y (pixels)')
    
    plt.savefig(output_file1, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved vesicles and active zones: {output_file1}")
    
    # Version 2: Vesicles and AuNPs
    output_file2 = output_dir / f"{tomo_name}_vesicles_aunps.png"
    if output_file2.exists() and not rerun:
        print(f"Skipping {output_file2}, already exists.")
    else:
        fig2, ax2 = plt.subplots(figsize=(12, 12))
        ax2.imshow(slice2d, cmap='gray', vmin=vmin, vmax=vmax)
    
    # Overlay vesicles with transparency
    for v in vesicles_in_slice:
        c = np.array(v['center'])
        r = v['radius']
        circ = Circle((c[0], c[1]), r, color='pink', fill=False, lw=1.5, alpha=0.7, 
                     label='Vesicle' if 'Vesicle' not in [l.get_label() for l in ax2.get_legend_handles_labels()[0]] else '')
        ax2.add_patch(circ)
    
    # Highlight vesicles within 10 nm with transparency
    for v in vesicles_in_slice:
        if v.get('distance_to_az', 99) <= 10:
            c = np.array(v['center'])
            r = v['radius']
            circ = Circle((c[0], c[1]), r, color='aqua', fill=False, lw=2, alpha=0.8, 
                         label='<=10nm' if '<=10nm' not in [l.get_label() for l in ax2.get_legend_handles_labels()[0]] else '')
            ax2.add_patch(circ)
    
    # Add AuNPs with transparency
    if aunps_near is not None:
        ax2.scatter(aunps_near['faCoordinateX'], aunps_near['faCoordinateY'], 
                  color='gold', s=30, alpha=0.8, label='AuNPs')
    
    # Add fusion points for vesicles within 10nm
    if fusion_points_near is not None and len(fusion_points_near) > 0:
        ax2.scatter(fusion_points_near[:, 0], fusion_points_near[:, 1], 
                   color='orange', s=100, alpha=0.9, marker='*', 
                   label='Fusion Sites' if 'Fusion Sites' not in [l.get_label() for l in ax2.get_legend_handles_labels()[0]] else '')
        print(f"Plotted {len(fusion_points_near)} fusion points within slice")
    
    # Only show fusion points for vesicles within 10nm that are also within 10 nm of slice
    vesicles_within_10nm = [v for v in vesicles_in_slice if v.get('distance_to_az', 99) <= 10]
    print(f"Found {len(vesicles_within_10nm)} vesicles within 10nm in slice")
    
    if vesicles_within_10nm and fusion_points is not None and len(fusion_points) > 0:
        print(f"Showing fusion points for vesicles within 10nm...")
        print(f"Fusion points z-range: {fusion_points[:, 2].min():.1f} to {fusion_points[:, 2].max():.1f}")
        
        # For each vesicle within 10nm, find its corresponding fusion point
        from scipy.spatial.distance import cdist
        vesicle_centers = np.array([v['center'] for v in vesicles_within_10nm])
        
        # Find the closest fusion point to each vesicle within 10nm
        if len(fusion_points) > 0:
            distances = cdist(vesicle_centers, fusion_points)
            closest_fusion_indices = np.argmin(distances, axis=1)
            
            # Plot the fusion point for each vesicle within 10nm, but only if fusion point is within 10 nm of slice
            plotted_fusion_points = set()
            for i, vesicle in enumerate(vesicles_within_10nm):
                fusion_point = fusion_points[closest_fusion_indices[i]]
                fusion_point_tuple = tuple(fusion_point)
                
                # Only plot if this fusion point is within 10 nm of the slice and hasn't been plotted yet
                if (abs(fusion_point[2] - z_center) <= z_thresh_aunps_fusion and 
                    fusion_point_tuple not in plotted_fusion_points):
                    ax2.scatter(fusion_point[0], fusion_point[1], 
                               color='orange', s=100, alpha=0.9, marker='*')
                    plotted_fusion_points.add(fusion_point_tuple)
            
            print(f"Plotted {len(plotted_fusion_points)} fusion points for vesicles within 10nm and within 10 nm of slice")
    elif vesicles_within_10nm:
        print("No fusion points available to plot for vesicles within 10nm")
    else:
        print("No vesicles within 10nm found in slice")
    
    # Add note about distance filtering to legend
    legend_elements = [
        Line2D([0], [0], color='pink', lw=1.5, label='Vesicles (intersecting slice)'),
        Line2D([0], [0], color='aqua', lw=2, label='Vesicles <10 nm from AZ'),
        plt.scatter([], [], color='gold', s=30, label='AuNPs'),
        plt.scatter([], [], color='orange', s=100, marker='*', label='Fusion Sites')
    ]
    ax2.legend(handles=legend_elements)
    ax2.set_title(f'Vesicles and AuNPs - {tomo_name}')
    ax2.set_xlabel('X (pixels)')
    ax2.set_ylabel('Y (pixels)')
    
    plt.savefig(output_file2, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved vesicles and AuNPs: {output_file2}")
    
    # Version 3: Combined - Vesicles, Active Zones, AuNPs, and Fusion Sites
    output_file3 = output_dir / f"{tomo_name}_combined.png"
    if output_file3.exists() and not rerun:
        print(f"Skipping {output_file3}, already exists.")
    else:
        fig3, ax3 = plt.subplots(figsize=(12, 12))
        ax3.imshow(slice2d, cmap='gray', vmin=vmin, vmax=vmax)
    
    # Overlay vesicles with transparency
    for v in vesicles_in_slice:
        c = np.array(v['center'])
        r = v['radius']
        circ = Circle((c[0], c[1]), r, color='pink', fill=False, lw=1.5, alpha=0.7, 
                     label='Vesicle' if 'Vesicle' not in [l.get_label() for l in ax3.get_legend_handles_labels()[0]] else '')
        ax3.add_patch(circ)
    
    # Highlight vesicles within 10 nm with transparency
    for v in vesicles_in_slice:
        if v.get('distance_to_az', 99) <= 10:
            c = np.array(v['center'])
            r = v['radius']
            circ = Circle((c[0], c[1]), r, color='aqua', fill=False, lw=2, alpha=0.8, 
                         label='<=10nm' if '<=10nm' not in [l.get_label() for l in ax3.get_legend_handles_labels()[0]] else '')
            ax3.add_patch(circ)
    
    # Overlay presynaptic active zone with transparent red
    for coords in azs_pre_in_slice:
        ax3.scatter(coords[:,0], coords[:,1], color='red', s=3, alpha=0.1, 
                label='Presynaptic Active Zone' if 'Presynaptic Active Zone' not in [l.get_label() for l in ax3.get_legend_handles_labels()[0]] else '')
    
    # Overlay postsynaptic active zone with transparent green
    for coords in azs_post_in_slice:
        ax3.scatter(coords[:,0], coords[:,1], color='green', s=3, alpha=0.1, 
                label='Postsynaptic Active Zone' if 'Postsynaptic Active Zone' not in [l.get_label() for l in ax3.get_legend_handles_labels()[0]] else '')
    
    # Add AuNPs with transparency
    if aunps_near is not None:
        ax3.scatter(aunps_near['faCoordinateX'], aunps_near['faCoordinateY'], 
                  color='gold', s=30, alpha=0.8, label='AuNPs')
    
    # Add fusion points for vesicles within 10nm
    if fusion_points_near is not None and len(fusion_points_near) > 0:
        ax3.scatter(fusion_points_near[:, 0], fusion_points_near[:, 1], 
                   color='orange', s=100, alpha=0.9, marker='*', 
                   label='Fusion Sites' if 'Fusion Sites' not in [l.get_label() for l in ax3.get_legend_handles_labels()[0]] else '')
        print(f"Plotted {len(fusion_points_near)} fusion points within slice")
    
    # Only show fusion points for vesicles within 10nm that are also within 10 nm of slice
    vesicles_within_10nm = [v for v in vesicles_in_slice if v.get('distance_to_az', 99) <= 10]
    print(f"Found {len(vesicles_within_10nm)} vesicles within 10nm in slice")
    
    if vesicles_within_10nm and fusion_points is not None and len(fusion_points) > 0:
        print(f"Showing fusion points for vesicles within 10nm...")
        print(f"Fusion points z-range: {fusion_points[:, 2].min():.1f} to {fusion_points[:, 2].max():.1f}")
        
        # For each vesicle within 10nm, find its corresponding fusion point
        from scipy.spatial.distance import cdist
        vesicle_centers = np.array([v['center'] for v in vesicles_within_10nm])
        
        # Find the closest fusion point to each vesicle within 10nm
        if len(fusion_points) > 0:
            distances = cdist(vesicle_centers, fusion_points)
            closest_fusion_indices = np.argmin(distances, axis=1)
            
            # Plot the fusion point for each vesicle within 10nm, but only if fusion point is within 10 nm of slice
            plotted_fusion_points = set()
            for i, vesicle in enumerate(vesicles_within_10nm):
                fusion_point = fusion_points[closest_fusion_indices[i]]
                fusion_point_tuple = tuple(fusion_point)
                
                # Only plot if this fusion point is within 10 nm of the slice and hasn't been plotted yet
                if (abs(fusion_point[2] - z_center) <= z_thresh_aunps_fusion and 
                    fusion_point_tuple not in plotted_fusion_points):
                    ax3.scatter(fusion_point[0], fusion_point[1], 
                               color='orange', s=100, alpha=0.9, marker='*')
                    plotted_fusion_points.add(fusion_point_tuple)
            
            print(f"Plotted {len(plotted_fusion_points)} fusion points for vesicles within 10nm and within 10 nm of slice")
    elif vesicles_within_10nm:
        print("No fusion points available to plot for vesicles within 10nm")
    else:
        print("No vesicles within 10nm found in slice")
    
    # Add note about distance filtering to legend
    legend_elements = [
        Line2D([0], [0], color='pink', lw=1.5, label='Vesicles (intersecting slice)'),
        Line2D([0], [0], color='aqua', lw=2, label='Vesicles <10 nm from AZ'),
        Line2D([0], [0], color='red', lw=1.5, label='Presynaptic Active Zone'),
        Line2D([0], [0], color='green', lw=1.5, label='Postsynaptic Active Zone'),
        plt.scatter([], [], color='gold', s=30, label='AuNPs'),
        plt.scatter([], [], color='orange', s=100, marker='*', label='Fusion Sites')
    ]
    ax3.legend(handles=legend_elements)
    ax3.set_title(f'Combined - Vesicles, Active Zones, AuNPs, and Fusion Sites - {tomo_name}')
    ax3.set_xlabel('X (pixels)')
    ax3.set_ylabel('Y (pixels)')
    
    plt.savefig(output_file3, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved combined visualization: {output_file3}")

    # Version 4: Vesicles colored by average signal intensity
    output_file4 = output_dir / f"{tomo_name}_vesicles_signal.png"
    if output_file4.exists() and not rerun:
        print(f"Skipping {output_file4}, already exists.")
    else:
        fig4, ax4 = plt.subplots(figsize=(12, 12))
        ax4.imshow(slice2d, cmap='gray', vmin=vmin, vmax=vmax)
        # Gather signal values for color mapping
        vesicle_signals = [v.get('average_signal', 0.0) for v in vesicles_in_slice]
        # Normalize signals per tomogram
        if len(vesicle_signals) == 0 or np.all(np.array(vesicle_signals) == vesicle_signals[0]):
            norm = plt.Normalize(vmin=0, vmax=1)
            colors = ['gray'] * len(vesicles_in_slice)
            vesicle_signals_norm = vesicle_signals
        else:
            min_signal = min(vesicle_signals)
            max_signal = max(vesicle_signals)
            if max_signal > min_signal:
                vesicle_signals_norm = [(s - min_signal) / (max_signal - min_signal) for s in vesicle_signals]
            else:
                vesicle_signals_norm = [0.0 for s in vesicle_signals]
            norm = plt.Normalize(vmin=0, vmax=1)
            cmap = plt.get_cmap('plasma')
            colors = [cmap(norm(s)) for s in vesicle_signals_norm]
        # Draw filled vesicles
        for v, color in zip(vesicles_in_slice, colors):
            c = np.array(v['center'])
            r = v['radius']
            circ = Circle((c[0], c[1]), r, color=color, fill=True, lw=0, alpha=0.8)
            ax4.add_patch(circ)
        # Overlay vesicle outlines as in original
        for v in vesicles_in_slice:
            c = np.array(v['center'])
            r = v['radius']
            circ = Circle((c[0], c[1]), r, color='black', fill=False, lw=1, alpha=0.7)
            ax4.add_patch(circ)
        # Overlay presynaptic active zone with transparent red
        for coords in azs_pre_in_slice:
            ax4.scatter(coords[:,0], coords[:,1], color='red', s=3, alpha=0.1)
        # Overlay postsynaptic active zone with transparent green
        for coords in azs_post_in_slice:
            ax4.scatter(coords[:,0], coords[:,1], color='green', s=3, alpha=0.1)
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap='plasma', norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax4, fraction=0.046, pad=0.04)
        cbar.set_label('Normalized Vesicle Signal Intensity (per tomogram)')
        ax4.set_title(f'Vesicles Colored by Normalized Signal Intensity - {tomo_name}')
        ax4.set_xlabel('X (pixels)')
        ax4.set_ylabel('Y (pixels)')
        plt.savefig(output_file4, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved vesicles colored by signal: {output_file4}")

    # --- AuNP Cluster Visualization ---
    # Try to load cluster assignments from aunp_clusters.star or aunp_nearest_neighbor_distances.csv
    import starfile
    aunps_results_dir = Path(tomo_path) / "best_alignment" / "STT_results" / "aunps"
    cluster_star = aunps_results_dir / "aunp_clusters.star"
    cluster_csv = aunps_results_dir / "aunp_nearest_neighbor_distances.csv"
    aunp_clusters = None
    if cluster_star.exists():
        try:
            aunp_clusters = starfile.read(cluster_star)
        except Exception:
            aunp_clusters = None
    if aunp_clusters is None and cluster_csv.exists():
        try:
            aunp_clusters = pd.read_csv(cluster_csv)
        except Exception:
            aunp_clusters = None
    if aunp_clusters is not None and not aunp_clusters.empty:
        # Assign colors to clusters
        import matplotlib.colors as mcolors
        import matplotlib.cm as cm
        clusters = aunp_clusters['aunp_cluster'].values
        unique_clusters = np.unique(clusters)
        n_clusters = len(unique_clusters[unique_clusters != -1])
        
        # Use a colormap that can handle more clusters and ensure distinct colors
        if n_clusters <= 20:
            cmap = cm.get_cmap('tab20', n_clusters)
        else:
            # For more than 20 clusters, use a different approach
            # Create a custom colormap that cycles through distinct colors
            import matplotlib.colors as mcolors
            base_colors = plt.cm.tab20(np.linspace(0, 1, 20))
            # Add more distinct colors by mixing and varying saturation
            additional_colors = []
            for i in range(n_clusters - 20):
                # Create variations of existing colors
                base_idx = i % 20
                base_color = base_colors[base_idx]
                # Vary saturation and lightness
                hsv = mcolors.rgb_to_hsv(base_color[:3])
                hsv[1] = min(1.0, hsv[1] * (0.7 + 0.3 * (i // 20)))  # Vary saturation
                hsv[2] = min(1.0, hsv[2] * (0.8 + 0.2 * (i % 3)))     # Vary lightness
                new_color = mcolors.hsv_to_rgb(hsv)
                additional_colors.append(np.append(new_color, 1.0))
            
            all_colors = np.vstack([base_colors, additional_colors])
            cmap = mcolors.ListedColormap(all_colors[:n_clusters])
        
        cluster_color_map = {c: cmap(i) for i, c in enumerate(unique_clusters) if c != -1}
        cluster_color_map[-1] = (0.5, 0.5, 0.5, 1.0)  # grey for noise
        colors = [cluster_color_map.get(c, (0.5, 0.5, 0.5, 1.0)) for c in clusters]
        # 1. Overlay all AuNPs on the combined visualization, colored by cluster
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(slice2d, cmap='gray', vmin=vmin, vmax=vmax)
        # Plot all AuNPs
        ax.scatter(aunp_clusters['faCoordinateX'], aunp_clusters['faCoordinateY'],
                   c=colors, s=40, edgecolor='k', linewidth=0.5, alpha=0.9, label='AuNPs (clustered)')
        ax.set_title(f"{tomo_name} - Combined Overlay with AuNP Clusters")
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        # Legend for clusters
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=cluster_color_map[c], edgecolor='k',
                                 label=f'Cluster {c}' if c != -1 else 'Noise')
                          for c in unique_clusters]
        ax.legend(handles=legend_elements, loc='best')
        output_dir_viz = Path('results/visualizations/aunps_and_vesicles')
        output_dir_viz.mkdir(parents=True, exist_ok=True)
        out_combined = output_dir_viz / f"{tomo_name}_combined_aunpclusters.png"
        if out_combined.exists() and not rerun:
            print(f"Skipping {out_combined}, already exists.")
        else:
            plt.savefig(out_combined, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved combined AuNP cluster overlay: {out_combined}")
        # 2. Save a separate image showing all AuNPs colored by cluster, best 2D projection
        coords = np.stack([aunp_clusters['faCoordinateX'],
                          aunp_clusters['faCoordinateY'],
                          aunp_clusters['faCoordinateZ']], axis=1)
        # Find projection with largest spread
        spreads = [np.ptp(coords[:, i]) for i in range(3)]
        proj_pairs = [(0, 1), (0, 2), (1, 2)]
        proj_spreads = [spreads[i] + spreads[j] for i, j in proj_pairs]
        best_proj = proj_pairs[np.argmax(proj_spreads)]
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.scatter(coords[:, best_proj[0]], coords[:, best_proj[1]],
                   c=colors, s=40, edgecolor='k', linewidth=0.5, alpha=0.9)
        ax.set_xlabel(['X', 'Y', 'Z'][best_proj[0]])
        ax.set_ylabel(['X', 'Y', 'Z'][best_proj[1]])
        ax.set_title(f"{tomo_name} - AuNP Clusters (Best 2D Projection)")
        ax.legend(handles=legend_elements, loc='best')
        out_clusters = output_dir_viz / f"{tomo_name}_aunpclusters.png"
        if out_clusters.exists() and not rerun:
            print(f"Skipping {out_clusters}, already exists.")
        else:
            plt.savefig(out_clusters, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved AuNP cluster summary image: {out_clusters}")
        # Save also to the tomogram's own visualization directory
        tomo_viz_dir = Path(tomo_path) / "best_alignment" / "STT_results" / "visualizations"
        tomo_viz_dir.mkdir(parents=True, exist_ok=True)
        out_combined_tomo = tomo_viz_dir / f"{tomo_name}_combined_aunpclusters.png"
        out_clusters_tomo = tomo_viz_dir / f"{tomo_name}_aunpclusters.png"
        # Save the same figures to the tomogram's visualization directory
        plt.figure(figsize=(12, 12))
        plt.imshow(slice2d, cmap='gray', vmin=vmin, vmax=vmax)
        plt.scatter(aunp_clusters['faCoordinateX'], aunp_clusters['faCoordinateY'],
                    c=colors, s=40, edgecolor='k', linewidth=0.5, alpha=0.9, label='AuNPs (clustered)')
        plt.title(f"{tomo_name} - Combined Overlay with AuNP Clusters")
        plt.xlabel('X (pixels)')
        plt.ylabel('Y (pixels)')
        plt.legend(handles=legend_elements, loc='best')
        plt.savefig(out_combined_tomo, dpi=300, bbox_inches='tight')
        plt.close()
        plt.figure(figsize=(10, 10))
        plt.scatter(coords[:, best_proj[0]], coords[:, best_proj[1]],
                    c=colors, s=40, edgecolor='k', linewidth=0.5, alpha=0.9)
        plt.xlabel(['X', 'Y', 'Z'][best_proj[0]])
        plt.ylabel(['X', 'Y', 'Z'][best_proj[1]])
        plt.title(f"{tomo_name} - AuNP Clusters (Best 2D Projection)")
        plt.legend(handles=legend_elements, loc='best')
        plt.savefig(out_clusters_tomo, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Also saved cluster visualizations to {tomo_viz_dir}")
    # --- End AuNP Cluster Visualization ---

def generate_summary_figures():
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    # Use seaborn-v0_8-whitegrid style if available, else fallback to ggplot
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except Exception:
        try:
            plt.style.use('ggplot')
            print("Warning: 'seaborn-v0_8-whitegrid' style not found. Using 'ggplot' instead.")
        except Exception:
            print("Warning: Could not set preferred matplotlib style. Using default.")
    summary_dir = os.path.join('results', 'summary_pdfs')
    os.makedirs(summary_dir, exist_ok=True)

    # Define colors for different sets
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F']
    
    def create_colored_boxplot(df, column, by_column, title, ylabel, filename):
        """Create a box plot with colored, transparent boxes for each set."""
        plt.figure(figsize=(10, 6))
        
        # Get unique sets and their data
        sets = df[by_column].unique()
        data_by_set = [df[df[by_column] == set_name][column].dropna() for set_name in sets]
        
        # Convert to Angstroms if this is a distance metric
        if 'nearest_neighbor_distance' in column or 'distance' in column:
            data_by_set = [data * 10 for data in data_by_set]  # Convert nm to Angstroms
            ylabel += ' (Å)'
        
        # Create box plot with custom colors
        bp = plt.boxplot(data_by_set, labels=sets, patch_artist=True)
        
        # Color the boxes with transparency
        for i, patch in enumerate(bp['boxes']):
            color = colors[i % len(colors)]
            patch.set_facecolor(color)
            patch.set_alpha(0.7)  # Transparency
            patch.set_edgecolor('black')
            patch.set_linewidth(1.5)
        
        # Style the other elements
        for element in ['whiskers', 'caps', 'medians']:
            plt.setp(bp[element], color='black', linewidth=1.5)
        
        for flier in bp['fliers']:
            flier.set(marker='o', markerfacecolor='red', markersize=4, alpha=0.7)
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Set', fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(summary_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()

    # --- AuNP Results ---
    aunp_path = os.path.join('results', 'aunps_results.csv')
    if os.path.exists(aunp_path):
        df_aunp = pd.read_csv(aunp_path)
        if 'set_name' in df_aunp.columns:
            for metric in ['aunp_analysis_aunp_count', 'aunp_analysis_nearest_neighbor_distance_mean']:
                if metric in df_aunp.columns:
                    title = f'{metric.replace("aunp_analysis_", "").replace("_", " ").title()} by Set'
                    ylabel = metric.replace('aunp_analysis_', '').replace('_', ' ').title()
                    filename = f'aunp_{metric.replace("aunp_analysis_", "")}_by_set.png'
                    create_colored_boxplot(df_aunp, metric, 'set_name', title, ylabel, filename)
            
            # New: AuNP density and avg distance to AZ center
            for metric in ['aunp_density', 'distance_to_active_zone_center_mean']:
                if metric in df_aunp.columns:
                    title = f'{metric.replace("_", " ").title()} by Set'
                    ylabel = metric.replace('_', ' ').title()
                    filename = f'aunp_{metric}_by_set.png'
                    create_colored_boxplot(df_aunp, metric, 'set_name', title, ylabel, filename)

    # --- AuNP Cluster Results ---
    cluster_path = os.path.join('results', 'aunp_cluster_results.csv')
    if os.path.exists(cluster_path):
        df_cluster = pd.read_csv(cluster_path)
        if 'set_name' in df_cluster.columns:
            for metric in ['n_aunps', 'cluster_area', 'cluster_density']:
                if metric in df_cluster.columns:
                    title = f'{metric.replace("_", " ").title()} by Set'
                    ylabel = metric.replace('_', ' ').title()
                    filename = f'aunp_cluster_{metric}_by_set.png'
                    create_colored_boxplot(df_cluster, metric, 'set_name', title, ylabel, filename)
            
            # Also plot number of clusters per tomogram by set
            if 'tomogram_name' in df_cluster.columns:
                cluster_counts = df_cluster.groupby(['set_name', 'tomogram_name']).size().reset_index(name='n_clusters')
                title = 'Number of AuNP Clusters per Tomogram by Set'
                ylabel = 'Number of Clusters'
                filename = 'aunp_cluster_count_by_set.png'
                create_colored_boxplot(cluster_counts, 'n_clusters', 'set_name', title, ylabel, filename)

    # --- Vesicle Results ---
    vesicle_path = os.path.join('results', 'vesicles_results.csv')
    if os.path.exists(vesicle_path):
        df_ves = pd.read_csv(vesicle_path)
        if 'set_name' in df_ves.columns:
            for metric in ['vesicle_detection_vesicle_count', 'vesicle_detection_nearby_vesicle_count', 'vesicle_detection_average_vesicle_diameter']:
                if metric in df_ves.columns:
                    title = f'{metric.replace("_", " ").title()} by Set'
                    ylabel = metric.replace('_', ' ').title()
                    filename = f'vesicle_{metric}_by_set.png'
                    create_colored_boxplot(df_ves, metric, 'set_name', title, ylabel, filename)

    # --- Active Zone Results ---
    az_path = os.path.join('results', 'activezone_results.csv')
    if os.path.exists(az_path):
        df_az = pd.read_csv(az_path)
        if 'set_name' in df_az.columns:
            for metric in ['active_zone_count', 'avg_active_zone_area', 'average_cleft_width']:
                title = f'{metric.replace("_", " ").title()} by Set'
                ylabel = metric.replace('_', ' ').title()
                filename = f'activezone_{metric}_by_set.png'
                create_colored_boxplot(df_az, metric, 'set_name', title, ylabel, filename)

    print(f"Summary figures saved to {summary_dir}")

def main():
    """Main function to process all tomograms and generate visualizations."""
    parser = argparse.ArgumentParser(description='Generate synaptic tomogram visualizations')
    parser.add_argument('--data-dir', default='../data/', help='Path to data directory')
    parser.add_argument('--output-dir', default='./visualization_output', help='Output directory for figures')
    parser.add_argument('--tomo-name', help='Process only specific tomogram (optional)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Find tomograms
    tomogram_paths = find_analyzed_tomograms(args.data_dir)
    
    if not tomogram_paths:
        print("No analyzed tomograms found!")
        return
    
    print(f"Found {len(tomogram_paths)} analyzed tomograms")
    
    # Filter by specific tomogram if requested
    if args.tomo_name:
        tomogram_paths = [p for p in tomogram_paths if args.tomo_name in p]
        if not tomogram_paths:
            print(f"No tomograms found matching '{args.tomo_name}'")
            return
    
    # Process each tomogram
    for i, tomo_path in enumerate(tomogram_paths, 1):
        print(f"\nProcessing tomogram {i}/{len(tomogram_paths)}: {Path(tomo_path).name}")
        
        try:
            # Generate 2D overlay
            plot_tomogram_overlays(tomo_path, output_dir)
            # 3D overlay removed
        except Exception as e:
            print(f"Error processing {tomo_path}: {e}")
            continue
    
    print(f"\nVisualization complete! Figures saved to: {output_dir.absolute()}")

    # Generate summary figures for all sets/metrics
    generate_summary_figures()

if __name__ == "__main__":
    main() 