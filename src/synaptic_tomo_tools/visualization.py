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
from scipy.spatial import KDTree

# Import from the same package
from .activezone import extract_active_zonogram

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

def load_tomogram_slice(tomo_path, z_center=None, alignment_dir='best_alignment'):
    """Load a 2D slice from the tomogram."""
    if mrcfile is None:
        print("mrcfile not available, skipping tomogram slice loading")
        return None, None
    
    mrcs = list((Path(tomo_path) / alignment_dir).glob('*ddw.mrc'))
    if not mrcs:
        return None, None
    with mrcfile.open(mrcs[0], 'r') as mrc:
        data = mrc.data
    if z_center is None:
        z_center = data.shape[0] // 2
    return data[z_center], z_center

def load_membrane_coords(tomo_path, kind='presynaptic', alignment_dir='best_alignment'):
    """Load membrane coordinates from text files."""
    aunps_dir = Path(tomo_path) / alignment_dir / 'aunps'
    files = sorted(aunps_dir.glob(f'{kind}membranes_*.txt'))
    coords = [np.loadtxt(f) for f in files if f.exists()]
    return coords

def load_active_zone_coords(tomo_path, alignment_dir='best_alignment'):
    """Load active zone coordinates."""
    az_dir = Path(tomo_path) / alignment_dir / 'STT_results' / 'activezone'
    files = sorted(az_dir.glob('active_zone_pre*_post*_pre_outer.txt'))
    coords = [np.loadtxt(f) for f in files if f.exists()]
    return coords

def load_vesicles(tomo_path, alignment_dir='best_alignment'):
    """Load vesicle data from JSON file."""
    ves_file = Path(tomo_path) / alignment_dir / 'STT_results' / 'vesicles' / 'vesicle_results.json'
    with open(ves_file) as f:
        data = json.load(f)
    return data['vesicles']

def load_aunps(tomo_path, active_zone_indices=None, alignment_dir='best_alignment'):
    """Load AuNP coordinates from filtered aunp_clusters.star file, optionally filtered by active_zone_indices."""
    aunps_results_dir = Path(tomo_path) / alignment_dir / 'STT_results' / 'aunps'
    import starfile
    import pandas as pd
    
    # Load from the filtered output file (aunp_clusters.star)
    cluster_star = aunps_results_dir / "aunp_clusters.star"
    
    if not cluster_star.exists():
        print(f"[viz] Warning: Filtered AuNP file not found at {cluster_star}")
        print("[viz] Falling back to original input files (this should not happen if analysis was run)")
        # Fallback to original files if filtered file doesn't exist (for backward compatibility)
        aunps_dir = Path(tomo_path) / alignment_dir / 'aunps'
        import glob
        import re
        star_dfs = []
        if active_zone_indices is not None:
            for idx in active_zone_indices:
                star_file = aunps_dir / f"aunp_tm_BP_active_zone_{idx}.star"
                print("[viz] Fallback: Trying to load:", star_file)
                if star_file.exists():
                    star_data = starfile.read(star_file)
                    if isinstance(star_data, dict):
                        for v in star_data.values():
                            if isinstance(v, pd.DataFrame):
                                v = v.copy()
                                if 'active_zone' not in v.columns:
                                    v['active_zone'] = idx
                                star_dfs.append(v)
                                break
                    elif isinstance(star_data, pd.DataFrame):
                        star_data = star_data.copy()
                        if 'active_zone' not in star_data.columns:
                            star_data['active_zone'] = idx
                        star_dfs.append(star_data)
        else:
            pattern = str(aunps_dir / "aunp_tm_BP_active_zone_*.star")
            for file in glob.glob(pattern):
                fname = Path(file).name
                m = re.match(r"aunp_tm_BP_active_zone_(\d+)\.star", fname)
                if m:
                    az_id = int(m.group(1))
                    star_data = starfile.read(Path(file))
                    if isinstance(star_data, dict):
                        for v in star_data.values():
                            if isinstance(v, pd.DataFrame):
                                v = v.copy()
                                if 'active_zone' not in v.columns:
                                    v['active_zone'] = az_id
                                star_dfs.append(v)
                                break
                    elif isinstance(star_data, pd.DataFrame):
                        star_data = star_data.copy()
                        if 'active_zone' not in star_data.columns:
                            star_data['active_zone'] = az_id
                        star_dfs.append(star_data)
        if not star_dfs:
            print("[viz] No AuNP files found.")
            return None
        return pd.concat(star_dfs, ignore_index=True)
    
    # Load filtered AuNP data
    print(f"[viz] Loading filtered AuNPs from: {cluster_star}")
    star_data = starfile.read(cluster_star)
    
    # Handle both dict and DataFrame formats
    if isinstance(star_data, dict):
        # Extract DataFrame from dict (usually has 'particles' key or similar)
        for v in star_data.values():
            if isinstance(v, pd.DataFrame):
                df = v.copy()
                break
        else:
            print("[viz] Error: Could not find DataFrame in star file dict")
            return None
    elif isinstance(star_data, pd.DataFrame):
        df = star_data.copy()
    else:
        print("[viz] Error: Unexpected star file format")
        return None
    
    # Filter by active_zone_indices if specified
    if active_zone_indices is not None:
        if 'active_zone' not in df.columns:
            print("[viz] Warning: 'active_zone' column not found in filtered AuNP file")
            return None
        
        # Convert active_zone column to int if needed, and handle any NaN values
        df['active_zone'] = pd.to_numeric(df['active_zone'], errors='coerce').astype('Int64')
        
        # Show what active zones are actually in the file for debugging
        unique_azs = sorted(df['active_zone'].dropna().unique().tolist())
        print(f"[viz] Active zones in file: {unique_azs}, filtering for: {active_zone_indices}")
        
        # Filter by active zone indices (convert to same type for comparison)
        active_zone_indices_int = [int(az) for az in active_zone_indices]
        df_filtered = df[df['active_zone'].isin(active_zone_indices_int)].copy()
        
        if len(df_filtered) == 0:
            # If no AuNPs found, check if all active_zone values are 0 (common issue from old data)
            # This can happen if the input files had active_zone=0 instead of the file index
            unique_azs = sorted(df['active_zone'].dropna().unique().tolist())
            if len(unique_azs) == 1 and unique_azs[0] == 0:
                print(f"[viz] Warning: All AuNPs have active_zone=0, but filtering for {active_zone_indices}")
                print(f"[viz] This suggests the analysis needs to be re-run with the fixed active_zone assignment.")
                print(f"[viz] Returning empty result - please re-run AuNP analysis to fix active_zone values.")
            else:
                print(f"[viz] Warning: No AuNPs found in active zones {active_zone_indices}")
                print(f"[viz] Available active zones in file: {unique_azs}")
        
        df = df_filtered
        print(f"[viz] Filtered to {len(df)} AuNPs in active zones {active_zone_indices}")
    
    return df

def load_fusion_points(tomo_path):
    """Load fusion points for vesicles within 20nm of active zone."""
    try:
        from scipy.spatial import KDTree
        from .aunps import compute_fusion_points
        
        fusion_points = compute_fusion_points(tomo_path, vesicle_distance_threshold=20.0)
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
    az_dir = Path(tomo_path) / 'best_alignment' / 'STT_results' / 'activezone'
    files = sorted(az_dir.glob('active_zone_pre*_post*_post_outer.txt'))
    coords = [np.loadtxt(f) for f in files if f.exists()]
    return coords


def _load_optional_az_surface_txt(path: Path) -> np.ndarray:
    """Load Nx3 coordinates from an active-zone surface txt; return empty (0, 3) if missing or invalid."""
    if not path.exists():
        return np.zeros((0, 3))
    try:
        arr = np.loadtxt(path, delimiter=None)
        arr = np.atleast_2d(arr)
        if arr.size == 0:
            return np.zeros((0, 3))
        if arr.shape[1] < 3:
            return np.zeros((0, 3))
        return arr
    except Exception:
        return np.zeros((0, 3))


def load_specific_active_zone_coords(tomo_path, active_zone_indices, aunps, alignment_dir: str = "best_alignment"):
    """Load outer and inner active zone coordinates for the given indices, using saved mapping."""
    from .activezone import load_active_zone_mapping
    
    az_dir = Path(tomo_path) / alignment_dir / "STT_results" / "activezone"
    
    azs_pre = []
    azs_post = []
    azs_pre_inner = []
    azs_post_inner = []
    
    if active_zone_indices is not None:
        # Load saved mapping
        az_mapping = load_active_zone_mapping(tomo_path, alignment_dir)
        
        if not az_mapping:
            # No mapping found - use all available zone files as fallback but print error
            print(f"No saved active zone mapping found for {Path(tomo_path).name}. Active zone analysis must be run first with smart matching to create the mapping.")
            print(f"FALLBACK: Loading all available active zone files (no filtering applied).")
            # Load all available zone files
            pre_files = sorted(list(az_dir.glob('active_zone_pre*_post*_pre_outer.txt')))
            post_files = sorted(list(az_dir.glob('active_zone_pre*_post*_post_outer.txt')))
            
            # Group files by active zone name to ensure paired matching
            active_zone_groups = {}
            for pre_file in pre_files:
                zone_name = pre_file.name.replace('_pre_outer.txt', '')
                if zone_name not in active_zone_groups:
                    active_zone_groups[zone_name] = {'pre': None, 'post': None}
                active_zone_groups[zone_name]['pre'] = pre_file
            
            for post_file in post_files:
                zone_name = post_file.name.replace('_post_outer.txt', '')
                if zone_name not in active_zone_groups:
                    active_zone_groups[zone_name] = {'pre': None, 'post': None}
                active_zone_groups[zone_name]['post'] = post_file
            
            # Load all paired zones
            for zone_name, files in active_zone_groups.items():
                if files['pre'] is not None and files['post'] is not None:
                    try:
                        pre_coords = np.loadtxt(files['pre'])
                        post_coords = np.loadtxt(files['post'])
                        if pre_coords.size > 0 and post_coords.size > 0:
                            azs_pre.append(pre_coords)
                            azs_post.append(post_coords)
                            azs_pre_inner.append(_load_optional_az_surface_txt(az_dir / f"{zone_name}_pre_inner.txt"))
                            azs_post_inner.append(_load_optional_az_surface_txt(az_dir / f"{zone_name}_post_inner.txt"))
                    except Exception as e:
                        print(f"Warning: Error loading {zone_name}: {e}")
        else:
            # Convert string keys to int (JSON stores dict keys as strings)
            az_mapping = {int(k): v for k, v in az_mapping.items()}
            
            # Use saved mapping to load zones directly
            for az_id in active_zone_indices:
                if az_id in az_mapping:
                    zone_name = az_mapping[az_id]
                    pre_file = az_dir / f"{zone_name}_pre_outer.txt"
                    post_file = az_dir / f"{zone_name}_post_outer.txt"
                    
                    if pre_file.exists() and post_file.exists():
                        try:
                            pre_coords = np.loadtxt(pre_file)
                            post_coords = np.loadtxt(post_file)
                            azs_pre.append(pre_coords)
                            azs_post.append(post_coords)
                            azs_pre_inner.append(_load_optional_az_surface_txt(az_dir / f"{zone_name}_pre_inner.txt"))
                            azs_post_inner.append(_load_optional_az_surface_txt(az_dir / f"{zone_name}_post_inner.txt"))
                        except Exception as e:
                            raise ValueError(f"Error loading zone {zone_name} from saved mapping: {e}")
                    else:
                        raise ValueError(f"Files not found for zone {zone_name} from saved mapping. Expected files: {pre_file} and {post_file}")
                else:
                    raise ValueError(f"Active zone index {az_id} not found in saved mapping. This indicates the active zone analysis was run with different indices.")
    
    return azs_pre, azs_post, azs_pre_inner, azs_post_inner

def plot_tomogram_overlays(tomo_path, output_dir, aunp_active_zone_indices=None, rerun=False, alignment_dir='best_alignment',
                           sphere_size=None, sphere_color=None, aunp_distance_min=None, aunp_distance_max=None,
                           aunp_distance_cutoff_direction=None, aunp_distance_cutoff_value=None):
    """Generate 2D overlay plot and save to file. Only processes CSV-specified active zones."""
    vesicles = load_vesicles(tomo_path, alignment_dir=alignment_dir)
    pre_mem = load_membrane_coords(tomo_path, 'presynatptic', alignment_dir=alignment_dir)
    post_mem = load_membrane_coords(tomo_path, 'postsynaptic', alignment_dir=alignment_dir)
    aunps = load_aunps(tomo_path, aunp_active_zone_indices, alignment_dir=alignment_dir)
    fusion_points = load_fusion_points(tomo_path)
    
    # Process active zones - auto-detect if none specified in CSV
    if aunp_active_zone_indices is None or len(aunp_active_zone_indices) == 0:
        print("No active zones specified in CSV, auto-detecting all available active zones")
        # Auto-detect all available active zone numbers from filtered AuNP file
        aunps_results_dir = Path(tomo_path) / "best_alignment" / "STT_results" / "aunps"
        cluster_star = aunps_results_dir / "aunp_clusters.star"
        
        if cluster_star.exists():
            try:
                import starfile
                star_data = starfile.read(cluster_star)
                if isinstance(star_data, dict):
                    for v in star_data.values():
                        if isinstance(v, pd.DataFrame):
                            df = v
                            break
                    else:
                        df = None
                else:
                    df = star_data
                
                if df is not None and 'active_zone' in df.columns:
                    aunp_az_numbers = sorted(df['active_zone'].unique().tolist())
                    # Remove -1 if present (means "not in any active zone")
                    aunp_az_numbers = [az for az in aunp_az_numbers if az != -1]
                    aunp_active_zone_indices = aunp_az_numbers
                    print(f"Auto-detected active zones from filtered AuNP file: {aunp_active_zone_indices}")
                else:
                    print("Warning: Could not extract active zones from filtered file, falling back to input files")
                    raise ValueError("No active_zone column in filtered file")
            except Exception as e:
                print(f"Error reading filtered file for auto-detection: {e}")
                # Fallback to original method
                aunps_dir = Path(tomo_path) / "best_alignment" / "aunps"
                import glob
                import re
                pattern = str(aunps_dir / "aunp_tm_BP_active_zone_*.star")
                aunp_az_numbers = []
                for file in glob.glob(pattern):
                    fname = Path(file).name
                    m = re.match(r"aunp_tm_BP_active_zone_(\d+)\.star", fname)
                    if m:
                        aunp_az_numbers.append(int(m.group(1)))
                aunp_az_numbers.sort()
                aunp_active_zone_indices = aunp_az_numbers
                print(f"Auto-detected active zones (fallback): {aunp_active_zone_indices}")
        else:
            print("Warning: Filtered AuNP file not found, falling back to input files")
            aunps_dir = Path(tomo_path) / "best_alignment" / "aunps"
            import glob
            import re
            pattern = str(aunps_dir / "aunp_tm_BP_active_zone_*.star")
            aunp_az_numbers = []
            for file in glob.glob(pattern):
                fname = Path(file).name
                m = re.match(r"aunp_tm_BP_active_zone_(\d+)\.star", fname)
                if m:
                    aunp_az_numbers.append(int(m.group(1)))
            aunp_az_numbers.sort()
            aunp_active_zone_indices = aunp_az_numbers
            print(f"Auto-detected active zones (fallback): {aunp_active_zone_indices}")
    
    # Load only the active zone membranes for CSV-specified or auto-detected active zones, matched by distance to AuNPs
    azs_pre, azs_post, azs_pre_inner, azs_post_inner = load_specific_active_zone_coords(
        tomo_path, aunp_active_zone_indices, aunps, alignment_dir=alignment_dir
    )
    
    if len(aunp_active_zone_indices) == 0:
        print("No active zones found, using middle of tomogram")
        # Fallback to middle of tomogram if no active zones found
        slice2d, z_center = load_tomogram_slice(tomo_path, None, alignment_dir=alignment_dir)
        if slice2d is None:
            print(f"Could not load tomogram slice for {tomo_path}")
            return
        _generate_visualizations_for_slice(tomo_path, output_dir, slice2d, z_center, vesicles, 
                                         pre_mem, post_mem, [], [], [], [], aunps, fusion_points, 
                                         aunp_active_zone_indices, rerun, "middle")
    else:
        # Generate visualizations for each active zone (CSV-specified or auto-detected)
        for az_id in aunp_active_zone_indices:
            # Processing active zone
            
            # Calculate z_center based on AuNPs within this specific active zone
            z_center = _calculate_active_zone_center_from_aunps(aunps, az_id)
            if z_center is None:
                print(f"Warning: No AuNPs found for active zone {az_id}, skipping visualization")
                continue
            
            # Generating visualizations for active zone
            
            slice2d, zc = load_tomogram_slice(tomo_path, z_center, alignment_dir=alignment_dir)
            if slice2d is None:
                print(f"Could not load tomogram slice for {tomo_path} at z={z_center}")
                continue
            
            _generate_visualizations_for_slice(tomo_path, output_dir, slice2d, z_center, vesicles, 
                                             pre_mem, post_mem, azs_pre, azs_post, azs_pre_inner, azs_post_inner,
                                             aunps, fusion_points, 
                                             aunp_active_zone_indices, rerun, f"az{az_id}")

def _calculate_active_zone_center_from_aunps(aunps, active_zone_id):
    """Calculate the z_center of an active zone based on the center of AuNPs within that active zone."""
    if aunps is None or aunps.empty:
        return None
    
    # Filter AuNPs for this specific active zone
    if 'active_zone' in aunps.columns:
        aunps_in_az = aunps[aunps['active_zone'] == active_zone_id]
    else:
        # If no active_zone column, we can't filter by active zone
        print(f"Warning: No 'active_zone' column in AuNP data, cannot calculate center for active zone {active_zone_id}")
        return None
    
    if aunps_in_az.empty:
        print(f"Warning: No AuNPs found in active zone {active_zone_id}")
        return None
    
    # Calculate the mean Z coordinate of AuNPs in this active zone
    z_center = int(np.mean(aunps_in_az['faCoordinateZ']))
    # Active zone AuNPs and z_center calculated
    
    return z_center

def _get_cluster_colors(n_clusters):
    """
    Get consistent cluster colors for both combined overlay and active zonogram visualizations.
    Returns a list of colors and a colormap.
    """
    import matplotlib.colors as mcolors
    import matplotlib.cm as cm
    
    # Use scientific publication colors as base, extended to 20 options
    # Note: Gray is excluded since noise points are always gray
    scientific_colors = [
        '#1f77b4',  # Blue
        '#ff7f0e',  # Orange
        '#2ca02c',  # Green
        '#d62728',  # Red
        '#9467bd',  # Purple
        '#8c564b',  # Brown
        '#e377c2',  # Pink
        '#bcbd22',  # Olive
        '#17becf',  # Cyan
        '#ff1493',  # Deep Pink
        '#aec7e8',  # Light Blue
        '#ffbb78',  # Light Orange
        '#98df8a',  # Light Green
        '#ff9896',  # Light Red
        '#c5b0d5',  # Light Purple
        '#c49c94',  # Light Brown
        '#f7b6d3',  # Light Pink
        '#dbdb8d',  # Light Olive
        '#9edae5',  # Light Cyan
        '#ffa500',  # Orange (alternative)
    ]
    
    if n_clusters <= 20:
        # Use the scientific colors directly
        cmap = mcolors.ListedColormap(scientific_colors[:n_clusters])
        colors = scientific_colors[:n_clusters]
    else:
        # For more than 20 clusters, extend the scientific colors
        base_colors = np.array([mcolors.to_rgba(color) for color in scientific_colors])
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
        colors = [mcolors.to_hex(color) for color in all_colors[:n_clusters]]
    
    return colors, cmap

def _generate_visualizations_for_slice(tomo_path, output_dir, slice2d, z_center, vesicles, 
                                     pre_mem, post_mem, azs_pre, azs_post, azs_pre_inner, azs_post_inner,
                                     aunps, fusion_points, 
                                     aunp_active_zone_indices, rerun, suffix):
    """Generate all visualization types for a specific slice."""
    tomo_name = Path(tomo_path).name
    
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
    azs_pre_inner_in_slice = filter_coords_in_slice(azs_pre_inner, z_center, z_thresh_az, None)
    azs_post_inner_in_slice = filter_coords_in_slice(azs_post_inner, z_center, z_thresh_az, None)
    aunps_near = filter_aunps_near_slice(aunps, z_center, z_thresh_aunps_fusion)
    
    # Inner AZ colors: lighter red / green than pure outer; drawn under outer scatters
    inner_pre_rgb = (1.0, 0.52, 0.52)
    inner_post_rgb = (0.52, 1.0, 0.52)
    inner_az_alpha = 0.06
    
    # Debug output (simplified)
    
    # Version 1: Vesicles and Active Zones
    output_file1 = output_dir / f"{tomo_name}_vesicles_active_zones_{suffix}.png"
    if output_file1.exists() and not rerun:
        print(f"Skipping {output_file1}, already exists.")
    else:
        fig1, ax1 = plt.subplots(figsize=(12, 12))
        ax1.imshow(slice2d, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
        
        # Overlay vesicles with transparency
        for v in vesicles_in_slice:
            c = np.array(v['center'])
            r = v['radius']
            circ = Circle((c[0], c[1]), r, color='pink', fill=False, lw=1.5, alpha=0.7, 
                         label='Vesicle' if 'Vesicle' not in [l.get_label() for l in ax1.get_legend_handles_labels()[0]] else '')
            ax1.add_patch(circ)
        
        # Highlight vesicles within 20 nm with transparency
        for v in vesicles_in_slice:
            if v.get('distance_to_az', 99) <= 20:
                c = np.array(v['center'])
                r = v['radius']
                circ = Circle((c[0], c[1]), r, color='aqua', fill=False, lw=2, alpha=0.8, 
                             label='<=20nm' if '<=20nm' not in [l.get_label() for l in ax1.get_legend_handles_labels()[0]] else '')
                ax1.add_patch(circ)
        
        # Inner active zones (faded; underneath outer)
        for coords in azs_pre_inner_in_slice:
            ax1.scatter(coords[:, 0], coords[:, 1], color=inner_pre_rgb, s=3, alpha=inner_az_alpha,
                        label='Presynaptic AZ (inner)' if 'Presynaptic AZ (inner)' not in [l.get_label() for l in ax1.get_legend_handles_labels()[0]] else '')
        for coords in azs_post_inner_in_slice:
            ax1.scatter(coords[:, 0], coords[:, 1], color=inner_post_rgb, s=3, alpha=inner_az_alpha,
                        label='Postsynaptic AZ (inner)' if 'Postsynaptic AZ (inner)' not in [l.get_label() for l in ax1.get_legend_handles_labels()[0]] else '')
        
        # Overlay presynaptic active zone (outer)
        for coords in azs_pre_in_slice:
            ax1.scatter(coords[:,0], coords[:,1], color='red', s=3, alpha=0.1, 
                    label='Presynaptic AZ (outer)' if 'Presynaptic AZ (outer)' not in [l.get_label() for l in ax1.get_legend_handles_labels()[0]] else '')
        
        # Overlay postsynaptic active zone (outer)
        for coords in azs_post_in_slice:
            ax1.scatter(coords[:,0], coords[:,1], color='green', s=3, alpha=0.1, 
                    label='Postsynaptic AZ (outer)' if 'Postsynaptic AZ (outer)' not in [l.get_label() for l in ax1.get_legend_handles_labels()[0]] else '')
        
        # Add note about distance filtering to legend
        legend_elements = [
            Line2D([0], [0], color='pink', lw=1.5, label='Vesicles (intersecting slice)'),
            Line2D([0], [0], color='aqua', lw=2, label='Vesicles <20 nm from AZ'),
            Line2D([0], [0], color=inner_pre_rgb, lw=1.5, label='Presynaptic AZ (inner)'),
            Line2D([0], [0], color=inner_post_rgb, lw=1.5, label='Postsynaptic AZ (inner)'),
            Line2D([0], [0], color='red', lw=1.5, label='Presynaptic AZ (outer)'),
            Line2D([0], [0], color='green', lw=1.5, label='Postsynaptic AZ (outer)'),
        ]
        ax1.legend(handles=legend_elements)
        ax1.set_title(f'Vesicles and Active Zones - {tomo_name}')
        ax1.set_xlabel('X (pixels)')
        ax1.set_ylabel('Y (pixels)')
        
        plt.savefig(output_file1, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved vesicles and active zones: {output_file1.name}")
    
    # Version 2: Vesicles and AuNPs
    output_file2 = output_dir / f"{tomo_name}_vesicles_aunps_{suffix}.png"
    if output_file2.exists() and not rerun:
        print(f"Skipping {output_file2}, already exists.")
    else:
        fig2, ax2 = plt.subplots(figsize=(12, 12))
        ax2.imshow(slice2d, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
        
        # Overlay vesicles with transparency
        for v in vesicles_in_slice:
            c = np.array(v['center'])
            r = v['radius']
            circ = Circle((c[0], c[1]), r, color='pink', fill=False, lw=1.5, alpha=0.7, 
                         label='Vesicle' if 'Vesicle' not in [l.get_label() for l in ax2.get_legend_handles_labels()[0]] else '')
            ax2.add_patch(circ)
        
        # Highlight vesicles within 20 nm with transparency
        for v in vesicles_in_slice:
            if v.get('distance_to_az', 99) <= 20:
                c = np.array(v['center'])
                r = v['radius']
                circ = Circle((c[0], c[1]), r, color='aqua', fill=False, lw=2, alpha=0.8, 
                             label='<=20nm' if '<=20nm' not in [l.get_label() for l in ax2.get_legend_handles_labels()[0]] else '')
                ax2.add_patch(circ)
        
        # Add AuNPs with transparency
        if aunps_near is not None:
            ax2.scatter(aunps_near['faCoordinateX'], aunps_near['faCoordinateY'], 
                      color='gold', s=30, alpha=0.8, label='AuNPs')
        
        # Show fusion points for membrane-adjacent vesicles that are being displayed
        if fusion_points is not None and len(fusion_points) > 0 and len(vesicles_in_slice) > 0:
            # Filter to only membrane-adjacent vesicles (≤20 nm from active zone) that are being displayed
            membrane_adjacent_vesicles_in_slice = [v for v in vesicles_in_slice if v.get('distance_to_az', 99) <= 20]
            
            if len(membrane_adjacent_vesicles_in_slice) > 0:
                from scipy.spatial.distance import cdist
                vesicle_centers = np.array([v['center'] for v in membrane_adjacent_vesicles_in_slice])
                
                # Find the closest fusion point to each membrane-adjacent vesicle being displayed
                distances = cdist(vesicle_centers, fusion_points)
                closest_fusion_indices = np.argmin(distances, axis=1)
                
                # Plot fusion points for membrane-adjacent vesicles being displayed
                plotted_fusion_points = set()
                fusion_points_plotted = 0
                
                for i, vesicle in enumerate(membrane_adjacent_vesicles_in_slice):
                    fusion_point = fusion_points[closest_fusion_indices[i]]
                    fusion_point_tuple = tuple(fusion_point)
                    
                    # Plot fusion point if it hasn't been plotted yet
                    if fusion_point_tuple not in plotted_fusion_points:
                        ax2.scatter(fusion_point[0], fusion_point[1], 
                                   color='orange', s=100, alpha=0.9, marker='*',
                                   label='Fusion Sites' if fusion_points_plotted == 0 else '')
                        plotted_fusion_points.add(fusion_point_tuple)
                        fusion_points_plotted += 1
            
            # Plotted fusion points for all membrane-adjacent vesicles being displayed and near the slice
        
        # Add note about distance filtering to legend
        legend_elements = [
            Line2D([0], [0], color='pink', lw=1.5, label='Vesicles (intersecting slice)'),
            Line2D([0], [0], color='aqua', lw=2, label='Vesicles <20 nm from AZ'),
            plt.scatter([], [], color='gold', s=30, label='AuNPs'),
            plt.scatter([], [], color='orange', s=100, marker='*', label='Fusion Sites')
        ]
        ax2.legend(handles=legend_elements)
        ax2.set_title(f'Vesicles and AuNPs - {tomo_name}')
        ax2.set_xlabel('X (pixels)')
        ax2.set_ylabel('Y (pixels)')
        
        plt.savefig(output_file2, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved vesicles and AuNPs: {output_file2.name}")
    
    # Version 3: Combined - Vesicles, Active Zones, AuNPs, and Fusion Sites
    output_file3 = output_dir / f"{tomo_name}_combined_{suffix}.png"
    if output_file3.exists() and not rerun:
        print(f"Skipping {output_file3}, already exists.")
    else:
        fig3, ax3 = plt.subplots(figsize=(12, 12))
        ax3.imshow(slice2d, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
        
        # Overlay vesicles with transparency
        for v in vesicles_in_slice:
            c = np.array(v['center'])
            r = v['radius']
            circ = Circle((c[0], c[1]), r, color='pink', fill=False, lw=1.5, alpha=0.7, 
                         label='Vesicle' if 'Vesicle' not in [l.get_label() for l in ax3.get_legend_handles_labels()[0]] else '')
            ax3.add_patch(circ)
        
        # Highlight vesicles within 20 nm with transparency
        for v in vesicles_in_slice:
            if v.get('distance_to_az', 99) <= 20:
                c = np.array(v['center'])
                r = v['radius']
                circ = Circle((c[0], c[1]), r, color='aqua', fill=False, lw=2, alpha=0.8, 
                             label='<=20nm' if '<=20nm' not in [l.get_label() for l in ax3.get_legend_handles_labels()[0]] else '')
                ax3.add_patch(circ)
        
        # Inner active zones (faded; underneath outer)
        for coords in azs_pre_inner_in_slice:
            ax3.scatter(coords[:, 0], coords[:, 1], color=inner_pre_rgb, s=3, alpha=inner_az_alpha,
                        label='Presynaptic AZ (inner)' if 'Presynaptic AZ (inner)' not in [l.get_label() for l in ax3.get_legend_handles_labels()[0]] else '')
        for coords in azs_post_inner_in_slice:
            ax3.scatter(coords[:, 0], coords[:, 1], color=inner_post_rgb, s=3, alpha=inner_az_alpha,
                        label='Postsynaptic AZ (inner)' if 'Postsynaptic AZ (inner)' not in [l.get_label() for l in ax3.get_legend_handles_labels()[0]] else '')
        
        # Overlay presynaptic active zone (outer)
        for coords in azs_pre_in_slice:
            ax3.scatter(coords[:,0], coords[:,1], color='red', s=3, alpha=0.1, 
                    label='Presynaptic AZ (outer)' if 'Presynaptic AZ (outer)' not in [l.get_label() for l in ax3.get_legend_handles_labels()[0]] else '')
        
        # Overlay postsynaptic active zone (outer)
        for coords in azs_post_in_slice:
            ax3.scatter(coords[:,0], coords[:,1], color='green', s=3, alpha=0.1, 
                    label='Postsynaptic AZ (outer)' if 'Postsynaptic AZ (outer)' not in [l.get_label() for l in ax3.get_legend_handles_labels()[0]] else '')
        
        # Add AuNPs with transparency
        if aunps_near is not None:
            ax3.scatter(aunps_near['faCoordinateX'], aunps_near['faCoordinateY'], 
                      color='gold', s=30, alpha=0.8, label='AuNPs')
        
        # Show fusion points for membrane-adjacent vesicles that are being displayed
        if fusion_points is not None and len(fusion_points) > 0 and len(vesicles_in_slice) > 0:
            # Filter to only membrane-adjacent vesicles (≤20 nm from active zone) that are being displayed
            membrane_adjacent_vesicles_in_slice = [v for v in vesicles_in_slice if v.get('distance_to_az', 99) <= 20]
            
            if len(membrane_adjacent_vesicles_in_slice) > 0:
                from scipy.spatial.distance import cdist
                vesicle_centers = np.array([v['center'] for v in membrane_adjacent_vesicles_in_slice])
                
                # Find the closest fusion point to each membrane-adjacent vesicle being displayed
                distances = cdist(vesicle_centers, fusion_points)
                closest_fusion_indices = np.argmin(distances, axis=1)
                
                # Plot fusion points for membrane-adjacent vesicles being displayed
                plotted_fusion_points = set()
                fusion_points_plotted = 0
                
                for i, vesicle in enumerate(membrane_adjacent_vesicles_in_slice):
                    fusion_point = fusion_points[closest_fusion_indices[i]]
                    fusion_point_tuple = tuple(fusion_point)
                    
                    # Plot fusion point if it hasn't been plotted yet
                    if fusion_point_tuple not in plotted_fusion_points:
                        ax3.scatter(fusion_point[0], fusion_point[1], 
                                   color='orange', s=100, alpha=0.9, marker='*',
                                   label='Fusion Sites' if fusion_points_plotted == 0 else '')
                        plotted_fusion_points.add(fusion_point_tuple)
                        fusion_points_plotted += 1
            
            # Plotted fusion points for all membrane-adjacent vesicles being displayed and near the slice
        
        # Add note about distance filtering to legend
        legend_elements = [
            Line2D([0], [0], color='pink', lw=1.5, label='Vesicles (intersecting slice)'),
            Line2D([0], [0], color='aqua', lw=2, label='Vesicles <20 nm from AZ'),
            Line2D([0], [0], color=inner_pre_rgb, lw=1.5, label='Presynaptic AZ (inner)'),
            Line2D([0], [0], color=inner_post_rgb, lw=1.5, label='Postsynaptic AZ (inner)'),
            Line2D([0], [0], color='red', lw=1.5, label='Presynaptic AZ (outer)'),
            Line2D([0], [0], color='green', lw=1.5, label='Postsynaptic AZ (outer)'),
            plt.scatter([], [], color='gold', s=30, label='AuNPs'),
            plt.scatter([], [], color='orange', s=100, marker='*', label='Fusion Sites')
        ]
        ax3.legend(handles=legend_elements)
        ax3.set_title(f'Combined - Vesicles, Active Zones, AuNPs, and Fusion Sites - {tomo_name}')
        ax3.set_xlabel('X (pixels)')
        ax3.set_ylabel('Y (pixels)')
        
        # Save the original combined image with legend
        plt.savefig(output_file3, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved combined visualization: {output_file3.name}")
        
        # Also save without suffix for PDF compatibility (only for the first active zone)
        if suffix == "az0" or suffix == "middle":
            output_file3_pdf = output_dir / f"{tomo_name}_combined.png"
            plt.savefig(output_file3_pdf, dpi=300, bbox_inches='tight')
            print(f"  ✓ Saved combined visualization for PDF: {output_file3_pdf.name}")
        
        # Save version without legend
        ax3.legend().set_visible(False)
        output_file3_no_legend = output_dir / f"{tomo_name}_combined_no_legend_{suffix}.png"
        plt.savefig(output_file3_no_legend, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved combined visualization (no legend): {output_file3_no_legend.name}")
        
        # Save legend only
        fig_legend, ax_legend = plt.subplots(figsize=(4, 6))
        ax_legend.legend(handles=legend_elements, loc='center')
        ax_legend.set_xlim(0, 1)
        ax_legend.set_ylim(0, 1)
        ax_legend.axis('off')
        ax_legend.set_title(f'Legend - {tomo_name}', pad=20)
        
        output_file3_legend = output_dir / f"{tomo_name}_combined_legend_{suffix}.png"
        plt.savefig(output_file3_legend, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved combined legend: {output_file3_legend.name}")

    # --- AuNP Cluster Visualization ---
    # Use the same filtered AuNPs that were used in the analysis
    # aunps already comes from aunp_clusters.star which includes cluster assignments
    aunp_clusters = aunps.copy() if aunps is not None else None
    
    if aunp_clusters is not None and not aunp_clusters.empty:
        # Check if cluster assignments are already in the dataframe (should be if loaded from aunp_clusters.star)
        if 'aunp_cluster' not in aunp_clusters.columns:
            # Fallback: Try to load cluster assignments separately (shouldn't be needed but kept for backward compatibility)
            aunps_results_dir = Path(tomo_path) / "best_alignment" / "STT_results" / "aunps"
            cluster_star = aunps_results_dir / "aunp_clusters.star"
            cluster_csv = aunps_results_dir / "aunp_nearest_neighbor_distances.csv"
            
            # Load cluster assignments
            cluster_assignments = None
            if cluster_star.exists():
                try:
                    import starfile
                    cluster_data = starfile.read(cluster_star)
                    if isinstance(cluster_data, dict):
                        for v in cluster_data.values():
                            if isinstance(v, pd.DataFrame):
                                cluster_assignments = v
                                break
                    elif isinstance(cluster_data, pd.DataFrame):
                        cluster_assignments = cluster_data
                except Exception:
                    cluster_assignments = None
            
            if cluster_assignments is None and cluster_csv.exists():
                try:
                    cluster_assignments = pd.read_csv(cluster_csv)
                except Exception:
                    cluster_assignments = None
            
            # Match cluster assignments to filtered AuNPs by coordinates
            if cluster_assignments is not None and not cluster_assignments.empty:
                if 'faCoordinateX' in aunp_clusters.columns and 'faCoordinateX' in cluster_assignments.columns:
                    # Create coordinate-based matching
                    from scipy.spatial.distance import cdist
                    coords_viz = aunp_clusters[['faCoordinateX', 'faCoordinateY', 'faCoordinateZ']].values
                    coords_cluster = cluster_assignments[['faCoordinateX', 'faCoordinateY', 'faCoordinateZ']].values
                    
                    # Find closest matches
                    distances = cdist(coords_viz, coords_cluster)
                    closest_indices = np.argmin(distances, axis=1)
                    
                    # Only keep matches that are very close (within 1 pixel)
                    close_matches = distances[np.arange(len(closest_indices)), closest_indices] < 1.0
                    
                    if np.any(close_matches) and 'aunp_cluster' in cluster_assignments.columns:
                        # Add cluster assignments to the filtered AuNPs
                        aunp_clusters['aunp_cluster'] = -1  # Default to noise
                        aunp_clusters.loc[close_matches, 'aunp_cluster'] = cluster_assignments.iloc[closest_indices[close_matches]]['aunp_cluster'].values
                        print(f"Matched {np.sum(close_matches)} AuNPs with cluster assignments")
                    else:
                        print("Warning: Could not match filtered AuNPs with cluster assignments")
                        aunp_clusters['aunp_cluster'] = -1
                else:
                    print("Warning: Coordinate columns not found for cluster matching")
                    aunp_clusters['aunp_cluster'] = -1
            else:
                print("Warning: No cluster assignments found")
                aunp_clusters['aunp_cluster'] = -1
        
        # Assign colors to clusters using shared color scheme
        clusters = aunp_clusters['aunp_cluster'].values
        unique_clusters = np.unique(clusters)
        n_clusters = len(unique_clusters[unique_clusters != -1])
        
        # Get consistent colors using shared function
        color_list, cmap = _get_cluster_colors(n_clusters)
        
        # Create mapping from cluster ID to color index
        valid_clusters = unique_clusters[unique_clusters != -1]
        cluster_color_map = {c: color_list[i] for i, c in enumerate(valid_clusters)}
        cluster_color_map[-1] = (0.5, 0.5, 0.5, 1.0)  # grey for noise
        colors = [cluster_color_map.get(c, (0.5, 0.5, 0.5, 1.0)) for c in clusters]
        # 1. Overlay all AuNPs on the combined visualization, colored by cluster
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(slice2d, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
        # Plot all AuNPs
        ax.scatter(aunp_clusters['faCoordinateX'], aunp_clusters['faCoordinateY'],
                   c=colors, s=30, alpha=0.8, label='AuNPs (clustered)')
        ax.set_title(f"{tomo_name} - Combined Overlay with AuNP Clusters")
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        # Legend for clusters
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=cluster_color_map[c], edgecolor='k',
                                 label=f'Cluster {c}' if c != -1 else 'Non-clustered')
                          for c in unique_clusters]
        ax.legend(handles=legend_elements, loc='best')
        # Use organized structure: results/visualizations/{tomo_name}/aunps_and_vesicles/
        output_dir_viz = Path('results') / 'visualizations' / tomo_name / 'aunps_and_vesicles'
        output_dir_viz.mkdir(parents=True, exist_ok=True)
        out_combined = output_dir_viz / f"{tomo_name}_combined_aunpclusters_{suffix}.png"
        if out_combined.exists() and not rerun:
            print(f"Skipping {out_combined}, already exists.")
        else:
            plt.savefig(out_combined, dpi=300, bbox_inches='tight')
            print(f"  ✓ Saved combined AuNP cluster overlay: {out_combined.name}")
            
            # Also save without suffix for PDF compatibility (only for the first active zone)
            if suffix == "az0" or suffix == "middle":
                out_combined_pdf = output_dir_viz / f"{tomo_name}_combined_aunpclusters.png"
                plt.savefig(out_combined_pdf, dpi=300, bbox_inches='tight')
                print(f"  ✓ Saved combined AuNP cluster overlay for PDF: {out_combined_pdf.name}")
            
            plt.close()
        # Save also to the tomogram's own visualization directory
        tomo_viz_dir = Path(tomo_path) / "best_alignment" / "STT_results" / "visualizations"
        tomo_viz_dir.mkdir(parents=True, exist_ok=True)
        out_combined_tomo = tomo_viz_dir / f"{tomo_name}_combined_aunpclusters_{suffix}.png"
        # Save the same figures to the tomogram's visualization directory
        plt.figure(figsize=(12, 12))
        plt.imshow(slice2d, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
        plt.scatter(aunp_clusters['faCoordinateX'], aunp_clusters['faCoordinateY'],
                    c=colors, s=30, alpha=0.8, label='AuNPs (clustered)')
        plt.title(f"{tomo_name} - Combined Overlay with AuNP Clusters")
        plt.xlabel('X (pixels)')
        plt.ylabel('Y (pixels)')
        plt.legend(handles=legend_elements, loc='best')
        plt.savefig(out_combined_tomo, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Also saved cluster visualizations to {tomo_viz_dir}")
    # --- End AuNP Cluster Visualization ---


def run_zonogram_analysis_for_all_tomograms(tomo_paths, output_dir, csv_path=None, root_dir=None, rerun=False,
                                            sphere_size=None, sphere_color=None, aunp_distance_min=None, aunp_distance_max=None,
                                            aunp_distance_cutoff_direction=None, aunp_distance_cutoff_value=None):
    """Run active zonogram analysis for all tomograms and generate PDF summaries."""
    try:
        # Import the combined zonogram analysis function
        from .activezone import (
            define_active_zone, define_active_zonogram, extract_active_zonogram,
            import_membrane_segmentations_from_glb, find_active_zones_from_glb
        )
        
        # Individual files are saved to organized structure: results/visualizations/{tomogram_name}/active_zonograms/
        
        print(f"Running active zonogram analysis for {len(tomo_paths)} tomograms...")
        
        # Process each tomogram with progress tracking
        successful_count = 0
        failed_count = 0
        
        for i, tomo_info in enumerate(tomo_paths, 1):
            # Handle both old format (just path) and new format (path, set_name, active_zones)
            if isinstance(tomo_info, tuple) and len(tomo_info) == 3:
                if len(tomo_info) >= 4:
                    tomo_path, set_name, aunp_active_zones, _alignment_dir = tomo_info
                else:
                    tomo_path, set_name, aunp_active_zones = tomo_info
            else:
                tomo_path = tomo_info
                set_name = None
                aunp_active_zones = None
            
            tomogram_name = Path(tomo_path).name
            print(f"[{i}/{len(tomo_paths)}] Processing {tomogram_name}...", end=" ", flush=True)
            
            try:
                # Run the combined active zonogram analysis for this tomogram
                result = run_combined_zonogram_analysis_single_tomogram(tomo_path, None, aunp_active_zones, rerun,
                                                                         sphere_size=sphere_size, sphere_color=sphere_color,
                                                                         aunp_distance_min=aunp_distance_min, aunp_distance_max=aunp_distance_max,
                                                                         aunp_distance_cutoff_direction=aunp_distance_cutoff_direction,
                                                                         aunp_distance_cutoff_value=aunp_distance_cutoff_value)
                
                if result.get('success', False):
                    print("✅")
                    successful_count += 1
                else:
                    print("❌")
                    failed_count += 1
                    print(f"    Error: {result.get('reason', 'Unknown error')}")
                    
            except Exception as e:
                print("❌")
                failed_count += 1
                print(f"    Error: {e}")
                # Only show full traceback for unexpected errors
                if "ValueError" not in str(type(e).__name__):
                    import traceback
                    traceback.print_exc()
                continue
        
        print(f"\nActive zonogram analysis complete: {successful_count} successful, {failed_count} failed")
        
        # Generate default visualization PDF summary
        print("\nGenerating PDF summary...")
        # Extract root directory from tomogram paths
        root_dir = None
        if tomo_paths:
            # Get the root directory from the first tomogram path
            first_tomo_path = Path(tomo_paths[0][0])
            # Go up to find the root (assuming structure: root/set/TOP_TOMOS/tomogram)
            if first_tomo_path.parent.name == "TOP_TOMOS":
                root_dir = str(first_tomo_path.parent.parent.parent)
        generate_default_visualization_pdf_summary(tomo_paths, csv_path, root_dir)
        
        # Generate zonogram PDF summaries (all zonograms and mini zonograms)
        print("\nGenerating zonogram PDF summaries...")
        # Extract data directory from tomogram paths
        data_dir = None
        if tomo_paths:
            # Get the data directory from the first tomogram path
            first_tomo_path = Path(tomo_paths[0][0])
            # Go up to find the data directory (assuming structure: data_dir/set/TOP_TOMOS/tomogram)
            if first_tomo_path.parent.name == "TOP_TOMOS":
                data_dir = str(first_tomo_path.parent.parent.parent)
        generate_zonogram_pdf_summaries(None, tomo_paths, data_dir)
        
        print(f"\nActive zonogram analysis complete! Results saved to organized structure: results/visualizations/{{tomogram_name}}/active_zonograms/")
        
    except ImportError as e:
        print(f"Warning: Could not import active zonogram analysis modules: {e}")
        print("Skipping active zonogram analysis.")
    except Exception as e:
        print(f"Error in active zonogram analysis: {e}")

def render_active_zonograms_findingampa_style(active_zone_data):
    """
    Render active zonogram using the exact same approach as findingampa.
    Based on findingampa/src/findingampa/utils/analysis.py:render_active_zonograms()
    """
    import torch
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    
    res_ddw = active_zone_data[2]
    width = (res_ddw.shape[2] + res_ddw.shape[0]) / 50
    height = (res_ddw.shape[1] + res_ddw.shape[0]) / 50
    fig = plt.figure(figsize=(width, height))
    gs = gridspec.GridSpec(2, 2, width_ratios=[res_ddw.shape[2], res_ddw.shape[0]], height_ratios=[res_ddw.shape[1], res_ddw.shape[0]])

    axxy = plt.subplot(gs[0, 0])
    axyz = plt.subplot(gs[0, 1], sharey=axxy)
    axxz = plt.subplot(gs[1, 0], sharex=axxy)
    
    axxy.imshow(torch.min(res_ddw, axis=0).values, cmap='gray', interpolation='mitchell', vmax=-0., vmin=-20*res_ddw.std(), origin='lower')

    # Draw arrows for coordinate system in xy plane
    axxy.quiver(0, 0, 0, 50, color='g', angles='xy', scale_units='xy', units="xy", width=1, scale=1, clip_on=False)
    axxy.quiver(0, 0, 50, 0, color='r', angles='xy', scale_units='xy', units="xy", width=1, scale=1, clip_on=False)

    # Add scale bar text for xy plane
    axxy.text(25, 5, '50 nm', color='white', fontsize=8, ha='center', va='bottom',
              bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
    
    axyz.imshow(torch.min(res_ddw, axis=2).values.T, cmap='gray', interpolation='mitchell', vmax=-0., vmin=-20*res_ddw.std(), origin='lower')

    axyz.quiver(0, 0, 0, 50, color='g', angles='xy', scale_units='xy', units="xy", width=1, scale=1, clip_on=False)
    axyz.quiver(0, 0, 50, 0, color='b', angles='xy', scale_units='xy', units="xy", width=1, scale=1, clip_on=False)

    # Add scale bar text for yz plane
    axyz.text(25, 5, '50 nm', color='white', fontsize=8, ha='center', va='bottom',
              bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
    
    axxz.imshow(torch.min(res_ddw, axis=1).values, cmap='gray', interpolation='mitchell', vmax=-0., vmin=-20*res_ddw.std(), origin='lower')

    axxz.quiver(0, 0, 0, 50, color='b', angles='xy', scale_units='xy', units="xy", width=1, scale=1, clip_on=False)
    axxz.quiver(0, 0, 50, 0, color='r', angles='xy', scale_units='xy', units="xy", width=1, scale=1, clip_on=False)

    # Add scale bar text for xz plane
    axxz.text(25, 5, '50 nm', color='white', fontsize=8, ha='center', va='bottom',
              bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
    # Hide axes 
    axxy.axis('off')
    axxz.axis('off')
    axyz.axis('off')
    plt.tight_layout()
    return fig

def render_mini_zonogram_xy_only(active_zone_data, include_legend_space=False, extra_width_multiplier=1.5):
    """
    Render mini zonogram showing only the xy slice (top-left view from regular zonograms).
    If include_legend_space is True, creates a wider figure to accommodate legend on the right.
    """
    import torch
    import matplotlib.pyplot as plt
    
    res_ddw = active_zone_data[2]
    # Use square aspect ratio for mini zonogram
    fig_size = max(res_ddw.shape[1], res_ddw.shape[2]) / 50
    
    if include_legend_space:
        # Make figure wider to accommodate legend on the right
        fig = plt.figure(figsize=(fig_size * extra_width_multiplier, fig_size))
    else:
        fig = plt.figure(figsize=(fig_size, fig_size))
    
    axxy = plt.subplot(111)
    
    axxy.imshow(torch.min(res_ddw, axis=0).values, cmap='gray', interpolation='mitchell', vmax=-0., vmin=-20*res_ddw.std(), origin='lower')

    # Draw arrows for coordinate system in xy plane
    axxy.quiver(0, 0, 0, 50, color='g', angles='xy', scale_units='xy', units="xy", width=1, scale=1, clip_on=False)
    axxy.quiver(0, 0, 50, 0, color='r', angles='xy', scale_units='xy', units="xy", width=1, scale=1, clip_on=False)

    # Add scale bar text for xy plane
    axxy.text(25, 5, '50 nm', color='white', fontsize=8, ha='center', va='bottom',
              bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
    
    # Hide axes 
    axxy.axis('off')
    plt.tight_layout()
    return fig, axxy

def select_aunps_findingampa_style(active_zone_data, aunp_data, tomogram_path, active_zone_id=0, original_zone_data=None):
    """
    Select AuNPs for visualization using findingampa-style approach.
    Only includes AuNPs that belong to the specified active zone.
    """
    from pathlib import Path
    import numpy as np
    import pandas as pd
    
    # Load AuNP data from filtered output file
    aunp_file = Path(tomogram_path) / "best_alignment" / "STT_results" / "aunps" / "aunp_clusters.star"

    if not aunp_file.exists():
        return []
    
    try:
        import starfile
        star_data = starfile.read(aunp_file)
        # Handle both dict and DataFrame formats
        if isinstance(star_data, dict):
            for v in star_data.values():
                if isinstance(v, pd.DataFrame):
                    aunp_df = v
                    break
            else:
                return []
        elif isinstance(star_data, pd.DataFrame):
            aunp_df = star_data
        else:
            return []
        
        # Filter AuNPs by active zone
        if 'active_zone' not in aunp_df.columns:
            return []
        aunp_df = aunp_df[aunp_df['active_zone'] == active_zone_id]
        if aunp_df.empty:
            return []
        aunp_positions = aunp_df[['faCoordinateX', 'faCoordinateY', 'faCoordinateZ']].values

    except Exception as e:
        print(f"Error loading AuNPs in select_aunps_findingampa_style: {e}")
        return []
    
    # Use proper transformation if original zone data is available
    if original_zone_data is not None:
        # Transform AuNP positions to zonogram coordinate system (same as original run_zonogram.py)
        center = original_zone_data['center']
        coordinate_system = original_zone_data['transformation_matrix'][:3, :3]
        
        selected_aunp_pos_transformed = (aunp_positions - center) @ coordinate_system.T
        selected_aunp_pos_transformed += np.floor(np.array(active_zone_data[2].shape)[[2,1,0]]/2)
        
        # Filter points within the volume
        valid_mask = np.all(selected_aunp_pos_transformed > 0, axis=1) & np.all(selected_aunp_pos_transformed < np.array(active_zone_data[2].shape)[[2,1,0]], axis=1)
        selected_aunp_positions = selected_aunp_pos_transformed[valid_mask]
        
        return selected_aunp_positions
    else:
        # Fallback: return empty array if no transformation data available
        return []


def select_aunps_with_distances_findingampa_style(active_zone_data, aunp_data, tomogram_path, active_zone_id=0, original_zone_data=None):
    """
    Select AuNPs for visualization with their distances to postsynaptic membrane.
    Only includes AuNPs that belong to the specified active zone.
    Returns a dict with 'positions' and 'distances' arrays.
    """
    from pathlib import Path
    import numpy as np
    import pandas as pd
    
    # Load AuNP data from filtered output file
    aunp_file = Path(tomogram_path) / "best_alignment" / "STT_results" / "aunps" / "aunp_clusters.star"

    if not aunp_file.exists():
        return {'positions': np.array([]), 'distances': np.array([])}
    
    try:
        import starfile
        star_data = starfile.read(aunp_file)
        # Handle both dict and DataFrame formats
        if isinstance(star_data, dict):
            for v in star_data.values():
                if isinstance(v, pd.DataFrame):
                    aunp_df = v
                    break
            else:
                return {'positions': np.array([]), 'distances': np.array([])}
        elif isinstance(star_data, pd.DataFrame):
            aunp_df = star_data
        else:
            return {'positions': np.array([]), 'distances': np.array([])}
        
        # Filter AuNPs by active zone
        if 'active_zone' not in aunp_df.columns:
            return {'positions': np.array([]), 'distances': np.array([])}
        aunp_df = aunp_df[aunp_df['active_zone'] == active_zone_id]
        if aunp_df.empty:
            return {'positions': np.array([]), 'distances': np.array([])}
        
        # Get positions and distances
        aunp_positions = aunp_df[['faCoordinateX', 'faCoordinateY', 'faCoordinateZ']].values
        
        # Get distance to postsynaptic membrane (try both column name variations)
        if 'distance_to_postsynaptic_nm' in aunp_df.columns:
            post_distances = aunp_df['distance_to_postsynaptic_nm'].values
        elif 'distance_to_postsynaptic' in aunp_df.columns:
            post_distances = aunp_df['distance_to_postsynaptic'].values
        else:
            print(f"Warning: distance_to_postsynaptic column not found in aunp_clusters.star")
            return {'positions': np.array([]), 'distances': np.array([])}

    except Exception as e:
        print(f"Error loading AuNPs with distances in select_aunps_with_distances_findingampa_style: {e}")
        return {'positions': np.array([]), 'distances': np.array([])}
    
    # Use proper transformation if original zone data is available
    if original_zone_data is not None:
        # Transform AuNP positions to zonogram coordinate system (same as original run_zonogram.py)
        center = original_zone_data['center']
        coordinate_system = original_zone_data['transformation_matrix'][:3, :3]
        
        selected_aunp_pos_transformed = (aunp_positions - center) @ coordinate_system.T
        selected_aunp_pos_transformed += np.floor(np.array(active_zone_data[2].shape)[[2,1,0]]/2)
        
        # Filter points within the volume
        valid_mask = np.all(selected_aunp_pos_transformed > 0, axis=1) & np.all(selected_aunp_pos_transformed < np.array(active_zone_data[2].shape)[[2,1,0]], axis=1)
        selected_aunp_positions = selected_aunp_pos_transformed[valid_mask]
        selected_distances = post_distances[valid_mask]
        
        return {'positions': selected_aunp_positions, 'distances': selected_distances}
    else:
        # Fallback: return empty arrays if no transformation data available
        return {'positions': np.array([]), 'distances': np.array([])}


def select_aunps_by_cluster_findingampa_style(active_zone_data, cluster_data, tomogram_path, active_zone_id=0, original_zone_data=None):
    """
    Select AuNPs by cluster for visualization using findingampa-style approach.
    Only includes AuNPs that belong to the specified active zone.
    """
    from pathlib import Path
    import numpy as np
    import pandas as pd
    
    # Load cluster data from filtered output file
    cluster_file = Path(tomogram_path) / "best_alignment" / "STT_results" / "aunps" / "aunp_clusters.star"

    if not cluster_file.exists():
        return [], []
    
    try:
        import starfile
        star_data = starfile.read(cluster_file)
        # Handle both dict and DataFrame formats
        if isinstance(star_data, dict):
            for v in star_data.values():
                if isinstance(v, pd.DataFrame):
                    cluster_df = v
                    break
            else:
                return [], []
        elif isinstance(star_data, pd.DataFrame):
            cluster_df = star_data
        else:
            return [], []
        
        # Filter AuNPs by active zone
        if 'active_zone' not in cluster_df.columns:
            return [], []
        cluster_df = cluster_df[cluster_df['active_zone'] == active_zone_id]
        if cluster_df.empty:
            return [], []
        aunp_positions = cluster_df[['faCoordinateX', 'faCoordinateY', 'faCoordinateZ']].values
        # Get cluster assignments (default to -1 if column missing)
        if 'aunp_cluster' in cluster_df.columns:
            cluster_assignments = cluster_df['aunp_cluster'].values
        else:
            cluster_assignments = np.full(len(cluster_df), -1)
    except Exception as e:
        print(f"Error loading cluster data in select_aunps_by_cluster_findingampa_style: {e}")
        return [], []
    
    # Use proper transformation if original zone data is available
    if original_zone_data is not None:
        # Transform AuNP positions to zonogram coordinate system (same as original run_zonogram.py)
        center = original_zone_data['center']
        coordinate_system = original_zone_data['transformation_matrix'][:3, :3]
        
        selected_aunp_pos_transformed = (aunp_positions - center) @ coordinate_system.T
        selected_aunp_pos_transformed += np.floor(np.array(active_zone_data[2].shape)[[2,1,0]]/2)
        
        # Filter points within the volume
        valid_mask = np.all(selected_aunp_pos_transformed > 0, axis=1) & np.all(selected_aunp_pos_transformed < np.array(active_zone_data[2].shape)[[2,1,0]], axis=1)
        selected_aunp_positions = selected_aunp_pos_transformed[valid_mask]
        selected_cluster_assignments = cluster_assignments[valid_mask]
        
        return selected_aunp_positions, selected_cluster_assignments
    else:
        # Fallback: return empty arrays if no transformation data available
        return [], []


def run_combined_zonogram_analysis_single_tomogram(tomo_path, output_dir, aunp_active_zones=None, rerun=False,
                                                    sphere_size=None, sphere_color=None, aunp_distance_min=None, aunp_distance_max=None,
                                                    aunp_distance_cutoff_direction=None, aunp_distance_cutoff_value=None):
    """Run combined active zonogram analysis for a single tomogram - EXACT SAME CODE as original script."""
    from .activezone import (
        define_active_zone, define_active_zonogram, extract_active_zonogram,
        import_membrane_segmentations_from_glb, find_active_zones_from_glb
    )
    from scipy.spatial import KDTree
    import numpy as np
    import mrcfile
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    import torch
    import pandas as pd
    from scipy.spatial import cKDTree
    from scipy.spatial.distance import pdist, squareform
    from pathlib import Path
    
    tomogram_path = str(tomo_path)
    tomogram_name = Path(tomo_path).name
    
    try:
        # Step 1: Load membrane data and active zones (shared between both analyses)
        membrane_data = import_membrane_segmentations_from_glb(tomogram_path)
        
        # Find active zones
        active_zones_data = find_active_zones_from_glb(membrane_data, distance_range=(10.0, 40.0))
        
        if not active_zones_data['active_zones']:
            print("No active zones found. Skipping active zonogram analysis.")
            return {"success": False, "reason": "No active zones found"}
        
        # Load AuNP data for smart active zone matching (always try to load if file exists)
        aunp_data = None
        try:
            # Load AuNP data to match active zones
            aunp_star_path = Path(tomogram_path) / "best_alignment" / "STT_results" / "aunps" / "aunp_clusters.star"
            if aunp_star_path.exists():
                import starfile
                star_data = starfile.read(aunp_star_path)
                # Handle both dict and DataFrame formats
                if isinstance(star_data, dict):
                    for v in star_data.values():
                        if isinstance(v, pd.DataFrame):
                            aunp_data = v
                            break
                    else:
                        aunp_data = None
                elif isinstance(star_data, pd.DataFrame):
                    aunp_data = star_data
                else:
                    aunp_data = None
                if aunp_data is not None:
                    # Loaded AuNP data for smart active zone matching
                    pass
                else:
                    print(f"Warning: Could not extract DataFrame from {aunp_star_path}, smart matching will not be available")
                    aunp_data = None
            else:
                print(f"Warning: AuNP data not found at {aunp_star_path}, smart matching will not be available")
                aunp_data = None
        except Exception as e:
            print(f"Warning: Error loading AuNP data: {e}, smart matching will not be available")
            aunp_data = None
        
        # Filter active zones based on CSV specification using smart matching
        if aunp_active_zones is not None and aunp_active_zones != []:
            # Handle both list and string inputs
            if isinstance(aunp_active_zones, list):
                # Already parsed list from CLI
                selected_az_indices = aunp_active_zones
                # CSV specified active zones
            else:
                # Parse active zone indices from CSV string (handle floats like "2.0")
                az_str = str(aunp_active_zones) if aunp_active_zones is not None else ""
                if az_str.strip() != "" and az_str.lower() != "nan":
                    try:
                        selected_az_indices = []
                        for x in az_str.split(","):
                            x = x.strip()
                            if x.isdigit():
                                selected_az_indices.append(int(x))
                            elif x.replace(".", "").isdigit():  # Handle floats like "2.0"
                                selected_az_indices.append(int(float(x)))
                        # CSV specified active zones
                    except Exception as e:
                        print(f"Warning: Error parsing active zone indices from CSV '{aunp_active_zones}': {e}")
                        print("Proceeding with all active zones")
                        selected_az_indices = None
                else:
                    print("No active zones specified in CSV, proceeding with all active zones")
                    selected_az_indices = None
        else:
            print("No active zone filtering specified, proceeding with all active zones")
            selected_az_indices = None
        
        # If no specific active zones were specified, get all available active zone numbers from filtered AuNP file
        if selected_az_indices is None:
            aunps_results_dir = Path(tomogram_path) / "best_alignment" / "STT_results" / "aunps"
            cluster_star = aunps_results_dir / "aunp_clusters.star"
            
            if cluster_star.exists():
                try:
                    import starfile
                    star_data = starfile.read(cluster_star)
                    if isinstance(star_data, dict):
                        for v in star_data.values():
                            if isinstance(v, pd.DataFrame):
                                df = v
                                break
                        else:
                            df = None
                    else:
                        df = star_data
                    
                    if df is not None and 'active_zone' in df.columns:
                        aunp_az_numbers = sorted(df['active_zone'].unique().tolist())
                        # Remove -1 if present (means "not in any active zone")
                        aunp_az_numbers = [az for az in aunp_az_numbers if az != -1]
                        selected_az_indices = aunp_az_numbers
                        print(f"Using all available active zones from filtered AuNP file: {selected_az_indices}")
                    else:
                        raise ValueError("No active_zone column in filtered file")
                except Exception as e:
                    print(f"Error reading filtered file for active zone detection: {e}, falling back to input files")
                    # Fallback to original method
                    aunps_dir = Path(tomogram_path) / "best_alignment" / "aunps"
                    import glob
                    import re
                    pattern = str(aunps_dir / "aunp_tm_BP_active_zone_*.star")
                    aunp_az_numbers = []
                    for file in glob.glob(pattern):
                        fname = Path(file).name
                        m = re.match(r"aunp_tm_BP_active_zone_(\d+)\.star", fname)
                        if m:
                            aunp_az_numbers.append(int(m.group(1)))
                    aunp_az_numbers.sort()
                    selected_az_indices = aunp_az_numbers
                    print(f"Using all available active zones from input files (fallback): {selected_az_indices}")
            else:
                print("Warning: Filtered AuNP file not found, falling back to input files")
                aunps_dir = Path(tomogram_path) / "best_alignment" / "aunps"
                import glob
                import re
                pattern = str(aunps_dir / "aunp_tm_BP_active_zone_*.star")
                aunp_az_numbers = []
                for file in glob.glob(pattern):
                    fname = Path(file).name
                    m = re.match(r"aunp_tm_BP_active_zone_(\d+)\.star", fname)
                    if m:
                        aunp_az_numbers.append(int(m.group(1)))
                aunp_az_numbers.sort()
                selected_az_indices = aunp_az_numbers
                print(f"Using all available active zones from input files (fallback): {selected_az_indices}")
        
        # Use saved mapping from activezone.py (created by define_active_zone)
        from .activezone import load_active_zone_mapping
        
        # Load saved mapping
        az_mapping = load_active_zone_mapping(tomogram_path)
        
        if not az_mapping:
            # No mapping found - use all active zones as fallback but print error
            print(f"No saved active zone mapping found for {tomogram_name}. Active zone analysis must be run first with smart matching to create the mapping.")
            print(f"FALLBACK: Using all {len(active_zones_data['active_zones'])} active zones found from GLB (no filtering applied).")
            # Use all zones, no filtering
            filtered_active_zones = active_zones_data['active_zones']
            # Create a dummy mapping for filename generation (use zone names as-is)
            az_mapping = {}
            for idx, zone_name in enumerate(active_zones_data['active_zones'].keys()):
                az_mapping[idx] = zone_name
        else:
            # Convert string keys to int (JSON stores dict keys as strings)
            az_mapping = {int(k): v for k, v in az_mapping.items()}
            
            # Filter to only include zones in the mapping
            filtered_active_zones = {}
            for az_index in selected_az_indices:
                if az_index in az_mapping:
                    zone_name = az_mapping[az_index]
                    if zone_name in active_zones_data['active_zones']:
                        filtered_active_zones[zone_name] = active_zones_data['active_zones'][zone_name]
                    else:
                        raise ValueError(f"Zone {zone_name} from saved mapping not found in active zones data. This indicates a mismatch between the mapping and current active zones.")
                else:
                    raise ValueError(f"Active zone index {az_index} not found in saved mapping. This indicates the active zone analysis was run with different indices.")
            
            print(f"Using saved active zone mapping for {len(filtered_active_zones)} zones")
        
        # Store the mapping for later use in filename generation
        active_zones_data['az_mapping'] = az_mapping
        active_zones_data['active_zones'] = filtered_active_zones
        
        # Step 2: Regular Active Zonogram Analysis
        
        # Define active zonograms
        zonogram_results = define_active_zonogram(active_zones_data)
        
        if zonogram_results['status'] == 'completed':
            # Defined active zonograms
            
            # Extract and save zonograms
            extracted_results = extract_active_zonogram(zonogram_results, active_zones_data, tomogram_path)
            
            if extracted_results and isinstance(extracted_results, dict) and 'rendered_zonograms' in extracted_results and extracted_results.get('rendered_zonograms'):
                # Create output directories
                # 1. In results/visualizations/{tomogram_name}/active_zonograms/full/ directory (new organized structure)
                results_active_zonograms_dir_full = Path("results") / "visualizations" / tomogram_name / "active_zonograms" / "full"
                results_active_zonograms_dir_full.mkdir(parents=True, exist_ok=True)
                
                # 2. In tomogram's STT_results/visualizations/active_zonograms directory
                tomogram_active_zonograms_dir = Path(tomogram_path) / "best_alignment" / "STT_results" / "visualizations" / "active_zonograms"
                tomogram_active_zonograms_dir.mkdir(parents=True, exist_ok=True)
                
                files_created = []
                
                # Create filename suffix using the az_mapping (define once for all zones)
                if 'az_mapping' in active_zones_data:
                    # Use the first active zone index from the mapping as the default suffix
                    first_az_index = list(active_zones_data['az_mapping'].keys())[0] if active_zones_data['az_mapping'] else 0
                    default_suffix = f"_az{first_az_index}"
                else:
                    # Fallback if no az_mapping available
                    default_suffix = "_az0"
                
                for zone_name, zone_data in extracted_results['rendered_zonograms'].items():
                    # Get the original zonogram data with transformation matrix and extent
                    original_zone_data = zonogram_results['zonogram_data'][zone_name]
                    
                    # Create zonogram data in findingampa format
                    zonogram_findingampa = (np.eye(3), np.zeros(3), torch.tensor(zone_data['transformed_tomogram']), ())
                    
                    # Create filename suffix using the az_mapping for this specific zone
                    if 'az_mapping' in active_zones_data:
                        # Find which active zone index maps to this zone_name
                        az_index = None
                        for idx, mapped_zone in active_zones_data['az_mapping'].items():
                            if mapped_zone == zone_name:
                                az_index = idx
                                break
                        
                        if az_index is not None:
                            suffix = f"_az{az_index}"
                        else:
                            # Fallback if zone_name not found in mapping
                            suffix = default_suffix
                    else:
                        # Fallback if no az_mapping available
                        suffix = default_suffix
                    
                    # Save MRC file to tomogram directory only
                    mrc_filename = f"{tomogram_name}_active_zonogram_{zone_name}{suffix}.mrc"
                    mrcfile.write(tomogram_active_zonograms_dir / mrc_filename, zone_data['transformed_tomogram'], overwrite=True)
                    print(f"    ✓ Saved MRC: {mrc_filename}")
                    
                    # Save NPY file to tomogram directory only
                    npy_filename = f"{tomogram_name}_active_zonogram_{zone_name}{suffix}.npy"
                    npy_data = {
                        "cs": np.eye(3),
                        "center": np.zeros(3),
                        "objects": ()
                    }
                    np.save(tomogram_active_zonograms_dir / npy_filename, npy_data, allow_pickle=True)
                    print(f"    ✓ Saved MRC: {mrc_filename}")
                    
                    # Generate main PNG and save to organized structure and tomogram directory
                    png_filename = f"{tomogram_name}_active_zonogram_{zone_name}{suffix}.png"
                    png_path_results_organized = results_active_zonograms_dir_full / png_filename
                    png_path_tomogram = tomogram_active_zonograms_dir / png_filename
                    
                    if png_path_results_organized.exists() and png_path_tomogram.exists() and not rerun:
                        print(f"    Skipping {png_filename}, already exists.")
                        files_created.append(png_filename)
                    else:
                        fig = render_active_zonograms_findingampa_style(zonogram_findingampa)
                        fig.savefig(png_path_results_organized)
                        fig.savefig(png_path_tomogram)
                        plt.close(fig)
                        print(f"    ✓ Saved PNG: {png_filename}")
                        files_created.append(png_filename)
                    
                    # Extract active zone ID from the az_mapping
                    active_zone_id = None
                    if 'az_mapping' in active_zones_data:
                        # Find which active zone index maps to this zone_name
                        for idx, mapped_zone in active_zones_data['az_mapping'].items():
                            if mapped_zone == zone_name:
                                active_zone_id = idx
                                break
                    
                    # Fallback if no mapping found
                    if active_zone_id is None:
                        if 'pre1_post1' in zone_name:
                            active_zone_id = 0
                        elif 'pre2_post1' in zone_name:
                            active_zone_id = 1
                        else:
                            active_zone_id = 0  # Default fallback
                    
                    # Generate AuNP visualization
                    selected_aunps = select_aunps_findingampa_style(zonogram_findingampa, None, tomogram_path, active_zone_id, original_zone_data)
                    if len(selected_aunps) > 0:
                        aunp_filename = f"{tomogram_name}_active_zonogram_{zone_name}_selected_aunps{suffix}.png"
                        aunp_path_results_organized = results_active_zonograms_dir_full / aunp_filename
                        aunp_path_tomogram = tomogram_active_zonograms_dir / aunp_filename
                        
                        if aunp_path_results_organized.exists() and aunp_path_tomogram.exists() and not rerun:
                            print(f"    Skipping {aunp_filename}, already exists.")
                            files_created.append(aunp_filename)
                        else:
                            fig = render_active_zonograms_findingampa_style(zonogram_findingampa)
                            (axxy, axxz, axyz) = fig.get_axes()
                            
                            # Use custom sphere size and color if provided, otherwise use defaults
                            circle_size = sphere_size if sphere_size is not None else 36  # 6nm diameter circles
                            sphere_edgecolor = sphere_color if sphere_color is not None else 'gold'
                            axxy.scatter(selected_aunps[:,0], selected_aunps[:,1], s=circle_size, c='none', alpha=1.0, edgecolors=sphere_edgecolor, linewidth=1.5)
                            axxz.scatter(selected_aunps[:,2], selected_aunps[:,1], s=circle_size, c='none', alpha=1.0, edgecolors=sphere_edgecolor, linewidth=1.5)
                            axyz.scatter(selected_aunps[:,0], selected_aunps[:,2], s=circle_size, c='none', alpha=1.0, edgecolors=sphere_edgecolor, linewidth=1.5)
                            
                            fig.savefig(aunp_path_results_organized)
                            fig.savefig(aunp_path_tomogram)
                            plt.close(fig)
                            print(f"    ✓ Saved PNG: {aunp_filename}")
                            files_created.append(aunp_filename)
                    
                    # Generate distance-colored AuNP visualization (colored by distance to postsynaptic membrane)
                    selected_aunps_with_distances = select_aunps_with_distances_findingampa_style(zonogram_findingampa, None, tomogram_path, active_zone_id, original_zone_data)
                    if len(selected_aunps_with_distances['positions']) > 0:
                        selected_aunps_dist = selected_aunps_with_distances['positions']
                        post_distances = selected_aunps_with_distances['distances']
                        
                        dist_filename = f"{tomogram_name}_active_zonogram_{zone_name}_selected_aunps_by_distance_to_post{suffix}.png"
                        dist_path_results_organized = results_active_zonograms_dir_full / dist_filename
                        dist_path_tomogram = tomogram_active_zonograms_dir / dist_filename
                        
                        if dist_path_results_organized.exists() and dist_path_tomogram.exists() and not rerun:
                            print(f"    Skipping {dist_filename}, already exists.")
                            files_created.append(dist_filename)
                        else:
                            fig = render_active_zonograms_findingampa_style(zonogram_findingampa)
                            (axxy, axxz, axyz) = fig.get_axes()
                            
                            # Use custom sphere size if provided, otherwise use default
                            circle_size = sphere_size if sphere_size is not None else 36
                            
                            # Create colormap for distances (use 'viridis' or 'plasma' for good visibility)
                            import matplotlib.cm as cm
                            import matplotlib.colors as mcolors
                            
                            # Normalize distances for colormap
                            if len(post_distances) > 0:
                                # Filter out NaN values for normalization
                                valid_distances = post_distances[~np.isnan(post_distances)]
                                if len(valid_distances) > 0:
                                    # Use custom min/max if provided, otherwise auto-calculate from data
                                    if aunp_distance_min is not None and aunp_distance_max is not None:
                                        vmin = aunp_distance_min
                                        vmax = aunp_distance_max
                                    else:
                                        vmin = np.min(valid_distances)
                                        vmax = np.max(valid_distances)
                                    if vmin == vmax:
                                        # All distances are the same, use a single color
                                        vmin = max(0, vmin - 1)
                                        vmax = vmax + 1
                                    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
                                    cmap = cm.get_cmap('viridis')
                                    
                                    # Get colors for each AuNP
                                    colors = cmap(norm(post_distances))
                                    
                                    # Plot AuNPs with distance-based colors
                                    for i, pos in enumerate(selected_aunps_dist):
                                        if not np.isnan(post_distances[i]):
                                            color = colors[i]
                                            axxy.scatter(pos[0], pos[1], s=circle_size, c='none', alpha=1.0, edgecolors=color, linewidth=1.5)
                                            axxz.scatter(pos[2], pos[1], s=circle_size, c='none', alpha=1.0, edgecolors=color, linewidth=1.5)
                                            axyz.scatter(pos[0], pos[2], s=circle_size, c='none', alpha=1.0, edgecolors=color, linewidth=1.5)
                                    
                                    # Add colorbar
                                    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
                                    sm.set_array([])
                                    cbar = fig.colorbar(sm, ax=[axxy, axxz, axyz], orientation='vertical', 
                                                       pad=0.02, aspect=30, shrink=0.8)
                                    cbar.set_label('Distance to Postsynaptic Membrane (nm)', rotation=270, labelpad=20, fontsize=9)
                                else:
                                    # All distances are NaN, plot with default color
                                    default_color = sphere_color if sphere_color is not None else 'gold'
                                    for pos in selected_aunps_dist:
                                        axxy.scatter(pos[0], pos[1], s=circle_size, c='none', alpha=1.0, edgecolors=default_color, linewidth=1.5)
                                        axxz.scatter(pos[2], pos[1], s=circle_size, c='none', alpha=1.0, edgecolors=default_color, linewidth=1.5)
                                        axyz.scatter(pos[0], pos[2], s=circle_size, c='none', alpha=1.0, edgecolors=default_color, linewidth=1.5)
                            
                            fig.savefig(dist_path_results_organized, bbox_inches='tight')
                            fig.savefig(dist_path_tomogram, bbox_inches='tight')
                            plt.close(fig)
                            print(f"    ✓ Saved PNG: {dist_filename}")
                            files_created.append(dist_filename)
                            
                            # Generate filtered distance-colored AuNP visualization (with cutoff)
                            # Use defaults if not provided
                            cutoff_direction = aunp_distance_cutoff_direction if aunp_distance_cutoff_direction is not None else "below"
                            cutoff_value = aunp_distance_cutoff_value if aunp_distance_cutoff_value is not None else 15.0
                            
                            # Filter AuNPs based on cutoff
                            if cutoff_direction == "below":
                                filter_mask = post_distances < cutoff_value
                            else:  # "above"
                                filter_mask = post_distances > cutoff_value
                            
                            # Also filter out NaN values
                            filter_mask = filter_mask & ~np.isnan(post_distances)
                            
                            filtered_positions = selected_aunps_dist[filter_mask]
                            filtered_distances = post_distances[filter_mask]
                            
                            if len(filtered_positions) > 0:
                                cutoff_filename = f"{tomogram_name}_active_zonogram_{zone_name}_selected_aunps_by_distance_to_post_{cutoff_direction}_{cutoff_value}nm{suffix}.png"
                                cutoff_path_results_organized = results_active_zonograms_dir_full / cutoff_filename
                                cutoff_path_tomogram = tomogram_active_zonograms_dir / cutoff_filename
                                
                                if cutoff_path_results_organized.exists() and cutoff_path_tomogram.exists() and not rerun:
                                    print(f"    Skipping {cutoff_filename}, already exists.")
                                    files_created.append(cutoff_filename)
                                else:
                                    fig = render_active_zonograms_findingampa_style(zonogram_findingampa)
                                    (axxy, axxz, axyz) = fig.get_axes()
                                    
                                    # Use custom sphere size if provided, otherwise use default
                                    circle_size = sphere_size if sphere_size is not None else 36
                                    
                                    # Create colormap for distances
                                    import matplotlib.cm as cm
                                    import matplotlib.colors as mcolors
                                    
                                    # Normalize distances for colormap - use range of filtered values only
                                    valid_distances = filtered_distances[~np.isnan(filtered_distances)]
                                    if len(valid_distances) > 0:
                                        # Always calculate colormap range from filtered values only
                                        vmin = np.min(valid_distances)
                                        vmax = np.max(valid_distances)
                                        if vmin == vmax:
                                            # All distances are the same, use a single color
                                            vmin = max(0, vmin - 1)
                                            vmax = vmax + 1
                                        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
                                        cmap = cm.get_cmap('viridis')
                                        
                                        # Get colors for each AuNP
                                        colors = cmap(norm(filtered_distances))
                                        
                                        # Plot filtered AuNPs with distance-based colors
                                        for i, pos in enumerate(filtered_positions):
                                            if not np.isnan(filtered_distances[i]):
                                                color = colors[i]
                                                axxy.scatter(pos[0], pos[1], s=circle_size, c='none', alpha=1.0, edgecolors=color, linewidth=1.5)
                                                axxz.scatter(pos[2], pos[1], s=circle_size, c='none', alpha=1.0, edgecolors=color, linewidth=1.5)
                                                axyz.scatter(pos[0], pos[2], s=circle_size, c='none', alpha=1.0, edgecolors=color, linewidth=1.5)
                                        
                                        # Add colorbar
                                        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
                                        sm.set_array([])
                                        cbar = fig.colorbar(sm, ax=[axxy, axxz, axyz], orientation='vertical', 
                                                           pad=0.02, aspect=30, shrink=0.8)
                                        cbar.set_label('Distance to Postsynaptic Membrane (nm)', rotation=270, labelpad=20, fontsize=9)
                                    
                                    fig.savefig(cutoff_path_results_organized, bbox_inches='tight')
                                    fig.savefig(cutoff_path_tomogram, bbox_inches='tight')
                                    plt.close(fig)
                                    print(f"    ✓ Saved PNG: {cutoff_filename}")
                                    files_created.append(cutoff_filename)
                            else:
                                print(f"    No AuNPs found {cutoff_direction} {cutoff_value} nm from postsynaptic membrane for {zone_name}")
                    
                    # Generate cluster-colored AuNP visualization
                    selected_aunps, cluster_assignments = select_aunps_by_cluster_findingampa_style(zonogram_findingampa, None, tomogram_path, active_zone_id, original_zone_data)
                    if len(selected_aunps) > 0:
                        fig = render_active_zonograms_findingampa_style(zonogram_findingampa)
                        (axxy, axxz, axyz) = fig.get_axes()
                        
                        unique_clusters = sorted(set(cluster_assignments))
                        non_noise_clusters = [c for c in unique_clusters if c != -1]
                        
                        # Use the same color scheme as the combined overlay
                        color_list, _ = _get_cluster_colors(len(non_noise_clusters))
                        
                        cluster_color_map = {}
                        cluster_color_map[-1] = 'grey'
                        for i, cluster in enumerate(non_noise_clusters):
                            cluster_color_map[cluster] = color_list[i]
                        
                        # Use custom sphere size if provided, otherwise use default
                        circle_size = sphere_size if sphere_size is not None else 36
                        for i, (pos, cluster) in enumerate(zip(selected_aunps, cluster_assignments)):
                            color = cluster_color_map.get(cluster, 'gray')
                            axxy.scatter(pos[0], pos[1], s=circle_size, c='none', alpha=1.0, edgecolors=color, linewidth=1.5)
                            axxz.scatter(pos[2], pos[1], s=circle_size, c='none', alpha=1.0, edgecolors=color, linewidth=1.5)
                            axyz.scatter(pos[0], pos[2], s=circle_size, c='none', alpha=1.0, edgecolors=color, linewidth=1.5)
                        
                        # Add fusion points if available
                        fusion_points = []
                        fusion_points_transformed = []
                        try:
                            # Try to load cached fusion points first
                            fusion_points_cache_path = Path(tomogram_path) / "best_alignment" / "STT_results" / "vesicles" / "fusion_points.npy"
                            
                            if fusion_points_cache_path.exists():
                                try:
                                    fusion_points = np.load(fusion_points_cache_path)
                                except Exception as e:
                                    print(f"Could not load cached fusion points: {e}")
                            
                            # Compute fusion points if not cached
                            if len(fusion_points) == 0:
                                try:
                                    from .aunps import compute_fusion_points
                                    fusion_points = compute_fusion_points(tomogram_path, vesicle_distance_threshold=20.0)
                                except Exception as e:
                                    print(f"Could not compute fusion points: {e}")
                                    fusion_points = []
                            
                            if len(fusion_points) > 0:
                                # Transform fusion points to the same coordinate system as AuNPs
                                from torch_affine_utils.utils import homogenise_coordinates
                                import einops
                                
                                # Convert to homogeneous coordinates
                                fusion_points_homog = homogenise_coordinates(torch.tensor(fusion_points, dtype=torch.float32))
                                
                                # Apply transformation matrix
                                M = torch.tensor(original_zone_data['transformation_matrix'], dtype=torch.float32)
                                transformed_fusion_points = M @ einops.rearrange(fusion_points_homog, 'b xyzw -> b xyzw 1')
                                transformed_fusion_points = einops.rearrange(transformed_fusion_points, 'b xyzw 1 -> b xyzw')[:, :3]
                                
                                # Add offset to center in the zonogram
                                center = original_zone_data['center']
                                extent = original_zone_data['extent']
                                new_center = extent // 2
                                fusion_points_transformed = transformed_fusion_points.numpy() + new_center
                                
                                # Filter points within the zonogram
                                valid_mask = np.all(fusion_points_transformed >= 0, axis=1) & np.all(fusion_points_transformed < extent.reshape(1, -1), axis=1)
                                fusion_points_transformed = fusion_points_transformed[valid_mask]
                                
                                if len(fusion_points_transformed) > 0:
                                    # Plot fusion points as orange stars with cyan circles on all three views
                                    for fp in fusion_points_transformed:
                                        # Plot the orange star
                                        axxy.scatter(fp[0], fp[1], color='orange', s=100, alpha=0.9, marker='*', 
                                                   edgecolors='darkorange', linewidth=0.5)
                                        axxz.scatter(fp[2], fp[1], color='orange', s=100, alpha=0.9, marker='*', 
                                                   edgecolors='darkorange', linewidth=0.5)
                                        axyz.scatter(fp[0], fp[2], color='orange', s=100, alpha=0.9, marker='*', 
                                                   edgecolors='darkorange', linewidth=0.5)
                                        
                                        # Add dotted cyan circle (40 nm diameter = 20 nm radius)
                                        # Note: Assuming 1 pixel = 1 nm for the circle radius
                                        circle_radius = 20  # 20 nm radius for 40 nm diameter
                                        circle_xy = plt.Circle((fp[0], fp[1]), circle_radius, fill=False, 
                                                             color='cyan', linestyle=':', linewidth=1.5, alpha=0.8)
                                        circle_xz = plt.Circle((fp[2], fp[1]), circle_radius, fill=False, 
                                                             color='cyan', linestyle=':', linewidth=1.5, alpha=0.8)
                                        circle_yz = plt.Circle((fp[0], fp[2]), circle_radius, fill=False, 
                                                             color='cyan', linestyle=':', linewidth=1.5, alpha=0.8)
                                        
                                        axxy.add_patch(circle_xy)
                                        axxz.add_patch(circle_xz)
                                        axyz.add_patch(circle_yz)
                        except Exception as e:
                            print(f"Warning: Could not load fusion points: {e}")
                        
                        legend_handles = []
                        legend_labels = []
                        if -1 in cluster_color_map:
                            noise_handle = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
                                                     markeredgecolor=cluster_color_map[-1], markersize=8, linewidth=1.5)
                            legend_handles.append(noise_handle)
                            legend_labels.append('Non-clustered')
                        for cluster in sorted(non_noise_clusters):
                            cluster_handle = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
                                                       markeredgecolor=cluster_color_map[cluster], markersize=8, linewidth=1.5)
                            legend_handles.append(cluster_handle)
                            legend_labels.append(f'Cluster {cluster}')
                        
                        # Add fusion points to legend if they were plotted
                        if len(fusion_points) > 0 and len(fusion_points_transformed) > 0:
                            # Create a custom legend entry showing both star and circle
                            from matplotlib.patches import Circle
                            from matplotlib.lines import Line2D
                            
                            # Create a custom legend handle that shows both elements
                            fusion_handle = Line2D([0], [0], marker='*', color='w', markerfacecolor='orange', 
                                                 markeredgecolor='darkorange', markersize=10, linewidth=0.5,
                                                 linestyle=':', markeredgewidth=1.5)
                            legend_handles.append(fusion_handle)
                            legend_labels.append('Fusion Sites (40nm)')
                        
                        if legend_handles:
                            fig.legend(legend_handles, legend_labels, loc='lower right', bbox_to_anchor=(1.0, 0.0),
                                      fontsize=8, frameon=True, fancybox=True, shadow=True)
                        
                        # Save the version with fusion points
                        cluster_filename = f"{tomogram_name}_active_zonogram_{zone_name}_selected_aunps_by_cluster{suffix}.png"
                        cluster_path_results_organized = results_active_zonograms_dir_full / cluster_filename
                        cluster_path_tomogram = tomogram_active_zonograms_dir / cluster_filename
                        
                        if cluster_path_results_organized.exists() and cluster_path_tomogram.exists() and not rerun:
                            print(f"    Skipping {cluster_filename}, already exists.")
                            files_created.append(cluster_filename)
                        else:
                            fig.savefig(cluster_path_results_organized)
                            fig.savefig(cluster_path_tomogram)
                            print(f"    ✓ Saved PNG: {cluster_filename}")
                            files_created.append(cluster_filename)
                        
                        # Create and save a version without fusion points for comparison
                        fig_no_fusion = render_active_zonograms_findingampa_style(zonogram_findingampa)
                        (axxy_no_fusion, axxz_no_fusion, axyz_no_fusion) = fig_no_fusion.get_axes()
                        
                        # Plot only the cluster-colored AuNPs (no fusion points)
                        for i, (pos, cluster) in enumerate(zip(selected_aunps, cluster_assignments)):
                            color = cluster_color_map.get(cluster, 'gray')
                            axxy_no_fusion.scatter(pos[0], pos[1], s=circle_size, c='none', alpha=1.0, edgecolors=color, linewidth=1.5)
                            axxz_no_fusion.scatter(pos[2], pos[1], s=circle_size, c='none', alpha=1.0, edgecolors=color, linewidth=1.5)
                            axyz_no_fusion.scatter(pos[0], pos[2], s=circle_size, c='none', alpha=1.0, edgecolors=color, linewidth=1.5)
                        
                        # Add legend without fusion points
                        legend_handles_no_fusion = []
                        legend_labels_no_fusion = []
                        if -1 in cluster_color_map:
                            noise_handle = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
                                                     markeredgecolor=cluster_color_map[-1], markersize=8, linewidth=1.5)
                            legend_handles_no_fusion.append(noise_handle)
                            legend_labels_no_fusion.append('Non-clustered')
                        for cluster in sorted(non_noise_clusters):
                            cluster_handle = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
                                                       markeredgecolor=cluster_color_map[cluster], markersize=8, linewidth=1.5)
                            legend_handles_no_fusion.append(cluster_handle)
                            legend_labels_no_fusion.append(f'Cluster {cluster}')
                        
                        if legend_handles_no_fusion:
                            fig_no_fusion.legend(legend_handles_no_fusion, legend_labels_no_fusion, loc='lower right', bbox_to_anchor=(1.0, 0.0),
                                               fontsize=8, frameon=True, fancybox=True, shadow=True)
                        
                        # Save the version without fusion points
                        cluster_no_fusion_filename = f"{tomogram_name}_active_zonogram_{zone_name}_selected_aunps_by_cluster_no_fusion{suffix}.png"
                        cluster_no_fusion_path_results_organized = results_active_zonograms_dir_full / cluster_no_fusion_filename
                        cluster_no_fusion_path_tomogram = tomogram_active_zonograms_dir / cluster_no_fusion_filename
                        
                        if cluster_no_fusion_path_results_organized.exists() and cluster_no_fusion_path_tomogram.exists() and not rerun:
                            print(f"    Skipping {cluster_no_fusion_filename}, already exists.")
                            files_created.append(cluster_no_fusion_filename)
                        else:
                            fig_no_fusion.savefig(cluster_no_fusion_path_results_organized)
                            fig_no_fusion.savefig(cluster_no_fusion_path_tomogram)
                            print(f"    ✓ Saved PNG: {cluster_no_fusion_filename}")
                            files_created.append(cluster_no_fusion_filename)
                        
                        plt.close(fig)
                        plt.close(fig_no_fusion)
                    
                    # Generate packing density visualization
                    packing_density_file = Path(tomogram_path) / "best_alignment" / "STT_results" / "aunps" / "packing_density_results.json"
                    if packing_density_file.exists() and zone_name in zonogram_results['zonogram_data']:
                        try:
                            import json
                            from scipy.interpolate import griddata
                            from .activezone import transform_coordinates_to_active_zonogram
                            
                            with open(packing_density_file, 'r') as f:
                                packing_density_data = json.load(f)
                            
                            if zone_name in packing_density_data:
                                zone_packing_data = packing_density_data[zone_name]
                                v_array = np.array(zone_packing_data['v_array'])
                                # Convert None (JSON null) back to NaN for masked edge vertices
                                packing_coefficient = np.array(
                                    [np.nan if x is None else x for x in zone_packing_data['packing_coefficient']]
                                )
                                # Use only interior vertices (non-NaN) for interpolation to avoid edge artifacts
                                valid_mask = ~np.isnan(packing_coefficient)
                                v_array_valid = v_array[valid_mask]
                                packing_valid = packing_coefficient[valid_mask]

                                # Transform coordinates to active zonogram space
                                transformed_v = transform_coordinates_to_active_zonogram(v_array_valid, original_zone_data)

                                if len(transformed_v) > 0:
                                    # Create figure EXACTLY the same way as regular active zonogram
                                    fig_packing = render_active_zonograms_findingampa_style(zonogram_findingampa)
                                    (axxy_packing, axxz_packing, axyz_packing) = fig_packing.get_axes()
                                    
                                    # Get the base image to determine its extent
                                    res_ddw = zonogram_findingampa[2]
                                    base_image = torch.min(res_ddw, axis=0).values
                                    base_image_shape = base_image.shape  # (y, x) = (shape[1], shape[2] from res_ddw)
                                    
                                    # Get the extent of the base image (imshow default is [left, right, bottom, top])
                                    # For a (y, x) array with origin='lower', extent is [0, x_size, 0, y_size]
                                    base_extent = [0, base_image_shape[1], 0, base_image_shape[0]]
                                    
                                    # Create grid for interpolation matching the base image shape exactly
                                    grid_y, grid_x = np.mgrid[0:base_image_shape[0], 0:base_image_shape[1]]
                                    
                                    # Interpolate packing density onto the grid (only interior vertices)
                                    # fill_value=nan so extrapolated regions stay transparent
                                    density_map = griddata(
                                        transformed_v[:, :2],  # (x, y) points
                                        packing_valid,
                                        (grid_x, grid_y),  # Grid points (x, y)
                                        method='linear',
                                        fill_value=np.nan
                                    )
                                    
                                    # Verify density_map shape matches base image exactly
                                    if density_map.shape != base_image_shape:
                                        # If shape doesn't match, we need to fix it
                                        from scipy.ndimage import zoom
                                        zoom_factors = (base_image_shape[0] / density_map.shape[0], 
                                                       base_image_shape[1] / density_map.shape[1])
                                        density_map = zoom(density_map, zoom_factors, order=1)
                                    
                                    # Overlay the heatmap (mask NaN/edge regions so they stay transparent)
                                    density_masked = np.ma.masked_invalid(density_map)
                                    im = axxy_packing.imshow(
                                        density_masked,
                                        cmap='hot',
                                        alpha=0.6,
                                        origin='lower',
                                        vmin=0.0,
                                        vmax=1.0,
                                        extent=base_extent,  # Use same extent as base image
                                        zorder=10  # Ensure it's on top
                                    )
                                    
                                    # Add colorbar
                                    cbar = fig_packing.colorbar(im, ax=axxy_packing, fraction=0.046, pad=0.04)
                                    cbar.set_label('Estimated AMPA Receptor Packing Coefficient', rotation=270, labelpad=15)
                                    
                                    # Save packing density visualization
                                    packing_filename = f"{tomogram_name}_active_zonogram_{zone_name}_packing_density{suffix}.png"
                                    packing_path_results_organized = results_active_zonograms_dir_full / packing_filename
                                    packing_path_tomogram = tomogram_active_zonograms_dir / packing_filename
                                    
                                    if packing_path_results_organized.exists() and packing_path_tomogram.exists() and not rerun:
                                        print(f"    Skipping {packing_filename}, already exists.")
                                        files_created.append(packing_filename)
                                    else:
                                        fig_packing.savefig(packing_path_results_organized)
                                        fig_packing.savefig(packing_path_tomogram)
                                        plt.close(fig_packing)
                                        print(f"    ✓ Saved PNG: {packing_filename}")
                                        files_created.append(packing_filename)
                        except Exception as e:
                            print(f"    Warning: Could not create packing density visualization for {zone_name}: {e}")
                            import traceback
                            traceback.print_exc()
            else:
                print("No active zonograms found")
        else:
            print("No active zonograms found")
            
        print()  # Spacer line
        
        # Step 3: Check if AuNP analysis was completed
        # Step 3: Checking AuNP analysis status
        
        # Check for required AuNP analysis files
        aunp_analysis_path = Path(tomogram_path) / "best_alignment" / "STT_results" / "aunps"
        cluster_data_path = aunp_analysis_path / "aunp_clusters.star"
        
        if not aunp_analysis_path.exists():
            print(f"Warning: AuNP analysis directory not found at {aunp_analysis_path}")
            print("Skipping mini zonogram analysis - AuNP analysis must be completed first.")
            return {"success": True, "regular_zonograms": len(files_created), "mini_zonograms": 0, "files_created": files_created}
        
        if not cluster_data_path.exists():
            print(f"Warning: AuNP cluster data not found at {cluster_data_path}")
            print("Skipping mini zonogram analysis - AuNP analysis must be completed first.")
            return {"success": True, "regular_zonograms": len(files_created), "mini_zonograms": 0, "files_created": files_created}
        
        # AuNP analysis files found, proceeding with mini zonogram analysis
        
        # Step 4: Mini Zonogram Analysis
        
        # Load cluster data
        import starfile
        cluster_df = starfile.read(cluster_data_path)
        # Loaded cluster data
        
        # Identify small clusters (excluding noise cluster -1)
        cluster_counts = cluster_df['aunp_cluster'].value_counts()
        small_clusters = cluster_counts[(cluster_counts < 11) & (cluster_counts.index != -1)]
        
        # Found small clusters for mini zonogram analysis
        
        mini_zonogram_count = 0
        if len(small_clusters) > 0:
            # Create cluster color map (same as regular zonogram analysis)
            unique_clusters = sorted(set(cluster_df['aunp_cluster'].values))
            non_noise_clusters = [c for c in unique_clusters if c != -1]
            
            # Use the same color scheme as the combined overlay
            color_list, _ = _get_cluster_colors(len(non_noise_clusters))
            
            cluster_color_map = {}
            cluster_color_map[-1] = 'grey'
            for i, cluster in enumerate(non_noise_clusters):
                cluster_color_map[cluster] = color_list[i]
            
            # Generate mini zonograms for each small cluster
            for cluster_id in small_clusters.index:
                # Get AuNPs for this cluster
                cluster_data = cluster_df[cluster_df['aunp_cluster'] == cluster_id]
                
                # Create mini zonogram (use filtered active zones)
                # Create mini zonogram directory in organized structure
                results_active_zonograms_dir_mini = Path("results") / "visualizations" / tomogram_name / "active_zonograms" / "mini"
                results_active_zonograms_dir_mini.mkdir(parents=True, exist_ok=True)
                
                success = create_mini_zonogram_for_cluster(
                    cluster_data, cluster_id, tomogram_path, tomogram_active_zonograms_dir, 
                    results_active_zonograms_dir_mini, active_zones_data, cluster_color_map, 
                    tomogram_name, default_suffix, rerun
                )
                
                if success:
                    mini_zonogram_count += 1
                    files_created.append(f"{tomogram_name}_mini_zonogram_cluster_{cluster_id}{default_suffix}.png")
            
            print(f"Successfully created {mini_zonogram_count} mini zonograms out of {len(small_clusters)} small clusters")
        else:
            print("No small clusters found (excluding noise).")
        
        print()  # Spacer line
        print("Combined active zonogram analysis completed successfully!")
        
        return {
            "success": True,
            "regular_zonograms": len([f for f in files_created if 'active_zonogram' in f and 'mini' not in f]),
            "mini_zonograms": mini_zonogram_count,
            "files_created": files_created
        }
            
    except Exception as e:
        print(f"Error in active zonogram analysis: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "reason": f"Error: {e}"}




def create_mini_zonogram_for_cluster(cluster_data, cluster_id, tomogram_path, tomogram_azograms_dir, results_azograms_dir, active_zones_data, cluster_color_map, tomogram_name, suffix="", rerun=False):
    """
    Create a mini zonogram centered on a specific small cluster.
    Uses the same transformation matrix calculation as regular active zonograms.
    Uses the same color scheme as the regular zonogram analysis.
    Saves files in both tomogram's STT_results/active_zonograms and results/visualizations/active_zonograms.
    """
    # Creating mini zonogram for cluster
    
    # Debug: Check available active zones
    # Available active zones checked
    
    # Get cluster center
    cluster_center = cluster_data[['faCoordinateX', 'faCoordinateY', 'faCoordinateZ']].mean().values
    
    # Calculate transformation matrix using the same approach as regular active zonograms
    from torch_affine_utils.transforms_3d import T
    from torch_affine_utils.utils import homogenise_coordinates
    import torch
    import einops
    
    # Find the closest active zone to this cluster
    closest_zone_name = None
    min_distance = float('inf')
    
    for zone_name, zone_data in active_zones_data['active_zones'].items():
        if len(zone_data['active_presynaptic_points']) > 0 and len(zone_data['active_postsynaptic_points']) > 0:
            # Calculate distance from cluster center to zone center
            center_presyn = np.mean(zone_data['active_presynaptic_points'], axis=0)
            center_postsyn = np.mean(zone_data['active_postsynaptic_points'], axis=0)
            zone_center = (center_presyn + center_postsyn) / 2.0
            distance = np.linalg.norm(cluster_center - zone_center)
            
            if distance < min_distance:
                min_distance = distance
                closest_zone_name = zone_name
    
    if closest_zone_name is None:
        print(f"    Warning: No active zones found, using identity matrix")
        coordinate_system = np.eye(3)
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, 3] = -cluster_center
        extent = np.array([100, 100, 100])
    else:
        # Use the membrane data from the closest active zone
        zone_data = active_zones_data['active_zones'][closest_zone_name]
        
        # Construct coordinate system using the same logic as regular active zonograms
        # Get 100 random points in postsynapse (or all if fewer than 100)
        post_points = zone_data['active_postsynaptic_points']
        if len(post_points) > 100:
            post_points_sel = post_points[np.random.choice(post_points.shape[0], 100, replace=False)]
        else:
            post_points_sel = post_points
            
        # Get closest points in presynapse
        pre_dis, pre_i = KDTree(zone_data['active_presynaptic_points']).query(post_points_sel)
        pre_points_el = zone_data['active_presynaptic_points'][pre_i]
        norm_vector = np.mean(post_points_sel - pre_points_el, axis=0)
        norm_vector = norm_vector / np.linalg.norm(norm_vector)
        
        z = np.array([0, 0, 1])
        xp = np.cross(norm_vector, z)
        yp = np.cross(norm_vector, xp)
        xp = xp / np.linalg.norm(xp) 
        yp = yp / np.linalg.norm(yp)

        # Generate 4x4 transformation matrix bringing center to 0 and make xp, yp, norm_vector the new axes
        M = torch.eye(4)
        M[0, :3] = torch.tensor(xp)
        M[1, :3] = torch.tensor(yp)
        M[2, :3] = torch.tensor(norm_vector)
        M = M @ T(-cluster_center)  # Center on cluster instead of zone center
        
        transformation_matrix = M.numpy()
        coordinate_system = transformation_matrix[:3, :3]
        
        # Calculate extent based on cluster spread plus padding
        cluster_positions = cluster_data[['faCoordinateX', 'faCoordinateY', 'faCoordinateZ']].values
        cluster_spread = np.max(cluster_positions, axis=0) - np.min(cluster_positions, axis=0)
        extent = np.maximum(cluster_spread + 50, 100).astype(int)  # At least 100nm, or cluster spread + 50nm padding
    
    # Create mini zonogram data structure
    # Use the closest zone name so extract_active_zonogram can find it
    zone_name_to_use = closest_zone_name if closest_zone_name else f'mini_cluster_{cluster_id}'

    mini_zonogram_data = {
        'center': cluster_center,
        'extent': extent,
        'transformation_matrix': transformation_matrix,
        'zone_name': zone_name_to_use
    }
    
    # Extract mini zonogram using the same approach as regular active zonograms
    try:
        # Use extract_active_zonogram function to properly extract with transformation
        # Create the expected data structure for extract_active_zonogram (matching define_active_zonogram output)
        mini_zonogram_dict = {
            'status': 'completed',
            'active_zone_count': 1,
            'zonogram_data': {zone_name_to_use: mini_zonogram_data}
        }
        extracted_data = extract_active_zonogram(mini_zonogram_dict, active_zones_data, tomogram_path)
        
        if extracted_data is None or 'rendered_zonograms' not in extracted_data or zone_name_to_use not in extracted_data['rendered_zonograms']:
            print(f"    Failed to extract mini zonogram for cluster {cluster_id}")
            return False
        
        # Get the transformed tomogram
        transformed_tomogram = torch.tensor(extracted_data['rendered_zonograms'][zone_name_to_use]['transformed_tomogram'])
        
        # Create mini zonogram data in findingampa format
        mini_zonogram_findingampa = (coordinate_system, cluster_center, transformed_tomogram, ())
        
        # Save mini zonogram files to both locations
        mini_filename_base = f"{tomogram_name}_mini_zonogram_cluster_{cluster_id}{suffix}"
        
        # 1. Save MRC file to tomogram directory only (not to results directory)
        mrc_filename = f"{mini_filename_base}.mrc"
        mrcfile.write(tomogram_azograms_dir / mrc_filename, transformed_tomogram.numpy(), overwrite=True)
        print(f"    ✓ Saved MRC: {mrc_filename}")
        
        # 2. Save NPY file to tomogram directory only (not to results directory)
        npy_filename = f"{mini_filename_base}.npy"
        npy_data = {
            "cs": coordinate_system, 
            "center": cluster_center, 
            "objects": (),
            "cluster_id": cluster_id,
            "aunp_count": len(cluster_data)
        }
        np.save(tomogram_azograms_dir / npy_filename, npy_data, allow_pickle=True)
        print(f"    ✓ Saved NPY: {npy_filename}")
        
        # 3. Generate main PNG and save to all locations
        png_filename = f"{mini_filename_base}.png"
        png_path_tomogram = tomogram_azograms_dir / png_filename
        png_path_results = results_azograms_dir / png_filename
        
        if png_path_tomogram.exists() and png_path_results.exists() and not rerun:
            print(f"    Skipping {png_filename}, already exists.")
        else:
            fig, axxy = render_mini_zonogram_xy_only(mini_zonogram_findingampa)
            fig.savefig(png_path_tomogram)
            fig.savefig(png_path_results)
            plt.close(fig)
            print(f"    ✓ Saved PNG: {png_filename}")
        
        # Transform cluster AuNPs to mini zonogram coordinates using the same transformation as the tomogram
        # (Calculate this once for use in both aunp visualization and comparison figure)
        cluster_positions = cluster_data[['faCoordinateX', 'faCoordinateY', 'faCoordinateZ']].values
        
        # Apply the same transformation matrix that was used for the tomogram
        from torch_affine_utils.utils import homogenise_coordinates
        import einops
        
        # Convert to homogeneous coordinates
        cluster_positions_homog = homogenise_coordinates(torch.tensor(cluster_positions, dtype=torch.float32))
        
        # Apply transformation matrix
        M = torch.tensor(transformation_matrix, dtype=torch.float32)
        transformed_cluster_positions = M @ einops.rearrange(cluster_positions_homog, 'b xyzw -> b xyzw 1')
        transformed_cluster_positions = einops.rearrange(transformed_cluster_positions, 'b xyzw 1 -> b xyzw')[:, :3]
        
        # Add offset to center in the mini zonogram
        new_center = extent // 2
        cluster_positions_transformed = transformed_cluster_positions.numpy() + new_center
        
        # Filter points within the mini zonogram
        valid_mask = np.all(cluster_positions_transformed >= 0, axis=1) & np.all(cluster_positions_transformed < extent.reshape(1, -1), axis=1)
        cluster_positions_transformed = cluster_positions_transformed[valid_mask]
        
        # 4. Generate AuNP visualization and save to all locations
        aunp_filename = f"{mini_filename_base}_aunps.png"
        aunp_path_tomogram = tomogram_azograms_dir / aunp_filename
        aunp_path_results = results_azograms_dir / aunp_filename
        
        if aunp_path_tomogram.exists() and aunp_path_results.exists() and not rerun:
            print(f"    Skipping {aunp_filename}, already exists.")
        else:
            # Create AuNP visualization
            fig, axxy = render_mini_zonogram_xy_only(mini_zonogram_findingampa)
            
            if len(cluster_positions_transformed) > 0:
                # Plot AuNPs with cluster colors
                circle_size = 36  # 6nm diameter circles
                cluster_color = cluster_color_map.get(cluster_id, 'red')  # Default to red if cluster not in map
                axxy.scatter(cluster_positions_transformed[:,0], cluster_positions_transformed[:,1],
                            s=circle_size, c='none', alpha=1.0, edgecolors=cluster_color, linewidth=1.5)
            
            fig.savefig(aunp_path_tomogram)
            fig.savefig(aunp_path_results)
            plt.close(fig)
            print(f"    ✓ Saved AuNPs: {aunp_filename}")
        
        # 5. Generate three-panel comparison PNG: mini zonogram, mini zonogram with AuNPs, and mini zonogram with distances
        comparison_filename = f"{mini_filename_base}_comparison.png"
        comparison_path_tomogram = tomogram_azograms_dir / comparison_filename
        comparison_path_results = results_azograms_dir / comparison_filename
        
        if comparison_path_tomogram.exists() and comparison_path_results.exists() and not rerun:
            print(f"    Skipping {comparison_filename}, already exists.")
        else:
            # Generate left panel (mini zonogram only) using the same function
            fig_left, axxy_left = render_mini_zonogram_xy_only(mini_zonogram_findingampa)
            
            # Generate middle panel (mini zonogram with AuNPs) using the same function
            fig_middle, axxy_middle = render_mini_zonogram_xy_only(mini_zonogram_findingampa)
            
            # Generate right panel (mini zonogram with AuNPs and distances) using the same function with extra legend space
            fig_right, axxy_right = render_mini_zonogram_xy_only(mini_zonogram_findingampa, include_legend_space=True, extra_width_multiplier=2.2)
            
            # Add AuNPs to middle panel
            if len(cluster_positions_transformed) > 0:
                circle_size = 36  # 6nm diameter circles
                cluster_color = cluster_color_map.get(cluster_id, 'red')  # Default to red if cluster not in map
                axxy_middle.scatter(cluster_positions_transformed[:,0], cluster_positions_transformed[:,1],
                                  s=circle_size, c='none', alpha=1.0, edgecolors=cluster_color, linewidth=1.5)
            
            # Add legend to the middle panel (top left)
            legend_text = f"Cluster {cluster_id}"
            cluster_color = cluster_color_map.get(cluster_id, 'red')
            legend_handle = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
                                     markeredgecolor=cluster_color, markersize=8, linewidth=1.5)
            axxy_middle.legend([legend_handle], [legend_text], loc='upper left', 
                             fontsize=10, frameon=True, fancybox=True, shadow=True)
            
            # Add AuNPs and distance lines to right panel
            if len(cluster_positions_transformed) > 0:
                circle_size = 36  # 6nm diameter circles
                cluster_color = cluster_color_map.get(cluster_id, 'red')  # Default to red if cluster not in map
                axxy_right.scatter(cluster_positions_transformed[:,0], cluster_positions_transformed[:,1],
                                  s=circle_size, c='none', alpha=1.0, edgecolors=cluster_color, linewidth=1.5)
                
                # Calculate distances between all pairs of AuNPs and draw lines for those < 15 nm apart
                from scipy.spatial.distance import pdist, squareform
                
                # Calculate pairwise distances in the original coordinate system (nm)
                # Use all original positions for distance calculation (before filtering)
                original_positions = cluster_data[['faCoordinateX', 'faCoordinateY', 'faCoordinateZ']].values
                distances = pdist(original_positions)
                distance_matrix = squareform(distances)
                
                # Define a set of distinct colors for the distance lines
                line_colors = ['yellow', 'cyan', 'magenta', 'orange', 'lime', 'red', 'blue', 'green',
                              'pink', 'purple', 'brown', 'gray', 'olive', 'navy', 'teal', 'maroon']
                
                # Find pairs of AuNPs that are less than 15 nm apart and collect distance info
                # Only include pairs where both positions are valid (within zonogram bounds)
                distance_pairs = []
                color_idx = 0
                
                # Build a mapping from original index to filtered index
                # valid_mask is a boolean array where True means the position is valid
                valid_indices_map = {}
                filtered_idx = 0
                for orig_idx in range(len(valid_mask)):
                    if valid_mask[orig_idx]:
                        valid_indices_map[orig_idx] = filtered_idx
                        filtered_idx += 1
                
                for orig_i in range(len(original_positions)):
                    for orig_j in range(orig_i+1, len(original_positions)):
                        if distance_matrix[orig_i, orig_j] < 15.0:  # Less than 15 nm apart
                            # Only include if both positions are valid (within zonogram bounds)
                            if valid_mask[orig_i] and valid_mask[orig_j]:
                                distance_pairs.append({
                                    'i': valid_indices_map[orig_i], 'j': valid_indices_map[orig_j],
                                    'orig_i': orig_i, 'orig_j': orig_j,  # Store original indices for legend
                                    'distance': distance_matrix[orig_i, orig_j],
                                    'color': line_colors[color_idx % len(line_colors)]
                                })
                                color_idx += 1
                
                # Draw lines for each distance pair
                for pair in distance_pairs:
                    # Get the transformed positions for these two AuNPs
                    pos1 = cluster_positions_transformed[pair['i']]
                    pos2 = cluster_positions_transformed[pair['j']]
                    
                    # Draw a line between them with unique color
                    axxy_right.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], 
                                 color=pair['color'], linewidth=1.5, alpha=0.8)
                
                # Create distance legend in top right corner
                if distance_pairs:
                    legend_text_distances = []
                    legend_colors_distances = []
                    for idx, pair in enumerate(distance_pairs):
                        # Use original indices (1-indexed) for legend
                        legend_text_distances.append(f"AuNP {pair['orig_i']+1}-{pair['orig_j']+1}: {pair['distance']:.1f}nm")
                        legend_colors_distances.append(pair['color'])
                    
                    # Create custom legend handles with matching colors
                    from matplotlib.lines import Line2D
                    legend_handles_distances = [Line2D([0], [0], color=color, linewidth=2) for color in legend_colors_distances]
                    
                    # Add legend to the right of the figure (outside the plot area)
                    axxy_right.legend(legend_handles_distances, legend_text_distances, loc='center left',
                                     fontsize=6, frameon=True, fancybox=True, shadow=True,
                                     bbox_to_anchor=(1.05, 0.5), framealpha=0.9)
            
            # Save individual figures to memory
            import io
            left_buffer = io.BytesIO()
            fig_left.savefig(left_buffer, format='png', bbox_inches='tight', pad_inches=0)
            left_buffer.seek(0)
            
            middle_buffer = io.BytesIO()
            fig_middle.savefig(middle_buffer, format='png', bbox_inches='tight', pad_inches=0)
            middle_buffer.seek(0)
            
            # Adjust subplot to create space only on the right for the legend
            plt.figure(fig_right.number)
            plt.subplots_adjust(right=0.7)  # Create space on right side only
            
            right_buffer = io.BytesIO()
            fig_right.savefig(right_buffer, format='png', bbox_inches='tight', pad_inches=0)
            right_buffer.seek(0)
            
            # Close the individual figures
            plt.close(fig_left)
            plt.close(fig_middle)
            plt.close(fig_right)
            
            # Load the images and combine them into a three-panel layout
            from PIL import Image
            left_img = Image.open(left_buffer)
            middle_img = Image.open(middle_buffer)
            right_img = Image.open(right_buffer)
            
            # Create combined image with white background, spacers, and border
            spacer_width = 12  # 12 pixel white spacer between panels
            border_width = 10  # 10 pixel white border around entire image
            total_width = left_img.width + spacer_width + middle_img.width + spacer_width + right_img.width + (2 * border_width)
            max_height = max(left_img.height, middle_img.height, right_img.height) + (2 * border_width)
            
            combined_img = Image.new('RGB', (total_width, max_height), 'white')
            combined_img.paste(left_img, (border_width, border_width))
            combined_img.paste(middle_img, (left_img.width + spacer_width + border_width, border_width))
            combined_img.paste(right_img, (left_img.width + spacer_width + middle_img.width + spacer_width + border_width, border_width))
            
            # Save the combined image to both locations
            combined_img.save(comparison_path_tomogram)
            combined_img.save(comparison_path_results)
            print(f"    ✓ Saved comparison: {comparison_filename}")
        
        return True
        
    except Exception as e:
        print(f"    Error creating mini zonogram for cluster {cluster_id}: {e}")
        return False

def generate_zonogram_pdf_summaries(zonogram_output_dir, tomo_paths, data_dir=None):
    """Generate PDF summaries for zonogram results - EXACT SAME CODE as original GUI."""
    try:
        # Generating all zonograms summary PDF
        generate_all_zonograms_pdf(tomo_paths, data_dir)
        
        # Generating mini zonograms summary PDF
        generate_mini_zonograms_pdf(tomo_paths, data_dir)
            
    except Exception as e:
        print(f"Error generating PDF summaries: {e}")
        import traceback
        traceback.print_exc()

def generate_all_zonograms_pdf(tomo_paths, data_dir=None):
    """Generate a comprehensive PDF showing all zonogram images from processed tomograms - EXACT SAME CODE as original GUI."""
    try:
        from pathlib import Path
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.platypus import SimpleDocTemplate, Image, PageBreak, Spacer, Paragraph
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.lib import colors
        from PIL import Image as PILImage
        
        # Use the processed tomograms list instead of reading CSV
        if not tomo_paths:
            print("    Warning: No processed tomograms found, skipping PDF generation")
            return
        
        # Extract tomogram names and sets from processed tomograms
        tomogram_data = []
        for tomo_info in tomo_paths:
            # Handle both old format (just path) and new format (path, set_name, active_zones)
            if isinstance(tomo_info, tuple) and len(tomo_info) == 3:
                if len(tomo_info) >= 4:
                    tomo_path, set_name, aunp_active_zones, _alignment_dir = tomo_info
                else:
                    tomo_path, set_name, aunp_active_zones = tomo_info
            else:
                tomo_path = tomo_info
                set_name = None
                aunp_active_zones = None
            
            tomo_path = Path(tomo_path)
            tomogram_name = tomo_path.name
            # Extract set from path (e.g., data/15F1/TOP_TOMOS/tomogram_name -> 15F1)
            tomogram_set = tomo_path.parent.parent.name
            tomogram_data.append((tomogram_name, tomogram_set))
        
        print(f"    Processing {len(tomogram_data)} tomograms for PDF generation")
        
        # Create output directory for summary PDFs
        output_dir = Path("results/visualizations/pdf_summaries")
        output_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = output_dir / "all_zonograms_summary.pdf"
        
        print(f"    Generating PDF: {pdf_path}")
        
        # Create PDF document
        doc = SimpleDocTemplate(str(pdf_path), pagesize=A4)
        story = []
        styles = getSampleStyleSheet()
        
        # Create custom style for tomogram names
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=16,
            spaceAfter=20,
            textColor=colors.darkblue
        )
        
        # Process each tomogram
        for i, (tomogram_name, tomogram_set) in enumerate(tomogram_data, 1):
            print(f"    [{i}/{len(tomogram_data)}] Processing {tomogram_name}...", end=" ", flush=True)
            
            # Get the active zone indices for this tomogram
            selected_az_indices = None
            for tomo_info in tomo_paths:
                if isinstance(tomo_info, tuple) and len(tomo_info) == 3:
                    if len(tomo_info) >= 4:
                        tomo_path, set_name, aunp_active_zones, _alignment_dir = tomo_info
                    else:
                        tomo_path, set_name, aunp_active_zones = tomo_info
                    if Path(tomo_path).name == tomogram_name:
                        if aunp_active_zones is not None:
                            az_str = str(aunp_active_zones)
                            if az_str.strip() != "" and az_str.lower() != "nan":
                                try:
                                    selected_az_indices = []
                                    for x in az_str.split(","):
                                        x = x.strip()
                                        if x.isdigit():
                                            selected_az_indices.append(int(x))
                                        elif x.replace(".", "").isdigit():  # Handle floats like "2.0"
                                            selected_az_indices.append(int(float(x)))
                                    # CSV specified active zones
                                except Exception as e:
                                    print(f"    Warning: Error parsing active zone indices for {tomogram_name}: {e}")
                        break
                else:
                    if Path(tomo_info).name == tomogram_name:
                        break
            
            # Look in the new organized structure
            azograms_dir = Path("results") / "visualizations" / tomogram_name / "active_zonograms" / "full"
            
            if not azograms_dir.exists():
                print(f"    Warning: Active zonograms directory not found: {azograms_dir}")
                continue
            
            # Add tomogram name as title
            story.append(Paragraph(f"Tomogram: {tomogram_name}", title_style))
            story.append(Spacer(1, 10))
            
            # Find regular active zonogram files (aunps_by_cluster.png) for this specific tomogram
            regular_zonogram_files = list(azograms_dir.glob(f"{tomogram_name}_active_zonogram_*_selected_aunps_by_cluster_az*.png"))
            
            # Add regular active zonograms first
            for zonogram_file in sorted(regular_zonogram_files):
                try:
                    # Get zone name from filename
                    zone_name = zonogram_file.stem.split('_active_zonogram_')[1].split('_selected_aunps_by_cluster')[0]
                    
                    # Filter by active zone indices if specified in CSV
                    if selected_az_indices is not None:
                        # Extract active zone index from filename suffix (e.g., "_az0" -> 0)
                        try:
                            az_suffix = zonogram_file.stem.split('_az')[-1]
                            az_index = int(az_suffix)
                            if az_index not in selected_az_indices:
                                print(f"      Skipping active zone {zone_name} (index {az_index}) - not in CSV")
                                continue
                        except (ValueError, IndexError):
                            print(f"      Warning: Could not parse active zone index from filename: {zonogram_file.name}")
                            # Include it by default if we can't parse the index
                    
                    # Add zone name as subtitle
                    zone_style = ParagraphStyle(
                        'ZoneTitle',
                        parent=styles['Heading2'],
                        fontSize=12,
                        spaceAfter=10,
                        textColor=colors.darkgreen
                    )
                    story.append(Paragraph(f"Active Zone: {zone_name}", zone_style))
                    
                    # Add the image (preserve aspect ratio but ensure it fits on page)
                    # First, get the original image dimensions
                    pil_img = PILImage.open(str(zonogram_file))
                    orig_width, orig_height = pil_img.size
                    aspect_ratio = orig_width / orig_height
                    
                    # Calculate maximum dimensions that fit on page
                    max_width = 7 * inch
                    max_height = 600  # Leave some margin
                    
                    # Calculate dimensions that preserve aspect ratio and fit within limits
                    if max_width / aspect_ratio <= max_height:
                        # Width is the limiting factor
                        final_width = max_width
                        final_height = max_width / aspect_ratio
                    else:
                        # Height is the limiting factor
                        final_height = max_height
                        final_width = max_height * aspect_ratio
                    
                    img = Image(str(zonogram_file), width=final_width, height=final_height)
                    story.append(img)
                    story.append(Spacer(1, 10))
                    
                    print(f"      Added regular zonogram: {zone_name}")
                    
                except Exception as e:
                    print(f"      Error adding regular zonogram {zonogram_file}: {e}")
            
            # Find mini zonogram comparison files for this specific tomogram
            # Look for mini zonograms in the mini subdirectory
            mini_azograms_dir = Path("results") / "visualizations" / tomogram_name / "active_zonograms" / "mini"
            mini_zonogram_files = list(mini_azograms_dir.glob(f"{tomogram_name}_mini_zonogram_cluster_*_comparison.png")) if mini_azograms_dir.exists() else []
            
            if mini_zonogram_files:
                # Add mini zonograms section title for this tomogram
                mini_style = ParagraphStyle(
                    'MiniTitle',
                    parent=styles['Heading2'],
                    fontSize=12,
                    spaceAfter=10,
                    textColor=colors.darkred
                )
                story.append(Paragraph("Mini Zonograms (Small Clusters)", mini_style))
                story.append(Spacer(1, 5))
                
                # Add mini zonograms in two columns
                for j in range(0, len(mini_zonogram_files), 2):
                    # Create a table-like layout for two columns
                    from reportlab.platypus import Table, TableStyle
                    
                    row_data = []
                    for k in range(2):
                        if j + k < len(mini_zonogram_files):
                            mini_file = mini_zonogram_files[j + k]
                            try:
                                # Get cluster number from filename
                                cluster_num = mini_file.stem.split('_cluster_')[1].split('_comparison')[0]
                                
                                # Add image (smaller size for two columns, preserve aspect ratio but ensure it fits)
                                # First, get the original image dimensions
                                pil_img = PILImage.open(str(mini_file))
                                orig_width, orig_height = pil_img.size
                                aspect_ratio = orig_width / orig_height
                                
                                # Calculate maximum dimensions that fit in two columns
                                max_width = 3.5 * inch
                                max_height = 300  # Leave some margin for mini zonograms
                                
                                # Calculate dimensions that preserve aspect ratio and fit within limits
                                if max_width / aspect_ratio <= max_height:
                                    # Width is the limiting factor
                                    final_width = max_width
                                    final_height = max_width / aspect_ratio
                                else:
                                    # Height is the limiting factor
                                    final_height = max_height
                                    final_width = max_height * aspect_ratio
                                
                                img = Image(str(mini_file), width=final_width, height=final_height)
                                row_data.append([img])
                                # Added mini zonogram
                                
                            except Exception as e:
                                print(f"      Error adding mini zonogram {mini_file}: {e}")
                                row_data.append([""])
                        else:
                            row_data.append([""])
                    
                    if any(cell != [""] for cell in row_data):
                        # Create table for this row
                        table = Table(row_data, colWidths=[3.5*inch, 3.5*inch])
                        table.setStyle(TableStyle([
                            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                            ('LEFTPADDING', (0, 0), (-1, -1), 5),
                            ('RIGHTPADDING', (0, 0), (-1, -1), 5),
                        ]))
                        story.append(table)
                        story.append(Spacer(1, 10))
            
            # Add page break between tomograms
            if i < len(tomogram_data):
                story.append(PageBreak())
        
        # Build PDF
        # Building all zonograms PDF
        doc.build(story)
        print(f"    ✓ All zonograms PDF saved to: {pdf_path}")
        
    except Exception as e:
        print(f"    Error generating all zonograms PDF: {e}")
        import traceback
        traceback.print_exc()

def generate_mini_zonograms_pdf(tomo_paths, data_dir=None):
    """Generate a PDF showing only the mini zonogram comparison images from processed tomograms - EXACT SAME CODE as original GUI."""
    try:
        from pathlib import Path
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.platypus import SimpleDocTemplate, Image, PageBreak, Spacer, Paragraph
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.lib import colors
        from PIL import Image as PILImage
        
        # Use the processed tomograms list instead of reading CSV
        if not tomo_paths:
            print("    Warning: No processed tomograms found, skipping PDF generation")
            return
        
        # Extract tomogram names and sets from processed tomograms
        tomogram_data = []
        for tomo_info in tomo_paths:
            # Handle both old format (just path) and new format (path, set_name, active_zones)
            if isinstance(tomo_info, tuple) and len(tomo_info) == 3:
                if len(tomo_info) >= 4:
                    tomo_path, set_name, aunp_active_zones, _alignment_dir = tomo_info
                else:
                    tomo_path, set_name, aunp_active_zones = tomo_info
            else:
                tomo_path = tomo_info
                set_name = None
                aunp_active_zones = None
            
            tomo_path = Path(tomo_path)
            tomogram_name = tomo_path.name
            # Extract set from path (e.g., data/15F1/TOP_TOMOS/tomogram_name -> 15F1)
            tomogram_set = tomo_path.parent.parent.name
            tomogram_data.append((tomogram_name, tomogram_set))
        
        # Processing tomograms for mini zonogram PDF generation
        
        # Create output directory for summary PDFs
        output_dir = Path("results/visualizations/pdf_summaries")
        output_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = output_dir / "mini_zonograms_summary.pdf"
        pdf_path_4aunps = output_dir / "mini_zonograms_4aunps_summary.pdf"
        
        # Generating mini zonogram PDFs
        
        # Create PDF documents
        doc = SimpleDocTemplate(str(pdf_path), pagesize=A4)
        doc_4aunps = SimpleDocTemplate(str(pdf_path_4aunps), pagesize=A4)
        
        story = []
        story_4aunps = []
        styles = getSampleStyleSheet()
        
        # Create custom style for tomogram names
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=16,
            spaceAfter=20,
            textColor=colors.darkblue
        )
        
        # Process each tomogram
        for i, (tomogram_name, tomogram_set) in enumerate(tomogram_data, 1):
            # Get the active zone indices for this tomogram
            selected_az_indices = None
            for tomo_info in tomo_paths:
                if isinstance(tomo_info, tuple) and len(tomo_info) == 3:
                    if len(tomo_info) >= 4:
                        tomo_path, set_name, aunp_active_zones, _alignment_dir = tomo_info
                    else:
                        tomo_path, set_name, aunp_active_zones = tomo_info
                    if Path(tomo_path).name == tomogram_name:
                        if aunp_active_zones is not None:
                            az_str = str(aunp_active_zones)
                            if az_str.strip() != "" and az_str.lower() != "nan":
                                try:
                                    selected_az_indices = [int(x) for x in az_str.split(",") if x.strip().isdigit()]
                                    # CSV specified active zones
                                except Exception as e:
                                    print(f"    Warning: Error parsing active zone indices for {tomogram_name}: {e}")
                        break
                else:
                    if Path(tomo_info).name == tomogram_name:
                        break
            
            print(f"    [{i}/{len(tomogram_data)}] Processing {tomogram_name}...", end=" ", flush=True)
            
            # Build tomogram path
            if data_dir:
                tomogram_path = Path(data_dir) / tomogram_set / "TOP_TOMOS" / tomogram_name
            else:
                tomogram_path = Path("data") / tomogram_set / "TOP_TOMOS" / tomogram_name
            
            # Look in the organized structure
            azograms_dir_organized = Path("results") / "visualizations" / tomogram_name / "active_zonograms" / "mini"
            
            if not azograms_dir_organized.exists():
                print(f"    Warning: Active zonograms directory not found: {azograms_dir_organized}")
                continue
            
            mini_zonogram_files = list(azograms_dir_organized.glob("*_mini_zonogram_cluster_*_comparison.png"))
            
            if not mini_zonogram_files:
                print(f"    Warning: No mini zonogram files found for {tomogram_name}")
                continue
            
            # Filter mini zonogram files by active zone indices if specified in CSV
            if selected_az_indices is not None and mini_zonogram_files:
                filtered_mini_files = []
                for mini_file in mini_zonogram_files:
                    # Extract active zone info from filename (e.g., "tomogram_mini_zonogram_cluster_1_comparison.png")
                    # We need to check if this mini zonogram belongs to a filtered active zone
                    # For now, we'll include all mini zonograms since they're cluster-specific, not active zone specific
                    # But we could add filtering logic here if needed
                    filtered_mini_files.append(mini_file)
                mini_zonogram_files = filtered_mini_files
                # Filtered to mini zonogram files
            
            # Get cluster data to identify clusters with 4 AuNPs
            cluster_data_path = tomogram_path / "best_alignment" / "STT_results" / "aunps" / "aunp_clusters.star"
            clusters_with_4_aunps = set()
            
            if cluster_data_path.exists():
                try:
                    import starfile
                    cluster_df = starfile.read(cluster_data_path)
                    # Count AuNPs per cluster
                    cluster_counts = cluster_df['aunp_cluster'].value_counts()
                    # Get clusters with exactly 4 AuNPs
                    clusters_with_4_aunps = set(cluster_counts[cluster_counts == 4].index)
                    # Found clusters with 4 AuNPs
                except Exception as e:
                    print(f"      Warning: Could not read cluster data: {e}")
            
            if mini_zonogram_files:
                # Add tomogram name as title
                story.append(Paragraph(f"Tomogram: {tomogram_name}", title_style))
                story.append(Spacer(1, 10))
                
                # Also add to 4 AuNP PDF if this tomogram has any 4 AuNP clusters
                has_4aunp_clusters = any(
                    int(f.stem.split('_cluster_')[1].split('_comparison')[0].split('_az')[0]) in clusters_with_4_aunps
                    for f in mini_zonogram_files
                )
                if has_4aunp_clusters:
                    story_4aunps.append(Paragraph(f"Tomogram: {tomogram_name}", title_style))
                    story_4aunps.append(Spacer(1, 10))
                
                # Add mini zonograms in two columns
                for j in range(0, len(mini_zonogram_files), 2):
                    # Create a table-like layout for two columns
                    from reportlab.platypus import Table, TableStyle
                    
                    row_data = []
                    for k in range(2):
                        if j + k < len(mini_zonogram_files):
                            mini_file = mini_zonogram_files[j + k]
                            try:
                                # Get cluster number from filename
                                cluster_num = mini_file.stem.split('_cluster_')[1].split('_comparison')[0]
                                
                                # Add image (preserve aspect ratio but ensure it fits)
                                # First, get the original image dimensions
                                pil_img = PILImage.open(str(mini_file))
                                orig_width, orig_height = pil_img.size
                                aspect_ratio = orig_width / orig_height
                                
                                # Calculate maximum dimensions that fit in two columns
                                max_width = 3.5 * inch
                                max_height = 300  # Leave some margin for mini zonograms
                                
                                # Calculate dimensions that preserve aspect ratio and fit within limits
                                if max_width / aspect_ratio <= max_height:
                                    # Width is the limiting factor
                                    final_width = max_width
                                    final_height = max_width / aspect_ratio
                                else:
                                    # Height is the limiting factor
                                    final_height = max_height
                                    final_width = max_height * aspect_ratio
                                
                                img = Image(str(mini_file), width=final_width, height=final_height)
                                row_data.append([img])
                                # Added mini zonogram
                                
                            except Exception as e:
                                print(f"      Error adding mini zonogram {mini_file}: {e}")
                                row_data.append([""])
                        else:
                            row_data.append([""])
                    
                    if any(cell != [""] for cell in row_data):
                        # Create table for this row
                        table = Table(row_data, colWidths=[3.5*inch, 3.5*inch])
                        table.setStyle(TableStyle([
                            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                            ('LEFTPADDING', (0, 0), (-1, -1), 5),
                            ('RIGHTPADDING', (0, 0), (-1, -1), 5),
                        ]))
                        story.append(table)
                        story.append(Spacer(1, 10))
                
                # Add 4 AuNP clusters to the separate PDF using same two-column layout
                clusters_4aunps = [f for f in mini_zonogram_files 
                                 if int(f.stem.split('_cluster_')[1].split('_comparison')[0].split('_az')[0]) in clusters_with_4_aunps]
                
                if clusters_4aunps:
                    for j in range(0, len(clusters_4aunps), 2):
                        # Create a table-like layout for two columns (same as main PDF)
                        from reportlab.platypus import Table, TableStyle
                        
                        row_data = []
                        for k in range(2):
                            if j + k < len(clusters_4aunps):
                                mini_file = clusters_4aunps[j + k]
                                try:
                                    # Get cluster number from filename
                                    cluster_num = mini_file.stem.split('_cluster_')[1].split('_comparison')[0]
                                    
                                    # Add image (preserve aspect ratio but ensure it fits) - same as main PDF
                                    pil_img = PILImage.open(str(mini_file))
                                    orig_width, orig_height = pil_img.size
                                    aspect_ratio = orig_width / orig_height
                                    
                                    # Calculate maximum dimensions that fit in two columns (same as main PDF)
                                    max_width = 3.5 * inch
                                    max_height = 300  # Leave some margin for mini zonograms
                                    
                                    # Calculate dimensions that preserve aspect ratio and fit within limits
                                    if max_width / aspect_ratio <= max_height:
                                        # Width is the limiting factor
                                        final_width = max_width
                                        final_height = max_width / aspect_ratio
                                    else:
                                        # Height is the limiting factor
                                        final_height = max_height
                                        final_width = max_height * aspect_ratio
                                    
                                    img = Image(str(mini_file), width=final_width, height=final_height)
                                    row_data.append([img])
                                    print(f"      Added to 4 AuNP PDF: Cluster {cluster_num}")
                                    
                                except Exception as e:
                                    print(f"      Error adding 4 AuNP mini zonogram {mini_file}: {e}")
                                    row_data.append([""])
                            else:
                                row_data.append([""])
                        
                        if any(cell != [""] for cell in row_data):
                            # Create table for this row (same as main PDF)
                            table = Table(row_data, colWidths=[3.5*inch, 3.5*inch])
                            table.setStyle(TableStyle([
                                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                                ('LEFTPADDING', (0, 0), (-1, -1), 5),
                                ('RIGHTPADDING', (0, 0), (-1, -1), 5),
                            ]))
                            story_4aunps.append(table)
                            story_4aunps.append(Spacer(1, 10))
            else:
                print(f"      No mini zonograms found for {tomogram_name}")
        
        # Build PDFs
        # Building mini zonogram PDFs
        doc.build(story)
        
        # Build 4 AuNP PDF if there are any clusters with 4 AuNPs
        if story_4aunps:
            doc_4aunps.build(story_4aunps)
            print(f"    4 AuNP mini zonogram PDF generation completed successfully!")
            print(f"    4 AuNP PDF saved to: {pdf_path_4aunps}")
        else:
            # No clusters with 4 AuNPs found, skipping 4 AuNP PDF
            pass
        
        print("✅")
        print(f"    ✓ Mini zonogram PDF saved to: {pdf_path}")
        
    except Exception as e:
        print(f"    Error generating mini zonograms PDF: {e}")
        import traceback
        traceback.print_exc()

def generate_default_visualization_pdf_summary(tomogram_paths, original_csv_path=None, root_dir=None):
    """Generate the default visualization PDF summary using the original CSV file."""
    try:
        import subprocess
        import sys
        from pathlib import Path
        
        # Get the script path
        script_path = Path(__file__).parent.parent.parent / "scripts" / "generate_tomogram_summary_pdf.py"
        
        if not script_path.exists():
            print("    Warning: generate_tomogram_summary_pdf.py not found, skipping PDF generation")
            return
        
        # Use the original CSV if provided, otherwise skip PDF generation
        if not original_csv_path or not Path(original_csv_path).exists():
            print("    Warning: No original CSV path provided or file doesn't exist, skipping PDF generation")
            return
        
        # Using original CSV file
        
        # Run the PDF generation script with the original CSV
        cmd = [sys.executable, str(script_path)]
        # Note: The PDF generation script may need to be updated to use the new organized structure
        # For now, point to the base visualizations directory - the script will need to handle the new structure
        cmd += ["--vis-dir", "results/visualizations"]
        # Use the provided root directory or default to "data"
        data_dir = root_dir if root_dir else "data"
        cmd += ["--data-dir", data_dir]
        cmd += ["--output-dir", "results/visualizations/pdf_summaries"]
        cmd += ["--tomocsv", str(original_csv_path)]
        
        print(f"    Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("    ✓ PDF summary generated successfully")
            if result.stdout:
                print(f"    Output: {result.stdout.strip()}")
        else:
            print(f"    Warning: PDF generation failed with return code {result.returncode}")
            if result.stderr:
                print(f"    Error: {result.stderr.strip()}")
    
    except Exception as e:
        print(f"    Error generating default visualization PDF summary: {e}")
        import traceback
        traceback.print_exc()

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

    
    # Run active zonogram analysis for all tomograms
    print("\n" + "="*60)
    print("RUNNING ACTIVE ZONOGRAM ANALYSIS")
    print("="*60)
    try:
        run_zonogram_analysis_for_all_tomograms(tomogram_paths, output_dir)
        print("Active zonogram analysis completed successfully!")
    except Exception as e:
        print(f"Error in active zonogram analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 