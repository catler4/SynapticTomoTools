#!/usr/bin/env python3
"""
Synaptic Tomogram Results Visualization Script

This script generates visualizations for analyzed synaptic tomograms, including overlays
of membranes, vesicles, synaptic clefts, and AuNPs. It processes all available tomograms
and saves the figures as output files.
"""

import os
import sys
from typing import Optional

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
from .cleft import extract_active_zonogram
from .alignment_utils import require_alignment_dir

# Tomogram slice overlays: vesicle distance-class edge colors (nm semantics in legends elsewhere)
VESICLE_VIZ_EDGE_FUSING = "aqua"
VESICLE_VIZ_EDGE_CLOSE = "#C8A8E0"  # light purple / lilac
VESICLE_VIZ_EDGE_FAR = "pink"


def unpack_tomo_csv_row(tomo_info):
    """
    Normalize CLI/GUI tomogram entries to
    (tomo_path, set_name, cleft_ids, alignment_dir).

    ``alignment_dir`` must come from the tomogram CSV ``alignment_dir`` column (fourth tuple element).
    """
    if isinstance(tomo_info, tuple) and len(tomo_info) >= 4:
        adir = require_alignment_dir(
            tomo_info[3],
            context="tuple from load_tomograms / CSV row (alignment_dir column)",
        )
        return tomo_info[0], tomo_info[1], tomo_info[2], adir
    raise ValueError(
        "Each tomogram entry must be a tuple "
        "(tomo_path, set_name, cleft_ids, alignment_dir) "
        "with alignment_dir read from the CSV. Got: "
        f"{type(tomo_info).__name__} with length "
        f"{len(tomo_info) if isinstance(tomo_info, tuple) else 'n/a'}."
    )


def organized_results_viz_path(results_root, tomogram_name: str, alignment_dir: str, *subpaths: str) -> Path:
    """results/visualizations/{tomogram}/{alignment}/… — avoids collisions when one tomogram uses multiple alignments."""
    alignment_dir = require_alignment_dir(alignment_dir)
    p = Path(results_root) / "visualizations" / tomogram_name / alignment_dir
    for s in subpaths:
        p = p / s
    return p


# Try to import mrcfile, but handle gracefully if not available
try:
    import mrcfile
except ImportError:
    print("Warning: mrcfile not available. Tomogram slice loading will be disabled.")
    mrcfile = None

# Helper to find analyzed tomograms
def find_analyzed_tomograms(base_dir="../data/"):
    """Find tomogram directories that contain vesicle results under any alignment subdirectory."""
    tomos = []
    base = Path(base_dir)
    if not base.exists():
        return []
    for vesicle_json in base.rglob("STT_results/vesicles/vesicle_results.json"):
        # .../<tomogram>/<alignment>/STT_results/vesicles/vesicle_results.json
        tomo_path = vesicle_json.parents[3]
        s = str(tomo_path)
        if s not in tomos:
            tomos.append(s)
    return sorted(tomos)

def load_tomogram_slice(tomo_path, z_center=None, *, alignment_dir: str):
    """Load a 2D slice from the tomogram."""
    alignment_dir = require_alignment_dir(alignment_dir)
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

def load_membrane_coords(tomo_path, kind='presynaptic', *, alignment_dir: str):
    """Load membrane coordinates from text files."""
    alignment_dir = require_alignment_dir(alignment_dir)
    aunps_dir = Path(tomo_path) / alignment_dir / 'aunps'
    files = sorted(aunps_dir.glob(f'{kind}membranes_*.txt'))
    coords = [np.loadtxt(f) for f in files if f.exists()]
    return coords

def load_cleft_coords(tomo_path, *, alignment_dir: str):
    """Load synaptic cleft coordinates."""
    alignment_dir = require_alignment_dir(alignment_dir)
    az_dir = Path(tomo_path) / alignment_dir / 'STT_results' / 'cleft'
    files = sorted(az_dir.glob('cleft_pre*_post*_pre_outer.txt'))
    coords = [np.loadtxt(f) for f in files if f.exists()]
    return coords

def load_vesicles(tomo_path, *, alignment_dir: str):
    """Load vesicle data from JSON file."""
    alignment_dir = require_alignment_dir(alignment_dir)
    ves_file = Path(tomo_path) / alignment_dir / 'STT_results' / 'vesicles' / 'vesicle_results.json'
    with open(ves_file) as f:
        data = json.load(f)
    return data['vesicles']

def load_aunps(tomo_path, cleft_indices=None, *, alignment_dir: str):
    """Load AuNP coordinates from ``STT_results/aunps/aunp_clusters.star``.

    Raises ``FileNotFoundError`` if that file is missing (run AuNP analysis first).
    """
    alignment_dir = require_alignment_dir(alignment_dir)
    aunps_results_dir = Path(tomo_path) / alignment_dir / 'STT_results' / 'aunps'
    import starfile
    import pandas as pd
    
    # Load from the filtered output file (aunp_clusters.star)
    cluster_star = aunps_results_dir / "aunp_clusters.star"
    
    if not cluster_star.exists():
        raise FileNotFoundError(
            f"Required AuNP cluster file not found: {cluster_star}. "
            "Run AuNP analysis (analyze_aunps) to produce aunp_clusters.star."
        )
    
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
    
    # Filter by cleft_indices if specified
    if cleft_indices is not None:
        if 'cleft' not in df.columns:
            print("[viz] Warning: 'cleft' column not found in filtered AuNP file")
            return None
        
        # Convert cleft column to int if needed, and handle any NaN values
        df['cleft'] = pd.to_numeric(df['cleft'], errors='coerce').astype('Int64')
        
        # Show what synaptic clefts are actually in the file for debugging
        unique_azs = sorted(df['cleft'].dropna().unique().tolist())
        print(f"[viz] Synaptic clefts in file: {unique_azs}, filtering for: {cleft_indices}")
        
        # Filter by synaptic cleft indices (convert to same type for comparison)
        cleft_indices_int = [int(az) for az in cleft_indices]
        df_filtered = df[df['cleft'].isin(cleft_indices_int)].copy()
        
        if len(df_filtered) == 0:
            # If no AuNPs found, check if all cleft values are 0 (common issue from old data)
            # This can happen if the input files had cleft=0 instead of the file index
            unique_azs = sorted(df['cleft'].dropna().unique().tolist())
            if len(unique_azs) == 1 and unique_azs[0] == 0:
                print(f"[viz] Warning: All AuNPs have cleft=0, but filtering for {cleft_indices}")
                print(f"[viz] This suggests the analysis needs to be re-run with the fixed cleft assignment.")
                print(f"[viz] Returning empty result - please re-run AuNP analysis to fix cleft values.")
            else:
                print(f"[viz] Warning: No AuNPs found in synaptic clefts {cleft_indices}")
                print(f"[viz] Available synaptic clefts in file: {unique_azs}")
        
        df = df_filtered
        print(f"[viz] Filtered to {len(df)} AuNPs in synaptic clefts {cleft_indices}")
    
    return df

def load_fusion_points(tomo_path, *, alignment_dir: str, vesicle_distance_threshold: float = 20.0):
    """Load fusion points for vesicles near the presynaptic synaptic cleft (see ``compute_fusion_points``)."""
    alignment_dir = require_alignment_dir(alignment_dir)
    try:
        from scipy.spatial import KDTree
        from .aunps import compute_fusion_points
        
        fusion_points = compute_fusion_points(
            tomo_path, vesicle_distance_threshold=vesicle_distance_threshold, alignment_dir=alignment_dir
        )
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
    """Filter AuNPs with cleft != -1 and z within z_thresh of z_center."""
    if aunps is None or aunps.empty:
        return None
    if 'cleft' not in aunps.columns:
        return None
    mask = (aunps['cleft'] != -1) & (np.abs(aunps['faCoordinateZ'] - z_center) <= z_thresh)
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

def load_postsynaptic_cleft_coords(tomo_path, *, alignment_dir: str):
    """Load postsynaptic synaptic cleft coordinates."""
    alignment_dir = require_alignment_dir(alignment_dir)
    az_dir = Path(tomo_path) / alignment_dir / 'STT_results' / 'cleft'
    files = sorted(az_dir.glob('cleft_pre*_post*_post_outer.txt'))
    coords = [np.loadtxt(f) for f in files if f.exists()]
    return coords


def _load_optional_az_surface_txt(path: Path) -> np.ndarray:
    """Load Nx3 coordinates from an synaptic-cleft surface txt; return empty (0, 3) if missing or invalid."""
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


def load_specific_cleft_coords(tomo_path, cleft_indices, aunps, *, alignment_dir: str):
    """Load outer and inner synaptic cleft coordinates for the given indices, using saved mapping."""
    from .cleft import load_cleft_mapping
    
    alignment_dir = require_alignment_dir(alignment_dir)
    az_dir = Path(tomo_path) / alignment_dir / "STT_results" / "cleft"
    
    azs_pre = []
    azs_post = []
    azs_pre_inner = []
    azs_post_inner = []
    
    if cleft_indices is not None:
        # Load saved mapping
        az_mapping = load_cleft_mapping(tomo_path, alignment_dir)
        
        if not az_mapping:
            # No mapping found - use all available zone files as fallback but print error
            print(f"No saved synaptic cleft mapping found for {Path(tomo_path).name}. Active zone analysis must be run first with smart matching to create the mapping.")
            print(f"FALLBACK: Loading all available synaptic cleft files (no filtering applied).")
            # Load all available zone files
            pre_files = sorted(list(az_dir.glob('cleft_pre*_post*_pre_outer.txt')))
            post_files = sorted(list(az_dir.glob('cleft_pre*_post*_post_outer.txt')))
            
            # Group files by synaptic cleft name to ensure paired matching
            cleft_groups = {}
            for pre_file in pre_files:
                zone_name = pre_file.name.replace('_pre_outer.txt', '')
                if zone_name not in cleft_groups:
                    cleft_groups[zone_name] = {'pre': None, 'post': None}
                cleft_groups[zone_name]['pre'] = pre_file
            
            for post_file in post_files:
                zone_name = post_file.name.replace('_post_outer.txt', '')
                if zone_name not in cleft_groups:
                    cleft_groups[zone_name] = {'pre': None, 'post': None}
                cleft_groups[zone_name]['post'] = post_file
            
            # Load all paired zones
            for zone_name, files in cleft_groups.items():
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
            for az_id in cleft_indices:
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
                    raise ValueError(f"Active zone index {az_id} not found in saved mapping. This indicates the synaptic cleft analysis was run with different indices.")
    
    return azs_pre, azs_post, azs_pre_inner, azs_post_inner

def plot_tomogram_overlays(tomo_path, output_dir, aunp_cleft_indices=None, rerun=False, *, alignment_dir: str,
                           vesicle_distance_threshold: float = 20.0,
                           fusing_perimeter_threshold: float = 1.0,
                           sphere_size=None, sphere_color=None, aunp_distance_min=None, aunp_distance_max=None,
                           aunp_distance_cutoff_direction=None, aunp_distance_cutoff_value=None):
    """Generate 2D overlay plot and save to file. Only processes CSV-specified synaptic clefts."""
    alignment_dir = require_alignment_dir(alignment_dir)
    vesicles = load_vesicles(tomo_path, alignment_dir=alignment_dir)
    pre_mem = load_membrane_coords(tomo_path, 'presynaptic', alignment_dir=alignment_dir)
    post_mem = load_membrane_coords(tomo_path, 'postsynaptic', alignment_dir=alignment_dir)
    aunps = load_aunps(tomo_path, aunp_cleft_indices, alignment_dir=alignment_dir)
    fusion_points = load_fusion_points(
        tomo_path, alignment_dir=alignment_dir, vesicle_distance_threshold=vesicle_distance_threshold
    )
    
    # Process synaptic clefts - auto-detect if none specified in CSV
    if aunp_cleft_indices is None or len(aunp_cleft_indices) == 0:
        print("No synaptic clefts specified in CSV, auto-detecting all available synaptic clefts")
        # Auto-detect all available synaptic cleft numbers from filtered AuNP file
        aunps_results_dir = Path(tomo_path) / alignment_dir / "STT_results" / "aunps"
        cluster_star = aunps_results_dir / "aunp_clusters.star"
        if not cluster_star.exists():
            raise FileNotFoundError(
                f"Required AuNP cluster file not found: {cluster_star}. "
                "Run AuNP analysis (analyze_aunps) to produce aunp_clusters.star."
            )
        import starfile
        star_data = starfile.read(cluster_star)
        if isinstance(star_data, dict):
            df = None
            for v in star_data.values():
                if isinstance(v, pd.DataFrame):
                    df = v
                    break
        else:
            df = star_data

        if df is not None and 'cleft' in df.columns:
            aunp_az_numbers = sorted(df['cleft'].unique().tolist())
            # Remove -1 if present (means "not in any synaptic cleft")
            aunp_az_numbers = [az for az in aunp_az_numbers if az != -1]
            aunp_cleft_indices = aunp_az_numbers
            print(f"Auto-detected synaptic clefts from filtered AuNP file: {aunp_cleft_indices}")
        else:
            raise ValueError(
                f"Could not read synaptic clefts from {cluster_star}: missing DataFrame or 'cleft' column."
            )
    
    # Load only the synaptic cleft membranes for CSV-specified or auto-detected synaptic clefts, matched by distance to AuNPs
    azs_pre, azs_post, azs_pre_inner, azs_post_inner = load_specific_cleft_coords(
        tomo_path, aunp_cleft_indices, aunps, alignment_dir=alignment_dir
    )
    
    if len(aunp_cleft_indices) == 0:
        print("No synaptic clefts found, using middle of tomogram")
        # Fallback to middle of tomogram if no synaptic clefts found
        slice2d, z_center = load_tomogram_slice(tomo_path, None, alignment_dir=alignment_dir)
        if slice2d is None:
            print(f"Could not load tomogram slice for {tomo_path}")
            return
        _generate_visualizations_for_slice(tomo_path, output_dir, slice2d, z_center, vesicles, 
                                         pre_mem, post_mem, [], [], [], [], aunps, fusion_points, 
                                         aunp_cleft_indices, rerun, alignment_dir,
                                         vesicle_distance_threshold, fusing_perimeter_threshold, "middle")
    else:
        # Generate visualizations for each synaptic cleft (CSV-specified or auto-detected)
        for az_id in aunp_cleft_indices:
            # Processing synaptic cleft
            
            # Calculate z_center based on AuNPs within this specific synaptic cleft
            z_center = _calculate_cleft_center_from_aunps(aunps, az_id)
            if z_center is None:
                print(f"Warning: No AuNPs found for synaptic cleft {az_id}, skipping visualization")
                continue
            
            # Generating visualizations for synaptic cleft
            
            slice2d, zc = load_tomogram_slice(tomo_path, z_center, alignment_dir=alignment_dir)
            if slice2d is None:
                print(f"Could not load tomogram slice for {tomo_path} at z={z_center}")
                continue
            
            _generate_visualizations_for_slice(tomo_path, output_dir, slice2d, z_center, vesicles, 
                                             pre_mem, post_mem, azs_pre, azs_post, azs_pre_inner, azs_post_inner,
                                             aunps, fusion_points, 
                                             aunp_cleft_indices, rerun, alignment_dir,
                                             vesicle_distance_threshold, fusing_perimeter_threshold, f"az{az_id}")

def _calculate_cleft_center_from_aunps(aunps, cleft_id):
    """Calculate the z_center of an synaptic cleft based on the center of AuNPs within that synaptic cleft."""
    if aunps is None or aunps.empty:
        return None
    
    # Filter AuNPs for this specific synaptic cleft
    if 'cleft' in aunps.columns:
        aunps_in_az = aunps[aunps['cleft'] == cleft_id]
    else:
        # If no cleft column, we can't filter by synaptic cleft
        print(f"Warning: No 'cleft' column in AuNP data, cannot calculate center for synaptic cleft {cleft_id}")
        return None
    
    if aunps_in_az.empty:
        print(f"Warning: No AuNPs found in synaptic cleft {cleft_id}")
        return None
    
    # Calculate the mean Z coordinate of AuNPs in this synaptic cleft
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


def _json_bool_truthy(x) -> bool:
    if x is True:
        return True
    if isinstance(x, str) and x.strip().lower() in ("true", "1", "yes"):
        return True
    return False


def _vesicle_az_proximate_for_viz(vesicle: dict, vesicle_distance_threshold_nm: float) -> bool:
    """
    Whether a vesicle counts as AZ-proximal for 2D overlays (fusion-site association, extra ring).

    Prefer ``vesicle_distance_class`` / legacy flags when present; otherwise fall back to
    ``distance_to_az <= vesicle_distance_threshold_nm``.
    """
    cls = vesicle.get("vesicle_distance_class")
    if cls is not None and str(cls).strip():
        s = str(cls).strip().lower()
        if s in ("fusing", "fusion", "close"):
            return True
        if s == "far":
            return False
    if _json_bool_truthy(vesicle.get("is_fusing")) or _json_bool_truthy(vesicle.get("is_close")):
        return True
    d = vesicle.get("distance_to_az", float("nan"))
    try:
        d = float(d)
    except (TypeError, ValueError):
        d = float("nan")
    thresh = float(vesicle_distance_threshold_nm)
    return bool(np.isfinite(d) and d <= thresh)


def _vesicle_viz_proximity_role(vesicle: dict, vesicle_distance_threshold_nm: float) -> Optional[str]:
    """
    For AZ-proximate vesicles in slice overlays, return ``\"fusing\"`` or ``\"close\"`` for coloring;
    ``None`` if not AZ-proximate (draw as far / pink).
    """
    if not _vesicle_az_proximate_for_viz(vesicle, vesicle_distance_threshold_nm):
        return None
    cls = vesicle.get("vesicle_distance_class")
    if cls is not None and str(cls).strip():
        s = str(cls).strip().lower()
        if s in ("fusing", "fusion"):
            return "fusing"
        if s == "close":
            return "close"
    if _json_bool_truthy(vesicle.get("is_fusing")):
        return "fusing"
    if _json_bool_truthy(vesicle.get("is_close")):
        return "close"
    return "close"


def _vesicle_fusion_ring_edgecolor(vesicle: dict, vesicle_distance_threshold_nm: float) -> str:
    """Edge color for the 40 nm fusion-site ring on active zonograms (fusing vs close)."""
    if _vesicle_viz_proximity_role(vesicle, vesicle_distance_threshold_nm) == "fusing":
        return VESICLE_VIZ_EDGE_FUSING
    return VESICLE_VIZ_EDGE_CLOSE


def _vesicle_distance_class_edge_style(
    v, vesicle_distance_threshold_nm: Optional[float] = None
) -> dict:
    """
    Matplotlib Circle kwargs from vesicle_results.json ``vesicle_distance_class``:
    fusing (aqua), close (light purple), far (pink), unknown (gray).
    Fallback: ``is_fusing`` / ``is_close``; then non-finite distance -> unknown; else if
    ``vesicle_distance_threshold_nm`` is set and ``distance_to_az`` is within threshold,
    treat as close (light purple); otherwise far (pink).
    """
    cls = v.get("vesicle_distance_class")
    if cls is not None and str(cls).strip():
        s = str(cls).strip().lower()
        if s in ("fusing", "fusion"):
            return {"edgecolor": VESICLE_VIZ_EDGE_FUSING, "linewidth": 2.5, "alpha": 0.95}
        if s == "close":
            return {"edgecolor": VESICLE_VIZ_EDGE_CLOSE, "linewidth": 2.0, "alpha": 0.85}
        if s == "far":
            return {"edgecolor": VESICLE_VIZ_EDGE_FAR, "linewidth": 1.5, "alpha": 0.7}
        if s == "unknown":
            return {"edgecolor": "gray", "linewidth": 1.5, "alpha": 0.65}
        return {"edgecolor": "gray", "linewidth": 1.5, "alpha": 0.65}
    if _json_bool_truthy(v.get("is_fusing")):
        return {"edgecolor": VESICLE_VIZ_EDGE_FUSING, "linewidth": 2.5, "alpha": 0.95}
    if _json_bool_truthy(v.get("is_close")):
        return {"edgecolor": VESICLE_VIZ_EDGE_CLOSE, "linewidth": 2.0, "alpha": 0.85}
    d = v.get("distance_to_az", float("nan"))
    try:
        d = float(d)
    except (TypeError, ValueError):
        return {"edgecolor": "gray", "linewidth": 1.5, "alpha": 0.65}
    if not np.isfinite(d):
        return {"edgecolor": "gray", "linewidth": 1.5, "alpha": 0.65}
    if vesicle_distance_threshold_nm is not None and d <= float(vesicle_distance_threshold_nm):
        return {"edgecolor": VESICLE_VIZ_EDGE_CLOSE, "linewidth": 2.0, "alpha": 0.85}
    return {"edgecolor": VESICLE_VIZ_EDGE_FAR, "linewidth": 1.5, "alpha": 0.7}


def _generate_visualizations_for_slice(tomo_path, output_dir, slice2d, z_center, vesicles, 
                                     pre_mem, post_mem, azs_pre, azs_post, azs_pre_inner, azs_post_inner,
                                     aunps, fusion_points, 
                                     aunp_cleft_indices, rerun, alignment_dir: str,
                                     vesicle_distance_threshold: float = 20.0,
                                     fusing_perimeter_threshold: float = 1.0,
                                     suffix: str = ""):
    """Generate all visualization types for a specific slice."""
    alignment_dir = require_alignment_dir(alignment_dir)
    tomo_name = Path(tomo_path).name
    
    # Contrast adjustment: use 2nd and 98th percentiles for vmin/vmax
    vmin, vmax = np.percentile(slice2d, [2, 98])
    
    # Filter objects for the slice
    z_thresh = 5  # Increased from 2 to 5 pixels
    z_thresh_az = 1  # Stricter threshold for synaptic clefts
    z_thresh_aunps_fusion = 10  # 10 nm threshold for AuNPs and fusion sites
    z_thresh_vesicles = 1  # 1 pixel threshold - vesicle must intersect with slice
    vesicles_in_slice = filter_vesicles_in_slice(vesicles, z_center, z_thresh_vesicles)
    azs_pre_in_slice = filter_coords_in_slice(azs_pre, z_center, z_thresh_az, None)
    azs_post_in_slice = filter_coords_in_slice(azs_post, z_center, z_thresh_az, None)
    azs_pre_inner_in_slice = filter_coords_in_slice(azs_pre_inner, z_center, z_thresh_az, None)
    azs_post_inner_in_slice = filter_coords_in_slice(azs_post_inner, z_center, z_thresh_az, None)
    aunps_near = filter_aunps_near_slice(aunps, z_center, z_thresh_aunps_fusion)
    vdist_nm = float(vesicle_distance_threshold)
    vfuse_nm = float(fusing_perimeter_threshold)
    lbl_legend_fusing = f"Vesicles fusing (< {vfuse_nm:g} nm)"
    lbl_legend_close = f"Vesicles close (< {vdist_nm:g} nm)"
    lbl_legend_far = f"Vesicles far (> {vdist_nm:g} nm)"
    lbl_v_fusing = lbl_legend_fusing
    lbl_v_close = lbl_legend_close
    lbl_v_far = lbl_legend_far

    # Inner AZ colors: lighter red / green than pure outer; drawn under outer scatters
    inner_pre_rgb = (1.0, 0.52, 0.52)
    inner_post_rgb = (0.52, 1.0, 0.52)
    inner_az_alpha = 0.06
    
    # Debug output (simplified)
    
    # Version 1: Vesicles and Clefts
    output_file1 = output_dir / f"{tomo_name}_vesicles_clefts_{suffix}.png"
    if output_file1.exists() and not rerun:
        print(f"Skipping {output_file1}, already exists.")
    else:
        fig1, ax1 = plt.subplots(figsize=(12, 12))
        ax1.imshow(slice2d, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
        
        # Overlay vesicles intersecting slice, colored by vesicle_distance_class (see vesicle_results.json)
        draw_order = ("far", "unknown", "close", "fusing")

        def _class_bucket(v):
            cls = v.get("vesicle_distance_class")
            if cls is not None and str(cls).strip():
                s = str(cls).strip().lower()
                if s in ("fusing", "fusion"):
                    return "fusing"
                if s == "close":
                    return "close"
                if s == "far":
                    return "far"
                if s == "unknown":
                    return "unknown"
                return "unknown"
            if _json_bool_truthy(v.get("is_fusing")):
                return "fusing"
            if _json_bool_truthy(v.get("is_close")):
                return "close"
            d = v.get("distance_to_az", float("nan"))
            if not np.isfinite(d):
                return "unknown"
            try:
                d = float(d)
            except (TypeError, ValueError):
                return "unknown"
            if d <= vdist_nm:
                return "close"
            return "far"

        for bucket in draw_order:
            for v in vesicles_in_slice:
                if _class_bucket(v) != bucket:
                    continue
                c = np.array(v["center"])
                r = v["radius"]
                st = _vesicle_distance_class_edge_style(v, vdist_nm)
                circ = Circle(
                    (c[0], c[1]),
                    r,
                    facecolor="none",
                    edgecolor=st["edgecolor"],
                    linewidth=st["linewidth"],
                    alpha=st["alpha"],
                )
                ax1.add_patch(circ)
        
        # Inner synaptic clefts (faded; underneath outer)
        for coords in azs_pre_inner_in_slice:
            ax1.scatter(coords[:, 0], coords[:, 1], color=inner_pre_rgb, s=3, alpha=inner_az_alpha,
                        label='Presynaptic AZ (inner)' if 'Presynaptic AZ (inner)' not in [l.get_label() for l in ax1.get_legend_handles_labels()[0]] else '')
        for coords in azs_post_inner_in_slice:
            ax1.scatter(coords[:, 0], coords[:, 1], color=inner_post_rgb, s=3, alpha=inner_az_alpha,
                        label='Postsynaptic AZ (inner)' if 'Postsynaptic AZ (inner)' not in [l.get_label() for l in ax1.get_legend_handles_labels()[0]] else '')
        
        # Overlay presynaptic synaptic cleft (outer)
        for coords in azs_pre_in_slice:
            ax1.scatter(coords[:,0], coords[:,1], color='red', s=3, alpha=0.1, 
                    label='Presynaptic AZ (outer)' if 'Presynaptic AZ (outer)' not in [l.get_label() for l in ax1.get_legend_handles_labels()[0]] else '')
        
        # Overlay postsynaptic synaptic cleft (outer)
        for coords in azs_post_in_slice:
            ax1.scatter(coords[:,0], coords[:,1], color='green', s=3, alpha=0.1, 
                    label='Postsynaptic AZ (outer)' if 'Postsynaptic AZ (outer)' not in [l.get_label() for l in ax1.get_legend_handles_labels()[0]] else '')
        
        legend_elements = [
            Line2D([0], [0], color=VESICLE_VIZ_EDGE_FUSING, lw=2.5, label=lbl_legend_fusing),
            Line2D([0], [0], color=VESICLE_VIZ_EDGE_CLOSE, lw=2, label=lbl_legend_close),
            Line2D([0], [0], color=VESICLE_VIZ_EDGE_FAR, lw=1.5, label=lbl_legend_far),
            Line2D([0], [0], color="gray", lw=1.5, label="Vesicles unknown"),
            Line2D([0], [0], color=inner_pre_rgb, lw=1.5, label='Presynaptic AZ (inner)'),
            Line2D([0], [0], color=inner_post_rgb, lw=1.5, label='Postsynaptic AZ (inner)'),
            Line2D([0], [0], color='red', lw=1.5, label='Presynaptic AZ (outer)'),
            Line2D([0], [0], color='green', lw=1.5, label='Postsynaptic AZ (outer)'),
        ]
        ax1.legend(handles=legend_elements)
        ax1.set_title(f'Vesicles and Clefts - {tomo_name}')
        ax1.set_xlabel('X (pixels)')
        ax1.set_ylabel('Y (pixels)')
        
        plt.savefig(output_file1, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved vesicles and synaptic clefts: {output_file1.name}")

    # Version 1b: All vesicles (XY projection on same slice), same class coloring as Version 1
    output_file1_all = output_dir / f"{tomo_name}_vesicles_clefts_all_{suffix}.png"
    vesicles_all_xy = list(vesicles) if vesicles is not None else []
    if output_file1_all.exists() and not rerun:
        print(f"Skipping {output_file1_all}, already exists.")
    else:
        fig1b, ax1b = plt.subplots(figsize=(12, 12))
        ax1b.imshow(slice2d, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')

        draw_order_all = ("far", "unknown", "close", "fusing")

        def _class_bucket_all(v):
            cls = v.get("vesicle_distance_class")
            if cls is not None and str(cls).strip():
                s = str(cls).strip().lower()
                if s in ("fusing", "fusion"):
                    return "fusing"
                if s == "close":
                    return "close"
                if s == "far":
                    return "far"
                if s == "unknown":
                    return "unknown"
                return "unknown"
            if _json_bool_truthy(v.get("is_fusing")):
                return "fusing"
            if _json_bool_truthy(v.get("is_close")):
                return "close"
            d = v.get("distance_to_az", float("nan"))
            if not np.isfinite(d):
                return "unknown"
            try:
                d = float(d)
            except (TypeError, ValueError):
                return "unknown"
            if d <= vdist_nm:
                return "close"
            return "far"

        for bucket in draw_order_all:
            for v in vesicles_all_xy:
                if _class_bucket_all(v) != bucket:
                    continue
                c = np.array(v["center"])
                r = v["radius"]
                st = _vesicle_distance_class_edge_style(v, vdist_nm)
                circ = Circle(
                    (c[0], c[1]),
                    r,
                    facecolor="none",
                    edgecolor=st["edgecolor"],
                    linewidth=st["linewidth"],
                    alpha=st["alpha"],
                )
                ax1b.add_patch(circ)

        for coords in azs_pre_inner_in_slice:
            ax1b.scatter(
                coords[:, 0],
                coords[:, 1],
                color=inner_pre_rgb,
                s=3,
                alpha=inner_az_alpha,
                label="Presynaptic AZ (inner)"
                if "Presynaptic AZ (inner)"
                not in [l.get_label() for l in ax1b.get_legend_handles_labels()[0]]
                else "",
            )
        for coords in azs_post_inner_in_slice:
            ax1b.scatter(
                coords[:, 0],
                coords[:, 1],
                color=inner_post_rgb,
                s=3,
                alpha=inner_az_alpha,
                label="Postsynaptic AZ (inner)"
                if "Postsynaptic AZ (inner)"
                not in [l.get_label() for l in ax1b.get_legend_handles_labels()[0]]
                else "",
            )
        for coords in azs_pre_in_slice:
            ax1b.scatter(
                coords[:, 0],
                coords[:, 1],
                color="red",
                s=3,
                alpha=0.1,
                label="Presynaptic AZ (outer)"
                if "Presynaptic AZ (outer)"
                not in [l.get_label() for l in ax1b.get_legend_handles_labels()[0]]
                else "",
            )
        for coords in azs_post_in_slice:
            ax1b.scatter(
                coords[:, 0],
                coords[:, 1],
                color="green",
                s=3,
                alpha=0.1,
                label="Postsynaptic AZ (outer)"
                if "Postsynaptic AZ (outer)"
                not in [l.get_label() for l in ax1b.get_legend_handles_labels()[0]]
                else "",
            )

        legend_all = [
            Line2D([0], [0], color=VESICLE_VIZ_EDGE_FUSING, lw=2.5, label=lbl_legend_fusing),
            Line2D([0], [0], color=VESICLE_VIZ_EDGE_CLOSE, lw=2, label=lbl_legend_close),
            Line2D([0], [0], color=VESICLE_VIZ_EDGE_FAR, lw=1.5, label=lbl_legend_far),
            Line2D([0], [0], color="gray", lw=1.5, label="Vesicles unknown"),
            Line2D([0], [0], color=inner_pre_rgb, lw=1.5, label="Presynaptic AZ (inner)"),
            Line2D([0], [0], color=inner_post_rgb, lw=1.5, label="Postsynaptic AZ (inner)"),
            Line2D([0], [0], color="red", lw=1.5, label="Presynaptic AZ (outer)"),
            Line2D([0], [0], color="green", lw=1.5, label="Postsynaptic AZ (outer)"),
        ]
        ax1b.legend(handles=legend_all)
        ax1b.set_title(
            f"All vesicles (XY projection) and synaptic clefts — {tomo_name} (slice z≈{z_center})"
        )
        ax1b.set_xlabel("X (pixels)")
        ax1b.set_ylabel("Y (pixels)")

        plt.savefig(output_file1_all, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  ✓ Saved all-vesicle XY overlay: {output_file1_all.name}")
    
    # Version 2: Vesicles and AuNPs
    output_file2 = output_dir / f"{tomo_name}_vesicles_aunps_{suffix}.png"
    if output_file2.exists() and not rerun:
        print(f"Skipping {output_file2}, already exists.")
    else:
        fig2, ax2 = plt.subplots(figsize=(12, 12))
        ax2.imshow(slice2d, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
        
        # Vesicles in slice: far / not AZ-proximate (pink), fusing (aqua), close (light purple)
        for v in vesicles_in_slice:
            c = np.array(v["center"])
            r = v["radius"]
            role = _vesicle_viz_proximity_role(v, vdist_nm)
            handles = ax2.get_legend_handles_labels()[0]
            existing = [l.get_label() for l in handles]
            if role == "fusing":
                ec, lw, alpha = VESICLE_VIZ_EDGE_FUSING, 2.5, 0.88
                lab = lbl_v_fusing if lbl_v_fusing not in existing else ""
            elif role == "close":
                ec, lw, alpha = VESICLE_VIZ_EDGE_CLOSE, 2.0, 0.88
                lab = lbl_v_close if lbl_v_close not in existing else ""
            else:
                ec, lw, alpha = VESICLE_VIZ_EDGE_FAR, 1.5, 0.7
                lab = lbl_v_far if lbl_v_far not in existing else ""
            circ = Circle(
                (c[0], c[1]),
                r,
                edgecolor=ec,
                facecolor="none",
                fill=False,
                lw=lw,
                alpha=alpha,
                label=lab,
            )
            ax2.add_patch(circ)
        
        # Add AuNPs with transparency
        if aunps_near is not None:
            ax2.scatter(aunps_near['faCoordinateX'], aunps_near['faCoordinateY'], 
                      color='gold', s=30, alpha=0.8, label='AuNPs')
        
        # Show fusion points for membrane-adjacent vesicles that are being displayed
        if fusion_points is not None and len(fusion_points) > 0 and len(vesicles_in_slice) > 0:
            # AZ-proximate vesicles in this slice (same gating as aqua highlight)
            membrane_adjacent_vesicles_in_slice = [
                v for v in vesicles_in_slice if _vesicle_az_proximate_for_viz(v, vdist_nm)
            ]
            
            if len(membrane_adjacent_vesicles_in_slice) > 0:
                from scipy.spatial.distance import cdist
                vesicle_centers = np.array([v['center'] for v in membrane_adjacent_vesicles_in_slice])
                
                # Find the closest fusion point to each AZ-proximate vesicle being displayed
                distances = cdist(vesicle_centers, fusion_points)
                closest_fusion_indices = np.argmin(distances, axis=1)
                
                # Plot fusion points for AZ-proximate vesicles being displayed
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
            
            # Plotted fusion points for all AZ-proximate vesicles being displayed and near the slice
        
        # Add note about distance filtering to legend
        legend_elements = [
            Line2D([0], [0], color=VESICLE_VIZ_EDGE_FUSING, lw=2.5, label=lbl_v_fusing),
            Line2D([0], [0], color=VESICLE_VIZ_EDGE_CLOSE, lw=2, label=lbl_v_close),
            Line2D([0], [0], color=VESICLE_VIZ_EDGE_FAR, lw=1.5, label=lbl_v_far),
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
    
    # Version 3: Combined - Vesicles, Clefts, AuNPs, and Fusion Sites
    output_file3 = output_dir / f"{tomo_name}_combined_{suffix}.png"
    if output_file3.exists() and not rerun:
        print(f"Skipping {output_file3}, already exists.")
    else:
        fig3, ax3 = plt.subplots(figsize=(12, 12))
        ax3.imshow(slice2d, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
        
        for v in vesicles_in_slice:
            c = np.array(v["center"])
            r = v["radius"]
            role = _vesicle_viz_proximity_role(v, vdist_nm)
            handles = ax3.get_legend_handles_labels()[0]
            existing = [l.get_label() for l in handles]
            if role == "fusing":
                ec, lw, alpha = VESICLE_VIZ_EDGE_FUSING, 2.5, 0.88
                lab = lbl_v_fusing if lbl_v_fusing not in existing else ""
            elif role == "close":
                ec, lw, alpha = VESICLE_VIZ_EDGE_CLOSE, 2.0, 0.88
                lab = lbl_v_close if lbl_v_close not in existing else ""
            else:
                ec, lw, alpha = VESICLE_VIZ_EDGE_FAR, 1.5, 0.7
                lab = lbl_v_far if lbl_v_far not in existing else ""
            circ = Circle(
                (c[0], c[1]),
                r,
                edgecolor=ec,
                facecolor="none",
                fill=False,
                lw=lw,
                alpha=alpha,
                label=lab,
            )
            ax3.add_patch(circ)
        
        # Inner synaptic clefts (faded; underneath outer)
        for coords in azs_pre_inner_in_slice:
            ax3.scatter(coords[:, 0], coords[:, 1], color=inner_pre_rgb, s=3, alpha=inner_az_alpha,
                        label='Presynaptic AZ (inner)' if 'Presynaptic AZ (inner)' not in [l.get_label() for l in ax3.get_legend_handles_labels()[0]] else '')
        for coords in azs_post_inner_in_slice:
            ax3.scatter(coords[:, 0], coords[:, 1], color=inner_post_rgb, s=3, alpha=inner_az_alpha,
                        label='Postsynaptic AZ (inner)' if 'Postsynaptic AZ (inner)' not in [l.get_label() for l in ax3.get_legend_handles_labels()[0]] else '')
        
        # Overlay presynaptic synaptic cleft (outer)
        for coords in azs_pre_in_slice:
            ax3.scatter(coords[:,0], coords[:,1], color='red', s=3, alpha=0.1, 
                    label='Presynaptic AZ (outer)' if 'Presynaptic AZ (outer)' not in [l.get_label() for l in ax3.get_legend_handles_labels()[0]] else '')
        
        # Overlay postsynaptic synaptic cleft (outer)
        for coords in azs_post_in_slice:
            ax3.scatter(coords[:,0], coords[:,1], color='green', s=3, alpha=0.1, 
                    label='Postsynaptic AZ (outer)' if 'Postsynaptic AZ (outer)' not in [l.get_label() for l in ax3.get_legend_handles_labels()[0]] else '')
        
        # Add AuNPs with transparency
        if aunps_near is not None:
            ax3.scatter(aunps_near['faCoordinateX'], aunps_near['faCoordinateY'], 
                      color='gold', s=30, alpha=0.8, label='AuNPs')
        
        # Show fusion points for membrane-adjacent vesicles that are being displayed
        if fusion_points is not None and len(fusion_points) > 0 and len(vesicles_in_slice) > 0:
            membrane_adjacent_vesicles_in_slice = [
                v for v in vesicles_in_slice if _vesicle_az_proximate_for_viz(v, vdist_nm)
            ]
            
            if len(membrane_adjacent_vesicles_in_slice) > 0:
                from scipy.spatial.distance import cdist
                vesicle_centers = np.array([v['center'] for v in membrane_adjacent_vesicles_in_slice])
                
                # Find the closest fusion point to each AZ-proximate vesicle being displayed
                distances = cdist(vesicle_centers, fusion_points)
                closest_fusion_indices = np.argmin(distances, axis=1)
                
                # Plot fusion points for AZ-proximate vesicles being displayed
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
            
            # Plotted fusion points for all AZ-proximate vesicles being displayed and near the slice
        
        # Add note about distance filtering to legend
        legend_elements = [
            Line2D([0], [0], color=VESICLE_VIZ_EDGE_FUSING, lw=2.5, label=lbl_v_fusing),
            Line2D([0], [0], color=VESICLE_VIZ_EDGE_CLOSE, lw=2, label=lbl_v_close),
            Line2D([0], [0], color=VESICLE_VIZ_EDGE_FAR, lw=1.5, label=lbl_v_far),
            Line2D([0], [0], color=inner_pre_rgb, lw=1.5, label='Presynaptic AZ (inner)'),
            Line2D([0], [0], color=inner_post_rgb, lw=1.5, label='Postsynaptic AZ (inner)'),
            Line2D([0], [0], color='red', lw=1.5, label='Presynaptic AZ (outer)'),
            Line2D([0], [0], color='green', lw=1.5, label='Postsynaptic AZ (outer)'),
            plt.scatter([], [], color='gold', s=30, label='AuNPs'),
            plt.scatter([], [], color='orange', s=100, marker='*', label='Fusion Sites')
        ]
        ax3.legend(handles=legend_elements)
        ax3.set_title(f'Combined - Vesicles, Clefts, AuNPs, and Fusion Sites - {tomo_name}')
        ax3.set_xlabel('X (pixels)')
        ax3.set_ylabel('Y (pixels)')
        
        # Save the original combined image with legend
        plt.savefig(output_file3, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved combined visualization: {output_file3.name}")
        
        # Also save without suffix for PDF compatibility (only for the first synaptic cleft)
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
            aunps_results_dir = Path(tomo_path) / alignment_dir / "STT_results" / "aunps"
            cluster_star = aunps_results_dir / "aunp_clusters.star"
            if not cluster_star.exists():
                raise FileNotFoundError(
                    f"Required AuNP cluster file not found: {cluster_star}. "
                    "Run AuNP analysis (analyze_aunps) to produce aunp_clusters.star."
                )
            import starfile
            cluster_data = starfile.read(cluster_star)
            cluster_assignments = None
            if isinstance(cluster_data, dict):
                for v in cluster_data.values():
                    if isinstance(v, pd.DataFrame):
                        cluster_assignments = v
                        break
            elif isinstance(cluster_data, pd.DataFrame):
                cluster_assignments = cluster_data
            if cluster_assignments is None:
                raise ValueError(f"Could not read DataFrame from {cluster_star}")
            
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
        # Save next to other outputs for this run (output_dir), and mirror under the tomogram alignment path
        out_primary = output_dir / f"{tomo_name}_combined_aunpclusters_{suffix}.png"
        tomo_viz_dir = Path(tomo_path) / alignment_dir / "STT_results" / "visualizations"
        tomo_viz_dir.mkdir(parents=True, exist_ok=True)
        out_tomo = tomo_viz_dir / f"{tomo_name}_combined_aunpclusters_{suffix}.png"

        if out_primary.exists() and out_tomo.exists() and not rerun:
            print(f"Skipping AuNP cluster overlays ({suffix}), already exist.")
            plt.close(fig)
        else:
            plt.savefig(out_primary, dpi=300, bbox_inches='tight')
            print(f"  ✓ Saved combined AuNP cluster overlay: {out_primary.name}")
            if suffix == "az0" or suffix == "middle":
                out_pdf = output_dir / f"{tomo_name}_combined_aunpclusters.png"
                plt.savefig(out_pdf, dpi=300, bbox_inches='tight')
                print(f"  ✓ Saved combined AuNP cluster overlay for PDF: {out_pdf.name}")
            plt.close(fig)

            fig2, ax2 = plt.subplots(figsize=(12, 12))
            ax2.imshow(slice2d, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
            ax2.scatter(aunp_clusters['faCoordinateX'], aunp_clusters['faCoordinateY'],
                        c=colors, s=30, alpha=0.8, label='AuNPs (clustered)')
            ax2.set_title(f"{tomo_name} - Combined Overlay with AuNP Clusters")
            ax2.set_xlabel('X (pixels)')
            ax2.set_ylabel('Y (pixels)')
            ax2.legend(handles=legend_elements, loc='best')
            plt.savefig(out_tomo, dpi=300, bbox_inches='tight')
            plt.close(fig2)
            print(f"  ✓ Also saved cluster overlay under {tomo_viz_dir}")

        # --- AuNP synaptic / extrasynaptic overlay (AZ membranes + AuNPs; same marker style as combined_aunpclusters) ---
        tomo_viz_dir_syn = Path(tomo_path) / alignment_dir / "STT_results" / "visualizations"
        tomo_viz_dir_syn.mkdir(parents=True, exist_ok=True)
        out_syn_primary = output_dir / f"{tomo_name}_combined_aunps_synaptic_designation_{suffix}.png"
        out_syn_tomo = tomo_viz_dir_syn / f"{tomo_name}_combined_aunps_synaptic_designation_{suffix}.png"
        if out_syn_primary.exists() and out_syn_tomo.exists() and not rerun:
            print(f"Skipping synaptic-designation AuNP overlay ({suffix}), already exist.")
        else:
            desig_col = None
            if "synaptic_designation" in aunp_clusters.columns:
                desig_col = "synaptic_designation"
            elif "synaptic_designation_nm" in aunp_clusters.columns:
                desig_col = "synaptic_designation_nm"
            if desig_col is None:
                print(
                    "  Warning: No synaptic_designation column in aunp data; "
                    f"skipping {out_syn_primary.name}"
                )
            else:
                syn_gold = "#DAA520"
                extra_orange = "#FF8C00"
                unknown_gray = (0.55, 0.55, 0.55, 1.0)
                raw = (
                    aunp_clusters[desig_col]
                    .astype(str)
                    .str.strip()
                    .str.lower()
                )
                point_colors = []
                for v in raw:
                    if v in ("synaptic", "true", "1"):
                        point_colors.append(syn_gold)
                    elif v in ("extrasynaptic", "false", "0"):
                        point_colors.append(extra_orange)
                    elif v in ("nan", "none", ""):
                        point_colors.append(unknown_gray)
                    else:
                        point_colors.append(unknown_gray)

                fig_syn, ax_syn = plt.subplots(figsize=(12, 12))
                ax_syn.imshow(slice2d, cmap="gray", vmin=vmin, vmax=vmax, origin="lower")

                for coords in azs_pre_inner_in_slice:
                    ax_syn.scatter(
                        coords[:, 0],
                        coords[:, 1],
                        color=inner_pre_rgb,
                        s=3,
                        alpha=inner_az_alpha,
                        label="Presynaptic AZ (inner)"
                        if "Presynaptic AZ (inner)"
                        not in [l.get_label() for l in ax_syn.get_legend_handles_labels()[0]]
                        else "",
                    )
                for coords in azs_post_inner_in_slice:
                    ax_syn.scatter(
                        coords[:, 0],
                        coords[:, 1],
                        color=inner_post_rgb,
                        s=3,
                        alpha=inner_az_alpha,
                        label="Postsynaptic AZ (inner)"
                        if "Postsynaptic AZ (inner)"
                        not in [l.get_label() for l in ax_syn.get_legend_handles_labels()[0]]
                        else "",
                    )
                for coords in azs_pre_in_slice:
                    ax_syn.scatter(
                        coords[:, 0],
                        coords[:, 1],
                        color="red",
                        s=3,
                        alpha=0.1,
                        label="Presynaptic AZ (outer)"
                        if "Presynaptic AZ (outer)"
                        not in [l.get_label() for l in ax_syn.get_legend_handles_labels()[0]]
                        else "",
                    )
                for coords in azs_post_in_slice:
                    ax_syn.scatter(
                        coords[:, 0],
                        coords[:, 1],
                        color="green",
                        s=3,
                        alpha=0.1,
                        label="Postsynaptic AZ (outer)"
                        if "Postsynaptic AZ (outer)"
                        not in [l.get_label() for l in ax_syn.get_legend_handles_labels()[0]]
                        else "",
                    )

                # Match combined_aunpclusters: filled markers, s=30, alpha=0.8 (no heavy edge stroke)
                ax_syn.scatter(
                    aunp_clusters["faCoordinateX"],
                    aunp_clusters["faCoordinateY"],
                    c=point_colors,
                    s=30,
                    alpha=0.8,
                    linewidths=0,
                    edgecolors="none",
                )
                ax_syn.set_title(
                    f"{tomo_name} - AuNPs by synaptic designation (with synaptic cleft membranes)"
                )
                ax_syn.set_xlabel("X (pixels)")
                ax_syn.set_ylabel("Y (pixels)")

                legend_syn = [
                    Line2D([0], [0], color=inner_pre_rgb, lw=1.5, label="Presynaptic AZ (inner)"),
                    Line2D([0], [0], color=inner_post_rgb, lw=1.5, label="Postsynaptic AZ (inner)"),
                    Line2D([0], [0], color="red", lw=1.5, label="Presynaptic AZ (outer)"),
                    Line2D([0], [0], color="green", lw=1.5, label="Postsynaptic AZ (outer)"),
                    plt.scatter([], [], c=syn_gold, s=30, alpha=0.8, linewidths=0, edgecolors="none", label="Synaptic AuNP"),
                    plt.scatter(
                        [],
                        [],
                        c=extra_orange,
                        s=30,
                        alpha=0.8,
                        linewidths=0,
                        edgecolors="none",
                        label="Extrasynaptic AuNP",
                    ),
                    plt.scatter(
                        [],
                        [],
                        c=[unknown_gray],
                        s=30,
                        alpha=0.8,
                        linewidths=0,
                        edgecolors="none",
                        label="Unknown designation",
                    ),
                ]
                ax_syn.legend(handles=legend_syn, loc="best")

                plt.savefig(out_syn_primary, dpi=300, bbox_inches="tight")
                print(f"  ✓ Saved synaptic-designation AuNP overlay: {out_syn_primary.name}")
                if suffix == "az0" or suffix == "middle":
                    out_syn_pdf = output_dir / f"{tomo_name}_combined_aunps_synaptic_designation.png"
                    plt.savefig(out_syn_pdf, dpi=300, bbox_inches="tight")
                    print(f"  ✓ Saved synaptic-designation overlay for PDF: {out_syn_pdf.name}")

                plt.savefig(out_syn_tomo, dpi=300, bbox_inches="tight")
                plt.close(fig_syn)
                print(f"  ✓ Also saved synaptic-designation overlay under {tomo_viz_dir_syn}")
    # --- End AuNP Cluster Visualization ---


def run_zonogram_analysis_for_all_tomograms(tomo_paths, output_dir, csv_path=None, root_dir=None, rerun=False,
                                            sphere_size=None, sphere_color=None, aunp_distance_min=None, aunp_distance_max=None,
                                            aunp_distance_cutoff_direction=None, aunp_distance_cutoff_value=None,
                                            vesicle_distance_threshold: float = 20.0,
                                            fusing_perimeter_threshold: float = 1.0):
    """Run active zonogram analysis for all tomograms and generate PDF summaries."""
    try:
        # Import the combined zonogram analysis function
        from .cleft import (
            define_cleft, define_active_zonogram, extract_active_zonogram,
            import_membrane_segmentations_from_glb, find_active_zones_from_glb
        )
        
        # Individual files are saved to organized structure: results/visualizations/{tomogram_name}/cleft_MIPs/
        
        print(f"Running active zonogram analysis for {len(tomo_paths)} tomograms...")
        
        # Process each tomogram with progress tracking
        successful_count = 0
        failed_count = 0
        
        for i, tomo_info in enumerate(tomo_paths, 1):
            tomo_path, set_name, cleft_ids, alignment_dir = unpack_tomo_csv_row(tomo_info)
            
            tomogram_name = Path(tomo_path).name
            print(f"[{i}/{len(tomo_paths)}] Processing {tomogram_name} ({alignment_dir})...", end=" ", flush=True)
            
            try:
                # Run the combined active zonogram analysis for this tomogram
                result = run_combined_zonogram_analysis_single_tomogram(
                    tomo_path, None, cleft_ids, rerun,
                    alignment_dir=alignment_dir,
                    sphere_size=sphere_size, sphere_color=sphere_color,
                    aunp_distance_min=aunp_distance_min, aunp_distance_max=aunp_distance_max,
                    aunp_distance_cutoff_direction=aunp_distance_cutoff_direction,
                    aunp_distance_cutoff_value=aunp_distance_cutoff_value,
                    vesicle_distance_threshold=vesicle_distance_threshold,
                    fusing_perimeter_threshold=fusing_perimeter_threshold,
                )
                
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
            first_path, _, _, _ = unpack_tomo_csv_row(tomo_paths[0])
            first_tomo_path = Path(first_path)
            # Go up to find the root (assuming structure: root/set/TOP_TOMOS/tomogram)
            if first_tomo_path.parent.name == "TOP_TOMOS":
                root_dir = str(first_tomo_path.parent.parent.parent)
        generate_default_visualization_pdf_summary(tomo_paths, csv_path, root_dir)
        
        # Generate zonogram PDF summaries (all zonograms and mini zonograms)
        print("\nGenerating zonogram PDF summaries...")
        # Extract data directory from tomogram paths
        data_dir = None
        if tomo_paths:
            first_path, _, _, _ = unpack_tomo_csv_row(tomo_paths[0])
            first_tomo_path = Path(first_path)
            # Go up to find the data directory (assuming structure: data_dir/set/TOP_TOMOS/tomogram)
            if first_tomo_path.parent.name == "TOP_TOMOS":
                data_dir = str(first_tomo_path.parent.parent.parent)
        generate_zonogram_pdf_summaries(None, tomo_paths, data_dir)
        
        print(
            "\nActive zonogram analysis complete! Organized outputs under "
            "results/visualizations/{tomogram_name}/{alignment_dir}/cleft_MIPs/"
        )
        
    except ImportError as e:
        print(f"Warning: Could not import active zonogram analysis modules: {e}")
        print("Skipping active zonogram analysis.")
    except Exception as e:
        print(f"Error in active zonogram analysis: {e}")


def _transform_packing_samples_to_zonogram_xy(
    vertices: np.ndarray,
    values: np.ndarray,
    zonogram_data: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Project packing-density sample vertices into zonogram XY coordinates, keeping
    values aligned with any in-bounds filtering (unlike transform_coordinates_to_active_zonogram
    alone, which can drop rows without updating the value array).
    """
    import torch
    from torch_affine_utils.transforms_3d import T
    from torch_affine_utils.utils import homogenise_coordinates
    import einops

    coords = torch.tensor(vertices, dtype=torch.float32)
    M = torch.tensor(zonogram_data["transformation_matrix"], dtype=torch.float32)
    new_center = zonogram_data["extent"] // 2
    M = T(new_center) @ M
    coords = homogenise_coordinates(coords)
    transformed = M @ einops.rearrange(coords, "b xyzw -> b xyzw 1")
    transformed = einops.rearrange(transformed, "b xyzw 1 -> b xyzw")[:, :3]
    extent = zonogram_data["extent"]
    in_bounds = (
        (transformed[:, 0] >= 0)
        & (transformed[:, 0] < extent[0])
        & (transformed[:, 1] >= 0)
        & (transformed[:, 1] < extent[1])
        & (transformed[:, 2] >= 0)
        & (transformed[:, 2] < extent[2])
    )
    mask = in_bounds.numpy()
    xy = transformed[mask, :2].numpy()
    vals = np.asarray(values, dtype=float)[mask]
    return xy, vals


# Packing-density heatmap colormap options.
# - inferno: black/purple → crimson → orange → yellow (high contrast on gray)
# - viridis: purple → teal → yellow (calmer, colorblind-friendly)
# - hot: black → red → yellow → white (legacy / familiar cryo-ET look)
PACKING_DENSITY_CMAP_OPTIONS: tuple[str, ...] = ("inferno", "viridis", "hot")
PACKING_DENSITY_CMAP = PACKING_DENSITY_CMAP_OPTIONS[0]
PACKING_DENSITY_OVERLAY_ALPHA = 0.6

# Value metrics written for each cmap (overlay + heatmap-only).
# packing_coefficient: dimensionless 0–1 receptor packing (fixed vmin/vmax).
# nm2_per_aunp: linear area per AuNP = 1 / (AuNPs per nm²); dense → low values.
PACKING_DENSITY_VALUE_METRICS: tuple[str, ...] = ("packing_coefficient", "nm2_per_aunp")
PACKING_COEFF_VMIN = 0.0
PACKING_COEFF_VMAX = 1.0
# Legacy aliases
PACKING_DENSITY_VMIN = PACKING_COEFF_VMIN
PACKING_DENSITY_VMAX = PACKING_COEFF_VMAX
NM2_PER_AUNP_VMIN = 0.0
NM2_PER_AUNP_VMAX_PERCENTILE = 98.0


def _packing_density_paths_for_cmap(
    base_overlay_path: Path,
    cmap: str,
    *,
    value_metric: str = "packing_coefficient",
) -> tuple[Path, Path]:
    """
    Overlay and heatmap-only paths for one colormap × value-metric option.

    packing_coefficient:
      ``..._packing_density.png`` → ``..._packing_density_inferno.png``
    nm2_per_aunp:
      ``..._packing_density.png`` → ``..._packing_density_nm2_per_aunp_inferno.png``
    """
    if value_metric == "packing_coefficient":
        tag = cmap
    elif value_metric == "nm2_per_aunp":
        tag = f"nm2_per_aunp_{cmap}"
    else:
        tag = f"{value_metric}_{cmap}"
    overlay = base_overlay_path.with_name(
        f"{base_overlay_path.stem}_{tag}{base_overlay_path.suffix}"
    )
    heatmap_only = overlay.with_name(f"{overlay.stem}_heatmap_only{overlay.suffix}")
    return overlay, heatmap_only


def _packing_density_heatmap_only_path(overlay_path: Path) -> Path:
    """Sibling path for the transparent heatmap-only PNG (cmap already in stem)."""
    return overlay_path.with_name(f"{overlay_path.stem}_heatmap_only{overlay_path.suffix}")


def _active_zonogram_figure_layout(cleft_data):
    """
    Figure size and GridSpec ratios matching ``render_active_zonograms_findingampa_style``,
    so overlay and heatmap-only PNGs share identical canvas geometry.
    """
    res_ddw = cleft_data[2]
    figsize = (
        (res_ddw.shape[2] + res_ddw.shape[0]) / 50,
        (res_ddw.shape[1] + res_ddw.shape[0]) / 50,
    )
    gs_kwargs = {
        "width_ratios": [res_ddw.shape[2], res_ddw.shape[0]],
        "height_ratios": [res_ddw.shape[1], res_ddw.shape[0]],
    }
    return figsize, gs_kwargs


def _interpolate_packing_density_on_zonogram_xy(
    xy_points: np.ndarray,
    packing_values: np.ndarray,
    image_shape: tuple[int, int],
) -> np.ndarray:
    """
    Linear griddata onto the zonogram XY slice, then mask to the concave hull of
    the projected sample points (same approach as
    ``scripts/test_sliding_cylinder_packing_density.py``).
    """
    from scipy.interpolate import griddata
    from scipy.ndimage import zoom

    from .aunps import concave_hull_grid_mask

    height, width = image_shape
    grid_y, grid_x = np.mgrid[0:height, 0:width]
    density_map = griddata(
        xy_points,
        packing_values,
        (grid_x, grid_y),
        method="linear",
        fill_value=np.nan,
    )
    if density_map.shape != image_shape:
        zoom_factors = (
            height / density_map.shape[0],
            width / density_map.shape[1],
        )
        density_map = zoom(density_map, zoom_factors, order=1)

    inside_mask, _ = concave_hull_grid_mask(xy_points, density_map.shape)
    return np.where(inside_mask, density_map, np.nan)


def _nm2_per_aunp_from_density(aunp_density_per_nm2: np.ndarray) -> np.ndarray:
    """Convert AuNP/nm² to linear nm²/AuNP; zero/non-finite density → NaN."""
    dens = np.asarray(aunp_density_per_nm2, dtype=float)
    out = np.full(dens.shape, np.nan, dtype=float)
    valid = np.isfinite(dens) & (dens > 0)
    out[valid] = 1.0 / dens[valid]
    return out


def _packing_density_scale_and_cmap(
    value_metric: str,
    sample_values: np.ndarray,
    cmap: str,
) -> tuple[float, float, str, str]:
    """
    Return (vmin, vmax, matplotlib_cmap, colorbar_label_suffix) for a value metric.

    For nm²/AuNP the colormap is reversed so dense regions (low nm²/AuNP) stay on the
    bright end of the same palette family as the packing-coefficient maps.
    """
    if value_metric == "packing_coefficient":
        return (
            PACKING_COEFF_VMIN,
            PACKING_COEFF_VMAX,
            cmap,
            "AMPA packing coeff.",
        )
    if value_metric == "nm2_per_aunp":
        finite = np.asarray(sample_values, dtype=float)
        finite = finite[np.isfinite(finite) & (finite > 0)]
        if len(finite):
            vmax = float(np.nanpercentile(finite, NM2_PER_AUNP_VMAX_PERCENTILE))
            vmax = max(vmax, 1.0)
        else:
            vmax = 1.0
        return (
            NM2_PER_AUNP_VMIN,
            vmax,
            f"{cmap}_r",
            "nm² / AuNP",
        )
    raise ValueError(f"Unknown packing-density value metric: {value_metric!r}")


def _draw_packing_density_heatmap(
    ax,
    density_map: np.ndarray,
    *,
    extent,
    alpha: float,
    cmap: str,
    vmin: float,
    vmax: float,
):
    """Draw packing-density heatmap with the given colormap and linear value scale."""
    return ax.imshow(
        density_map,
        cmap=cmap,
        alpha=alpha,
        origin="lower",
        vmin=vmin,
        vmax=vmax,
        extent=extent,
        interpolation="mitchell",
        zorder=10,
    )


def _create_packing_density_zonogram_figure(
    zonogram_findingampa,
    density_map: np.ndarray,
    *,
    probe_radius_nm: float,
    with_tomogram_underlay: bool,
    heatmap_alpha: float,
    cmap: str,
    vmin: float,
    vmax: float,
    colorbar_quantity: str,
):
    """
    Build overlay or heatmap-only figure with identical figsize / GridSpec / colorbar.

    ``with_tomogram_underlay=True`` draws the usual gray XY/YZ/XZ tomogram slices.
    ``False`` leaves axes empty/transparent so only the XY heatmap (+ colorbar) shows.
    """
    import torch
    import matplotlib.pyplot as plt
    from matplotlib import gridspec

    res_ddw = zonogram_findingampa[2]
    base_image = torch.min(res_ddw, axis=0).values
    base_extent = [0, base_image.shape[1], 0, base_image.shape[0]]
    figsize, gs_kwargs = _active_zonogram_figure_layout(zonogram_findingampa)

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 2, **gs_kwargs)
    axxy = plt.subplot(gs[0, 0])
    axyz = plt.subplot(gs[0, 1], sharey=axxy)
    axxz = plt.subplot(gs[1, 0], sharex=axxy)

    if with_tomogram_underlay:
        gray_vmin = -20 * float(res_ddw.std())
        axxy.imshow(
            torch.min(res_ddw, axis=0).values,
            cmap="gray",
            interpolation="mitchell",
            vmax=-0.0,
            vmin=gray_vmin,
            origin="lower",
        )
        axxy.quiver(
            0, 0, 0, 50, color="g", angles="xy", scale_units="xy",
            units="xy", width=1, scale=1, clip_on=False,
        )
        axxy.quiver(
            0, 0, 50, 0, color="r", angles="xy", scale_units="xy",
            units="xy", width=1, scale=1, clip_on=False,
        )
        axyz.imshow(
            torch.min(res_ddw, axis=2).values.T,
            cmap="gray",
            interpolation="mitchell",
            vmax=-0.0,
            vmin=gray_vmin,
            origin="lower",
        )
        axyz.quiver(
            0, 0, 0, 50, color="g", angles="xy", scale_units="xy",
            units="xy", width=1, scale=1, clip_on=False,
        )
        axyz.quiver(
            0, 0, 50, 0, color="b", angles="xy", scale_units="xy",
            units="xy", width=1, scale=1, clip_on=False,
        )
        axxz.imshow(
            torch.min(res_ddw, axis=1).values,
            cmap="gray",
            interpolation="mitchell",
            vmax=-0.0,
            vmin=gray_vmin,
            origin="lower",
        )
        axxz.quiver(
            0, 0, 0, 50, color="b", angles="xy", scale_units="xy",
            units="xy", width=1, scale=1, clip_on=False,
        )
        axxz.quiver(
            0, 0, 50, 0, color="r", angles="xy", scale_units="xy",
            units="xy", width=1, scale=1, clip_on=False,
        )
    else:
        for ax in (axxy, axyz, axxz):
            ax.set_facecolor("none")
        axxy.set_xlim(base_extent[0], base_extent[1])
        axxy.set_ylim(base_extent[2], base_extent[3])
        # Preserve side-panel aspect slots without drawing content.
        axyz.set_xlim(0, float(res_ddw.shape[0]))
        axxz.set_ylim(0, float(res_ddw.shape[0]))

    im = _draw_packing_density_heatmap(
        axxy,
        density_map,
        extent=base_extent,
        alpha=heatmap_alpha,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    cbar = fig.colorbar(im, ax=axxy, fraction=0.046, pad=0.04)
    cmap_label = cmap[:-2] if str(cmap).endswith("_r") else cmap
    cbar.set_label(
        f"{colorbar_quantity} (probe r={int(round(probe_radius_nm))} nm; {cmap_label})",
        rotation=270,
        labelpad=15,
    )

    axxy.axis("off")
    axyz.axis("off")
    axxz.axis("off")
    plt.tight_layout()
    return fig


def save_packing_density_zonogram_overlay(
    zone_packing_data: dict,
    *,
    zonogram_findingampa,
    original_zone_data: dict,
    probe_radius_nm: float,
    packing_path_results_organized: Path,
    packing_path_tomogram: Path,
    rerun: bool,
    cmaps: tuple[str, ...] | None = None,
    value_metrics: tuple[str, ...] | None = None,
) -> list[str]:
    """
    Render and save packing-density heatmaps for a given probe radius.

    For each value metric × colormap, writes two matched PNGs (identical figure size /
    color scale) to both the organized-results and tomogram viz folders:

    - packing coefficient (0–1): ``*_packing_density_<cmap>*.png``
    - linear nm²/AuNP: ``*_packing_density_nm2_per_aunp_<cmap>*.png``
    - ``*_heatmap_only.png`` sibling — heatmap alone, transparent background

    The density field is masked to the concave hull of projected sample vertices.
    Returns the list of filenames written or already present under the results path.
    """
    import matplotlib.pyplot as plt

    cmaps = tuple(cmaps) if cmaps is not None else PACKING_DENSITY_CMAP_OPTIONS
    value_metrics = (
        tuple(value_metrics)
        if value_metrics is not None
        else PACKING_DENSITY_VALUE_METRICS
    )

    v_array = np.array(zone_packing_data["v_array"])
    packing_coefficient = np.array(zone_packing_data["packing_coefficient"])
    if packing_coefficient.dtype == object:
        packing_coefficient = np.array(
            [np.nan if x is None else float(x) for x in zone_packing_data["packing_coefficient"]],
            dtype=np.float64,
        )
    aunp_density = np.array(
        zone_packing_data.get("aunp_density_per_nm2", [np.nan] * len(packing_coefficient)),
        dtype=float,
    )
    if aunp_density.dtype == object:
        aunp_density = np.array(
            [np.nan if x is None else float(x) for x in aunp_density],
            dtype=np.float64,
        )
    if len(aunp_density) != len(packing_coefficient):
        aunp_density = np.full(len(packing_coefficient), np.nan, dtype=float)

    valid = np.isfinite(packing_coefficient)
    v_array = v_array[valid]
    packing_coefficient = packing_coefficient[valid]
    aunp_density = aunp_density[valid]
    nm2_per_aunp = _nm2_per_aunp_from_density(aunp_density)

    metric_sample_values = {
        "packing_coefficient": packing_coefficient,
        "nm2_per_aunp": nm2_per_aunp,
    }

    res_ddw = zonogram_findingampa[2]
    base_image_shape = (int(res_ddw.shape[1]), int(res_ddw.shape[2]))

    packing_path_results_organized.parent.mkdir(parents=True, exist_ok=True)
    packing_path_tomogram.parent.mkdir(parents=True, exist_ok=True)

    created_or_present: list[str] = []
    for value_metric in value_metrics:
        sample_values = metric_sample_values[value_metric]
        xy_points, metric_values = _transform_packing_samples_to_zonogram_xy(
            v_array,
            sample_values,
            original_zone_data,
        )
        # Drop non-finite metric samples (e.g. zero AuNP count → undefined nm²/AuNP).
        finite_metric = np.isfinite(metric_values)
        xy_points = xy_points[finite_metric]
        metric_values = metric_values[finite_metric]
        if len(xy_points) < 3:
            print(f"    Skipping {value_metric} heatmaps (<3 finite sample points).")
            continue

        density_map = _interpolate_packing_density_on_zonogram_xy(
            xy_points,
            metric_values,
            base_image_shape,
        )

        for cmap in cmaps:
            vmin, vmax, plot_cmap, quantity = _packing_density_scale_and_cmap(
                value_metric, metric_values, cmap
            )
            overlay_results, heatmap_only_results = _packing_density_paths_for_cmap(
                packing_path_results_organized, cmap, value_metric=value_metric
            )
            overlay_tomogram, heatmap_only_tomogram = _packing_density_paths_for_cmap(
                packing_path_tomogram, cmap, value_metric=value_metric
            )

            overlay_exists = overlay_results.exists() and overlay_tomogram.exists()
            heatmap_only_exists = heatmap_only_results.exists() and heatmap_only_tomogram.exists()
            if overlay_exists and heatmap_only_exists and not rerun:
                print(f"    Skipping {overlay_results.name} (+ heatmap-only), already exists.")
                created_or_present.extend([overlay_results.name, heatmap_only_results.name])
                continue

            if (not overlay_exists) or rerun:
                fig_overlay = _create_packing_density_zonogram_figure(
                    zonogram_findingampa,
                    density_map,
                    probe_radius_nm=probe_radius_nm,
                    with_tomogram_underlay=True,
                    heatmap_alpha=PACKING_DENSITY_OVERLAY_ALPHA,
                    cmap=plot_cmap,
                    vmin=vmin,
                    vmax=vmax,
                    colorbar_quantity=quantity,
                )
                fig_overlay.savefig(overlay_results)
                fig_overlay.savefig(overlay_tomogram)
                plt.close(fig_overlay)
                print(f"    ✓ Saved PNG: {overlay_results.name}")
            else:
                print(f"    Skipping {overlay_results.name}, already exists.")
            created_or_present.append(overlay_results.name)

            if (not heatmap_only_exists) or rerun:
                fig_only = _create_packing_density_zonogram_figure(
                    zonogram_findingampa,
                    density_map,
                    probe_radius_nm=probe_radius_nm,
                    with_tomogram_underlay=False,
                    heatmap_alpha=1.0,
                    cmap=plot_cmap,
                    vmin=vmin,
                    vmax=vmax,
                    colorbar_quantity=quantity,
                )
                fig_only.patch.set_facecolor("none")
                fig_only.patch.set_alpha(0.0)
                fig_only.savefig(heatmap_only_results, transparent=True)
                fig_only.savefig(heatmap_only_tomogram, transparent=True)
                plt.close(fig_only)
                print(f"    ✓ Saved PNG: {heatmap_only_results.name}")
            else:
                print(f"    Skipping {heatmap_only_results.name}, already exists.")
            created_or_present.append(heatmap_only_results.name)

    return created_or_present


def render_active_zonograms_findingampa_style(cleft_data):
    """
    Render active zonogram using the exact same approach as findingampa.
    Based on findingampa/src/findingampa/utils/analysis.py:render_active_zonograms()
    """
    import torch
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    
    res_ddw = cleft_data[2]
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
    
    axyz.imshow(torch.min(res_ddw, axis=2).values.T, cmap='gray', interpolation='mitchell', vmax=-0., vmin=-20*res_ddw.std(), origin='lower')

    axyz.quiver(0, 0, 0, 50, color='g', angles='xy', scale_units='xy', units="xy", width=1, scale=1, clip_on=False)
    axyz.quiver(0, 0, 50, 0, color='b', angles='xy', scale_units='xy', units="xy", width=1, scale=1, clip_on=False)
    
    axxz.imshow(torch.min(res_ddw, axis=1).values, cmap='gray', interpolation='mitchell', vmax=-0., vmin=-20*res_ddw.std(), origin='lower')

    axxz.quiver(0, 0, 0, 50, color='b', angles='xy', scale_units='xy', units="xy", width=1, scale=1, clip_on=False)
    axxz.quiver(0, 0, 50, 0, color='r', angles='xy', scale_units='xy', units="xy", width=1, scale=1, clip_on=False)
    # Hide axes 
    axxy.axis('off')
    axxz.axis('off')
    axyz.axis('off')
    plt.tight_layout()
    return fig

def render_mini_zonogram_xy_only(cleft_data, include_legend_space=False, extra_width_multiplier=1.5):
    """
    Render mini zonogram showing only the xy slice (top-left view from regular zonograms).
    If include_legend_space is True, creates a wider figure to accommodate legend on the right.
    """
    import torch
    import matplotlib.pyplot as plt
    
    res_ddw = cleft_data[2]
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

def select_aunps_findingampa_style(cleft_data, aunp_data, tomogram_path, cleft_id=0, original_zone_data=None,
                                   *, alignment_dir: str):
    """
    Select AuNPs for visualization using findingampa-style approach.
    Only includes AuNPs that belong to the specified synaptic cleft.
    """
    alignment_dir = require_alignment_dir(alignment_dir)
    from pathlib import Path
    import numpy as np
    import pandas as pd
    
    # Load AuNP data from filtered output file
    aunp_file = Path(tomogram_path) / alignment_dir / "STT_results" / "aunps" / "aunp_clusters.star"

    if not aunp_file.exists():
        raise FileNotFoundError(
            f"Required AuNP cluster file not found: {aunp_file}. "
            "Run AuNP analysis (analyze_aunps) to produce aunp_clusters.star."
        )
    
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
        
        # Filter AuNPs by synaptic cleft
        if 'cleft' not in aunp_df.columns:
            return []
        aunp_df = aunp_df[aunp_df['cleft'] == cleft_id]
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
        selected_aunp_pos_transformed += np.floor(np.array(cleft_data[2].shape)[[2,1,0]]/2)
        
        # Filter points within the volume
        valid_mask = np.all(selected_aunp_pos_transformed > 0, axis=1) & np.all(selected_aunp_pos_transformed < np.array(cleft_data[2].shape)[[2,1,0]], axis=1)
        selected_aunp_positions = selected_aunp_pos_transformed[valid_mask]
        
        return selected_aunp_positions
    else:
        # Fallback: return empty array if no transformation data available
        return []


def transform_positions_to_zonogram_coords(
    positions: np.ndarray,
    zonogram_findingampa,
    original_zone_data: dict,
) -> np.ndarray:
    """Map world-space coordinates (nm) to active zonogram panel coordinates."""
    center = np.asarray(original_zone_data["center"], dtype=float)
    coordinate_system = np.asarray(original_zone_data["transformation_matrix"][:3, :3], dtype=float)
    pts = np.atleast_2d(np.asarray(positions, dtype=float))
    transformed = (pts - center) @ coordinate_system.T
    transformed += np.floor(np.array(zonogram_findingampa[2].shape)[[2, 1, 0]] / 2)
    return transformed


def _filter_positions_inside_zonogram_extent(
    world_xyz: np.ndarray,
    zonogram_findingampa,
    original_zone_data: dict,
) -> np.ndarray:
    """Transform world coordinates and keep only points inside the zonogram volume."""
    world_xyz = np.atleast_2d(np.asarray(world_xyz, dtype=float))
    if world_xyz.size == 0:
        return np.zeros((0, 3), dtype=float)
    transformed = transform_positions_to_zonogram_coords(
        world_xyz, zonogram_findingampa, original_zone_data
    )
    extent = np.asarray(original_zone_data["extent"], dtype=float).reshape(1, -1)
    valid = np.all(transformed >= 0, axis=1) & np.all(transformed < extent, axis=1)
    return transformed[valid]


def _scatter_points_on_zonogram_axes(
    axxy,
    axxz,
    axyz,
    zonogram_xyz: np.ndarray,
    *,
    marker: str = "o",
    color: str = "cyan",
    size: float = 28,
    alpha: float = 0.85,
    edgecolors: Optional[str] = None,
    linewidths: float = 0.5,
    zorder: int = 6,
) -> None:
    """Scatter Nx3 zonogram-panel coordinates on all three active-zonogram views."""
    if len(zonogram_xyz) == 0:
        return
    scatter_kw = dict(
        s=size,
        c=color,
        alpha=alpha,
        marker=marker,
        zorder=zorder,
    )
    if edgecolors is not None:
        scatter_kw["edgecolors"] = edgecolors
        scatter_kw["linewidths"] = linewidths
    axxy.scatter(zonogram_xyz[:, 0], zonogram_xyz[:, 1], **scatter_kw)
    axxz.scatter(zonogram_xyz[:, 2], zonogram_xyz[:, 1], **scatter_kw)
    axyz.scatter(zonogram_xyz[:, 0], zonogram_xyz[:, 2], **scatter_kw)



def _compute_fusion_null_query_point_dataframes(
    tomogram_path: Path,
    alignment_dir: str,
    az_mapping: dict,
    *,
    vesicle_distance_threshold: float = 20.0,
    fusion_point_threshold: float = 20.0,
) -> dict[str, pd.DataFrame]:
    """
    Build long-form 40 nm shift and label-permutation tables for zonogram overlays.

    Uses the same 3D geometry and null models as
    ``fusion_point_aunp_position_distance_and_Ripleys_analyses``.
    """
    from .fusion_point_aunp_position_distance_and_Ripleys_analyses import (
        build_fusion_null_query_point_dataframes_for_zonograms,
    )

    alignment_dir = require_alignment_dir(alignment_dir)
    try:
        return build_fusion_null_query_point_dataframes_for_zonograms(
            tomogram_path,
            alignment_dir,
            az_mapping,
            vesicle_distance_threshold=vesicle_distance_threshold,
            fusion_point_threshold=fusion_point_threshold,
        )
    except Exception as exc:
        print(f"Warning: could not compute fusion null query points for zonograms: {exc}")
        return {"40nm_shift": pd.DataFrame(), "label_permutation": pd.DataFrame()}


def _unique_shift_query_points_for_zone(df: pd.DataFrame, zone_name: str) -> np.ndarray:
    if df is None or df.empty or "cleft_name" not in df.columns:
        return np.zeros((0, 3), dtype=float)
    sub = df[df["cleft_name"] == zone_name]
    if sub.empty:
        return np.zeros((0, 3), dtype=float)
    cols = ["query_point_x_nm", "query_point_y_nm", "query_point_z_nm"]
    return sub.drop_duplicates(subset=["vesicle_id", "shift_replicate_id"])[cols].to_numpy(dtype=float)


def _unique_label_perm_query_points_for_zone(df: pd.DataFrame, zone_name: str) -> np.ndarray:
    if df is None or df.empty or "cleft_name" not in df.columns:
        return np.zeros((0, 3), dtype=float)
    sub = df[df["cleft_name"] == zone_name]
    if sub.empty:
        return np.zeros((0, 3), dtype=float)
    cols = ["query_point_x_nm", "query_point_y_nm", "query_point_z_nm"]
    return (
        sub.drop_duplicates(subset=["permutation_id", "fusion_site_index"])[cols]
        .to_numpy(dtype=float)
    )


def _fusing_fusion_points_by_zone(
    tomogram_path: Path,
    alignment_dir: str,
    *,
    vesicle_distance_threshold: float = 20.0,
    fusion_point_threshold: float = 20.0,
) -> dict[str, np.ndarray]:
    """Real fusing-vesicle fusion points grouped by synaptic cleft name."""
    from .aunps import enumerate_close_vesicle_fusion_points
    from .fusion_point_aunp_position_distance_and_Ripleys_analyses import (
        zone_name_for_presynaptic_membrane,
    )

    by_zone: dict[str, list[list[float]]] = {}
    for fp in enumerate_close_vesicle_fusion_points(
        tomogram_path,
        alignment_dir=alignment_dir,
        vesicle_distance_threshold=vesicle_distance_threshold,
        fusion_point_threshold=fusion_point_threshold,
        fusing_only=True,
    ):
        zone_name = zone_name_for_presynaptic_membrane(fp.get("closest_membrane"))
        if not zone_name:
            continue
        by_zone.setdefault(zone_name, []).append(
            [
                float(fp["fusion_point_x_nm"]),
                float(fp["fusion_point_y_nm"]),
                float(fp["fusion_point_z_nm"]),
            ]
        )
    return {
        zone_name: np.asarray(pts, dtype=float)
        for zone_name, pts in by_zone.items()
        if pts
    }


def _save_active_zonogram_query_point_overlay(
    *,
    zonogram_findingampa,
    original_zone_data: dict,
    query_world_xyz: np.ndarray,
    reference_world_xyz: np.ndarray | None,
    output_path_results: Path,
    output_path_tomogram: Path,
    overlay_label: str,
    overlay_color: str,
    overlay_marker: str = "o",
    overlay_size: float = 30,
    overlay_alpha: float = 0.8,
    reference_label: str = "Fusing fusion points",
    rerun: bool = False,
) -> bool:
    """Save a three-panel active zonogram with null-model and optional reference fusion sites."""
    if output_path_results.exists() and output_path_tomogram.exists() and not rerun:
        print(f"    Skipping {output_path_results.name}, already exists.")
        return False

    query_zono = _filter_positions_inside_zonogram_extent(
        query_world_xyz, zonogram_findingampa, original_zone_data
    )
    reference_zono = (
        _filter_positions_inside_zonogram_extent(
            reference_world_xyz, zonogram_findingampa, original_zone_data
        )
        if reference_world_xyz is not None and len(reference_world_xyz) > 0
        else np.zeros((0, 3), dtype=float)
    )
    if len(query_zono) == 0 and len(reference_zono) == 0:
        return False

    fig = render_active_zonograms_findingampa_style(zonogram_findingampa)
    axxy, axxz, axyz = fig.get_axes()

    if len(reference_zono) > 0:
        _scatter_points_on_zonogram_axes(
            axxy,
            axxz,
            axyz,
            reference_zono,
            marker="*",
            color="orange",
            size=100,
            alpha=0.9,
            edgecolors="darkorange",
            linewidths=0.5,
            zorder=8,
        )
    if len(query_zono) > 0:
        _scatter_points_on_zonogram_axes(
            axxy,
            axxz,
            axyz,
            query_zono,
            marker=overlay_marker,
            color=overlay_color,
            size=overlay_size,
            alpha=overlay_alpha,
            edgecolors="white" if overlay_marker != "." else None,
            linewidths=0.4,
            zorder=7,
        )

    legend_handles: list = []
    legend_labels: list[str] = []
    if len(reference_zono) > 0:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="*",
                color="w",
                markerfacecolor="orange",
                markeredgecolor="darkorange",
                markersize=10,
                linewidth=0.5,
            )
        )
        legend_labels.append(reference_label)
    if len(query_zono) > 0:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker=overlay_marker,
                color="w",
                markerfacecolor=overlay_color,
                markeredgecolor="white" if overlay_marker != "." else overlay_color,
                markersize=8,
                linewidth=0.5,
                alpha=overlay_alpha,
            )
        )
        legend_labels.append(f"{overlay_label} (n={len(query_zono)})")

    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower right",
            bbox_to_anchor=(1.0, 0.0),
            fontsize=8,
            frameon=True,
            fancybox=True,
            shadow=True,
        )

    output_path_results.parent.mkdir(parents=True, exist_ok=True)
    output_path_tomogram.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path_results)
    fig.savefig(output_path_tomogram)
    plt.close(fig)
    print(f"    ✓ Saved PNG: {output_path_results.name}")
    return True


def _postsynaptic_center_distance_column(aunp_df: "pd.DataFrame") -> Optional[str]:
    """Column for mean of active-zone postsynaptic inner/outer distances (from analyze_aunps)."""
    for col in (
        "distance_to_postsynaptic_active_outer_inner_mean_nm",
        "distance_to_postsynaptic_active_outer_inner_mean",
    ):
        if col in aunp_df.columns:
            return col
    return None


def select_aunps_with_distances_findingampa_style(cleft_data, aunp_data, tomogram_path, cleft_id=0, original_zone_data=None,
                                                   *, alignment_dir: str):
    """
    Select AuNPs for visualization with distance to postsynaptic synaptic-cleft center
    (mean of inner/outer active membrane distances from analyze_aunps).
    Only includes AuNPs that belong to the specified synaptic cleft.
    Returns a dict with 'positions' and 'distances' arrays.
    """
    alignment_dir = require_alignment_dir(alignment_dir)
    from pathlib import Path
    import numpy as np
    import pandas as pd
    
    # Load AuNP data from filtered output file
    aunp_file = Path(tomogram_path) / alignment_dir / "STT_results" / "aunps" / "aunp_clusters.star"

    if not aunp_file.exists():
        raise FileNotFoundError(
            f"Required AuNP cluster file not found: {aunp_file}. "
            "Run AuNP analysis (analyze_aunps) to produce aunp_clusters.star."
        )
    
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
        
        # Filter AuNPs by synaptic cleft
        if 'cleft' not in aunp_df.columns:
            return {'positions': np.array([]), 'distances': np.array([])}
        aunp_df = aunp_df[aunp_df['cleft'] == cleft_id]
        if aunp_df.empty:
            return {'positions': np.array([]), 'distances': np.array([])}
        
        # Get positions and distances
        aunp_positions = aunp_df[['faCoordinateX', 'faCoordinateY', 'faCoordinateZ']].values
        
        dist_col = _postsynaptic_center_distance_column(aunp_df)
        if dist_col is None:
            print(
                "Warning: distance_to_postsynaptic_active_outer_inner_mean not found in "
                "aunp_clusters.star (re-run AuNP analysis to compute synaptic-cleft center distances)"
            )
            return {'positions': np.array([]), 'distances': np.array([])}
        # Coerce to float so downstream np.isnan works even if the STAR column
        # was read as object dtype (e.g. empty/non-numeric entries -> NaN).
        post_distances = pd.to_numeric(aunp_df[dist_col], errors='coerce').to_numpy(dtype=float)

    except Exception as e:
        print(f"Error loading AuNPs with distances in select_aunps_with_distances_findingampa_style: {e}")
        return {'positions': np.array([]), 'distances': np.array([])}
    
    # Use proper transformation if original zone data is available
    if original_zone_data is not None:
        # Transform AuNP positions to zonogram coordinate system (same as original run_zonogram.py)
        center = original_zone_data['center']
        coordinate_system = original_zone_data['transformation_matrix'][:3, :3]
        
        selected_aunp_pos_transformed = (aunp_positions - center) @ coordinate_system.T
        selected_aunp_pos_transformed += np.floor(np.array(cleft_data[2].shape)[[2,1,0]]/2)
        
        # Filter points within the volume
        valid_mask = np.all(selected_aunp_pos_transformed > 0, axis=1) & np.all(selected_aunp_pos_transformed < np.array(cleft_data[2].shape)[[2,1,0]], axis=1)
        selected_aunp_positions = selected_aunp_pos_transformed[valid_mask]
        selected_distances = post_distances[valid_mask]
        
        return {'positions': selected_aunp_positions, 'distances': selected_distances}
    else:
        # Fallback: return empty arrays if no transformation data available
        return {'positions': np.array([]), 'distances': np.array([])}


def select_aunps_by_cluster_findingampa_style(cleft_data, cluster_data, tomogram_path, cleft_id=0, original_zone_data=None,
                                               *, alignment_dir: str):
    """
    Select AuNPs by cluster for visualization using findingampa-style approach.
    Only includes AuNPs that belong to the specified synaptic cleft.
    """
    alignment_dir = require_alignment_dir(alignment_dir)
    from pathlib import Path
    import numpy as np
    import pandas as pd
    
    # Load cluster data from filtered output file
    cluster_file = Path(tomogram_path) / alignment_dir / "STT_results" / "aunps" / "aunp_clusters.star"

    if not cluster_file.exists():
        raise FileNotFoundError(
            f"Required AuNP cluster file not found: {cluster_file}. "
            "Run AuNP analysis (analyze_aunps) to produce aunp_clusters.star."
        )
    
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
        
        # Filter AuNPs by synaptic cleft
        if 'cleft' not in cluster_df.columns:
            return [], []
        cluster_df = cluster_df[cluster_df['cleft'] == cleft_id]
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
        selected_aunp_pos_transformed += np.floor(np.array(cleft_data[2].shape)[[2,1,0]]/2)
        
        # Filter points within the volume
        valid_mask = np.all(selected_aunp_pos_transformed > 0, axis=1) & np.all(selected_aunp_pos_transformed < np.array(cleft_data[2].shape)[[2,1,0]], axis=1)
        selected_aunp_positions = selected_aunp_pos_transformed[valid_mask]
        selected_cluster_assignments = cluster_assignments[valid_mask]
        
        return selected_aunp_positions, selected_cluster_assignments
    else:
        # Fallback: return empty arrays if no transformation data available
        return [], []


def run_combined_zonogram_analysis_single_tomogram(tomo_path, output_dir, cleft_ids=None, rerun=False,
                                                    *, alignment_dir: str,
                                                    sphere_size=None, sphere_color=None, aunp_distance_min=None, aunp_distance_max=None,
                                                    aunp_distance_cutoff_direction=None, aunp_distance_cutoff_value=None,
                                                    vesicle_distance_threshold: float = 20.0,
                                                    fusing_perimeter_threshold: float = 1.0):
    """Run combined active zonogram analysis for a single tomogram - EXACT SAME CODE as original script."""
    from .cleft import (
        define_cleft, define_active_zonogram, extract_active_zonogram,
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
    
    alignment_dir = require_alignment_dir(alignment_dir)
    tomogram_path = str(tomo_path)
    tomogram_name = Path(tomo_path).name
    
    try:
        # Step 1: Load membrane data and synaptic clefts (shared between both analyses)
        membrane_data = import_membrane_segmentations_from_glb(tomogram_path, alignment_dir=alignment_dir)
        
        # Find synaptic clefts
        clefts_data = find_active_zones_from_glb(membrane_data, distance_range=(10.0, 40.0))
        
        if not clefts_data['clefts']:
            print("No synaptic clefts found. Skipping active zonogram analysis.")
            return {"success": False, "reason": "No synaptic clefts found"}
        
        # Load AuNP data to match synaptic clefts (required)
        aunp_star_path = Path(tomogram_path) / alignment_dir / "STT_results" / "aunps" / "aunp_clusters.star"
        if not aunp_star_path.exists():
            raise FileNotFoundError(
                f"Required AuNP cluster file not found: {aunp_star_path}. "
                "Run AuNP analysis (analyze_aunps) to produce aunp_clusters.star."
            )
        import starfile
        star_data = starfile.read(aunp_star_path)
        aunp_data = None
        if isinstance(star_data, dict):
            for v in star_data.values():
                if isinstance(v, pd.DataFrame):
                    aunp_data = v
                    break
        elif isinstance(star_data, pd.DataFrame):
            aunp_data = star_data
        if aunp_data is None:
            raise ValueError(f"Could not read AuNP DataFrame from {aunp_star_path}")
        
        # Filter synaptic clefts based on CSV specification using smart matching
        if cleft_ids is not None and cleft_ids != []:
            # Handle both list and string inputs
            if isinstance(cleft_ids, list):
                # Already parsed list from CLI
                selected_az_indices = cleft_ids
                # CSV specified synaptic clefts
            else:
                # Parse synaptic cleft indices from CSV string (handle floats like "2.0")
                az_str = str(cleft_ids) if cleft_ids is not None else ""
                if az_str.strip() != "" and az_str.lower() != "nan":
                    try:
                        selected_az_indices = []
                        for x in az_str.split(","):
                            x = x.strip()
                            if x.isdigit():
                                selected_az_indices.append(int(x))
                            elif x.replace(".", "").isdigit():  # Handle floats like "2.0"
                                selected_az_indices.append(int(float(x)))
                        # CSV specified synaptic clefts
                    except Exception as e:
                        print(f"Warning: Error parsing synaptic cleft indices from CSV '{cleft_ids}': {e}")
                        print("Proceeding with all synaptic clefts")
                        selected_az_indices = None
                else:
                    print("No synaptic clefts specified in CSV, proceeding with all synaptic clefts")
                    selected_az_indices = None
        else:
            print("No synaptic cleft filtering specified, proceeding with all synaptic clefts")
            selected_az_indices = None
        
        # If no specific synaptic clefts were specified, get all available synaptic cleft numbers from filtered AuNP file
        if selected_az_indices is None:
            aunps_results_dir = Path(tomogram_path) / alignment_dir / "STT_results" / "aunps"
            cluster_star = aunps_results_dir / "aunp_clusters.star"
            if not cluster_star.exists():
                raise FileNotFoundError(
                    f"Required AuNP cluster file not found: {cluster_star}. "
                    "Run AuNP analysis (analyze_aunps) to produce aunp_clusters.star."
                )
            import starfile
            star_data = starfile.read(cluster_star)
            if isinstance(star_data, dict):
                df = None
                for v in star_data.values():
                    if isinstance(v, pd.DataFrame):
                        df = v
                        break
            else:
                df = star_data

            if df is not None and 'cleft' in df.columns:
                aunp_az_numbers = sorted(df['cleft'].unique().tolist())
                # Remove -1 if present (means "not in any synaptic cleft")
                aunp_az_numbers = [az for az in aunp_az_numbers if az != -1]
                selected_az_indices = aunp_az_numbers
                print(f"Using all available synaptic clefts from filtered AuNP file: {selected_az_indices}")
            else:
                raise ValueError(
                    f"Could not read synaptic clefts from {cluster_star}: missing DataFrame or 'cleft' column."
                )
        
        # Use saved mapping from cleft.py (created by define_cleft)
        from .cleft import load_cleft_mapping
        
        # Load saved mapping
        az_mapping = load_cleft_mapping(tomogram_path, alignment_dir)
        
        if not az_mapping:
            # No mapping found - use all synaptic clefts as fallback but print error
            print(f"No saved synaptic cleft mapping found for {tomogram_name}. Active zone analysis must be run first with smart matching to create the mapping.")
            print(f"FALLBACK: Using all {len(clefts_data['clefts'])} synaptic clefts found from GLB (no filtering applied).")
            # Use all zones, no filtering
            filtered_clefts = clefts_data['clefts']
            # Create a dummy mapping for filename generation (use zone names as-is)
            az_mapping = {}
            for idx, zone_name in enumerate(clefts_data['clefts'].keys()):
                az_mapping[idx] = zone_name
        else:
            # Convert string keys to int (JSON stores dict keys as strings)
            az_mapping = {int(k): v for k, v in az_mapping.items()}
            
            # Filter to only include zones in the mapping
            filtered_clefts = {}
            for az_index in selected_az_indices:
                if az_index in az_mapping:
                    zone_name = az_mapping[az_index]
                    if zone_name in clefts_data['clefts']:
                        filtered_clefts[zone_name] = clefts_data['clefts'][zone_name]
                    else:
                        raise ValueError(f"Zone {zone_name} from saved mapping not found in synaptic clefts data. This indicates a mismatch between the mapping and current synaptic clefts.")
                else:
                    raise ValueError(f"Active zone index {az_index} not found in saved mapping. This indicates the synaptic cleft analysis was run with different indices.")
            
            print(f"Using saved synaptic cleft mapping for {len(filtered_clefts)} zones")
        
        # Store the mapping for later use in filename generation
        clefts_data['az_mapping'] = az_mapping
        clefts_data['clefts'] = filtered_clefts
        
        # Step 2: Regular Active Zonogram Analysis
        
        # Define active zonograms
        zonogram_results = define_active_zonogram(clefts_data)
        
        if zonogram_results['status'] == 'completed':
            # Defined active zonograms
            
            # Extract and save zonograms
            extracted_results = extract_active_zonogram(
                zonogram_results, clefts_data, tomogram_path, alignment_dir=alignment_dir
            )
            
            if extracted_results and isinstance(extracted_results, dict) and 'rendered_zonograms' in extracted_results and extracted_results.get('rendered_zonograms'):
                # Create output directories (per alignment to avoid collisions)
                # 1. results/visualizations/{tomogram_name}/{alignment_dir}/cleft_MIPs/full/
                results_cleft_mips_dir_full = organized_results_viz_path(
                    "results", tomogram_name, alignment_dir, "cleft_MIPs", "full"
                )
                results_cleft_mips_dir_full.mkdir(parents=True, exist_ok=True)
                
                # 2. In tomogram's STT_results/visualizations/cleft_MIPs directory
                tomogram_cleft_mips_dir = Path(tomogram_path) / alignment_dir / "STT_results" / "visualizations" / "cleft_MIPs"
                tomogram_cleft_mips_dir.mkdir(parents=True, exist_ok=True)
                
                files_created = []
                
                fusion_null_query_dfs = _compute_fusion_null_query_point_dataframes(
                    Path(tomogram_path),
                    alignment_dir,
                    clefts_data.get("az_mapping", {}),
                    vesicle_distance_threshold=vesicle_distance_threshold,
                )
                fusing_fusion_points_by_zone = _fusing_fusion_points_by_zone(
                    Path(tomogram_path),
                    alignment_dir,
                    vesicle_distance_threshold=vesicle_distance_threshold,
                )
                
                # Create filename suffix using the az_mapping (define once for all zones)
                if 'az_mapping' in clefts_data:
                    # Use the first synaptic cleft index from the mapping as the default suffix
                    first_az_index = list(clefts_data['az_mapping'].keys())[0] if clefts_data['az_mapping'] else 0
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
                    if 'az_mapping' in clefts_data:
                        # Find which synaptic cleft index maps to this zone_name
                        az_index = None
                        for idx, mapped_zone in clefts_data['az_mapping'].items():
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
                    mrc_filename = f"{tomogram_name}_cleft_MIP_{zone_name}{suffix}.mrc"
                    mrcfile.write(tomogram_cleft_mips_dir / mrc_filename, zone_data['transformed_tomogram'], overwrite=True)
                    print(f"    ✓ Saved MRC: {mrc_filename}")
                    
                    # Save NPY file to tomogram directory only
                    npy_filename = f"{tomogram_name}_cleft_MIP_{zone_name}{suffix}.npy"
                    npy_data = {
                        "cs": np.eye(3),
                        "center": np.zeros(3),
                        "objects": ()
                    }
                    np.save(tomogram_cleft_mips_dir / npy_filename, npy_data, allow_pickle=True)
                    print(f"    ✓ Saved MRC: {mrc_filename}")
                    
                    # Generate main PNG and save to organized structure and tomogram directory
                    png_filename = f"{tomogram_name}_cleft_MIP_{zone_name}{suffix}.png"
                    png_path_results_organized = results_cleft_mips_dir_full / png_filename
                    png_path_tomogram = tomogram_cleft_mips_dir / png_filename
                    
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

                    # Active zonogram with synaptic-cleft center marked (mean pre/post membrane center)
                    center_png_filename = (
                        f"{tomogram_name}_cleft_MIP_{zone_name}{suffix}_center.png"
                    )
                    center_path_results_organized = (
                        results_cleft_mips_dir_full / center_png_filename
                    )
                    center_path_tomogram = tomogram_cleft_mips_dir / center_png_filename

                    if (
                        center_path_results_organized.exists()
                        and center_path_tomogram.exists()
                        and not rerun
                    ):
                        print(f"    Skipping {center_png_filename}, already exists.")
                        files_created.append(center_png_filename)
                    else:
                        fig_center = render_active_zonograms_findingampa_style(
                            zonogram_findingampa
                        )
                        axxy_c, axxz_c, axyz_c = fig_center.get_axes()
                        az_center = np.asarray(original_zone_data["center"], dtype=float)
                        center_pos = transform_positions_to_zonogram_coords(
                            az_center.reshape(1, 3),
                            zonogram_findingampa,
                            original_zone_data,
                        )[0]
                        center_marker = dict(
                            marker="x",
                            c="red",
                            s=120,
                            linewidths=2.5,
                            zorder=20,
                        )
                        axxy_c.scatter(center_pos[0], center_pos[1], **center_marker)
                        axxz_c.scatter(center_pos[2], center_pos[1], **center_marker)
                        axyz_c.scatter(center_pos[0], center_pos[2], **center_marker)
                        fig_center.savefig(center_path_results_organized)
                        fig_center.savefig(center_path_tomogram)
                        plt.close(fig_center)
                        print(f"    ✓ Saved PNG: {center_png_filename}")
                        files_created.append(center_png_filename)
                    
                    # Extract synaptic cleft ID from the az_mapping
                    cleft_id = None
                    if 'az_mapping' in clefts_data:
                        # Find which synaptic cleft index maps to this zone_name
                        for idx, mapped_zone in clefts_data['az_mapping'].items():
                            if mapped_zone == zone_name:
                                cleft_id = idx
                                break
                    
                    # Fallback if no mapping found
                    if cleft_id is None:
                        if 'pre1_post1' in zone_name:
                            cleft_id = 0
                        elif 'pre2_post1' in zone_name:
                            cleft_id = 1
                        else:
                            cleft_id = 0  # Default fallback
                    
                    # Generate AuNP visualization
                    selected_aunps = select_aunps_findingampa_style(
                        zonogram_findingampa, None, tomogram_path, cleft_id, original_zone_data,
                        alignment_dir=alignment_dir,
                    )
                    if len(selected_aunps) > 0:
                        aunp_filename = f"{tomogram_name}_cleft_MIP_{zone_name}_selected_aunps{suffix}.png"
                        aunp_path_results_organized = results_cleft_mips_dir_full / aunp_filename
                        aunp_path_tomogram = tomogram_cleft_mips_dir / aunp_filename
                        
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
                    
                    # Generate distance-colored AuNP visualization (postsynaptic synaptic-cleft center distance)
                    selected_aunps_with_distances = select_aunps_with_distances_findingampa_style(
                        zonogram_findingampa, None, tomogram_path, cleft_id, original_zone_data,
                        alignment_dir=alignment_dir,
                    )
                    if len(selected_aunps_with_distances['positions']) > 0:
                        selected_aunps_dist = selected_aunps_with_distances['positions']
                        post_distances = selected_aunps_with_distances['distances']
                        
                        dist_filename = f"{tomogram_name}_cleft_MIP_{zone_name}_selected_aunps_by_distance_to_post{suffix}.png"
                        dist_path_results_organized = results_cleft_mips_dir_full / dist_filename
                        dist_path_tomogram = tomogram_cleft_mips_dir / dist_filename
                        
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
                                    cbar.set_label(
                                        'Distance to Postsynaptic Active-Zone Center (nm)',
                                        rotation=270, labelpad=20, fontsize=9,
                                    )
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
                                cutoff_filename = f"{tomogram_name}_cleft_MIP_{zone_name}_selected_aunps_by_distance_to_post_{cutoff_direction}_{cutoff_value}nm{suffix}.png"
                                cutoff_path_results_organized = results_cleft_mips_dir_full / cutoff_filename
                                cutoff_path_tomogram = tomogram_cleft_mips_dir / cutoff_filename
                                
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
                                        cbar.set_label(
                                            'Distance to Postsynaptic Active-Zone Center (nm)',
                                            rotation=270, labelpad=20, fontsize=9,
                                        )
                                    
                                    fig.savefig(cutoff_path_results_organized, bbox_inches='tight')
                                    fig.savefig(cutoff_path_tomogram, bbox_inches='tight')
                                    plt.close(fig)
                                    print(f"    ✓ Saved PNG: {cutoff_filename}")
                                    files_created.append(cutoff_filename)
                            else:
                                print(f"    No AuNPs found {cutoff_direction} {cutoff_value} nm from postsynaptic membrane for {zone_name}")
                    
                    # Generate cluster-colored AuNP visualization
                    selected_aunps, cluster_assignments = select_aunps_by_cluster_findingampa_style(
                        zonogram_findingampa, None, tomogram_path, cleft_id, original_zone_data,
                        alignment_dir=alignment_dir,
                    )
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
                        
                        # Add fusion points if available (rings colored by source vesicle: fusing=aqua, close=purple)
                        fusion_points = []
                        fusion_points_transformed = []
                        fusion_sources_filtered: list = []
                        try:
                            from .aunps import compute_fusion_points_with_sources

                            fusion_points, fusion_sources = compute_fusion_points_with_sources(
                                tomogram_path,
                                vesicle_distance_threshold=vesicle_distance_threshold,
                                alignment_dir=alignment_dir,
                            )
                            fusion_points = np.asarray(fusion_points)

                            if len(fusion_points) > 0:
                                # Transform fusion points to the same coordinate system as AuNPs
                                from torch_affine_utils.utils import homogenise_coordinates
                                import einops

                                fusion_points_homog = homogenise_coordinates(
                                    torch.tensor(fusion_points, dtype=torch.float32)
                                )

                                M = torch.tensor(original_zone_data['transformation_matrix'], dtype=torch.float32)
                                transformed_fusion_points = M @ einops.rearrange(
                                    fusion_points_homog, "b xyzw -> b xyzw 1"
                                )
                                transformed_fusion_points = einops.rearrange(
                                    transformed_fusion_points, "b xyzw 1 -> b xyzw"
                                )[:, :3]

                                extent = original_zone_data['extent']
                                new_center = extent // 2
                                fusion_points_transformed = transformed_fusion_points.numpy() + new_center

                                valid_mask = np.all(fusion_points_transformed >= 0, axis=1) & np.all(
                                    fusion_points_transformed < extent.reshape(1, -1), axis=1
                                )
                                fusion_sources_filtered = [
                                    fusion_sources[i]
                                    for i in range(len(fusion_sources))
                                    if valid_mask[i]
                                ]
                                fusion_points_transformed = fusion_points_transformed[valid_mask]

                                vdist_ring = float(vesicle_distance_threshold)
                                if len(fusion_points_transformed) > 0:
                                    circle_radius = 20  # 20 nm radius for 40 nm diameter
                                    for fp, src_ves in zip(
                                        fusion_points_transformed, fusion_sources_filtered
                                    ):
                                        ring_ec = _vesicle_fusion_ring_edgecolor(src_ves, vdist_ring)
                                        axxy.scatter(
                                            fp[0],
                                            fp[1],
                                            color="orange",
                                            s=100,
                                            alpha=0.9,
                                            marker="*",
                                            edgecolors="darkorange",
                                            linewidth=0.5,
                                        )
                                        axxz.scatter(
                                            fp[2],
                                            fp[1],
                                            color="orange",
                                            s=100,
                                            alpha=0.9,
                                            marker="*",
                                            edgecolors="darkorange",
                                            linewidth=0.5,
                                        )
                                        axyz.scatter(
                                            fp[0],
                                            fp[2],
                                            color="orange",
                                            s=100,
                                            alpha=0.9,
                                            marker="*",
                                            edgecolors="darkorange",
                                            linewidth=0.5,
                                        )

                                        circle_xy = plt.Circle(
                                            (fp[0], fp[1]),
                                            circle_radius,
                                            fill=False,
                                            edgecolor=ring_ec,
                                            linestyle=":",
                                            linewidth=1.5,
                                            alpha=0.85,
                                        )
                                        circle_xz = plt.Circle(
                                            (fp[2], fp[1]),
                                            circle_radius,
                                            fill=False,
                                            edgecolor=ring_ec,
                                            linestyle=":",
                                            linewidth=1.5,
                                            alpha=0.85,
                                        )
                                        circle_yz = plt.Circle(
                                            (fp[0], fp[2]),
                                            circle_radius,
                                            fill=False,
                                            edgecolor=ring_ec,
                                            linestyle=":",
                                            linewidth=1.5,
                                            alpha=0.85,
                                        )

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
                            fusion_handle = Line2D(
                                [0],
                                [0],
                                marker="*",
                                color="w",
                                markerfacecolor="orange",
                                markeredgecolor="darkorange",
                                markersize=10,
                                linewidth=0.5,
                                linestyle=":",
                                markeredgewidth=1.5,
                            )
                            legend_handles.append(fusion_handle)
                            legend_labels.append("Fusion sites")
                        
                        if legend_handles:
                            fig.legend(legend_handles, legend_labels, loc='lower right', bbox_to_anchor=(1.0, 0.0),
                                      fontsize=8, frameon=True, fancybox=True, shadow=True)
                        
                        # Save the version with fusion points
                        cluster_filename = f"{tomogram_name}_cleft_MIP_{zone_name}_selected_aunps_by_cluster{suffix}.png"
                        cluster_path_results_organized = results_cleft_mips_dir_full / cluster_filename
                        cluster_path_tomogram = tomogram_cleft_mips_dir / cluster_filename
                        
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
                        cluster_no_fusion_filename = f"{tomogram_name}_cleft_MIP_{zone_name}_selected_aunps_by_cluster_no_fusion{suffix}.png"
                        cluster_no_fusion_path_results_organized = results_cleft_mips_dir_full / cluster_no_fusion_filename
                        cluster_no_fusion_path_tomogram = tomogram_cleft_mips_dir / cluster_no_fusion_filename
                        
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
                    packing_density_file = (
                        Path(tomogram_path) / alignment_dir / "STT_results" / "aunps"
                        / "packing_density_results.json"
                    )
                    if packing_density_file.exists() and zone_name in zonogram_results['zonogram_data']:
                        try:
                            with open(packing_density_file, 'r') as f:
                                packing_density_data = json.load(f)

                            if zone_name in packing_density_data:
                                zone_packing_data = packing_density_data[zone_name]
                                probe_radius_nm = float(
                                    zone_packing_data.get(
                                        "cylinder_radius_nm",
                                        zone_packing_data.get("probe_radius_nm", 25.0),
                                    )
                                )
                                packing_filename = (
                                    f"{tomogram_name}_cleft_MIP_{zone_name}_packing_density{suffix}.png"
                                )
                                packing_path_results_organized = (
                                    results_cleft_mips_dir_full / packing_filename
                                )
                                packing_path_tomogram = (
                                    tomogram_cleft_mips_dir / packing_filename
                                )
                                created_names = save_packing_density_zonogram_overlay(
                                    zone_packing_data,
                                    zonogram_findingampa=zonogram_findingampa,
                                    original_zone_data=original_zone_data,
                                    probe_radius_nm=probe_radius_nm,
                                    packing_path_results_organized=packing_path_results_organized,
                                    packing_path_tomogram=packing_path_tomogram,
                                    rerun=rerun,
                                )
                                files_created.extend(created_names)
                        except Exception as e:
                            print(
                                f"    Warning: Could not create packing density visualization "
                                f"for {zone_name}: {e}"
                            )
                            import traceback
                            traceback.print_exc()

                    reference_fusion_world = fusing_fusion_points_by_zone.get(
                        zone_name, np.zeros((0, 3), dtype=float)
                    )
                    shift_query_world = _unique_shift_query_points_for_zone(
                        fusion_null_query_dfs.get("40nm_shift", pd.DataFrame()),
                        zone_name,
                    )
                    if len(shift_query_world) > 0 or len(reference_fusion_world) > 0:
                        shift_png_filename = (
                            f"{tomogram_name}_cleft_MIP_{zone_name}"
                            f"_fusing_40nm_shift_controls{suffix}.png"
                        )
                        shift_path_results = (
                            results_cleft_mips_dir_full / shift_png_filename
                        )
                        shift_path_tomogram = (
                            tomogram_cleft_mips_dir / shift_png_filename
                        )
                        if _save_active_zonogram_query_point_overlay(
                            zonogram_findingampa=zonogram_findingampa,
                            original_zone_data=original_zone_data,
                            query_world_xyz=shift_query_world,
                            reference_world_xyz=reference_fusion_world,
                            output_path_results=shift_path_results,
                            output_path_tomogram=shift_path_tomogram,
                            overlay_label="40 nm tangential shift controls (100 replicates)",
                            overlay_color="deepskyblue",
                            overlay_marker="o",
                            overlay_size=26,
                            overlay_alpha=0.75,
                            rerun=rerun,
                        ) or (
                            shift_path_results.exists() and shift_path_tomogram.exists()
                        ):
                            files_created.append(shift_png_filename)

                    perm_query_world = _unique_label_perm_query_points_for_zone(
                        fusion_null_query_dfs.get("label_permutation", pd.DataFrame()),
                        zone_name,
                    )
                    if len(perm_query_world) > 0 or len(reference_fusion_world) > 0:
                        perm_png_filename = (
                            f"{tomogram_name}_cleft_MIP_{zone_name}"
                            f"_label_permutation_fusion_sites{suffix}.png"
                        )
                        perm_path_results = (
                            results_cleft_mips_dir_full / perm_png_filename
                        )
                        perm_path_tomogram = (
                            tomogram_cleft_mips_dir / perm_png_filename
                        )
                        if _save_active_zonogram_query_point_overlay(
                            zonogram_findingampa=zonogram_findingampa,
                            original_zone_data=original_zone_data,
                            query_world_xyz=perm_query_world,
                            reference_world_xyz=reference_fusion_world,
                            output_path_results=perm_path_results,
                            output_path_tomogram=perm_path_tomogram,
                            overlay_label="Label-permutation fusion sites (100 replicates)",
                            overlay_color="mediumorchid",
                            overlay_marker=".",
                            overlay_size=18,
                            overlay_alpha=0.55,
                            rerun=rerun,
                        ) or (
                            perm_path_results.exists() and perm_path_tomogram.exists()
                        ):
                            files_created.append(perm_png_filename)
            else:
                print("No active zonograms found")
        else:
            print("No active zonograms found")
            
        print()  # Spacer line
        
        # Step 3: Check if AuNP analysis was completed
        # Step 3: Checking AuNP analysis status
        
        # Check for required AuNP analysis files
        aunp_analysis_path = Path(tomogram_path) / alignment_dir / "STT_results" / "aunps"
        cluster_data_path = aunp_analysis_path / "aunp_clusters.star"
        
        if not aunp_analysis_path.exists():
            raise FileNotFoundError(
                f"Required AuNP results directory not found: {aunp_analysis_path}. "
                "Run AuNP analysis (analyze_aunps) first."
            )

        if not cluster_data_path.exists():
            raise FileNotFoundError(
                f"Required AuNP cluster file not found: {cluster_data_path}. "
                "Run AuNP analysis (analyze_aunps) to produce aunp_clusters.star."
            )
        
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
                
                # Create mini zonogram (use filtered synaptic clefts)
                # Create mini zonogram directory in organized structure
                results_cleft_mips_dir_mini = organized_results_viz_path(
                    "results", tomogram_name, alignment_dir, "cleft_MIPs", "mini"
                )
                results_cleft_mips_dir_mini.mkdir(parents=True, exist_ok=True)
                
                success = create_mini_zonogram_for_cluster(
                    cluster_data,
                    cluster_id,
                    tomogram_path,
                    tomogram_cleft_mips_dir,
                    results_cleft_mips_dir_mini,
                    clefts_data,
                    cluster_color_map,
                    tomogram_name,
                    alignment_dir=alignment_dir,
                    suffix=default_suffix,
                    rerun=rerun,
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
            "regular_zonograms": len([f for f in files_created if 'cleft_MIP' in f and 'mini' not in f]),
            "mini_zonograms": mini_zonogram_count,
            "files_created": files_created
        }
            
    except Exception as e:
        print(f"Error in active zonogram analysis: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "reason": f"Error: {e}"}




def create_mini_zonogram_for_cluster(
    cluster_data,
    cluster_id,
    tomogram_path,
    tomogram_azograms_dir,
    results_azograms_dir,
    clefts_data,
    cluster_color_map,
    tomogram_name,
    *,
    alignment_dir: str,
    suffix="",
    rerun=False,
):
    """
    Create a mini zonogram centered on a specific small cluster.
    Uses the same transformation matrix calculation as regular active zonograms.
    Uses the same color scheme as the regular zonogram analysis.
    Saves files in both tomogram's STT_results/visualizations/cleft_MIPs and results/visualizations/cleft_MIPs.
    """
    alignment_dir = require_alignment_dir(alignment_dir)
    # Creating mini zonogram for cluster
    
    # Debug: Check available synaptic clefts
    # Available synaptic clefts checked
    
    # Get cluster center
    cluster_center = cluster_data[['faCoordinateX', 'faCoordinateY', 'faCoordinateZ']].mean().values
    
    # Calculate transformation matrix using the same approach as regular active zonograms
    from torch_affine_utils.transforms_3d import T
    from torch_affine_utils.utils import homogenise_coordinates
    import torch
    import einops
    
    # Find the closest synaptic cleft to this cluster
    closest_zone_name = None
    min_distance = float('inf')
    
    for zone_name, zone_data in clefts_data['clefts'].items():
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
        print(f"    Warning: No synaptic clefts found, using identity matrix")
        coordinate_system = np.eye(3)
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, 3] = -cluster_center
        extent = np.array([100, 100, 100])
    else:
        # Use the membrane data from the closest synaptic cleft
        zone_data = clefts_data['clefts'][closest_zone_name]
        
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
            'cleft_count': 1,
            'zonogram_data': {zone_name_to_use: mini_zonogram_data}
        }
        extracted_data = extract_active_zonogram(
            mini_zonogram_dict, clefts_data, tomogram_path, alignment_dir=alignment_dir
        )
        
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
        
        print(f"    Processing {len(tomo_paths)} tomogram CSV rows for PDF generation")
        
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
        
        # One PDF section per CSV row (same tomogram name may appear with different alignments)
        for i, tomo_info in enumerate(tomo_paths, 1):
            tomo_path, _set_name, cleft_ids, alignment_dir = unpack_tomo_csv_row(tomo_info)
            tomogram_name = Path(tomo_path).name
            print(f"    [{i}/{len(tomo_paths)}] Processing {tomogram_name} ({alignment_dir})...", end=" ", flush=True)

            selected_az_indices = None
            if cleft_ids is not None:
                az_str = str(cleft_ids)
                if az_str.strip() != "" and az_str.lower() != "nan":
                    try:
                        selected_az_indices = []
                        for x in az_str.split(","):
                            x = x.strip()
                            if x.isdigit():
                                selected_az_indices.append(int(x))
                            elif x.replace(".", "").isdigit():
                                selected_az_indices.append(int(float(x)))
                    except Exception as e:
                        print(f"    Warning: Error parsing synaptic cleft indices for {tomogram_name}: {e}")

            azograms_dir = organized_results_viz_path(
                "results", tomogram_name, alignment_dir, "cleft_MIPs", "full"
            )
            
            if not azograms_dir.exists():
                print(f"    Warning: Cleft MIPs directory not found: {azograms_dir}")
                continue
            
            story.append(
                Paragraph(f"Tomogram: {tomogram_name} — alignment: {alignment_dir}", title_style)
            )
            story.append(Spacer(1, 10))
            
            # Find regular active zonogram files (aunps_by_cluster.png) for this specific tomogram
            regular_zonogram_files = list(azograms_dir.glob(f"{tomogram_name}_cleft_MIP_*_selected_aunps_by_cluster_az*.png"))
            
            # Add regular active zonograms first
            for zonogram_file in sorted(regular_zonogram_files):
                try:
                    # Get zone name from filename
                    zone_name = zonogram_file.stem.split('_cleft_MIP_')[1].split('_selected_aunps_by_cluster')[0]
                    
                    # Filter by synaptic cleft indices if specified in CSV
                    if selected_az_indices is not None:
                        # Extract synaptic cleft index from filename suffix (e.g., "_az0" -> 0)
                        try:
                            az_suffix = zonogram_file.stem.split('_az')[-1]
                            az_index = int(az_suffix)
                            if az_index not in selected_az_indices:
                                print(f"      Skipping synaptic cleft {zone_name} (index {az_index}) - not in CSV")
                                continue
                        except (ValueError, IndexError):
                            print(f"      Warning: Could not parse synaptic cleft index from filename: {zonogram_file.name}")
                            # Include it by default if we can't parse the index
                    
                    # Add zone name as subtitle
                    zone_style = ParagraphStyle(
                        'ZoneTitle',
                        parent=styles['Heading2'],
                        fontSize=12,
                        spaceAfter=10,
                        textColor=colors.darkgreen
                    )
                    story.append(Paragraph(f"Cleft: {zone_name}", zone_style))
                    
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
            mini_azograms_dir = organized_results_viz_path(
                "results", tomogram_name, alignment_dir, "cleft_MIPs", "mini"
            )
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
            
            if i < len(tomo_paths):
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
        
        for i, tomo_info in enumerate(tomo_paths, 1):
            tomo_path, _set_name, cleft_ids, alignment_dir = unpack_tomo_csv_row(tomo_info)
            tomogram_path = Path(tomo_path)
            tomogram_name = tomogram_path.name

            selected_az_indices = None
            if cleft_ids is not None:
                az_str = str(cleft_ids)
                if az_str.strip() != "" and az_str.lower() != "nan":
                    try:
                        selected_az_indices = []
                        for x in az_str.split(","):
                            x = x.strip()
                            if x.isdigit():
                                selected_az_indices.append(int(x))
                            elif x.replace(".", "").isdigit():
                                selected_az_indices.append(int(float(x)))
                    except Exception as e:
                        print(f"    Warning: Error parsing synaptic cleft indices for {tomogram_name}: {e}")

            print(f"    [{i}/{len(tomo_paths)}] Processing {tomogram_name} ({alignment_dir})...", end=" ", flush=True)

            azograms_dir_organized = organized_results_viz_path(
                "results", tomogram_name, alignment_dir, "cleft_MIPs", "mini"
            )
            
            if not azograms_dir_organized.exists():
                print(f"    Warning: Cleft MIPs directory not found: {azograms_dir_organized}")
                continue
            
            mini_zonogram_files = list(azograms_dir_organized.glob("*_mini_zonogram_cluster_*_comparison.png"))
            
            if not mini_zonogram_files:
                print(f"    Warning: No mini zonogram files found for {tomogram_name}")
                continue
            
            # Filter mini zonogram files by synaptic cleft indices if specified in CSV
            if selected_az_indices is not None and mini_zonogram_files:
                filtered_mini_files = []
                for mini_file in mini_zonogram_files:
                    # Extract synaptic cleft info from filename (e.g., "tomogram_mini_zonogram_cluster_1_comparison.png")
                    # We need to check if this mini zonogram belongs to a filtered synaptic cleft
                    # For now, we'll include all mini zonograms since they're cluster-specific, not synaptic cleft specific
                    # But we could add filtering logic here if needed
                    filtered_mini_files.append(mini_file)
                mini_zonogram_files = filtered_mini_files
                # Filtered to mini zonogram files
            
            # Get cluster data to identify clusters with 4 AuNPs
            cluster_data_path = tomogram_path / alignment_dir / "STT_results" / "aunps" / "aunp_clusters.star"
            clusters_with_4_aunps = set()

            if not cluster_data_path.exists():
                raise FileNotFoundError(
                    f"Required AuNP cluster file not found: {cluster_data_path}. "
                    "Run AuNP analysis (analyze_aunps) to produce aunp_clusters.star."
                )
            try:
                import starfile
                cluster_df = starfile.read(cluster_data_path)
                cluster_counts = cluster_df['aunp_cluster'].value_counts()
                clusters_with_4_aunps = set(cluster_counts[cluster_counts == 4].index)
            except Exception as e:
                raise RuntimeError(f"Could not read cluster data from {cluster_data_path}: {e}") from e
            
            if mini_zonogram_files:
                story.append(
                    Paragraph(f"Tomogram: {tomogram_name} — alignment: {alignment_dir}", title_style)
                )
                story.append(Spacer(1, 10))
                
                # Also add to 4 AuNP PDF if this tomogram has any 4 AuNP clusters
                has_4aunp_clusters = any(
                    int(f.stem.split('_cluster_')[1].split('_comparison')[0].split('_az')[0]) in clusters_with_4_aunps
                    for f in mini_zonogram_files
                )
                if has_4aunp_clusters:
                    story_4aunps.append(
                        Paragraph(f"Tomogram: {tomogram_name} — alignment: {alignment_dir}", title_style)
                    )
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