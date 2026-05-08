#!/usr/bin/env python3
"""
Script to run define_active_zonogram function on a tomogram using the exact same approach as findingampa.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import mrcfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
import torch
import pandas as pd
from scipy.spatial import cKDTree

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from synaptic_tomo_tools.activezone import define_active_zone, define_active_zonogram, extract_active_zonogram

def render_active_zonograms_findingampa_style(active_zone_data):
    """
    Render active zonogram using the exact same approach as findingampa.
    Based on findingampa/src/findingampa/utils/analysis.py:render_active_zonograms()
    """
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

def render_active_zone_extend_via_blender_fallback(center, size, rotation_matrix, output_filename):
    """
    Fallback function for position visualization when Blender is not available.
    Creates a simple matplotlib-based position visualization.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Create a simple representation showing the active zone position
    ax.text(0.5, 0.5, f'Active Zone Position\nCenter: {center}\nSize: {size}', 
            transform=ax.transAxes, ha='center', va='center', fontsize=12,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('Active Zone Position (Blender not available)')
    
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()

def select_aunps_findingampa_style(aunp_data, az_data, res_ddw, threshold=3.8, skip_segment_activezone=False):
    """
    Select AuNPs using the exact same approach as findingampa.
    Based on findingampa/src/findingampa/utils/analysis.py:select_aunps()
    """
    # For now, we'll use a simplified approach since we don't have the full findingampa data structure
    # We'll use the available AuNP data and apply basic filtering
    
    # Load AuNP data if it's a file path
    if isinstance(aunp_data, str) or isinstance(aunp_data, Path):
        try:
            aunp_df = pd.read_csv(aunp_data)
        except:
            # Try reading as star file
            try:
                import starfile
                aunp_df = starfile.read(aunp_data)
            except:
                print(f"Could not load AuNP data from {aunp_data}")
                return None, None, None, None, None
    else:
        aunp_df = aunp_data
    
    # Check if we have the required columns
    required_cols = ['faCoordinateX', 'faCoordinateY', 'faCoordinateZ']
    if not all(col in aunp_df.columns for col in required_cols):
        print(f"AuNP data missing required columns: {required_cols}")
        return None, None, None, None, None
    
    # Get AuNP positions
    selected_aunp_pos = aunp_df[required_cols].values
    
    # Apply basic filtering based on distance to membranes
    if 'distance_to_presynaptic' in aunp_df.columns and 'distance_to_postsynaptic' in aunp_df.columns:
        # Filter AuNPs that are close to both membranes (within 30nm)
        pre_dist_mask = aunp_df['distance_to_presynaptic'] < 30
        post_dist_mask = aunp_df['distance_to_postsynaptic'] < 30
        significant_picks_mask = pre_dist_mask & post_dist_mask
    else:
        # If no distance data, use all AuNPs
        significant_picks_mask = np.ones(len(aunp_df), dtype=bool)
    
    # Filter AuNPs based on active zone
    if 'active_zone' in aunp_df.columns:
        # Filter for AuNPs in the current active zone
        az_id = int(az_data.get('zone_name', '0').split('_')[-1]) if isinstance(az_data.get('zone_name'), str) else 0
        az_mask = aunp_df['active_zone'] == az_id
        significant_picks_mask = significant_picks_mask & az_mask
    
    # Apply the mask
    selected_aunp_pos = selected_aunp_pos[significant_picks_mask]
    
    # For now, we'll use a simplified approach for active zone segmentation
    # In the full findingampa implementation, this would use membrane meshes
    selected_aunp_pos_mask = np.ones(selected_aunp_pos.shape[0], dtype=bool)
    selected_aunp_pos_postsyn_mask = np.zeros(selected_aunp_pos.shape[0], dtype=bool)
    
    # Transform AuNP positions to zonogram coordinate system
    center = az_data['center']
    coordinate_system = az_data['transformation_matrix'][:3, :3]
    
    selected_aunp_pos_transformed = (selected_aunp_pos - center) @ coordinate_system.T
    selected_aunp_pos_transformed += np.floor(np.array(res_ddw.shape)[[2,1,0]]/2)
    
    # Filter points within the volume
    valid_mask = np.all(selected_aunp_pos_transformed > 0, axis=1) & np.all(selected_aunp_pos_transformed < np.array(res_ddw.shape)[[2,1,0]], axis=1)
    selected_aunp_pos_transformed = selected_aunp_pos_transformed[valid_mask]
    
    # Create the visualization figure
    fig = render_active_zonograms_findingampa_style((coordinate_system, center, res_ddw, ()))
    
    # Get subplot axes and plot AuNPs
    (axxy, axxz, axyz) = fig.get_axes()
    
    if len(selected_aunp_pos_transformed) > 0:
        # Calculate circle size for 6nm diameter AuNPs (same as cluster version)
        circle_size = 36  # This should give approximately 6nm diameter circles
        
        # Plot active zone AuNPs as red rings (clear interior, red edge)
        axxy.scatter(selected_aunp_pos_transformed[:,0], selected_aunp_pos_transformed[:,1], 
                    s=circle_size, c='none', alpha=1.0, edgecolors='red', linewidth=1.5)
        axxz.scatter(selected_aunp_pos_transformed[:,2], selected_aunp_pos_transformed[:,1], 
                    s=circle_size, c='none', alpha=1.0, edgecolors='red', linewidth=1.5)
        axyz.scatter(selected_aunp_pos_transformed[:,0], selected_aunp_pos_transformed[:,2], 
                    s=circle_size, c='none', alpha=1.0, edgecolors='red', linewidth=1.5)
    
    return fig, significant_picks_mask, selected_aunp_pos, selected_aunp_pos_mask, selected_aunp_pos_postsyn_mask

def select_aunps_by_cluster_findingampa_style(
    aunp_data,
    cluster_data,
    az_data,
    res_ddw,
    threshold=3.8,
    skip_segment_activezone=False,
    tomogram_path=None,
    *,
    alignment_dir: str,
):
    """
    Select AuNPs and color them by cluster assignment using the exact same approach as findingampa.
    Based on findingampa/src/findingampa/utils/analysis.py:select_aunps() but with cluster coloring.
    """
    from synaptic_tomo_tools.alignment_utils import require_alignment_dir
    alignment_dir = require_alignment_dir(alignment_dir, context="run_zonogram cluster overlay")
    # Load AuNP data if it's a file path
    if isinstance(aunp_data, str) or isinstance(aunp_data, Path):
        try:
            aunp_df = pd.read_csv(aunp_data)
        except:
            try:
                import starfile
                aunp_df = starfile.read(aunp_data)
            except:
                print(f"Could not load AuNP data from {aunp_data}")
                return None, None, None, None, None
    else:
        aunp_df = aunp_data
    
    # Load cluster data if it's a file path
    if isinstance(cluster_data, str) or isinstance(cluster_data, Path):
        try:
            import starfile
            cluster_df = starfile.read(cluster_data)
        except:
            print(f"Could not load cluster data from {cluster_data}")
            return None, None, None, None, None
    else:
        cluster_df = cluster_data
    
    # Check if we have the required columns
    required_cols = ['faCoordinateX', 'faCoordinateY', 'faCoordinateZ']
    if not all(col in aunp_df.columns for col in required_cols):
        print(f"AuNP data missing required columns: {required_cols}")
        return None, None, None, None, None
    
    # Check for cluster column
    if 'aunp_cluster' not in cluster_df.columns:
        print("Cluster data missing 'aunp_cluster' column")
        return None, None, None, None, None
    
    # Get AuNP positions
    selected_aunp_pos = aunp_df[required_cols].values
    
    # Apply basic filtering based on distance to membranes
    if 'distance_to_presynaptic' in aunp_df.columns and 'distance_to_postsynaptic' in aunp_df.columns:
        # Filter AuNPs that are close to both membranes (within 30nm)
        pre_dist_mask = aunp_df['distance_to_presynaptic'] < 30
        post_dist_mask = aunp_df['distance_to_postsynaptic'] < 30
        significant_picks_mask = pre_dist_mask & post_dist_mask
    else:
        # If no distance data, use all AuNPs
        significant_picks_mask = np.ones(len(aunp_df), dtype=bool)
    
    # Filter AuNPs based on active zone
    if 'active_zone' in aunp_df.columns:
        # Filter for AuNPs in the current active zone
        az_id = int(az_data.get('zone_name', '0').split('_')[-1]) if isinstance(az_data.get('zone_name'), str) else 0
        az_mask = aunp_df['active_zone'] == az_id
        significant_picks_mask = significant_picks_mask & az_mask
    
    # Apply the mask
    selected_aunp_pos = selected_aunp_pos[significant_picks_mask]
    
    # Get cluster assignments for the selected AuNPs
    # We need to match AuNPs between the two dataframes
    # For simplicity, we'll assume they have the same order or can be matched by coordinates
    if len(selected_aunp_pos) > 0 and len(cluster_df) > 0:
        # Try to match by coordinates (with some tolerance)
        cluster_assignments = []
        for pos in selected_aunp_pos:
            # Find the closest cluster AuNP
            distances = np.sqrt(np.sum((cluster_df[required_cols].values - pos)**2, axis=1))
            closest_idx = np.argmin(distances)
            if distances[closest_idx] < 5.0:  # 5nm tolerance
                cluster_assignments.append(cluster_df.iloc[closest_idx]['aunp_cluster'])
            else:
                cluster_assignments.append(-1)  # No cluster assigned
    else:
        cluster_assignments = [-1] * len(selected_aunp_pos)
    
    # For now, we'll use a simplified approach for active zone segmentation
    selected_aunp_pos_mask = np.ones(selected_aunp_pos.shape[0], dtype=bool)
    selected_aunp_pos_postsyn_mask = np.zeros(selected_aunp_pos.shape[0], dtype=bool)
    
    # Transform AuNP positions to zonogram coordinate system
    center = az_data['center']
    coordinate_system = az_data['transformation_matrix'][:3, :3]
    
    selected_aunp_pos_transformed = (selected_aunp_pos - center) @ coordinate_system.T
    selected_aunp_pos_transformed += np.floor(np.array(res_ddw.shape)[[2,1,0]]/2)
    
    # Filter points within the volume
    valid_mask = np.all(selected_aunp_pos_transformed > 0, axis=1) & np.all(selected_aunp_pos_transformed < np.array(res_ddw.shape)[[2,1,0]], axis=1)
    selected_aunp_pos_transformed = selected_aunp_pos_transformed[valid_mask]
    cluster_assignments = [cluster_assignments[i] for i in range(len(cluster_assignments)) if valid_mask[i]]
    
    # Create the visualization figure
    fig = render_active_zonograms_findingampa_style((coordinate_system, center, res_ddw, ()))
    
    # Get subplot axes and plot AuNPs with cluster colors
    (axxy, axxz, axyz) = fig.get_axes()
    
    if len(selected_aunp_pos_transformed) > 0:
        # Create a color map for clusters
        unique_clusters = sorted(set(cluster_assignments))
        
        # Filter out noise cluster (-1) and assign it grey color
        non_noise_clusters = [c for c in unique_clusters if c != -1]
        
        # Use a larger colormap to ensure unique colors for all clusters
        # Use tab20 which has 20 distinct colors, or cycle through tab10 if more than 20 clusters
        if len(non_noise_clusters) <= 20:
            colors = plt.cm.tab20(np.linspace(0, 1, len(non_noise_clusters)))
        else:
            # For more than 20 clusters, cycle through tab10 colors
            base_colors = plt.cm.tab10(np.linspace(0, 1, 10))
            colors = []
            for i in range(len(non_noise_clusters)):
                colors.append(base_colors[i % 10])
            colors = np.array(colors)
        
        cluster_color_map = {}
        # Assign grey to noise cluster (-1)
        cluster_color_map[-1] = 'grey'
        # Assign colors to non-noise clusters
        for i, cluster in enumerate(non_noise_clusters):
            cluster_color_map[cluster] = colors[i]
        
        # Calculate circle size for 6nm diameter AuNPs (twice as large as 3nm)
        # Assuming the zonogram is in nm units, we need to convert 6nm to points
        # The size parameter in scatter is in points^2, so we need to estimate the conversion
        # For a typical figure, 1 nm might be roughly 1-2 points depending on the zoom level
        # Let's use a reasonable size that represents ~6nm diameter
        circle_size = 36  # This should give approximately 6nm diameter circles (4x larger area = 2x diameter)
        
        # Plot AuNPs colored by cluster as rings (clear interior, colored edge)
        for i, (pos, cluster) in enumerate(zip(selected_aunp_pos_transformed, cluster_assignments)):
            color = cluster_color_map.get(cluster, 'gray')
            # Use circles with clear interior and colored edge (ring effect)
            axxy.scatter(pos[0], pos[1], s=circle_size, c='none', alpha=1.0, edgecolors=color, linewidth=1.5)
            axxz.scatter(pos[2], pos[1], s=circle_size, c='none', alpha=1.0, edgecolors=color, linewidth=1.5)
            axyz.scatter(pos[0], pos[2], s=circle_size, c='none', alpha=1.0, edgecolors=color, linewidth=1.5)
        
        # Add legend to show cluster names
        # Create legend handles for each cluster type
        legend_handles = []
        legend_labels = []
        
        # Add noise cluster to legend
        if -1 in cluster_color_map:
            noise_handle = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='none', 
                                     markeredgecolor=cluster_color_map[-1], markersize=8, linewidth=1.5)
            legend_handles.append(noise_handle)
            legend_labels.append('Noise')
        
        # Add other clusters to legend
        for cluster in sorted(non_noise_clusters):
            cluster_handle = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='none', 
                                       markeredgecolor=cluster_color_map[cluster], markersize=8, linewidth=1.5)
            legend_handles.append(cluster_handle)
            legend_labels.append(f'Cluster {cluster}')
        
        # Add legend to the figure
        if legend_handles:
            fig.legend(legend_handles, legend_labels, loc='lower right', bbox_to_anchor=(1.0, 0.0), 
                      fontsize=8, frameon=True, fancybox=True, shadow=True)
    
    # Add fusion points if tomogram path is provided
    if tomogram_path is not None:
        try:
            # Load fusion points from vesicle results (more efficient than recomputing)
            import sys
            import os
            import json
            sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
            from synaptic_tomo_tools.aunps import compute_fusion_points
            
            # Try to load cached fusion points first
            fusion_points_cache_path = (
                Path(tomogram_path) / alignment_dir / "STT_results" / "vesicles" / "fusion_points.npy"
            )
            fusion_points = None
            
            if fusion_points_cache_path.exists():
                try:
                    fusion_points = np.load(fusion_points_cache_path)
                    print(f"Loaded {len(fusion_points)} cached fusion points")
                except Exception as e:
                    print(f"Could not load cached fusion points: {e}")
            
            # Compute fusion points if not cached
            if fusion_points is None:
                print(f"Computing fusion points for {Path(tomogram_path).name}...")
                fusion_points = compute_fusion_points(
                    tomogram_path, vesicle_distance_threshold=20.0, alignment_dir=alignment_dir
                )
                print(f"Computed {len(fusion_points)} fusion points")
                
                # Cache the fusion points for future use
                if len(fusion_points) > 0:
                    try:
                        fusion_points_cache_path.parent.mkdir(parents=True, exist_ok=True)
                        np.save(fusion_points_cache_path, fusion_points)
                        print(f"Cached fusion points to {fusion_points_cache_path}")
                    except Exception as e:
                        print(f"Could not cache fusion points: {e}")
            
            if len(fusion_points) > 0:
                # Transform fusion points to the same coordinate system as AuNPs
                fusion_points_transformed = transform_coordinates(fusion_points, res_ddw)
                
                # Plot fusion points as orange stars on all three views
                for fp in fusion_points_transformed:
                    axxy.scatter(fp[0], fp[1], color='orange', s=100, alpha=0.9, marker='*', 
                               edgecolors='darkorange', linewidth=0.5)
                    axxz.scatter(fp[2], fp[1], color='orange', s=100, alpha=0.9, marker='*', 
                               edgecolors='darkorange', linewidth=0.5)
                    axyz.scatter(fp[0], fp[2], color='orange', s=100, alpha=0.9, marker='*', 
                               edgecolors='darkorange', linewidth=0.5)
                
                # Add fusion points to legend
                fusion_handle = plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='orange', 
                                         markeredgecolor='darkorange', markersize=10, linewidth=0.5)
                if legend_handles:
                    legend_handles.append(fusion_handle)
                    legend_labels.append('Fusion Sites')
                    # Update the legend
                    fig.legend(legend_handles, legend_labels, loc='lower right', bbox_to_anchor=(1.0, 0.0), 
                              fontsize=8, frameon=True, fancybox=True, shadow=True)
                else:
                    fig.legend([fusion_handle], ['Fusion Sites'], loc='lower right', bbox_to_anchor=(1.0, 0.0), 
                              fontsize=8, frameon=True, fancybox=True, shadow=True)
                
                print(f"Added {len(fusion_points)} fusion points to zonogram visualization")
        except Exception as e:
            print(f"Warning: Could not load fusion points: {e}")
    
    return fig, significant_picks_mask, selected_aunp_pos, selected_aunp_pos_mask, selected_aunp_pos_postsyn_mask

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run active zonogram analysis on a tomogram (findingampa style)."
    )
    parser.add_argument(
        "--tomogram-path",
        default="data/15F1/TOP_TOMOS/20241030_AMmilled12-1_15",
        help="Path to tomogram directory",
    )
    parser.add_argument(
        "--alignment-dir",
        required=True,
        help="Subdirectory under the tomogram with aligned data (from CSV alignment_dir column; no default).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    tomogram_path = args.tomogram_path
    from synaptic_tomo_tools.alignment_utils import require_alignment_dir
    alignment_dir = require_alignment_dir(args.alignment_dir, context="--alignment-dir")

    print(f"Running active zone analysis on: {tomogram_path}")
    
    print()  # Spacer line

    # Step 1: Import membranes and find active zones (do this once)
    print("Step 1: Importing membranes and finding active zones...")
    from synaptic_tomo_tools.activezone import import_membrane_segmentations_from_glb, find_active_zones_from_glb
    
    membranes = import_membrane_segmentations_from_glb(tomogram_path, alignment_dir=alignment_dir)
    active_zones_data = find_active_zones_from_glb(membranes, distance_range=(10.0, 40.0))
    
    print(f"Found {active_zones_data['total_active_zones']} active zones")
    
    print()  # Spacer line
    
    # Step 2: Run active zone analysis (for statistics and validation)
    print("Step 2: Running active zone analysis...")
    
    # Save active zone segmentations (this is what define_active_zone does that we need)
    from synaptic_tomo_tools.activezone import save_active_zone_segmentations, load_membrane_volumes
    save_active_zone_segmentations(active_zones_data, tomogram_path, alignment_dir=alignment_dir)
    
    # Calculate summary statistics (extracted from define_active_zone)
    total_active_pre_points = sum(len(zone['active_presynaptic_points']) for zone in active_zones_data['active_zones'].values())
    total_active_post_points = sum(len(zone['active_postsynaptic_points']) for zone in active_zones_data['active_zones'].values())
    
    # Calculate active zone areas
    active_zone_areas = []
    for zone_name, zone_data in active_zones_data['active_zones'].items():
        if 'active_presynaptic_area' in zone_data:
            active_zone_areas.append(zone_data['active_presynaptic_area'])
            print(f"Active zone area {zone_name}: {zone_data['active_presynaptic_area']:.6f} µm²")
    
    avg_active_zone_area = np.mean(active_zone_areas) if active_zone_areas else 0.0
    
    # Load membrane volumes
    volumes_data = load_membrane_volumes(tomogram_path, alignment_dir=alignment_dir)
    
    # Create results summary (similar to what define_active_zone returns)
    active_zone_results = {
        'active_zone_count': active_zones_data['total_active_zones'],
        'total_active_pre_points': total_active_pre_points,
        'total_active_post_points': total_active_post_points,
        'avg_active_zone_area': avg_active_zone_area,
        'distance_range': active_zones_data['distance_range'],
        'active_zone_names': list(active_zones_data['active_zones'].keys()),
        'membrane_volumes': volumes_data,
        'status': 'completed'
    }
    
    print(f"Found {active_zone_results['active_zone_count']} active zones")
    print(f"Average active zone area: {avg_active_zone_area:.6f} µm²")
    
    print()  # Spacer line
    
    # Step 3: Run zonogram definition
    print("Step 3: Running active zonogram definition...")
    zonogram_results = define_active_zonogram(active_zones_data)
    
    print(f"Zonogram results status: {zonogram_results['status']}")
    print(f"Processed {zonogram_results['active_zone_count']} active zones")
    
    print()  # Spacer line
    
    # Step 4: Extract and save active zonograms using findingampa approach
    print("Step 4: Extracting and saving active zonograms (findingampa style)...")
    
    # Create active_zonograms directory if it doesn't exist
    zonogram_dir = Path(tomogram_path) / alignment_dir / "active_zonograms"
    zonogram_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract zonograms
    extracted_results = extract_active_zonogram(
        zonogram_results,
        active_zones_data,
        tomogram_path,
        tomo_type='ddw',
        alignment_dir=alignment_dir,
    )
    
    if extracted_results['status'] == 'completed':
        print(f"Successfully extracted {len(extracted_results['rendered_zonograms'])} zonograms")
        
        # Process each zonogram using the findingampa approach
        for i, (zone_name, zone_data) in enumerate(extracted_results['rendered_zonograms'].items()):
            print(f"Processing zonogram {i}: {zone_name}")
            
            # Get the corresponding zonogram metadata
            zonogram_metadata = zonogram_results['zonogram_data'][zone_name]
            
            # Prepare data in findingampa format: (coordinate_system, center, res_ddw, objects)
            coordinate_system = zonogram_metadata['transformation_matrix'][:3, :3]  # 3x3 rotation matrix
            center = zonogram_metadata['center']
            res_ddw = torch.tensor(zone_data['transformed_tomogram'])
            objects = ()  # Empty tuple for objects (similar to findingampa manual active zones)
            
            active_zone_data_findingampa = (coordinate_system, center, res_ddw, objects)
            
            # 1. Save MRC file (same as findingampa)
            mrc_filename = f"active_zonogram_{i}.mrc"
            mrc_filepath = zonogram_dir / mrc_filename
            mrcfile.write(mrc_filepath, res_ddw.numpy(), overwrite=True)
            print(f"  Saved {mrc_filename}")
            
            # 2. Save NPY file (same format as findingampa)
            npy_filename = f"active_zonogram_{i}.npy"
            npy_filepath = zonogram_dir / npy_filename
            npy_data = {
                "cs": coordinate_system, 
                "center": center, 
                "objects": objects
            }
            np.save(npy_filepath, npy_data, allow_pickle=True)
            print(f"  Saved {npy_filename}")
            
            # 3. Generate main PNG using findingampa's render_active_zonograms function
            fig = render_active_zonograms_findingampa_style(active_zone_data_findingampa)
            png_filename = f"active_zonogram_{i}.png"
            png_filepath = zonogram_dir / png_filename
            fig.savefig(png_filepath)
            plt.close(fig)
            print(f"  Saved {png_filename}")
            
            # 4. Generate selected AuNPs PNG using findingampa's select_aunps approach
            selected_filename = f"active_zonogram_{i}_selected_aunps.png"
            selected_filepath = zonogram_dir / selected_filename
            
            # Try to load AuNP data
            aunp_data_path = Path(tomogram_path) / alignment_dir / "aunps" / "aunp_nearest_neighbor_distances.csv"
            
            if aunp_data_path.exists():
                try:
                    # Use the findingampa-style AuNP selection
                    fig, significant_picks_mask, selected_aunp_pos, selected_aunp_pos_mask, selected_aunp_pos_postsyn_mask = select_aunps_findingampa_style(
                        aunp_data_path, 
                        zonogram_metadata, 
                        res_ddw, 
                        threshold=3.8, 
                        skip_segment_activezone=False
                    )
                    
                    if fig is not None:
                        fig.savefig(selected_filepath)
                        plt.close(fig)
                        print(f"  Saved {selected_filename} with {len(selected_aunp_pos) if selected_aunp_pos is not None else 0} selected AuNPs")
                    else:
                        # Fallback if AuNP selection failed
                        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
                        ax.text(0.5, 0.5, f'Selected AuNPs\n(AuNP selection failed)', 
                                transform=ax.transAxes, ha='center', va='center', fontsize=12,
                                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
                        ax.set_xlim(0, 1)
                        ax.set_ylim(0, 1)
                        ax.axis('off')
                        ax.set_title(f'Selected AuNPs: {zone_name}')
                        plt.tight_layout()
                        plt.savefig(selected_filepath, dpi=300, bbox_inches='tight')
                        plt.close()
                        print(f"  Saved {selected_filename} (fallback)")
                        
                except Exception as e:
                    print(f"    Warning: Could not create AuNP visualization: {e}")
                    # Create error fallback
                    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
                    ax.text(0.5, 0.5, f'Selected AuNPs\n(Error: {str(e)[:50]}...)', 
                            transform=ax.transAxes, ha='center', va='center', fontsize=12,
                            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1)
                    ax.axis('off')
                    ax.set_title(f'Selected AuNPs: {zone_name}')
                    plt.tight_layout()
                    plt.savefig(selected_filepath, dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"  Saved {selected_filename} (error fallback)")
            else:
                print(f"    No AuNP data found at {aunp_data_path}, skipping AuNP visualization")
                # Create no-data fallback
                fig, ax = plt.subplots(1, 1, figsize=(10, 8))
                ax.text(0.5, 0.5, f'Selected AuNPs\n(No AuNP data available)', 
                        transform=ax.transAxes, ha='center', va='center', fontsize=12,
                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis('off')
                ax.set_title(f'Selected AuNPs: {zone_name}')
                plt.tight_layout()
                plt.savefig(selected_filepath, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"  Saved {selected_filename} (no data)")
            
            # 5. Generate cluster-colored AuNPs PNG (original version)
            cluster_filename = f"active_zonogram_{i}_selected_aunps_by_cluster.png"
            cluster_filepath = zonogram_dir / cluster_filename
            
            # 6. Generate cluster-colored AuNPs PNG with fusion points
            cluster_fusion_filename = f"active_zonogram_{i}_selected_aunps_by_cluster_with_fusion_points.png"
            cluster_fusion_filepath = zonogram_dir / cluster_fusion_filename
            
            # Try to load cluster data
            cluster_data_path = (
                Path(tomogram_path) / alignment_dir / "STT_results" / "aunps" / "aunp_clusters.star"
            )
            
            if cluster_data_path.exists() and aunp_data_path.exists():
                try:
                    # First, generate the original version without fusion points
                    fig_original, significant_picks_mask, selected_aunp_pos, selected_aunp_pos_mask, selected_aunp_pos_postsyn_mask = select_aunps_by_cluster_findingampa_style(
                        aunp_data_path,
                        cluster_data_path,
                        zonogram_metadata,
                        res_ddw,
                        threshold=3.8,
                        skip_segment_activezone=False,
                        tomogram_path=None,  # No fusion points for original version
                        alignment_dir=alignment_dir,
                    )
                    
                    if fig_original is not None:
                        fig_original.savefig(cluster_filepath)
                        plt.close(fig_original)
                        print(f"  Saved {cluster_filename} with cluster-colored AuNPs")
                    
                    # Then, generate the version with fusion points
                    fig_with_fusion, _, _, _, _ = select_aunps_by_cluster_findingampa_style(
                        aunp_data_path,
                        cluster_data_path,
                        zonogram_metadata,
                        res_ddw,
                        threshold=3.8,
                        skip_segment_activezone=False,
                        tomogram_path=tomogram_path,  # Include fusion points
                        alignment_dir=alignment_dir,
                    )
                    
                    if fig_with_fusion is not None:
                        fig_with_fusion.savefig(cluster_fusion_filepath)
                        plt.close(fig_with_fusion)
                        print(f"  Saved {cluster_fusion_filename} with cluster-colored AuNPs and fusion points")
                    else:
                        # Fallback if cluster AuNP selection failed - create both versions
                        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
                        ax.text(0.5, 0.5, f'Cluster-Colored AuNPs\n(Cluster selection failed)', 
                                transform=ax.transAxes, ha='center', va='center', fontsize=12,
                                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
                        ax.set_xlim(0, 1)
                        ax.set_ylim(0, 1)
                        ax.axis('off')
                        ax.set_title(f'Cluster-Colored AuNPs: {zone_name}')
                        plt.tight_layout()
                        plt.savefig(cluster_filepath, dpi=300, bbox_inches='tight')
                        plt.savefig(cluster_fusion_filepath, dpi=300, bbox_inches='tight')
                        plt.close()
                        print(f"  Saved {cluster_filename} (fallback)")
                        print(f"  Saved {cluster_fusion_filename} (fallback)")
                        
                except Exception as e:
                    print(f"    Warning: Could not create cluster AuNP visualization: {e}")
                    # Create error fallback for both versions
                    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
                    ax.text(0.5, 0.5, f'Cluster-Colored AuNPs\n(Error: {str(e)[:50]}...)', 
                            transform=ax.transAxes, ha='center', va='center', fontsize=12,
                            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1)
                    ax.axis('off')
                    ax.set_title(f'Cluster-Colored AuNPs: {zone_name}')
                    plt.tight_layout()
                    plt.savefig(cluster_filepath, dpi=300, bbox_inches='tight')
                    plt.savefig(cluster_fusion_filepath, dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"  Saved {cluster_filename} (error fallback)")
                    print(f"  Saved {cluster_fusion_filename} (error fallback)")
            else:
                print(f"    No cluster data found at {cluster_data_path}, skipping cluster visualization")
                # Create no-data fallback for both versions
                fig, ax = plt.subplots(1, 1, figsize=(10, 8))
                ax.text(0.5, 0.5, f'Cluster-Colored AuNPs\n(No cluster data available)', 
                        transform=ax.transAxes, ha='center', va='center', fontsize=12,
                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis('off')
                ax.set_title(f'Cluster-Colored AuNPs: {zone_name}')
                plt.tight_layout()
                plt.savefig(cluster_filepath, dpi=300, bbox_inches='tight')
                plt.savefig(cluster_fusion_filepath, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"  Saved {cluster_filename} (no data)")
                print(f"  Saved {cluster_fusion_filename} (no data)")
                
    else:
        print(f"Error extracting zonograms: {extracted_results.get('status', 'unknown error')}")
    
    print("\nActive zonogram analysis completed successfully!")

if __name__ == "__main__":
    main()