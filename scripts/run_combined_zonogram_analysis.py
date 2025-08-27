#!/usr/bin/env python3
"""
Combined script to run both regular active zonogram analysis and mini zonogram analysis.
Eliminates redundancy by reusing membrane data and active zones.
"""

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
from scipy.spatial.distance import pdist, squareform

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from synaptic_tomo_tools.activezone import (
    define_active_zone, define_active_zonogram, extract_active_zonogram,
    import_membrane_segmentations_from_glb, find_active_zones_from_glb
)
from scipy.spatial import KDTree

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

def render_mini_zonogram_xy_only(active_zone_data, include_legend_space=False, extra_width_multiplier=1.5):
    """
    Render mini zonogram showing only the xy slice (top-left view from regular zonograms).
    If include_legend_space is True, creates a wider figure to accommodate legend on the right.
    """
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
    # Load AuNP data
    aunp_file = Path(tomogram_path) / "best_alignment" / "STT_results" / "aunps" / "aunp_clusters.star"
    if not aunp_file.exists():
        return []
    
    try:
        import starfile
        aunp_df = starfile.read(aunp_file)
        # Filter AuNPs by active zone
        aunp_df = aunp_df[aunp_df['active_zone'] == active_zone_id]
        aunp_positions = aunp_df[['faCoordinateX', 'faCoordinateY', 'faCoordinateZ']].values
    except:
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

def select_aunps_by_cluster_findingampa_style(active_zone_data, cluster_data, tomogram_path, active_zone_id=0, original_zone_data=None):
    """
    Select AuNPs by cluster for visualization using findingampa-style approach.
    Only includes AuNPs that belong to the specified active zone.
    """
    # Load cluster data
    cluster_file = Path(tomogram_path) / "best_alignment" / "STT_results" / "aunps" / "aunp_clusters.star"
    if not cluster_file.exists():
        return [], []
    
    try:
        import starfile
        cluster_df = starfile.read(cluster_file)
        # Filter AuNPs by active zone
        cluster_df = cluster_df[cluster_df['active_zone'] == active_zone_id]
        aunp_positions = cluster_df[['faCoordinateX', 'faCoordinateY', 'faCoordinateZ']].values
        cluster_assignments = cluster_df['aunp_cluster'].values
    except:
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

def create_mini_zonogram_for_cluster(cluster_data, cluster_id, tomogram_path, tomogram_activezonograms_dir, results_activezonograms_dir, active_zones_data, cluster_color_map, tomogram_name):
    """
    Create a mini zonogram centered on a specific small cluster.
    Uses the same transformation matrix calculation as regular active zonograms.
    Uses the same color scheme as the regular zonogram analysis.
    Saves files in both tomogram's STT_results/activezonograms and results/visualizations/activezonograms.
    """
    print(f"  Creating mini zonogram for cluster {cluster_id} with {len(cluster_data)} AuNPs")
    
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
        mini_filename_base = f"{tomogram_name}_mini_zonogram_cluster_{cluster_id}"
        
        # 1. Save MRC file to tomogram directory only
        mrc_filename = f"{mini_filename_base}.mrc"
        mrcfile.write(tomogram_activezonograms_dir / mrc_filename, transformed_tomogram.numpy(), overwrite=True)
        print(f"    Saved {mrc_filename} to tomogram directory")
        
        # 2. Save NPY file to tomogram directory only
        npy_filename = f"{mini_filename_base}.npy"
        npy_data = {
            "cs": coordinate_system, 
            "center": cluster_center, 
            "objects": (),
            "cluster_id": cluster_id,
            "aunp_count": len(cluster_data)
        }
        np.save(tomogram_activezonograms_dir / npy_filename, npy_data, allow_pickle=True)
        print(f"    Saved {npy_filename} to tomogram directory")
        
        # 3. Generate main PNG and save to both locations
        fig, axxy = render_mini_zonogram_xy_only(mini_zonogram_findingampa)
        png_filename = f"{mini_filename_base}.png"
        fig.savefig(tomogram_activezonograms_dir / png_filename)
        fig.savefig(results_activezonograms_dir / png_filename)
        plt.close(fig)
        print(f"    Saved {png_filename} to both locations")
        
        # 4. Generate AuNP visualization and save to both locations
        aunp_filename = f"{mini_filename_base}_aunps.png"
        
        # Create AuNP visualization
        fig, axxy = render_mini_zonogram_xy_only(mini_zonogram_findingampa)
        
        # Transform cluster AuNPs to mini zonogram coordinates using the same transformation as the tomogram
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
        
        if len(cluster_positions_transformed) > 0:
            # Plot AuNPs with cluster colors
            circle_size = 36  # 6nm diameter circles
            cluster_color = cluster_color_map.get(cluster_id, 'red')  # Default to red if cluster not in map
            axxy.scatter(cluster_positions_transformed[:,0], cluster_positions_transformed[:,1], 
                        s=circle_size, c='none', alpha=1.0, edgecolors=cluster_color, linewidth=1.5)
        
        fig.savefig(tomogram_activezonograms_dir / aunp_filename)
        fig.savefig(results_activezonograms_dir / aunp_filename)
        plt.close(fig)
        print(f"    Saved {aunp_filename} to both locations")
        
        # 4.5. Generate AuNP visualization with distance lines (< 20 nm) and save to both locations
        aunp_distances_filename = f"{mini_filename_base}_aunps_with_distances.png"
        
        # Create AuNP visualization with distance lines
        fig, axxy = render_mini_zonogram_xy_only(mini_zonogram_findingampa, include_legend_space=True)
        
        if len(cluster_positions_transformed) > 0:
            # Plot AuNPs with cluster colors
            circle_size = 36  # 6nm diameter circles
            cluster_color = cluster_color_map.get(cluster_id, 'red')  # Default to red if cluster not in map
            axxy.scatter(cluster_positions_transformed[:,0], cluster_positions_transformed[:,1], 
                        s=circle_size, c='none', alpha=1.0, edgecolors=cluster_color, linewidth=1.5)
            
            # Calculate distances between all pairs of AuNPs and draw lines for those < 20 nm apart
            
            # Calculate pairwise distances in the original coordinate system (nm)
            original_positions = cluster_data[['faCoordinateX', 'faCoordinateY', 'faCoordinateZ']].values
            distances = pdist(original_positions)
            distance_matrix = squareform(distances)
            
            # Define a set of distinct colors for the distance lines (no grey to avoid confusion with noise)
            line_colors = ['yellow', 'cyan', 'magenta', 'orange', 'lime', 'red', 'blue', 'green', 
                          'pink', 'purple', 'brown', 'olive', 'navy', 'teal', 'maroon', 'darkorange']
            
            # Find pairs of AuNPs that are less than 15 nm apart and collect distance info
            distance_pairs = []
            color_idx = 0
            
            for i in range(len(original_positions)):
                for j in range(i+1, len(original_positions)):
                    if distance_matrix[i, j] < 15.0:  # Less than 15 nm apart
                        distance_pairs.append({
                            'i': i, 'j': j, 
                            'distance': distance_matrix[i, j],
                            'color': line_colors[color_idx % len(line_colors)]
                        })
                        color_idx += 1
            
            # Draw lines for each distance pair
            for pair in distance_pairs:
                # Get the transformed positions for these two AuNPs
                pos1 = cluster_positions_transformed[pair['i']]
                pos2 = cluster_positions_transformed[pair['j']]
                
                # Draw a line between them with unique color
                axxy.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], 
                         color=pair['color'], linewidth=1.5, alpha=0.8)
            
            # Create distance legend in top right corner
            if distance_pairs:
                legend_text = []
                legend_colors = []
                for idx, pair in enumerate(distance_pairs):
                    legend_text.append(f"AuNP {pair['i']+1}-{pair['j']+1}: {pair['distance']:.1f}nm")
                    legend_colors.append(pair['color'])
                
                # Create custom legend handles with matching colors
                from matplotlib.lines import Line2D
                legend_handles = [Line2D([0], [0], color=color, linewidth=2) for color in legend_colors]
                
                # Add legend to the right of the figure (outside the plot area)
                axxy.legend(legend_handles, legend_text, loc='center left', 
                           fontsize=6, frameon=True, fancybox=True, shadow=True,
                           bbox_to_anchor=(1.05, 0.5), framealpha=0.9)
        
        fig.savefig(tomogram_activezonograms_dir / aunp_distances_filename, bbox_inches='tight')
        fig.savefig(results_activezonograms_dir / aunp_distances_filename, bbox_inches='tight')
        plt.close(fig)
        print(f"    Saved {aunp_distances_filename} to both locations")
        
        # 5. Generate three-panel comparison PNG: mini zonogram, mini zonogram with AuNPs, and mini zonogram with distances
        comparison_filename = f"{mini_filename_base}_comparison.png"
        
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
            # Calculate pairwise distances in the original coordinate system (nm)
            original_positions = cluster_data[['faCoordinateX', 'faCoordinateY', 'faCoordinateZ']].values
            distances = pdist(original_positions)
            distance_matrix = squareform(distances)
            
            # Define a set of distinct colors for the distance lines (no grey to avoid confusion with noise)
            line_colors = ['yellow', 'cyan', 'magenta', 'orange', 'lime', 'red', 'blue', 'green', 
                          'pink', 'purple', 'brown', 'olive', 'navy', 'teal', 'maroon', 'darkorange']
            
            # Find pairs of AuNPs that are less than 15 nm apart and collect distance info
            distance_pairs = []
            color_idx = 0
            
            for i in range(len(original_positions)):
                for j in range(i+1, len(original_positions)):
                    if distance_matrix[i, j] < 15.0:  # Less than 15 nm apart
                        distance_pairs.append({
                            'i': i, 'j': j, 
                            'distance': distance_matrix[i, j],
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
                    legend_text_distances.append(f"AuNP {pair['i']+1}-{pair['j']+1}: {pair['distance']:.1f}nm")
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
        combined_img.save(tomogram_activezonograms_dir / comparison_filename)
        combined_img.save(results_activezonograms_dir / comparison_filename)
        print(f"    Saved {comparison_filename} to both locations")
        
        return True
        
    except Exception as e:
        print(f"    Error creating mini zonogram for cluster {cluster_id}: {e}")
        return False

def main():
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run combined zonogram analysis on a tomogram')
    parser.add_argument('--tomogram-path', type=str, required=True, help='Path to the tomogram directory')
    parser.add_argument('--output-dir', type=str, default='results/visualizations/active_zonograms', help='Output directory for results')
    parser.add_argument('--tomogram-name', type=str, required=True, help='Name of the tomogram for file naming')
    
    args = parser.parse_args()
    
    tomogram_path = args.tomogram_path
    output_dir = args.output_dir
    tomogram_name = args.tomogram_name
    
    print(f"Running combined zonogram analysis on: {tomogram_path}")
    print(f"Output directory: {output_dir}")
    print(f"Tomogram name: {tomogram_name}")
    
    print()  # Spacer line
    
    # Step 1: Load membrane data and active zones (shared between both analyses)
    print("Step 1: Loading membrane data and active zones...")
    try:
        # Load membrane data using GLB method
        membrane_data = import_membrane_segmentations_from_glb(tomogram_path)
        print(f"Loaded membrane data")
        
        # Find active zones
        active_zones_data = find_active_zones_from_glb(membrane_data, distance_range=(10.0, 40.0))
        print(f"Found {len(active_zones_data['active_zones'])} active zones")
        
    except Exception as e:
        print(f"Error loading membrane data or active zones: {e}")
        return
    
    print()  # Spacer line
    
    # Step 2: Regular Active Zonogram Analysis
    print("Step 2: Running regular active zonogram analysis...")
    try:
        # Define active zonograms
        zonogram_results = define_active_zonogram(active_zones_data)
        
        if zonogram_results['status'] == 'completed':
            print(f"Defined {zonogram_results['active_zone_count']} active zonograms")
            
            # Extract and save zonograms
            extracted_results = extract_active_zonogram(zonogram_results, active_zones_data, tomogram_path)
            
            if extracted_results and 'rendered_zonograms' in extracted_results:
                # Create output directories for both locations
                # 1. In results/visualizations/activezonograms directory
                results_activezonograms_dir = Path(output_dir) / "visualizations" / "activezonograms"
                results_activezonograms_dir.mkdir(parents=True, exist_ok=True)
                
                # 2. In tomogram's STT_results/activezonograms directory
                tomogram_activezonograms_dir = Path(tomogram_path) / "best_alignment" / "STT_results" / "activezonograms"
                tomogram_activezonograms_dir.mkdir(parents=True, exist_ok=True)
                
                for zone_name, zone_data in extracted_results['rendered_zonograms'].items():
                    # Get the original zonogram data with transformation matrix and extent
                    original_zone_data = zonogram_results['zonogram_data'][zone_name]
                    
                    # Create zonogram data in findingampa format
                    zonogram_findingampa = (np.eye(3), np.zeros(3), torch.tensor(zone_data['transformed_tomogram']), ())
                    
                    # Save MRC file to tomogram directory only
                    mrc_filename = f"{tomogram_name}_active_zonogram_{zone_name}.mrc"
                    mrcfile.write(tomogram_activezonograms_dir / mrc_filename, zone_data['transformed_tomogram'], overwrite=True)
                    print(f"  Saved {mrc_filename} to tomogram directory")
                    
                    # Save NPY file to tomogram directory only
                    npy_filename = f"{tomogram_name}_active_zonogram_{zone_name}.npy"
                    npy_data = {
                        "cs": np.eye(3),
                        "center": np.zeros(3),
                        "objects": ()
                    }
                    np.save(tomogram_activezonograms_dir / npy_filename, npy_data, allow_pickle=True)
                    print(f"  Saved {npy_filename} to tomogram directory")
                    
                    # Generate main PNG and save to both locations
                    fig = render_active_zonograms_findingampa_style(zonogram_findingampa)
                    png_filename = f"{tomogram_name}_active_zonogram_{zone_name}.png"
                    fig.savefig(results_activezonograms_dir / png_filename)
                    fig.savefig(tomogram_activezonograms_dir / png_filename)
                    plt.close(fig)
                    print(f"  Saved {png_filename} to both locations")
                    
                    # Extract active zone ID from zone_name (e.g., "active_zone_pre1_post1" -> 0, "active_zone_pre2_post1" -> 1)
                    # For now, we'll use a simple mapping since the zone names don't contain numeric IDs
                    if 'pre1_post1' in zone_name:
                        active_zone_id = 0
                    elif 'pre2_post1' in zone_name:
                        active_zone_id = 0  # Both zones map to active_zone 0 in the data
                    else:
                        active_zone_id = 0  # Default fallback
                    
                    # Generate AuNP visualization
                    selected_aunps = select_aunps_findingampa_style(zonogram_findingampa, None, tomogram_path, active_zone_id, original_zone_data)
                    if len(selected_aunps) > 0:
                        fig = render_active_zonograms_findingampa_style(zonogram_findingampa)
                        (axxy, axxz, axyz) = fig.get_axes()
                        
                        circle_size = 36  # 6nm diameter circles
                        axxy.scatter(selected_aunps[:,0], selected_aunps[:,1], s=circle_size, c='none', alpha=1.0, edgecolors='red', linewidth=1.5)
                        axxz.scatter(selected_aunps[:,2], selected_aunps[:,1], s=circle_size, c='none', alpha=1.0, edgecolors='red', linewidth=1.5)
                        axyz.scatter(selected_aunps[:,0], selected_aunps[:,2], s=circle_size, c='none', alpha=1.0, edgecolors='red', linewidth=1.5)
                        
                        aunp_filename = f"{tomogram_name}_active_zonogram_{zone_name}_selected_aunps.png"
                        fig.savefig(results_activezonograms_dir / aunp_filename)
                        fig.savefig(tomogram_activezonograms_dir / aunp_filename)
                        plt.close(fig)
                        print(f"  Saved {aunp_filename} to both locations")
                    
                    # Generate cluster-colored AuNP visualization
                    selected_aunps, cluster_assignments = select_aunps_by_cluster_findingampa_style(zonogram_findingampa, None, tomogram_path, active_zone_id, original_zone_data)
                    if len(selected_aunps) > 0:
                        fig = render_active_zonograms_findingampa_style(zonogram_findingampa)
                        (axxy, axxz, axyz) = fig.get_axes()
                        
                        unique_clusters = sorted(set(cluster_assignments))
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
                        cluster_color_map[-1] = 'grey'
                        for i, cluster in enumerate(non_noise_clusters):
                            cluster_color_map[cluster] = colors[i]
                        
                        circle_size = 36
                        for i, (pos, cluster) in enumerate(zip(selected_aunps, cluster_assignments)):
                            color = cluster_color_map.get(cluster, 'gray')
                            axxy.scatter(pos[0], pos[1], s=circle_size, c='none', alpha=1.0, edgecolors=color, linewidth=1.5)
                            axxz.scatter(pos[2], pos[1], s=circle_size, c='none', alpha=1.0, edgecolors=color, linewidth=1.5)
                            axyz.scatter(pos[0], pos[2], s=circle_size, c='none', alpha=1.0, edgecolors=color, linewidth=1.5)
                        
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
                        if legend_handles:
                            fig.legend(legend_handles, legend_labels, loc='lower right', bbox_to_anchor=(1.0, 0.0), 
                                      fontsize=8, frameon=True, fancybox=True, shadow=True)
                        
                        cluster_filename = f"{tomogram_name}_active_zonogram_{zone_name}_selected_aunps_by_cluster.png"
                        fig.savefig(results_activezonograms_dir / cluster_filename)
                        fig.savefig(tomogram_activezonograms_dir / cluster_filename)
                        plt.close(fig)
                        print(f"  Saved {cluster_filename} to both locations")
        else:
            print("No active zonograms found")
            
    except Exception as e:
        print(f"Error in regular zonogram analysis: {e}")
    
    print()  # Spacer line
    
    # Step 3: Check if AuNP analysis was completed
    print("Step 3: Checking AuNP analysis status...")
    
    # Check for required AuNP analysis files
    aunp_analysis_path = Path(tomogram_path) / "best_alignment" / "STT_results" / "aunps"
    cluster_data_path = aunp_analysis_path / "aunp_clusters.star"
    
    if not aunp_analysis_path.exists():
        print(f"Warning: AuNP analysis directory not found at {aunp_analysis_path}")
        print("Skipping zonogram analysis - AuNP analysis must be completed first.")
        return
    
    if not cluster_data_path.exists():
        print(f"Warning: AuNP cluster data not found at {cluster_data_path}")
        print("Skipping zonogram analysis - AuNP analysis must be completed first.")
        return
    
    print("AuNP analysis files found. Proceeding with zonogram analysis.")
    
    # Step 4: Mini Zonogram Analysis
    print("Step 4: Running mini zonogram analysis...")
    try:
        # Load cluster data
        
        import starfile
        cluster_df = starfile.read(cluster_data_path)
        print(f"Loaded cluster data with {len(cluster_df)} AuNPs")
        
        # Identify small clusters (excluding noise cluster -1)
        cluster_counts = cluster_df['aunp_cluster'].value_counts()
        small_clusters = cluster_counts[(cluster_counts < 11) & (cluster_counts.index != -1)]
        
        print(f"Found {len(small_clusters)} small clusters (< 11 AuNPs, excluding noise):")
        for cluster_id, count in small_clusters.items():
            print(f"  Cluster {cluster_id}: {count} AuNPs")
        
        if len(small_clusters) > 0:
            # Create mini zonograms directories in both locations
            # 1. In tomogram's STT_results directory
            tomogram_activezonograms_dir = Path(tomogram_path) / "best_alignment" / "STT_results" / "activezonograms"
            tomogram_activezonograms_dir.mkdir(parents=True, exist_ok=True)
            
            # 2. In results/visualizations directory
            results_activezonograms_dir = Path(output_dir) / "visualizations" / "activezonograms"
            results_activezonograms_dir.mkdir(parents=True, exist_ok=True)
            
            # Create cluster color map (same as regular zonogram analysis)
            unique_clusters = sorted(set(cluster_df['aunp_cluster'].values))
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
            cluster_color_map[-1] = 'grey'
            for i, cluster in enumerate(non_noise_clusters):
                cluster_color_map[cluster] = colors[i]
            
            # Generate mini zonograms for each small cluster
            successful_mini_zonograms = 0
            
            for cluster_id in small_clusters.index:
                # Get AuNPs for this cluster
                cluster_data = cluster_df[cluster_df['aunp_cluster'] == cluster_id]
                
                # Create mini zonogram
                success = create_mini_zonogram_for_cluster(
                    cluster_data, cluster_id, tomogram_path, tomogram_activezonograms_dir, results_activezonograms_dir, active_zones_data, cluster_color_map, tomogram_name
                )
                
                if success:
                    successful_mini_zonograms += 1
            
            print(f"Successfully created {successful_mini_zonograms} mini zonograms out of {len(small_clusters)} small clusters")
        else:
            print("No small clusters found (excluding noise).")
            
    except Exception as e:
        print(f"Error in mini zonogram analysis: {e}")
    
    print()  # Spacer line
    
    print("Combined zonogram analysis completed successfully!")

if __name__ == "__main__":
    main()
