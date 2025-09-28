#!/usr/bin/env python3
"""
Script to generate mini active zonograms for small clusters (< 11 AuNPs).
Based on run_zonogram.py but creates smaller, cluster-specific zonograms.
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

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from synaptic_tomo_tools.activezone import (
    define_active_zone, define_active_zonogram, extract_active_zonogram,
    import_membrane_segmentations_from_glb, find_active_zones_from_glb
)
from scipy.spatial import KDTree

def render_mini_zonogram_xy_only(active_zone_data):
    """
    Render mini zonogram showing only the xy slice (top-left view from regular zonograms).
    """
    res_ddw = active_zone_data[2]
    # Use square aspect ratio for mini zonogram
    fig_size = max(res_ddw.shape[1], res_ddw.shape[2]) / 50
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

def create_mini_zonogram_for_cluster(cluster_data, cluster_id, tomogram_data, tomogram_path, zonogram_dir, membrane_data, active_zones_data):
    """
    Create a mini zonogram centered on a specific small cluster.
    Uses the same transformation matrix calculation as regular active zonograms.
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
        
        # Save mini zonogram files
        mini_filename_base = f"mini_zonogram_cluster_{cluster_id}"
        
        # 1. Save MRC file
        mrc_filename = f"{mini_filename_base}.mrc"
        mrc_filepath = zonogram_dir / mrc_filename
        mrcfile.write(mrc_filepath, transformed_tomogram.numpy(), overwrite=True)
        print(f"    Saved {mrc_filename}")
        
        # 2. Save NPY file
        npy_filename = f"{mini_filename_base}.npy"
        npy_filepath = zonogram_dir / npy_filename
        npy_data = {
            "cs": coordinate_system, 
            "center": cluster_center, 
            "objects": (),
            "cluster_id": cluster_id,
            "aunp_count": len(cluster_data)
        }
        np.save(npy_filepath, npy_data, allow_pickle=True)
        print(f"    Saved {npy_filename}")
        
        # 3. Generate main PNG
        fig, axxy = render_mini_zonogram_xy_only(mini_zonogram_findingampa)
        png_filename = f"{mini_filename_base}.png"
        png_filepath = zonogram_dir / png_filename
        fig.savefig(png_filepath)
        plt.close(fig)
        print(f"    Saved {png_filename}")
        
        # 4. Generate AuNP visualization
        aunp_filename = f"{mini_filename_base}_aunps.png"
        aunp_filepath = zonogram_dir / aunp_filename
        
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
        valid_mask = np.all(cluster_positions_transformed >= 0, axis=1) & np.all(cluster_positions_transformed < extent, axis=1)
        cluster_positions_transformed = cluster_positions_transformed[valid_mask]
        
        if len(cluster_positions_transformed) > 0:
            # Plot AuNPs as red rings
            circle_size = 36  # 6nm diameter circles
            axxy.scatter(cluster_positions_transformed[:,0], cluster_positions_transformed[:,1], 
                        s=circle_size, c='none', alpha=1.0, edgecolors='red', linewidth=1.5)
        
        fig.savefig(aunp_filepath)
        plt.close(fig)
        print(f"    Saved {aunp_filename}")
        
        # 5. Generate side-by-side comparison PNG using exact same format as individual PNGs
        comparison_filename = f"{mini_filename_base}_comparison.png"
        comparison_filepath = zonogram_dir / comparison_filename
        
        # Generate left panel (mini zonogram only) using the same function
        fig_left, axxy_left = render_mini_zonogram_xy_only(mini_zonogram_findingampa)
        
        # Generate right panel (mini zonogram with AuNPs) using the same function
        fig_right, axxy_right = render_mini_zonogram_xy_only(mini_zonogram_findingampa)
        
        # Add AuNPs to right panel
        if len(cluster_positions_transformed) > 0:
            circle_size = 36  # 6nm diameter circles
            axxy_right.scatter(cluster_positions_transformed[:,0], cluster_positions_transformed[:,1], 
                              s=circle_size, c='none', alpha=1.0, edgecolors='red', linewidth=1.5)
        
        # Save individual figures to memory
        import io
        left_buffer = io.BytesIO()
        fig_left.savefig(left_buffer, format='png', bbox_inches='tight', pad_inches=0)
        left_buffer.seek(0)
        
        right_buffer = io.BytesIO()
        fig_right.savefig(right_buffer, format='png', bbox_inches='tight', pad_inches=0)
        right_buffer.seek(0)
        
        # Close the individual figures
        plt.close(fig_left)
        plt.close(fig_right)
        
        # Load the images and combine them
        from PIL import Image
        left_img = Image.open(left_buffer)
        right_img = Image.open(right_buffer)
        
        # Create combined image with white background, spacer, and border
        spacer_width = 50  # 50 pixel white spacer
        border_width = 20  # 20 pixel white border around entire image
        total_width = left_img.width + spacer_width + right_img.width + (2 * border_width)
        max_height = max(left_img.height, right_img.height) + (2 * border_width)
        
        combined_img = Image.new('RGB', (total_width, max_height), 'white')
        combined_img.paste(left_img, (border_width, border_width))
        combined_img.paste(right_img, (left_img.width + spacer_width + border_width, border_width))
        
        # Save the combined image
        combined_img.save(comparison_filepath)
        print(f"    Saved {comparison_filename}")
        
        return True
        
    except Exception as e:
        print(f"    Error creating mini zonogram for cluster {cluster_id}: {e}")
        return False

def main():
    # Tomogram path
    tomogram_path = "data/15F1/TOP_TOMOS/20241030_AMmilled12-1_15"
    
    print(f"Running mini zonogram analysis on: {tomogram_path}")
    
    print()  # Spacer line
    
    # Step 1: Load membrane data and active zones
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
    
    # Step 2: Load cluster data
    print("Step 2: Loading cluster data...")
    cluster_data_path = Path(tomogram_path) / "best_alignment" / "STT_results" / "aunps" / "aunp_clusters.star"
    
    if not cluster_data_path.exists():
        print(f"Error: Cluster data not found at {cluster_data_path}")
        return
    
    try:
        import starfile
        cluster_df = starfile.read(cluster_data_path)
        print(f"Loaded cluster data with {len(cluster_df)} AuNPs")
    except Exception as e:
        print(f"Error loading cluster data: {e}")
        return
    
    print()  # Spacer line
    
    # Step 3: Identify small clusters (excluding noise cluster -1)
    print("Step 3: Identifying small clusters...")
    cluster_counts = cluster_df['aunp_cluster'].value_counts()
    # Filter out noise cluster (-1) and get only small clusters
    small_clusters = cluster_counts[(cluster_counts < 11) & (cluster_counts.index != -1)]
    
    print(f"Found {len(small_clusters)} small clusters (< 11 AuNPs, excluding noise):")
    for cluster_id, count in small_clusters.items():
        print(f"  Cluster {cluster_id}: {count} AuNPs")
    
    if len(small_clusters) == 0:
        print("No small clusters found (excluding noise). Exiting.")
        return
    
    print()  # Spacer line
    
    # Step 4: Create mini zonograms directory
    print("Step 4: Creating mini zonograms...")
    mini_zonogram_dir = Path(tomogram_path) / "best_alignment" / "active_zonograms" / "aunp_cluster_mini_active_zonograms"
    mini_zonogram_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 5: Generate mini zonograms for each small cluster
    successful_mini_zonograms = 0
    
    for cluster_id in small_clusters.index:
        # Get AuNPs for this cluster
        cluster_data = cluster_df[cluster_df['aunp_cluster'] == cluster_id]
        
        # Create mini zonogram
        success = create_mini_zonogram_for_cluster(
            cluster_data, cluster_id, None, tomogram_path, mini_zonogram_dir, membrane_data, active_zones_data
        )
        
        if success:
            successful_mini_zonograms += 1
    
    print()  # Spacer line
    
    print(f"Successfully created {successful_mini_zonograms} mini zonograms out of {len(small_clusters)} small clusters")
    print("\nMini zonogram analysis completed successfully!")

if __name__ == "__main__":
    main()
