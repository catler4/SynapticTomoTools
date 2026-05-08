# src/synaptic_tomo_tools/aunps.py

from typing import List
import pandas as pd
import numpy as np
from scipy.spatial import KDTree, cKDTree
from sklearn.cluster import DBSCAN
from pathlib import Path
import starfile
from .activezone import import_membrane_segmentations
from .activezone import import_active_zone_segmentations
from datetime import datetime
import json
from .vesicles import import_presynaptic_membranes_and_active_zones
import re

# CSV export functions removed - now handled by ResultsManager

def calculate_packing_density_using_sliding_cylinder(
    active_zone: dict,
    active_zonogram: dict,
    aunp_coordinates: np.ndarray,
    cylinder_radius: float = 25.0,
    receptor_crosssection_nm_squared: float = 122.0,
    aunps_per_receptor: float = 2.0,
    vertex_sampling_step: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate packing density of AuNPs (receptors) on postsynaptic membrane using sliding cylinder method.
    
    Args:
        active_zone: Dictionary containing 'active_postsynaptic_mesh' with vertices and vertex_normals
        active_zonogram: Dictionary with zonogram data (not currently used but kept for API consistency)
        aunp_coordinates: Array of shape (N, 3) with AuNP 3D coordinates in nm
        cylinder_radius: Radius of the sliding cylinder probe in nm (default: 25.0)
        receptor_crosssection_nm_squared: Cross-sectional area of a single receptor in nm² (default: 122.0)
        aunps_per_receptor: AuNPs per receptor (e.g. 2 for full receptor, 1 for dimer half) (default: 2.0)
        vertex_sampling_step: Sample every Nth mesh vertex (1=all, 50=every 50th) (default: 50)
    
    Returns:
        Tuple of (v_array, packing_coefficient) where:
        - v_array: Subset of postsynaptic mesh vertices used (every vertex_sampling_step-th)
        - packing_coefficient: Calculated packing density values for each vertex
    """
    ps_mesh = active_zone['active_postsynaptic_mesh']
    subset_vertices = ps_mesh.vertices[::vertex_sampling_step]
    
    tree = cKDTree(ps_mesh.vertices)
    # Generate a cKDTree of aunps
    tree_aunps = cKDTree(aunp_coordinates)
    # Iterate over vertices in ps_mesh_simplified and find all vertices in ps_mesh within cylinder_radius and average their normals
    num_aunps_at_vertex = []
    for v in subset_vertices:
        idxs = tree.query_ball_point(v, cylinder_radius)
        normals = ps_mesh.vertex_normals[idxs]
        avg_normal = np.mean(normals, axis=0)
        avg_normal /= np.linalg.norm(avg_normal)
        # Find all aunps within cylinder_radius of line through v in direction of avg_normal
        line_points = np.array([v + t * avg_normal for t in np.linspace(0, 50, 100)])
        idxs_aunps = tree_aunps.query_ball_point(line_points, cylinder_radius)
        # Generate list of unique inds in idxs_aunps
        unique_idxs_aunps = set()
        for idx_list in idxs_aunps:
            unique_idxs_aunps.update(idx_list)
        num_aunps_at_vertex.append(len(unique_idxs_aunps))

    v_array = np.array(subset_vertices)
    area_of_circle = np.pi * (cylinder_radius ** 2)  # Area = πr²
    packing_coefficient = ((np.array(num_aunps_at_vertex) / aunps_per_receptor) * receptor_crosssection_nm_squared) / area_of_circle

    # Mask vertices near mesh boundary to avoid edge artifacts (cylinder extends past boundary)
    try:
        import trimesh
        boundary_edge_indices = trimesh.grouping.group_rows(ps_mesh.edges_sorted, require_count=1)
        if len(boundary_edge_indices) > 0:
            boundary_edges = ps_mesh.edges_sorted[boundary_edge_indices]
            boundary_vertex_indices = np.unique(boundary_edges.flatten())
            boundary_vertex_coords = ps_mesh.vertices[boundary_vertex_indices]
            boundary_tree = cKDTree(boundary_vertex_coords)
            dist_to_boundary, _ = boundary_tree.query(subset_vertices, k=1)
            # Set packing to NaN where cylinder would extend past mesh boundary
            edge_mask = dist_to_boundary < cylinder_radius
            packing_coefficient = np.asarray(packing_coefficient, dtype=float)
            packing_coefficient[edge_mask] = np.nan
    except Exception:
        # If boundary detection fails (e.g. watertight mesh), skip masking
        pass

    return (v_array, packing_coefficient)

def compute_fusion_points(tomogram_path, vesicle_distance_threshold=20.0, fusion_point_threshold=20.0, alignment_dir: str = "best_alignment"):
    """
    For each vesicle within 20 nm of the presynaptic active zone, compute the putative fusion point as the average
    of all presynaptic active zone points within 20 nm of any vesicle point. Supports multiple active zones.
    Returns a list of fusion points (np.ndarray shape (N, 3)).
    """
    # Load vesicle results
    vesicles_file = Path(tomogram_path) / alignment_dir / "STT_results" / "vesicles" / "vesicle_results.json"
    if not vesicles_file.exists():
        print(f"No vesicle results found: {vesicles_file}")
        return []
    with open(vesicles_file, 'r') as f:
        vesicle_data = json.load(f)
    vesicles = vesicle_data['vesicles']
    # Load presynaptic membranes and active zones
    membrane_active_zone_pairs = import_presynaptic_membranes_and_active_zones(tomogram_path, alignment_dir=alignment_dir)
    fusion_points = []
    for vesicle in vesicles:
        # Only consider vesicles within 10 nm of the presynaptic active zone
        if vesicle.get('distance_to_az', 0.0) > vesicle_distance_threshold:
            continue
        vesicle_points = np.array(vesicle['coordinates'])
        # Find closest presynaptic membrane and its active zone points
        membrane_name = vesicle.get('closest_membrane', None)
        if not membrane_name or membrane_name not in membrane_active_zone_pairs:
            continue
        active_zone_points = membrane_active_zone_pairs[membrane_name]['active_zone_points']
        if active_zone_points is None or len(active_zone_points) == 0:
            continue
        # For each vesicle point, find all active zone points within fusion_point_threshold
        tree = KDTree(active_zone_points)
        close_points = []
        for pt in vesicle_points:
            idxs = tree.query_ball_point(pt, r=fusion_point_threshold)
            if idxs:
                close_points.extend(active_zone_points[idxs])
        if close_points:
            fusion_point = np.mean(np.vstack(close_points), axis=0)
            fusion_points.append(fusion_point)
    if fusion_points:
        return np.vstack(fusion_points)
    else:
        return np.zeros((0, 3))


def compute_aunp_distance_histograms_per_vesicle(tomogram_path, aunp_coords, vesicle_distance_threshold=20.0, 
                                                  fusion_point_threshold=20.0, max_distance=500.0, bin_width=5.0,
                                                  fusing_only=False, fusing_perimeter_threshold=1.0,
                                                  alignment_dir: str = "best_alignment"):
    """
    For each vesicle within vesicle_distance_threshold of the presynaptic active zone:
    1. Compute the putative fusion point
    2. Calculate distances from all AuNPs to this fusion point
    3. Bin the AuNPs into distance histogram bins
    
    Args:
        tomogram_path: Path to tomogram directory
        aunp_coords: Array of AuNP coordinates (N, 3)
        vesicle_distance_threshold: Max distance from vesicle to AZ to be considered "close" (default 20 nm)
        fusion_point_threshold: Distance threshold for computing fusion points (default 20 nm)
        max_distance: Maximum distance for histogram bins (default 500 nm)
        bin_width: Width of histogram bins (default 5 nm)
        fusing_only: If True, only include fusing vesicles (perimeter within fusing_perimeter_threshold)
        fusing_perimeter_threshold: Distance threshold for fusing vesicles (default 1.0 nm)
        
    Returns:
        DataFrame with vesicle info and AuNP distance histogram bins
    """
    # Get tomogram name
    tomogram_name = Path(tomogram_path).name
    
    # Load vesicle results
    vesicles_file = Path(tomogram_path) / alignment_dir / "STT_results" / "vesicles" / "vesicle_results.json"
    if not vesicles_file.exists():
        print(f"No vesicle results found: {vesicles_file}")
        return pd.DataFrame()
    
    with open(vesicles_file, 'r') as f:
        vesicle_data = json.load(f)
    vesicles = vesicle_data['vesicles']
    
    # Load presynaptic membranes and active zones
    membrane_active_zone_pairs = import_presynaptic_membranes_and_active_zones(tomogram_path, alignment_dir=alignment_dir)
    
    # Create histogram bin edges
    bin_edges = np.arange(0, max_distance + bin_width, bin_width)
    bin_labels = [f"{int(bin_edges[i])}-{int(bin_edges[i+1])}" for i in range(len(bin_edges)-1)]
    
    # Results list
    results = []
    
    for vesicle_idx, vesicle in enumerate(vesicles):
        # Only consider vesicles within vesicle_distance_threshold of the presynaptic active zone
        distance_to_az = vesicle.get('distance_to_az', 0.0)
        if distance_to_az > vesicle_distance_threshold:
            continue
        
        vesicle_points = np.array(vesicle['coordinates'])
        
        # Find closest presynaptic membrane and its active zone points
        membrane_name = vesicle.get('closest_membrane', None)
        if not membrane_name or membrane_name not in membrane_active_zone_pairs:
            continue
        
        active_zone_points = membrane_active_zone_pairs[membrane_name]['active_zone_points']
        if active_zone_points is None or len(active_zone_points) == 0:
            continue
        
        # Check if this is a fusing vesicle (if fusing_only is True).
        # Strict mode: require precomputed vesicle label from vesicles.py.
        if fusing_only:
            if 'is_fusing' in vesicle:
                is_fusing = bool(vesicle.get('is_fusing'))
            else:
                print(
                    f"Skipping vesicle {vesicle_idx} in {tomogram_name}: "
                    "missing 'is_fusing' label from vesicles.py."
                )
                continue
            if not is_fusing:
                print(f"Skipping vesicle {vesicle_idx} in {tomogram_name}: not fusing.")
                continue
        
        # Compute fusion point for this vesicle
        tree = KDTree(active_zone_points)
        close_points = []
        for pt in vesicle_points:
            idxs = tree.query_ball_point(pt, r=fusion_point_threshold)
            if idxs:
                close_points.extend(active_zone_points[idxs])
        
        if not close_points:
            continue
        
        fusion_point = np.mean(np.vstack(close_points), axis=0)
        
        # Calculate distances from all AuNPs to this fusion point
        aunp_distances = np.linalg.norm(aunp_coords - fusion_point, axis=1)
        
        # Bin the distances into histogram
        hist, _ = np.histogram(aunp_distances, bins=bin_edges)
        
        # Create vesicle name identifier
        vesicle_name = f"{tomogram_name}_vesicle_{vesicle_idx}"
        
        # Create result row
        result_row = {
            'tomogram_name': tomogram_name,
            'alignment_dir': alignment_dir,
            'vesicle_name': vesicle_name,
            'vesicle_id': vesicle_idx,
            'distance_to_presynaptic_az_nm': distance_to_az,
            'vesicle_center_x_nm': vesicle['center'][0],
            'vesicle_center_y_nm': vesicle['center'][1],
            'vesicle_center_z_nm': vesicle['center'][2],
            'vesicle_diameter_nm': vesicle['diameter'],
            'vesicle_volume_nm3': vesicle['volume'],
            'fusion_point_x_nm': fusion_point[0],
            'fusion_point_y_nm': fusion_point[1],
            'fusion_point_z_nm': fusion_point[2],
            'total_aunps_analyzed': len(aunp_coords)
        }
        
        # Add histogram bins
        for i, label in enumerate(bin_labels):
            result_row[f'aunps_{label}nm'] = int(hist[i])
        
        results.append(result_row)
    
    return pd.DataFrame(results)

def analyze_aunps(tomogram_path, active_zone_indices=None, set_name=None, alignment_dir: str = "best_alignment",
                  vesicle_distance_threshold=20.0, dbscan_eps=16.0, dbscan_min_samples=1,
                  cylinder_radius=25.0, receptor_crosssection=122.0, aunps_per_receptor=2.0,
                  vertex_sampling_step=1, synaptic_designation_cutoff=30.0,
                  min_cluster_size=4, fusion_point_threshold=20.0,
                  fusing_perimeter_threshold=1.0):
    """
    Performs analysis of gold nanoparticles (AuNPs) in the tomogram.

    Parameters:
        tomogram_path (str or Path): Path to the tomogram file.
        active_zone_indices (list of int or None): Which active zone .star files to read. If None, read all.
        set_name (str, optional): Name of the experimental set.
        vesicle_distance_threshold (float): Distance threshold for "close" vesicles (nm). Default: 20.0.
        dbscan_eps (float): DBSCAN eps parameter for clustering (nm). Default: 16.0.
        dbscan_min_samples (int): DBSCAN min_samples parameter for clustering. Default: 1.
        cylinder_radius (float): Sliding cylinder radius for packing density heat map (nm). Default: 25.0.
        receptor_crosssection (float): Receptor cross-sectional area for packing density (nm²). Default: 122.0.
        aunps_per_receptor (float): AuNPs per receptor (e.g. 2 for dimer, 1 for monomer). Default: 2.0.
        vertex_sampling_step (int): Sample every Nth mesh vertex for packing (1=all, 50=every 50th). Default: 50.
        synaptic_designation_cutoff (float): Distance cutoff (nm) to postsynaptic active outer membrane for
            synaptic vs extrasynaptic label. Default: 30.0.
        min_cluster_size (int): Minimum cluster size to keep after DBSCAN (smaller clusters -> noise). Default: 4.
        fusion_point_threshold (float): Radius (nm) for AZ points contributing to fusion point. Default: 20.0.
        fusing_perimeter_threshold (float): Max perimeter-to-AZ distance (nm) for fusing vesicles. Default: 1.0.
    """
    print(f"Analyzing AuNPs in {Path(tomogram_path).name}")
    
    try:
        aunps_dir = Path(tomogram_path) / alignment_dir / "aunps"
        import glob
        import os
        star_dfs = []
        if active_zone_indices is not None:
            for idx in active_zone_indices:
                star_file = aunps_dir / f"aunp_tm_BP_active_zone_{idx}_manual_refined.star"
                print("Trying to load:", star_file)
                if star_file.exists():
                    star_data = starfile.read(star_file)
                    if isinstance(star_data, dict):
                        for v in star_data.values():
                            if isinstance(v, pd.DataFrame):
                                df = v.copy()
                                # Ensure active_zone column matches the file index
                                df['active_zone'] = idx
                                star_dfs.append(df)
                                break
                    elif isinstance(star_data, pd.DataFrame):
                        df = star_data.copy()
                        # Ensure active_zone column matches the file index
                        df['active_zone'] = idx
                        star_dfs.append(df)
        else:
            # Load all aunp_tm_BP_active_zone_<N>_manual_refined.star files with numeric suffix
            pattern = str(aunps_dir / "aunp_tm_BP_active_zone_*_manual_refined.star")
            for file in glob.glob(pattern):
                fname = Path(file).name
                m = re.match(r"aunp_tm_BP_active_zone_(\d+)_manual_refined\.star", fname)
                if m:
                    az_id = int(m.group(1))
                    star_data = starfile.read(Path(file))
                    if isinstance(star_data, dict):
                        for v in star_data.values():
                            if isinstance(v, pd.DataFrame):
                                df = v.copy()
                                # Ensure active_zone column matches the file index
                                df['active_zone'] = az_id
                                star_dfs.append(df)
                                break
                    elif isinstance(star_data, pd.DataFrame):
                        df = star_data.copy()
                        # Ensure active_zone column matches the file index
                        df['active_zone'] = az_id
                        star_dfs.append(df)
        
        if not star_dfs:
            print("No numeric aunp_tm_BP_active_zone_<N>_manual_refined.star files found.")
            return {
                'aunp_count': 0,
                'status': 'completed',
                'error': 'No AuNP data found'
            }
        
        # Combine all dataframes
        df = pd.concat(star_dfs, ignore_index=True)
        
        if df is None:
            print("No DataFrame found in .star file.")
            return {
                'aunp_count': 0,
                'status': 'error',
                'error': 'No DataFrame found in .star file'
            }
        
        # Only consider AuNPs within an active zone (active_zone != -1)
        if 'active_zone' not in df.columns:
            print("Column 'active_zone' not found in .star file.")
            return {
                'aunp_count': 0,
                'status': 'error',
                'error': 'Column active_zone not found in .star file'
            }
        
        df_valid = df[df['active_zone'] != -1].copy()
        if df_valid.empty:
            print("No AuNPs within active zones found.")
            return {
                'aunp_count': 0,
                'status': 'completed',
                'error': 'No AuNPs within active zones found'
            }
        
        coord_cols = ['faCoordinateX', 'faCoordinateY', 'faCoordinateZ']
        coords = np.asarray(df_valid[coord_cols]).astype(float)
        
        # Note: Nearest neighbor and clustering analysis will be done AFTER membrane filtering

        aunps_results_dir = Path(tomogram_path) / alignment_dir / "STT_results" / "aunps"
        aunps_results_dir.mkdir(parents=True, exist_ok=True)
        output_file = aunps_results_dir / "aunp_nearest_neighbor_distances.csv"

        # --- Calculate distance to closest pre/post membrane segmentation (before filtering to determine filter) ---
        membranes = import_membrane_segmentations(tomogram_path)
        # Ensure all arrays are 2D and have shape (N, 3)
        pre_arrays = [np.atleast_2d(arr) for arr in membranes['presynaptic'] if np.atleast_2d(arr).shape[1] == 3]
        post_arrays = [np.atleast_2d(arr) for arr in membranes['postsynaptic'] if np.atleast_2d(arr).shape[1] == 3]
        pre_points = np.concatenate(pre_arrays, axis=0) if pre_arrays else np.zeros((0, 3))
        post_points = np.concatenate(post_arrays, axis=0) if post_arrays else np.zeros((0, 3))
        if len(pre_points) > 0:
            pre_tree = KDTree(pre_points)
            pre_dists, _ = pre_tree.query(coords)
        else:
            pre_dists = np.full(coords.shape[0], np.nan)
        if len(post_points) > 0:
            post_tree = KDTree(post_points)
            post_dists, _ = post_tree.query(coords)
        else:
            post_dists = np.full(coords.shape[0], np.nan)
        df_valid['distance_to_presynaptic'] = pre_dists
        df_valid['distance_to_postsynaptic'] = post_dists
        # --- End new ---
        
        # --- No membrane distance filtering ---
        # Keep all AuNPs in analysis; downstream consumers can filter later using
        # distance columns and synaptic_designation.
        print("Skipping membrane distance filtering (all AuNPs retained).")
        coords = np.asarray(df_valid[coord_cols]).astype(float)
        
        # --- Calculate active-zone distances (on filtered AuNPs) ---
        def _nearest_distances_to_cloud(points: np.ndarray, cloud: np.ndarray) -> np.ndarray:
            """Return nearest-neighbor distances from points to cloud (NaN if cloud empty)."""
            if cloud is None or len(cloud) == 0:
                return np.full(points.shape[0], np.nan)
            tree = KDTree(cloud)
            dists, _ = tree.query(points)
            return dists

        try:
            az_segmentations = import_active_zone_segmentations(tomogram_path, alignment_dir=alignment_dir)
            all_az_points = []
            pre_outer_clouds = []
            post_outer_clouds = []
            pre_inner_clouds = []
            post_inner_clouds = []
            for az in az_segmentations.values():
                if 'presynaptic_coords' in az and len(az['presynaptic_coords']) > 0:
                    all_az_points.append(np.atleast_2d(np.asarray(az['presynaptic_coords'])))
                if 'postsynaptic_coords' in az and len(az['postsynaptic_coords']) > 0:
                    all_az_points.append(np.atleast_2d(np.asarray(az['postsynaptic_coords'])))
                if 'presynaptic_outer_coords' in az and len(az['presynaptic_outer_coords']) > 0:
                    pre_outer_clouds.append(np.atleast_2d(np.asarray(az['presynaptic_outer_coords'])))
                if 'postsynaptic_outer_coords' in az and len(az['postsynaptic_outer_coords']) > 0:
                    post_outer_clouds.append(np.atleast_2d(np.asarray(az['postsynaptic_outer_coords'])))
                if 'presynaptic_inner_coords' in az and len(az['presynaptic_inner_coords']) > 0:
                    pre_inner_clouds.append(np.atleast_2d(np.asarray(az['presynaptic_inner_coords'])))
                if 'postsynaptic_inner_coords' in az and len(az['postsynaptic_inner_coords']) > 0:
                    post_inner_clouds.append(np.atleast_2d(np.asarray(az['postsynaptic_inner_coords'])))

            pre_outer_points = np.vstack(pre_outer_clouds) if pre_outer_clouds else np.array([])
            post_outer_points = np.vstack(post_outer_clouds) if post_outer_clouds else np.array([])
            pre_inner_points = np.vstack(pre_inner_clouds) if pre_inner_clouds else np.array([])
            post_inner_points = np.vstack(post_inner_clouds) if post_inner_clouds else np.array([])

            df_valid['distance_to_presynaptic_active_outer'] = _nearest_distances_to_cloud(coords, pre_outer_points)
            df_valid['distance_to_postsynaptic_active_outer'] = _nearest_distances_to_cloud(coords, post_outer_points)
            df_valid['distance_to_presynaptic_active_inner'] = _nearest_distances_to_cloud(coords, pre_inner_points)
            df_valid['distance_to_postsynaptic_active_inner'] = _nearest_distances_to_cloud(coords, post_inner_points)

            pre_center_stack = np.vstack([
                df_valid['distance_to_presynaptic_active_outer'].to_numpy(),
                df_valid['distance_to_presynaptic_active_inner'].to_numpy()
            ])
            post_center_stack = np.vstack([
                df_valid['distance_to_postsynaptic_active_outer'].to_numpy(),
                df_valid['distance_to_postsynaptic_active_inner'].to_numpy()
            ])
            # Hard-coded behavior: use mean of outer/inner distances.
            df_valid['distance_to_presynaptic_active_outer_inner_mean'] = np.nanmean(pre_center_stack, axis=0)
            df_valid['distance_to_postsynaptic_active_outer_inner_mean'] = np.nanmean(post_center_stack, axis=0)

            # Synaptic if within cutoff of postsynaptic active outer membrane; else extrasynaptic
            synaptic_mask = df_valid['distance_to_postsynaptic_active_outer'] <= synaptic_designation_cutoff
            # If distance is NaN, default to extrasynaptic for explicit labeling
            synaptic_mask = synaptic_mask.fillna(False)
            df_valid['synaptic_designation'] = np.where(synaptic_mask, "synaptic", "extrasynaptic")

            if all_az_points:
                all_az_points = np.vstack(all_az_points)
                az_center = np.mean(all_az_points, axis=0)
                distances_to_center = np.linalg.norm(coords - az_center, axis=1)
            else:
                az_center = np.array([np.nan, np.nan, np.nan])
                distances_to_center = np.full(coords.shape[0], np.nan)
        except Exception as e:
            print(f"Error calculating active zone center: {e}")
            az_center = np.array([np.nan, np.nan, np.nan])
            distances_to_center = np.full(coords.shape[0], np.nan)
            df_valid['distance_to_presynaptic_active_outer'] = np.full(coords.shape[0], np.nan)
            df_valid['distance_to_postsynaptic_active_outer'] = np.full(coords.shape[0], np.nan)
            df_valid['distance_to_presynaptic_active_inner'] = np.full(coords.shape[0], np.nan)
            df_valid['distance_to_postsynaptic_active_inner'] = np.full(coords.shape[0], np.nan)
            df_valid['distance_to_presynaptic_active_outer_inner_mean'] = np.full(coords.shape[0], np.nan)
            df_valid['distance_to_postsynaptic_active_outer_inner_mean'] = np.full(coords.shape[0], np.nan)
            df_valid['synaptic_designation'] = "extrasynaptic"
        df_valid['distance_to_active_zone_center'] = distances_to_center
        # --- End active zone center calculation ---

        # --- KDTree nearest neighbor analysis (on filtered AuNPs) ---
        tree = KDTree(coords)
        dists, idxs = tree.query(coords, k=2)
        df_valid['nearest_neighbor_distance'] = dists[:, 1]
        
        # --- AuNP clustering analysis using DBSCAN (on filtered AuNPs) ---
        db = None  # Initialize so it's accessible for cluster summary
        try:
            # Use DBSCAN with custom parameters, then filter out clusters with < min_cluster_size points
            db = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit(coords)
            initial_labels = db.labels_
            
            # Count points in each cluster
            unique_labels, counts = np.unique(initial_labels, return_counts=True)
            
            # Create a mapping: clusters with < min_cluster_size points become noise (-1)
            label_mapping = {}
            valid_cluster_count = 0
            for label, count in zip(unique_labels, counts):
                if label == -1:  # Keep noise as noise
                    label_mapping[label] = -1
                elif count < min_cluster_size:  # Small clusters become noise
                    label_mapping[label] = -1
                else:  # Renumber larger clusters starting from 1
                    valid_cluster_count += 1
                    label_mapping[label] = valid_cluster_count
            
            # Apply the mapping to get final cluster assignments
            final_labels = np.array([label_mapping[label] for label in initial_labels])
            df_valid['aunp_cluster'] = final_labels
            
            # Count final clusters (excluding noise)
            n_clusters = len(set(final_labels)) - (1 if -1 in final_labels else 0)
            n_small_clusters_filtered = len([count for label, count in zip(unique_labels, counts) 
                                           if label != -1 and count < min_cluster_size])
            
            print(f"DBSCAN found {n_clusters} AuNP clusters (eps={dbscan_eps} nm, min_samples={dbscan_min_samples})")
            print(f"Filtered out {n_small_clusters_filtered} small clusters (< {min_cluster_size} points) and reassigned to noise")
        except Exception as e:
            print(f"Error in DBSCAN clustering: {e}")
            df_valid['aunp_cluster'] = -1
            db = None  # Ensure db is None if clustering failed
        # --- End clustering ---

        # --- Output cluster summary CSV ---
        try:
            from scipy.spatial import ConvexHull, distance_matrix
            cluster_labels = np.unique(df_valid['aunp_cluster'].to_numpy())
            cluster_rows = []
            for label in cluster_labels:
                if label == -1:
                    continue  # Skip noise
                cluster_points = coords[df_valid['aunp_cluster'].to_numpy() == label]
                n_points = len(cluster_points)
                if n_points < 3:
                    area = np.nan
                    max_dim = np.nan
                else:
                    try:
                        hull = ConvexHull(cluster_points)
                        area = hull.area / 2.0
                    except Exception:
                        area = np.nan
                    try:
                        dists = distance_matrix(cluster_points, cluster_points)
                        max_dim = np.nanmax(dists)
                    except Exception:
                        max_dim = np.nan
                density = n_points / area if area and area > 0 else np.nan
                cluster_rows.append({
                    'cluster_label': label,
                    'n_aunps': n_points,
                    'cluster_area_nm2': area,
                    'cluster_max_dimension_nm': max_dim,
                    'cluster_density_aunps_per_nm2': density
                })
            cluster_df = pd.DataFrame(cluster_rows)
            cluster_csv = aunps_results_dir / "aunp_clusters.csv"
            cluster_df.to_csv(cluster_csv, index=False)
            print(f"Saved AuNP cluster summary to {cluster_csv}")
            # --- Append to global results/aunps/aunp_cluster_results.csv ---
            tomogram_name = Path(tomogram_path).name
            if set_name is None or set_name == "unknown":
                path_parts = Path(tomogram_path).parts
                set_name = "unknown"
                for i, part in enumerate(path_parts):
                    if part.endswith("_tomograms") and i > 0:
                        set_name = part.replace("_tomograms", "")
                        break
            cluster_df['tomogram_name'] = tomogram_name
            cluster_df['set_name'] = set_name
            cluster_df['alignment_dir'] = alignment_dir
            global_csv = Path("results/aunps/aunp_cluster_results.csv")
            global_csv.parent.mkdir(parents=True, exist_ok=True)
            if global_csv.exists():
                try:
                    df_existing = pd.read_csv(global_csv)
                    if 'alignment_dir' not in df_existing.columns:
                        df_existing['alignment_dir'] = ''
                    df_existing = df_existing[
                        ~(
                            (df_existing['tomogram_name'] == tomogram_name) &
                            (df_existing['alignment_dir'] == alignment_dir)
                        )
                    ]
                    df_combined = pd.concat([df_existing, cluster_df], ignore_index=True)
                    df_combined.to_csv(global_csv, index=False)
                except Exception as e:
                    print(f"Error updating global aunp_cluster_results.csv: {e}")
                    cluster_df.to_csv(global_csv, index=False)
            else:
                cluster_df.to_csv(global_csv, index=False)
            print(f"Appended cluster info to {global_csv}")
        except Exception as e:
            print(f"Error writing aunp_clusters.csv: {e}")
        
        # --- Output .star file with cluster assignments (after filtering and clustering) ---
        try:
            # Use the same columns as the imported .star, plus calculated distance columns and aunp_cluster
            # Get base columns from original star file
            base_cols = [col for col in df.columns if col in df_valid.columns]
            # Add calculated distance columns if they exist
            distance_cols = [
                'nearest_neighbor_distance', 'distance_to_presynaptic', 'distance_to_postsynaptic',
                'distance_to_fusion_point', 'distance_to_active_zone_center',
                'distance_to_presynaptic_active_outer', 'distance_to_postsynaptic_active_outer',
                'distance_to_presynaptic_active_inner', 'distance_to_postsynaptic_active_inner',
                'distance_to_presynaptic_active_outer_inner_mean', 'distance_to_postsynaptic_active_outer_inner_mean',
            ]
            additional_cols = [col for col in distance_cols if col in df_valid.columns]
            star_cols = base_cols + additional_cols + ['synaptic_designation', 'aunp_cluster']
            star_df = df_valid[star_cols].copy()
            if not isinstance(star_df, pd.DataFrame):
                star_df = pd.DataFrame(star_df)
            star_outfile = aunps_results_dir / "aunp_clusters.star"
            starfile.write(star_df, star_outfile, overwrite=True)
            print(f"Saved AuNP cluster assignments (filtered) to {star_outfile}")
        except Exception as e:
            print(f"Error writing .star file with clusters: {e}")
        # --- End .star output ---
        
        # Save nearest neighbor distances in STT_results/aunps directory
        # Compute fusion points
        fusion_points = compute_fusion_points(
            tomogram_path,
            vesicle_distance_threshold=vesicle_distance_threshold,
            fusion_point_threshold=fusion_point_threshold,
            alignment_dir=alignment_dir,
        )
        fusion_points = np.asarray(fusion_points)
        if len(fusion_points) > 0 and fusion_points.shape[0] > 0:
            fusion_tree = KDTree(fusion_points)
            fusion_dists, _ = fusion_tree.query(coords)
        else:
            fusion_dists = np.full(coords.shape[0], np.nan)
        df_valid['distance_to_fusion_point'] = fusion_dists
        # Update output columns (rename distance columns to include units)
        cols_out = [
            'active_zone', 'faCoordinateX_nm', 'faCoordinateY_nm', 'faCoordinateZ_nm',
            'nearest_neighbor_distance_nm', 'distance_to_presynaptic_nm', 'distance_to_postsynaptic_nm',
            'distance_to_fusion_point_nm', 'distance_to_active_zone_center_nm',
            'distance_to_presynaptic_active_outer_nm', 'distance_to_postsynaptic_active_outer_nm',
            'distance_to_presynaptic_active_inner_nm', 'distance_to_postsynaptic_active_inner_nm',
            'distance_to_presynaptic_active_outer_inner_mean_nm', 'distance_to_postsynaptic_active_outer_inner_mean_nm',
            'synaptic_designation', 'aunp_cluster'
        ]
        # Create a copy with renamed columns for CSV output
        df_output = df_valid[['active_zone', 'faCoordinateX', 'faCoordinateY', 'faCoordinateZ',
                              'nearest_neighbor_distance', 'distance_to_presynaptic', 'distance_to_postsynaptic',
                              'distance_to_fusion_point', 'distance_to_active_zone_center',
                              'distance_to_presynaptic_active_outer', 'distance_to_postsynaptic_active_outer',
                              'distance_to_presynaptic_active_inner', 'distance_to_postsynaptic_active_inner',
                              'distance_to_presynaptic_active_outer_inner_mean', 'distance_to_postsynaptic_active_outer_inner_mean',
                              'synaptic_designation', 'aunp_cluster']].copy()
        df_output.columns = cols_out
        df_output.to_csv(output_file, index=False)
        print(f"Saved nearest neighbor, membrane, and fusion distances for AuNPs to {output_file}")
        
        # --- Append to global results/aunps/all_aunp_distances.csv ---
        tomogram_name = Path(tomogram_path).name
        # Use provided set_name or extract from tomogram path
        if set_name is None or set_name == "unknown":
            path_parts = Path(tomogram_path).parts
            set_name = "unknown"
            for i, part in enumerate(path_parts):
                if part.endswith("_tomograms") and i > 0:
                    set_name = part.replace("_tomograms", "")
                    break
        
        # Add tomogram and set info to the dataframe
        df_valid['tomogram_name'] = tomogram_name
        df_valid['set_name'] = set_name
        df_valid['alignment_dir'] = alignment_dir
        
        # Create a copy with renamed columns for global CSV (add units to distance/coordinate columns)
        df_global = df_valid.copy()
        rename_dict = {
            'faCoordinateX': 'faCoordinateX_nm',
            'faCoordinateY': 'faCoordinateY_nm',
            'faCoordinateZ': 'faCoordinateZ_nm',
            'nearest_neighbor_distance': 'nearest_neighbor_distance_nm',
            'distance_to_presynaptic': 'distance_to_presynaptic_nm',
            'distance_to_postsynaptic': 'distance_to_postsynaptic_nm',
            'distance_to_fusion_point': 'distance_to_fusion_point_nm',
            'distance_to_active_zone_center': 'distance_to_active_zone_center_nm',
            'distance_to_presynaptic_active_outer': 'distance_to_presynaptic_active_outer_nm',
            'distance_to_postsynaptic_active_outer': 'distance_to_postsynaptic_active_outer_nm',
            'distance_to_presynaptic_active_inner': 'distance_to_presynaptic_active_inner_nm',
            'distance_to_postsynaptic_active_inner': 'distance_to_postsynaptic_active_inner_nm',
            'distance_to_presynaptic_active_outer_inner_mean': 'distance_to_presynaptic_active_outer_inner_mean_nm',
            'distance_to_postsynaptic_active_outer_inner_mean': 'distance_to_postsynaptic_active_outer_inner_mean_nm',
        }
        # Only rename columns that exist
        rename_dict = {k: v for k, v in rename_dict.items() if k in df_global.columns}
        df_global = df_global.rename(columns=rename_dict)
        
        global_csv = Path("results/aunps/all_aunp_distances.csv")
        global_csv.parent.mkdir(parents=True, exist_ok=True)
        if global_csv.exists():
            try:
                df_existing = pd.read_csv(global_csv)
                if 'alignment_dir' not in df_existing.columns:
                    df_existing['alignment_dir'] = ''
                # Remove existing data for this tomogram+alignment pair
                df_existing = df_existing[
                    ~(
                        (df_existing['tomogram_name'] == tomogram_name) &
                        (df_existing['alignment_dir'] == alignment_dir)
                    )
                ]
                df_combined = pd.concat([df_existing, df_global], ignore_index=True)
                df_combined.to_csv(global_csv, index=False)
            except Exception as e:
                print(f"Error updating global all_aunp_distances.csv: {e}")
                df_global.to_csv(global_csv, index=False)
        else:
            df_global.to_csv(global_csv, index=False)
        print(f"Appended AuNP distances to {global_csv}")
        # --- End global results ---
        
        # Helper function to save histogram CSV
        def save_histogram_csv(df_hist, csv_path, vesicle_type, bin_width):
            """Save histogram DataFrame to CSV, updating existing file if it exists."""
            if not df_hist.empty:
                # Add set info (tomogram_name already included in the dataframe)
                df_hist['set_name'] = set_name
                df_hist['alignment_dir'] = alignment_dir
                
                csv_path.parent.mkdir(parents=True, exist_ok=True)
                if csv_path.exists():
                    try:
                        df_existing = pd.read_csv(csv_path)
                        if 'alignment_dir' not in df_existing.columns:
                            df_existing['alignment_dir'] = ''
                        # Remove existing data for this tomogram+alignment pair
                        df_existing = df_existing[
                            ~(
                                (df_existing['tomogram_name'] == tomogram_name) &
                                (df_existing['alignment_dir'] == alignment_dir)
                            )
                        ]
                        df_combined = pd.concat([df_existing, df_hist], ignore_index=True)
                        df_combined.to_csv(csv_path, index=False)
                    except Exception as e:
                        print(f"Error updating {csv_path.name}: {e}")
                        df_hist.to_csv(csv_path, index=False)
                else:
                    df_hist.to_csv(csv_path, index=False)
                print(f"Saved AuNP histograms (bin{int(bin_width)}) for {len(df_hist)} {vesicle_type} vesicles to {csv_path}")
            else:
                print(f"No {vesicle_type} vesicles found for AuNP histogram analysis (bin{int(bin_width)})")
        
        # --- Compute AuNP distance histograms per close vesicle (bin5) ---
        print("Computing AuNP distance histograms for close vesicles (bin5)...")
        df_vesicle_aunp_hist_bin5 = compute_aunp_distance_histograms_per_vesicle(
            tomogram_path, coords, 
            vesicle_distance_threshold=vesicle_distance_threshold,
            fusion_point_threshold=fusion_point_threshold,
            max_distance=500.0,
            bin_width=5.0,
            alignment_dir=alignment_dir,
        )
        save_histogram_csv(df_vesicle_aunp_hist_bin5, Path("results/aunps/close_vesicles_aunp_histograms_bin5.csv"), "close", 5.0)
        
        # --- Compute AuNP distance histograms per close vesicle (bin10) ---
        print("Computing AuNP distance histograms for close vesicles (bin10)...")
        df_vesicle_aunp_hist_bin10 = compute_aunp_distance_histograms_per_vesicle(
            tomogram_path, coords, 
            vesicle_distance_threshold=vesicle_distance_threshold,
            fusion_point_threshold=fusion_point_threshold,
            max_distance=500.0,
            bin_width=10.0,
            alignment_dir=alignment_dir,
        )
        save_histogram_csv(df_vesicle_aunp_hist_bin10, Path("results/aunps/close_vesicles_aunp_histograms_bin10.csv"), "close", 10.0)
        
        # --- Compute AuNP distance histograms per close vesicle (bin50) ---
        print("Computing AuNP distance histograms for close vesicles (bin50)...")
        df_vesicle_aunp_hist_bin50 = compute_aunp_distance_histograms_per_vesicle(
            tomogram_path, coords, 
            vesicle_distance_threshold=vesicle_distance_threshold,
            fusion_point_threshold=fusion_point_threshold,
            max_distance=500.0,
            bin_width=50.0,
            alignment_dir=alignment_dir,
        )
        save_histogram_csv(df_vesicle_aunp_hist_bin50, Path("results/aunps/close_vesicles_aunp_histograms_bin50.csv"), "close", 50.0)
        # --- End close vesicle AuNP histograms ---
        
        # --- Compute AuNP distance histograms per FUSING vesicle (bin5) ---
        print(f"Computing AuNP distance histograms for fusing vesicles (distance <= {fusing_perimeter_threshold} nm to AZ, bin5)...")
        df_fusing_vesicle_aunp_hist_bin5 = compute_aunp_distance_histograms_per_vesicle(
            tomogram_path, coords, 
            vesicle_distance_threshold=vesicle_distance_threshold,
            fusion_point_threshold=fusion_point_threshold,
            max_distance=500.0,
            bin_width=5.0,
            fusing_only=True,
            fusing_perimeter_threshold=fusing_perimeter_threshold,
            alignment_dir=alignment_dir,
        )
        save_histogram_csv(df_fusing_vesicle_aunp_hist_bin5, Path("results/aunps/fusing_vesicles_aunp_histograms_bin5.csv"), "fusing", 5.0)
        
        # --- Compute AuNP distance histograms per FUSING vesicle (bin10) ---
        print(f"Computing AuNP distance histograms for fusing vesicles (distance <= {fusing_perimeter_threshold} nm to AZ, bin10)...")
        df_fusing_vesicle_aunp_hist_bin10 = compute_aunp_distance_histograms_per_vesicle(
            tomogram_path, coords, 
            vesicle_distance_threshold=vesicle_distance_threshold,
            fusion_point_threshold=fusion_point_threshold,
            max_distance=500.0,
            bin_width=10.0,
            fusing_only=True,
            fusing_perimeter_threshold=fusing_perimeter_threshold,
            alignment_dir=alignment_dir,
        )
        save_histogram_csv(df_fusing_vesicle_aunp_hist_bin10, Path("results/aunps/fusing_vesicles_aunp_histograms_bin10.csv"), "fusing", 10.0)
        
        # --- Compute AuNP distance histograms per FUSING vesicle (bin50) ---
        print(f"Computing AuNP distance histograms for fusing vesicles (distance <= {fusing_perimeter_threshold} nm to AZ, bin50)...")
        df_fusing_vesicle_aunp_hist_bin50 = compute_aunp_distance_histograms_per_vesicle(
            tomogram_path, coords, 
            vesicle_distance_threshold=vesicle_distance_threshold,
            fusion_point_threshold=fusion_point_threshold,
            max_distance=500.0,
            bin_width=50.0,
            fusing_only=True,
            fusing_perimeter_threshold=fusing_perimeter_threshold,
            alignment_dir=alignment_dir,
        )
        save_histogram_csv(df_fusing_vesicle_aunp_hist_bin50, Path("results/aunps/fusing_vesicles_aunp_histograms_bin50.csv"), "fusing", 50.0)
        # --- End fusing vesicle AuNP histograms ---
        
        # --- Calculate packing density for each active zone ---
        packing_density_results = {}
        try:
            from .activezone import import_membrane_segmentations_from_glb, find_active_zones_from_glb, define_active_zonogram
            
            print("Calculating packing density for active zones...")
            # Load active zones from GLB
            membrane_data = import_membrane_segmentations_from_glb(tomogram_path, alignment_dir=alignment_dir)
            active_zones_glb = find_active_zones_from_glb(membrane_data, distance_range=(10.0, 40.0))
            
            if active_zones_glb and 'active_zones' in active_zones_glb and len(active_zones_glb['active_zones']) > 0:
                # Define active zonograms
                zonogram_results = define_active_zonogram(active_zones_glb)
                
                if zonogram_results['status'] == 'completed' and 'zonogram_data' in zonogram_results:
                    # Group AuNPs by active zone
                    aunps_by_az = {}
                    for az_idx in df_valid['active_zone'].unique():
                        if az_idx != -1:
                            az_df = df_valid[df_valid['active_zone'] == az_idx]
                            aunps_by_az[az_idx] = az_df[coord_cols].values
                    
                    # Calculate packing density for each active zone
                    all_packing_coefficients = []
                    for zone_name, zone_data in active_zones_glb['active_zones'].items():
                        # Check if this zone has the required mesh
                        if 'active_postsynaptic_mesh' not in zone_data:
                            continue
                        
                        # Find matching AuNP data for this zone
                        # Try to match by zone name pattern (e.g., "active_zone_pre1_post1")
                        zone_aunps = None
                        for az_idx, aunp_coords in aunps_by_az.items():
                            # For now, use the first available AuNP set
                            # In the future, could match more precisely using zone centers
                            if zone_aunps is None:
                                zone_aunps = aunp_coords
                        
                        if zone_aunps is None or len(zone_aunps) == 0:
                            print(f"  No AuNPs found for {zone_name}, skipping packing density calculation")
                            continue
                        
                        # Get zonogram data for this zone
                        if zone_name not in zonogram_results['zonogram_data']:
                            print(f"  No zonogram data found for {zone_name}, skipping packing density calculation")
                            continue
                        
                        zonogram_data = zonogram_results['zonogram_data'][zone_name]
                        
                        try:
                            # Calculate packing density
                            v_array, packing_coefficient = calculate_packing_density_using_sliding_cylinder(
                                zone_data,
                                zonogram_data,
                                zone_aunps,
                                cylinder_radius=cylinder_radius,
                                receptor_crosssection_nm_squared=receptor_crosssection,
                                aunps_per_receptor=aunps_per_receptor,
                                vertex_sampling_step=vertex_sampling_step
                            )
                            
                            # Store results (NaN -> None for JSON compatibility)
                            packing_list = [None if np.isnan(x) else float(x) for x in packing_coefficient]
                            valid = packing_coefficient[~np.isnan(packing_coefficient)]
                            if len(valid) > 0:
                                pmax, pmin = float(np.nanmax(packing_coefficient)), float(np.nanmin(packing_coefficient))
                                pavg, pstd = float(np.nanmean(packing_coefficient)), float(np.nanstd(packing_coefficient))
                            else:
                                pmax = pmin = pavg = pstd = None
                            packing_density_results[zone_name] = {
                                'v_array': v_array.tolist(),  # Convert to list for JSON serialization
                                'packing_coefficient': packing_list,
                                'max_packing_coefficient': pmax,
                                'avg_packing_coefficient': pavg,
                                'min_packing_coefficient': pmin,
                                'std_packing_coefficient': pstd,
                            }
                            all_packing_coefficients.extend([x for x in packing_coefficient if not np.isnan(x)])
                            
                            m = packing_density_results[zone_name]
                            max_s = f"{m['max_packing_coefficient']:.4f}" if m['max_packing_coefficient'] is not None else "N/A"
                            avg_s = f"{m['avg_packing_coefficient']:.4f}" if m['avg_packing_coefficient'] is not None else "N/A"
                            print(f"  ✓ Calculated packing density for {zone_name}: max={max_s}, avg={avg_s}")
                        except Exception as e:
                            print(f"  Error calculating packing density for {zone_name}: {e}")
                            continue
                    
                    # Save packing density results to file
                    if packing_density_results:
                        packing_density_file = aunps_results_dir / "packing_density_results.json"
                        import json
                        with open(packing_density_file, 'w') as f:
                            json.dump(packing_density_results, f, indent=2)
                        print(f"Saved packing density results to {packing_density_file}")
                else:
                    print("  Could not define active zonograms for packing density calculation")
            else:
                print("  No active zones found for packing density calculation")
        except Exception as e:
            print(f"Error calculating packing density: {e}")
            import traceback
            traceback.print_exc()
        # --- End packing density calculation ---
        
        # Prepare summary statistics for ResultsManager
        n_aunps = len(df_valid)
        summary_stats = {
            'aunp_count': n_aunps,
        }
        
        # Calculate statistics for each distance column
        for col in ['nearest_neighbor_distance', 'distance_to_presynaptic', 'distance_to_postsynaptic', 'distance_to_fusion_point']:
            if col in df_valid.columns and n_aunps > 0:
                summary_stats[f'{col}_mean'] = float(df_valid[col].mean())
                summary_stats[f'{col}_std'] = float(df_valid[col].std())
                summary_stats[f'{col}_min'] = float(df_valid[col].min())
                summary_stats[f'{col}_max'] = float(df_valid[col].max())
            else:
                summary_stats[f'{col}_mean'] = 0.0
                summary_stats[f'{col}_std'] = 0.0
                summary_stats[f'{col}_min'] = 0.0
                summary_stats[f'{col}_max'] = 0.0
        
        # Add cluster information if available
        if 'aunp_cluster' in df_valid.columns:
            n_clusters = len(df_valid['aunp_cluster'].unique()) - (1 if -1 in df_valid['aunp_cluster'].values else 0)
            summary_stats['aunp_cluster_count'] = n_clusters
        
        # Add AuNP density (AuNPs per unit active zone area)
        # Use the saved active zone results which already contain the filtered zones
        if n_aunps > 0:
            try:
                # Load active zone results (already filtered by active zone analysis step)
                from .results_manager import ResultsManager
                results_manager = ResultsManager("results")
                analysis_name = f"{tomogram_name}__{alignment_dir}"
                active_zone_results = results_manager.get_tomogram_results(analysis_name, 'activezone')
                
                if active_zone_results and 'results' in active_zone_results:
                    az_data = active_zone_results['results'].get('active_zone', {})
                    
                    # Use total postsynaptic active zone area (already filtered to only include zones with AuNPs)
                    if 'total_active_zone_post_area' not in az_data:
                        raise ValueError("No total postsynaptic active zone area available. Cannot calculate AuNP density. Active zone analysis must be run with postsynaptic area calculation.")
                    
                    total_active_zone_area = az_data['total_active_zone_post_area']
                    
                    if total_active_zone_area <= 0:
                        raise ValueError(f"Invalid total active zone area: {total_active_zone_area}. Area must be positive.")
                    
                    summary_stats['aunp_density'] = float(n_aunps / total_active_zone_area)  # AuNPs per µm²
                else:
                    raise ValueError(
                        f"No active zone results available for '{analysis_name}'. "
                        "Cannot calculate AuNP density. Active zone analysis must be run first for this alignment_dir."
                    )
                
            except Exception as e:
                print(f"Error getting active zone area for density calculation: {e}")
                raise  # Re-raise the exception instead of silently setting to 0.0
        else:
            summary_stats['aunp_density'] = 0.0
        
        # Add distance to active zone center mean
        if 'distance_to_active_zone_center' in df_valid.columns and n_aunps > 0:
            summary_stats['distance_to_active_zone_center_mean'] = float(df_valid['distance_to_active_zone_center'].mean())
        else:
            summary_stats['distance_to_active_zone_center_mean'] = 0.0
        
        # Add packing density statistics
        if packing_density_results:
            # Calculate overall max and average across all active zones (filter None from masked edge zones)
            all_max_values = [x for x in (z['max_packing_coefficient'] for z in packing_density_results.values()) if x is not None]
            all_avg_values = [x for x in (z['avg_packing_coefficient'] for z in packing_density_results.values()) if x is not None]
            all_min_values = [x for x in (z['min_packing_coefficient'] for z in packing_density_results.values()) if x is not None]
            if all_max_values and all_avg_values and all_min_values:
                summary_stats['packing_density_max'] = float(np.max(all_max_values))
                summary_stats['packing_density_avg'] = float(np.mean(all_avg_values))
                summary_stats['packing_density_min'] = float(np.min(all_min_values))
        else:
            # Packing density calculation was attempted but no results were produced
            # This is acceptable - not all analyses may have packing density data
            # Don't set to 0.0, just don't include these stats
            pass
        
        # Add completion status
        summary_stats['status'] = 'completed'
        
        return summary_stats
        
    except Exception as e:
        print(f"Error in AuNP analysis: {e}")
        return {
            'aunp_count': 0,
            'status': 'error',
            'error': str(e)
        }
