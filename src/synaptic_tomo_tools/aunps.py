# src/synaptic_tomo_tools/aunps.py

from typing import List
import pandas as pd
import numpy as np
from scipy.spatial import KDTree
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

def compute_fusion_points(tomogram_path, vesicle_distance_threshold=20.0, fusion_point_threshold=20.0):
    """
    For each vesicle within 20 nm of the presynaptic active zone, compute the putative fusion point as the average
    of all presynaptic active zone points within 20 nm of any vesicle point. Supports multiple active zones.
    Returns a list of fusion points (np.ndarray shape (N, 3)).
    """
    # Load vesicle results
    vesicles_file = Path(tomogram_path) / "best_alignment" / "STT_results" / "vesicles" / "vesicle_results.json"
    if not vesicles_file.exists():
        print(f"No vesicle results found: {vesicles_file}")
        return []
    with open(vesicles_file, 'r') as f:
        vesicle_data = json.load(f)
    vesicles = vesicle_data['vesicles']
    # Load presynaptic membranes and active zones
    membrane_active_zone_pairs = import_presynaptic_membranes_and_active_zones(tomogram_path)
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


def check_if_fusing_vesicle(vesicle, active_zone_points, perimeter_threshold=5.0):
    """
    Check if a vesicle is a "fusing vesicle" by determining if its spherical perimeter
    is within perimeter_threshold of the presynaptic active zone.
    
    Uses analytical distance formula: distance_to_surface = |distance(center, point) - radius|
    This is exact and doesn't require sampling points on the sphere.
    
    Args:
        vesicle: Vesicle dictionary with 'center' and 'radius'
        active_zone_points: Array of active zone coordinates
        perimeter_threshold: Distance threshold in nm (default 5.0)
        
    Returns:
        Boolean indicating if vesicle is fusing
    """
    if active_zone_points is None or len(active_zone_points) == 0:
        return False
    
    center = np.array(vesicle['center'])
    radius = vesicle['radius']
    
    # Calculate distances from vesicle center to all active zone points
    distances_from_center = np.linalg.norm(active_zone_points - center, axis=1)
    
    # Distance from sphere surface to each active zone point
    # Positive if point is outside sphere, negative if inside
    distances_to_surface = np.abs(distances_from_center - radius)
    
    # Find minimum distance from sphere surface to active zone
    min_distance_to_az = np.min(distances_to_surface)
    
    return min_distance_to_az <= perimeter_threshold


def compute_aunp_distance_histograms_per_vesicle(tomogram_path, aunp_coords, vesicle_distance_threshold=20.0, 
                                                  fusion_point_threshold=20.0, max_distance=500.0, bin_width=5.0,
                                                  fusing_only=False, fusing_perimeter_threshold=5.0):
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
        fusing_perimeter_threshold: Distance threshold for fusing vesicles (default 5.0 nm)
        
    Returns:
        DataFrame with vesicle info and AuNP distance histogram bins
    """
    # Get tomogram name
    tomogram_name = Path(tomogram_path).name
    
    # Load vesicle results
    vesicles_file = Path(tomogram_path) / "best_alignment" / "STT_results" / "vesicles" / "vesicle_results.json"
    if not vesicles_file.exists():
        print(f"No vesicle results found: {vesicles_file}")
        return pd.DataFrame()
    
    with open(vesicles_file, 'r') as f:
        vesicle_data = json.load(f)
    vesicles = vesicle_data['vesicles']
    
    # Load presynaptic membranes and active zones
    membrane_active_zone_pairs = import_presynaptic_membranes_and_active_zones(tomogram_path)
    
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
        
        # Check if this is a fusing vesicle (if fusing_only is True)
        if fusing_only:
            is_fusing = check_if_fusing_vesicle(vesicle, active_zone_points, 
                                               perimeter_threshold=fusing_perimeter_threshold)
            if not is_fusing:
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
            'vesicle_name': vesicle_name,
            'vesicle_id': vesicle_idx,
            'distance_to_presynaptic_az': distance_to_az,
            'vesicle_center_x': vesicle['center'][0],
            'vesicle_center_y': vesicle['center'][1],
            'vesicle_center_z': vesicle['center'][2],
            'vesicle_diameter': vesicle['diameter'],
            'vesicle_volume': vesicle['volume'],
            'fusion_point_x': fusion_point[0],
            'fusion_point_y': fusion_point[1],
            'fusion_point_z': fusion_point[2],
            'total_aunps_analyzed': len(aunp_coords)
        }
        
        # Add histogram bins
        for i, label in enumerate(bin_labels):
            result_row[f'aunps_{label}nm'] = int(hist[i])
        
        results.append(result_row)
    
    return pd.DataFrame(results)

def analyze_aunps(tomogram_path, active_zone_indices=None, set_name=None):
    """
    Performs analysis of gold nanoparticles (AuNPs) in the tomogram.

    Parameters:
        tomogram_path (str or Path): Path to the tomogram file.
        active_zone_indices (list of int or None): Which active zone .star files to read. If None, read all.
    """
    print(f"Analyzing AuNPs in {Path(tomogram_path).name}")
    
    try:
        aunps_dir = Path(tomogram_path) / "best_alignment" / "aunps"
        
        # Check if aunps directory exists
        if not aunps_dir.exists():
            print(f"Error: AuNPs directory not found: {aunps_dir}")
            return {
                'aunp_count': 0,
                'status': 'error',
                'error': f'AuNPs directory not found: {aunps_dir}'
            }
        
        print(f"Looking for AuNP files in: {aunps_dir}")
        star_dfs = []
        if active_zone_indices is not None:
            for idx in active_zone_indices:
                star_file = aunps_dir / f"aunp_tm_BP_active_zone_{idx}.star"
                print("Trying to load:", star_file)
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
            # Load all aunp_tm_BP_active_zone_*.star files with numeric suffix (not _all.star)
            # Use Path.glob() for better path handling
            all_star_files = list(aunps_dir.glob("aunp_tm_BP_active_zone_*.star"))
            print(f"Found {len(all_star_files)} files matching pattern aunp_tm_BP_active_zone_*.star")
            for file in all_star_files:
                fname = file.name
                m = re.match(r"aunp_tm_BP_active_zone_(\d+)\.star", fname)
                if m:
                    print(f"Loading numeric file: {fname}")
                    star_data = starfile.read(file)
                    if isinstance(star_data, dict):
                        for v in star_data.values():
                            if isinstance(v, pd.DataFrame):
                                star_dfs.append(v)
                                break
                    elif isinstance(star_data, pd.DataFrame):
                        star_dfs.append(star_data)
                else:
                    print(f"Skipping non-numeric file: {fname}")
        
        if not star_dfs:
            print(f"No numeric aunp_tm_BP_active_zone_*.star files found in {aunps_dir}")
            # List what files are actually in the directory
            all_files = list(aunps_dir.glob("*"))
            if all_files:
                print(f"Files found in {aunps_dir}:")
                for f in all_files:
                    print(f"  - {f.name}")
            else:
                print(f"Directory is empty or does not exist: {aunps_dir}")
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
        # KDTree nearest neighbor analysis
        tree = KDTree(coords)
        dists, idxs = tree.query(coords, k=2)
        df_valid['nearest_neighbor_distance'] = dists[:, 1]

        # --- AuNP clustering analysis using DBSCAN ---
        try:
            # Use DBSCAN with min_samples=1, then filter out clusters with < 4 points
            db = DBSCAN(eps=16, min_samples=1).fit(coords)
            initial_labels = db.labels_
            
            # Count points in each cluster
            unique_labels, counts = np.unique(initial_labels, return_counts=True)
            
            # Create a mapping: clusters with < 4 points become noise (-1)
            label_mapping = {}
            valid_cluster_count = 0
            for label, count in zip(unique_labels, counts):
                if label == -1:  # Keep noise as noise
                    label_mapping[label] = -1
                elif count < 4:  # Small clusters become noise
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
                                           if label != -1 and count < 4])
            
            print(f"DBSCAN found {n_clusters} AuNP clusters (eps=16 nm, min_samples=1)")
            print(f"Filtered out {n_small_clusters_filtered} small clusters (< 4 points) and reassigned to noise")
        except Exception as e:
            print(f"Error in DBSCAN clustering: {e}")
            df_valid['aunp_cluster'] = -1
        # --- End clustering ---

        aunps_results_dir = Path(tomogram_path) / "best_alignment" / "STT_results" / "aunps"
        aunps_results_dir.mkdir(parents=True, exist_ok=True)
        output_file = aunps_results_dir / "aunp_nearest_neighbor_distances.csv"

        # --- New: Output .star file with cluster assignments ---
        try:
            # Use the same columns as the imported .star, plus aunp_cluster
            star_cols = [col for col in df.columns if col in df_valid.columns] + ['aunp_cluster']
            star_df = df_valid[star_cols].copy()
            if not isinstance(star_df, pd.DataFrame):
                star_df = pd.DataFrame(star_df)
            star_outfile = aunps_results_dir / "aunp_clusters.star"
            starfile.write(star_df, star_outfile, overwrite=True)
            print(f"Saved AuNP cluster assignments to {star_outfile}")
        except Exception as e:
            print(f"Error writing .star file with clusters: {e}")
        # --- End .star output ---

        # --- Output cluster summary CSV ---
        try:
            from scipy.spatial import ConvexHull, distance_matrix
            cluster_labels = np.unique(db.labels_)
            cluster_rows = []
            for label in cluster_labels:
                if label == -1:
                    continue  # Skip noise
                cluster_points = coords[db.labels_ == label]
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
                    'cluster_area': area,
                    'cluster_max_dimension': max_dim,
                    'cluster_density': density
                })
            cluster_df = pd.DataFrame(cluster_rows)
            cluster_csv = aunps_results_dir / "aunp_clusters.csv"
            cluster_df.to_csv(cluster_csv, index=False)
            print(f"Saved AuNP cluster summary to {cluster_csv}")
            # --- Append to global results/aunp_cluster_results.csv ---
            tomogram_name = Path(tomogram_path).name
            # Use provided set_name or extract from tomogram path
            if set_name is None or set_name == "unknown":
                path_parts = Path(tomogram_path).parts
                set_name = "unknown"
                for i, part in enumerate(path_parts):
                    if part.endswith("_tomograms") and i > 0:
                        set_name = part.replace("_tomograms", "")
                        break
            cluster_df['tomogram_name'] = tomogram_name
            cluster_df['set_name'] = set_name
            global_csv = Path("results/aunp_cluster_results.csv")
            global_csv.parent.mkdir(parents=True, exist_ok=True)
            if global_csv.exists():
                try:
                    df_existing = pd.read_csv(global_csv)
                    df_existing = df_existing[df_existing['tomogram_name'] != tomogram_name]
                    df_combined = pd.concat([df_existing, cluster_df], ignore_index=True)
                    df_combined.to_csv(global_csv, index=False)
                except Exception as e:
                    print(f"Error updating global aunp_cluster_results.csv: {e}")
                    cluster_df.to_csv(global_csv, index=False)
            else:
                cluster_df.to_csv(global_csv, index=False)
            print(f"Appended cluster info to {global_csv}")
            # --- End global results ---
        except Exception as e:
            print(f"Error writing aunp_clusters.csv: {e}")
        # --- End cluster summary ---

        # Calculate distance to active zone center
        try:
            az_segmentations = import_active_zone_segmentations(tomogram_path)
            all_az_points = []
            for az in az_segmentations.values():
                if 'presynaptic_coords' in az and len(az['presynaptic_coords']) > 0:
                    all_az_points.append(np.asarray(az['presynaptic_coords']))
                if 'postsynaptic_coords' in az and len(az['postsynaptic_coords']) > 0:
                    all_az_points.append(np.asarray(az['postsynaptic_coords']))
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
        df_valid['distance_to_active_zone_center'] = distances_to_center
        # --- End new ---

        # --- New: Calculate distance to closest pre/post membrane segmentation ---
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
        # Save nearest neighbor distances in STT_results/aunps directory
        # Compute fusion points
        fusion_points = compute_fusion_points(tomogram_path, vesicle_distance_threshold=20.0)
        fusion_points = np.asarray(fusion_points)
        if len(fusion_points) > 0 and fusion_points.shape[0] > 0:
            fusion_tree = KDTree(fusion_points)
            fusion_dists, _ = fusion_tree.query(coords)
        else:
            fusion_dists = np.full(coords.shape[0], np.nan)
        df_valid['distance_to_fusion_point'] = fusion_dists
        # Update output columns
        cols_out = [
            'active_zone', 'faCoordinateX', 'faCoordinateY', 'faCoordinateZ',
            'nearest_neighbor_distance', 'distance_to_presynaptic', 'distance_to_postsynaptic',
            'distance_to_fusion_point', 'distance_to_active_zone_center', 'aunp_cluster'
        ]
        df_valid.loc[:, cols_out].to_csv(output_file, index=False)
        print(f"Saved nearest neighbor, membrane, and fusion distances for AuNPs to {output_file}")
        
        # --- Append to global results/all_aunp_distances.csv ---
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
        
        global_csv = Path("results/all_aunp_distances.csv")
        global_csv.parent.mkdir(parents=True, exist_ok=True)
        if global_csv.exists():
            try:
                df_existing = pd.read_csv(global_csv)
                # Remove existing data for this tomogram
                df_existing = df_existing[df_existing['tomogram_name'] != tomogram_name]
                df_combined = pd.concat([df_existing, df_valid], ignore_index=True)
                df_combined.to_csv(global_csv, index=False)
            except Exception as e:
                print(f"Error updating global all_aunp_distances.csv: {e}")
                df_valid.to_csv(global_csv, index=False)
        else:
            df_valid.to_csv(global_csv, index=False)
        print(f"Appended AuNP distances to {global_csv}")
        # --- End global results ---
        
        # --- Compute AuNP distance histograms per close vesicle ---
        print("Computing AuNP distance histograms for close vesicles...")
        df_vesicle_aunp_hist = compute_aunp_distance_histograms_per_vesicle(
            tomogram_path, coords, 
            vesicle_distance_threshold=20.0,
            fusion_point_threshold=20.0,
            max_distance=500.0,
            bin_width=5.0
        )
        
        if not df_vesicle_aunp_hist.empty:
            # Add set info (tomogram_name already included in the dataframe)
            df_vesicle_aunp_hist['set_name'] = set_name
            
            # Save to global CSV
            vesicle_aunp_csv = Path("results/close_vesicles_aunp_histograms.csv")
            vesicle_aunp_csv.parent.mkdir(parents=True, exist_ok=True)
            if vesicle_aunp_csv.exists():
                try:
                    df_existing = pd.read_csv(vesicle_aunp_csv)
                    # Remove existing data for this tomogram
                    df_existing = df_existing[df_existing['tomogram_name'] != tomogram_name]
                    df_combined = pd.concat([df_existing, df_vesicle_aunp_hist], ignore_index=True)
                    df_combined.to_csv(vesicle_aunp_csv, index=False)
                except Exception as e:
                    print(f"Error updating close_vesicles_aunp_histograms.csv: {e}")
                    df_vesicle_aunp_hist.to_csv(vesicle_aunp_csv, index=False)
            else:
                df_vesicle_aunp_hist.to_csv(vesicle_aunp_csv, index=False)
            print(f"Saved AuNP histograms for {len(df_vesicle_aunp_hist)} close vesicles to {vesicle_aunp_csv}")
        else:
            print("No close vesicles found for AuNP histogram analysis")
        # --- End vesicle AuNP histograms ---
        
        # --- Compute AuNP distance histograms per FUSING vesicle ---
        print("Computing AuNP distance histograms for fusing vesicles (perimeter within 5 nm of AZ)...")
        df_fusing_vesicle_aunp_hist = compute_aunp_distance_histograms_per_vesicle(
            tomogram_path, coords, 
            vesicle_distance_threshold=20.0,
            fusion_point_threshold=20.0,
            max_distance=500.0,
            bin_width=5.0,
            fusing_only=True,
            fusing_perimeter_threshold=5.0
        )
        
        if not df_fusing_vesicle_aunp_hist.empty:
            # Add set info (tomogram_name already included in the dataframe)
            df_fusing_vesicle_aunp_hist['set_name'] = set_name
            
            # Save to global CSV
            fusing_vesicle_aunp_csv = Path("results/fusing_vesicles_aunp_histograms.csv")
            fusing_vesicle_aunp_csv.parent.mkdir(parents=True, exist_ok=True)
            if fusing_vesicle_aunp_csv.exists():
                try:
                    df_existing = pd.read_csv(fusing_vesicle_aunp_csv)
                    # Remove existing data for this tomogram
                    df_existing = df_existing[df_existing['tomogram_name'] != tomogram_name]
                    df_combined = pd.concat([df_existing, df_fusing_vesicle_aunp_hist], ignore_index=True)
                    df_combined.to_csv(fusing_vesicle_aunp_csv, index=False)
                except Exception as e:
                    print(f"Error updating fusing_vesicles_aunp_histograms.csv: {e}")
                    df_fusing_vesicle_aunp_hist.to_csv(fusing_vesicle_aunp_csv, index=False)
            else:
                df_fusing_vesicle_aunp_hist.to_csv(fusing_vesicle_aunp_csv, index=False)
            print(f"Saved AuNP histograms for {len(df_fusing_vesicle_aunp_hist)} fusing vesicles to {fusing_vesicle_aunp_csv}")
        else:
            print("No fusing vesicles found for AuNP histogram analysis")
        # --- End fusing vesicle AuNP histograms ---
        
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
        if n_aunps > 0:
            # Get active zone area from existing results
            try:
                # Load active zone results to get the surface area
                from .results_manager import ResultsManager
                results_manager = ResultsManager("results")
                active_zone_results = results_manager.get_tomogram_results(tomogram_name, 'activezone')
                if active_zone_results and 'results' in active_zone_results:
                    az_data = active_zone_results['results'].get('active_zone', {})
                    # Calculate total active zone area (sum of all active zones)
                    active_zone_count = az_data.get('active_zone_count', 0)
                    avg_active_zone_area = az_data.get('avg_active_zone_area', 0.0)
                    total_active_zone_area = active_zone_count * avg_active_zone_area
                    
                    if total_active_zone_area > 0:
                        summary_stats['aunp_density'] = float(n_aunps / total_active_zone_area)  # AuNPs per µm²
                    else:
                        summary_stats['aunp_density'] = 0.0
                else:
                    summary_stats['aunp_density'] = 0.0
            except Exception as e:
                print(f"Error getting active zone area for density calculation: {e}")
                summary_stats['aunp_density'] = 0.0
        else:
            summary_stats['aunp_density'] = 0.0
        
        # Add distance to active zone center mean
        if 'distance_to_active_zone_center' in df_valid.columns and n_aunps > 0:
            summary_stats['distance_to_active_zone_center_mean'] = float(df_valid['distance_to_active_zone_center'].mean())
        else:
            summary_stats['distance_to_active_zone_center_mean'] = 0.0
        
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
