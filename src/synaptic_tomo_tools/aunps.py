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

def compute_fusion_points(tomogram_path, vesicle_distance_threshold=10.0, fusion_point_threshold=10.0):
    """
    For each vesicle within 10 nm of the presynaptic active zone, compute the putative fusion point as the average
    of all presynaptic active zone points within 10 nm of any vesicle point. Supports multiple active zones.
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

def analyze_aunps(tomogram_path, active_zone_indices=None, set_name=None):
    """
    Performs analysis of gold nanoparticles (AuNPs) in the tomogram.

    Parameters:
        tomogram_path (str or Path): Path to the tomogram file.
        active_zone_indices (list of int or None): Which active zone .star files to read. If None, read all.
    """
    print(f"Analyzing AuNPs in {Path(tomogram_path).name}")
    aunps_dir = Path(tomogram_path) / "best_alignment" / "aunps"
    import glob
    import os
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
            print("No numeric aunp_tm_BP_active_zone_*.star files found and _all.star fallback is disabled.")
            return None
    df = pd.concat(star_dfs, ignore_index=True)
    if df is None:
        print("No DataFrame found in .star file.")
        return None
    # Only consider AuNPs within an active zone (active_zone != -1)
    if 'active_zone' not in df.columns:
        print("Column 'active_zone' not found in .star file.")
        return None
    df_valid = df[df['active_zone'] != -1].copy()
    if df_valid.empty:
        print("No AuNPs within active zones found.")
        return None
    coord_cols = ['faCoordinateX', 'faCoordinateY', 'faCoordinateZ']
    coords = np.asarray(df_valid[coord_cols]).astype(float)
    # KDTree nearest neighbor analysis
    tree = KDTree(coords)
    dists, idxs = tree.query(coords, k=2)
    df_valid['nearest_neighbor_distance'] = dists[:, 1]

    # --- New: AuNP clustering analysis using DBSCAN ---
    try:
        db = DBSCAN(eps=20, min_samples=4).fit(coords)
        df_valid['aunp_cluster'] = db.labels_
        n_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
        print(f"DBSCAN found {n_clusters} AuNP clusters (eps=20 nm, min_samples=4)")
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
    fusion_points = compute_fusion_points(tomogram_path)
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
    
    return summary_stats
