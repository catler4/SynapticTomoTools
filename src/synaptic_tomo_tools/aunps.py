# src/synaptic_tomo_tools/aunps.py

from typing import List
import pandas as pd
import numpy as np
from scipy.spatial import KDTree
from pathlib import Path
import starfile
from .activezone import import_membrane_segmentations
from datetime import datetime
import json
from .vesicles import import_presynaptic_membranes_and_active_zones

def save_aunp_results_to_csv(tomogram_path, aunp_df, csv_file=None, overwrite=True):
    tomogram_path = Path(tomogram_path)
    tomogram_name = tomogram_path.name
    # Extract set name from tomogram path
    path_parts = tomogram_path.parts
    set_name = "unknown"
    for i, part in enumerate(path_parts):
        if part.endswith("_tomograms") and i > 0:
            set_name = part.replace("_tomograms", "")
            break
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Summary statistics
    n_aunps = len(aunp_df)
    def stats(col):
        return {
            f'{col}_mean': float(aunp_df[col].mean()) if n_aunps else 0.0,
            f'{col}_std': float(aunp_df[col].std()) if n_aunps else 0.0,
            f'{col}_min': float(aunp_df[col].min()) if n_aunps else 0.0,
            f'{col}_max': float(aunp_df[col].max()) if n_aunps else 0.0,
        }
    row = {
        'tomogram_name': tomogram_name,
        'set_name': set_name,
        'timestamp': timestamp,
        'aunp_count': n_aunps,
    }
    for col in ['nearest_neighbor_distance', 'distance_to_presynaptic', 'distance_to_postsynaptic', 'distance_to_fusion_point']:
        row.update(stats(col))
    # Save to CSV
    if csv_file is None:
        csv_file = "results/aunp_results.csv"
    csv_path = Path(csv_file)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite and csv_path.exists():
        try:
            df_existing = pd.read_csv(csv_path)
            df_existing = df_existing[df_existing['tomogram_name'] != tomogram_name]
            df_new = pd.DataFrame([row])
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined.to_csv(csv_path, index=False)
            print(f"Updated AuNP results in CSV: {csv_path}")
        except Exception as e:
            print(f"Error updating existing CSV, creating new file: {e}")
            df_new = pd.DataFrame([row])
            df_new.to_csv(csv_path, index=False)
    else:
        if csv_path.exists():
            df_new = pd.DataFrame([row])
            df_new.to_csv(csv_path, mode='a', header=False, index=False)
        else:
            df_new = pd.DataFrame([row])
            df_new.to_csv(csv_path, index=False)
    print(f"Saved AuNP results to CSV: {csv_path}")
    return csv_path

def save_all_aunp_distances_to_csv(aunp_df, tomogram_path, csv_file="results/all_aunp_distances.csv"):
    tomogram_path = Path(tomogram_path)
    tomogram_name = tomogram_path.name
    # Extract set name from tomogram path
    path_parts = tomogram_path.parts
    set_name = "unknown"
    for i, part in enumerate(path_parts):
        if part.endswith("_tomograms") and i > 0:
            set_name = part.replace("_tomograms", "")
            break
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Add tomogram info columns
    aunp_df = aunp_df.copy()
    aunp_df['tomogram_name'] = tomogram_name
    aunp_df['set_name'] = set_name
    aunp_df['timestamp'] = timestamp
    # Reorder columns
    cols_out = [
        'tomogram_name', 'set_name', 'timestamp',
        'active_zone', 'faCoordinateX', 'faCoordinateY', 'faCoordinateZ',
        'nearest_neighbor_distance', 'distance_to_presynaptic', 'distance_to_postsynaptic',
        'distance_to_fusion_point'
    ]
    aunp_df = aunp_df[cols_out]
    # Create or update CSV file
    csv_path = Path(csv_file)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if csv_path.exists():
        try:
            df_existing = pd.read_csv(csv_path)
            # Remove existing rows for this tomogram
            df_existing = df_existing[df_existing['tomogram_name'] != tomogram_name]
            df_new = pd.concat([df_existing, aunp_df], ignore_index=True)
            df_new.to_csv(csv_path, index=False)
            print(f"Updated all AuNP distances CSV: {csv_path} (added {len(aunp_df)} AuNPs from {tomogram_name})")
        except Exception as e:
            print(f"Error updating existing CSV, creating new file: {e}")
            aunp_df.to_csv(csv_path, index=False)
    else:
        aunp_df.to_csv(csv_path, index=False)
        print(f"Created all AuNP distances CSV: {csv_path} (added {len(aunp_df)} AuNPs from {tomogram_name})")
    return csv_path

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

def analyze_aunps(tomogram_path):
    """
    Performs analysis of gold nanoparticles (AuNPs) in the tomogram.

    Parameters:
        tomogram_path (str or Path): Path to the tomogram file.
    """
    print(f"Analyzing AuNPs in {Path(tomogram_path).name}")
    # Nearest neighbor analysis
    aunps_dir = Path(tomogram_path) / "best_alignment" / "aunps"
    star_file = aunps_dir / "aunp_tm_BP_active_zone_all.star"
    if not star_file.exists():
        print(f"AuNP .star file not found: {star_file}")
        return None

    # Use starfile to read the .star file
    star_data = starfile.read(star_file)
    df = None
    if isinstance(star_data, dict):
        # Use the first DataFrame block
        for v in star_data.values():
            if isinstance(v, pd.DataFrame):
                df = v
                break
    elif isinstance(star_data, pd.DataFrame):
        df = star_data
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
    output_file = aunps_dir / "aunp_nearest_neighbor_distances.csv"
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
        'distance_to_fusion_point'
    ]
    df_valid.loc[:, cols_out].to_csv(output_file, index=False)
    print(f"Saved nearest neighbor, membrane, and fusion distances for AuNPs to {output_file}")
    # Save summary results to results/aunp_results.csv
    save_aunp_results_to_csv(tomogram_path, df_valid)
    # Save all per-AuNP results to results/all_aunp_distances.csv
    save_all_aunp_distances_to_csv(df_valid, tomogram_path)
    return df_valid.loc[:, cols_out]
