# src/synaptic_tomo_tools/vesicles.py

from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import ConvexHull, KDTree
from scipy.optimize import minimize
import json
import glob
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import mrcfile


def load_tomogram_data(tomogram_path, alignment_dir: str = "best_alignment") -> Optional[np.ndarray]:
    """
    Load tomogram data from the *ddw.mrc file.
    
    Args:
        tomogram_path: Path to the tomogram directory
        
    Returns:
        Tomogram data as numpy array, or None if file not found
    """
    tomogram_path = Path(tomogram_path)
    
    # Look for the *ddw.mrc file in the selected alignment subdirectory
    align_dir = tomogram_path / alignment_dir
    if not align_dir.exists():
        print(f"Alignment directory not found: {align_dir}")
        return None
    
    # Find the *ddw.mrc file in selected alignment directory
    ddw_files = list(align_dir.glob("*ddw.mrc"))
    
    if not ddw_files:
        print(f"No *ddw.mrc file found in {align_dir}")
        return None
    
    # Use the first ddw file found
    ddw_file = ddw_files[0]
    print(f"Loading tomogram data from {ddw_file.name}")
    
    try:
        with mrcfile.open(ddw_file, 'r') as mrc:
            data = mrc.data
            print(f"Tomogram data shape: {data.shape}")
            return data
    except Exception as e:
        print(f"Error loading tomogram data: {e}")
        return None


def fit_sphere_to_points(points: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Fit a sphere to a set of 3D points using least squares optimization.
    
    Args:
        points: Nx3 array of (x, y, z) coordinates
        
    Returns:
        Tuple of (center, radius)
    """
    if len(points) < 4:
        # Need at least 4 points for sphere fitting
        center = np.mean(points, axis=0)
        radius = np.mean(np.linalg.norm(points - center, axis=1))
        return center, radius
    
    # Initial guess: center is mean of points, radius is mean distance to center
    initial_center = np.mean(points, axis=0)
    initial_radius = np.mean(np.linalg.norm(points - initial_center, axis=1))
    
    def sphere_error(params):
        center = params[:3]
        radius = params[3]
        distances = np.linalg.norm(points - center, axis=1)
        return np.sum((distances - radius) ** 2)
    
    # Optimize sphere parameters
    initial_params = np.concatenate([initial_center, [initial_radius]])
    result = minimize(sphere_error, initial_params, method='L-BFGS-B')
    
    if result.success:
        center = result.x[:3]
        radius = result.x[3]
        return center, radius
    else:
        raise ValueError(f"Sphere fitting optimization failed: {result.message}. Cannot calculate vesicle center and radius.")


def calculate_sphere_volume(radius: float) -> float:
    """
    Calculate volume of a sphere.
    
    Args:
        radius: Radius in nm
        
    Returns:
        Volume in nm³
    """
    return (4/3) * np.pi * radius**3


def check_sphere_overlap(center1: np.ndarray, radius1: float, 
                        center2: np.ndarray, radius2: float) -> bool:
    """
    Check if two spheres overlap.
    
    Args:
        center1, center2: Sphere centers
        radius1, radius2: Sphere radii
        
    Returns:
        True if spheres overlap
    """
    distance = np.linalg.norm(center1 - center2)
    return bool(distance < (radius1 + radius2))


def import_vesicle_segmentations(tomogram_path, alignment_dir: str = "best_alignment") -> List[Dict[str, Any]]:
    """
    Import vesicle segmentation files and fit spheres to each.
    
    Args:
        tomogram_path: Path to the tomogram directory (str or Path)
        
    Returns:
        List of dictionaries containing vesicle data
    """
    tomogram_path = Path(tomogram_path)
    aunps_dir = tomogram_path / alignment_dir / "aunps"
    
    if not aunps_dir.exists():
        raise FileNotFoundError(f"AuNPs directory not found: {aunps_dir}")
    
    # Find all vesicle segmentation files
    vesicle_files = list(aunps_dir.glob("synapticvesicles_*.txt"))
    print(f"Found {len(vesicle_files)} vesicle segmentation files")
    
    vesicles = []
    diameters = []
    
    # Process vesicles with progress bar
    for file_path in tqdm(sorted(vesicle_files), desc="Fitting spheres to vesicles"):
        try:
            # Load coordinates
            coords = np.loadtxt(file_path, delimiter=None)
            
            if len(coords) < 3:
                continue  # Skip vesicles with fewer than 3 points
            
            # Fit sphere to points
            center, radius = fit_sphere_to_points(coords)
            
            # Calculate volume
            volume = calculate_sphere_volume(radius)
            diameter = 2 * radius
            
            # Store vesicle data
            vesicle_data = {
                'file_name': file_path.name,
                'center': center.tolist(),
                'radius': float(radius),
                'diameter': float(diameter),
                'volume': float(volume),
                'point_count': len(coords),
                'coordinates': coords.tolist()
            }
            
            vesicles.append(vesicle_data)
            diameters.append(diameter)
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Report summary statistics
    if diameters:
        avg_diameter = np.mean(diameters)
        std_diameter = np.std(diameters)
        print(f"Vesicle fitting complete: {len(vesicles)} vesicles processed")
        print(f"Average diameter: {avg_diameter:.2f} ± {std_diameter:.2f} nm")
    else:
        print("No vesicles were successfully processed")
    
    return vesicles


def remove_overlapping_vesicles(vesicles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remove smaller vesicles that overlap with larger ones.
    
    Args:
        vesicles: List of vesicle dictionaries
        
    Returns:
        List of vesicles with overlaps removed
    """
    if len(vesicles) <= 1:
        return vesicles
    
    # Sort vesicles by volume (largest first)
    sorted_vesicles = sorted(vesicles, key=lambda v: v['volume'], reverse=True)
    
    # Check for overlaps
    non_overlapping = []
    
    for i, vesicle in enumerate(sorted_vesicles):
        is_overlapping = False
        
        # Check against all larger vesicles (already processed)
        for j in range(i):
            other_vesicle = sorted_vesicles[j]
            
            if check_sphere_overlap(
                np.array(vesicle['center']), vesicle['radius'],
                np.array(other_vesicle['center']), other_vesicle['radius']
            ):
                is_overlapping = True
                break
        
        if not is_overlapping:
            non_overlapping.append(vesicle)
    
    print(f"Removed {len(vesicles) - len(non_overlapping)} overlapping vesicles")
    return non_overlapping


def import_presynaptic_membranes_and_active_zones(tomogram_path, alignment_dir: str = "best_alignment") -> Dict[str, Dict[str, np.ndarray]]:
    """
    Import presynaptic membranes and their associated active zones.
    Note: Active zones are already filtered by the active zone analysis step, so only
    relevant zones (those with AuNPs) will be present in the saved files.
    
    Args:
        tomogram_path: Path to the tomogram directory (str or Path)
        
    Returns:
        Dictionary mapping presynaptic membrane names to their active zone points
    """
    tomogram_path = Path(tomogram_path)
    aunps_dir = tomogram_path / alignment_dir / "aunps"
    stt_results_dir = tomogram_path / alignment_dir / "STT_results" / "activezone"
    
    if not aunps_dir.exists():
        raise FileNotFoundError(f"AuNPs directory not found: {aunps_dir}")
    
    if not stt_results_dir.exists():
        raise FileNotFoundError(f"Active zones directory not found: {stt_results_dir}")
    
    # Find all presynaptic membrane files in aunps directory
    pre_membrane_files = list(aunps_dir.glob("presynapticmembranes_*.txt"))
    
    if not pre_membrane_files:
        print("No presynaptic membrane files found")
        return {}
    
    membrane_active_zone_pairs = {}
    
    for membrane_file in pre_membrane_files:
        try:
            # Load presynaptic membrane points
            membrane_points = np.loadtxt(membrane_file, delimiter=None)
            
            # Find corresponding active zone files
            membrane_name = membrane_file.stem  # e.g., "presynapticmembranes_1"
            membrane_number = membrane_name.split('_')[-1]  # e.g., "1"
            
            # Look for active zone files with matching number in STT_results/activezone
            # Note: These files are already filtered by the active zone analysis step
            # to only include zones with AuNPs, so no additional filtering is needed
            active_zone_files = list(stt_results_dir.glob(f"active_zone_pre{membrane_number}_post*_pre_inner.txt"))
            
            if active_zone_files:
                # Load all active zones for this membrane
                all_active_zone_points = []
                for active_zone_file in sorted(active_zone_files):
                    active_zone_points = np.loadtxt(active_zone_file, delimiter=None)
                    all_active_zone_points.append(active_zone_points)
                
                # Combine all active zone points into a single array
                if all_active_zone_points:
                    combined_active_zone_points = np.vstack(all_active_zone_points)
                else:
                    combined_active_zone_points = np.array([])
                
                membrane_active_zone_pairs[membrane_name] = {
                    'membrane_points': membrane_points,
                    'active_zone_points': combined_active_zone_points,
                    'individual_active_zones': all_active_zone_points  # Keep individual zones for detailed analysis
                }
                # Loaded membrane with active zone points
            else:
                # Warning: No active zone files found
                # Use empty active zone if none found
                membrane_active_zone_pairs[membrane_name] = {
                    'membrane_points': membrane_points,
                    'active_zone_points': np.array([]),
                    'individual_active_zones': []
                }
                
        except Exception as e:
            print(f"Error loading {membrane_file}: {e}")
    
    return membrane_active_zone_pairs


def find_closest_active_zone(vesicle_center: np.ndarray, 
                            membrane_active_zone_pairs: Dict[str, Dict[str, np.ndarray]]) -> Tuple[str, np.ndarray]:
    """
    Find the active zone closest to a vesicle across all filtered active zones.
    Only considers active zones that passed the filtering step (those with AuNPs).
    
    Args:
        vesicle_center: (x, y, z) coordinates of vesicle center
        membrane_active_zone_pairs: Dictionary mapping membrane names to their data
        
    Returns:
        Tuple of (membrane_name, active_zone_points) for the closest active zone
    """
    if not membrane_active_zone_pairs:
        return "unknown", np.array([])
    
    closest_membrane = None
    closest_active_zone_points = np.array([])
    min_distance = float('inf')
    
    # Check all membranes and their active zones
    for membrane_name, data in membrane_active_zone_pairs.items():
        active_zone_points = data['active_zone_points']
        
        # Only consider membranes that have active zones (filtered zones with AuNPs)
        if len(active_zone_points) == 0:
            continue
        
        # Calculate distance from vesicle center to active zone points
        distances = np.linalg.norm(active_zone_points - vesicle_center, axis=1)
        min_active_zone_distance = np.min(distances)
        
        # Find the closest active zone across all membranes
        if min_active_zone_distance < min_distance:
            min_distance = min_active_zone_distance
            closest_membrane = membrane_name
            closest_active_zone_points = active_zone_points
    
    if closest_membrane is None:
        return "unknown", np.array([])
    
    return closest_membrane, closest_active_zone_points


def calculate_vesicle_distance_to_closest_active_zone(vesicle_data: Tuple[int, Dict[str, Any]], 
                                                    membrane_active_zone_pairs: Dict[str, Dict[str, np.ndarray]]) -> Tuple[int, float, str]:
    """
    Calculate distance for a single vesicle to its closest active zone (for parallel processing).
    
    Args:
        vesicle_data: Tuple of (index, vesicle_dict)
        membrane_active_zone_pairs: Dictionary mapping membrane names to their data
        
    Returns:
        Tuple of (index, distance, membrane_name)
    """
    index, vesicle = vesicle_data
    
    if not membrane_active_zone_pairs:
        return index, float('nan'), "unknown"
    
    vesicle_center = np.array(vesicle['center'])
    
    # Find closest active zone across all filtered active zones (only those with AuNPs)
    membrane_name, active_zone_points = find_closest_active_zone(vesicle_center, membrane_active_zone_pairs)
    
    if len(active_zone_points) == 0:
        return index, float('nan'), membrane_name
    
    # Get all vesicle segmentation points
    vesicle_points = np.array(vesicle['coordinates'])
    
    # Build KDTree for efficient nearest neighbor search
    tree = KDTree(active_zone_points)
    
    # Calculate distances from all vesicle points to active zone
    distances, _ = tree.query(vesicle_points)
    
    # Return the minimum distance (closest point on vesicle to active zone)
    return index, float(np.min(distances)), membrane_name


def calculate_vesicle_distances_to_closest_active_zones(vesicles: List[Dict[str, Any]], 
                                                       membrane_active_zone_pairs: Dict[str, Dict[str, np.ndarray]],
                                                       n_processes: Optional[int] = None) -> Tuple[List[float], List[str]]:
    """
    Calculate distances for all vesicles to their closest active zones using parallel processing.
    
    Args:
        vesicles: List of vesicle dictionaries
        membrane_active_zone_pairs: Dictionary mapping membrane names to their data
        n_processes: Number of processes to use (default: CPU count)
        
    Returns:
        Tuple of (distances, membrane_names) corresponding to each vesicle
    """
    if not membrane_active_zone_pairs:
        return [float('nan')] * len(vesicles), ["unknown"] * len(vesicles)
    
    if n_processes is None:
        n_processes = mp.cpu_count()
    
    # Create partial function with membrane data
    calc_func = partial(calculate_vesicle_distance_to_closest_active_zone, 
                       membrane_active_zone_pairs=membrane_active_zone_pairs)
    
    # Prepare data for parallel processing
    vesicle_data = [(i, vesicle) for i, vesicle in enumerate(vesicles)]
    
    # Use multiprocessing to calculate distances
    with mp.Pool(processes=n_processes) as pool:
        results = list(tqdm(
            pool.imap(calc_func, vesicle_data),
            total=len(vesicle_data),
            desc=f"Calculating vesicle distances to closest active zones (using {n_processes} processes)"
        ))
    
    # Sort results by index and extract distances and membrane names
    results.sort(key=lambda x: x[0])
    distances = [result[1] for result in results]
    membrane_names = [result[2] for result in results]
    
    return distances, membrane_names


def save_vesicle_results(vesicles: List[Dict[str, Any]], tomogram_path, alignment_dir: str = "best_alignment"):
    """
    Save vesicle analysis results to JSON file.
    
    Args:
        vesicles: List of vesicle dictionaries
        tomogram_path: Path to tomogram directory (str or Path)
    """
    tomogram_path = Path(tomogram_path)
    stt_results_dir = tomogram_path / alignment_dir / "STT_results"
    
    # Create vesicles directory
    vesicles_dir = stt_results_dir / "vesicles"
    vesicles_dir.mkdir(parents=True, exist_ok=True)
    
    # Calculate summary statistics
    if vesicles:
        volumes = [v['volume'] for v in vesicles]
        diameters = [v['diameter'] for v in vesicles]
        distances_to_az = [v.get('distance_to_az', np.nan) for v in vesicles]
        valid_distances = [d for d in distances_to_az if np.isfinite(d)]
        
        summary = {
            'total_vesicles': len(vesicles),
            'total_vesicle_volume_nm3': float(np.sum(volumes)),
            'average_vesicle_diameter_nm': float(np.mean(diameters)),
            'std_vesicle_diameter_nm': float(np.std(diameters)),
            'min_vesicle_diameter_nm': float(np.min(diameters)),
            'max_vesicle_diameter_nm': float(np.max(diameters)),
            'average_distance_to_az_nm': float(np.mean(valid_distances)) if valid_distances else np.nan,
            'min_distance_to_az_nm': float(np.min(valid_distances)) if valid_distances else np.nan,
            'max_distance_to_az_nm': float(np.max(valid_distances)) if valid_distances else np.nan,
            'std_distance_to_az_nm': float(np.std(valid_distances)) if valid_distances else np.nan
        }
    else:
        summary = {
            'total_vesicles': 0,
            'total_vesicle_volume_nm3': 0.0,
            'average_vesicle_diameter_nm': 0.0,
            'std_vesicle_diameter_nm': 0.0,
            'min_vesicle_diameter_nm': 0.0,
            'max_vesicle_diameter_nm': 0.0,
            'average_distance_to_az_nm': 0.0,
            'min_distance_to_az_nm': 0.0,
            'max_distance_to_az_nm': 0.0,
            'std_distance_to_az_nm': 0.0
        }
    
    # Save detailed results
    results = {
        'summary': summary,
        'vesicles': vesicles
    }
    
    results_file = vesicles_dir / "vesicle_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Saved vesicle results to {results_file}")
    return results


# CSV export functions removed - now handled by ResultsManager




def detect_vesicles(
    tomogram_path,
    set_name=None,
    vesicle_distance_threshold=10.0,
    alignment_dir: str = "best_alignment",
    fusing_perimeter_threshold: float = 1.0,
) -> Dict[str, Any]:
    """
    Detect synaptic vesicles in tomogram.
    
    Args:
        tomogram_path (str or Path): Path to the tomogram file.
        set_name (str, optional): Name of the experimental set.
        vesicle_distance_threshold (float): Distance threshold for "close" vesicles (nm). Default: 10.0.
    
    Returns:
        Dictionary containing vesicle detection results.
    """
    print(f"Detecting vesicles in {Path(tomogram_path).name}")
    
    try:
        # Import vesicle segmentations
        vesicles = import_vesicle_segmentations(tomogram_path, alignment_dir=alignment_dir)
        
        # Remove overlapping vesicles
        vesicles = remove_overlapping_vesicles(vesicles)
        
        # Import presynaptic membranes and their associated active zones
        # Note: Active zones are already filtered by the active zone analysis step
        # to only include zones with AuNPs, so only relevant zones will be loaded
        membrane_active_zone_pairs = import_presynaptic_membranes_and_active_zones(tomogram_path, alignment_dir=alignment_dir)
        
        # Calculate distances from vesicle segmentation points to closest active zones using parallel processing
        if membrane_active_zone_pairs:
            # Use parallel processing for distance calculations to closest active zones
            distances_to_az, membrane_names = calculate_vesicle_distances_to_closest_active_zones(vesicles, membrane_active_zone_pairs)
            
            # Assign distances and membrane names to vesicles
            for i, vesicle in enumerate(vesicles):
                vesicle['distance_to_az'] = distances_to_az[i]
                vesicle['closest_membrane'] = membrane_names[i]
        else:
            # No presynaptic membranes available
            for vesicle in vesicles:
                vesicle['distance_to_az'] = np.nan
                vesicle['closest_membrane'] = "unknown"
        
        # Assign all vesicle distance classes in one step based on distance_to_az.
        for vesicle in vesicles:
            distance_to_az = vesicle.get('distance_to_az', np.nan)
            if not np.isfinite(distance_to_az):
                vesicle['vesicle_distance_class'] = 'unknown'
                vesicle['is_fusing'] = False
                vesicle['is_close'] = False
            elif distance_to_az <= fusing_perimeter_threshold:
                vesicle['vesicle_distance_class'] = 'fusing'
                vesicle['is_fusing'] = True
                vesicle['is_close'] = False
            elif distance_to_az <= vesicle_distance_threshold:
                vesicle['vesicle_distance_class'] = 'close'
                vesicle['is_fusing'] = False
                vesicle['is_close'] = True
            else:
                vesicle['vesicle_distance_class'] = 'far'
                vesicle['is_fusing'] = False
                vesicle['is_close'] = False

        # Calculate sphericity for all vesicles
        print("Calculating vesicle sphericity...")
        sphericities = calculate_all_vesicle_sphericities(vesicles)
        
        # Add sphericity data to vesicles
        for i, vesicle in enumerate(vesicles):
            vesicle.update(sphericities[i])
        
        # Calculate summary statistics
        if vesicles:
            volumes = [v['volume'] for v in vesicles]
            diameters = [v['diameter'] for v in vesicles]
            distances_to_az = [v.get('distance_to_az', np.nan) for v in vesicles]
            valid_distances = [d for d in distances_to_az if np.isfinite(d)]
            # Extract sphericity values (volume-based only)
            sphericity_volume = [s['sphericity_volume'] for s in sphericities]
            
            # Count vesicles within threshold of active zone using the same logic as save_nearby_vesicles
            nearby_vesicles = [v for v in vesicles if v.get('vesicle_distance_class') in ('fusing', 'close')]
            nearby_vesicle_count = len(nearby_vesicles)
            
            # --- Append to global results/all_vesicle_data.csv ---
            tomogram_name = Path(tomogram_path).name
            # Use provided set_name or extract from tomogram path
            if set_name is None or set_name == "unknown":
                path_parts = Path(tomogram_path).parts
                set_name = "unknown"
                for i, part in enumerate(path_parts):
                    if part.endswith("_tomograms") and i > 0:
                        set_name = part.replace("_tomograms", "")
                        break
            
            # Prepare individual vesicle data for CSV
            vesicle_rows = []
            for i, vesicle in enumerate(vesicles):
                row = {
                    'tomogram_name': tomogram_name,
                    'set_name': set_name,
                    'alignment_dir': alignment_dir,
                    'vesicle_id': i,
                    'center_x': vesicle['center'][0],
                    'center_y': vesicle['center'][1],
                    'center_z': vesicle['center'][2],
                    'radius': vesicle['radius'],
                    'diameter': vesicle['diameter'],
                    'volume': vesicle['volume'],
                    'distance_to_az': vesicle.get('distance_to_az', np.nan),
                    'closest_membrane': vesicle.get('closest_membrane', 'unknown'),
                    'vesicle_distance_class': vesicle.get('vesicle_distance_class', 'unknown'),
                    'is_fusing': bool(vesicle.get('is_fusing', False)),
                    'is_close': bool(vesicle.get('is_close', False)),
                    'sphericity_volume': sphericities[i]['sphericity_volume']
                }
                vesicle_rows.append(row)
            
            # Save to global CSV
            df_vesicles = pd.DataFrame(vesicle_rows)
            global_csv = Path("results/vesicles/all_vesicle_data.csv")
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
                    df_combined = pd.concat([df_existing, df_vesicles], ignore_index=True)
                    df_combined.to_csv(global_csv, index=False)
                except Exception as e:
                    print(f"Error updating global all_vesicle_data.csv: {e}")
                    df_vesicles.to_csv(global_csv, index=False)
            else:
                df_vesicles.to_csv(global_csv, index=False)
            print(f"Appended {len(vesicle_rows)} vesicles to {global_csv}")
            # --- End global results ---
            
            # Print sphericity statistics
            print(f"Volume-based Sphericity (Wadell): {np.mean(sphericity_volume):.3f} ± {np.std(sphericity_volume):.3f}")
            
            results = {
                'vesicle_count': len(vesicles),
                'vesicle_volumes': volumes,
                'vesicle_diameters': diameters,
                'vesicle_positions': [v['center'] for v in vesicles],
                'total_vesicle_volume': float(np.sum(volumes)),
                'average_vesicle_diameter': float(np.mean(diameters)),
                'average_distance_to_az': float(np.mean(valid_distances)) if valid_distances else np.nan,
                'min_distance_to_az': float(np.min(valid_distances)) if valid_distances else np.nan,
                'max_distance_to_az': float(np.max(valid_distances)) if valid_distances else np.nan,
                'distance_std': float(np.std(valid_distances)) if valid_distances else np.nan,
                'nearby_vesicle_count': nearby_vesicle_count,
                'fusing_vesicle_count': int(sum(1 for v in vesicles if v.get('vesicle_distance_class') == 'fusing')),
                'close_vesicle_count': int(sum(1 for v in vesicles if v.get('vesicle_distance_class') == 'close')),
                'far_vesicle_count': int(sum(1 for v in vesicles if v.get('vesicle_distance_class') == 'far')),
                'unknown_distance_vesicle_count': int(sum(1 for v in vesicles if v.get('vesicle_distance_class') == 'unknown')),
                'average_sphericity_volume': float(np.mean(sphericity_volume)),
                'min_sphericity_volume': float(np.min(sphericity_volume)),
                'max_sphericity_volume': float(np.max(sphericity_volume)),
                'sphericity_volume_std': float(np.std(sphericity_volume)),
                'status': 'completed'
            }
        else:
            results = {
                'vesicle_count': 0,
                'vesicle_volumes': [],
                'vesicle_diameters': [],
                'vesicle_positions': [],
                'total_vesicle_volume': 0.0,
                'average_vesicle_diameter': 0.0,
                'nearby_vesicle_count': 0,
                'average_sphericity_volume': 0.0,
                'min_sphericity_volume': 0.0,
                'max_sphericity_volume': 0.0,
                'sphericity_volume_std': 0.0,
                'status': 'completed'
            }
        
        # Save results
        save_vesicle_results(vesicles, tomogram_path, alignment_dir=alignment_dir)
        
        # CSV export now handled by ResultsManager
        return results
        
    except Exception as e:
        print(f"Error in vesicle detection: {e}")
        return {
            'vesicle_count': 0,
            'vesicle_volumes': [],
            'vesicle_diameters': [],
            'vesicle_positions': [],
            'total_vesicle_volume': 0.0,
            'average_vesicle_diameter': 0.0,
            'status': 'error',
            'error': str(e)
        }


def measure_distances_to_az(tomogram_path, alignment_dir: str = "best_alignment") -> Dict[str, Any]:
    """
    Measure distances from vesicles to active zone.
    This function now uses pre-calculated distances from detect_vesicles.
    
    Args:
        tomogram_path (str or Path): Path to the tomogram file.
    
    Returns:
        Dictionary containing distance measurement results.
    """
    print(f"Loading vesicle distance measurements for {Path(tomogram_path).name}")
    
    try:
        # Load existing vesicle results
        tomogram_path = Path(tomogram_path)
        vesicles_file = tomogram_path / alignment_dir / "STT_results" / "vesicles" / "vesicle_results.json"
        
        if not vesicles_file.exists():
            print("No vesicle results found. Run detect_vesicles first.")
            return {
                'average_distance_to_az': 0.0,
                'min_distance_to_az': 0.0,
                'max_distance_to_az': 0.0,
                'distance_std': 0.0,
                'vesicle_az_distances': [],
                'status': 'error',
                'error': 'No vesicle results found'
            }
        
        with open(vesicles_file, 'r') as f:
            vesicle_data = json.load(f)
        
        vesicles = vesicle_data['vesicles']
        
        if not vesicles:
            return {
                'average_distance_to_az': 0.0,
                'min_distance_to_az': 0.0,
                'max_distance_to_az': 0.0,
                'distance_std': 0.0,
                'vesicle_az_distances': [],
                'status': 'completed'
            }
        
        # Extract pre-calculated distances
        distances_to_az = [v.get('distance_to_az', np.nan) for v in vesicles]
        valid_distances = [d for d in distances_to_az if np.isfinite(d)]
        
        # Calculate distance statistics from pre-calculated distances
        if valid_distances:
            results = {
                'average_distance_to_az': float(np.mean(valid_distances)),
                'min_distance_to_az': float(np.min(valid_distances)),
                'max_distance_to_az': float(np.max(valid_distances)),
                'distance_std': float(np.std(valid_distances)),
                'vesicle_az_distances': distances_to_az,
                'status': 'completed'
            }
        else:
            results = {
                'average_distance_to_az': 0.0,
                'min_distance_to_az': 0.0,
                'max_distance_to_az': 0.0,
                'distance_std': 0.0,
                'vesicle_az_distances': [],
                'status': 'completed'
            }
        
        # Calculate nearby vesicle count using the same logic as detect_vesicles
        nearby_vesicles = []
        for vesicle in vesicles:
            distance_class = vesicle.get('vesicle_distance_class')
            if distance_class in ('fusing', 'close'):
                nearby_vesicles.append(vesicle)
                continue
            distance_to_az = vesicle.get('distance_to_az', np.nan)
            if np.isfinite(distance_to_az) and distance_to_az <= 10.0:
                nearby_vesicles.append(vesicle)
        nearby_vesicle_count = len(nearby_vesicles)
        
        return results
        
    except Exception as e:
        print(f"Error in distance measurement: {e}")
        return {
            'average_distance_to_az': 0.0,
            'min_distance_to_az': 0.0,
            'max_distance_to_az': 0.0,
            'distance_std': 0.0,
            'vesicle_az_distances': [],
            'status': 'error',
            'error': str(e)
        }


def calculate_vesicle_sphericity(vesicle_data: Dict[str, Any]) -> Dict[str, float]:
    """
    Calculate sphericity of a vesicle using volume-based method (Wadell sphericity).
    
    Args:
        vesicle_data: Dictionary containing vesicle data with coordinates and fitted sphere
        
    Returns:
        Dictionary with sphericity measures
    """
    coordinates = np.array(vesicle_data['coordinates'])
    fitted_center = np.array(vesicle_data['center'])
    fitted_radius = vesicle_data['radius']
    
    if len(coordinates) < 4:
        return {
            'sphericity_volume': 0.0,
            'actual_volume': 0.0,
            'actual_surface_area': 0.0
        }
    
    # Method 1: Volume-based sphericity using fitted sphere
    fitted_volume = (4/3) * np.pi * fitted_radius**3
    fitted_surface_area = 4 * np.pi * fitted_radius**2
    
    # Calculate actual surface area using convex hull
    try:
        hull = ConvexHull(coordinates)
        actual_surface_area = hull.area
        actual_volume = hull.volume
        
        if actual_surface_area <= 0:
            raise ValueError(f"Invalid surface area from convex hull: {actual_surface_area}. Must be positive.")
        if actual_volume <= 0:
            raise ValueError(f"Invalid volume from convex hull: {actual_volume}. Must be positive.")
        
        # Volume-based sphericity (ψ = (π^(1/3) * (6V)^(2/3)) / A)
        sphericity_volume = (np.pi**(1/3) * (6 * actual_volume)**(2/3)) / actual_surface_area
        
    except Exception as e:
        raise ValueError(f"Convex hull calculation failed for sphericity: {e}. Cannot calculate sphericity.")
    
    # Calculate sphericity data using only volume-based metric
    sphericity_data = {
        'sphericity_volume': float(sphericity_volume),
        'actual_volume': float(actual_volume),
        'actual_surface_area': float(actual_surface_area)
    }
    
    return sphericity_data


def calculate_all_vesicle_sphericities(vesicles: List[Dict[str, Any]]) -> List[Dict[str, float]]:
    """
    Calculate sphericity for all vesicles.
    
    Args:
        vesicles: List of vesicle dictionaries
        
    Returns:
        List of sphericity dictionaries for each vesicle
    """
    sphericities = []
    
    for vesicle in vesicles:
        sphericity_data = calculate_vesicle_sphericity(vesicle)
        sphericities.append(sphericity_data)
    
    return sphericities
