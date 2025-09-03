# src/synaptic_tomo_tools/activezone.py

from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import ConvexHull
from scipy.spatial import KDTree


def calculate_membrane_volume(coordinates: np.ndarray) -> float:
    """
    Calculate volume of membrane segmentation using convex hull.
    
    Args:
        coordinates: Nx3 array of (x, y, z) coordinates in nm
        
    Returns:
        Volume in µm³
    """
    if len(coordinates) < 4:
        return 0.0  # Need at least 4 points for 3D volume
    
    try:
        # Calculate convex hull volume (in nm³)
        hull = ConvexHull(coordinates)
        volume_nm3 = hull.volume
        
        # Convert to µm³ (1 µm³ = 10^9 nm³)
        volume_um3 = volume_nm3 / 1e9
        
        return volume_um3
    except Exception as e:
        print(f"Error calculating volume: {e}")
        return 0.0


def save_membrane_volumes(membranes: Dict[str, List[np.ndarray]], tomogram_path):
    """
    Calculate and save volumes for each membrane segmentation.
    
    Args:
        membranes: Dictionary containing membrane coordinate arrays
        tomogram_path: Path to tomogram directory (str or Path)
    """
    tomogram_path = Path(tomogram_path)
    stt_results_dir = tomogram_path / "best_alignment" / "STT_results"
    
    # Create volumes directory
    volumes_dir = stt_results_dir / "volumes"
    volumes_dir.mkdir(parents=True, exist_ok=True)
    
    volumes_data = {}
    
    # Calculate presynaptic membrane volumes
    for i, coords in enumerate(membranes['presynaptic']):
        volume = calculate_membrane_volume(coords)
        membrane_name = f"presynaptic_membrane_{i+1}"
        volumes_data[membrane_name] = {
            'volume_um3': volume,
            'point_count': len(coords),
            'coordinates_file': f"presynapticmembranes_{i+1}.txt"
        }
        print(f"Presynaptic membrane {i+1} volume: {volume:.6f} µm³ ({len(coords)} points)")
    
    # Calculate postsynaptic membrane volumes
    for i, coords in enumerate(membranes['postsynaptic']):
        volume = calculate_membrane_volume(coords)
        membrane_name = f"postsynaptic_membrane_{i+1}"
        volumes_data[membrane_name] = {
            'volume_um3': volume,
            'point_count': len(coords),
            'coordinates_file': f"postsynapticmembranes_{i+1}.txt"
        }
        print(f"Postsynaptic membrane {i+1} volume: {volume:.6f} µm³ ({len(coords)} points)")
    
    # Save volumes to JSON file
    import json
    volumes_file = volumes_dir / "membrane_volumes.json"
    with open(volumes_file, 'w') as f:
        json.dump(volumes_data, f, indent=2, default=str)
    
    return volumes_data


def save_membrane_volumes_from_glb(membranes: Dict[str, List[Dict[str, np.ndarray]]], tomogram_path):
    """
    Calculate and save volumes for each membrane segmentation from GLB mesh data.
    
    Args:
        membranes: Dictionary containing membrane mesh data from GLB
        tomogram_path: Path to tomogram directory (str or Path)
    """
    import trimesh
    
    tomogram_path = Path(tomogram_path)
    stt_results_dir = tomogram_path / "best_alignment" / "STT_results"
    
    # Create volumes directory
    volumes_dir = stt_results_dir / "volumes"
    volumes_dir.mkdir(parents=True, exist_ok=True)
    
    volumes_data = {}
    
    # Calculate presynaptic membrane volumes from meshes
    for i, mesh_data in enumerate(membranes['presynaptic']):
        try:
            # Create trimesh object from vertices and faces
            mesh = trimesh.Trimesh(vertices=mesh_data['vertices'], faces=mesh_data['faces'])
            
            # Calculate volume (trimesh gives volume in nm³, convert to µm³)
            volume_um3 = mesh.volume / 1e9  # Convert nm³ to µm³
            
            membrane_name = f"presynaptic_membrane_{i+1}"
            volumes_data[membrane_name] = {
                'volume_um3': volume_um3,
                'vertex_count': len(mesh_data['vertices']),
                'face_count': len(mesh_data['faces']),
                'surface_area_um2': mesh.area / 1e6,  # Convert nm² to µm²
                'source': 'glb_mesh'
            }
            print(f"Presynaptic membrane {i+1} volume: {volume_um3:.6f} µm³ ({len(mesh_data['vertices'])} vertices, {len(mesh_data['faces'])} faces)")
        except Exception as e:
            print(f"Error calculating volume for presynaptic membrane {i+1}: {e}")
            # Fallback to convex hull method
            volume = calculate_membrane_volume(mesh_data['vertices'])
            membrane_name = f"presynaptic_membrane_{i+1}"
            volumes_data[membrane_name] = {
                'volume_um3': volume,
                'vertex_count': len(mesh_data['vertices']),
                'face_count': len(mesh_data['faces']),
                'source': 'convex_hull_fallback'
            }
            print(f"Presynaptic membrane {i+1} volume (fallback): {volume:.6f} µm³")
    
    # Calculate postsynaptic membrane volumes from meshes
    for i, mesh_data in enumerate(membranes['postsynaptic']):
        try:
            # Create trimesh object from vertices and faces
            mesh = trimesh.Trimesh(vertices=mesh_data['vertices'], faces=mesh_data['faces'])
            
            # Calculate volume (trimesh gives volume in nm³, convert to µm³)
            volume_um3 = mesh.volume / 1e9  # Convert nm³ to µm³
            
            membrane_name = f"postsynaptic_membrane_{i+1}"
            volumes_data[membrane_name] = {
                'volume_um3': volume_um3,
                'vertex_count': len(mesh_data['vertices']),
                'face_count': len(mesh_data['faces']),
                'surface_area_um2': mesh.area / 1e6,  # Convert nm² to µm²
                'source': 'glb_mesh'
            }
            print(f"Postsynaptic membrane {i+1} volume: {volume_um3:.6f} µm³ ({len(mesh_data['vertices'])} vertices, {len(mesh_data['faces'])} faces)")
        except Exception as e:
            print(f"Error calculating volume for postsynaptic membrane {i+1}: {e}")
            # Fallback to convex hull method
            volume = calculate_membrane_volume(mesh_data['vertices'])
            membrane_name = f"postsynaptic_membrane_{i+1}"
            volumes_data[membrane_name] = {
                'volume_um3': volume,
                'vertex_count': len(mesh_data['vertices']),
                'face_count': len(mesh_data['faces']),
                'source': 'convex_hull_fallback'
            }
            print(f"Postsynaptic membrane {i+1} volume (fallback): {volume:.6f} µm³")
    
    # Save volumes to JSON file
    import json
    volumes_file = volumes_dir / "membrane_volumes.json"
    with open(volumes_file, 'w') as f:
        json.dump(volumes_data, f, indent=2, default=str)
    
    return volumes_data


def load_membrane_volumes(tomogram_path) -> Dict[str, Any]:
    """
    Load membrane volumes from JSON file and calculate averages.
    
    Args:
        tomogram_path: Path to tomogram directory (str or Path)
        
    Returns:
        Dictionary containing volume statistics with averages for pre and post
    """
    tomogram_path = Path(tomogram_path)
    volumes_file = tomogram_path / "best_alignment" / "STT_results" / "volumes" / "membrane_volumes.json"
    
    if not volumes_file.exists():
        print(f"Warning: Membrane volumes file not found: {volumes_file}")
        return {
            'avg_presynaptic_volume_um3': 0.0,
            'avg_postsynaptic_volume_um3': 0.0,
            'presynaptic_count': 0,
            'postsynaptic_count': 0
        }
    
    try:
        import json
        with open(volumes_file, 'r') as f:
            volumes_data = json.load(f)
        
        # Separate presynaptic and postsynaptic volumes
        presynaptic_volumes = []
        postsynaptic_volumes = []
        
        for membrane_name, data in volumes_data.items():
            if membrane_name.startswith('presynaptic_membrane'):
                presynaptic_volumes.append(data['volume_um3'])
            elif membrane_name.startswith('postsynaptic_membrane'):
                postsynaptic_volumes.append(data['volume_um3'])
        
        # Calculate averages
        avg_presynaptic = np.mean(presynaptic_volumes) if presynaptic_volumes else 0.0
        avg_postsynaptic = np.mean(postsynaptic_volumes) if postsynaptic_volumes else 0.0
        
        print(f"Average presynaptic membrane volume: {avg_presynaptic:.6f} µm³ ({len(presynaptic_volumes)} membranes)")
        print(f"Average postsynaptic membrane volume: {avg_postsynaptic:.6f} µm³ ({len(postsynaptic_volumes)} membranes)")
        
        return {
            'avg_presynaptic_volume_um3': float(avg_presynaptic),
            'avg_postsynaptic_volume_um3': float(avg_postsynaptic),
            'presynaptic_count': len(presynaptic_volumes),
            'postsynaptic_count': len(postsynaptic_volumes)
        }
        
    except Exception as e:
        print(f"Error loading membrane volumes: {e}")
        return {
            'avg_presynaptic_volume_um3': 0.0,
            'avg_postsynaptic_volume_um3': 0.0,
            'presynaptic_count': 0,
            'postsynaptic_count': 0,
            'error': str(e)
        }


def import_membrane_segmentations(tomogram_path) -> Dict[str, List[np.ndarray]]:
    """
    Import presynaptic and postsynaptic membrane segmentation files.
    
    Args:
        tomogram_path: Path to the tomogram directory (str or Path)
        
    Returns:
        Dictionary containing lists of coordinate arrays for each membrane type
    """
    tomogram_path = Path(tomogram_path)
    aunps_dir = tomogram_path / "best_alignment" / "aunps"
    
    if not aunps_dir.exists():
        raise FileNotFoundError(f"AuNPs directory not found: {aunps_dir}")
    
    # Initialize results
    membranes = {
        'presynaptic': [],
        'postsynaptic': []
    }
    
    # Find all presynaptic membrane files
    presyn_files = list(aunps_dir.glob("presynapticmembranes_*.txt"))
    postsyn_files = list(aunps_dir.glob("postsynapticmembranes_*.txt"))
    
    print(f"Found {len(presyn_files)} presynaptic membrane files")
    print(f"Found {len(postsyn_files)} postsynaptic membrane files")
    
    # Import presynaptic membranes
    for file_path in sorted(presyn_files):
        try:
            coords = np.loadtxt(file_path, delimiter=None)  # Auto-detect delimiter
            membranes['presynaptic'].append(coords)
        except Exception as e:
            print(f"Error importing {file_path}: {e}")
    
    # Import postsynaptic membranes
    for file_path in sorted(postsyn_files):
        try:
            coords = np.loadtxt(file_path, delimiter=None)  # Auto-detect delimiter
            membranes['postsynaptic'].append(coords)
        except Exception as e:
            print(f"Error importing {file_path}: {e}")
    
    return membranes

def import_membrane_segmentations_from_glb(tomogram_path) -> Dict[str, List[Dict[str, np.ndarray]]]:
    """
    Import presynaptic and postsynaptic membrane segmentations from a GLB file.
    
    Args:
        tomogram_path: Path to the tomogram directory (str or Path)
        
    Returns:
        Dictionary containing lists of coordinate arrays, faces, and normals for each membrane type
    """
    import trimesh

    tomogram_path = Path(tomogram_path)
    aunps_dir = tomogram_path / "best_alignment" / "aunps"
    
    if not aunps_dir.exists():
        raise FileNotFoundError(f"AuNPs directory not found: {aunps_dir}")
    
    # Initialize results
    membranes = {
        'presynaptic': [],
        'postsynaptic': []
    }

    presyn_glb = aunps_dir / "presynapticmembranes.glb"
    with open(presyn_glb, "rb") as f:
        presyn = trimesh.exchange.gltf.load_glb(f)
    postsyn_glb = aunps_dir / "postsynapticmembranes.glb"
    with open(postsyn_glb, "rb") as f:
        postsyn = trimesh.exchange.gltf.load_glb(f)

    membranes['presynaptic'] = [{'vertices': mesh['vertices'][:, [0, 2, 1]] * np.array([10, -10, 10]),'faces':mesh['faces'],'normals':mesh['vertex_normals'][:, [0, 2, 1]] * np.array([10, -10, 10])} for mesh in presyn["geometry"].values()]
    membranes['postsynaptic'] = [{'vertices': mesh['vertices'][:, [0, 2, 1]] * np.array([10, -10, 10]),'faces':mesh['faces'],'normals':mesh['vertex_normals'][:, [0, 2, 1]] * np.array([10, -10, 10])} for mesh in postsyn["geometry"].values()]

    # Save volumes to STT_results/volumes
    save_membrane_volumes_from_glb(membranes, tomogram_path)

    return membranes


def find_active_zones(membranes: Dict[str, List[np.ndarray]], distance_threshold: float = 40.0) -> Dict[str, Any]:
    """
    Find active zones by identifying presynaptic points within distance_threshold of postsynaptic points.
    Uses KD-tree for efficient spatial queries.
    
    Args:
        membranes: Dictionary containing membrane coordinate arrays
        distance_threshold: Distance threshold in nm (default: 40.0)
        
    Returns:
        Dictionary containing active zone information and segmentations
    """
    active_zones = {}
    active_zone_count = 0
    
    presyn_membranes = membranes['presynaptic']
    postsyn_membranes = membranes['postsynaptic']
        
    for pre_idx, presyn_coords in enumerate(presyn_membranes):
        for post_idx, postsyn_coords in enumerate(postsyn_membranes):

            # Build KD-tree for postsynaptic points
            post_tree = KDTree(postsyn_coords)
            
            # Find presynaptic points within threshold of any postsynaptic point
            distances_pre, indices_pre = post_tree.query(presyn_coords, distance_upper_bound=distance_threshold)
            
            # Get active presynaptic points (those within threshold)
            active_pre_mask = distances_pre <= distance_threshold
            active_pre_indices = np.where(active_pre_mask)[0]
            active_pre_coords = presyn_coords[active_pre_indices] if len(active_pre_indices) > 0 else np.array([])
            
            # Find postsynaptic points within threshold of active presynaptic points
            active_post_indices = np.array([])
            if len(active_pre_coords) > 0:
                # Build KD-tree for active presynaptic points
                pre_tree = KDTree(active_pre_coords)
                
                # Find postsynaptic points within threshold of active presynaptic points
                distances_post, indices_post = pre_tree.query(postsyn_coords, distance_upper_bound=distance_threshold)
                
                # Get active postsynaptic points
                active_post_mask = distances_post <= distance_threshold
                active_post_indices = np.where(active_post_mask)[0]
            
            # If we found an active zone
            if len(active_pre_indices) > 0 or len(active_post_indices) > 0:
                active_zone_count += 1
                zone_name = f"active_zone_pre{pre_idx+1}_post{post_idx+1}"
                
                # Calculate distance statistics for active presynaptic points
                if len(active_pre_coords) > 0:
                    distances_active = distances_pre[active_pre_indices]
                    min_dist = np.min(distances_active)
                    max_dist = np.max(distances_active)
                    avg_dist = np.mean(distances_active)
                else:
                    min_dist = float('inf')
                    max_dist = 0
                    avg_dist = 0
                
                active_zones[zone_name] = {
                    'presynaptic_membrane_index': pre_idx + 1,
                    'postsynaptic_membrane_index': post_idx + 1,
                    'active_presynaptic_points': active_pre_coords,
                    'active_postsynaptic_points': postsyn_coords[active_post_indices] if len(active_post_indices) > 0 else np.array([]),
                    'active_presynaptic_indices': active_pre_indices,
                    'active_postsynaptic_indices': active_post_indices,
                    'min_distance': min_dist,
                    'max_distance': max_dist,
                    'avg_distance': avg_dist,
                    'active_pre_count': len(active_pre_indices),
                    'active_post_count': len(active_post_indices)
                }
                
                print(f"Found active zone: {zone_name} with {len(active_pre_indices)} presynaptic and {len(active_post_indices)} postsynaptic points")
            else:
                print(f"No active zone found between presynaptic {pre_idx+1} and postsynaptic {post_idx+1}")
    
    return {
        'active_zones': active_zones,
        'total_active_zones': active_zone_count,
        'distance_threshold': distance_threshold
    }

def find_active_zones_from_glb(membranes: Dict[str, List[Dict[str, np.ndarray]]], distance_range: Tuple[float, float] = (10.0, 40.0)) -> Dict[str, Any]:
    """
    Find active zones by identifying presynaptic points within distance_range of postsynaptic points.
    Uses KD-tree for efficient spatial queries. Only vertices with normals pointing towards the other side are considered.
    
    Args:
        membranes: Dictionary containing membrane coordinate arrays from GLB
        distance_range: Distance range in nm as (min_distance, max_distance) (default: (10.0, 40.0))
        
    Returns:
        Dictionary containing active zone information and segmentations
    """
    import trimesh 

    min_distance, max_distance = distance_range
    
    active_zones = {}
    active_zone_count = 0
    presyn_membranes = membranes['presynaptic']
    postsyn_membranes = membranes['postsynaptic']
    for pre_idx, presyn_data in enumerate(presyn_membranes):
        presyn_coords = presyn_data['vertices']
        presyn_normals = presyn_data['normals']
        
        for post_idx, postsyn_data in enumerate(postsyn_membranes):
            postsyn_coords = postsyn_data['vertices']
            postsyn_normals = postsyn_data['normals']
            
            # Filter presynaptic points with normals pointing towards postsynaptic points               
            # Build KD-tree for postsynaptic points
            post_tree = KDTree(postsyn_coords)
            
            # Find presynaptic points within max_distance of any postsynaptic point
            distances_pre, indices_pre = post_tree.query(presyn_coords, distance_upper_bound=max_distance)
            
            # Get active presynaptic points (those within distance range)
            active_pre_mask = (distances_pre >= min_distance) & (distances_pre <= max_distance)
            # Filter presynaptic points based on normals pointing towards postsynaptic points
            active_pre_mask[active_pre_mask] = (np.sum(presyn_normals[active_pre_mask] * postsyn_normals[indices_pre[active_pre_mask]], axis=1) < 0)

            active_pre_indices = np.where(active_pre_mask)[0]
            active_pre_coords = presyn_coords[active_pre_indices] if len(active_pre_indices) > 0 else np.array([])
            
            # Find postsynaptic points within distance range of active presynaptic points
            active_post_indices = np.array([])
            if len(active_pre_coords) > 0:
                # Build KD-tree for active presynaptic points
                pre_tree = KDTree(active_pre_coords)
                
                # Find postsynaptic points within max_distance of active presynaptic points
                distances_post, indices_post = pre_tree.query(postsyn_coords, distance_upper_bound=max_distance)
                
                # Get active postsynaptic points (those within distance range)
                active_post_mask = (distances_post >= min_distance) & (distances_post <= max_distance)
                # Filter postsynaptic points based on normals pointing towards presynaptic points
                active_post_mask[active_post_mask] = (np.sum(postsyn_normals[active_post_mask] * presyn_normals[active_pre_mask][indices_post[active_post_mask]], axis=1) < 0)
                active_post_indices = np.where(active_post_mask)[0]
            
            # If we found an active zone
            if len(active_pre_indices) > 0 or len(active_post_indices) > 0:
                active_zone_count += 1
                zone_name = f"active_zone_pre{pre_idx+1}_post{post_idx+1}"
                
                # Calculate distance statistics for active presynaptic points
                if len(active_pre_coords) > 0:
                    distances_active = distances_pre[active_pre_indices]
                    min_dist = np.min(distances_active)
                    max_dist = np.max(distances_active)
                    avg_dist = np.mean(distances_active)
                else:
                    min_dist = float('inf')
                    max_dist = 0
                    avg_dist = 0
                
                pre_mesh = trimesh.Trimesh(vertices=presyn_data['vertices'], faces=presyn_data['faces'])
                post_mesh = trimesh.Trimesh(vertices=postsyn_data['vertices'], faces=postsyn_data['faces'])
                # Get only faces that contain only active presynaptic points
                active_pre_faces_mask = np.isin(pre_mesh.faces, active_pre_indices).all(axis=1)
                active_pre_faces_indices = np.where(active_pre_faces_mask)[0]
                active_pre_mesh = pre_mesh.submesh([active_pre_faces_indices], append=True)

                active_post_faces_mask = np.isin(postsyn_data['faces'], active_post_indices).all(axis=1)
                active_post_faces_indices = np.where(active_post_faces_mask)[0]
                active_post_mesh = post_mesh.submesh([active_post_faces_indices], append=True)
                # Get area of active presynaptic mesh
                active_pre_area = active_pre_mesh.area / 1e6  # Convert to µm²
                active_zones[zone_name] = {
                    'presynaptic_membrane_index': pre_idx + 1,
                    'postsynaptic_membrane_index': post_idx + 1,
                    'active_presynaptic_points': active_pre_coords,
                    'active_presynaptic_faces': active_pre_mesh.faces,
                    'active_presynaptic_mesh': active_pre_mesh,
                    'active_presynaptic_area': active_pre_area,
                    'active_postsynaptic_points': postsyn_coords[active_post_indices] if len(active_post_indices) > 0 else np.array([]),
                    'active_postsynaptic_mesh': active_post_mesh,
                    'active_presynaptic_indices': active_pre_indices,
                    'active_postsynaptic_indices': active_post_indices,
                    'min_distance': min_dist,
                    'max_distance': max_dist,
                    'avg_distance': avg_dist,
                    'active_pre_count': len(active_pre_indices),
                    'active_post_count': len(active_post_indices)
                }
                #
                print(f"Found active zone: {zone_name} with {len(active_pre_indices)} presynaptic and {len(active_post_indices)} postsynaptic points")
            else:
                print(f"No active zone found between presynaptic {pre_idx+1} and postsynaptic {post_idx+1}")
    return {
        'active_zones': active_zones,
        'total_active_zones': active_zone_count,
        'distance_range': distance_range
    }


def save_active_zone_segmentations(active_zones: Dict[str, Any], tomogram_path):
    """
    Save active zone segmentations to files.
    
    Args:
        active_zones: Active zones dictionary from find_active_zones
        tomogram_path: Path to tomogram directory (str or Path)
    """
    tomogram_path = Path(tomogram_path)
    stt_results_dir = tomogram_path / "best_alignment" / "STT_results"
    
    # Create active zone directory
    active_zone_dir = stt_results_dir / "active_zones"
    active_zone_dir.mkdir(parents=True, exist_ok=True)
    
    for zone_name, zone_data in active_zones['active_zones'].items():
        # Save active presynaptic points
        if len(zone_data['active_presynaptic_points']) > 0:
            pre_filename = f"{zone_name}_pre.txt"
            pre_filepath = active_zone_dir / pre_filename
            np.savetxt(pre_filepath, zone_data['active_presynaptic_points'], fmt='%.6e')
        
        # Save active postsynaptic points
        if len(zone_data['active_postsynaptic_points']) > 0:
            post_filename = f"{zone_name}_post.txt"
            post_filepath = active_zone_dir / post_filename
            np.savetxt(post_filepath, zone_data['active_postsynaptic_points'], fmt='%.6e')


def define_active_zone(tomogram_path) -> Dict[str, Any]:
    """
    Define active zone in tomogram.
    
    Args:
        tomogram_path: Path to the tomogram file.
    
    Returns:
        Dictionary containing active zone analysis results.
    """
    print(f"Defining active zone in {Path(tomogram_path).name}")
    
    # Import membrane segmentations from GLB
    try:
        membranes = import_membrane_segmentations_from_glb(tomogram_path)
        
        # Find active zones from GLB
        active_zones = find_active_zones_from_glb(membranes, distance_range=(10.0, 40.0))
        
        # Save active zone segmentations
        save_active_zone_segmentations(active_zones, tomogram_path)
        
        # Calculate summary statistics
        total_active_pre_points = sum(len(zone['active_presynaptic_points']) for zone in active_zones['active_zones'].values())
        total_active_post_points = sum(len(zone['active_postsynaptic_points']) for zone in active_zones['active_zones'].values())
        
        # Calculate active zone areas using mesh surface area from GLB
        active_zone_areas = []
        for zone_name, zone_data in active_zones['active_zones'].items():
            # Use the active presynaptic area calculated from GLB mesh
            if 'active_presynaptic_area' in zone_data:
                active_zone_areas.append(zone_data['active_presynaptic_area'])
                print(f"Active zone area {zone_name}: {zone_data['active_presynaptic_area']:.6f} µm²")
            else:
                # Fallback to convex hull calculation if mesh area not available
                presynaptic_points = zone_data['active_presynaptic_points']
                
                if len(presynaptic_points) >= 4:  # Need at least 4 points for 3D convex hull
                    try:
                        # Calculate 3D convex hull surface area of presynaptic points
                        hull = ConvexHull(presynaptic_points)
                        surface_area_nm2 = hull.area  # Surface area in nm²
                        
                        # Divide by 2 to estimate presynaptic area
                        presynaptic_area_nm2 = surface_area_nm2 / 2.0
                        
                        # Convert to µm² (1 µm² = 10^6 nm²)
                        presynaptic_area_um2 = presynaptic_area_nm2 / 1e6
                        
                        active_zone_areas.append(presynaptic_area_um2)
                        print(f"Active zone area {zone_name} (convex hull): {presynaptic_area_um2:.6f} µm²")
                        
                    except Exception as e:
                        print(f"Error calculating convex hull for {zone_name}: {e}")
                        # Fallback to 2D convex hull area
                        try:
                            coords_2d = presynaptic_points[:, :2]  # Use only X and Y
                            hull_2d = ConvexHull(coords_2d)
                            area_nm2 = hull_2d.area
                            area_um2 = area_nm2 / 1e6
                            active_zone_areas.append(area_um2)
                            print(f"Active zone {zone_name} (2D fallback): {area_um2:.6f} µm²")
                        except Exception as e2:
                            print(f"Error with 2D fallback for {zone_name}: {e2}")
                            # Final fallback to bounding box
                            bbox_size = np.max(presynaptic_points[:, :2], axis=0) - np.min(presynaptic_points[:, :2], axis=0)
                            area_nm2 = bbox_size[0] * bbox_size[1]
                            area_um2 = area_nm2 / 1e6
                            active_zone_areas.append(area_um2)
                            print(f"Active zone {zone_name} (bbox fallback): {area_um2:.6f} µm²")
        
        avg_active_zone_area = np.mean(active_zone_areas) if active_zone_areas else 0.0
        
        # Load membrane volumes
        volumes_data = load_membrane_volumes(tomogram_path)
        
        results = {
            'active_zone_count': active_zones['total_active_zones'],
            'total_active_pre_points': total_active_pre_points,
            'total_active_post_points': total_active_post_points,
            'avg_active_zone_area': avg_active_zone_area,
            'distance_range': active_zones['distance_range'],
            'active_zone_names': list(active_zones['active_zones'].keys()),
            'membrane_volumes': volumes_data,
            'status': 'completed'
        }
        
    except Exception as e:
        print(f"Error defining active zones: {e}")
        results = {
            'active_zone_count': 0,
            'total_active_pre_points': 0,
            'total_active_post_points': 0,
            'avg_active_zone_area': 0.0,
            'distance_range': (10.0, 40.0),
            'active_zone_names': [],
            'membrane_volumes': {},
            'status': 'error',
            'error_message': str(e)
        }
    
    return results


def import_active_zone_segmentations(tomogram_path) -> Dict[str, Any]:
    """
    Import active zone segmentation files.
    
    Args:
        tomogram_path: Path to tomogram directory (str or Path)
        
    Returns:
        Dictionary containing active zone segmentations
    """
    tomogram_path = Path(tomogram_path)
    active_zone_dir = tomogram_path / "best_alignment" / "STT_results" / "active_zones"
    
    if not active_zone_dir.exists():
        raise FileNotFoundError(f"Active zones directory not found: {active_zone_dir}")
    
    active_zones = {}
    
    # Find all active zone files
    pre_files = list(active_zone_dir.glob("*_pre.txt"))
    post_files = list(active_zone_dir.glob("*_post.txt"))
    
    print(f"Found {len(pre_files)} active presynaptic files")
    print(f"Found {len(post_files)} active postsynaptic files")
    
    # Group files by active zone name
    for pre_file in pre_files:
        # Extract zone name by removing the final '_pre' from the filename stem
        stem = pre_file.stem
        if stem.endswith('_pre'):
            zone_name = stem[:-4]  # Remove last 4 characters ('_pre')
        else:
            zone_name = stem  # Fallback if no '_pre' suffix
        
        post_file = active_zone_dir / f"{zone_name}_post.txt"
        
        if post_file.exists():
            try:
                pre_coords = np.loadtxt(pre_file, delimiter=None)
                post_coords = np.loadtxt(post_file, delimiter=None)
                
                active_zones[zone_name] = {
                    'presynaptic_coords': pre_coords,
                    'postsynaptic_coords': post_coords,
                    'presynaptic_count': len(pre_coords),
                    'postsynaptic_count': len(post_coords)
                }
                
                print(f"  ✓ Imported active zone: {zone_name} ({len(pre_coords)} pre, {len(post_coords)} post points)")
                
            except Exception as e:
                print(f"  ✗ Error importing active zone {zone_name}: {e}")
        else:
            print(f"  ✗ Post file not found: {post_file}")
            # Try to find any post file that might match
            potential_matches = list(active_zone_dir.glob(f"*{zone_name}*"))
            if potential_matches:
                print(f"    Potential matches: {[f.name for f in potential_matches]}")
    
    return active_zones


def calculate_cleft_width_for_active_zone(pre_coords: np.ndarray, post_coords: np.ndarray) -> Dict[str, Any]:
    """
    Calculate cleft width for a single active zone using KD-tree.
    
    Args:
        pre_coords: Presynaptic active zone coordinates
        post_coords: Postsynaptic active zone coordinates
        
    Returns:
        Dictionary with cleft width statistics
    """
    if len(pre_coords) == 0 or len(post_coords) == 0:
        return {
            'average_cleft_width': 0.0,
            'cleft_width_std': 0.0,
            'min_cleft_width': 0.0,
            'max_cleft_width': 0.0,
            'measurement_count': 0
        }
    
    # Build KD-tree for postsynaptic points
    post_tree = KDTree(post_coords)
    
    # Find closest postsynaptic point for each presynaptic point
    distances_pre_to_post, indices_pre_to_post = post_tree.query(pre_coords)
    
    # Build KD-tree for presynaptic points
    pre_tree = KDTree(pre_coords)
    
    # Find closest presynaptic point for each postsynaptic point
    distances_post_to_pre, indices_post_to_pre = pre_tree.query(post_coords)
    
    # Combine all distances
    all_distances = np.concatenate([distances_pre_to_post, distances_post_to_pre])
    
    return {
        'average_cleft_width': float(np.mean(all_distances)),
        'cleft_width_std': float(np.std(all_distances)),
        'min_cleft_width': float(np.min(all_distances)),
        'max_cleft_width': float(np.max(all_distances)),
        'measurement_count': len(all_distances)
    }


def calculate_cleft_width(tomogram_path) -> Dict[str, Any]:
    """
    Calculate synaptic cleft width for all active zones.
    
    Args:
        tomogram_path: Path to the tomogram file.
    
    Returns:
        Dictionary containing cleft width analysis results.
    """
    print(f"Calculating cleft width in {Path(tomogram_path).name}")
    
    try:
        # Import active zone segmentations
        active_zones = import_active_zone_segmentations(tomogram_path)
        
        if not active_zones:
            print("No active zones found for cleft width calculation")
            return {
                'average_cleft_width': 0.0,
                'cleft_width_std': 0.0,
                'min_cleft_width': 0.0,
                'max_cleft_width': 0.0,
                'active_zone_count': 0,
                'status': 'no_active_zones'
            }
        
        # Calculate cleft width for each active zone
        cleft_results = {}
        all_distances = []
        
        for zone_name, zone_data in active_zones.items():
            cleft_stats = calculate_cleft_width_for_active_zone(
                zone_data['presynaptic_coords'],
                zone_data['postsynaptic_coords']
            )
            
            cleft_results[zone_name] = cleft_stats
            
            # Collect all individual distance measurements for true min/max calculation
            if cleft_stats['measurement_count'] > 0:
                # Recalculate the actual distances for this zone to get all measurements
                pre_coords = zone_data['presynaptic_coords']
                post_coords = zone_data['postsynaptic_coords']
                
                if len(pre_coords) > 0 and len(post_coords) > 0:
                    # Build KD-tree for postsynaptic points
                    post_tree = KDTree(post_coords)
                    
                    # Find closest postsynaptic point for each presynaptic point
                    distances_pre_to_post, _ = post_tree.query(pre_coords)
                    
                    # Build KD-tree for presynaptic points
                    pre_tree = KDTree(pre_coords)
                    
                    # Find closest presynaptic point for each postsynaptic point
                    distances_post_to_pre, _ = pre_tree.query(post_coords)
                    
                    # Add all distances to the global list
                    all_distances.extend(distances_pre_to_post)
                    all_distances.extend(distances_post_to_pre)
        
        # Calculate overall statistics from all individual measurements
        if all_distances:
            overall_stats = {
                'average_cleft_width': float(np.mean(all_distances)),
                'cleft_width_std': float(np.std(all_distances)),
                'min_cleft_width': float(np.min(all_distances)),
                'max_cleft_width': float(np.max(all_distances)),
                'active_zone_count': len(active_zones),
                'total_measurements': len(all_distances),
                'individual_zone_results': cleft_results,
                'status': 'completed'
            }
            
            # --- Append to global results/all_cleft_distances.csv ---
            tomogram_name = Path(tomogram_path).name
            # Extract set_name from tomogram path
            path_parts = Path(tomogram_path).parts
            set_name = "unknown"
            for i, part in enumerate(path_parts):
                if part.endswith("_tomograms") and i > 0:
                    set_name = part.replace("_tomograms", "")
                    break
            
            # Prepare individual cleft distance data for CSV
            import pandas as pd
            cleft_rows = []
            for distance in all_distances:
                row = {
                    'tomogram_name': tomogram_name,
                    'set_name': set_name,
                    'cleft_width': distance
                }
                cleft_rows.append(row)
            
            # Save to global CSV
            df_cleft = pd.DataFrame(cleft_rows)
            global_csv = Path("results/all_cleft_distances.csv")
            global_csv.parent.mkdir(parents=True, exist_ok=True)
            if global_csv.exists():
                try:
                    df_existing = pd.read_csv(global_csv)
                    # Remove existing data for this tomogram
                    df_existing = df_existing[df_existing['tomogram_name'] != tomogram_name]
                    df_combined = pd.concat([df_existing, df_cleft], ignore_index=True)
                    df_combined.to_csv(global_csv, index=False)
                except Exception as e:
                    print(f"Error updating global all_cleft_distances.csv: {e}")
                    df_cleft.to_csv(global_csv, index=False)
            else:
                df_cleft.to_csv(global_csv, index=False)
            print(f"Appended {len(cleft_rows)} cleft measurements to {global_csv}")
            # --- End global results ---
        else:
            overall_stats = {
                'average_cleft_width': 0.0,
                'cleft_width_std': 0.0,
                'min_cleft_width': 0.0,
                'max_cleft_width': 0.0,
                'active_zone_count': len(active_zones),
                'total_measurements': 0,
                'individual_zone_results': cleft_results,
                'status': 'no_measurements'
            }
        
        print(f"Overall cleft width: {overall_stats['average_cleft_width']:.2f} ± {overall_stats['cleft_width_std']:.2f} nm")
        print(f"Calculated cleft width for {len(active_zones)} active zones")
        
        return overall_stats
        
    except Exception as e:
        print(f"Error calculating cleft width: {e}")
        return {
            'average_cleft_width': 0.0,
            'cleft_width_std': 0.0,
            'min_cleft_width': 0.0,
            'max_cleft_width': 0.0,
            'active_zone_count': 0,
            'total_measurements': 0,
            'individual_zone_results': {},
            'status': 'error',
            'error_message': str(e)
        }


def define_active_zonogram(active_zones):
    """
    Define active zonogram from active zones.
    
    Args:
        active_zones: Dictionary containing active zone data.
        
    Returns:
        Dictionary containing active zonogram results.
    """
    print("Defining active zonogram")
    from torch_affine_utils.transforms_3d import T
    from torch_affine_utils.utils import homogenise_coordinates
    import torch
    import einops
    
    if not active_zones or 'active_zones' not in active_zones:
        print("No active zones found for zonogram definition")
        return {
            'status': 'no_active_zones',
            'active_zone_count': 0,
            'zonogram_data': {}
        }
    
    # Prepare zonogram data
    zonogram_data = {}
    
    for zone_name, zone_data in active_zones['active_zones'].items():
        # Define center of active zonogram
        center_presyn = np.mean(zone_data['active_presynaptic_points'], axis=0) if len(zone_data['active_presynaptic_points']) > 0 else np.zeros(3)
        center_postsyn = np.mean(zone_data['active_postsynaptic_points'], axis=0) if len(zone_data['active_postsynaptic_points']) > 0 else np.zeros(3)
        center = (center_presyn + center_postsyn) / 2.0
        # Construct coordinate system
        # Get 100 random points in postsynapse
        post_points_sel = zone_data['active_postsynaptic_points'][np.random.choice(zone_data['active_postsynaptic_points'].shape[0], 100, replace=False)]
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
        M = M @ T(-center)

        # Calculate the extent of the active zonogram
        # Use the maximum distance from the center to any active presynaptic or postsynaptic point
        all_points = homogenise_coordinates(torch.tensor(np.concatenate([zone_data['active_presynaptic_points'], zone_data['active_postsynaptic_points']]), dtype=torch.float32))
        transformed_points = M @ einops.rearrange(all_points, 'b xyzw -> b xyzw 1')
        transformed_points = einops.rearrange(transformed_points, 'b xyzw 1 -> b xyzw')[:, :3]
        max_extent = torch.max(torch.abs(transformed_points), dim=0)[0].numpy().astype(int) * 2 + 10  # Add 10 nm padding
        # Define extent of active zonogram
        zonogram_data[zone_name] = {
            'center': center,
            'transformation_matrix': M.numpy(),
            'extent': max_extent
        }

    return {
        'status': 'completed',
        'active_zone_count': len(active_zones['active_zones']),
        'zonogram_data': zonogram_data
    }


def extract_active_zonogram(active_zonograms, active_zones, tomo_path, tomo_type='ddw'):
    """
    Extract active zonogram data.
    
    Args:
        active_zonograms: Dictionary containing active zonogram data.
        active_zones: Dictionary containing active zone data.
    """
    import mrcfile
    import torch
    from torch_transform_image import affine_transform_image_3d
    from torch_affine_utils.transforms_3d import T
    
    print("Rendering active zonogram")
    
    if not active_zonograms or 'zonogram_data' not in active_zonograms:
        print("No active zonograms found for rendering")
        return {
            'status': 'no_active_zonograms',
            'active_zone_count': 0,
            'rendered_zonograms': []
        }
    
    # Prepare rendering data
    rendered_zonograms = {}
    # Open tomogram

    mrcs = list((Path(tomo_path) / 'best_alignment').glob(f'*{tomo_type}.mrc'))
    if not mrcs:
        return None, None
    with mrcfile.open(mrcs[0], 'r') as mrc:
        data = torch.tensor(mrc.data)
    
    for zone_name, zone_data in active_zonograms['zonogram_data'].items():
        if zone_name not in active_zones['active_zones']:
            continue
        
        # Get active zone information
        active_zone = active_zones['active_zones'][zone_name]

        new_center = zone_data['extent'] // 2

        M = torch.tensor(zone_data['transformation_matrix'], dtype=torch.float32)
        M = T(new_center) @ M

        transformed_tomo = affine_transform_image_3d(
            image=data,
            matrices=torch.linalg.inv(M),
            interpolation='trilinear',
            zyx_matrices=False,
            output_shape=tuple(zone_data['extent'][::-1]),

        )
        rendered_zonograms[zone_name] = {
            'transformed_tomogram': transformed_tomo.numpy(),
        }

        
        
    
    return {
        'status': 'completed',
        'active_zone_count': len(active_zones['active_zones']),
        'rendered_zonograms': rendered_zonograms
    }


def transform_coordinates_to_active_zonogram(
        coordinates: np.ndarray, 
        active_zonogram: Dict[str, Any],
        eliminate_coordinates_outside = True
        ) -> np.ndarray:
    """
        Transforms a set of 3D coordinates into the space of an active zonogram using a provided transformation matrix.

        Parameters
        ----------
        coordinates : np.ndarray
            An array of shape (N, 3) or (N, 4) containing the coordinates to be transformed.
        active_zonogram : Dict[str, Any]
            A dictionary containing the zonogram's transformation matrix ('transformation_matrix') and its spatial extent ('extent').
            - 'transformation_matrix': A 4x4 affine transformation matrix.
            - 'extent': A tuple or array specifying the size of the zonogram in each dimension (x, y, z).
        eliminate_coordinates_outside : bool, optional
            If True, coordinates that fall outside the zonogram's extent after transformation are removed from the output.
            Default is True.

        Returns
        -------
        np.ndarray
            The transformed coordinates as a NumPy array of shape (M, 3), where M <= N depending on filtering.

        Notes
        -----
        - The function uses PyTorch for tensor operations and applies affine transformations in homogeneous coordinates.
        - Coordinates outside the zonogram's extent are optionally eliminated.
        - The transformation centers the zonogram before applying the affine matrix.    
    """
    import torch
    from torch_affine_utils.transforms_3d import T
    from torch_affine_utils.utils import homogenise_coordinates
    import einops

    coordinates = torch.tensor(coordinates, dtype=torch.float32)
    M = torch.tensor(active_zonogram['transformation_matrix'])
    new_center = active_zonogram['extent'] // 2
    M = T(new_center) @ M
    coordinates = homogenise_coordinates(coordinates)
    transformed_coordinates = M @ einops.rearrange(coordinates, 'b xyzw -> b xyzw 1')
    transformed_coordinates = einops.rearrange(transformed_coordinates, 'b xyzw 1 -> b xyzw')[:, :3]
    if eliminate_coordinates_outside:
        transformed_coordinates = transformed_coordinates[(
            (transformed_coordinates[:, 0] >= 0) &
            (transformed_coordinates[:, 0] < active_zonogram['extent'][0]) &
            (transformed_coordinates[:, 1] >= 0) &
            (transformed_coordinates[:, 1] < active_zonogram['extent'][1]) &
            (transformed_coordinates[:, 2] >= 0) &
            (transformed_coordinates[:, 2] < active_zonogram['extent'][2])
        )]
    return transformed_coordinates.numpy()