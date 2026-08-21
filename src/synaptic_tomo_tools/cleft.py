# src/synaptic_tomo_tools/cleft.py

from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np

from .alignment_utils import require_alignment_dir
from scipy.spatial.distance import cdist, pdist
from scipy.spatial import ConvexHull, KDTree


def _collect_cleft_surface_points(zone_data: Dict[str, Any]) -> np.ndarray:
    """All designated synaptic-cleft surface points (pre/post synaptic, inner and outer)."""
    point_keys = (
        'active_presynaptic_points',
        'active_presynaptic_outer_points',
        'active_presynaptic_inner_points',
        'active_postsynaptic_points',
        'active_postsynaptic_outer_points',
        'active_postsynaptic_inner_points',
    )
    chunks: List[np.ndarray] = []
    for key in point_keys:
        pts = zone_data.get(key)
        if pts is not None and len(pts) > 0:
            chunks.append(np.asarray(pts, dtype=float))
    if not chunks:
        return np.zeros((0, 3))
    return np.vstack(chunks)


def _cleft_membrane_area_from_hull_um2(
    inner_points: np.ndarray,
    outer_points: np.ndarray,
) -> float:
    """
    Estimate one membrane sheet area (µm²) from inner+outer synaptic-cleft surface points.

    Builds a 3D convex hull over both sheets and uses half the hull surface area as the
    synaptic-cleft membrane area (coordinates in nm; converted to µm²).
    """
    chunks: List[np.ndarray] = []
    for pts in (inner_points, outer_points):
        if pts is not None and len(pts) > 0:
            chunks.append(np.asarray(pts, dtype=float))
    if not chunks:
        return 0.0
    points = np.vstack(chunks)
    if len(points) < 4:
        return 0.0
    try:
        hull = ConvexHull(points, qhull_options="QJ")
        if hull.area <= 0:
            return 0.0
        return float(hull.area) / 2.0 / 1e6
    except Exception:
        return 0.0


def compute_cleft_max_distance_nm(zone_data: Dict[str, Any]) -> float:
    """
    Farthest distance between any two synaptic-cleft surface points (nm).

    Includes all pre- and postsynaptic synaptic-cleft designated points (inner and outer).
    Uses the 3D convex hull: the maximum pairwise distance in the full point set is always
    attained by two hull vertices, so ``pdist`` is evaluated only on hull vertices.
    Falls back to all points if the hull cannot be built (degenerate geometry).
    """
    points = _collect_cleft_surface_points(zone_data)
    n = len(points)
    if n < 2:
        return 0.0
    if n == 2:
        return float(np.linalg.norm(points[1] - points[0]))

    try:
        hull = ConvexHull(points, qhull_options="QJ")
        hull_vertices = points[hull.vertices]
        if len(hull_vertices) < 2:
            return 0.0
        if len(hull_vertices) == 2:
            return float(np.linalg.norm(hull_vertices[1] - hull_vertices[0]))
        return float(np.max(pdist(hull_vertices)))
    except Exception:
        return float(np.max(pdist(points)))


def save_membrane_volumes_from_glb(membranes: Dict[str, List[Dict[str, np.ndarray]]], tomogram_path, alignment_dir: str):
    """
    Calculate and save volumes for each membrane segmentation from GLB mesh data using convex hull.
    This method is more robust for non-watertight meshes that may extend beyond tomogram boundaries.
    
    Args:
        membranes: Dictionary containing membrane mesh data from GLB
        tomogram_path: Path to tomogram directory (str or Path)
    """
    alignment_dir = require_alignment_dir(alignment_dir)
    import trimesh
    
    tomogram_path = Path(tomogram_path)
    stt_results_dir = tomogram_path / alignment_dir / "STT_results"
    
    # Create cleft directory
    clefts_dir = stt_results_dir / "cleft"
    clefts_dir.mkdir(parents=True, exist_ok=True)
    
    volumes_data = {}
    
    # Calculate presynaptic membrane volumes from meshes using convex hull
    for i, mesh_data in enumerate(membranes['presynaptic']):
        try:
            # Create trimesh object from vertices and faces
            mesh = trimesh.Trimesh(vertices=mesh_data['vertices'], faces=mesh_data['faces'])
            
            # Calculate convex hull volume (more robust for non-watertight meshes)
            convex_hull = mesh.convex_hull
            volume_um3 = convex_hull.volume / 1e9  # Convert nm³ to µm³
            
            membrane_name = f"presynaptic_membrane_{i+1}"
            volumes_data[membrane_name] = {
                'volume_um3': volume_um3,
                'vertex_count': len(mesh_data['vertices']),
                'face_count': len(mesh_data['faces']),
                'surface_area_um2': mesh.area / 1e6,  # Convert nm² to µm²
                'convex_hull_volume_um3': volume_um3,
                'source': 'convex_hull'
            }
            # Presynaptic membrane volume calculated
        except Exception as e:
            print(f"Error calculating volume for presynaptic membrane {i+1}: {e}")
            # Skip this membrane if mesh calculation fails
            continue
    
    # Calculate postsynaptic membrane volumes from meshes using convex hull
    for i, mesh_data in enumerate(membranes['postsynaptic']):
        try:
            # Create trimesh object from vertices and faces
            mesh = trimesh.Trimesh(vertices=mesh_data['vertices'], faces=mesh_data['faces'])
            
            # Calculate convex hull volume (more robust for non-watertight meshes)
            convex_hull = mesh.convex_hull
            volume_um3 = convex_hull.volume / 1e9  # Convert nm³ to µm³
            
            membrane_name = f"postsynaptic_membrane_{i+1}"
            volumes_data[membrane_name] = {
                'volume_um3': volume_um3,
                'vertex_count': len(mesh_data['vertices']),
                'face_count': len(mesh_data['faces']),
                'surface_area_um2': mesh.area / 1e6,  # Convert nm² to µm²
                'convex_hull_volume_um3': volume_um3,
                'source': 'convex_hull'
            }
            # Postsynaptic membrane volume calculated
        except Exception as e:
            print(f"Error calculating volume for postsynaptic membrane {i+1}: {e}")
            # Skip this membrane if mesh calculation fails
            continue
    
    # Save volumes to JSON file
    import json
    volumes_file = clefts_dir / "membrane_volumes.json"
    with open(volumes_file, 'w') as f:
        json.dump(volumes_data, f, indent=2, default=str)
    
    return volumes_data


def load_membrane_volumes(tomogram_path, alignment_dir: str) -> Dict[str, Any]:
    """
    Load membrane volumes from JSON file and calculate averages.
    
    Args:
        tomogram_path: Path to tomogram directory (str or Path)
        
    Returns:
        Dictionary containing volume statistics with averages for pre and post
    """
    alignment_dir = require_alignment_dir(alignment_dir)
    tomogram_path = Path(tomogram_path)
    volumes_file = tomogram_path / alignment_dir / "STT_results" / "cleft" / "membrane_volumes.json"
    
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


def import_membrane_segmentations(tomogram_path, alignment_dir: str) -> Dict[str, List[np.ndarray]]:
    """
    Import presynaptic and postsynaptic membrane segmentation files.
    
    Args:
        tomogram_path: Path to the tomogram directory (str or Path)
        
    Returns:
        Dictionary containing lists of coordinate arrays for each membrane type
    """
    alignment_dir = require_alignment_dir(alignment_dir)
    tomogram_path = Path(tomogram_path)
    aunps_dir = tomogram_path / alignment_dir / "aunps"
    
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

def import_membrane_segmentations_from_glb(tomogram_path, alignment_dir: str) -> Dict[str, List[Dict[str, np.ndarray]]]:
    """
    Import presynaptic and postsynaptic membrane segmentations from a GLB file.
    
    Args:
        tomogram_path: Path to the tomogram directory (str or Path)
        
    Returns:
        Dictionary containing lists of coordinate arrays, faces, and normals for each membrane type
    """
    alignment_dir = require_alignment_dir(alignment_dir)
    import trimesh

    tomogram_path = Path(tomogram_path)
    aunps_dir = tomogram_path / alignment_dir / "aunps"
    
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
    save_membrane_volumes_from_glb(membranes, tomogram_path, alignment_dir=alignment_dir)

    return membranes


def find_clefts(membranes: Dict[str, List[np.ndarray]], distance_threshold: float = 40.0) -> Dict[str, Any]:
    """
    Find synaptic clefts by identifying presynaptic points within distance_threshold of postsynaptic points.
    Uses KD-tree for efficient spatial queries.
    
    Note: This function is not used in the current pipeline, after switching to using the find_active_zones_from_glb function.

    Args:
        membranes: Dictionary containing membrane coordinate arrays
        distance_threshold: Distance threshold in nm (default: 40.0)
        
    Returns:
        Dictionary containing synaptic cleft information and segmentations
    """
    clefts = {}
    cleft_count = 0
    
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
            
            # If we found an synaptic cleft
            if len(active_pre_indices) > 0 or len(active_post_indices) > 0:
                cleft_count += 1
                zone_name = f"cleft_pre{pre_idx+1}_post{post_idx+1}"
                
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
                
                clefts[zone_name] = {
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
                
                # Found synaptic cleft with presynaptic and postsynaptic points
            else:
                print(f"No synaptic cleft found between presynaptic {pre_idx+1} and postsynaptic {post_idx+1}")
    
    return {
        'clefts': clefts,
        'total_clefts': cleft_count,
        'distance_threshold': distance_threshold
    }

def find_active_zones_from_glb(membranes: Dict[str, List[Dict[str, np.ndarray]]], distance_range: Tuple[float, float] = (10.0, 40.0)) -> Dict[str, Any]:
    """
    Find synaptic clefts from GLB membrane meshes using KD-trees.

    Membership (distance gate): presynaptic vertices whose nearest postsynaptic neighbor lies in
    ``distance_range``; postsynaptic vertices whose nearest such presynaptic vertex lies in the same
    band. No normal filter on membership.

    Outer vs inner (vertex normals): **outer** if ``presyn_normal · postsyn_normal_at_nearest < 0``
    on the presynaptic side (and the symmetric dot on postsynaptic); **inner** is the complement on
    that active vertex set.

    Triangle patches use faces whose three vertices are all distance-active. Outer (resp. inner)
    area sums triangles whose three vertices are all classified outer (resp. inner).

    Args:
        membranes: Dictionary containing membrane coordinate arrays from GLB
        distance_range: Distance range in nm as (min_distance, max_distance) (default: (10.0, 40.0))

    Returns:
        Dictionary containing synaptic cleft information and segmentations
    """
    import trimesh

    min_distance, max_distance = distance_range

    clefts = {}
    cleft_count = 0
    presyn_membranes = membranes['presynaptic']
    postsyn_membranes = membranes['postsynaptic']
    for pre_idx, presyn_data in enumerate(presyn_membranes):
        presyn_coords = presyn_data['vertices']
        presyn_normals = presyn_data['normals']

        for post_idx, postsyn_data in enumerate(postsyn_membranes):
            postsyn_coords = postsyn_data['vertices']
            postsyn_normals = postsyn_data['normals']

            post_tree = KDTree(postsyn_coords)
            distances_pre, indices_pre = post_tree.query(presyn_coords, distance_upper_bound=max_distance)

            active_pre_mask = (distances_pre >= min_distance) & (distances_pre <= max_distance)
            active_pre_indices = np.where(active_pre_mask)[0]
            active_pre_coords = presyn_coords[active_pre_indices] if len(active_pre_indices) > 0 else np.array([])

            active_post_indices = np.array([], dtype=int)
            if len(active_pre_coords) > 0:
                pre_tree = KDTree(active_pre_coords)
                distances_post, indices_post = pre_tree.query(postsyn_coords, distance_upper_bound=max_distance)
                active_post_mask = (distances_post >= min_distance) & (distances_post <= max_distance)
                active_post_indices = np.where(active_post_mask)[0]

            if len(active_pre_indices) == 0 or len(active_post_indices) == 0:
                print(f"No synaptic cleft found between presynaptic {pre_idx + 1} and postsynaptic {post_idx + 1}")
                continue

            cleft_count += 1
            zone_name = f"cleft_pre{pre_idx + 1}_post{post_idx + 1}"

            distances_active = distances_pre[active_pre_indices]
            min_dist = float(np.min(distances_active))
            max_dist = float(np.max(distances_active))
            avg_dist = float(np.mean(distances_active))

            dots_pre = np.sum(
                presyn_normals[active_pre_indices] * postsyn_normals[indices_pre[active_pre_indices]],
                axis=1,
            )
            pre_outer_local = dots_pre < 0
            pre_outer_global = active_pre_indices[pre_outer_local]
            pre_inner_global = active_pre_indices[~pre_outer_local]
            active_pre_outer_points = presyn_coords[pre_outer_global]
            active_pre_inner_points = presyn_coords[pre_inner_global]

            nn_pre_for_post = active_pre_indices[indices_post[active_post_indices]]
            dots_post = np.sum(
                postsyn_normals[active_post_indices] * presyn_normals[nn_pre_for_post],
                axis=1,
            )
            post_outer_local = dots_post < 0
            post_outer_global = active_post_indices[post_outer_local]
            post_inner_global = active_post_indices[~post_outer_local]
            active_post_outer_points = postsyn_coords[post_outer_global]
            active_post_inner_points = postsyn_coords[post_inner_global]

            pre_mesh = trimesh.Trimesh(vertices=presyn_data['vertices'], faces=presyn_data['faces'])
            post_mesh = trimesh.Trimesh(vertices=postsyn_data['vertices'], faces=postsyn_data['faces'])

            active_pre_faces_mask = np.isin(pre_mesh.faces, active_pre_indices).all(axis=1)
            active_pre_faces_indices = np.where(active_pre_faces_mask)[0]
            active_pre_mesh = pre_mesh.submesh([active_pre_faces_indices], append=True)

            active_post_faces_mask = np.isin(postsyn_data['faces'], active_post_indices).all(axis=1)
            active_post_faces_indices = np.where(active_post_faces_mask)[0]
            active_post_mesh = post_mesh.submesh([active_post_faces_indices], append=True)

            total_faces = len(active_pre_mesh.faces)
            total_post_faces = len(active_post_mesh.faces)

            pre_outer_face_mask = np.isin(active_pre_mesh.faces, pre_outer_global).all(axis=1)
            pre_inner_face_mask = np.isin(active_pre_mesh.faces, pre_inner_global).all(axis=1)
            front_facing_faces = int(np.sum(pre_outer_face_mask))
            back_facing_faces = int(np.sum(pre_inner_face_mask))

            active_pre_area = _cleft_membrane_area_from_hull_um2(
                active_pre_inner_points, active_pre_outer_points
            )

            if total_post_faces > 0:
                post_outer_face_mask = np.isin(active_post_mesh.faces, post_outer_global).all(axis=1)
                post_inner_face_mask = np.isin(active_post_mesh.faces, post_inner_global).all(axis=1)
                post_front_facing_faces = int(np.sum(post_outer_face_mask))
                post_back_facing_faces = int(np.sum(post_inner_face_mask))
            else:
                post_front_facing_faces = 0
                post_back_facing_faces = 0

            active_post_area = _cleft_membrane_area_from_hull_um2(
                active_post_inner_points, active_post_outer_points
            )

            clefts[zone_name] = {
                'presynaptic_membrane_index': pre_idx + 1,
                'postsynaptic_membrane_index': post_idx + 1,
                'active_presynaptic_points': active_pre_coords,
                'active_presynaptic_outer_points': active_pre_outer_points,
                'active_presynaptic_inner_points': active_pre_inner_points,
                'active_presynaptic_faces': active_pre_mesh.faces,
                'active_presynaptic_mesh': active_pre_mesh,
                'active_presynaptic_area': active_pre_area,
                'total_faces': total_faces,
                'front_facing_faces': front_facing_faces,
                'back_facing_faces': back_facing_faces,
                'active_postsynaptic_points': postsyn_coords[active_post_indices],
                'active_postsynaptic_outer_points': active_post_outer_points,
                'active_postsynaptic_inner_points': active_post_inner_points,
                'active_postsynaptic_faces': active_post_mesh.faces,
                'active_postsynaptic_mesh': active_post_mesh,
                'active_postsynaptic_area': active_post_area,
                'postsynaptic_total_faces': total_post_faces,
                'postsynaptic_front_facing_faces': post_front_facing_faces,
                'postsynaptic_back_facing_faces': post_back_facing_faces,
                'active_presynaptic_indices': active_pre_indices,
                'active_postsynaptic_indices': active_post_indices,
                'min_distance': min_dist,
                'max_distance': max_dist,
                'avg_distance': avg_dist,
                'active_pre_count': len(active_pre_indices),
                'active_post_count': len(active_post_indices),
            }

    return {
        'clefts': clefts,
        'total_clefts': cleft_count,
        'distance_range': distance_range,
    }


def save_cleft_segmentations(clefts: Dict[str, Any], tomogram_path, alignment_dir: str):
    """
    Save synaptic cleft segmentations to files.
    
    Args:
        clefts: Clefts dictionary from find_clefts
        tomogram_path: Path to tomogram directory (str or Path)
    """
    alignment_dir = require_alignment_dir(alignment_dir)
    tomogram_path = Path(tomogram_path)
    stt_results_dir = tomogram_path / alignment_dir / "STT_results"
    
    # Create cleft directory
    cleft_dir = stt_results_dir / "cleft"
    cleft_dir.mkdir(parents=True, exist_ok=True)
    
    for zone_name, zone_data in clefts['clefts'].items():
        # Save outer-facing (toward cleft) membranes for downstream usage
        pre_outer = zone_data.get('active_presynaptic_outer_points', zone_data.get('active_presynaptic_points', np.array([])))
        post_outer = zone_data.get('active_postsynaptic_outer_points', zone_data.get('active_postsynaptic_points', np.array([])))
        if len(pre_outer) > 0:
            np.savetxt(cleft_dir / f"{zone_name}_pre_outer.txt", pre_outer, fmt='%.6e')
        if len(post_outer) > 0:
            np.savetxt(cleft_dir / f"{zone_name}_post_outer.txt", post_outer, fmt='%.6e')

        # Save inner-facing (away from cleft) membranes for future use
        pre_inner = zone_data.get('active_presynaptic_inner_points', np.array([]))
        post_inner = zone_data.get('active_postsynaptic_inner_points', np.array([]))
        if len(pre_inner) > 0:
            np.savetxt(cleft_dir / f"{zone_name}_pre_inner.txt", pre_inner, fmt='%.6e')
        if len(post_inner) > 0:
            np.savetxt(cleft_dir / f"{zone_name}_post_inner.txt", post_inner, fmt='%.6e')


def match_clefts_by_aunps(
    tomogram_path,
    cleft_indices,
    all_clefts,
    alignment_dir: str,
    *,
    aunp_pick_star_pattern=None,
) -> Dict[int, str]:
    """
    Match synaptic cleft indices to zone names using smart matching based on AuNP locations.
    This is done once and the mapping can be reused.
    
    Args:
        tomogram_path: Path to the tomogram file
        cleft_indices: List of synaptic cleft indices to match
        all_clefts: Dictionary of all synaptic clefts from GLB
        
    Returns:
        Dictionary mapping cleft_index -> zone_name
    """
    alignment_dir = require_alignment_dir(alignment_dir)
    az_mapping = {}
    
    # Load AuNP data for smart matching
    try:
        import pandas as pd
        from .aunps import load_aunp_pick_star_dataframes

        aunps_dir = Path(tomogram_path) / alignment_dir / "aunps"
        star_dfs = load_aunp_pick_star_dataframes(
            aunps_dir,
            list(cleft_indices),
            pattern=aunp_pick_star_pattern,
        )
        
        if not star_dfs:
            return az_mapping
        
        aunp_data = pd.concat(star_dfs, ignore_index=True)
        
        if 'cleft' not in aunp_data.columns or 'faCoordinateX' not in aunp_data.columns:
            return az_mapping
        
        # Match each index to a zone name
        for az_idx in cleft_indices:
            # Get AuNPs for this synaptic cleft index
            aunps_in_az = aunp_data[aunp_data['cleft'] == az_idx]
            
            if aunps_in_az.empty:
                print(f"Warning: No AuNPs found for synaptic cleft index {az_idx}, skipping")
                continue
            
            # Calculate center of AuNPs for this synaptic cleft index
            aunp_center = np.mean(aunps_in_az[['faCoordinateX', 'faCoordinateY', 'faCoordinateZ']].values, axis=0)
            
            # Find the synaptic cleft closest to these AuNPs
            best_az_name = None
            min_distance = float('inf')
            
            for zone_name, zone_data in all_clefts.items():
                if len(zone_data['active_presynaptic_points']) > 0 and len(zone_data['active_postsynaptic_points']) > 0:
                    # Calculate center of this synaptic cleft (paired pre/post membranes)
                    pre_center = np.mean(zone_data['active_presynaptic_points'], axis=0)
                    post_center = np.mean(zone_data['active_postsynaptic_points'], axis=0)
                    az_center = (pre_center + post_center) / 2.0
                    
                    # Calculate distance from AuNP center to synaptic cleft center
                    distance = np.linalg.norm(aunp_center - az_center)
                    
                    if distance < min_distance:
                        min_distance = distance
                        best_az_name = zone_name
            
            if best_az_name is not None:
                az_mapping[az_idx] = best_az_name
                print(f"Matched synaptic cleft index {az_idx} to {best_az_name} (distance: {min_distance:.2f} nm)")
            else:
                print(f"Warning: No synaptic cleft found for synaptic cleft index {az_idx}")
    
    except Exception as e:
        print(f"Warning: Could not perform smart matching: {e}")
    
    return az_mapping


def save_cleft_mapping(tomogram_path, az_mapping: Dict[int, str], alignment_dir: str):
    """Save the synaptic cleft index to zone name mapping to a JSON file."""
    alignment_dir = require_alignment_dir(alignment_dir)
    tomogram_path = Path(tomogram_path)
    cleft_dir = tomogram_path / alignment_dir / "STT_results" / "cleft"
    cleft_dir.mkdir(parents=True, exist_ok=True)
    
    mapping_file = cleft_dir / "cleft_mapping.json"
    import json
    with open(mapping_file, 'w') as f:
        json.dump(az_mapping, f, indent=2)
    print(f"Saved synaptic cleft mapping to {mapping_file}")


def load_cleft_mapping(tomogram_path, alignment_dir: str) -> Dict[int, str]:
    """Load the synaptic cleft index to zone name mapping from JSON file."""
    alignment_dir = require_alignment_dir(alignment_dir)
    tomogram_path = Path(tomogram_path)
    mapping_file = tomogram_path / alignment_dir / "STT_results" / "cleft" / "cleft_mapping.json"

    if mapping_file.exists():
        try:
            import json
            with open(mapping_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load synaptic cleft mapping: {e}")
    
    return {}


def define_cleft(
    tomogram_path,
    cleft_indices=None,
    distance_range=None,
    *,
    alignment_dir: str,
    aunp_pick_star_pattern=None,
) -> Dict[str, Any]:
    """
    Define synaptic cleft in tomogram.
    If cleft_indices is specified, only includes synaptic clefts that correspond to those indices.
    
    Args:
        tomogram_path: Path to the tomogram file.
        cleft_indices: List of synaptic cleft indices to include (None = all).
        distance_range: Tuple of (min_distance, max_distance) in nm for synaptic cleft definition (default: (10.0, 40.0)).
    
    Returns:
        Dictionary containing synaptic cleft analysis results.
    """
    alignment_dir = require_alignment_dir(alignment_dir)
    print(f"Defining synaptic cleft in {Path(tomogram_path).name}")
    
    # Use custom distance range if provided, otherwise use default
    if distance_range is None:
        distance_range = (10.0, 40.0)
    
    # Import membrane segmentations from GLB
    try:
        membranes = import_membrane_segmentations_from_glb(tomogram_path, alignment_dir=alignment_dir)
        
        # Find synaptic clefts from GLB
        clefts = find_active_zones_from_glb(membranes, distance_range=distance_range)
        
        # Filter synaptic clefts if indices are specified using smart matching based on AuNP locations
        az_mapping = {}
        if cleft_indices is not None and len(cleft_indices) > 0:
            # Do smart matching once
            az_mapping = match_clefts_by_aunps(
                tomogram_path,
                cleft_indices,
                clefts['clefts'],
                alignment_dir=alignment_dir,
                aunp_pick_star_pattern=aunp_pick_star_pattern,
            )
            
            if az_mapping:
                # Filter to only include matched zones
                zones_to_include = set(az_mapping.values())
                filtered_zones = {name: data for name, data in clefts['clefts'].items() if name in zones_to_include}
                clefts['clefts'] = filtered_zones
                clefts['total_clefts'] = len(filtered_zones)
                
                # Save the mapping for reuse by other functions
                save_cleft_mapping(tomogram_path, az_mapping, alignment_dir=alignment_dir)
                print(f"Filtered to {len(filtered_zones)} synaptic clefts using smart matching (indices: {cleft_indices})")
            else:
                # Fallback to order-based matching if smart matching failed
                print("Warning: Smart matching failed. Using order-based matching (may be incorrect).")
                cleft_names = list(clefts['clefts'].keys())
                zones_to_include = set()
                for az_idx in cleft_indices:
                    if 0 <= az_idx < len(cleft_names):
                        zone_name = cleft_names[az_idx]
                        zones_to_include.add(zone_name)
                        az_mapping[az_idx] = zone_name
                
                # Filter to only include specified zones
                filtered_zones = {name: data for name, data in clefts['clefts'].items() if name in zones_to_include}
                clefts['clefts'] = filtered_zones
                clefts['total_clefts'] = len(filtered_zones)
                save_cleft_mapping(tomogram_path, az_mapping, alignment_dir=alignment_dir)
                print(f"Filtered to {len(filtered_zones)} synaptic clefts using order-based matching (indices: {cleft_indices})")
        
        # Save synaptic cleft segmentations
        save_cleft_segmentations(clefts, tomogram_path, alignment_dir=alignment_dir)
        
        # Calculate summary statistics
        total_active_pre_points = sum(len(zone['active_presynaptic_points']) for zone in clefts['clefts'].values())
        total_active_post_points = sum(len(zone['active_postsynaptic_points']) for zone in clefts['clefts'].values())
        
        # Calculate synaptic cleft areas and max point-to-point span per zone
        cleft_pre_areas = []
        cleft_post_areas = []
        cleft_max_distances_nm: List[float] = []
        for zone_name, zone_data in clefts['clefts'].items():
            max_dist_nm = compute_cleft_max_distance_nm(zone_data)
            zone_data['cleft_max_distance_nm'] = max_dist_nm
            cleft_max_distances_nm.append(max_dist_nm)
            print(f"Cleft max distance {zone_name}: {max_dist_nm:.2f} nm")

            # Require presynaptic area data - raise error if missing
            if 'active_presynaptic_area' not in zone_data:
                raise ValueError(f"No presynaptic area data available for {zone_name}. All synaptic clefts must have area data.")
            cleft_pre_areas.append(zone_data['active_presynaptic_area'])
            total_faces = zone_data.get('total_faces', 0)
            front_facing_faces = zone_data.get('front_facing_faces', 0)
            back_facing_faces = zone_data.get('back_facing_faces', 0)
            print(
                f"Presynaptic synaptic cleft area {zone_name}: {zone_data['active_presynaptic_area']:.6f} µm² "
                f"(3D hull inner+outer, area/2; mesh faces: {front_facing_faces}/{total_faces} outer, "
                f"{back_facing_faces} inner)"
            )
            
            # Require postsynaptic area data - raise error if missing
            if 'active_postsynaptic_area' not in zone_data:
                raise ValueError(f"No postsynaptic area data available for {zone_name}. All synaptic clefts must have area data.")
            cleft_post_areas.append(zone_data['active_postsynaptic_area'])
            post_total_faces = zone_data.get('postsynaptic_total_faces', 0)
            post_front_facing_faces = zone_data.get('postsynaptic_front_facing_faces', 0)
            post_back_facing_faces = zone_data.get('postsynaptic_back_facing_faces', 0)
            print(
                f"Postsynaptic synaptic cleft area {zone_name}: {zone_data['active_postsynaptic_area']:.6f} µm² "
                f"(3D hull inner+outer, area/2; mesh faces: {post_front_facing_faces}/{post_total_faces} outer, "
                f"{post_back_facing_faces} inner)"
            )
        
        if not cleft_pre_areas:
            raise ValueError("No synaptic cleft presynaptic areas calculated. Cannot compute average area.")
        if not cleft_post_areas:
            raise ValueError("No synaptic cleft postsynaptic areas calculated. Cannot compute average area.")
        avg_cleft_pre_area = np.mean(cleft_pre_areas)
        avg_cleft_post_area = np.mean(cleft_post_areas)
        total_cleft_pre_area = np.sum(cleft_pre_areas)
        total_cleft_post_area = np.sum(cleft_post_areas)
        cleft_max_distance = (
            float(np.max(cleft_max_distances_nm)) if cleft_max_distances_nm else 0.0
        )

        if az_mapping:
            az_index_by_zone = {zone_name: int(idx) for idx, zone_name in az_mapping.items()}
        else:
            az_index_by_zone = {
                zone_name: i
                for i, zone_name in enumerate(sorted(clefts['clefts'].keys()))
            }

        individual_zone_results: Dict[str, Dict[str, Any]] = {}
        for zone_name, zone_data in clefts['clefts'].items():
            individual_zone_results[zone_name] = {
                'cleft_index': az_index_by_zone.get(zone_name),
                'active_presynaptic_area': float(zone_data['active_presynaptic_area']),
                'active_postsynaptic_area': float(zone_data['active_postsynaptic_area']),
                'cleft_max_distance_nm': float(zone_data['cleft_max_distance_nm']),
                'active_pre_count': int(zone_data['active_pre_count']),
                'active_post_count': int(zone_data['active_post_count']),
                'az_min_distance_nm': float(zone_data['min_distance']),
                'az_max_distance_nm': float(zone_data['max_distance']),
                'az_avg_distance_nm': float(zone_data['avg_distance']),
                'presynaptic_membrane_index': int(zone_data.get('presynaptic_membrane_index', 0)),
                'postsynaptic_membrane_index': int(zone_data.get('postsynaptic_membrane_index', 0)),
                'pre_total_faces': int(zone_data.get('total_faces', 0)),
                'pre_front_facing_faces': int(zone_data.get('front_facing_faces', 0)),
                'pre_back_facing_faces': int(zone_data.get('back_facing_faces', 0)),
                'post_total_faces': int(zone_data.get('postsynaptic_total_faces', 0)),
                'post_front_facing_faces': int(zone_data.get('postsynaptic_front_facing_faces', 0)),
                'post_back_facing_faces': int(zone_data.get('postsynaptic_back_facing_faces', 0)),
            }
        
        # Load membrane volumes
        volumes_data = load_membrane_volumes(tomogram_path, alignment_dir=alignment_dir)
        
        results = {
            'cleft_count': clefts['total_clefts'],
            'total_active_pre_points': total_active_pre_points,
            'total_active_post_points': total_active_post_points,
            'avg_cleft_area': avg_cleft_pre_area,  # Presynaptic area (kept for backward compatibility)
            'avg_cleft_pre_area': avg_cleft_pre_area,  # Presynaptic area
            'avg_cleft_post_area': avg_cleft_post_area,  # Postsynaptic area
            'total_cleft_pre_area': total_cleft_pre_area,  # Sum of all presynaptic areas
            'total_cleft_post_area': total_cleft_post_area,  # Sum of all postsynaptic areas (use this for AuNP density)
            'cleft_max_distance': cleft_max_distance,  # Max span (nm) across zones; exported as *_nm in CSV
            'distance_range': clefts['distance_range'],
            'cleft_names': list(clefts['clefts'].keys()),
            'individual_zone_results': individual_zone_results,
            'membrane_volumes': volumes_data,
            'status': 'completed'
        }
        
    except Exception as e:
        print(f"Error defining synaptic clefts: {e}")
        results = {
            'cleft_count': 0,
            'total_active_pre_points': 0,
            'total_active_post_points': 0,
            'avg_cleft_area': 0.0,
            'avg_cleft_pre_area': 0.0,
            'avg_cleft_post_area': 0.0,
            'total_cleft_pre_area': 0.0,
            'total_cleft_post_area': 0.0,
            'cleft_max_distance': 0.0,
            'distance_range': (10.0, 40.0),
            'cleft_names': [],
            'individual_zone_results': {},
            'membrane_volumes': {},
            'status': 'error',
            'error_message': str(e)
        }
    
    return results


CLEFT_RESULTS_CSV = Path("results/cleft/cleft_results.csv")


def build_cleft_per_zone_rows(
    *,
    tomogram_name: str,
    set_name: str,
    alignment_dir: str,
    az_results: Dict[str, Any],
    cleft_results: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Build one CSV row per synaptic cleft from define_cleft + calculate_cleft_width results."""
    az_zones = az_results.get("individual_zone_results") or {}
    cleft_zones = cleft_results.get("individual_zone_results") or {}
    az_status = az_results.get("status", "")
    cleft_status = cleft_results.get("status", "")

    rows: List[Dict[str, Any]] = []
    for zone_name in sorted(set(az_zones) | set(cleft_zones)):
        z = az_zones.get(zone_name, {})
        c = cleft_zones.get(zone_name, {})
        rows.append({
            "tomogram_name": tomogram_name,
            "set_name": set_name or "",
            "alignment_dir": alignment_dir,
            "cleft": zone_name,
            "cleft_index": z.get("cleft_index"),
            "az_status": az_status,
            "cleft_status": cleft_status,
            "active_presynaptic_area_um2": z.get("active_presynaptic_area"),
            "active_postsynaptic_area_um2": z.get("active_postsynaptic_area"),
            "cleft_max_distance_nm": z.get("cleft_max_distance_nm"),
            "active_pre_count": z.get("active_pre_count"),
            "active_post_count": z.get("active_post_count"),
            "az_min_distance_nm": z.get("az_min_distance_nm"),
            "az_max_distance_nm": z.get("az_max_distance_nm"),
            "az_avg_distance_nm": z.get("az_avg_distance_nm"),
            "presynaptic_membrane_index": z.get("presynaptic_membrane_index"),
            "postsynaptic_membrane_index": z.get("postsynaptic_membrane_index"),
            "pre_total_faces": z.get("pre_total_faces"),
            "pre_front_facing_faces": z.get("pre_front_facing_faces"),
            "pre_back_facing_faces": z.get("pre_back_facing_faces"),
            "post_total_faces": z.get("post_total_faces"),
            "post_front_facing_faces": z.get("post_front_facing_faces"),
            "post_back_facing_faces": z.get("post_back_facing_faces"),
            "average_cleft_width_nm": c.get("average_cleft_width"),
            "cleft_width_std_nm": c.get("cleft_width_std"),
            "min_cleft_width_nm": c.get("min_cleft_width"),
            "max_cleft_width_nm": c.get("max_cleft_width"),
            "cleft_n_measurements": c.get("measurement_count"),
        })
    return rows


def upsert_cleft_per_zone_csv(
    rows: List[Dict[str, Any]],
    tomogram_name: str,
    alignment_dir: str,
    results_dir: str = "results",
) -> Path:
    """Upsert per-zone synaptic cleft rows for one tomogram into the global CSV."""
    import pandas as pd

    csv_path = Path(results_dir) / "cleft" / "cleft_results.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_new = pd.DataFrame(rows)
    if csv_path.exists():
        try:
            df_existing = pd.read_csv(csv_path)
            if "alignment_dir" not in df_existing.columns:
                df_existing["alignment_dir"] = ""
            df_existing = df_existing[
                ~(
                    (df_existing["tomogram_name"] == tomogram_name)
                    & (df_existing["alignment_dir"] == alignment_dir)
                )
            ]
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined.to_csv(csv_path, index=False)
        except Exception as e:
            print(f"Error updating {csv_path}: {e}")
            df_new.to_csv(csv_path, index=False)
    else:
        df_new.to_csv(csv_path, index=False)
    print(f"Saved {len(rows)} synaptic cleft result row(s) for {tomogram_name} to {csv_path}")
    return csv_path


def import_cleft_segmentations(tomogram_path, alignment_dir: str) -> Dict[str, Any]:
    """
    Import synaptic cleft segmentation files.
    
    Args:
        tomogram_path: Path to tomogram directory (str or Path)
        
    Returns:
        Dictionary containing synaptic cleft segmentations
    """
    alignment_dir = require_alignment_dir(alignment_dir)
    tomogram_path = Path(tomogram_path)
    cleft_dir = tomogram_path / alignment_dir / "STT_results" / "cleft"
    
    if not cleft_dir.exists():
        raise FileNotFoundError(f"Cleft directory not found: {cleft_dir}")
    
    clefts = {}
    
    # Find all synaptic cleft outer files
    pre_files = list(cleft_dir.glob("*_pre_outer.txt"))
    post_files = list(cleft_dir.glob("*_post_outer.txt"))
    
    print(f"Found {len(pre_files)} active presynaptic files")
    print(f"Found {len(post_files)} active postsynaptic files")
    
    # Group files by synaptic cleft name
    for pre_file in pre_files:
        # Extract zone name from *_pre_outer
        stem = pre_file.stem
        if stem.endswith('_pre_outer'):
            zone_name = stem[:-10]  # remove '_pre_outer'
        else:
            zone_name = stem

        post_file = cleft_dir / f"{zone_name}_post_outer.txt"
        
        if post_file.exists():
            try:
                pre_coords = np.atleast_2d(np.loadtxt(pre_file, delimiter=None))
                post_coords = np.atleast_2d(np.loadtxt(post_file, delimiter=None))
                
                pre_inner_file = cleft_dir / f"{zone_name}_pre_inner.txt"
                post_inner_file = cleft_dir / f"{zone_name}_post_inner.txt"
                pre_inner_coords = np.atleast_2d(np.loadtxt(pre_inner_file, delimiter=None)) if pre_inner_file.exists() else np.array([])
                post_inner_coords = np.atleast_2d(np.loadtxt(post_inner_file, delimiter=None)) if post_inner_file.exists() else np.array([])

                clefts[zone_name] = {
                    # Outer membranes (also exposed as generic coords keys)
                    'presynaptic_coords': pre_coords,
                    'postsynaptic_coords': post_coords,
                    # Explicit outer/inner keys
                    'presynaptic_outer_coords': pre_coords,
                    'postsynaptic_outer_coords': post_coords,
                    'presynaptic_inner_coords': pre_inner_coords,
                    'postsynaptic_inner_coords': post_inner_coords,
                    'presynaptic_count': len(pre_coords),
                    'postsynaptic_count': len(post_coords),
                    'presynaptic_inner_count': len(pre_inner_coords) if np.size(pre_inner_coords) > 0 else 0,
                    'postsynaptic_inner_count': len(post_inner_coords) if np.size(post_inner_coords) > 0 else 0,
                }
                
                print(f"  ✓ Imported synaptic cleft: {zone_name} ({len(pre_coords)} pre, {len(post_coords)} post points)")
                
            except Exception as e:
                print(f"  ✗ Error importing synaptic cleft {zone_name}: {e}")
        else:
            print(f"  ✗ Post file not found: {post_file}")
            # Try to find any post file that might match
            potential_matches = list(cleft_dir.glob(f"*{zone_name}*"))
            if potential_matches:
                print(f"    Potential matches: {[f.name for f in potential_matches]}")
    
    return clefts


def calculate_cleft_width_for_cleft(pre_coords: np.ndarray, post_coords: np.ndarray) -> Dict[str, Any]:
    """
    Calculate cleft width for a single synaptic cleft using KD-tree.
    
    Args:
        pre_coords: Presynaptic synaptic cleft coordinates
        post_coords: Postsynaptic synaptic cleft coordinates
        
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


def calculate_cleft_width(
    tomogram_path,
    cleft_indices=None,
    set_name=None,
    *,
    alignment_dir: str,
) -> Dict[str, Any]:
    """
    Calculate synaptic cleft width for synaptic clefts.
    If cleft_indices is specified, only includes synaptic clefts that correspond to those indices.
    
    Args:
        tomogram_path: Path to the tomogram file.
        cleft_indices: List of synaptic cleft indices to include (None = all).
        set_name: Name of the dataset/set this tomogram belongs to (from CSV).
    
    Returns:
        Dictionary containing cleft width analysis results.
    """
    alignment_dir = require_alignment_dir(alignment_dir)
    print(f"Calculating cleft width in {Path(tomogram_path).name}")
    
    try:
        # Import synaptic cleft segmentations
        clefts = import_cleft_segmentations(tomogram_path, alignment_dir=alignment_dir)
        
        # Filter synaptic clefts if indices are specified - use saved mapping from define_cleft
        if cleft_indices is not None and len(cleft_indices) > 0:
            # Load the saved mapping (created by define_cleft)
            az_mapping = load_cleft_mapping(tomogram_path, alignment_dir=alignment_dir)
            
            if az_mapping:
                # Convert string keys to int (JSON stores dict keys as strings)
                az_mapping = {int(k): v for k, v in az_mapping.items()}
                
                # Filter to only include zones in the mapping
                zones_to_include = {az_mapping[az_idx] for az_idx in cleft_indices if az_idx in az_mapping}
                clefts = {name: data for name, data in clefts.items() if name in zones_to_include}
                print(f"Filtered to {len(clefts)} synaptic clefts for cleft width using saved mapping (indices: {cleft_indices})")
            else:
                # If no saved mapping exists, the zones were already filtered when saved
                # Just use all loaded zones (they're already filtered)
                print(f"Using {len(clefts)} synaptic clefts for cleft width (already filtered by define_cleft)")
        
        if not clefts:
            print("No synaptic clefts found for cleft width calculation")
            return {
                'average_cleft_width': 0.0,
                'cleft_width_std': 0.0,
                'min_cleft_width': 0.0,
                'max_cleft_width': 0.0,
                'cleft_count': 0,
                'status': 'no_clefts'
            }
        
        tomogram_name = Path(tomogram_path).name
        if set_name is None or set_name == "unknown":
            path_parts = Path(tomogram_path).parts
            set_name = "unknown"
            for i, part in enumerate(path_parts):
                if part.endswith("_tomograms") and i > 0:
                    set_name = part.replace("_tomograms", "")
                    break
        
        # Calculate cleft width for each synaptic cleft
        cleft_results = {}
        all_distances = []
        measurement_rows: List[Dict[str, Any]] = []
        
        for zone_name, zone_data in clefts.items():
            cleft_stats = calculate_cleft_width_for_cleft(
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
                    for d in distances_pre_to_post:
                        measurement_rows.append({
                            'tomogram_name': tomogram_name,
                            'set_name': set_name,
                            'alignment_dir': alignment_dir,
                            'cleft': zone_name,
                            'direction': 'pre_to_post',
                            'cleft_distance_nm': float(d),
                        })
                    for d in distances_post_to_pre:
                        measurement_rows.append({
                            'tomogram_name': tomogram_name,
                            'set_name': set_name,
                            'alignment_dir': alignment_dir,
                            'cleft': zone_name,
                            'direction': 'post_to_pre',
                            'cleft_distance_nm': float(d),
                        })
        
        # Calculate overall statistics from all individual measurements
        if all_distances:
            overall_stats = {
                'average_cleft_width': float(np.mean(all_distances)),
                'cleft_width_std': float(np.std(all_distances)),
                'min_cleft_width': float(np.min(all_distances)),
                'max_cleft_width': float(np.max(all_distances)),
                'cleft_count': len(clefts),
                'total_measurements': len(all_distances),
                'individual_zone_results': cleft_results,
                'status': 'completed'
            }
            
            # --- Append to global results/all_cleft_distances.csv ---
            # One row per tomogram + synaptic cleft (unique tomogram+AZ).
            import pandas as pd

            cleft_rows = []
            for zone_name, zone_stats in cleft_results.items():
                cleft_rows.append({
                    'tomogram_name': tomogram_name,
                    'set_name': set_name,
                    'alignment_dir': alignment_dir,
                    'cleft': zone_name,
                    'average_cleft_width_nm': zone_stats['average_cleft_width'],
                    'cleft_width_std_nm': zone_stats['cleft_width_std'],
                    'min_cleft_width_nm': zone_stats['min_cleft_width'],
                    'max_cleft_width_nm': zone_stats['max_cleft_width'],
                    'n_measurements': zone_stats['measurement_count'],
                })

            # Save to global CSV
            df_cleft = pd.DataFrame(cleft_rows)
            global_csv = Path("results/cleft/all_cleft_distances.csv")
            global_csv.parent.mkdir(parents=True, exist_ok=True)
            if global_csv.exists():
                try:
                    df_existing = pd.read_csv(global_csv)
                    if 'alignment_dir' not in df_existing.columns:
                        df_existing['alignment_dir'] = ''
                    # Remove existing rows for this tomogram+alignment pair
                    df_existing = df_existing[
                        ~(
                            (df_existing['tomogram_name'] == tomogram_name) &
                            (df_existing['alignment_dir'] == alignment_dir)
                        )
                    ]
                    df_combined = pd.concat([df_existing, df_cleft], ignore_index=True)
                    df_combined.to_csv(global_csv, index=False)
                except Exception as e:
                    print(f"Error updating global all_cleft_distances.csv: {e}")
                    df_cleft.to_csv(global_csv, index=False)
            else:
                df_cleft.to_csv(global_csv, index=False)
            print(
                f"Saved cleft width for {len(cleft_rows)} synaptic cleft(s) in "
                f"{tomogram_name} to {global_csv}"
            )
            
            meas_csv = Path("results/cleft/all_cleft_measurements.csv")
            df_meas = pd.DataFrame(measurement_rows)
            if meas_csv.exists():
                try:
                    df_meas_existing = pd.read_csv(meas_csv)
                    if 'alignment_dir' not in df_meas_existing.columns:
                        df_meas_existing['alignment_dir'] = ''
                    df_meas_existing = df_meas_existing[
                        ~(
                            (df_meas_existing['tomogram_name'] == tomogram_name)
                            & (df_meas_existing['alignment_dir'] == alignment_dir)
                        )
                    ]
                    pd.concat([df_meas_existing, df_meas], ignore_index=True).to_csv(
                        meas_csv, index=False
                    )
                except Exception as e:
                    print(f"Error updating global all_cleft_measurements.csv: {e}")
                    df_meas.to_csv(meas_csv, index=False)
            else:
                df_meas.to_csv(meas_csv, index=False)
            print(
                f"Saved {len(measurement_rows)} individual cleft measurements for "
                f"{tomogram_name} to {meas_csv}"
            )
            overall_stats['cleft_measurements_csv'] = str(meas_csv)
            # --- End global results ---
        else:
            raise ValueError("No cleft width measurements found. Cannot calculate cleft width statistics. Synaptic clefts must have both presynaptic and postsynaptic points.")
        
        print(f"Overall cleft width: {overall_stats['average_cleft_width']:.2f} ± {overall_stats['cleft_width_std']:.2f} nm")
        print(f"Calculated cleft width for {len(clefts)} synaptic clefts")
        
        return overall_stats
        
    except Exception as e:
        print(f"Error calculating cleft width: {e}")
        return {
            'average_cleft_width': 0.0,
            'cleft_width_std': 0.0,
            'min_cleft_width': 0.0,
            'max_cleft_width': 0.0,
            'cleft_count': 0,
            'total_measurements': 0,
            'individual_zone_results': {},
            'status': 'error',
            'error_message': str(e)
        }


def define_active_zonogram(clefts):
    """
    Define active zonogram from synaptic clefts.
    
    Args:
        clefts: Dictionary containing synaptic cleft data.
        
    Returns:
        Dictionary containing active zonogram results.
    """
    # Defining active zonogram
    from torch_affine_utils.transforms_3d import T
    from torch_affine_utils.utils import homogenise_coordinates
    import torch
    import einops
    
    if not clefts or 'clefts' not in clefts:
        print("No synaptic clefts found for zonogram definition")
        return {
            'status': 'no_clefts',
            'cleft_count': 0,
            'zonogram_data': {}
        }
    
    # Prepare zonogram data
    zonogram_data = {}
    
    for zone_name, zone_data in clefts['clefts'].items():
        # Define center of active zonogram
        if len(zone_data['active_presynaptic_points']) == 0:
            raise ValueError(f"No presynaptic points found for {zone_name}. Cannot calculate synaptic cleft center.")
        if len(zone_data['active_postsynaptic_points']) == 0:
            raise ValueError(f"No postsynaptic points found for {zone_name}. Cannot calculate synaptic cleft center.")
        
        center_presyn = np.mean(zone_data['active_presynaptic_points'], axis=0)
        center_postsyn = np.mean(zone_data['active_postsynaptic_points'], axis=0)
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
        'cleft_count': len(clefts['clefts']),
        'zonogram_data': zonogram_data
    }


def extract_active_zonogram(
    active_zonograms,
    clefts,
    tomo_path,
    tomo_type="ddw",
    *,
    alignment_dir: str,
):
    """
    Extract active zonogram data.
    
    Args:
        active_zonograms: Dictionary containing active zonogram data.
        clefts: Dictionary containing synaptic cleft data.
    """
    alignment_dir = require_alignment_dir(alignment_dir)
    import mrcfile
    import torch
    from torch_transform_image import affine_transform_image_3d
    from torch_affine_utils.transforms_3d import T
    
    # Rendering active zonogram
    
    if not active_zonograms or 'zonogram_data' not in active_zonograms:
        print("No active zonograms found for rendering")
        return {
            'status': 'no_active_zonograms',
            'cleft_count': 0,
            'rendered_zonograms': []
        }
    
    # Prepare rendering data
    rendered_zonograms = {}
    # Open tomogram

    mrcs = list((Path(tomo_path) / alignment_dir).glob(f'*{tomo_type}.mrc'))
    if not mrcs:
        print(f"No {tomo_type} MRC files found in {Path(tomo_path) / alignment_dir}")
        return {
            'status': 'no_mrc_files',
            'cleft_count': 0,
            'rendered_zonograms': {}
        }
    with mrcfile.open(mrcs[0], 'r') as mrc:
        data = torch.tensor(mrc.data)
    
    for zone_name, zone_data in active_zonograms['zonogram_data'].items():
        if zone_name not in clefts['clefts']:
            continue
        
        # Get synaptic cleft information
        cleft = clefts['clefts'][zone_name]

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
        'cleft_count': len(clefts['clefts']),
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