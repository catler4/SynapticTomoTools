"""
AMPA poses estimation module for SynapticTomoTools.

This module implements the functionality from findingampa's create-relion-starfile command
to estimate AMPA receptor poses based on AuNP pair analysis.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial import cKDTree as KDTree
from scipy.spatial import distance
from scipy.spatial.transform import Rotation as R
import starfile
import trimesh


def find_pair_centers(coordinates, distances=(6, 12)):
    """
    Find pairs of AuNPs within the specified distance range.
    
    Args:
        coordinates: Array of AuNP coordinates
        distances: Tuple of (min_distance, max_distance) in nm, or None for no cutoff
        
    Returns:
        Array of pairs with their center coordinates
    """
    tree = KDTree(coordinates)
    
    if distances is None:
        # No distance cutoff - find all pairs
        all_pairs = tree.query_pairs(float('inf'))
    else:
        all_pairs = tree.query_pairs(distances[1])
        all_pairs = [pair for pair in all_pairs if distances[0] < distance.euclidean(coordinates[pair[0]], coordinates[pair[1]]) < distances[1]]
    
    return np.array([[pair[0], pair[1], *((coordinates[pair[0]] + coordinates[pair[1]]) / 2)] for pair in all_pairs])


def find_closest_point_on_membrane(centers, postsynaptic_coordinates):
    """
    Find the closest point on the postsynaptic membrane for each center.
    
    Args:
        centers: Array of center coordinates
        postsynaptic_coordinates: Array of postsynaptic membrane coordinates
        
    Returns:
        Array of closest membrane points
    """
    postsynaptic_tree = KDTree(postsynaptic_coordinates)
    return np.array([postsynaptic_coordinates[postsynaptic_tree.query(center)[1]] for center in centers])


def load_postsynaptic_coordinates(tomo_path):
    """
    Load postsynaptic membrane coordinates from GLB file.
    
    Args:
        tomo_path: Path to tomogram directory
        
    Returns:
        Array of postsynaptic membrane coordinates
    """
    postsynaptic_glb_path = Path(tomo_path) / "best_alignment" / "aunps" / "postsynapticmembranes.glb"
    
    if not postsynaptic_glb_path.exists():
        raise FileNotFoundError(f"Postsynaptic membrane GLB file not found: {postsynaptic_glb_path}")
    
    # Load the GLB file using trimesh
    loaded = trimesh.load(str(postsynaptic_glb_path))
    
    # Handle both Mesh and Scene objects
    if hasattr(loaded, 'vertices'):
        # It's a Mesh object
        mesh = loaded
        vertices = mesh.vertices
    else:
        # It's a Scene object - combine all meshes
        vertices_list = []
        for mesh in loaded.geometry.values():
            if hasattr(mesh, 'vertices'):
                vertices_list.append(mesh.vertices)
        
        if not vertices_list:
            raise ValueError("No valid meshes found in the GLB file")
        
        # Combine all vertices
        vertices = np.vstack(vertices_list)
    
    # Get vertex coordinates and transform them
    # The transformation from findingampa: [0,2,1] * [10,-10,10]
    vertices_transformed = vertices[:, [0, 2, 1]] * np.array([10, -10, 10])
    
    return vertices_transformed


def estimate_ampa_poses(
    tomo_name,
    aunp_coordinates,
    postsynaptic_data,
    output_dir,
    output_filename,
    inter_aunp_distance=(6, 12),
    aunp_membrane_distance=(17, 23)
):
    """
    Estimate AMPA receptor poses based on AuNP pair analysis.
    
    Args:
        tomo_name: Name of the tomogram
        aunp_coordinates: Array of AuNP coordinates
        postsynaptic_data: Array of postsynaptic membrane coordinates
        output_dir: Directory to save output files
        output_filename: Base filename for output files
        inter_aunp_distance: Tuple of (min, max) distance between AuNPs in nm
        aunp_membrane_distance: Tuple of (min, max) distance from AuNP to membrane in nm
        
    Returns:
        Dictionary with results summary
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find AuNP pairs within distance range
    pairs = find_pair_centers(aunp_coordinates, distances=inter_aunp_distance)
    if inter_aunp_distance is None:
        print(f"Potential pairs (no AuNP distance cutoff): {len(pairs)}")
    else:
        print(f"Potential pairs after AuNP distance filtering ({inter_aunp_distance}): {len(pairs)}")
    
    if len(pairs) == 0:
        print("No AuNP pairs found within specified distance range")
        return {"status": "no_pairs", "pairs_found": 0}
    
    center_coordinates = pairs[:, 2:]
    aunp1_coordinates = aunp_coordinates[pairs[:, 0].astype(int)]
    aunp2_coordinates = aunp_coordinates[pairs[:, 1].astype(int)]
    
    # Find closest points on membrane
    closest_points = find_closest_point_on_membrane(center_coordinates, postsynaptic_data)
    closest_points_aunp1 = find_closest_point_on_membrane(aunp1_coordinates, postsynaptic_data)
    closest_points_aunp2 = find_closest_point_on_membrane(aunp2_coordinates, postsynaptic_data)
    
    # Filter pairs based on membrane distance (if cutoff is enabled)
    if aunp_membrane_distance is None:
        # No membrane distance cutoff - use all pairs
        valid_pairs_mask = np.ones(len(pairs), dtype=bool)
        print(f"Using all pairs (no membrane distance cutoff): {len(pairs)}")
    else:
        aunp1_selection = np.logical_and(
            aunp_membrane_distance[0] < np.linalg.norm(closest_points_aunp1 - aunp1_coordinates, axis=1),
            np.linalg.norm(closest_points_aunp1 - aunp1_coordinates, axis=1) < aunp_membrane_distance[1]
        )
        aunp2_selection = np.logical_and(
            aunp_membrane_distance[0] < np.linalg.norm(closest_points_aunp2 - aunp2_coordinates, axis=1),
            np.linalg.norm(closest_points_aunp2 - aunp2_coordinates, axis=1) < aunp_membrane_distance[1]
        )
        valid_pairs_mask = np.logical_and(aunp1_selection, aunp2_selection)
        print(f"Pairs after membrane distance filtering ({aunp_membrane_distance}): {np.sum(valid_pairs_mask)}")
    
    # Apply the filtering mask
    pairs = pairs[valid_pairs_mask]
    closest_points = closest_points[valid_pairs_mask]
    
    if aunp_membrane_distance is None:
        print(f"Final pairs (no membrane distance cutoff): {len(pairs)}")
    else:
        print(f"Final pairs after membrane distance filtering ({aunp_membrane_distance}): {len(pairs)}")
    
    if len(pairs) == 0:
        print("No AuNP pairs found within specified membrane distance range")
        return {"status": "no_pairs_after_filtering", "pairs_found": 0}
    
    # Generate AMPA poses
    ampa_positions = []
    results_relion = []
    
    for x, (pair, cp) in enumerate(zip(pairs, closest_points)):
        ampa_positions.append([])
        
        # Add AuNP 1
        ampa_positions[-1].append({
            "o": 1,
            "c": x + 1,
            "x": aunp_coordinates[int(pair[0])][0],
            "y": aunp_coordinates[int(pair[0])][1],
            "z": aunp_coordinates[int(pair[0])][2]
        })
        
        # Add AuNP 2
        ampa_positions[-1].append({
            "o": 1,
            "c": x + 1,
            "x": aunp_coordinates[int(pair[1])][0],
            "y": aunp_coordinates[int(pair[1])][1],
            "z": aunp_coordinates[int(pair[1])][2]
        })
        
        # Add membrane point
        ampa_positions[-1].append({
            "o": 1,
            "c": x + 1,
            "x": cp[0],
            "y": cp[1],
            "z": cp[2]
        })
        
        # Calculate AMPA receptor pose
        vector_center = pair[2:] - cp
        vector_aunp1 = aunp_coordinates[int(pair[0])] - cp
        vector_aunp2 = aunp_coordinates[int(pair[1])] - cp
        vector_norm = vector_center / np.linalg.norm(vector_center)
        
        # Position AMPA receptor 6 nm from membrane (fudge factor from findingampa)
        center = cp + vector_norm * 6
        
        # Calculate rotation between AuNP vectors and reference vectors
        rot, _ = R.align_vectors([[-0.5, 0, 1], [0.5, 0, 1]], [vector_aunp1, vector_aunp2])
        eulers = rot.as_euler("ZYZ", degrees=True)
        
        # Add to RELION results
        results_relion.append({
            "rlnTomoName": tomo_name,
            "rlnCoordinateX": center[0],
            "rlnCoordinateY": center[1],
            "rlnCoordinateZ": center[2],
            "rlnAngleRot": eulers[0],
            "rlnAngleTilt": eulers[1],
            "rlnAnglePsi": eulers[2],
            "faAuNPSeparation": np.linalg.norm(aunp_coordinates[int(pair[0])] - aunp_coordinates[int(pair[1])]),
            "faAuNPMembraneDistance": np.linalg.norm(vector_center),
        })
    
    # Save RELION star file
    starfile.write({
        'particles': pd.DataFrame(results_relion),
        'optics': pd.DataFrame([{'rlnOpticsGroup': 1}])
    }, output_path / f"{output_filename}.star")
    print(f"Saved RELION star file of AMPA poses to {output_path / f'{output_filename}.star'}")
    
    # Save AuNPs used in analysis
    aunps_used_data = []
    for i, result in enumerate(results_relion):
        # Get the pair indices for this AMPA pose
        pair_idx = i
        if pair_idx < len(pairs):
            pair = pairs[pair_idx]
            aunp1_idx = int(pair[0])
            aunp2_idx = int(pair[1])
            
            # Add AuNP 1
            aunps_used_data.append({
                'rlnTomoName': tomo_name,
                'rlnCoordinateX': aunp_coordinates[aunp1_idx][0],
                'rlnCoordinateY': aunp_coordinates[aunp1_idx][1],
                'rlnCoordinateZ': aunp_coordinates[aunp1_idx][2],
                'faAMPA_ID': i + 1,
                'faAuNP_Type': 'AuNP_1',
                'faAuNP_Pair_Index': pair_idx + 1
            })
            
            # Add AuNP 2
            aunps_used_data.append({
                'rlnTomoName': tomo_name,
                'rlnCoordinateX': aunp_coordinates[aunp2_idx][0],
                'rlnCoordinateY': aunp_coordinates[aunp2_idx][1],
                'rlnCoordinateZ': aunp_coordinates[aunp2_idx][2],
                'faAMPA_ID': i + 1,
                'faAuNP_Type': 'AuNP_2',
                'faAuNP_Pair_Index': pair_idx + 1
            })
    
    # Save AuNPs star file (only those used for AMPA poses)
    if aunps_used_data:
        starfile.write({
            'particles': pd.DataFrame(aunps_used_data),
            'optics': pd.DataFrame([{'rlnOpticsGroup': 1}])
        }, output_path / f"{output_filename}_aunps.star")
        print(f"Saved AuNPs used in AMPA poses analysis to {output_path / f'{output_filename}_aunps.star'}")
    
    # Save ALL AuNPs from chosen active zones
    all_aunps_data = []
    for i, coord in enumerate(aunp_coordinates):
        all_aunps_data.append({
            'rlnTomoName': tomo_name,
            'rlnCoordinateX': coord[0],
            'rlnCoordinateY': coord[1],
            'rlnCoordinateZ': coord[2],
            'faAuNP_ID': i + 1,
            'faActive_Zone': 'all'  # Indicates this includes all AuNPs from active zones
        })
    
    if all_aunps_data:
        starfile.write({
            'particles': pd.DataFrame(all_aunps_data),
            'optics': pd.DataFrame([{'rlnOpticsGroup': 1}])
        }, output_path / f"{output_filename}_all_aunps.star")
        print(f"Saved ALL AuNPs from chosen active zones to {output_path / f'{output_filename}_all_aunps.star'}")
    
    # Save summary CSV
    summary_data = []
    for i, result in enumerate(results_relion):
        summary_data.append({
            'AMPA_ID': i + 1,
            'X': result['rlnCoordinateX'],
            'Y': result['rlnCoordinateY'],
            'Z': result['rlnCoordinateZ'],
            'Rot': result['rlnAngleRot'],
            'Tilt': result['rlnAngleTilt'],
            'Psi': result['rlnAnglePsi'],
            'AuNP_Separation_nm': result['faAuNPSeparation'],
            'Membrane_Distance_nm': result['faAuNPMembraneDistance']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_path / f"{output_filename}_summary.csv", index=False)
    print(f"Saved AMPA poses summary to {output_path / f'{output_filename}_summary.csv'}")
    
    return {
        "status": "success",
        "pairs_found": len(pairs),
        "star_file": str(output_path / f"{output_filename}.star"),
        "aunps_file": str(output_path / f"{output_filename}_aunps.star"),
        "all_aunps_file": str(output_path / f"{output_filename}_all_aunps.star"),
        "summary_file": str(output_path / f"{output_filename}_summary.csv"),
        "particles_data": results_relion,
        "aunps_data": aunps_used_data
    }


def run_ampa_poses_analysis(tomo_path, output_dir, aunp_active_zones=None, 
                           inter_aunp_distance=(6, 12), aunp_membrane_distance=(17, 23)):
    """
    Run AMPA poses analysis for a tomogram.
    
    Args:
        tomo_path: Path to tomogram directory
        output_dir: Directory to save results
        aunp_active_zones: List of active zone indices to analyze (None for all)
        inter_aunp_distance: Tuple of (min, max) distance between AuNPs in nm, or None for no cutoff
        aunp_membrane_distance: Tuple of (min, max) distance from AuNP to membrane in nm, or None for no cutoff
        
    Returns:
        Dictionary with analysis results
    """
    tomo_path = Path(tomo_path)
    tomo_name = tomo_path.name
    
    print(f"Running AMPA poses analysis for {tomo_name}")
    
    # Load AuNP data
    aunps_dir = tomo_path / "best_alignment" / "aunps"
    
    if aunp_active_zones is None:
        # Load all AuNPs
        aunp_file = aunps_dir / "aunp_tm_BP_active_zone_all.star"
        if not aunp_file.exists():
            raise FileNotFoundError(f"AuNP file not found: {aunp_file}")
        aunp_data = starfile.read(aunp_file)
    else:
        # Load specific active zones
        aunp_files = []
        for az_id in aunp_active_zones:
            az_file = aunps_dir / f"aunp_tm_BP_active_zone_{az_id}.star"
            if az_file.exists():
                aunp_files.append(az_file)
        
        if not aunp_files:
            raise FileNotFoundError(f"No AuNP files found for active zones: {aunp_active_zones}")
        
        # Load and combine AuNP data
        aunp_data_list = []
        for aunp_file in aunp_files:
            aunp_data = starfile.read(aunp_file)
            aunp_data_list.append(aunp_data)
        
        aunp_data = pd.concat(aunp_data_list, ignore_index=True)
    
    aunp_coordinates = aunp_data[["faCoordinateX", "faCoordinateY", "faCoordinateZ"]].values
    
    # Load postsynaptic membrane data
    postsynaptic_data = load_postsynaptic_coordinates(tomo_path)
    
    # Run AMPA poses estimation
    # Include distance cutoffs in filename for easy identification
    if inter_aunp_distance is None:
        aunp_str = "aunpNONE"
    else:
        aunp_min, aunp_max = inter_aunp_distance
        aunp_str = f"aunp{aunp_min}-{aunp_max}nm"
    
    if aunp_membrane_distance is None:
        membrane_str = "memNONE"
    else:
        membrane_min, membrane_max = aunp_membrane_distance
        membrane_str = f"mem{membrane_min}-{membrane_max}nm"
    
    output_filename = f"{tomo_name}_ampa_poses_{aunp_str}_{membrane_str}"
    results = estimate_ampa_poses(
        tomo_name,
        aunp_coordinates,
        postsynaptic_data,
        output_dir,
        output_filename,
        inter_aunp_distance,
        aunp_membrane_distance
    )
    
    return results
