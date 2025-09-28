"""
Optimized AMPA poses estimation module with optimal AuNP pairing.

This module implements an enhanced version of AMPA receptor pose estimation
that uses maximum matching algorithms to find optimal AuNP pairs while
avoiding steric clashes between predicted AMPA positions.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial import cKDTree as KDTree
from scipy.spatial import distance
from scipy.spatial.transform import Rotation as R
from scipy.optimize import linear_sum_assignment
import pulp
import starfile
import trimesh
from typing import List, Tuple, Dict, Optional
import itertools


def calculate_ampa_position(aunp1_coord, aunp2_coord, postsynaptic_data):
    """
    Calculate AMPA position from AuNP pair and postsynaptic membrane.
    
    Args:
        aunp1_coord: First AuNP coordinate
        aunp2_coord: Second AuNP coordinate  
        postsynaptic_data: Postsynaptic membrane coordinates
        
    Returns:
        AMPA position coordinate
    """
    # Calculate pair center
    pair_center = (aunp1_coord + aunp2_coord) / 2
    
    # Find closest point on membrane
    postsynaptic_tree = KDTree(postsynaptic_data)
    closest_membrane_point = postsynaptic_data[postsynaptic_tree.query(pair_center)[1]]
    
    # Calculate direction vector from membrane to pair center
    direction_vector = pair_center - closest_membrane_point
    direction_norm = direction_vector / np.linalg.norm(direction_vector)
    
    # Position AMPA 6 nm from membrane
    ampa_position = closest_membrane_point + direction_norm * 6
    
    return ampa_position


def check_aunp_membrane_distance_validity(aunp1_coord, aunp2_coord, postsynaptic_data,
                                        aunp_membrane_distance=(17, 23)):
    """
    Check if AuNPs meet membrane distance criteria (AuNP-AuNP distance already checked).
    
    Args:
        aunp1_coord: First AuNP coordinate
        aunp2_coord: Second AuNP coordinate
        postsynaptic_data: Postsynaptic membrane coordinates
        aunp_membrane_distance: (min, max) distance from AuNP to membrane
        
    Returns:
        Boolean indicating if pair meets membrane distance criteria
    """
    if aunp_membrane_distance is None:
        return True
    
    # Check AuNP-membrane distances
    postsynaptic_tree = KDTree(postsynaptic_data)
    
    # Check first AuNP
    dist1 = postsynaptic_tree.query(aunp1_coord)[0]
    if not (aunp_membrane_distance[0] < dist1 < aunp_membrane_distance[1]):
        return False
    
    # Check second AuNP
    dist2 = postsynaptic_tree.query(aunp2_coord)[0]
    if not (aunp_membrane_distance[0] < dist2 < aunp_membrane_distance[1]):
        return False
    
    return True


def check_ampa_steric_clash(ampa_positions, min_distance=5.0):
    """
    Check for steric clashes between AMPA positions.
    
    Args:
        ampa_positions: List of AMPA position coordinates
        min_distance: Minimum allowed distance between AMPA positions
        
    Returns:
        List of clash indices (pairs that are too close)
    """
    if len(ampa_positions) < 2:
        return []
    
    clashes = []
    for i in range(len(ampa_positions)):
        for j in range(i + 1, len(ampa_positions)):
            distance = np.linalg.norm(ampa_positions[i] - ampa_positions[j])
            if distance < min_distance:
                clashes.append((i, j))
    
    return clashes




def find_optimal_ampa_pairs_ilp(aunp_coordinates, postsynaptic_data,
                               inter_aunp_distance=(6, 12),
                               aunp_membrane_distance=(17, 23),
                               ampa_steric_radius=5.0):
    """
    Find the absolute optimal AMPA pairs using Integer Linear Programming (ILP).
    
    This method formulates the problem as an ILP where we maximize the number of
    selected pairs subject to constraints that prevent conflicts.
    
    Args:
        aunp_coordinates: Array of AuNP coordinates
        postsynaptic_data: Array of postsynaptic membrane coordinates
        inter_aunp_distance: Tuple of (min, max) distance between AuNPs in a pair
        aunp_membrane_distance: Tuple of (min, max) distance from AuNP to membrane
        ampa_steric_radius: Minimum distance between AMPA positions
        
    Returns:
        Dictionary with optimal solution details
    """
    print("🔬 Using Integer Linear Programming (ILP) exact optimization method...")
    
    # Create postsynaptic KDTree for efficient membrane distance queries
    postsynaptic_tree = KDTree(postsynaptic_data)
    
    # Find all valid AuNP pairs using KDTree (same as original method)
    print("📊 Finding valid AuNP pairs...")
    tree = KDTree(aunp_coordinates)
    
    if inter_aunp_distance is None:
        all_pairs = tree.query_pairs(float('inf'))
    else:
        all_pairs = tree.query_pairs(inter_aunp_distance[1])
        all_pairs = [pair for pair in all_pairs 
                    if inter_aunp_distance[0] < distance.euclidean(aunp_coordinates[pair[0]], aunp_coordinates[pair[1]]) < inter_aunp_distance[1]]
    
    print(f"   Found {len(all_pairs)} potential AuNP pairs")
    
    # Filter pairs by membrane distance constraints
    print("🔍 Filtering pairs by membrane distance constraints...")
    valid_pairs = []
    for pair in all_pairs:
        if check_aunp_membrane_distance_validity(
            aunp_coordinates[pair[0]], aunp_coordinates[pair[1]], 
            postsynaptic_data, aunp_membrane_distance
        ):
            valid_pairs.append(pair)
    
    print(f"   {len(valid_pairs)} pairs passed membrane distance filtering")
    
    if len(valid_pairs) == 0:
        print("❌ No valid pairs found")
        return {
            'pairs': [],
            'ampa_positions': [],
            'ampa_orientations': [],
            'total_pairs': 0,
            'steric_clashes': 0,
            'unpaired_aunps': [],
            'valid_aunps_count': 0,
            'method': 'ilp_exact'
        }
    
    # Check if problem size is manageable for exact solution
    if len(valid_pairs) > 1000:
        print(f"⚠️  Warning: {len(valid_pairs)} pairs is large for ILP solution.")
        print("   Consider using greedy method for better performance.")
    
    print("🧮 Building Integer Linear Programming model...")
    
    # Create ILP problem
    prob = pulp.LpProblem("AMPA_Pairing", pulp.LpMaximize)
    
    # Decision variables: x[i] = 1 if pair i is selected, 0 otherwise
    x = [pulp.LpVariable(f"pair_{i}", cat='Binary') for i in range(len(valid_pairs))]
    
    # Objective: maximize number of selected pairs
    prob += pulp.lpSum(x)
    
    print("🔗 Adding conflict constraints...")
    conflicts_found = 0
    
    # Add constraints for conflicts
    for i, pair1 in enumerate(valid_pairs):
        for j, pair2 in enumerate(valid_pairs[i+1:], i+1):
            # Check for shared AuNPs
            if (pair1[0] == pair2[0] or pair1[0] == pair2[1] or 
                pair1[1] == pair2[0] or pair1[1] == pair2[1]):
                prob += x[i] + x[j] <= 1  # Can't select both conflicting pairs
                conflicts_found += 1
                continue
            
            # Check for steric clashes between predicted AMPA positions
            ampa1_pos = calculate_ampa_position_from_pair(pair1, aunp_coordinates, postsynaptic_tree)
            ampa2_pos = calculate_ampa_position_from_pair(pair2, aunp_coordinates, postsynaptic_tree)
            
            if distance.euclidean(ampa1_pos, ampa2_pos) < ampa_steric_radius:
                prob += x[i] + x[j] <= 1  # Can't select both conflicting pairs
                conflicts_found += 1
    
    print(f"   Added {conflicts_found} conflict constraints")
    
    # Solve the ILP
    print("🎯 Solving Integer Linear Programming model...")
    print("   This may take several minutes for large problems...")
    
    try:
        # Set solver options for better performance
        solver = pulp.PULP_CBC_CMD(msg=1, timeLimit=1800)  # 30 minute timeout
        
        # Solve
        prob.solve(solver)
        
        # Check if solution was found
        if pulp.LpStatus[prob.status] == 'Optimal':
            print("   ✅ ILP optimization completed successfully")
            
            # Extract solution
            optimal_pairs = []
            for i, var in enumerate(x):
                if var.varValue == 1:
                    optimal_pairs.append(valid_pairs[i])
            
            print(f"✅ Found optimal solution with {len(optimal_pairs)} pairs")
            
            # Calculate AMPA positions and orientations for optimal pairs
            print("🧮 Calculating AMPA positions and orientations...")
            ampa_positions = []
            ampa_orientations = []
            
            for pair in optimal_pairs:
                aunp1_coord = aunp_coordinates[pair[0]]
                aunp2_coord = aunp_coordinates[pair[1]]
                
                # Calculate AMPA position
                ampa_pos = calculate_ampa_position_from_pair(pair, aunp_coordinates, postsynaptic_tree)
                ampa_positions.append(ampa_pos)
                
                # Calculate orientation using original method's strategy
                pair_center = (aunp1_coord + aunp2_coord) / 2
                closest_membrane_point = postsynaptic_data[postsynaptic_tree.query(pair_center)[1]]
                vector_center = pair_center - closest_membrane_point
                vector_aunp1 = aunp1_coord - closest_membrane_point
                vector_aunp2 = aunp2_coord - closest_membrane_point
                rot, _ = R.align_vectors([[-0.5, 0, 1], [0.5, 0, 1]], [vector_aunp1, vector_aunp2])
                euler_angles = rot.as_euler("ZYZ", degrees=True)
                ampa_orientations.append(euler_angles)
            
            # Calculate unpaired AuNPs
            used_aunps = set()
            for pair in optimal_pairs:
                used_aunps.add(pair[0])
                used_aunps.add(pair[1])
            
            unpaired_aunps = [i for i in range(len(aunp_coordinates)) if i not in used_aunps]
            
            # Calculate valid AuNPs (those that were in valid pairs)
            valid_aunps = set()
            for pair in valid_pairs:
                valid_aunps.add(pair[0])
                valid_aunps.add(pair[1])
            
            return {
                'pairs': optimal_pairs,
                'ampa_positions': ampa_positions,
                'ampa_orientations': ampa_orientations,
                'total_pairs': len(optimal_pairs),
                'steric_clashes': 0,  # No clashes in optimal solution
                'unpaired_aunps': unpaired_aunps,
                'valid_aunps_count': len(valid_aunps),  # AuNPs that passed distance cutoffs
                'valid_pairs': valid_pairs,  # Store valid pairs for consistency
                'method': 'ilp_exact',
                'valid_pairs_considered': len(valid_pairs),
                'conflicts_found': conflicts_found
            }
            
        elif pulp.LpStatus[prob.status] == 'Not Solved':
            print("   ⏰ ILP optimization timed out")
            raise TimeoutError("ILP optimization timed out")
        else:
            print(f"   ❌ ILP optimization failed with status: {pulp.LpStatus[prob.status]}")
            raise Exception(f"ILP optimization failed: {pulp.LpStatus[prob.status]}")
            
    except Exception as e:
        print(f"❌ Error in ILP optimization: {e}")
        print("   ILP optimization failed. Please try using the greedy method instead.")
        return {
            'pairs': [],
            'ampa_positions': [],
            'ampa_orientations': [],
            'total_pairs': 0,
            'steric_clashes': 0,
            'unpaired_aunps': [],
            'valid_aunps_count': 0,
            'method': 'ilp_failed',
            'error': str(e)
        }


def calculate_ampa_position_from_pair(pair, aunp_coordinates, postsynaptic_tree):
    """Helper function to calculate AMPA position from a pair."""
    aunp1_coord = aunp_coordinates[pair[0]]
    aunp2_coord = aunp_coordinates[pair[1]]
    pair_center = (aunp1_coord + aunp2_coord) / 2
    
    # Find closest point on postsynaptic membrane
    closest_membrane_point = postsynaptic_tree.data[postsynaptic_tree.query(pair_center)[1]]
    
    # Calculate vector from membrane to pair center
    vector_center = pair_center - closest_membrane_point
    vector_norm = vector_center / np.linalg.norm(vector_center)
    
    # Position AMPA receptor 6 nm from membrane
    ampa_position = closest_membrane_point + vector_norm * 6
    
    return ampa_position


def find_optimal_ampa_pairs(aunp_coordinates, postsynaptic_data,
                           inter_aunp_distance=(6, 12),
                           aunp_membrane_distance=(17, 23),
                           ampa_steric_radius=5.0,
                           convergence_threshold=100,
                           enable_early_termination=True):
    """
    Find optimal AuNP pairs for AMPA pose prediction using iterative optimization.
    
    This function implements a greedy algorithm with backtracking to find
    the maximum number of non-overlapping AuNP pairs that result in
    non-clashing AMPA positions.
    
    Args:
        aunp_coordinates: Array of AuNP coordinates
        postsynaptic_data: Postsynaptic membrane coordinates
        inter_aunp_distance: (min, max) distance between AuNPs
        aunp_membrane_distance: (min, max) distance from AuNP to membrane
        ampa_steric_radius: Minimum distance between AMPA positions
        convergence_threshold: Number of iterations without improvement before convergence
        
    Returns:
        Dictionary containing optimal pairs and AMPA positions
    """
    n_aunps = len(aunp_coordinates)
    
    # Step 1: Generate all valid AuNP pairs using efficient KDTree approach
    print("🔍 Generating valid AuNP pairs...")
    valid_pairs = []
    pair_ampa_positions = []
    valid_aunps = set()  # Track AuNPs that pass distance cutoffs
    
    # Use KDTree for efficient pair finding (same as original method)
    print("  Building KDTree for efficient pair finding...")
    tree = KDTree(aunp_coordinates)
    
    if inter_aunp_distance is None:
        # No distance cutoff - find all pairs
        print("  Finding all AuNP pairs (no distance cutoff)...")
        all_pairs = tree.query_pairs(float('inf'))
    else:
        print(f"  Finding AuNP pairs within {inter_aunp_distance[1]} nm...")
        all_pairs = tree.query_pairs(inter_aunp_distance[1])
        print(f"  Found {len(all_pairs)} candidate pairs, filtering by minimum distance...")
        # Filter by minimum distance
        all_pairs = [pair for pair in all_pairs if inter_aunp_distance[0] < distance.euclidean(aunp_coordinates[pair[0]], aunp_coordinates[pair[1]]) < inter_aunp_distance[1]]
        print(f"  After distance filtering: {len(all_pairs)} pairs")
    
    # Now check membrane distance constraints for each pair
    print(f"  Checking membrane distance constraints for {len(all_pairs)} pairs...")
    for pair_idx, (i, j) in enumerate(all_pairs):
        if pair_idx % 1000 == 0 and pair_idx > 0:
            print(f"    Processed {pair_idx}/{len(all_pairs)} pairs...")
        
        if check_aunp_membrane_distance_validity(
            aunp_coordinates[i], aunp_coordinates[j], postsynaptic_data,
            aunp_membrane_distance
        ):
            ampa_pos = calculate_ampa_position(
                aunp_coordinates[i], aunp_coordinates[j], postsynaptic_data
            )
            valid_pairs.append((i, j))
            pair_ampa_positions.append(ampa_pos)
            valid_aunps.add(i)
            valid_aunps.add(j)
    
    print(f"✅ Found {len(valid_pairs)} valid AuNP pairs from {len(all_pairs)} candidates")
    
    if len(valid_pairs) == 0:
        return {
            'pairs': [],
            'ampa_positions': [],
            'unpaired_aunps': list(range(n_aunps)),
            'total_pairs': 0,
            'steric_clashes': 0,
            'valid_aunps_count': len(valid_aunps),
            'valid_pairs': valid_pairs
        }
    
    # Step 2: Find optimal non-overlapping pairs using iterative optimization
    # Track top 3 solutions
    top_solutions = []
    
    def add_solution(pairs, ampa_positions, ampa_orientations, unpaired_aunps, total_pairs, steric_clashes):
        """Add a solution to the top solutions list, maintaining top 3"""
        solution = {
            'pairs': pairs,
            'ampa_positions': ampa_positions,
            'ampa_orientations': ampa_orientations,
            'unpaired_aunps': unpaired_aunps,
            'total_pairs': total_pairs,
            'steric_clashes': steric_clashes
        }
        
        # Add to list
        top_solutions.append(solution)
        
        # Sort by total_pairs (descending), then by steric_clashes (ascending)
        top_solutions.sort(key=lambda x: (-x['total_pairs'], x['steric_clashes']))
        
        # Keep only top 3
        if len(top_solutions) > 3:
            top_solutions.pop()
    
    # Initialize with empty solution
    add_solution([], [], [], list(range(n_aunps)), 0, float('inf'))
    
    # Always run minimum iterations before considering convergence
    # Minimum is at least 100 or number of AuNPs, whichever is larger
    min_iterations = max(100, n_aunps)
    # No maximum iterations limit - only terminate on convergence
    
    # Create postsynaptic KDTree once for efficient membrane distance queries
    postsynaptic_tree = KDTree(postsynaptic_data)
    
    # Convergence tracking variables
    no_improvement_count = 0
    solution_qualities = []
    max_possible_pairs = len(valid_pairs)
    
    print(f"Running optimization with minimum {min_iterations} iterations, no maximum limit (convergence-based termination)")
    
    # Progress tracking
    progress_interval = max(1, min_iterations // 20)  # Update every 5% of minimum iterations
    last_progress_update = 0
    
    # Try different starting points for the greedy algorithm
    iteration = 0
    while True:  # Run until convergence
        # Progress updates
        if iteration % progress_interval == 0:
            # Get current best solution info
            current_best = top_solutions[0] if top_solutions else {'total_pairs': 0, 'steric_clashes': float('inf')}
            
            print(f"\r🔄 Iteration {iteration + 1}: Best: {current_best['total_pairs']} pairs, {current_best['steric_clashes']} clashes | "
                  f"No improvement: {no_improvement_count}/{convergence_threshold}", end='', flush=True)
            
            last_progress_update = iteration
        # Create a copy of valid pairs for this iteration
        remaining_pairs = valid_pairs.copy()
        remaining_ampa_positions = pair_ampa_positions.copy()
        
        # Shuffle to try different starting points
        if iteration > 0:
            indices = np.random.permutation(len(remaining_pairs))
            remaining_pairs = [remaining_pairs[i] for i in indices]
            remaining_ampa_positions = [remaining_ampa_positions[i] for i in indices]
        
        # Greedy selection with steric constraint checking
        selected_pairs = []
        selected_ampa_positions = []
        used_aunps = set()
        
        for pair_idx, (i, j) in enumerate(remaining_pairs):
            # Check if AuNPs are already used
            if i in used_aunps or j in used_aunps:
                continue
            
            # Check for steric clashes with already selected AMPA positions
            temp_ampa_positions = selected_ampa_positions + [remaining_ampa_positions[pair_idx]]
            clashes = check_ampa_steric_clash(temp_ampa_positions, ampa_steric_radius)
            
            # If no clashes, add this pair
            if len(clashes) == 0:
                selected_pairs.append((i, j))
                selected_ampa_positions.append(remaining_ampa_positions[pair_idx])
                used_aunps.add(i)
                used_aunps.add(j)
        
        # Evaluate this solution
        unpaired_aunps = [i for i in range(n_aunps) if i not in used_aunps]
        final_clashes = check_ampa_steric_clash(selected_ampa_positions, ampa_steric_radius)
        current_pairs = len(selected_pairs)
        
        # Calculate orientations for this solution using original method's strategy
        selected_ampa_orientations = []
        for pair in selected_pairs:
            aunp1_idx, aunp2_idx = pair
            aunp1_coord = aunp_coordinates[aunp1_idx]
            aunp2_coord = aunp_coordinates[aunp2_idx]
            
            # Use original method's orientation calculation strategy
            pair_center = (aunp1_coord + aunp2_coord) / 2
            closest_membrane_point = postsynaptic_data[
                postsynaptic_tree.query(pair_center)[1]
            ]
            vector_center = pair_center - closest_membrane_point
            vector_aunp1 = aunp1_coord - closest_membrane_point
            vector_aunp2 = aunp2_coord - closest_membrane_point
            rot, _ = R.align_vectors([[-0.5, 0, 1], [0.5, 0, 1]], [vector_aunp1, vector_aunp2])
            euler_angles = rot.as_euler("ZYZ", degrees=True)
            selected_ampa_orientations.append(euler_angles)
        
        # Track solution quality
        solution_qualities.append(current_pairs)
        
        # Check for improvement (compare with best solution BEFORE adding current solution)
        best_solution = top_solutions[0]  # Best solution is always first
        solution_improved = False
        if (current_pairs > best_solution['total_pairs'] or 
            (current_pairs == best_solution['total_pairs'] and 
             len(final_clashes) < best_solution['steric_clashes'])):
            solution_improved = True
            no_improvement_count = 0
            
            # Milestone messages for significant improvements
            if current_pairs > best_solution['total_pairs']:
                improvement = current_pairs - best_solution['total_pairs']
                print(f"\n🎯 New best solution found! +{improvement} pairs (now {current_pairs} pairs, {len(final_clashes)} clashes)")
        else:
            no_improvement_count += 1
        
        # Add this solution to top solutions (after checking for improvement)
        add_solution(selected_pairs, selected_ampa_positions, selected_ampa_orientations, unpaired_aunps, current_pairs, len(final_clashes))
        
        # Progress indicator
        if iteration == min_iterations - 1:
            print(f"\nCompleted minimum {min_iterations} iterations. Now monitoring for convergence...")
        
        # Early termination conditions (only after minimum iterations)
        if enable_early_termination and iteration >= min_iterations:
            # Perfect solution found
            if (best_solution['total_pairs'] == max_possible_pairs and 
                best_solution['steric_clashes'] == 0):
                print(f"Found perfect solution with {max_possible_pairs} pairs and no clashes after {iteration + 1} iterations!")
                break
            
            # No improvement for many iterations
            if no_improvement_count >= convergence_threshold:
                print(f"Converged after {iteration + 1} iterations (no improvement for {convergence_threshold} iterations)")
                break
            
            # Statistical convergence (check last 100 iterations)
            if len(solution_qualities) >= 100:
                recent_qualities = solution_qualities[-100:]
                std_quality = np.std(recent_qualities)
                if std_quality < 0.5:  # Less than 0.5 pairs variation
                    print(f"\nConverged after {iteration + 1} iterations (std={std_quality:.2f} over last 100 iterations)")
                    break
        
        # Increment iteration counter
        iteration += 1
    
    # Clear progress bar and add final newline
    print()  # Final newline after progress bar
    
    # Completion message
    total_iterations = iteration + 1
    print(f"✅ Optimization completed after {total_iterations} iterations")
    
    # Print top 3 solutions
    print(f"\nTop 3 solutions found:")
    for i, solution in enumerate(top_solutions[:3]):
        print(f"  Solution {i+1}: {solution['total_pairs']} pairs, "
              f"{len(solution['unpaired_aunps'])} unpaired AuNPs, "
              f"{solution['steric_clashes']} steric clashes")
    
    # Use the best solution as the primary result
    best_solution = top_solutions[0]
    
    # Add metadata to best solution
    best_solution['valid_aunps_count'] = len(valid_aunps)
    best_solution['valid_pairs'] = valid_pairs
    best_solution['top_3_solutions'] = top_solutions[:3]  # Include all top 3 solutions
    
    return best_solution


def estimate_ampa_poses_optimized(
    tomo_name,
    aunp_coordinates,
    postsynaptic_data,
    output_dir,
    output_filename,
    inter_aunp_distance=(6, 12),
    aunp_membrane_distance=(17, 23),
    ampa_steric_radius=5.0,
    method="greedy"
):
    """
    Estimate AMPA receptor poses using optimized AuNP pairing.
    
    Args:
        tomo_name: Name of the tomogram
        aunp_coordinates: Array of AuNP coordinates
        postsynaptic_data: Array of postsynaptic membrane coordinates
        output_dir: Directory to save output files
        output_filename: Base filename for output files
        inter_aunp_distance: Tuple of (min, max) distance between AuNPs in nm
        aunp_membrane_distance: Tuple of (min, max) distance from AuNP to membrane in nm
        ampa_steric_radius: Minimum distance between AMPA positions in nm
        method: Optimization method ("greedy" or "ilp")
        
    Returns:
        Dictionary with results summary
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find optimal AuNP pairs using specified method
    if method == "ilp":
        optimal_solution = find_optimal_ampa_pairs_ilp(
            aunp_coordinates, postsynaptic_data,
            inter_aunp_distance, aunp_membrane_distance, ampa_steric_radius
        )
    else:  # default to greedy
        optimal_solution = find_optimal_ampa_pairs(
            aunp_coordinates, postsynaptic_data,
            inter_aunp_distance, aunp_membrane_distance, ampa_steric_radius,
            convergence_threshold=100, enable_early_termination=True
        )
    
    if optimal_solution['total_pairs'] == 0:
        print("No valid AuNP pairs found")
        return {"status": "no_pairs", "pairs_found": 0}
    
    # Create postsynaptic KDTree for efficient membrane distance queries
    postsynaptic_tree = KDTree(postsynaptic_data)
    
    # Generate AMPA poses for optimal pairs
    results_relion = []
    aunps_used_data = []
    
    for i, (pair, ampa_pos, eulers) in enumerate(zip(optimal_solution['pairs'], 
                                                      optimal_solution['ampa_positions'],
                                                      optimal_solution['ampa_orientations'])):
        aunp1_idx, aunp2_idx = pair
        aunp1_coord = aunp_coordinates[aunp1_idx]
        aunp2_coord = aunp_coordinates[aunp2_idx]
        
        # Calculate distances (use stored orientation instead of recalculating)
        pair_center = (aunp1_coord + aunp2_coord) / 2
        closest_membrane_point = postsynaptic_data[
            postsynaptic_tree.query(pair_center)[1]
        ]
        vector_center = pair_center - closest_membrane_point
        vector_aunp1 = aunp1_coord - closest_membrane_point
        vector_aunp2 = aunp2_coord - closest_membrane_point
        
        # Calculate detailed distance measurements
        aunp_separation = np.linalg.norm(aunp1_coord - aunp2_coord)
        ampa_membrane_distance = np.linalg.norm(ampa_pos - closest_membrane_point)
        aunp1_membrane_distance = np.linalg.norm(aunp1_coord - closest_membrane_point)
        aunp2_membrane_distance = np.linalg.norm(aunp2_coord - closest_membrane_point)
        
        # Add to RELION results
        results_relion.append({
            "rlnTomoName": tomo_name,
            "rlnCoordinateX": ampa_pos[0],
            "rlnCoordinateY": ampa_pos[1],
            "rlnCoordinateZ": ampa_pos[2],
            "rlnAngleRot": eulers[0],
            "rlnAngleTilt": eulers[1],
            "rlnAnglePsi": eulers[2],
            "faAuNPSeparation": aunp_separation,
            "faAuNPMembraneDistance": ampa_membrane_distance,
            "faAuNP1MembraneDistance": aunp1_membrane_distance,
            "faAuNP2MembraneDistance": aunp2_membrane_distance,
            "faAMPA_ID": i + 1,
            "faAuNP1_ID": aunp1_idx + 1,
            "faAuNP2_ID": aunp2_idx + 1
        })
        
        # Add AuNPs to used data with distance measurements
        aunps_used_data.extend([
            {
                'rlnTomoName': tomo_name,
                'rlnCoordinateX': aunp1_coord[0],
                'rlnCoordinateY': aunp1_coord[1],
                'rlnCoordinateZ': aunp1_coord[2],
                'faAMPA_ID': i + 1,
                'faAuNP_Type': 'AuNP_1',
                'faAuNP_Pair_Index': i + 1,
                'faAuNP_Original_ID': aunp1_idx + 1,
                'faAuNPSeparation': aunp_separation,
                'faAuNPMembraneDistance': aunp1_membrane_distance,
                'faAMPA_MembraneDistance': ampa_membrane_distance
            },
            {
                'rlnTomoName': tomo_name,
                'rlnCoordinateX': aunp2_coord[0],
                'rlnCoordinateY': aunp2_coord[1],
                'rlnCoordinateZ': aunp2_coord[2],
                'faAMPA_ID': i + 1,
                'faAuNP_Type': 'AuNP_2',
                'faAuNP_Pair_Index': i + 1,
                'faAuNP_Original_ID': aunp2_idx + 1,
                'faAuNPSeparation': aunp_separation,
                'faAuNPMembraneDistance': aunp2_membrane_distance,
                'faAMPA_MembraneDistance': ampa_membrane_distance
            }
        ])
    
    # Save RELION star file
    starfile.write({
        'particles': pd.DataFrame(results_relion),
        'optics': pd.DataFrame([{'rlnOpticsGroup': 1}])
    }, output_path / f"{output_filename}.star")
    print(f"Saved optimized RELION star file to {output_path / f'{output_filename}.star'}")
    
    # Save AuNPs used in analysis
    if aunps_used_data:
        starfile.write({
            'particles': pd.DataFrame(aunps_used_data),
            'optics': pd.DataFrame([{'rlnOpticsGroup': 1}])
        }, output_path / f"{output_filename}_paired_aunps.star")
        print(f"Saved paired AuNPs from optimized analysis to {output_path / f'{output_filename}_paired_aunps.star'}")
    
    # Save unpaired AuNPs with membrane distance
    unpaired_aunps_data = []
    postsynaptic_tree = KDTree(postsynaptic_data)
    for aunp_idx in optimal_solution['unpaired_aunps']:
        coord = aunp_coordinates[aunp_idx]
        membrane_distance = postsynaptic_tree.query(coord)[0]
        unpaired_aunps_data.append({
                'rlnTomoName': tomo_name,
            'rlnCoordinateX': coord[0],
            'rlnCoordinateY': coord[1],
            'rlnCoordinateZ': coord[2],
            'faAuNP_ID': aunp_idx + 1,
            'faStatus': 'unpaired',
            'faAuNPMembraneDistance': membrane_distance
        })
    
    if unpaired_aunps_data:
        starfile.write({
            'particles': pd.DataFrame(unpaired_aunps_data),
            'optics': pd.DataFrame([{'rlnOpticsGroup': 1}])
        }, output_path / f"{output_filename}_unpaired_aunps.star")
        print(f"Saved {len(unpaired_aunps_data)} unpaired AuNPs to {output_path / f'{output_filename}_unpaired_aunps.star'}")
    
    # Save AuNPs that passed initial filtering but were excluded during optimization
    filtered_out_aunps_data = []
    used_aunps_in_pairs = set()
    for pair in optimal_solution['pairs']:
        used_aunps_in_pairs.add(pair[0])
        used_aunps_in_pairs.add(pair[1])
    
    # Find AuNPs that were in valid pairs but not selected
    valid_aunps_in_pairs = set()
    for pair in optimal_solution.get('valid_pairs', []):
        valid_aunps_in_pairs.add(pair[0])
        valid_aunps_in_pairs.add(pair[1])
    
    # AuNPs that passed filtering but were excluded during optimization
    filtered_out_aunps = valid_aunps_in_pairs - used_aunps_in_pairs
    
    for aunp_idx in filtered_out_aunps:
        coord = aunp_coordinates[aunp_idx]
        membrane_distance = postsynaptic_tree.query(coord)[0]
        filtered_out_aunps_data.append({
                'rlnTomoName': tomo_name,
            'rlnCoordinateX': coord[0],
            'rlnCoordinateY': coord[1],
            'rlnCoordinateZ': coord[2],
            'faAuNP_ID': aunp_idx + 1,
            'faStatus': 'filtered_out_during_optimization',
            'faAuNPMembraneDistance': membrane_distance
        })
    
    if filtered_out_aunps_data:
        starfile.write({
            'particles': pd.DataFrame(filtered_out_aunps_data),
            'optics': pd.DataFrame([{'rlnOpticsGroup': 1}])
        }, output_path / f"{output_filename}_filtered_out_aunps.star")
        print(f"Saved {len(filtered_out_aunps_data)} filtered out AuNPs to {output_path / f'{output_filename}_filtered_out_aunps.star'}")
    
    # Save top 3 solutions to separate directories (only for greedy method)
    if method == "greedy":
        top_3_solutions = optimal_solution.get('top_3_solutions', [])
        if len(top_3_solutions) > 1:  # Only save if we have multiple solutions
            print(f"\n💾 Saving top {len(top_3_solutions)} solutions to separate directories...")
        
        for i, solution in enumerate(top_3_solutions[:3]):
            if solution['total_pairs'] > 0:  # Only save non-empty solutions
                # Create solution directory
                solution_dir = output_path / f"solution_{i+1}"
                solution_dir.mkdir(exist_ok=True)
                
                # Generate AMPA poses for this solution
                solution_results_relion = []
                solution_aunps_used_data = []
                solution_unpaired_data = []
                solution_filtered_out_data = []
                
                # Track used AuNPs for this solution
                solution_used_aunps = set()
                for pair in solution['pairs']:
                    solution_used_aunps.add(pair[0])
                    solution_used_aunps.add(pair[1])
                
                for j, (pair, ampa_pos, euler_angles) in enumerate(zip(solution['pairs'], solution['ampa_positions'], solution['ampa_orientations'])):
                    aunp1_idx, aunp2_idx = pair
                    aunp1_coord = aunp_coordinates[aunp1_idx]
                    aunp2_coord = aunp_coordinates[aunp2_idx]
                    
                    # Calculate distances
                    aunp_separation = np.linalg.norm(aunp2_coord - aunp1_coord)
                    aunp1_membrane_distance = postsynaptic_tree.query(aunp1_coord)[0]
                    aunp2_membrane_distance = postsynaptic_tree.query(aunp2_coord)[0]
                    ampa_membrane_distance = postsynaptic_tree.query(ampa_pos)[0]
                    
                    # Use stored orientation instead of recalculating
                    
                    # Add to RELION results
                    solution_results_relion.append({
                        'rlnTomoName': tomo_name,
                        'rlnCoordinateX': ampa_pos[0],
                        'rlnCoordinateY': ampa_pos[1],
                        'rlnCoordinateZ': ampa_pos[2],
                        'rlnAngleRot': euler_angles[0],
                        'rlnAngleTilt': euler_angles[1],
                        'rlnAnglePsi': euler_angles[2],
                        'faAMPA_ID': j + 1,
                        'faAuNP1_ID': aunp1_idx + 1,
                        'faAuNP2_ID': aunp2_idx + 1,
                        'faAuNPSeparation': aunp_separation,
                        'faAuNPMembraneDistance': (aunp1_membrane_distance + aunp2_membrane_distance) / 2,
                        'faAuNP1MembraneDistance': aunp1_membrane_distance,
                        'faAuNP2MembraneDistance': aunp2_membrane_distance
                    })
                    
                    # Add AuNPs to used data
                    solution_aunps_used_data.extend([
                        {
                            'rlnTomoName': tomo_name,
                            'rlnCoordinateX': aunp1_coord[0],
                            'rlnCoordinateY': aunp1_coord[1],
                            'rlnCoordinateZ': aunp1_coord[2],
                            'faAMPA_ID': j + 1,
                            'faAuNP_Type': 'AuNP_1',
                            'faAuNP_Pair_Index': j + 1,
                            'faAuNP_Original_ID': aunp1_idx + 1,
                            'faAuNPSeparation': aunp_separation,
                            'faAuNPMembraneDistance': aunp1_membrane_distance,
                            'faAMPA_MembraneDistance': ampa_membrane_distance
                        },
                        {
                            'rlnTomoName': tomo_name,
                            'rlnCoordinateX': aunp2_coord[0],
                            'rlnCoordinateY': aunp2_coord[1],
                            'rlnCoordinateZ': aunp2_coord[2],
                            'faAMPA_ID': j + 1,
                            'faAuNP_Type': 'AuNP_2',
                            'faAuNP_Pair_Index': j + 1,
                            'faAuNP_Original_ID': aunp2_idx + 1,
                            'faAuNPSeparation': aunp_separation,
                            'faAuNPMembraneDistance': aunp2_membrane_distance,
                            'faAMPA_MembraneDistance': ampa_membrane_distance
                        }
                    ])
                
                # Generate unpaired AuNPs for this solution
                for aunp_idx in solution['unpaired_aunps']:
                    coord = aunp_coordinates[aunp_idx]
                    membrane_distance = postsynaptic_tree.query(coord)[0]
                    solution_unpaired_data.append({
            'rlnTomoName': tomo_name,
            'rlnCoordinateX': coord[0],
            'rlnCoordinateY': coord[1],
            'rlnCoordinateZ': coord[2],
                        'faAuNP_ID': aunp_idx + 1,
                        'faStatus': 'unpaired',
                        'faAuNPMembraneDistance': membrane_distance
                    })
                
                # Generate filtered out AuNPs for this solution
                valid_aunps_in_pairs = set()
                for pair in optimal_solution.get('valid_pairs', []):
                    valid_aunps_in_pairs.add(pair[0])
                    valid_aunps_in_pairs.add(pair[1])
                
                solution_filtered_out_aunps = valid_aunps_in_pairs - solution_used_aunps
                for aunp_idx in solution_filtered_out_aunps:
                    coord = aunp_coordinates[aunp_idx]
                    membrane_distance = postsynaptic_tree.query(coord)[0]
                    solution_filtered_out_data.append({
                        'rlnTomoName': tomo_name,
                        'rlnCoordinateX': coord[0],
                        'rlnCoordinateY': coord[1],
                        'rlnCoordinateZ': coord[2],
                        'faAuNP_ID': aunp_idx + 1,
                        'faStatus': 'filtered_out_during_optimization',
                        'faAuNPMembraneDistance': membrane_distance
                    })
                
                # Save all files for this solution
                solution_base_filename = f"{tomo_name}_ampa_poses_optimized_solution_{i+1}"
                
                # Save RELION star file (AMPA poses)
        starfile.write({
                    'particles': pd.DataFrame(solution_results_relion),
            'optics': pd.DataFrame([{'rlnOpticsGroup': 1}])
                }, solution_dir / f"{solution_base_filename}.star")
                
                # Save paired AuNPs star file
                if solution_aunps_used_data:
                    starfile.write({
                        'particles': pd.DataFrame(solution_aunps_used_data),
                        'optics': pd.DataFrame([{'rlnOpticsGroup': 1}])
                    }, solution_dir / f"{solution_base_filename}_paired_aunps.star")
                
                # Save unpaired AuNPs star file
                if solution_unpaired_data:
                    starfile.write({
                        'particles': pd.DataFrame(solution_unpaired_data),
                        'optics': pd.DataFrame([{'rlnOpticsGroup': 1}])
                    }, solution_dir / f"{solution_base_filename}_unpaired_aunps.star")
                
                # Save filtered out AuNPs star file
                if solution_filtered_out_data:
                    starfile.write({
                        'particles': pd.DataFrame(solution_filtered_out_data),
                        'optics': pd.DataFrame([{'rlnOpticsGroup': 1}])
                    }, solution_dir / f"{solution_base_filename}_filtered_out_aunps.star")
                
                # Save summary CSV for this solution
                solution_summary_data = []
                for j, result in enumerate(solution_results_relion):
                    solution_summary_data.append({
                        'AMPA_ID': j + 1,
                        'X': result['rlnCoordinateX'],
                        'Y': result['rlnCoordinateY'],
                        'Z': result['rlnCoordinateZ'],
                        'Rot': result['rlnAngleRot'],
                        'Tilt': result['rlnAngleTilt'],
                        'Psi': result['rlnAnglePsi'],
                        'AuNP1_ID': result['faAuNP1_ID'],
                        'AuNP2_ID': result['faAuNP2_ID'],
                        'AuNP_Separation': result['faAuNPSeparation'],
                        'AuNP_Membrane_Distance': result['faAuNPMembraneDistance'],
                        'AuNP1_Membrane_Distance': result['faAuNP1MembraneDistance'],
                        'AuNP2_Membrane_Distance': result['faAuNP2MembraneDistance']
                    })
                
                solution_summary_df = pd.DataFrame(solution_summary_data)
                solution_summary_df.to_csv(solution_dir / f"{solution_base_filename}_summary.csv", index=False)
                
                print(f"  Solution {i+1}: {solution['total_pairs']} pairs, {solution['steric_clashes']} clashes → {solution_dir.name}/")
                print(f"    Files: {solution_base_filename}.star, {solution_base_filename}_paired_aunps.star, {solution_base_filename}_unpaired_aunps.star, {solution_base_filename}_filtered_out_aunps.star, {solution_base_filename}_summary.csv")
        
        # Generate consensus poses file - pairs that appear in all 3 solutions
        # Since pose calculation is deterministic, same AuNP pair = same pose
        print(f"\n🔗 Generating consensus poses from top {len(top_3_solutions)} solutions...")
        print(f"   Looking for AuNP pairs that appear in all 3 solutions...")
        consensus_data = []
        consensus_aunps_data = []
        
        # Track which AuNP pairs appear in each solution
        pair_occurrence_count = {}
        pair_to_pose_data = {}  # Store pose data for the first occurrence of each pair
        
        for i, solution in enumerate(top_3_solutions[:3]):
            if solution['total_pairs'] > 0:
                for j, (pair, ampa_pos, euler_angles) in enumerate(zip(solution['pairs'], solution['ampa_positions'], solution['ampa_orientations'])):
                    pair_key = tuple(sorted(pair))  # Sort to ensure consistent key
                    
                    if pair_key not in pair_occurrence_count:
                        pair_occurrence_count[pair_key] = 0
                        # Store pose data from first occurrence (all occurrences should be identical)
                        aunp1_idx, aunp2_idx = pair
                        aunp1_coord = aunp_coordinates[aunp1_idx]
                        aunp2_coord = aunp_coordinates[aunp2_idx]
                        
                        pair_to_pose_data[pair_key] = {
                            'pair': pair,
                            'ampa_pos': ampa_pos,
                            'euler_angles': euler_angles,  # Use stored orientation instead of recalculating
                            'aunp1_coord': aunp1_coord,
                            'aunp2_coord': aunp2_coord
                        }
                    
                    pair_occurrence_count[pair_key] += 1
        
        # Create consensus entries for pairs that appear in all 3 solutions
        consensus_id = 1
        for pair_key, count in pair_occurrence_count.items():
            if count >= 3:  # Appears in all 3 solutions
                pose_data = pair_to_pose_data[pair_key]
                pair = pose_data['pair']
                ampa_pos = pose_data['ampa_pos']
                euler_angles = pose_data['euler_angles']
                aunp1_coord = pose_data['aunp1_coord']
                aunp2_coord = pose_data['aunp2_coord']
                
                aunp1_idx, aunp2_idx = pair
                
                # Calculate distances
                aunp_separation = np.linalg.norm(aunp2_coord - aunp1_coord)
                aunp1_membrane_distance = postsynaptic_tree.query(aunp1_coord)[0]
                aunp2_membrane_distance = postsynaptic_tree.query(aunp2_coord)[0]
                ampa_membrane_distance = postsynaptic_tree.query(ampa_pos)[0]
                
                # Add to consensus data
                consensus_data.append({
                    'rlnTomoName': tomo_name,
                    'rlnCoordinateX': ampa_pos[0],
                    'rlnCoordinateY': ampa_pos[1],
                    'rlnCoordinateZ': ampa_pos[2],
                    'rlnAngleRot': euler_angles[0],
                    'rlnAngleTilt': euler_angles[1],
                    'rlnAnglePsi': euler_angles[2],
                    'faAMPA_ID': consensus_id,
                    'faAuNP1_ID': aunp1_idx + 1,
                    'faAuNP2_ID': aunp2_idx + 1,
                    'faAuNPSeparation': aunp_separation,
                    'faAuNPMembraneDistance': (aunp1_membrane_distance + aunp2_membrane_distance) / 2,
                    'faAuNP1MembraneDistance': aunp1_membrane_distance,
                    'faAuNP2MembraneDistance': aunp2_membrane_distance,
                    'faConsensusCount': count,
                    'faConsensusSolutions': '1/2/3'  # All 3 solutions
                })
                
                # Add AuNPs to consensus data
                consensus_aunps_data.extend([
                    {
                        'rlnTomoName': tomo_name,
                        'rlnCoordinateX': aunp1_coord[0],
                        'rlnCoordinateY': aunp1_coord[1],
                        'rlnCoordinateZ': aunp1_coord[2],
                        'faAMPA_ID': consensus_id,
                        'faAuNP_Type': 'AuNP_1',
                        'faAuNP_Pair_Index': consensus_id,
                        'faAuNP_Original_ID': aunp1_idx + 1,
                        'faAuNPSeparation': aunp_separation,
                        'faAuNPMembraneDistance': aunp1_membrane_distance,
                        'faAMPA_MembraneDistance': ampa_membrane_distance,
                        'faConsensusCount': count
                    },
                    {
                        'rlnTomoName': tomo_name,
                        'rlnCoordinateX': aunp2_coord[0],
                        'rlnCoordinateY': aunp2_coord[1],
                        'rlnCoordinateZ': aunp2_coord[2],
                        'faAMPA_ID': consensus_id,
                        'faAuNP_Type': 'AuNP_2',
                        'faAuNP_Pair_Index': consensus_id,
                        'faAuNP_Original_ID': aunp2_idx + 1,
                        'faAuNPSeparation': aunp_separation,
                        'faAuNPMembraneDistance': aunp2_membrane_distance,
                        'faAMPA_MembraneDistance': ampa_membrane_distance,
                        'faConsensusCount': count
                    }
                ])
                
                consensus_id += 1
        
        # Save consensus files
        if consensus_data:
            consensus_filename = f"{output_filename}_consensus"
            
            # Save consensus RELION star file
            starfile.write({
                'particles': pd.DataFrame(consensus_data),
                'optics': pd.DataFrame([{'rlnOpticsGroup': 1}])
            }, output_path / f"{consensus_filename}.star")
            
            # Save consensus AuNPs star file
            if consensus_aunps_data:
                starfile.write({
                    'particles': pd.DataFrame(consensus_aunps_data),
                    'optics': pd.DataFrame([{'rlnOpticsGroup': 1}])
                }, output_path / f"{consensus_filename}_paired_aunps.star")
            
            # Save consensus summary CSV
            consensus_summary_data = []
            for result in consensus_data:
                consensus_summary_data.append({
                    'AMPA_ID': result['faAMPA_ID'],
                    'X': result['rlnCoordinateX'],
                    'Y': result['rlnCoordinateY'],
                    'Z': result['rlnCoordinateZ'],
                    'Rot': result['rlnAngleRot'],
                    'Tilt': result['rlnAngleTilt'],
                    'Psi': result['rlnAnglePsi'],
                    'AuNP1_ID': result['faAuNP1_ID'],
                    'AuNP2_ID': result['faAuNP2_ID'],
                    'AuNP_Separation': result['faAuNPSeparation'],
                    'AuNP_Membrane_Distance': result['faAuNPMembraneDistance'],
                    'AuNP1_Membrane_Distance': result['faAuNP1MembraneDistance'],
                    'AuNP2_Membrane_Distance': result['faAuNP2MembraneDistance'],
                    'Consensus_Count': result['faConsensusCount'],
                    'Consensus_Solutions': result['faConsensusSolutions']
                })
            
            consensus_summary_df = pd.DataFrame(consensus_summary_data)
            consensus_summary_df.to_csv(output_path / f"{consensus_filename}_summary.csv", index=False)
            
            print(f"  Consensus: {len(consensus_data)} AuNP pairs found in all 3 solutions → {consensus_filename}.star")
        else:
            print(f"  Consensus: No AuNP pairs found in all 3 solutions")
    
    # End of greedy-specific code
    # Save summary CSV with optimization details
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
            'AMPA_Membrane_Distance_nm': result['faAuNPMembraneDistance'],
            'AuNP1_Membrane_Distance_nm': result['faAuNP1MembraneDistance'],
            'AuNP2_Membrane_Distance_nm': result['faAuNP2MembraneDistance'],
            'AuNP1_ID': result['faAuNP1_ID'],
            'AuNP2_ID': result['faAuNP2_ID']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_path / f"{output_filename}_summary.csv", index=False)
    print(f"Saved optimized AMPA poses summary to {output_path / f'{output_filename}_summary.csv'}")
    
    # Save optimization statistics
    total_aunps = len(aunp_coordinates)
    valid_aunps_count = optimal_solution.get('valid_aunps_count', 0)
    total_pairs = optimal_solution['total_pairs']
    
    # Calculate both efficiency metrics
    overall_efficiency = (total_pairs * 2) / total_aunps if total_aunps > 0 else 0.0
    normalized_efficiency = (total_pairs * 2) / valid_aunps_count if valid_aunps_count > 0 else 0.0
    
    # Calculate average membrane distances for different AuNP categories
    postsynaptic_tree = KDTree(postsynaptic_data)
    
    # Get membrane distances for paired AuNPs
    paired_aunp_distances = []
    for pair in optimal_solution['pairs']:
        for aunp_idx in pair:
            coord = aunp_coordinates[aunp_idx]
            membrane_distance = postsynaptic_tree.query(coord)[0]
            paired_aunp_distances.append(membrane_distance)
    
    # Get membrane distances for filtered out AuNPs
    filtered_out_aunp_distances = []
    used_aunps_in_pairs = set()
    for pair in optimal_solution['pairs']:
        used_aunps_in_pairs.add(pair[0])
        used_aunps_in_pairs.add(pair[1])
    
    valid_aunps_in_pairs = set()
    for pair in optimal_solution.get('valid_pairs', []):
        valid_aunps_in_pairs.add(pair[0])
        valid_aunps_in_pairs.add(pair[1])
    
    filtered_out_aunps = valid_aunps_in_pairs - used_aunps_in_pairs
    for aunp_idx in filtered_out_aunps:
        coord = aunp_coordinates[aunp_idx]
        membrane_distance = postsynaptic_tree.query(coord)[0]
        filtered_out_aunp_distances.append(membrane_distance)
    
    # Calculate averages
    avg_paired_membrane_distance = np.mean(paired_aunp_distances) if paired_aunp_distances else 0.0
    avg_filtered_out_membrane_distance = np.mean(filtered_out_aunp_distances) if filtered_out_aunp_distances else 0.0
    
    # Get top 3 solutions statistics (only for greedy method)
    if method == "greedy":
        top_3_solutions = optimal_solution.get('top_3_solutions', [])
        solution_1_pairs = top_3_solutions[0]['total_pairs'] if len(top_3_solutions) > 0 else 0
        solution_2_pairs = top_3_solutions[1]['total_pairs'] if len(top_3_solutions) > 1 else 0
        solution_3_pairs = top_3_solutions[2]['total_pairs'] if len(top_3_solutions) > 2 else 0
        solution_1_clashes = top_3_solutions[0]['steric_clashes'] if len(top_3_solutions) > 0 else 0
        solution_2_clashes = top_3_solutions[1]['steric_clashes'] if len(top_3_solutions) > 1 else 0
        solution_3_clashes = top_3_solutions[2]['steric_clashes'] if len(top_3_solutions) > 2 else 0
    else:
        # For ILP, set default values
        top_3_solutions = []
        solution_1_pairs = optimal_solution['total_pairs']
        solution_2_pairs = 0
        solution_3_pairs = 0
        solution_1_clashes = optimal_solution['steric_clashes']
        solution_2_clashes = 0
        solution_3_clashes = 0
    
    optimization_stats = {
        'total_aunps': total_aunps,
        'valid_aunps_after_cutoffs': valid_aunps_count,
        'total_pairs_found': total_pairs,
        'unpaired_aunps': len(optimal_solution['unpaired_aunps']),
        'steric_clashes': optimal_solution['steric_clashes'],
        'overall_pairing_efficiency': overall_efficiency,
        'normalized_pairing_efficiency': normalized_efficiency,
        'avg_paired_membrane_distance': avg_paired_membrane_distance,
        'avg_filtered_out_membrane_distance': avg_filtered_out_membrane_distance,
        'solution_1_pairs': solution_1_pairs,
        'solution_1_clashes': solution_1_clashes,
        'solution_2_pairs': solution_2_pairs,
        'solution_2_clashes': solution_2_clashes,
        'solution_3_pairs': solution_3_pairs,
        'solution_3_clashes': solution_3_clashes,
        'inter_aunp_distance': inter_aunp_distance,
        'aunp_membrane_distance': aunp_membrane_distance,
        'ampa_steric_radius': ampa_steric_radius
    }
    
    stats_df = pd.DataFrame([optimization_stats])
    stats_df.to_csv(output_path / f"{output_filename}_optimization_stats.csv", index=False)
    print(f"Saved optimization statistics to {output_path / f'{output_filename}_optimization_stats.csv'}")
    
    # Print membrane distance analysis
    print(f"\nMembrane distance analysis:")
    print(f"  Paired AuNPs: {avg_paired_membrane_distance:.2f} nm (n={len(paired_aunp_distances)})")
    print(f"  Filtered out AuNPs: {avg_filtered_out_membrane_distance:.2f} nm (n={len(filtered_out_aunp_distances)})")
    if avg_paired_membrane_distance > 0 and avg_filtered_out_membrane_distance > 0:
        difference = avg_paired_membrane_distance - avg_filtered_out_membrane_distance
        print(f"  Difference: {difference:.2f} nm")
    
    # Print top 3 solutions analysis (only for greedy method)
    if method == "greedy":
        print(f"\nTop 3 solutions analysis:")
        for i, solution in enumerate(top_3_solutions[:3]):
            efficiency = (solution['total_pairs'] * 2) / total_aunps if total_aunps > 0 else 0.0
            print(f"  Solution {i+1}: {solution['total_pairs']} pairs, {solution['steric_clashes']} clashes, {efficiency:.1%} efficiency")
        
        if len(top_3_solutions) >= 2:
            solution_diversity = solution_1_pairs - solution_3_pairs
            print(f"  Solution diversity: {solution_diversity} pairs difference between best and 3rd best")
    
    return {
        "status": "success",
        "pairs_found": optimal_solution['total_pairs'],
        "unpaired_aunps": len(optimal_solution['unpaired_aunps']),
        "steric_clashes": optimal_solution['steric_clashes'],
        "overall_pairing_efficiency": optimization_stats['overall_pairing_efficiency'],
        "normalized_pairing_efficiency": optimization_stats['normalized_pairing_efficiency'],
        "star_file": str(output_path / f"{output_filename}.star"),
        "paired_aunps_file": str(output_path / f"{output_filename}_paired_aunps.star"),
        "unpaired_file": str(output_path / f"{output_filename}_unpaired_aunps.star"),
        "filtered_out_file": str(output_path / f"{output_filename}_filtered_out_aunps.star"),
        "summary_file": str(output_path / f"{output_filename}_summary.csv"),
        "stats_file": str(output_path / f"{output_filename}_optimization_stats.csv"),
        "particles_data": results_relion,
        "aunps_data": aunps_used_data,
        "optimization_stats": optimization_stats
    }


# Import trimesh for GLB loading
import trimesh


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
    
    print(f"    Loading GLB file: {postsynaptic_glb_path.name}")
    
    # Load the GLB file using trimesh
    loaded = trimesh.load(str(postsynaptic_glb_path))
    print(f"    📦 GLB file loaded successfully")
    
    # Handle both Mesh and Scene objects
    if hasattr(loaded, 'vertices'):
        # It's a Mesh object
        print(f"    🔍 Processing single mesh...")
        mesh = loaded
        vertices = mesh.vertices
        print(f"    📊 Found {len(vertices)} vertices in single mesh")
    else:
        # It's a Scene object - combine all meshes
        print(f"    🔍 Processing scene with {len(loaded.geometry)} meshes...")
        vertices_list = []
        for i, (name, mesh) in enumerate(loaded.geometry.items(), 1):
            if hasattr(mesh, 'vertices'):
                vertices_list.append(mesh.vertices)
                print(f"      Mesh {i}/{len(loaded.geometry)}: {name} ({len(mesh.vertices)} vertices)")
            else:
                print(f"      ⚠️  Mesh {i}/{len(loaded.geometry)}: {name} (no vertices)")
        
        if not vertices_list:
            raise ValueError("No valid meshes found in the GLB file")
        
        # Combine all vertices
        print(f"    🔗 Combining {len(vertices_list)} meshes...")
        vertices = np.vstack(vertices_list)
        print(f"    📊 Combined total: {len(vertices)} vertices")
    
    # Get vertex coordinates and transform them
    # The transformation from findingampa: [0,2,1] * [10,-10,10]
    print(f"    🔄 Applying coordinate transformation...")
    vertices_transformed = vertices[:, [0, 2, 1]] * np.array([10, -10, 10])
    print(f"    ✅ Transformation complete")
    
    return vertices_transformed


def run_ampa_poses_analysis_optimized(tomo_path, output_dir, aunp_active_zones=None,
                                     inter_aunp_distance=(6, 12), 
                                     aunp_membrane_distance=(17, 23),
                                     ampa_steric_radius=5.0,
                                     method="greedy"):
    """
    Run optimized AMPA poses analysis for a tomogram.
    
    Args:
        tomo_path: Path to tomogram directory
        output_dir: Directory to save results
        aunp_active_zones: List of active zone indices to analyze (None for all)
        inter_aunp_distance: Tuple of (min, max) distance between AuNPs in nm
        aunp_membrane_distance: Tuple of (min, max) distance from AuNP to membrane in nm
        ampa_steric_radius: Minimum distance between AMPA positions in nm
        method: Optimization method ("greedy" or "ilp")
        
    Returns:
        Dictionary with analysis results
    """
    tomo_path = Path(tomo_path)
    tomo_name = tomo_path.name
    
    print(f"Running optimized AMPA poses analysis for {tomo_name}")
    
    # Load AuNP data (reuse original function logic)
    print("📁 Loading AuNP data...")
    aunps_dir = tomo_path / "best_alignment" / "aunps"
    
    if aunp_active_zones is None:
        # Load all AuNPs
        aunp_file = aunps_dir / "aunp_tm_BP_active_zone_all.star"
        if not aunp_file.exists():
            raise FileNotFoundError(f"AuNP file not found: {aunp_file}")
        print(f"  Loading from: {aunp_file.name}")
        aunp_data = starfile.read(aunp_file)
        print(f"  ✅ Loaded {len(aunp_data)} AuNPs from all active zones")
    else:
        # Load specific active zones
        print(f"  Loading active zones: {aunp_active_zones}")
        aunp_files = []
        for az_id in aunp_active_zones:
            az_file = aunps_dir / f"aunp_tm_BP_active_zone_{az_id}.star"
            if az_file.exists():
                aunp_files.append(az_file)
                print(f"    Found: {az_file.name}")
            else:
                print(f"    ⚠️  Missing: {az_file.name}")
        
        if not aunp_files:
            raise FileNotFoundError(f"No AuNP files found for active zones: {aunp_active_zones}")
        
        # Load and combine AuNP data
        print(f"  Loading {len(aunp_files)} AuNP files...")
        aunp_data_list = []
        for i, aunp_file in enumerate(aunp_files, 1):
            print(f"    Loading file {i}/{len(aunp_files)}: {aunp_file.name}")
            aunp_data = starfile.read(aunp_file)
            aunp_data_list.append(aunp_data)
        
        aunp_data = pd.concat(aunp_data_list, ignore_index=True)
        print(f"  ✅ Combined {len(aunp_data)} AuNPs from {len(aunp_files)} files")
    
    aunp_coordinates = aunp_data[["faCoordinateX", "faCoordinateY", "faCoordinateZ"]].values
    print(f"  📊 Extracted {len(aunp_coordinates)} AuNP coordinates")
    
    # Load postsynaptic membrane data
    print("🧠 Loading postsynaptic membrane data...")
    postsynaptic_data = load_postsynaptic_coordinates(tomo_path)
    print(f"  ✅ Loaded {len(postsynaptic_data)} membrane vertices")
    
    # Generate output filename with parameters
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
    
    output_filename = f"{tomo_name}_ampa_poses_{method}_{aunp_str}_{membrane_str}_steric{ampa_steric_radius}nm"
    
    # Create method-specific output directory
    # Check if the output directory already contains the method name
    output_dir_str = str(output_dir)
    if method == "ilp":
        if "ilp" in output_dir_str:
            method_output_dir = Path(output_dir)
            print(f"📁 Using ILP method - output directory: {method_output_dir}")
        else:
            method_output_dir = Path(output_dir) / "ilp"
            print(f"📁 Using ILP method - output directory: {method_output_dir}")
    else:
        if "greedy" in output_dir_str:
            method_output_dir = Path(output_dir)
            print(f"📁 Using Greedy method - output directory: {method_output_dir}")
        else:
            method_output_dir = Path(output_dir) / "greedy"
            print(f"📁 Using Greedy method - output directory: {method_output_dir}")
    
    print(f"🚀 Starting optimization with {len(aunp_coordinates)} AuNPs and {len(postsynaptic_data)} membrane vertices")
    print(f"📋 Parameters: AuNP distance {inter_aunp_distance}, Membrane distance {aunp_membrane_distance}, Steric radius {ampa_steric_radius}nm")
    print()
    
    # Run optimized AMPA poses estimation
    results = estimate_ampa_poses_optimized(
        tomo_name,
        aunp_coordinates,
        postsynaptic_data,
        method_output_dir,
        output_filename,
        inter_aunp_distance,
        aunp_membrane_distance,
        ampa_steric_radius,
        method
    )
    
    return results
