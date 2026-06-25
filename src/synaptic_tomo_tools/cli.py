import argparse
import pandas as pd
from pathlib import Path
import sys
import os
import shutil
from .activezone import define_active_zone, calculate_cleft_width
from .vesicles import detect_vesicles, measure_distances_to_az
from .aunps import analyze_aunps
from .results_manager import ResultsManager

# Import visualization module
try:
    from .visualization import plot_tomogram_overlays
except ImportError:
    print("Warning: Could not import visualization module. Visualizations will be skipped.")
    plot_tomogram_overlays = None


# Map each tomogram set name to its specific root directory
# Uses TOMO_ROOT_BASE environment variable if set (same approach as GUI)
# Otherwise defaults to "data" directory relative to current working directory
TOMO_ROOT_BASE = os.environ.get("TOMO_ROOT_BASE")
if not TOMO_ROOT_BASE:
    # Default to "data" directory if TOMO_ROOT_BASE is not set
    TOMO_ROOT_BASE = "data"

# SET_ROOTS will be dynamically constructed as needed in load_tomograms
# This allows any set name from the CSV to work without hardcoding
SET_ROOTS = {}

def load_tomograms(csv_path, analysis_type, set_name=None):
    """
    Load filtered tomogram paths based on analysis type and set,
    with support for per-set data root directories.
    Returns a list of (path, set_name, aunp_active_zones, alignment_dir) tuples.
    """
    df = pd.read_csv(csv_path)

    # Handle 'all' and 'visualizations' analysis types - include tomograms that have any analysis enabled
    if analysis_type in ['all', 'visualizations']:
        # Include tomograms that have at least one analysis enabled
        analysis_columns = ['activezone', 'vesicles', 'aunps']
        available_columns = [col for col in analysis_columns if col in df.columns]
        if not available_columns:
            raise ValueError(f"No analysis columns found in {csv_path}.")
        
        # Create a mask for tomograms that have any analysis enabled
        analysis_mask = df[available_columns].any(axis=1)
        filtered = df[analysis_mask]
    else:
        # For individual analysis types, check if the column exists
        if analysis_type not in df.columns:
            raise ValueError(f"Column '{analysis_type}' not found in {csv_path}.")
        filtered = df[df[analysis_type] == True]
    
    if set_name:
        filtered = filtered[filtered["set"] == set_name]

    # Ensure filtered is a DataFrame (for linter and runtime safety)
    assert isinstance(filtered, pd.DataFrame)
    # type: ignore[union-attr]

    # Require explicit alignment_dir in CSV (no fallback).
    if "alignment_dir" not in df.columns:
        raise ValueError(
            f"Column 'alignment_dir' not found in {csv_path}. "
            "Each CSV row must specify alignment_dir (e.g., best_alignment, liza_az0)."
        )

    # Construct full paths based on set-specific root
    # Dynamically construct paths using TOMO_ROOT_BASE (same approach as GUI)
    paths = []
    for _, row in filtered.iterrows():
        row: pd.Series  # type hint for linter
        set_name = str(row["set"])
        # Dynamically construct path if not already in SET_ROOTS
        if set_name not in SET_ROOTS:
            SET_ROOTS[set_name] = Path(TOMO_ROOT_BASE) / set_name / "TOP_TOMOS"
        root = SET_ROOTS[set_name]
        full_path = root / row["tomoname"]
        alignment_dir = str(row["alignment_dir"]).strip()
        if alignment_dir == "" or alignment_dir.lower() == "nan":
            raise ValueError(
                f"Missing alignment_dir for tomogram '{row['tomoname']}' in {csv_path}. "
                "Please set alignment_dir explicitly for every row."
            )
        # Get aunp_active_zones if present, else empty string
        aunp_active_zones = row.get("aunp_active_zones", "") if "aunp_active_zones" in row else ""
        paths.append((full_path, set_name, aunp_active_zones, alignment_dir))

    return paths

def print_synapse_ascii_art():
    synapse_art = r"""
================================================================================
SYNAPTIC TOMO TOOLS
================================================================================

                                    .:::::..             .......                                    
                                .:::.     ..::.      .....     .....                                
                            .:::.     .....  .::    ...            ....                             
                        .:::..       ..    .. .:.  ...                ......                        
                  ..::::..   ..      ..    .  .:.  ..                     ......                    
   :::::::::::::::..       ..   ..      ..     ::  ..                          .........            
    .....                 ..    ..   .....     ::  ..                                 ...........   
                           ......   ..    .   .::  ..                                               
                      ......        ..    .   .:. ...                                               
                     ..    ..    ...  ...     .:.:=--.                                              
                      .    ..  .    ..        ::. ==-.                                              
                        ..    ..    ..  ..... .:..-:..                                              
                                .....  .     ..:..=::.                                              
                            ....       ..   ...::  ..                                               
                          ..    ..       ..    ::  ..                                               
   ::::::::::...          ..    ..    ..       ::  ..                                  ..........   
              .:::::..      ....    .    ..    ::  ..                          .............   ..   
                     .:::..        ..    ..   .:.  ..                     ......                    
                         .:::.       .....    ::.  ...                .....                         
                             .:::            .:.    ...           .....                             
                                 :::..   ..:::       .....    .....                                 
                                     .....                .....                                     
"""
    print(synapse_art)

def run_activezone(tomo_paths, results_manager, rerun=False, print_ascii=True, az_distance_min=None, az_distance_max=None,
                   aunp_pick_star_pattern=None):
    if print_ascii:
        print_synapse_ascii_art()
    for i, (tomo, set_name, aunp_active_zones, alignment_dir) in enumerate(tomo_paths):
        tomogram_name = Path(tomo).name
        analysis_name = f"{tomogram_name}__{alignment_dir}"
        print(f"\n{'='*33} TOMOGRAM {i+1}/{len(tomo_paths)} {'='*33}")
        print(f"Analyzing: {tomogram_name}")
        print("="*80)
        
        # Check if analysis already completed successfully
        existing_results = results_manager.get_tomogram_results(analysis_name, 'activezone')
        has_completed = (existing_results and 
                        'results' in existing_results and 
                        'active_zone' in existing_results['results'] and
                        existing_results['results']['active_zone'].get('status') == 'completed')
        
        if has_completed and not rerun:
            print(f"Skipping active zone analysis for {analysis_name} (already completed successfully)")
            continue
        
        print(f"Running active zone analysis on {analysis_name}")
        try:
            # Parse aunp_active_zones to get active zone indices (same as in run_aunps and run_vesicles)
            az_str = str(aunp_active_zones) if aunp_active_zones is not None else ""
            if az_str.strip() == "" or az_str.lower() == "nan":
                az_indices = None
            else:
                # Handle both integer strings and float strings (e.g., "0.0" -> 0)
                az_indices = []
                for x in az_str.split(","):
                    x = x.strip()
                    if x.isdigit():
                        az_indices.append(int(x))
                    elif x.replace(".", "").isdigit():  # Handle floats like "0.0"
                        az_indices.append(int(float(x)))
                # If parsing resulted in empty list, treat as None (use all active zones)
                if len(az_indices) == 0:
                    az_indices = None
            
            # Use custom distance range if provided, otherwise use defaults
            distance_range = None
            if az_distance_min is not None or az_distance_max is not None:
                min_dist = az_distance_min if az_distance_min is not None else 10.0
                max_dist = az_distance_max if az_distance_max is not None else 40.0
                distance_range = (min_dist, max_dist)
            
            az_results = define_active_zone(
                tomo,
                active_zone_indices=az_indices,
                distance_range=distance_range,
                alignment_dir=alignment_dir,
                aunp_pick_star_pattern=aunp_pick_star_pattern,
            )
            cleft_results = calculate_cleft_width(tomo, active_zone_indices=az_indices, set_name=set_name, alignment_dir=alignment_dir)
            combined_results = {
                'active_zone': az_results,
                'cleft_width': cleft_results
            }
            results_manager.store_tomogram_results(
                analysis_name, 'activezone', combined_results, overwrite=rerun, set_name=set_name, alignment_dir=alignment_dir
            )
        except Exception as e:
            print(f"Error in active zone analysis for {analysis_name}: {e}")
            # Store error results so we know this analysis failed
            error_results = {
                'active_zone': {
                    'status': 'error',
                    'error': str(e)
                },
                'cleft_width': {
                    'status': 'error',
                    'error': str(e)
                }
            }
            results_manager.store_tomogram_results(
                analysis_name, 'activezone', error_results, overwrite=True, set_name=set_name, alignment_dir=alignment_dir
            )

def run_vesicles(
    tomo_paths,
    results_manager,
    rerun=False,
    print_ascii=True,
    vesicle_distance_threshold=None,
    fusing_perimeter_threshold=None,
):
    if print_ascii:
        print_synapse_ascii_art()
    for i, (tomo, set_name, aunp_active_zones, alignment_dir) in enumerate(tomo_paths):
        tomogram_name = Path(tomo).name
        analysis_name = f"{tomogram_name}__{alignment_dir}"
        if i > 0:
            print("\n" + "="*80)
        print(f"\n{'='*33} TOMOGRAM {i+1}/{len(tomo_paths)} {'='*33}")
        print(f"Analyzing: {tomogram_name}")
        print("="*80)
        
        # Check if analysis already completed successfully
        existing_results = results_manager.get_tomogram_results(analysis_name, 'vesicles')
        has_completed = (existing_results and 
                        'results' in existing_results and 
                        'vesicle_detection' in existing_results['results'] and
                        existing_results['results']['vesicle_detection'].get('status') == 'completed')
        
        if has_completed and not rerun:
            print(f"Skipping vesicle analysis for {analysis_name} (already completed successfully)")
            continue
        
        print(f"Running vesicle analysis on {analysis_name}")
        try:
            # Note: Active zones are already filtered by the active zone analysis step
            # to only include zones with AuNPs, so vesicle analysis will automatically
            # use only the relevant active zones
            # Use custom vesicle distance threshold if provided, otherwise use default (20.0)
            threshold = vesicle_distance_threshold if vesicle_distance_threshold is not None else 20.0
            fusing_threshold = fusing_perimeter_threshold if fusing_perimeter_threshold is not None else 1.0
            vesicle_results = detect_vesicles(
                tomo,
                set_name=set_name,
                vesicle_distance_threshold=threshold,
                alignment_dir=alignment_dir,
                fusing_perimeter_threshold=fusing_threshold,
            )
            distance_results = measure_distances_to_az(tomo, alignment_dir=alignment_dir)
            combined_results = {
                'vesicle_detection': vesicle_results,
                'distance_measurements': distance_results
            }
            results_manager.store_tomogram_results(
                analysis_name, 'vesicles', combined_results, overwrite=rerun, set_name=set_name, alignment_dir=alignment_dir
            )
        except Exception as e:
            print(f"Error in vesicle analysis for {analysis_name}: {e}")
            # Store error results so we know this analysis failed
            error_results = {
                'vesicle_detection': {
                    'status': 'error',
                    'error': str(e),
                    'vesicle_count': 0
                },
                'distance_measurements': {
                    'status': 'error',
                    'error': str(e)
                }
            }
            results_manager.store_tomogram_results(
                analysis_name, 'vesicles', error_results, overwrite=True, set_name=set_name, alignment_dir=alignment_dir
            )

def generate_visualizations(tomo_paths, results_manager, rerun=False, print_ascii=True, csv_path=None, 
                            sphere_size=None, sphere_color=None, aunp_distance_min=None, aunp_distance_max=None,
                            aunp_distance_cutoff_direction=None, aunp_distance_cutoff_value=None,
                            vesicle_distance_threshold=None, fusing_perimeter_threshold=None):
    """Generate visualization images for each tomogram after analysis is complete."""
    if print_ascii:
        print_synapse_ascii_art()
    
    if plot_tomogram_overlays is None:
        print("Skipping visualization generation (visualization module not available)")
        return

    from .visualization import run_combined_zonogram_analysis_single_tomogram

    vdist = vesicle_distance_threshold if vesicle_distance_threshold is not None else 20.0
    vfuse = fusing_perimeter_threshold if fusing_perimeter_threshold is not None else 1.0

    print("\nGenerating visualizations...")
    
    # Create combined visualization directory in results (structure: visualizations/{tomogram_name}/aunps_and_vesicles/full/)
    # We'll create tomogram-specific directories as we process each tomogram
    base_viz_dir = Path(results_manager.results_dir) / 'visualizations'
    base_viz_dir.mkdir(parents=True, exist_ok=True)
    
    for i, (tomo, set_name, aunp_active_zones, alignment_dir) in enumerate(tomo_paths):
        tomogram_name = Path(tomo).name
        analysis_name = f"{tomogram_name}__{alignment_dir}"

        # Check if visualization step already completed successfully.
        existing_results = results_manager.get_tomogram_results(analysis_name, 'visualizations')
        has_completed = (
            existing_results and
            'results' in existing_results and
            'visualizations' in existing_results['results'] and
            existing_results['results']['visualizations'].get('status') == 'completed'
        )
        if has_completed and not rerun:
            print(f"Skipping visualization for {analysis_name} (already completed successfully)")
            continue
        
        # Create visualization output directory within the tomogram's results folder
        viz_output_dir = Path(tomo) / alignment_dir / 'STT_results' / 'visualizations'
        viz_output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"[{i+1}/{len(tomo_paths)}] Processing {tomogram_name}...", end=" ", flush=True)
        
        try:
            # Parse aunp_active_zones string to list of ints or None
            az_str = str(aunp_active_zones) if aunp_active_zones is not None else ""
            if az_str.strip() == "" or az_str.lower() == "nan":
                az_indices = None
            else:
                # Handle both integer strings and float strings (e.g., "0.0" -> 0)
                az_indices = []
                for x in az_str.split(","):
                    x = x.strip()
                    if x.isdigit():
                        az_indices.append(int(x))
                    elif x.replace(".", "").isdigit():  # Handle floats like "0.0"
                        az_indices.append(int(float(x)))
            # Per-alignment under results/visualizations (same tomogram + different alignment_dir → separate dirs)
            tomogram_viz_dir = base_viz_dir / tomogram_name / alignment_dir / 'aunps_and_vesicles'
            tomogram_viz_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate the three visualization types
            # 1. In tomogram's own directory
            plot_tomogram_overlays(tomo, viz_output_dir, az_indices, rerun=rerun, alignment_dir=alignment_dir,
                                   vesicle_distance_threshold=vdist,
                                   fusing_perimeter_threshold=vfuse,
                                   sphere_size=sphere_size, sphere_color=sphere_color,
                                   aunp_distance_min=aunp_distance_min, aunp_distance_max=aunp_distance_max,
                                   aunp_distance_cutoff_direction=aunp_distance_cutoff_direction,
                                   aunp_distance_cutoff_value=aunp_distance_cutoff_value)
            
            # 2. In the new organized structure
            plot_tomogram_overlays(tomo, tomogram_viz_dir, az_indices, rerun=rerun, alignment_dir=alignment_dir,
                                   vesicle_distance_threshold=vdist,
                                   fusing_perimeter_threshold=vfuse,
                                   sphere_size=sphere_size, sphere_color=sphere_color,
                                   aunp_distance_min=aunp_distance_min, aunp_distance_max=aunp_distance_max,
                                   aunp_distance_cutoff_direction=aunp_distance_cutoff_direction,
                                   aunp_distance_cutoff_value=aunp_distance_cutoff_value)

            zonogram_result = run_combined_zonogram_analysis_single_tomogram(
                tomo, None, aunp_active_zones, rerun,
                alignment_dir=alignment_dir,
                vesicle_distance_threshold=vdist,
                fusing_perimeter_threshold=vfuse,
                sphere_size=sphere_size, sphere_color=sphere_color,
                aunp_distance_min=aunp_distance_min, aunp_distance_max=aunp_distance_max,
                aunp_distance_cutoff_direction=aunp_distance_cutoff_direction,
                aunp_distance_cutoff_value=aunp_distance_cutoff_value,
            )
            if not zonogram_result.get("success"):
                raise RuntimeError(
                    zonogram_result.get("reason", "Active zonogram analysis failed")
                )

            viz_results = {
                'visualizations': {
                    'status': 'completed',
                }
            }
            results_manager.store_tomogram_results(
                analysis_name,
                'visualizations',
                viz_results,
                overwrite=rerun,
                set_name=set_name,
                alignment_dir=alignment_dir
            )
            
            print("✅")
        except Exception as e:
            print("❌")
            print(f"    Error: {e}")
            error_results = {
                'visualizations': {
                    'status': 'error',
                    'error': str(e)
                }
            }
            results_manager.store_tomogram_results(
                analysis_name,
                'visualizations',
                error_results,
                overwrite=True,
                set_name=set_name,
                alignment_dir=alignment_dir
            )
            continue
    
    print(f"\nAll visualizations saved to:")
    print(f"  Individual tomogram directories: {viz_output_dir}")
    print(f"  Organized results directory: {base_viz_dir}")

    print("\n" + "=" * 60)
    print("AGGREGATING FUSION-POINT VS AUNP DENSITY RESULTS (PER TOMOGRAM)")
    print("=" * 60)
    try:
        from .fusion_point_vs_aunp_density import aggregate_fusion_point_per_tomogram_visualizations

        aggregate_fusion_point_per_tomogram_visualizations(tomo_paths)
    except Exception as e:
        print(f"Warning: Could not aggregate per-tomogram fusion-point vs AuNP density figures: {e}")
    
    # Per-tomogram active zonogram analysis runs in the loop above. PDF summaries are generated once here.
    print("\n" + "="*60)
    print("GENERATING VISUALIZATION PDF SUMMARIES")
    print("="*60)
    try:
        from .visualization import (
            unpack_tomo_csv_row,
            generate_default_visualization_pdf_summary,
            generate_zonogram_pdf_summaries,
        )

        root_dir = None
        data_dir = None
        if tomo_paths:
            first_path, _, _, _ = unpack_tomo_csv_row(tomo_paths[0])
            first_tomo_path = Path(first_path)
            if first_tomo_path.parent.name == "TOP_TOMOS":
                root_dir = str(first_tomo_path.parent.parent.parent)
                data_dir = root_dir

        print("\nGenerating PDF summary...")
        generate_default_visualization_pdf_summary(tomo_paths, csv_path, root_dir)
        print("\nGenerating zonogram PDF summaries...")
        generate_zonogram_pdf_summaries(None, tomo_paths, data_dir)
        print("Visualization PDF summaries completed successfully!")
    except Exception as e:
        print(f"Error generating visualization PDF summaries (tomogram results unchanged): {e}")
        import traceback
        traceback.print_exc()

def run_all_analyses(tomo_paths, results_manager, rerun=False, csv_path=None, 
                     az_distance_min=None, az_distance_max=None, vesicle_distance_threshold=None,
                     dbscan_eps=None, dbscan_min_samples=None, sphere_size=None, sphere_color=None,
                     aunp_distance_min=None, aunp_distance_max=None,
                     aunp_distance_cutoff_direction=None, aunp_distance_cutoff_value=None,
                     cylinder_radius=None, receptor_crosssection=None, aunps_per_receptor=None,
                     vertex_sampling_step=None, synaptic_designation_cutoff=None,
                     min_cluster_size=None, fusion_point_threshold=None,
                     fusing_perimeter_threshold=None,
                     aunp_pick_star_pattern=None):
    """Run all analyses in the correct order: activezone, vesicles, aunps, visualizations."""
    print_synapse_ascii_art()
    print("="*80)
    print("RUNNING ALL ANALYSES")
    print("="*80)
    print("Order: activezone → vesicles → aunps → visualizations")
    print("="*80)
    
    # Step 1: Active Zone Analysis
    print("\n" + "="*80)
    print("STEP 1: ACTIVE ZONE ANALYSIS")
    print("="*80)
    run_activezone(tomo_paths, results_manager, rerun, print_ascii=False, 
                   az_distance_min=az_distance_min, az_distance_max=az_distance_max,
                   aunp_pick_star_pattern=aunp_pick_star_pattern)
    
    # Step 2: Vesicle Analysis
    print("\n" + "="*80)
    print("STEP 2: VESICLE ANALYSIS")
    print("="*80)
    run_vesicles(
        tomo_paths,
        results_manager,
        rerun,
        print_ascii=False,
        vesicle_distance_threshold=vesicle_distance_threshold,
        fusing_perimeter_threshold=fusing_perimeter_threshold,
    )
    
    # Step 3: AuNP Analysis
    print("\n" + "="*80)
    print("STEP 3: AUNP ANALYSIS")
    print("="*80)
    run_aunps(tomo_paths, results_manager, rerun, print_ascii=False, 
              vesicle_distance_threshold=vesicle_distance_threshold,
              dbscan_eps=dbscan_eps, dbscan_min_samples=dbscan_min_samples,
              cylinder_radius=cylinder_radius, receptor_crosssection=receptor_crosssection,
              aunps_per_receptor=aunps_per_receptor,
              vertex_sampling_step=vertex_sampling_step,
              synaptic_designation_cutoff=synaptic_designation_cutoff,
              min_cluster_size=min_cluster_size, fusion_point_threshold=fusion_point_threshold,
              fusing_perimeter_threshold=fusing_perimeter_threshold,
              aunp_pick_star_pattern=aunp_pick_star_pattern)
    
    # Step 4: Visualizations
    print("\n" + "="*80)
    print("STEP 4: VISUALIZATION GENERATION")
    print("="*80)
    generate_visualizations(tomo_paths, results_manager, rerun, print_ascii=False, csv_path=csv_path,
                            sphere_size=sphere_size, sphere_color=sphere_color,
                            aunp_distance_min=aunp_distance_min, aunp_distance_max=aunp_distance_max,
                            aunp_distance_cutoff_direction=aunp_distance_cutoff_direction,
                            aunp_distance_cutoff_value=aunp_distance_cutoff_value,
                            vesicle_distance_threshold=vesicle_distance_threshold,
                            fusing_perimeter_threshold=fusing_perimeter_threshold)
    
    print("\n" + "="*80)
    print("ALL ANALYSES COMPLETED!")
    print("="*80)

def run_aunps(tomo_paths, results_manager, rerun=False, print_ascii=True, 
              vesicle_distance_threshold=None, dbscan_eps=None, dbscan_min_samples=None,
              cylinder_radius=None, receptor_crosssection=None, aunps_per_receptor=None,
              vertex_sampling_step=None, synaptic_designation_cutoff=None,
              min_cluster_size=None, fusion_point_threshold=None,
              fusing_perimeter_threshold=None,
              aunp_pick_star_pattern=None):
    if print_ascii:
        print_synapse_ascii_art()
    for i, (tomo, set_name, aunp_active_zones, alignment_dir) in enumerate(tomo_paths):
        tomogram_name = Path(tomo).name
        analysis_name = f"{tomogram_name}__{alignment_dir}"
        
        # Print separator between tomograms
        if i > 0:
            print("\n" + "="*80)
        print(f"\n{'='*33} TOMOGRAM {i+1}/{len(tomo_paths)} {'='*33}")
        print(f"Analyzing: {tomogram_name}")
        print("="*80)
        
        # Check if analysis already completed successfully
        existing_results = results_manager.get_tomogram_results(analysis_name, 'aunps')
        has_completed = (existing_results and 
                        'results' in existing_results and 
                        'aunp_analysis' in existing_results['results'] and
                        existing_results['results']['aunp_analysis'].get('status') == 'completed')
        
        if has_completed and not rerun:
            print(f"Skipping AuNP analysis for {analysis_name} (already completed successfully)")
            continue
            
        print(f"Running AuNP analysis on {analysis_name}")
        
        try:
            # Parse aunp_active_zones string to list of ints or None
            az_str = str(aunp_active_zones) if aunp_active_zones is not None else ""
            if az_str.strip() == "" or az_str.lower() == "nan":
                az_indices = None
            else:
                # Handle both integer strings and float strings (e.g., "0.0" -> 0)
                az_indices = []
                for x in az_str.split(","):
                    x = x.strip()
                    if x.isdigit():
                        az_indices.append(int(x))
                    elif x.replace(".", "").isdigit():  # Handle floats like "0.0"
                        az_indices.append(int(float(x)))
            # Use custom parameters if provided, otherwise use defaults
            vesicle_threshold = vesicle_distance_threshold if vesicle_distance_threshold is not None else 20.0
            eps = dbscan_eps if dbscan_eps is not None else 16.0
            min_samples = dbscan_min_samples if dbscan_min_samples is not None else 1
            
            # Run analyses and collect results
            cylinder_rad = cylinder_radius if cylinder_radius is not None else 25.0
            receptor_cs = receptor_crosssection if receptor_crosssection is not None else 122.0
            aunps_per_rec = aunps_per_receptor if aunps_per_receptor is not None else 2.0
            vert_step = vertex_sampling_step if vertex_sampling_step is not None else 50
            syn_cutoff = synaptic_designation_cutoff if synaptic_designation_cutoff is not None else 30.0
            min_clust = min_cluster_size if min_cluster_size is not None else 4
            fusion_thresh = fusion_point_threshold if fusion_point_threshold is not None else 20.0
            fusing_thresh = fusing_perimeter_threshold if fusing_perimeter_threshold is not None else 1.0
            aunp_results = analyze_aunps(tomo, az_indices, set_name=set_name, alignment_dir=alignment_dir,
                                         vesicle_distance_threshold=vesicle_threshold,
                                         dbscan_eps=eps, dbscan_min_samples=min_samples,
                                         cylinder_radius=cylinder_rad,
                                         receptor_crosssection=receptor_cs,
                                         aunps_per_receptor=aunps_per_rec,
                                         vertex_sampling_step=vert_step,
                                         synaptic_designation_cutoff=syn_cutoff,
                                         min_cluster_size=min_clust,
                                         fusion_point_threshold=fusion_thresh,
                                         fusing_perimeter_threshold=fusing_thresh,
                                         aunp_pick_star_pattern=aunp_pick_star_pattern)
            
            # Store combined results
            combined_results = {
                'aunp_analysis': aunp_results,
            }
            
            # Auto-overwrite if not using skip_completed (more intuitive behavior)
            results_manager.store_tomogram_results(
                analysis_name, 'aunps', combined_results, overwrite=rerun, set_name=set_name, alignment_dir=alignment_dir
            )
        except Exception as e:
            print(f"Error in AuNP analysis for {analysis_name}: {e}")
            # Store error results so we know this analysis failed
            error_results = {
                'aunp_analysis': {
                    'status': 'error',
                    'error': str(e)
                }
            }
            results_manager.store_tomogram_results(
                analysis_name, 'aunps', error_results, overwrite=True, set_name=set_name, alignment_dir=alignment_dir
            )

    print("\n" + "=" * 60)
    print("AGGREGATING FUSION-POINT VS AUNP DENSITY RESULTS (POOLED)")
    print("=" * 60)
    try:
        from .fusion_point_aunp_position_distance_and_Ripleys_analyses import (
            plot_pooled_fusion_point_aunp_ripley_l12_visualizations,
        )

        plot_pooled_fusion_point_aunp_ripley_l12_visualizations()
    except Exception as e:
        print(f"Warning: Could not write pooled fusion-point/AuNP Ripley L₁₂ figures: {e}")
    try:
        from .fusion_point_vs_aunp_density import aggregate_fusion_point_pooled_visualizations

        aggregate_fusion_point_pooled_visualizations(tomo_paths)
    except Exception as e:
        print(f"Warning: Could not aggregate pooled fusion-point vs AuNP density results: {e}")

def delete_csv_tomogram_results(csv_path, results_dir="results", data_dir="data"):
    """Delete results only for tomograms specified in the CSV file."""
    print(f"Deleting results for tomograms specified in {csv_path}")
    
    # Load CSV to get list of tomograms
    try:
        df = pd.read_csv(csv_path)
        if "alignment_dir" not in df.columns:
            raise ValueError(
                f"Column 'alignment_dir' not found in {csv_path}. "
                "Cannot resolve results keys (tomogram__alignment_dir)."
            )
        csv_tomograms = set(df["tomoname"].astype(str).str.strip().tolist())
        print(f"Found {len(csv_tomograms)} distinct tomogram names in CSV")
    except Exception as e:
        print(f"Error reading CSV file {csv_path}: {e}")
        return
    
    # Delete specific tomogram results from the results directory
    results_path = Path(results_dir)
    if results_path.exists():
        # Load existing results
        results_manager = ResultsManager(results_dir)
        existing_results = results_manager.get_all_results()
        
        # Keys in analysis_results.json are tomogram__alignment_dir (see run_* in this module).
        # Also remove legacy bare tomogram_name keys when that tomogram appears in the CSV.
        keys_to_remove = set()
        for _, row in df.iterrows():
            tomo = str(row["tomoname"]).strip()
            alignment_dir = str(row["alignment_dir"]).strip()
            if alignment_dir == "" or alignment_dir.lower() == "nan":
                raise ValueError(
                    f"Missing alignment_dir for tomogram '{tomo}' in {csv_path}."
                )
            keys_to_remove.add(f"{tomo}__{alignment_dir}")
            keys_to_remove.add(tomo)

        deleted_count = 0
        for key in keys_to_remove:
            if key in existing_results:
                del existing_results[key]
                deleted_count += 1
                print(f"  Deleted results for {key}")
        
        # Save updated results
        results_manager.results = existing_results
        results_manager._save_results()
        print(f"Deleted {deleted_count} results entries from results directory")
    
    # Delete STT_results directories for CSV tomograms only
    data_path = Path(data_dir)
    deleted_stt_count = 0
    
    for root, dirs, files in os.walk(data_path):
        for d in dirs:
            if d == "STT_results":
                stt_path = Path(root) / d
                # Check if this STT_results belongs to a CSV tomogram
                # The path should be: data/set_name/TOP_TOMOS/tomogram_name/best_alignment/STT_results
                path_parts = stt_path.parts
                
                # Find the tomogram name in the path
                tomogram_name = None
                for i, part in enumerate(path_parts):
                    if part == "TOP_TOMOS" and i + 1 < len(path_parts):
                        tomogram_name = path_parts[i + 1]
                        break
                
                if tomogram_name and tomogram_name in csv_tomograms:
                    print(f"  Deleting {stt_path}...")
                    shutil.rmtree(stt_path)
                    deleted_stt_count += 1
    
    print(f"Deleted STT_results directories for {deleted_stt_count} tomograms")
    print(f"Total: Deleted results for {len(csv_tomograms)} tomograms specified in CSV")

def check_analysis_status(results_manager, results_key, analysis_type):
    """Check if an analysis completed successfully, failed, or hasn't been run.

    results_key must match the top-level key in analysis_results.json
    (typically tomogram_name__alignment_dir).
    """
    existing_results = results_manager.get_tomogram_results(results_key, analysis_type)
    
    if not existing_results:
        return 'not_run'
    
    if 'results' not in existing_results:
        return 'not_run'
    
    # Check for error status in the results
    if analysis_type == 'vesicles':
        if 'vesicle_detection' in existing_results['results']:
            status = existing_results['results']['vesicle_detection'].get('status')
            if status == 'error':
                return 'failed'
            elif status == 'completed':
                return 'completed'
    elif analysis_type == 'activezone':
        if 'active_zone' in existing_results['results']:
            status = existing_results['results']['active_zone'].get('status')
            if status == 'error':
                return 'failed'
            elif status == 'completed':
                return 'completed'
    elif analysis_type == 'aunps':
        if 'aunp_analysis' in existing_results['results']:
            status = existing_results['results']['aunp_analysis'].get('status')
            if status == 'error':
                return 'failed'
            elif status == 'completed':
                return 'completed'
    
    # If we have results but no clear status, assume it's completed
    return 'completed'

def print_analysis_summary(results_manager, tomos):
    """Print a summary of analysis status for all tomograms."""
    print("\n" + "="*80)
    print("ANALYSIS STATUS SUMMARY")
    print("="*80)
    
    analysis_types = ['activezone', 'vesicles', 'aunps']
    status_counts = {analysis_type: {'completed': 0, 'failed': 0, 'not_run': 0} for analysis_type in analysis_types}
    
    for tomo_path, _set_name, _, alignment_dir in tomos:
        tomogram_name = Path(tomo_path).name
        results_key = f"{tomogram_name}__{alignment_dir}"
        print(f"\n{tomogram_name} ({alignment_dir}):")
        for analysis_type in analysis_types:
            status = check_analysis_status(results_manager, results_key, analysis_type)
            status_counts[analysis_type][status] += 1
            
            status_symbol = {
                'completed': '✅',
                'failed': '❌', 
                'not_run': '⏳'
            }[status]
            
            print(f"  {status_symbol} {analysis_type}")
    
    print(f"\n{'='*80}")
    print("SUMMARY:")
    for analysis_type in analysis_types:
        counts = status_counts[analysis_type]
        total = sum(counts.values())
        print(f"{analysis_type}: {counts['completed']}/{total} completed, {counts['failed']} failed, {counts['not_run']} not run")
    print("="*80)

def main():
    parser = argparse.ArgumentParser(
        description="Run SynapticTomoTools analysis on selected tomograms."
    )
    parser.add_argument(
        "--analysis", required=True, choices=["activezone", "vesicles", "aunps", "visualizations", "all"],
        help="Which analysis to run. Use 'all' to run activezone, vesicles, aunps, and visualizations in sequence. Use 'visualizations' to generate images from existing analysis results."
    )
    parser.add_argument(
        "--set", default=None,
        help="(Optional) Filter tomograms by experimental set name (e.g., 15F1, unlabeled)."
    )
    parser.add_argument(
        "--csv", default=None,
        help="Path to CSV file listing tomograms and analysis flags. Default is 'data/tomograms.csv', or 'data/tomograms-test.csv' if --test is used."
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Use test tomogram roots and default to 'data/tomograms-test.csv' unless --csv is specified."
    )
    parser.add_argument(
        "--rerun", action="store_true",
        help="Rerun analysis on already completed steps and overwrite existing results."
    )
    parser.add_argument(
        "--results-dir", default="results",
        help="Directory to store analysis results."
    )

    parser.add_argument(
        "--delete-results", action="store_true",
        help="Delete analysis results for tomograms specified in the CSV file (not all results)."
    )
    parser.add_argument(
        "--check-files", action="store_true",
        help="Only check that all expected files for the tomograms listed in the CSV are present in the expected locations."
    )
    parser.add_argument(
        "--generate-pdf-summary", action="store_true",
        help="Generate PDF summary for all tomograms at the end of the analysis pipeline."
    )
    parser.add_argument(
        "--show-status", action="store_true",
        help="Show detailed status of all analyses for the selected tomograms."
    )
    
    # Custom parameter arguments
    parser.add_argument(
        "--az-distance-min", type=float, default=None,
        help="Custom minimum distance for active zone definition (nm). Default: 10.0"
    )
    parser.add_argument(
        "--az-distance-max", type=float, default=None,
        help="Custom maximum distance for active zone definition (nm). Default: 40.0"
    )
    parser.add_argument(
        "--vesicle-distance-threshold", type=float, default=None,
        help="Custom distance threshold for 'close' vesicles (nm). Default: 20.0"
    )
    parser.add_argument(
        "--dbscan-eps", type=float, default=None,
        help="Custom DBSCAN eps parameter for AuNP clustering (nm). Default: 16.0"
    )
    parser.add_argument(
        "--dbscan-min-samples", type=int, default=None,
        help="Custom DBSCAN min_samples parameter for AuNP clustering. Default: 1"
    )
    parser.add_argument(
        "--sphere-size", type=int, default=None,
        help="Custom sphere size for active zonogram overlays. Default: 36"
    )
    parser.add_argument(
        "--sphere-color", type=str, default=None,
        help="Custom sphere color for active zonogram overlays. Default: 'gold'"
    )
    parser.add_argument(
        "--aunp-distance-min", type=float, default=None,
        help="Custom minimum distance for AuNP distance color scale (nm). Default: auto from data"
    )
    parser.add_argument(
        "--aunp-distance-max", type=float, default=None,
        help="Custom maximum distance for AuNP distance color scale (nm). Default: auto from data"
    )
    parser.add_argument(
        "--aunp-distance-cutoff-direction", type=str, default="below",
        choices=["below", "above"],
        help="Direction for AuNP distance cutoff filter. Options: 'below' or 'above'. Default: 'below'"
    )
    parser.add_argument(
        "--aunp-distance-cutoff-value", type=float, default=15.0,
        help="Cutoff value for AuNP distance filter (nm). Default: 15.0"
    )
    parser.add_argument(
        "--cylinder-radius", type=float, default=None,
        help="Sliding cylinder radius for packing density heat map (nm). Default: 25.0"
    )
    parser.add_argument(
        "--receptor-crosssection", type=float, default=None,
        help="Receptor cross-sectional area for packing density (nm²). Default: 122.0"
    )
    parser.add_argument(
        "--aunps-per-receptor", type=float, default=None,
        help="AuNPs per receptor for packing density (2=dimer, 1=monomer). Default: 2.0"
    )
    parser.add_argument(
        "--vertex-sampling-step", type=int, default=None,
        help="Sample every Nth mesh vertex for packing density (1=all, 50=every 50th). Default: 50"
    )
    parser.add_argument(
        "--synaptic-designation-cutoff", type=float, default=None,
        help="Distance cutoff (nm) to postsynaptic active outer membrane for synaptic/extrasynaptic designation. Default: 30.0"
    )
    parser.add_argument(
        "--min-cluster-size", type=int, default=None,
        help="Minimum cluster size retained after DBSCAN (smaller clusters become noise). Default: 4"
    )
    parser.add_argument(
        "--fusion-point-threshold", type=float, default=None,
        help="Distance threshold (nm) for AZ points contributing to fusion point. Default: 20.0"
    )
    parser.add_argument(
        "--fusing-perimeter-threshold", type=float, default=None,
        help="Minimum vesicle-segmentation-point to presynaptic active-zone distance (nm) for classifying fusing vesicles. Default: 1.0"
    )
    parser.add_argument(
        "--aunp-pick-star-pattern", type=str, default=None,
        help=(
            "Per-active-zone AuNP pick STAR filename pattern; use '*' for the active zone index "
            "(default: aunp_tm_BP_active_zone_*_manual_refined.star)"
        ),
    )

    args = parser.parse_args()

    # Handle test mode - set TOMO_ROOT_BASE to repo's data directory
    global TOMO_ROOT_BASE, SET_ROOTS
    if args.test:
        repo_root = Path(__file__).parent.parent.parent.resolve()
        TOMO_ROOT_BASE = str(repo_root / "data")
        SET_ROOTS = {}  # Will be dynamically constructed as needed
        if args.csv is None:
            args.csv = str(repo_root / "data" / "tomograms-test.csv")
    else:
        if args.csv is None:
            args.csv = "data/tomograms.csv"
    
    if args.delete_results:
        delete_csv_tomogram_results(args.csv, args.results_dir, "data")

    if args.check_files:
        tomos = load_tomograms(args.csv, args.analysis, args.set)
        if not tomos:
            print("No matching tomograms found.")
            return
        print(f"Checking required files for {len(tomos)} tomograms...")
        missing = False
        for tomo, set_name, _, alignment_dir in tomos:
            missing_files = []
            base = Path(tomo) / alignment_dir
            # Check for main reconstruction
            rec_file = list(base.glob("*_ddw.mrc"))
            if not rec_file:
                missing_files.append("main reconstruction (*.mrc)")
            # Check for vesicle files
            ves_dir = base / "aunps"
            ves_files = list(ves_dir.glob("synapticvesicles_*.txt"))
            if not ves_files:
                missing_files.append("vesicle files (synapticvesicles_*.txt)")
            # Check for membrane files
            pre_mem = list(ves_dir.glob("presynapticmembranes_*.txt"))
            post_mem = list(ves_dir.glob("postsynapticmembranes_*.txt"))
            if not pre_mem:
                missing_files.append("presynaptic membrane files (presynapticmembranes_*.txt)")
            if not post_mem:
                missing_files.append("postsynaptic membrane files (postsynapticmembranes_*.txt)")
            # Check for active zone segmentations only if not running activezone or all
            if args.analysis not in ["activezone", "all"]:
                az_dir = base / "STT_results" / "activezone"
                az_pre = list(az_dir.glob("*_pre_outer.txt"))
                az_post = list(az_dir.glob("*_post_outer.txt"))
                if not az_pre:
                    missing_files.append("active zone pre files (*_pre_outer.txt)")
                if not az_post:
                    missing_files.append("active zone post files (*_post_outer.txt)")
            # Check for MemBrain segmentation
            membrain_dir = base / "membrain"
            membrain_files = list(membrain_dir.glob("*.mrc"))
            if not membrain_files:
                missing_files.append("MemBrain segmentation (*.mrc)")
            if missing_files:
                missing = True
                print(f"[MISSING] {Path(tomo).name}:")
                for f in missing_files:
                    print(f"  - {f}")
            else:
                print(f"[OK] {Path(tomo).name}: All required files present.")
        if not missing:
            print("All required files are present for all tomograms.")
        return

    # Initialize results manager
    results_manager = ResultsManager(args.results_dir)
    
    # Show completed analyses
    completed = results_manager.list_completed_analyses()
    if completed:
        print("Previously completed analyses:")
        for tomogram, analyses in completed.items():
            print(f"  {tomogram}: {', '.join(analyses)}")
        print()

    tomos = load_tomograms(args.csv, args.analysis, args.set)

    if not tomos:
        print("No matching tomograms found.")
        return

    print(f"Found {len(tomos)} tomograms for analysis '{args.analysis}'")
    
    # Show detailed status if requested
    if args.show_status:
        print_analysis_summary(results_manager, tomos)
        return

    # Show what will be run
    if args.analysis == "all":
        print("\nWill run: activezone → vesicles → aunps → visualizations")
        print("Analyses that have already completed successfully will be skipped.")
        print("Use --rerun to force re-run of completed analyses.")
    elif args.analysis in ["activezone", "vesicles", "aunps"]:
        print(f"\nWill run: {args.analysis} analysis")
        print("Analyses that have already completed successfully will be skipped.")
        print("Use --rerun to force re-run of completed analyses.")
    elif args.analysis == "visualizations":
        print("\nWill run: visualization generation")
        print("Visualizations that have already been generated will be skipped.")
        print("Use --rerun to force re-generation of existing visualizations.")

    # Extract custom parameters from args
    az_distance_min = args.az_distance_min
    az_distance_max = args.az_distance_max
    vesicle_distance_threshold = args.vesicle_distance_threshold
    dbscan_eps = args.dbscan_eps
    dbscan_min_samples = args.dbscan_min_samples
    sphere_size = args.sphere_size
    sphere_color = args.sphere_color
    aunp_distance_min = args.aunp_distance_min
    aunp_distance_max = args.aunp_distance_max
    aunp_distance_cutoff_direction = args.aunp_distance_cutoff_direction
    aunp_distance_cutoff_value = args.aunp_distance_cutoff_value
    cylinder_radius = args.cylinder_radius
    receptor_crosssection = args.receptor_crosssection
    aunps_per_receptor = args.aunps_per_receptor
    vertex_sampling_step = args.vertex_sampling_step
    synaptic_designation_cutoff = args.synaptic_designation_cutoff
    min_cluster_size = args.min_cluster_size
    fusion_point_threshold = args.fusion_point_threshold
    fusing_perimeter_threshold = args.fusing_perimeter_threshold
    aunp_pick_star_pattern = args.aunp_pick_star_pattern
    
    if args.analysis == "activezone":
        run_activezone(tomos, results_manager, rerun=args.rerun, 
                       az_distance_min=az_distance_min, az_distance_max=az_distance_max,
                       aunp_pick_star_pattern=aunp_pick_star_pattern)
    elif args.analysis == "vesicles":
        run_vesicles(
            tomos,
            results_manager,
            rerun=args.rerun,
            vesicle_distance_threshold=vesicle_distance_threshold,
            fusing_perimeter_threshold=fusing_perimeter_threshold,
        )
    elif args.analysis == "aunps":
        run_aunps(tomos, results_manager, rerun=args.rerun, 
                  vesicle_distance_threshold=vesicle_distance_threshold,
                  dbscan_eps=dbscan_eps, dbscan_min_samples=dbscan_min_samples,
                  cylinder_radius=cylinder_radius, receptor_crosssection=receptor_crosssection,
                  aunps_per_receptor=aunps_per_receptor,
                  vertex_sampling_step=vertex_sampling_step,
                  synaptic_designation_cutoff=synaptic_designation_cutoff,
                  min_cluster_size=min_cluster_size, fusion_point_threshold=fusion_point_threshold,
                  fusing_perimeter_threshold=fusing_perimeter_threshold,
                  aunp_pick_star_pattern=aunp_pick_star_pattern)
    elif args.analysis == "visualizations":
        generate_visualizations(tomos, results_manager, rerun=args.rerun, csv_path=args.csv,
                                sphere_size=sphere_size, sphere_color=sphere_color,
                                aunp_distance_min=aunp_distance_min, aunp_distance_max=aunp_distance_max,
                                aunp_distance_cutoff_direction=aunp_distance_cutoff_direction,
                                aunp_distance_cutoff_value=aunp_distance_cutoff_value,
                                vesicle_distance_threshold=vesicle_distance_threshold,
                                fusing_perimeter_threshold=fusing_perimeter_threshold)
    elif args.analysis == "all":
        run_all_analyses(tomos, results_manager, rerun=args.rerun, csv_path=args.csv,
                         az_distance_min=az_distance_min, az_distance_max=az_distance_max,
                         vesicle_distance_threshold=vesicle_distance_threshold,
                         dbscan_eps=dbscan_eps, dbscan_min_samples=dbscan_min_samples,
                         sphere_size=sphere_size, sphere_color=sphere_color,
                         aunp_distance_min=aunp_distance_min, aunp_distance_max=aunp_distance_max,
                         aunp_distance_cutoff_direction=aunp_distance_cutoff_direction,
                         aunp_distance_cutoff_value=aunp_distance_cutoff_value,
                         cylinder_radius=cylinder_radius,
                         receptor_crosssection=receptor_crosssection,
                         aunps_per_receptor=aunps_per_receptor,
                         vertex_sampling_step=vertex_sampling_step,
                         synaptic_designation_cutoff=synaptic_designation_cutoff,
                         min_cluster_size=min_cluster_size, fusion_point_threshold=fusion_point_threshold,
                         fusing_perimeter_threshold=fusing_perimeter_threshold,
                         aunp_pick_star_pattern=aunp_pick_star_pattern)

    # Always export summary CSVs at the end
    print("\nExporting all summary CSVs from stored results...")
    results_manager.export_to_csv()

    # If requested, generate PDF summary at the end
    if args.generate_pdf_summary:
        import subprocess
        print("\nGenerating PDF summary...")
        subprocess.run([
            sys.executable, "scripts/generate_tomogram_summary_pdf.py",
            "--vis-dir", "results/visualizations",
            "--data-dir", "data",
            "--output-dir", "results/visualizations/pdf_summaries"
        ], check=True)

    if args.show_status:
        print_analysis_summary(results_manager, tomos)

if __name__ == "__main__":
    main()
