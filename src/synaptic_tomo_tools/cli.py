import argparse
import pandas as pd
from pathlib import Path
import sys
import os
import shutil
from .cleft import define_cleft, calculate_cleft_width, build_cleft_per_zone_rows, upsert_cleft_per_zone_csv
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

def parse_cleft_ids(value):
    """Parse CSV/CLI ``cleft_IDs`` to a list of ints, or None for all clefts."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return None
        out = []
        for x in value:
            try:
                out.append(int(float(x)))
            except (TypeError, ValueError):
                continue
        return out or None
    s = str(value).strip()
    if s == "" or s.lower() == "nan":
        return None
    # Allow comma- and/or whitespace-separated values
    parts = s.replace(",", " ").split()
    out = []
    for x in parts:
        x = x.strip()
        if not x:
            continue
        try:
            out.append(int(float(x)))
        except ValueError:
            continue
    return out or None


def load_tomograms(csv_path, analysis_type=None, set_name=None):
    """
    Load tomogram paths from CSV.

    ``analysis_type`` is kept for call-site compatibility but no longer filters rows;
    every CSV row is included (optionally filtered by ``set_name``). Analyses to run
    are chosen by CLI/GUI (``--analysis`` / selected tabs), not by per-row flags.

    Returns a list of ``(path, set_name, cleft_ids, alignment_dir)`` tuples.
    ``cleft_ids`` is the raw CSV ``cleft_IDs`` cell (empty string if absent).
    """
    df = pd.read_csv(csv_path)

    for required in ("tomoname", "set", "alignment_dir"):
        if required not in df.columns:
            raise ValueError(
                f"Column '{required}' not found in {csv_path}."
            )

    filtered = df
    if set_name:
        filtered = filtered[filtered["set"] == set_name]

    assert isinstance(filtered, pd.DataFrame)

    paths = []
    for _, row in filtered.iterrows():
        row: pd.Series  # type hint for linter
        row_set = str(row["set"])
        if row_set not in SET_ROOTS:
            SET_ROOTS[row_set] = Path(TOMO_ROOT_BASE) / row_set / "TOP_TOMOS"
        root = SET_ROOTS[row_set]
        full_path = root / row["tomoname"]
        alignment_dir = str(row["alignment_dir"]).strip()
        if alignment_dir == "" or alignment_dir.lower() == "nan":
            raise ValueError(
                f"Missing alignment_dir for tomogram '{row['tomoname']}' in {csv_path}. "
                "Please set alignment_dir explicitly for every row."
            )
        if "cleft_IDs" in row.index:
            cleft_ids = row.get("cleft_IDs", "")
            if pd.isna(cleft_ids):
                cleft_ids = ""
        else:
            cleft_ids = ""
        paths.append((full_path, row_set, cleft_ids, alignment_dir))

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

def run_cleft(tomo_paths, results_manager, rerun=False, print_ascii=True, az_distance_min=None, az_distance_max=None,
                   aunp_pick_star_pattern=None):
    if print_ascii:
        print_synapse_ascii_art()
    for i, (tomo, set_name, cleft_ids, alignment_dir) in enumerate(tomo_paths):
        tomogram_name = Path(tomo).name
        analysis_name = f"{tomogram_name}__{alignment_dir}"
        print(f"\n{'='*33} TOMOGRAM {i+1}/{len(tomo_paths)} {'='*33}")
        print(f"Analyzing: {tomogram_name}")
        print("="*80)
        
        # Check if analysis already completed successfully
        existing_results = results_manager.get_tomogram_results(analysis_name, 'cleft')
        has_completed = (existing_results and 
                        'results' in existing_results and 
                        'cleft' in existing_results['results'] and
                        existing_results['results']['cleft'].get('status') == 'completed')
        
        if has_completed and not rerun:
            print(f"Skipping synaptic cleft analysis for {analysis_name} (already completed successfully)")
            continue
        
        print(f"Running synaptic cleft analysis on {analysis_name}")
        try:
            az_indices = parse_cleft_ids(cleft_ids)
            
            # Use custom distance range if provided, otherwise use defaults
            distance_range = None
            if az_distance_min is not None or az_distance_max is not None:
                min_dist = az_distance_min if az_distance_min is not None else 10.0
                max_dist = az_distance_max if az_distance_max is not None else 40.0
                distance_range = (min_dist, max_dist)
            
            az_results = define_cleft(
                tomo,
                cleft_indices=az_indices,
                distance_range=distance_range,
                alignment_dir=alignment_dir,
                aunp_pick_star_pattern=aunp_pick_star_pattern,
            )
            cleft_results = calculate_cleft_width(tomo, cleft_indices=az_indices, set_name=set_name, alignment_dir=alignment_dir)
            combined_results = {
                'cleft': az_results,
                'cleft_width': cleft_results
            }
            results_manager.store_tomogram_results(
                analysis_name, 'cleft', combined_results, overwrite=rerun, set_name=set_name, alignment_dir=alignment_dir
            )
            per_zone_rows = build_cleft_per_zone_rows(
                tomogram_name=tomogram_name,
                set_name=set_name or "",
                alignment_dir=alignment_dir,
                az_results=az_results,
                cleft_results=cleft_results,
            )
            if per_zone_rows:
                upsert_cleft_per_zone_csv(
                    per_zone_rows,
                    tomogram_name=tomogram_name,
                    alignment_dir=alignment_dir,
                    results_dir=str(results_manager.results_dir),
                )
        except Exception as e:
            print(f"Error in synaptic cleft analysis for {analysis_name}: {e}")
            # Store error results so we know this analysis failed
            error_results = {
                'cleft': {
                    'status': 'error',
                    'error': str(e)
                },
                'cleft_width': {
                    'status': 'error',
                    'error': str(e)
                }
            }
            results_manager.store_tomogram_results(
                analysis_name, 'cleft', error_results, overwrite=True, set_name=set_name, alignment_dir=alignment_dir
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
    for i, (tomo, set_name, cleft_ids, alignment_dir) in enumerate(tomo_paths):
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
            # Note: Synaptic clefts are already filtered by the synaptic cleft analysis step
            # to only include zones with AuNPs, so vesicle analysis will automatically
            # use only the relevant synaptic clefts
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
                            vesicle_distance_threshold=None, fusing_perimeter_threshold=None,
                            generate_combined_pdf=False):
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
    
    for i, (tomo, set_name, cleft_ids, alignment_dir) in enumerate(tomo_paths):
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
            az_indices = parse_cleft_ids(cleft_ids)
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
                tomo, None, cleft_ids, rerun,
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

    # Per-tomogram active zonogram analysis runs in the loop above. Combined PDF
    # summaries are generated once here, only when explicitly requested.
    if not generate_combined_pdf:
        print(
            "\nSkipping combined visualization PDF generation "
            "(disabled by default; enable via --generate-combined-pdf or the GUI toggle)."
        )
    else:
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

# Canonical order of pipeline steps for `--analysis all`.
PIPELINE_STEPS = ("cleft", "vesicles", "aunps", "visualizations")
# Steps that can be cleared for CSV tomograms (pipeline + pose prediction).
CLEARABLE_ANALYSIS_STEPS = PIPELINE_STEPS + ("poses",)


def run_all_analyses(tomo_paths, results_manager, rerun=False, csv_path=None, 
                     az_distance_min=None, az_distance_max=None, vesicle_distance_threshold=None,
                     dbscan_eps=None, dbscan_min_samples=None, sphere_size=None, sphere_color=None,
                     aunp_distance_min=None, aunp_distance_max=None,
                     aunp_distance_cutoff_direction=None, aunp_distance_cutoff_value=None,
                     cylinder_radius=None, receptor_crosssection=None, aunps_per_receptor=None,
                     vertex_sampling_step=None, synaptic_designation_cutoff=None,
                     min_cluster_size=None, fusion_point_threshold=None,
                     fusing_perimeter_threshold=None,
                     aunp_pick_star_pattern=None,
                     use_monomer_dimer_aunp_labeling=False,
                     run_fusion_point_aunp_analyses=False,
                     run_aunp_vs_az_center_ripley=False,
                     run_aunp_monomer_dimer_ripley=False,
                     monomer_star_pattern=None,
                     dimer_star_pattern=None,
                     monomer_dimer_ripley_n_perm=None,
                     steps=None,
                     generate_combined_pdf=False):
    """Run selected pipeline steps in canonical order: cleft, vesicles, aunps, visualizations.

    steps: iterable of step names to run (subset of PIPELINE_STEPS). None = all steps.
    """
    if steps is None:
        steps_to_run = list(PIPELINE_STEPS)
    else:
        requested = {str(s).strip().lower() for s in steps}
        steps_to_run = [s for s in PIPELINE_STEPS if s in requested]
    if not steps_to_run:
        print("No valid pipeline steps selected; nothing to run.")
        return

    print_synapse_ascii_art()
    print("="*80)
    print("RUNNING PIPELINE STEPS")
    print("="*80)
    print("Selected steps (canonical order): " + " → ".join(steps_to_run))
    print("="*80)

    if "cleft" in steps_to_run:
        print("\n" + "="*80)
        print("STEP: ACTIVE ZONE ANALYSIS")
        print("="*80)
        run_cleft(tomo_paths, results_manager, rerun, print_ascii=False, 
                       az_distance_min=az_distance_min, az_distance_max=az_distance_max,
                       aunp_pick_star_pattern=aunp_pick_star_pattern)
    
    if "vesicles" in steps_to_run:
        print("\n" + "="*80)
        print("STEP: VESICLE ANALYSIS")
        print("="*80)
        run_vesicles(
            tomo_paths,
            results_manager,
            rerun,
            print_ascii=False,
            vesicle_distance_threshold=vesicle_distance_threshold,
            fusing_perimeter_threshold=fusing_perimeter_threshold,
        )
    
    if "aunps" in steps_to_run:
        print("\n" + "="*80)
        print("STEP: AUNP ANALYSIS")
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
                  aunp_pick_star_pattern=aunp_pick_star_pattern,
                  use_monomer_dimer_aunp_labeling=use_monomer_dimer_aunp_labeling,
                  run_fusion_point_aunp_analyses=run_fusion_point_aunp_analyses,
                  run_aunp_vs_az_center_ripley=run_aunp_vs_az_center_ripley,
                  run_aunp_monomer_dimer_ripley=run_aunp_monomer_dimer_ripley,
                  monomer_star_pattern=monomer_star_pattern,
                  dimer_star_pattern=dimer_star_pattern,
                  monomer_dimer_ripley_n_perm=monomer_dimer_ripley_n_perm)
    
    if "visualizations" in steps_to_run:
        print("\n" + "="*80)
        print("STEP: VISUALIZATION GENERATION")
        print("="*80)
        generate_visualizations(tomo_paths, results_manager, rerun, print_ascii=False, csv_path=csv_path,
                                sphere_size=sphere_size, sphere_color=sphere_color,
                                aunp_distance_min=aunp_distance_min, aunp_distance_max=aunp_distance_max,
                                aunp_distance_cutoff_direction=aunp_distance_cutoff_direction,
                                aunp_distance_cutoff_value=aunp_distance_cutoff_value,
                                vesicle_distance_threshold=vesicle_distance_threshold,
                                fusing_perimeter_threshold=fusing_perimeter_threshold,
                                generate_combined_pdf=generate_combined_pdf)
    
    print("\n" + "="*80)
    print("SELECTED ANALYSES COMPLETED!")
    print("="*80)

def run_aunps(tomo_paths, results_manager, rerun=False, print_ascii=True, 
              vesicle_distance_threshold=None, dbscan_eps=None, dbscan_min_samples=None,
              cylinder_radius=None, receptor_crosssection=None, aunps_per_receptor=None,
              vertex_sampling_step=None, synaptic_designation_cutoff=None,
              min_cluster_size=None, fusion_point_threshold=None,
              fusing_perimeter_threshold=None,
              aunp_pick_star_pattern=None,
              use_monomer_dimer_aunp_labeling=False,
              run_fusion_point_aunp_analyses=False,
              run_aunp_vs_az_center_ripley=False,
              run_aunp_monomer_dimer_ripley=False,
              monomer_star_pattern=None,
              dimer_star_pattern=None,
              monomer_dimer_ripley_n_perm=None):
    if print_ascii:
        print_synapse_ascii_art()
    for i, (tomo, set_name, cleft_ids, alignment_dir) in enumerate(tomo_paths):
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
            az_indices = parse_cleft_ids(cleft_ids)
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
                                         aunp_pick_star_pattern=aunp_pick_star_pattern,
                                         use_monomer_dimer_aunp_labeling=use_monomer_dimer_aunp_labeling,
                                         run_fusion_point_aunp_analyses=run_fusion_point_aunp_analyses,
                                         run_aunp_vs_az_center_ripley=run_aunp_vs_az_center_ripley,
                                         run_aunp_monomer_dimer_ripley=run_aunp_monomer_dimer_ripley,
                                         monomer_star_pattern=monomer_star_pattern,
                                         dimer_star_pattern=dimer_star_pattern,
                                         monomer_dimer_ripley_n_perm=monomer_dimer_ripley_n_perm)
            
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
    if run_fusion_point_aunp_analyses:
        print("AGGREGATING FUSION-POINT/AUNP POOLED RESULTS")
        print("=" * 60)
        try:
            from .fusion_point_aunp_position_distance_and_Ripleys_analyses import (
                plot_pooled_fusion_point_aunp_ripley_bidirectional_visualizations,
                plot_pooled_fusion_point_aunp_ripley_g12_visualizations,
                plot_pooled_fusion_point_aunp_ripley_l12_visualizations,
                write_pooled_fusion_point_aunp_distance_column_csvs,
            )

            roots = [Path(tomo) for tomo, _, _, _ in tomo_paths]
            write_pooled_fusion_point_aunp_distance_column_csvs(roots)
            plot_pooled_fusion_point_aunp_ripley_l12_visualizations()
            plot_pooled_fusion_point_aunp_ripley_g12_visualizations()
            plot_pooled_fusion_point_aunp_ripley_bidirectional_visualizations()
        except Exception as e:
            print(f"Warning: Could not write pooled fusion-point/AuNP outputs: {e}")

def _filter_pooled_results_csvs_for_tomograms(
    step_dir: Path,
    tomogram_alignment_pairs: set[tuple[str, str]],
    tomogram_names: set[str],
) -> int:
    """Remove rows matching CSV tomograms from pooled CSVs under ``results/{step}/``.

    Returns the number of CSV files modified.
    """
    if not step_dir.is_dir():
        return 0
    modified = 0
    for csv_path in sorted(step_dir.rglob("*.csv")):
        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            print(f"  Skipping pooled CSV {csv_path}: {exc}")
            continue
        if df.empty:
            continue
        before = len(df)
        if "tomogram_name" in df.columns and "alignment_dir" in df.columns:
            pair_mask = df.apply(
                lambda r: (
                    str(r["tomogram_name"]).strip(),
                    str(r["alignment_dir"]).strip(),
                )
                in tomogram_alignment_pairs,
                axis=1,
            )
            df = df.loc[~pair_mask].copy()
        elif "tomogram_name" in df.columns:
            df = df.loc[
                ~df["tomogram_name"].astype(str).str.strip().isin(tomogram_names)
            ].copy()
        else:
            continue
        if len(df) == before:
            continue
        df.to_csv(csv_path, index=False)
        modified += 1
        print(
            f"  Filtered {before - len(df)} row(s) from {csv_path.relative_to(step_dir.parent)}"
        )
    return modified


def delete_csv_tomogram_results(
    csv_path,
    results_dir="results",
    data_dir="data",
    analysis_type=None,
):
    """Delete results for tomograms specified in the CSV file.

    If ``analysis_type`` is None, delete all analysis steps (JSON entries + whole
    ``STT_results`` trees) for those tomograms.

    If ``analysis_type`` is one of ``CLEARABLE_ANALYSIS_STEPS`` (e.g. ``aunps`` or
    ``poses``), delete only that step: JSON step key when present,
    ``STT_results/{analysis_type}/``, matching pooled CSV rows under
    ``results/{analysis_type}/``, and for visualizations also
    ``results/visualizations/{tomo}/{alignment}/``. For ``poses``, also removes
    combined ``results/poses/all_ampa_poses*`` aggregate STAR files (rebuilt on
    the next Pose Prediction run).
    """
    step = None if analysis_type is None else str(analysis_type).strip().lower()
    if step is not None and step not in CLEARABLE_ANALYSIS_STEPS:
        raise ValueError(
            f"Unknown analysis step {analysis_type!r}; "
            f"expected one of {', '.join(CLEARABLE_ANALYSIS_STEPS)}"
        )

    scope = f"step '{step}'" if step else "all steps"
    print(f"Deleting {scope} results for tomograms specified in {csv_path}")

    try:
        df = pd.read_csv(csv_path)
        for required in ("tomoname", "alignment_dir"):
            if required not in df.columns:
                raise ValueError(
                    f"Column '{required}' not found in {csv_path}. "
                    "Cannot resolve results keys (tomogram__alignment_dir)."
                )
        csv_tomograms = set(df["tomoname"].astype(str).str.strip().tolist())
        print(f"Found {len(csv_tomograms)} distinct tomogram names in CSV")
    except Exception as e:
        print(f"Error reading CSV file {csv_path}: {e}")
        return

    tomogram_alignment_pairs: set[tuple[str, str]] = set()
    results_keys: set[str] = set()
    for _, row in df.iterrows():
        tomo = str(row["tomoname"]).strip()
        alignment_dir = str(row["alignment_dir"]).strip()
        if alignment_dir == "" or alignment_dir.lower() == "nan":
            raise ValueError(
                f"Missing alignment_dir for tomogram '{tomo}' in {csv_path}."
            )
        tomogram_alignment_pairs.add((tomo, alignment_dir))
        results_keys.add(f"{tomo}__{alignment_dir}")
        results_keys.add(tomo)  # legacy bare keys

    results_path = Path(results_dir)
    deleted_json_count = 0
    # Pose prediction is not stored in analysis_results.json today; skip JSON for poses.
    if step != "poses" and results_path.exists():
        results_manager = ResultsManager(results_dir)
        for key in sorted(results_keys):
            if results_manager.delete_tomogram_results(
                key, analysis_type=step, save=False
            ):
                deleted_json_count += 1
                print(f"  Deleted {scope} JSON results for {key}")
        if deleted_json_count:
            results_manager._save_results()
        print(f"Deleted {deleted_json_count} JSON result entr{'y' if deleted_json_count == 1 else 'ies'}")
    elif step == "poses":
        print("Skipping analysis_results.json (poses are not tracked there)")

    # Prefer CSV set/tomoname/alignment paths when available; fall back to walk.
    deleted_stt_count = 0
    data_path = Path(data_dir)
    has_set = "set" in df.columns

    if has_set:
        for _, row in df.iterrows():
            tomo = str(row["tomoname"]).strip()
            alignment_dir = str(row["alignment_dir"]).strip()
            row_set = str(row["set"]).strip()
            base = Path(TOMO_ROOT_BASE) / row_set / "TOP_TOMOS" / tomo / alignment_dir / "STT_results"
            target = base if step is None else base / step
            if target.exists():
                print(f"  Deleting {target}...")
                shutil.rmtree(target)
                deleted_stt_count += 1
            # If step-only and STT_results is now empty, leave the empty parent
            # (harmless; next run recreates subdirs).
    else:
        for root, dirs, _files in os.walk(data_path):
            for d in dirs:
                if d != "STT_results":
                    continue
                stt_path = Path(root) / d
                path_parts = stt_path.parts
                tomogram_name = None
                for i, part in enumerate(path_parts):
                    if part == "TOP_TOMOS" and i + 1 < len(path_parts):
                        tomogram_name = path_parts[i + 1]
                        break
                if not tomogram_name or tomogram_name not in csv_tomograms:
                    continue
                target = stt_path if step is None else stt_path / step
                if target.exists():
                    print(f"  Deleting {target}...")
                    shutil.rmtree(target)
                    deleted_stt_count += 1

    print(
        f"Deleted STT_results"
        f"{'' if step is None else '/' + step} "
        f"for {deleted_stt_count} path(s)"
    )

    if step is not None and results_path.exists():
        step_dir = results_path / step
        n_csv = _filter_pooled_results_csvs_for_tomograms(
            step_dir, tomogram_alignment_pairs, csv_tomograms
        )
        print(f"Updated {n_csv} pooled CSV file(s) under results/{step}/")

        if step == "visualizations":
            viz_root = results_path / "visualizations"
            deleted_viz = 0
            if viz_root.is_dir():
                for tomo, alignment_dir in tomogram_alignment_pairs:
                    viz_dir = viz_root / tomo / alignment_dir
                    if viz_dir.exists():
                        shutil.rmtree(viz_dir)
                        deleted_viz += 1
                        print(f"  Deleted {viz_dir}")
                    # Remove empty tomogram parent if nothing else remains
                    tomo_parent = viz_root / tomo
                    if tomo_parent.is_dir() and not any(tomo_parent.iterdir()):
                        tomo_parent.rmdir()
            print(f"Deleted {deleted_viz} visualization output dir(s)")

        if step == "poses":
            # Combined aggregate STARs are rebuilt by the next Pose Prediction run.
            poses_dir = results_path / "poses"
            removed_agg = 0
            if poses_dir.is_dir():
                for path in poses_dir.glob("all_ampa_poses*"):
                    if path.is_file():
                        path.unlink()
                        removed_agg += 1
                        print(f"  Deleted aggregate {path}")
            print(f"Deleted {removed_agg} aggregate poses file(s) under results/poses/")

    print(f"Total: cleared {scope} for {len(csv_tomograms)} tomogram name(s) from CSV")


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
    elif analysis_type == 'cleft':
        if 'cleft' in existing_results['results']:
            status = existing_results['results']['cleft'].get('status')
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
    
    analysis_types = ['cleft', 'vesicles', 'aunps']
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
        "--analysis", required=True, choices=["cleft", "vesicles", "aunps", "visualizations", "all"],
        help="Which analysis to run. Use 'all' to run cleft, vesicles, aunps, and visualizations in sequence. Use 'visualizations' to generate images from existing analysis results."
    )
    parser.add_argument(
        "--steps", default=None,
        help=(
            "Only used with --analysis all. Comma-separated subset of pipeline steps to run "
            "(choices: cleft, vesicles, aunps, visualizations). Steps always run in canonical "
            "order regardless of listing order. Default: all four steps."
        ),
    )
    parser.add_argument(
        "--generate-combined-pdf", action="store_true",
        help=(
            "At the end of the visualization step, combine all per-tomogram figures into the "
            "combined visualization/zonogram PDF summaries. Off by default."
        ),
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
        "--delete-analysis-step",
        type=str,
        default=None,
        choices=list(CLEARABLE_ANALYSIS_STEPS),
        help=(
            "With --delete-results, delete only this analysis step "
            f"({', '.join(CLEARABLE_ANALYSIS_STEPS)}) instead of all steps for the CSV tomograms."
        ),
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
        help="Custom minimum distance for synaptic cleft definition (nm). Default: 10.0"
    )
    parser.add_argument(
        "--az-distance-max", type=float, default=None,
        help="Custom maximum distance for synaptic cleft definition (nm). Default: 40.0"
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
        help="Minimum vesicle-segmentation-point to presynaptic synaptic-cleft distance (nm) for classifying fusing vesicles. Default: 1.0"
    )
    parser.add_argument(
        "--aunp-pick-star-pattern", type=str, default=None,
        help=(
            "Per-synaptic-cleft AuNP pick STAR filename pattern; use '*' for the synaptic cleft index "
            "(default: aunp_tm_BP_active_zone_*_manual_refined.star). Ignored when "
            "--use-monomer-dimer-aunp-labeling is set."
        ),
    )
    parser.add_argument(
        "--use-monomer-dimer-aunp-labeling",
        dest="use_monomer_dimer_aunp_labeling",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use separate monomer/dimer AuNP pick STAR files: main AuNP analysis runs twice "
            "(tagged monomer/dimer outputs), and monomer/dimer Ripley options apply. "
            "When off, fusion-point vs AuNP analyses use the general pick STAR as a single pool."
        ),
    )
    parser.add_argument(
        "--fusion-point-aunp-analyses",
        dest="run_fusion_point_aunp_analyses",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Run 3D fusion-point vs AuNP distance and Ripley L₁₂ analyses "
            "(default: disabled). Use --fusion-point-aunp-analyses to enable. "
            "With --use-monomer-dimer-aunp-labeling, uses monomer/dimer STAR patterns; "
            "otherwise uses --aunp-pick-star-pattern as a single AuNP pool."
        ),
    )
    parser.add_argument(
        "--monomer-star-pattern", type=str, default=None,
        help=(
            "Per-synaptic-cleft monomer AuNP STAR filename pattern; use '*' for the synaptic cleft index "
            "(default: aunp_tm_BP_active_zone_*_manual_refined_monomer.star)"
        ),
    )
    parser.add_argument(
        "--dimer-star-pattern", type=str, default=None,
        help=(
            "Per-synaptic-cleft dimer AuNP STAR filename pattern; use '*' for the synaptic cleft index "
            "(default: aunp_tm_BP_active_zone_*_manual_refined_dimer.star)"
        ),
    )
    parser.add_argument(
        "--aunp-vs-az-center-ripley",
        dest="run_aunp_vs_az_center_ripley",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Run 3D Ripley L₁₂ of AuNP positions relative to the synaptic cleft center "
            "(default: disabled). Use --aunp-vs-az-center-ripley to enable."
        ),
    )
    parser.add_argument(
        "--aunp-monomer-dimer-ripley",
        dest="run_aunp_monomer_dimer_ripley",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Run 3D bivariate Ripley L₁₂ of monomer vs dimer AuNP positions with a "
            "label-permutation control (default: disabled). Requires "
            "--use-monomer-dimer-aunp-labeling (uses the monomer/dimer STAR patterns). "
            "Use --aunp-monomer-dimer-ripley to enable."
        ),
    )
    parser.add_argument(
        "--monomer-dimer-ripley-n-perm",
        type=int,
        default=None,
        help=(
            "Label-permutation (and matching greedy-segregation) replicate count for "
            f"monomer vs dimer Ripley L₁₂ (default: {1000}). Segregation always uses "
            "the same count. Only used with --aunp-monomer-dimer-ripley."
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
        delete_csv_tomogram_results(
            args.csv,
            args.results_dir,
            "data",
            analysis_type=args.delete_analysis_step,
        )

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
            # Check for synaptic cleft segmentations only if not running cleft or all
            if args.analysis not in ["cleft", "all"]:
                az_dir = base / "STT_results" / "cleft"
                az_pre = list(az_dir.glob("*_pre_outer.txt"))
                az_post = list(az_dir.glob("*_post_outer.txt"))
                if not az_pre:
                    missing_files.append("synaptic cleft pre files (*_pre_outer.txt)")
                if not az_post:
                    missing_files.append("synaptic cleft post files (*_post_outer.txt)")
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
        print("\nWill run: cleft → vesicles → aunps → visualizations")
        print("Analyses that have already completed successfully will be skipped.")
        print("Use --rerun to force re-run of completed analyses.")
    elif args.analysis in ["cleft", "vesicles", "aunps"]:
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
    use_monomer_dimer_aunp_labeling = args.use_monomer_dimer_aunp_labeling
    run_fusion_point_aunp_analyses = args.run_fusion_point_aunp_analyses
    run_aunp_vs_az_center_ripley = args.run_aunp_vs_az_center_ripley
    run_aunp_monomer_dimer_ripley = args.run_aunp_monomer_dimer_ripley
    monomer_star_pattern = args.monomer_star_pattern
    dimer_star_pattern = args.dimer_star_pattern
    monomer_dimer_ripley_n_perm = args.monomer_dimer_ripley_n_perm
    
    if args.analysis == "cleft":
        run_cleft(tomos, results_manager, rerun=args.rerun, 
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
                  aunp_pick_star_pattern=aunp_pick_star_pattern,
                  use_monomer_dimer_aunp_labeling=use_monomer_dimer_aunp_labeling,
                  run_fusion_point_aunp_analyses=run_fusion_point_aunp_analyses,
                  run_aunp_vs_az_center_ripley=run_aunp_vs_az_center_ripley,
                  run_aunp_monomer_dimer_ripley=run_aunp_monomer_dimer_ripley,
                  monomer_star_pattern=monomer_star_pattern,
                  dimer_star_pattern=dimer_star_pattern,
                  monomer_dimer_ripley_n_perm=monomer_dimer_ripley_n_perm)
    elif args.analysis == "visualizations":
        generate_visualizations(tomos, results_manager, rerun=args.rerun, csv_path=args.csv,
                                sphere_size=sphere_size, sphere_color=sphere_color,
                                aunp_distance_min=aunp_distance_min, aunp_distance_max=aunp_distance_max,
                                aunp_distance_cutoff_direction=aunp_distance_cutoff_direction,
                                aunp_distance_cutoff_value=aunp_distance_cutoff_value,
                                vesicle_distance_threshold=vesicle_distance_threshold,
                                fusing_perimeter_threshold=fusing_perimeter_threshold,
                                generate_combined_pdf=args.generate_combined_pdf)
    elif args.analysis == "all":
        selected_steps = None
        if getattr(args, "steps", None):
            requested = [s.strip().lower() for s in args.steps.split(",") if s.strip()]
            invalid = [s for s in requested if s not in PIPELINE_STEPS]
            if invalid:
                print(f"Warning: ignoring unknown --steps values: {', '.join(invalid)}")
            selected_steps = [s for s in requested if s in PIPELINE_STEPS]
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
                         aunp_pick_star_pattern=aunp_pick_star_pattern,
                         use_monomer_dimer_aunp_labeling=use_monomer_dimer_aunp_labeling,
                         run_fusion_point_aunp_analyses=run_fusion_point_aunp_analyses,
                         run_aunp_vs_az_center_ripley=run_aunp_vs_az_center_ripley,
                         run_aunp_monomer_dimer_ripley=run_aunp_monomer_dimer_ripley,
                         monomer_star_pattern=monomer_star_pattern,
                         dimer_star_pattern=dimer_star_pattern,
                         monomer_dimer_ripley_n_perm=monomer_dimer_ripley_n_perm,
                         steps=selected_steps,
                         generate_combined_pdf=args.generate_combined_pdf)

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
