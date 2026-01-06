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
    Returns a list of (path, set_name) tuples.
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
        # Get aunp_active_zones if present, else empty string
        aunp_active_zones = row.get("aunp_active_zones", "") if "aunp_active_zones" in row else ""
        paths.append((full_path, set_name, aunp_active_zones))

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

def run_activezone(tomo_paths, results_manager, rerun=False, print_ascii=True):
    if print_ascii:
        print_synapse_ascii_art()
    for i, (tomo, set_name) in enumerate(tomo_paths):
        tomogram_name = Path(tomo).name
        print(f"\n{'='*33} TOMOGRAM {i+1}/{len(tomo_paths)} {'='*33}")
        print(f"Analyzing: {tomogram_name}")
        print("="*80)
        
        # Check if analysis already completed successfully
        existing_results = results_manager.get_tomogram_results(tomogram_name, 'activezone')
        has_completed = (existing_results and 
                        'results' in existing_results and 
                        'active_zone' in existing_results['results'] and
                        existing_results['results']['active_zone'].get('status') == 'completed')
        
        if has_completed and not rerun:
            print(f"Skipping active zone analysis for {tomogram_name} (already completed successfully)")
            continue
        
        print(f"Running active zone analysis on {tomogram_name}")
        try:
            az_results = define_active_zone(tomo)
            cleft_results = calculate_cleft_width(tomo)
            combined_results = {
                'active_zone': az_results,
                'cleft_width': cleft_results
            }
            results_manager.store_tomogram_results(tomogram_name, 'activezone', combined_results, overwrite=rerun, set_name=set_name)
        except Exception as e:
            print(f"Error in active zone analysis for {tomogram_name}: {e}")
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
            results_manager.store_tomogram_results(tomogram_name, 'activezone', error_results, overwrite=True, set_name=set_name)

def run_vesicles(tomo_paths, results_manager, rerun=False, print_ascii=True):
    if print_ascii:
        print_synapse_ascii_art()
    for i, (tomo, set_name, aunp_active_zones) in enumerate(tomo_paths):
        tomogram_name = Path(tomo).name
        if i > 0:
            print("\n" + "="*80)
        print(f"\n{'='*33} TOMOGRAM {i+1}/{len(tomo_paths)} {'='*33}")
        print(f"Analyzing: {tomogram_name}")
        print("="*80)
        
        # Check if analysis already completed successfully
        existing_results = results_manager.get_tomogram_results(tomogram_name, 'vesicles')
        has_completed = (existing_results and 
                        'results' in existing_results and 
                        'vesicle_detection' in existing_results['results'] and
                        existing_results['results']['vesicle_detection'].get('status') == 'completed')
        
        if has_completed and not rerun:
            print(f"Skipping vesicle analysis for {tomogram_name} (already completed successfully)")
            continue
        
        print(f"Running vesicle analysis on {tomogram_name}")
        try:
            # Parse aunp_active_zones to get active zone indices (same as in run_aunps)
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
            
            vesicle_results = detect_vesicles(tomo, set_name=set_name, active_zone_indices=az_indices)
            distance_results = measure_distances_to_az(tomo)
            combined_results = {
                'vesicle_detection': vesicle_results,
                'distance_measurements': distance_results
            }
            results_manager.store_tomogram_results(tomogram_name, 'vesicles', combined_results, overwrite=rerun, set_name=set_name)
        except Exception as e:
            print(f"Error in vesicle analysis for {tomogram_name}: {e}")
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
            results_manager.store_tomogram_results(tomogram_name, 'vesicles', error_results, overwrite=True, set_name=set_name)

def generate_visualizations(tomo_paths, results_manager, rerun=False, print_ascii=True, csv_path=None):
    """Generate visualization images for each tomogram after analysis is complete."""
    if print_ascii:
        print_synapse_ascii_art()
    
    if plot_tomogram_overlays is None:
        print("Skipping visualization generation (visualization module not available)")
        return
        
    print("\nGenerating visualizations...")
    
    # Create combined visualization directory in results (structure: visualizations/{tomogram_name}/aunps_and_vesicles/full/)
    # We'll create tomogram-specific directories as we process each tomogram
    base_viz_dir = Path(results_manager.results_dir) / 'visualizations'
    base_viz_dir.mkdir(parents=True, exist_ok=True)
    
    for i, (tomo, set_name, aunp_active_zones) in enumerate(tomo_paths):
        tomogram_name = Path(tomo).name
        
        # Create visualization output directory within the tomogram's results folder
        viz_output_dir = Path(tomo) / 'best_alignment' / 'STT_results' / 'visualizations'
        viz_output_dir.mkdir(parents=True, exist_ok=True)

        # Check if all expected visualization files exist and skip_completed is True
        # Basic visualization files
        basic_files = [
            viz_output_dir / f"{tomogram_name}_vesicles_active_zones.png",
            viz_output_dir / f"{tomogram_name}_vesicles_aunps.png",
            viz_output_dir / f"{tomogram_name}_combined.png",
        ]
        
        # Active zonogram files (check for at least one of each type)
        active_zonogram_files = list(viz_output_dir.glob(f"{tomogram_name}_active_zonogram_*.png"))
        mini_zonogram_files = list(viz_output_dir.glob(f"{tomogram_name}_mini_zonogram_*.png"))
        
        # Check if we have the minimum required files
        basic_files_exist = all(f.exists() for f in basic_files)
        zonogram_files_exist = len(active_zonogram_files) > 0 and len(mini_zonogram_files) > 0
        
        if basic_files_exist and zonogram_files_exist and not rerun:
            print(f"Skipping visualization for {tomogram_name} (already completed)")
            continue
        
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
            # Create tomogram-specific directory structure: visualizations/{tomogram_name}/aunps_and_vesicles/
            tomogram_viz_dir = base_viz_dir / tomogram_name / 'aunps_and_vesicles'
            tomogram_viz_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate the three visualization types
            # 1. In tomogram's own directory
            plot_tomogram_overlays(tomo, viz_output_dir, az_indices, rerun=rerun)
            
            # 2. In the new organized structure
            plot_tomogram_overlays(tomo, tomogram_viz_dir, az_indices, rerun=rerun)
            
            print("✅")
        except Exception as e:
            print("❌")
            print(f"    Error: {e}")
            continue
    
    print(f"\nAll visualizations saved to:")
    print(f"  Individual tomogram directories: {viz_output_dir}")
    print(f"  Organized results directory: {base_viz_dir}")

    
    # Run active zonogram analysis for all tomograms
    print("\n" + "="*60)
    print("RUNNING ACTIVE ZONOGRAM ANALYSIS")
    print("="*60)
    try:
        # Import the active zonogram analysis function using absolute import
        import sys
        from pathlib import Path as PathLib
        
        # Add the src directory to the path if not already there
        src_path = PathLib(__file__).parent.parent
        if str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))
        
        from synaptic_tomo_tools.visualization import run_zonogram_analysis_for_all_tomograms
        
        # Create output directory for active zonogram analysis
        output_dir = Path("results")
        # Extract root directory from tomogram paths for PDF generation
        root_dir = None
        if tomo_paths:
            # Get the root directory from the first tomogram path
            first_tomo_path = Path(tomo_paths[0][0])
            # Go up to find the root (assuming structure: root/set/TOP_TOMOS/tomogram)
            if first_tomo_path.parent.name == "TOP_TOMOS":
                root_dir = str(first_tomo_path.parent.parent.parent)
        
        run_zonogram_analysis_for_all_tomograms(tomo_paths, output_dir, csv_path=csv_path, root_dir=root_dir, rerun=rerun)
        print("Active zonogram analysis completed successfully!")
    except Exception as e:
        print(f"Error in active zonogram analysis: {e}")
        import traceback
        traceback.print_exc()

def run_all_analyses(tomo_paths, results_manager, rerun=False, csv_path=None):
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
    activezone_paths = [(tomo, set_name) for (tomo, set_name, _) in tomo_paths]
    run_activezone(activezone_paths, results_manager, rerun, print_ascii=False)
    
    # Step 2: Vesicle Analysis
    print("\n" + "="*80)
    print("STEP 2: VESICLE ANALYSIS")
    print("="*80)
    vesicles_paths = [(tomo, set_name) for (tomo, set_name, _) in tomo_paths]
    run_vesicles(vesicles_paths, results_manager, rerun, print_ascii=False)
    
    # Step 3: AuNP Analysis
    print("\n" + "="*80)
    print("STEP 3: AUNP ANALYSIS")
    print("="*80)
    run_aunps(tomo_paths, results_manager, rerun, print_ascii=False)
    
    # Step 4: Visualizations
    print("\n" + "="*80)
    print("STEP 4: VISUALIZATION GENERATION")
    print("="*80)
    generate_visualizations(tomo_paths, results_manager, rerun, print_ascii=False, csv_path=csv_path)
    
    print("\n" + "="*80)
    print("ALL ANALYSES COMPLETED!")
    print("="*80)

def run_aunps(tomo_paths, results_manager, rerun=False, print_ascii=True):
    if print_ascii:
        print_synapse_ascii_art()
    for i, (tomo, set_name, aunp_active_zones) in enumerate(tomo_paths):
        tomogram_name = Path(tomo).name
        
        # Print separator between tomograms
        if i > 0:
            print("\n" + "="*80)
        print(f"\n{'='*33} TOMOGRAM {i+1}/{len(tomo_paths)} {'='*33}")
        print(f"Analyzing: {tomogram_name}")
        print("="*80)
        
        # Check if analysis already completed successfully
        existing_results = results_manager.get_tomogram_results(tomogram_name, 'aunps')
        has_completed = (existing_results and 
                        'results' in existing_results and 
                        'aunp_analysis' in existing_results['results'] and
                        existing_results['results']['aunp_analysis'].get('status') == 'completed')
        
        if has_completed and not rerun:
            print(f"Skipping AuNP analysis for {tomogram_name} (already completed successfully)")
            continue
            
        print(f"Running AuNP analysis on {tomogram_name}")
        
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
            # Run analyses and collect results
            aunp_results = analyze_aunps(tomo, az_indices, set_name=set_name)
            
            # Store combined results
            combined_results = {
                'aunp_analysis': aunp_results,
            }
            
            # Auto-overwrite if not using skip_completed (more intuitive behavior)
            results_manager.store_tomogram_results(tomogram_name, 'aunps', combined_results, overwrite=rerun, set_name=set_name)
        except Exception as e:
            print(f"Error in AuNP analysis for {tomogram_name}: {e}")
            # Store error results so we know this analysis failed
            error_results = {
                'aunp_analysis': {
                    'status': 'error',
                    'error': str(e)
                }
            }
            results_manager.store_tomogram_results(tomogram_name, 'aunps', error_results, overwrite=True, set_name=set_name)

def delete_csv_tomogram_results(csv_path, results_dir="results", data_dir="data"):
    """Delete results only for tomograms specified in the CSV file."""
    print(f"Deleting results for tomograms specified in {csv_path}")
    
    # Load CSV to get list of tomograms
    try:
        df = pd.read_csv(csv_path)
        csv_tomograms = set(df['tomoname'].tolist())
        print(f"Found {len(csv_tomograms)} tomograms in CSV")
    except Exception as e:
        print(f"Error reading CSV file {csv_path}: {e}")
        return
    
    # Delete specific tomogram results from the results directory
    results_path = Path(results_dir)
    if results_path.exists():
        # Load existing results
        results_manager = ResultsManager(results_dir)
        existing_results = results_manager.get_all_results()
        
        # Remove results for CSV tomograms only
        deleted_count = 0
        for tomogram_name in csv_tomograms:
            if tomogram_name in existing_results:
                del existing_results[tomogram_name]
                deleted_count += 1
                print(f"  Deleted results for {tomogram_name}")
        
        # Save updated results
        results_manager.results = existing_results
        results_manager._save_results()
        print(f"Deleted results for {deleted_count} tomograms from results directory")
    
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

def check_analysis_status(results_manager, tomogram_name, analysis_type):
    """Check if an analysis completed successfully, failed, or hasn't been run."""
    existing_results = results_manager.get_tomogram_results(tomogram_name, analysis_type)
    
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
    
    for tomogram_name, set_name, _ in tomos:
        print(f"\n{Path(tomogram_name).name}:")
        for analysis_type in analysis_types:
            status = check_analysis_status(results_manager, tomogram_name, analysis_type)
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
        for tomo, set_name, _ in tomos:
            missing_files = []
            base = Path(tomo) / "best_alignment"
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
                az_dir = base.parent / "STT_results" / "activezone"
                az_pre = list(az_dir.glob("*_pre.txt"))
                az_post = list(az_dir.glob("*_post.txt"))
                if not az_pre:
                    missing_files.append("active zone pre files (*_pre.txt)")
                if not az_post:
                    missing_files.append("active zone post files (*_post.txt)")
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

    if args.analysis == "activezone":
        activezone_paths = [(tomo, set_name) for (tomo, set_name, _) in tomos]
        run_activezone(activezone_paths, results_manager, rerun=args.rerun)
    elif args.analysis == "vesicles":
        vesicles_paths = [(tomo, set_name) for (tomo, set_name, _) in tomos]
        run_vesicles(vesicles_paths, results_manager, rerun=args.rerun)
    elif args.analysis == "aunps":
        run_aunps(tomos, results_manager, rerun=args.rerun)
    elif args.analysis == "visualizations":
        generate_visualizations(tomos, results_manager, rerun=args.rerun, csv_path=args.csv)
    elif args.analysis == "all":
        run_all_analyses(tomos, results_manager, rerun=args.rerun, csv_path=args.csv)

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
