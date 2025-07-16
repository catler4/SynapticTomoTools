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
SET_ROOTS = {
    "15F1": Path("/goliath/processing/Gouaux/CJS/BestTomo/ProcessingCJS/tomograms/15F1_tomograms/TOP_TOMOS"),
    "5F11": Path("/goliath/processing/Gouaux/CJS/BestTomo/ProcessingCJS/tomograms/5F11_tomograms/TOP_TOMOS"),
    "15F1and5F11": Path("/goliath/processing/Gouaux/CJS/BestTomo/ProcessingCJS/tomograms/15F1and5F11_tomograms/TOP_TOMOS"),
    "15F1and5F11dimer": Path("/goliath/processing/Gouaux/CJS/BestTomo/ProcessingCJS/tomograms/15F1and5F11dimer_tomograms/TOP_TOMOS"),
    "11B8": Path("/goliath/processing/Gouaux/CJS/BestTomo/ProcessingCJS/tomograms/11B8_tomograms/TOP_TOMOS"),
    "unlabeled": Path("/goliath/processing/Gouaux/CJS/BestTomo/ProcessingCJS/tomograms/unlabeled_tomograms/TOP_TOMOS"),
    # Add more sets here if needed
}

def load_tomograms(csv_path, analysis_type, set_name=None):
    """
    Load filtered tomogram paths based on analysis type and set,
    with support for per-set data root directories.
    Returns a list of (path, set_name) tuples.
    """
    df = pd.read_csv(csv_path)

    # Handle 'all' analysis type - include tomograms that have any analysis enabled
    if analysis_type == 'all':
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
    paths = []
    for _, row in filtered.iterrows():
        row: pd.Series  # type hint for linter
        root = SET_ROOTS.get(str(row["set"]))
        if root is None:
            raise ValueError(f"No root path defined for set: {row['set']}")
        full_path = root / row["tomoname"]
        # Get aunp_active_zones if present, else empty string
        aunp_active_zones = row.get("aunp_active_zones", "") if "aunp_active_zones" in row else ""
        paths.append((full_path, row["set"], aunp_active_zones))

    return paths

def print_synapse_ascii_art():
    synapse_art = r"""
╔═╗┬ ┬┌┐┌┌─┐┌─┐┌┬┐┬┌─┐╔╦╗┌─┐┌┬┐┌─┐╔╦╗┌─┐┌─┐┬  ┌─┐
╚═╗└┬┘│││├─┤├─┘ │ ││   ║ │ │││││ │ ║ │ ││ ││  └─┐
╚═╝ ┴ ┘└┘┴ ┴┴   ┴ ┴└─┘ ╩ └─┘┴ ┴└─┘ ╩ └─┘└─┘┴─┘└─┘

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

def print_vesicle_ascii_art():
    vesicle_art = r"""
                                  
\  /_ _. _| _   /\  _  _ |   _. _ 
 \/(-_)|(_|(-  /--\| )(_||\/_)|_) 
                          /       
"""
    print(vesicle_art)

def print_activezone_ascii_art():
    activezone_art = r"""
               ___                             
 /\  _|_.   _   _/ _  _  _   /\  _  _ |   _. _ 
/--\(_|_|\/(-  /__(_)| )(-  /--\| )(_||\/_)|_) 
                                       /       
"""
    print(activezone_art)

def print_aunps_ascii_art():
    aunps_art = r"""
            __                      
 /\    |\ ||__)   /\  _  _ |   _. _ 
/--\|_|| \||     /--\| )(_||\/_)|_) 
                            /       
"""
    print(aunps_art)

def run_activezone(tomo_paths, results_manager, rerun=False, print_ascii=True):
    if print_ascii:
        print_synapse_ascii_art()
        print_activezone_ascii_art()
    for i, (tomo, set_name) in enumerate(tomo_paths):
        tomogram_name = Path(tomo).name
        print(f"\n{'='*33} TOMOGRAM {i+1}/{len(tomo_paths)} {'='*33}")
        print(f"Analyzing: {tomogram_name}")
        print("="*80)
        has_existing = results_manager.has_results(tomogram_name, 'activezone')
        if has_existing and not rerun:
            print(f"Skipping active zone analysis for {tomogram_name} (already completed)")
            continue
        print(f"Running active zone analysis on {tomogram_name}")
        az_results = define_active_zone(tomo)
        cleft_results = calculate_cleft_width(tomo)
        combined_results = {
            'active_zone': az_results,
            'cleft_width': cleft_results
        }
        results_manager.store_tomogram_results(tomogram_name, 'activezone', combined_results, overwrite=rerun, set_name=set_name)

def run_vesicles(tomo_paths, results_manager, rerun=False, print_ascii=True):
    if print_ascii:
        print_synapse_ascii_art()
        print_vesicle_ascii_art()
    for i, (tomo, set_name) in enumerate(tomo_paths):
        tomogram_name = Path(tomo).name
        if i > 0:
            print("\n" + "="*80)
        print(f"\n{'='*33} TOMOGRAM {i+1}/{len(tomo_paths)} {'='*33}")
        print(f"Analyzing: {tomogram_name}")
        print("="*80)
        if results_manager.has_results(tomogram_name, 'vesicles') and not rerun:
            print(f"Skipping vesicle analysis for {tomogram_name} (already completed)")
            continue
        print(f"Running vesicle analysis on {tomogram_name}")
        vesicle_results = detect_vesicles(tomo)
        distance_results = measure_distances_to_az(tomo)
        combined_results = {
            'vesicle_detection': vesicle_results,
            'distance_measurements': distance_results
        }
        results_manager.store_tomogram_results(tomogram_name, 'vesicles', combined_results, overwrite=rerun, set_name=set_name)

def generate_visualizations(tomo_paths, results_manager, rerun=False):
    """Generate visualization images for each tomogram after analysis is complete."""
    if plot_tomogram_overlays is None:
        print("Skipping visualization generation (visualization module not available)")
        return
        
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    # Create combined visualization directory in results
    combined_viz_dir = Path(results_manager.results_dir) / 'visualizations'
    combined_viz_dir.mkdir(exist_ok=True)
    print(f"Combined visualizations will be saved to: {combined_viz_dir}")
    
    for i, (tomo, set_name, aunp_active_zones) in enumerate(tomo_paths):
        tomogram_name = Path(tomo).name
        
        # Create visualization output directory within the tomogram's results folder
        viz_output_dir = Path(tomo) / 'best_alignment' / 'STT_results' / 'visualizations'
        viz_output_dir.mkdir(parents=True, exist_ok=True)

        # Check if all expected visualization files exist and skip_completed is True
        expected_files = [
            viz_output_dir / f"{tomogram_name}_activezone.png",
            viz_output_dir / f"{tomogram_name}_aunps.png",
            viz_output_dir / f"{tomogram_name}_combined.png",
        ]
        if all(f.exists() for f in expected_files) and not rerun:
            print(f"Skipping visualization for {tomogram_name} (already completed)")
            continue
        
        print(f"\nGenerating visualizations for {tomogram_name}")
        print(f"Individual output directory: {viz_output_dir}")
        
        try:
            # Parse aunp_active_zones string to list of ints or None
            az_str = str(aunp_active_zones) if aunp_active_zones is not None else ""
            if az_str.strip() == "" or az_str.lower() == "nan":
                az_indices = None
            else:
                az_indices = [int(x) for x in az_str.split(",") if x.strip().isdigit()]
            # Generate the three visualization types in the tomogram's directory
            plot_tomogram_overlays(tomo, viz_output_dir, az_indices)
            
            # Copy the generated files to the combined directory with tomogram name prefix
            for viz_file in viz_output_dir.glob(f"{tomogram_name}_*.png"):
                combined_file = combined_viz_dir / viz_file.name
                import shutil
                shutil.copy2(viz_file, combined_file)
                print(f"  Copied {viz_file.name} to combined directory")
            
            print(f"✓ Successfully generated visualizations for {tomogram_name}")
        except Exception as e:
            print(f"✗ Failed to generate visualizations for {tomogram_name}: {e}")
            continue
    
    print(f"\nAll visualizations saved to:")
    print(f"  Individual: {viz_output_dir}")
    print(f"  Combined: {combined_viz_dir}")

def run_all_analyses(tomo_paths, results_manager, rerun=False):
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
    generate_visualizations(tomo_paths, results_manager, rerun)
    
    print("\n" + "="*80)
    print("ALL ANALYSES COMPLETED!")
    print("="*80)

def run_aunps(tomo_paths, results_manager, rerun=False, print_ascii=True):
    if print_ascii:
        print_synapse_ascii_art()
        print_aunps_ascii_art()
    for i, (tomo, set_name, aunp_active_zones) in enumerate(tomo_paths):
        tomogram_name = Path(tomo).name
        
        # Print separator between tomograms
        if i > 0:
            print("\n" + "="*80)
        print(f"\n{'='*33} TOMOGRAM {i+1}/{len(tomo_paths)} {'='*33}")
        print(f"Analyzing: {tomogram_name}")
        print("="*80)
        
        # Check if analysis already completed
        if results_manager.has_results(tomogram_name, 'aunps') and not rerun:
            print(f"Skipping AuNP analysis for {tomogram_name} (already completed)")
            continue
            
        print(f"Running AuNP analysis on {tomogram_name}")
        
        # Parse aunp_active_zones string to list of ints or None
        az_str = str(aunp_active_zones) if aunp_active_zones is not None else ""
        if az_str.strip() == "" or az_str.lower() == "nan":
            az_indices = None
        else:
            az_indices = [int(x) for x in az_str.split(",") if x.strip().isdigit()]
        # Run analyses and collect results
        aunp_results = analyze_aunps(tomo, az_indices)
        
        # Store combined results
        combined_results = {
            'aunp_analysis': aunp_results,
        }
        
        # Auto-overwrite if not using skip_completed (more intuitive behavior)
        results_manager.store_tomogram_results(tomogram_name, 'aunps', combined_results, overwrite=rerun, set_name=set_name)

def delete_all_analysis_results(results_dir="results", data_dir="data"):
    """Delete all files in the results directory and all STT_results subdirectories under best_alignment for all tomograms."""
    # Delete everything in the results directory
    results_path = Path(results_dir)
    if results_path.exists():
        print(f"Deleting all files in {results_path}...")
        shutil.rmtree(results_path)
        results_path.mkdir(exist_ok=True)
    # Delete all STT_results subdirs under best_alignment for all tomograms
    data_path = Path(data_dir)
    for root, dirs, files in os.walk(data_path):
        for d in dirs:
            if d == "STT_results":
                stt_path = Path(root) / d
                print(f"Deleting {stt_path}...")
                shutil.rmtree(stt_path)

def main():
    parser = argparse.ArgumentParser(
        description="Run SynapticTomoTools analysis on selected tomograms."
    )
    parser.add_argument(
        "--analysis", required=True, choices=["activezone", "vesicles", "aunps", "all"],
        help="Which analysis to run. Use 'all' to run activezone, vesicles, aunps, and visualizations in sequence."
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
        "--generate-visualizations", action="store_true",
        help="Generate visualization images for each tomogram after analysis completion."
    )
    parser.add_argument(
        "--delete-results", action="store_true",
        help="Delete all analysis results files before running analysis."
    )

    args = parser.parse_args()

    # Handle test mode
    global SET_ROOTS
    if args.test:
        repo_root = Path(__file__).parent.parent.parent.resolve()
        SET_ROOTS = {
            "15F1": repo_root / "data" / "15F1_tomograms" / "TOP_TOMOS",
            "5F11": repo_root / "data" / "5F11_tomograms" / "TOP_TOMOS",
            "15F1and5F11": repo_root / "data" / "15F1and5F11_tomograms" / "TOP_TOMOS",
            "15F1and5F11dimer": repo_root / "data" / "15F1and5F11dimer_tomograms" / "TOP_TOMOS",
            "11B8": repo_root / "data" / "11B8_tomograms" / "TOP_TOMOS",
            "unlabeled": repo_root / "data" / "unlabeled_tomograms" / "TOP_TOMOS",
        }
        if args.csv is None:
            args.csv = str(repo_root / "data" / "tomograms-test.csv")
    else:
        if args.csv is None:
            args.csv = "data/tomograms.csv"
    
    if args.delete_results:
        delete_all_analysis_results(args.results_dir, "data")

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

    if args.analysis == "activezone":
        run_activezone(tomos, results_manager, rerun=args.rerun)
    elif args.analysis == "vesicles":
        run_vesicles(tomos, results_manager, rerun=args.rerun)
    elif args.analysis == "aunps":
        run_aunps(tomos, results_manager, rerun=args.rerun)
    elif args.analysis == "all":
        run_all_analyses(tomos, results_manager, rerun=args.rerun)
    else:
        # Generate visualizations if requested (only for individual analyses)
        if args.generate_visualizations:
            generate_visualizations(tomos, results_manager, rerun=args.rerun)

    # Always export summary CSVs at the end
    print("\nExporting all summary CSVs from stored results...")
    results_manager.export_to_csv()

if __name__ == "__main__":
    main()
