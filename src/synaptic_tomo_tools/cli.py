import argparse
import pandas as pd
from pathlib import Path
from synaptic_tomo_tools import (
    define_active_zone,
    calculate_cleft_width,
    detect_vesicles,
    measure_distances_to_az,
    analyze_aunps,
    compute_vesicle_aunp_distances,
)
from .results_manager import ResultsManager

# Map each tomogram set name to its specific root directory
SET_ROOTS = {
    "15F1": Path("data/15F1_tomograms/TOP_TOMOS"),
    "unlabeled": Path("data/unlabeled_tomograms/TOP_TOMOS"),
    # Add more sets here if needed
}

def load_tomograms(csv_path, analysis_type, set_name=None):
    """
    Load filtered tomogram paths based on analysis type and set,
    with support for per-set data root directories.
    Returns a list of (path, set_name) tuples.
    """
    df = pd.read_csv(csv_path)

    if analysis_type not in df.columns:
        raise ValueError(f"Column '{analysis_type}' not found in {csv_path}.")

    filtered = df[df[analysis_type] == True]
    if set_name:
        filtered = filtered[filtered["set"] == set_name]

    # Construct full paths based on set-specific root
    paths = []
    for _, row in filtered.iterrows():
        root = SET_ROOTS.get(row["set"])
        if root is None:
            raise ValueError(f"No root path defined for set: {row['set']}")
        full_path = root / row["tomoname"]
        paths.append((full_path, row["set"]))

    return paths

def run_activezone(tomo_paths, results_manager, skip_completed=False, overwrite=False):
    for i, (tomo, set_name) in enumerate(tomo_paths):
        tomogram_name = Path(tomo).name
        
        # Print separator between tomograms
        print(f"\n{'='*33} TOMOGRAM {i+1}/{len(tomo_paths)} {'='*33}")
        print(f"Analyzing: {tomogram_name}")
        print("="*80)
        
        # Check if analysis already completed
        has_existing = results_manager.has_results(tomogram_name, 'activezone')
        
        if skip_completed and has_existing and not overwrite:
            print(f"Skipping active zone analysis for {tomogram_name} (already completed)")
            continue
            
        print(f"Running active zone analysis on {tomogram_name}")
        
        # Run analyses and collect results
        az_results = define_active_zone(tomo)
        cleft_results = calculate_cleft_width(tomo)
        
        # Store combined results
        combined_results = {
            'active_zone': az_results,
            'cleft_width': cleft_results
        }
        
        # Auto-overwrite if not using skip_completed (more intuitive behavior)
        auto_overwrite = not skip_completed or overwrite
        results_manager.store_tomogram_results(tomogram_name, 'activezone', combined_results, overwrite=auto_overwrite, set_name=set_name)

def run_vesicles(tomo_paths, results_manager, skip_completed=False):
    for i, (tomo, set_name) in enumerate(tomo_paths):
        tomogram_name = Path(tomo).name
        
        # Print separator between tomograms
        if i > 0:
            print("\n" + "="*80)
        print(f"\n{'='*33} TOMOGRAM {i+1}/{len(tomo_paths)} {'='*33}")
        print(f"Analyzing: {tomogram_name}")
        print("="*80)
        
        # Check if analysis already completed
        if skip_completed and results_manager.has_results(tomogram_name, 'vesicles'):
            print(f"Skipping vesicle analysis for {tomogram_name} (already completed)")
            continue
            
        print(f"Running vesicle analysis on {tomogram_name}")
        
        # Run analyses and collect results
        vesicle_results = detect_vesicles(tomo)
        distance_results = measure_distances_to_az(tomo)
        
        # Store combined results
        combined_results = {
            'vesicle_detection': vesicle_results,
            'distance_measurements': distance_results
        }
        
        results_manager.store_tomogram_results(tomogram_name, 'vesicles', combined_results, set_name=set_name)

def run_aunps(tomo_paths, results_manager, skip_completed=False):
    for i, (tomo, set_name) in enumerate(tomo_paths):
        tomogram_name = Path(tomo).name
        
        # Print separator between tomograms
        if i > 0:
            print("\n" + "="*80)
        print(f"\n{'='*33} TOMOGRAM {i+1}/{len(tomo_paths)} {'='*33}")
        print(f"Analyzing: {tomogram_name}")
        print("="*80)
        
        # Check if analysis already completed
        if skip_completed and results_manager.has_results(tomogram_name, 'aunps'):
            print(f"Skipping AuNP analysis for {tomogram_name} (already completed)")
            continue
            
        print(f"Running AuNP analysis on {tomogram_name}")
        
        # Run analyses and collect results
        aunp_results = analyze_aunps(tomo)
        distance_results = compute_vesicle_aunp_distances(tomo)
        
        # Store combined results
        combined_results = {
            'aunp_analysis': aunp_results,
            'vesicle_aunp_distances': distance_results
        }
        
        results_manager.store_tomogram_results(tomogram_name, 'aunps', combined_results, set_name=set_name)

def main():
    parser = argparse.ArgumentParser(
        description="Run SynapticTomoTools analysis on selected tomograms."
    )
    parser.add_argument(
        "--analysis", required=True, choices=["activezone", "vesicles", "aunps"],
        help="Which analysis to run."
    )
    parser.add_argument(
        "--set", default=None,
        help="(Optional) Filter tomograms by experimental set name (e.g., 15F1, unlabeled)."
    )
    parser.add_argument(
        "--csv", default="data/tomograms.csv",
        help="Path to CSV file listing tomograms and analysis flags."
    )
    parser.add_argument(
        "--skip-completed", action="store_true",
        help="Skip analyses that have already been completed."
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing results even if analysis is already completed."
    )
    parser.add_argument(
        "--export-csv", action="store_true",
        help="Export results to CSV file after completion."
    )
    parser.add_argument(
        "--results-dir", default="results",
        help="Directory to store analysis results."
    )

    args = parser.parse_args()
    
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
        run_activezone(tomos, results_manager, args.skip_completed, args.overwrite)
    elif args.analysis == "vesicles":
        run_vesicles(tomos, results_manager, args.skip_completed)
    elif args.analysis == "aunps":
        run_aunps(tomos, results_manager, args.skip_completed)

    # Always export to CSV
    results_manager.export_to_csv()

if __name__ == "__main__":
    main()
