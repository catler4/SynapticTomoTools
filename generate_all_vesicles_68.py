#!/usr/bin/env python3
"""
Script to generate combined visualization with ALL vesicles for tomogram 68.
"""

import sys
from pathlib import Path

# Add the src directory to the path so we can import the modules
sys.path.insert(0, str(Path(__file__).parent / "src"))

from synaptic_tomo_tools.visualization import plot_tomogram_overlays, filter_vesicles_in_slice

def main():
    # Path to tomogram 68
    tomo_path = Path("data/15F1_tomograms/TOP_TOMOS/20231017_EGmilled24-2_68")
    
    # Output directory
    output_dir = tomo_path / "best_alignment" / "STT_results" / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating ALL vesicles visualization for tomogram 68...")
    print(f"Tomogram path: {tomo_path}")
    print(f"Output directory: {output_dir}")
    
    # Temporarily modify the filter_vesicles_in_slice function to return all vesicles
    original_filter_vesicles_in_slice = filter_vesicles_in_slice
    
    def filter_vesicles_in_slice_all(vesicles, z_center, z_thresh):
        """Modified version that returns ALL vesicles, not just those in slice."""
        print(f"Showing ALL {len(vesicles)} vesicles (not filtered by slice)")
        return vesicles
    
    # Replace the function temporarily
    import synaptic_tomo_tools.visualization as viz
    viz.filter_vesicles_in_slice = filter_vesicles_in_slice_all
    
    try:
        # Generate the visualization with all vesicles
        # Use active zones 0 and 1 as specified in the CSV
        plot_tomogram_overlays(tomo_path, output_dir, aunp_active_zone_indices=[0, 1])
        
        print("Done! Check the output directory for the combined visualization.")
        
    finally:
        # Restore the original function
        viz.filter_vesicles_in_slice = original_filter_vesicles_in_slice

if __name__ == "__main__":
    main() 