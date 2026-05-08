#!/usr/bin/env python3
"""
Combined script to run both regular active zonogram analysis and mini zonogram analysis.
Eliminates redundancy by reusing membrane data and active zones.
"""

import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def main():
    import argparse
    import sys
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run combined zonogram analysis on a tomogram')
    parser.add_argument('--tomogram-path', type=str, required=True, help='Path to the tomogram directory')
    parser.add_argument('--output-dir', type=str, default='results', help='Base output directory for results')
    parser.add_argument('--tomogram-name', type=str, required=True, help='Name of the tomogram for file naming')
    
    args = parser.parse_args()
    
    # Call the function from the activezone module
    from synaptic_tomo_tools.activezone import run_combined_zonogram_analysis
    
    result = run_combined_zonogram_analysis(
        tomogram_path=args.tomogram_path,
        output_dir=args.output_dir,
        tomogram_name=args.tomogram_name
    )
    
    if result.get("success", False):
        print(f"Combined zonogram analysis completed successfully!")
        print(f"Regular zonograms: {result.get('regular_zonograms', 0)}")
        print(f"Mini zonograms: {result.get('mini_zonograms', 0)}")
        print(f"Total files created: {len(result.get('files_created', []))}")
    else:
        print(f"Zonogram analysis failed: {result.get('reason', 'Unknown error')}")
        sys.exit(1)


if __name__ == "__main__":
    main()

