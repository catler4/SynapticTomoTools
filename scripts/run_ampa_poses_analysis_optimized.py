#!/usr/bin/env python3
"""
CLI script for running optimized AMPA poses analysis.

This script estimates AMPA receptor poses using an optimized AuNP pairing algorithm
that maximizes the number of valid pairs while avoiding steric clashes between
predicted AMPA positions.
"""

import argparse
import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from synaptic_tomo_tools.ampa_poses_optimized import run_ampa_poses_analysis_optimized


def main():
    parser = argparse.ArgumentParser(
        description="Run optimized AMPA poses analysis on a tomogram",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic optimized analysis with default parameters
  python scripts/run_ampa_poses_analysis_optimized.py --tomogram-path data/15F1/TOP_TOMOS/20241030_AMmilled12-1_15 --output-dir results/ampa_poses_optimized
  
  # Custom distance parameters and steric radius
  python scripts/run_ampa_poses_analysis_optimized.py --tomogram-path data/15F1/TOP_TOMOS/20241030_AMmilled12-1_15 --output-dir results/ampa_poses_optimized --aunp-min-distance 5 --aunp-max-distance 12 --membrane-min-distance 15 --membrane-max-distance 25 --steric-radius 6
  
  # Disable distance cutoffs (use all pairs)
  python scripts/run_ampa_poses_analysis_optimized.py --tomogram-path data/15F1/TOP_TOMOS/20241030_AMmilled12-1_15 --output-dir results/ampa_poses_optimized --no-aunp-distance-cutoff --no-membrane-distance-cutoff
  
  # Use NetworkX exact optimization method (saves to networkx/ subdirectory)
  python scripts/run_ampa_poses_analysis_optimized.py --tomogram-path data/15F1/TOP_TOMOS/20241030_AMmilled12-1_15 --output-dir results/ampa_poses_optimized --method networkx
  
  # Use ILP exact optimization method (saves to ilp/ subdirectory)
  python scripts/run_ampa_poses_analysis_optimized.py --tomogram-path data/15F1/TOP_TOMOS/20241030_AMmilled12-1_15 --output-dir results/ampa_poses_optimized --method ilp
        """
    )
    
    parser.add_argument(
        "--tomogram-path",
        required=True,
        help="Path to the tomogram directory"
    )
    
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save results"
    )
    
    parser.add_argument(
        "--aunp-min-distance",
        type=float,
        default=6.0,
        help="Minimum distance between AuNPs in nm (default: 6.0)"
    )
    
    parser.add_argument(
        "--aunp-max-distance",
        type=float,
        default=12.0,
        help="Maximum distance between AuNPs in nm (default: 12.0)"
    )
    
    parser.add_argument(
        "--no-aunp-distance-cutoff",
        action="store_true",
        help="Disable AuNP distance cutoff (use all AuNP pairs)"
    )
    
    parser.add_argument(
        "--membrane-min-distance",
        type=float,
        default=17.0,
        help="Minimum distance from AuNP to membrane in nm (default: 17.0)"
    )
    
    parser.add_argument(
        "--membrane-max-distance",
        type=float,
        default=23.0,
        help="Maximum distance from AuNP to membrane in nm (default: 23.0)"
    )
    
    parser.add_argument(
        "--no-membrane-distance-cutoff",
        action="store_true",
        help="Disable membrane distance cutoff (use all pairs regardless of membrane distance)"
    )
    
    parser.add_argument(
        "--steric-radius",
        type=float,
        default=5.0,
        help="Minimum distance between AMPA positions to avoid steric clashes in nm (default: 5.0)"
    )
    
    parser.add_argument(
        "--method",
        choices=["greedy", "networkx", "ilp"],
        default="greedy",
        help="Optimization method: 'greedy' for fast heuristic solution (saves to optimized/), 'networkx' for exact optimal solution using graph theory (saves to networkx/), 'ilp' for exact optimal solution using integer linear programming (saves to ilp/) (default: greedy)"
    )
    
    parser.add_argument(
        "--aunp-active-zones",
        nargs="+",
        type=int,
        help="Specific active zone indices to analyze (default: all active zones)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments only if cutoffs are enabled
    if not args.no_aunp_distance_cutoff and args.aunp_min_distance >= args.aunp_max_distance:
        print("Error: AuNP minimum distance must be less than maximum distance")
        sys.exit(1)
    
    if not args.no_membrane_distance_cutoff and args.membrane_min_distance >= args.membrane_max_distance:
        print("Error: Membrane minimum distance must be less than maximum distance")
        sys.exit(1)
    
    if args.steric_radius <= 0:
        print("Error: Steric radius must be positive")
        sys.exit(1)
    
    # Run the analysis
    try:
        print(f"Running optimized AMPA poses analysis on {args.tomogram_path}")
        print(f"Output directory: {args.output_dir}")
        
        if args.no_aunp_distance_cutoff:
            print("AuNP distance range: No cutoff (using all AuNP pairs)")
        else:
            print(f"AuNP distance range: {args.aunp_min_distance}-{args.aunp_max_distance} nm")
            
        if args.no_membrane_distance_cutoff:
            print("Membrane distance range: No cutoff (using all pairs regardless of membrane distance)")
        else:
            print(f"Membrane distance range: {args.membrane_min_distance}-{args.membrane_max_distance} nm")
            
        print(f"Steric radius: {args.steric_radius} nm")
            
        if args.aunp_active_zones:
            print(f"Active zones: {args.aunp_active_zones}")
        else:
            print("Active zones: all")
        print()
        
        # Set distance parameters based on cutoff flags
        if args.no_aunp_distance_cutoff:
            inter_aunp_distance = None
        else:
            inter_aunp_distance = (args.aunp_min_distance, args.aunp_max_distance)
            
        if args.no_membrane_distance_cutoff:
            aunp_membrane_distance = None
        else:
            aunp_membrane_distance = (args.membrane_min_distance, args.membrane_max_distance)
        
        results = run_ampa_poses_analysis_optimized(
            tomo_path=args.tomogram_path,
            output_dir=args.output_dir,
            aunp_active_zones=args.aunp_active_zones,
            inter_aunp_distance=inter_aunp_distance,
            aunp_membrane_distance=aunp_membrane_distance,
            ampa_steric_radius=args.steric_radius,
            method=args.method
        )
        
        if results["status"] == "success":
            print(f"Optimized AMPA poses analysis completed successfully!")
            print(f"Found {results['pairs_found']} AMPA receptor poses")
            print(f"Unpaired AuNPs: {results['unpaired_aunps']}")
            print(f"Steric clashes: {results['steric_clashes']}")
            print(f"Overall pairing efficiency: {results['overall_pairing_efficiency']:.2%}")
            print(f"Normalized pairing efficiency: {results['normalized_pairing_efficiency']:.2%}")
            print(f"RELION star file: {results['star_file']}")
            print(f"Paired AuNPs star file: {results['paired_aunps_file']}")
            print(f"Unpaired AuNPs star file: {results['unpaired_file']}")
            print(f"Summary CSV: {results['summary_file']}")
            print(f"Optimization stats: {results['stats_file']}")
        elif results["status"] == "no_pairs":
            print("No valid AuNP pairs found within specified distance ranges")
        else:
            print(f"Analysis failed with status: {results['status']}")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error running optimized AMPA poses analysis: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
