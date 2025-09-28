#!/usr/bin/env python3
"""
Script to compare original and optimized AMPA poses analysis methods.

This script runs both the original and optimized AMPA poses analysis on the same
tomogram and provides a detailed comparison of the results.
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from synaptic_tomo_tools.ampa_poses import run_ampa_poses_analysis_optimized


def compare_results(original_results, optimized_results, output_dir):
    """
    Compare results from original and optimized methods.
    
    Args:
        original_results: Results from original method
        optimized_results: Results from optimized method
        output_dir: Directory to save comparison results
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create comparison summary
    comparison_data = {
        'Method': ['Original', 'Optimized'],
        'Total_Pairs': [
            original_results.get('pairs_found', 0),
            optimized_results.get('pairs_found', 0)
        ],
        'Unpaired_AuNPs': [
            'N/A',  # Original method doesn't track this
            optimized_results.get('unpaired_aunps', 0)
        ],
        'Steric_Clashes': [
            'N/A',  # Original method doesn't check this
            optimized_results.get('steric_clashes', 0)
        ],
        'Pairing_Efficiency': [
            'N/A',  # Original method doesn't calculate this
            optimized_results.get('pairing_efficiency', 0)
        ]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_file = output_path / "ampa_poses_comparison.csv"
    comparison_df.to_csv(comparison_file, index=False)
    
    print(f"Comparison saved to {comparison_file}")
    print("\nComparison Summary:")
    print(comparison_df.to_string(index=False))
    
    # Calculate improvement metrics
    if original_results.get('pairs_found', 0) > 0:
        pairs_improvement = (optimized_results.get('pairs_found', 0) - 
                           original_results.get('pairs_found', 0)) / original_results.get('pairs_found', 0) * 100
        print(f"\nPairs improvement: {pairs_improvement:+.1f}%")
    
    if optimized_results.get('steric_clashes', 0) == 0:
        print("✓ Optimized method has no steric clashes")
    else:
        print(f"⚠ Optimized method has {optimized_results.get('steric_clashes', 0)} steric clashes")
    
    if optimized_results.get('pairing_efficiency', 0) > 0:
        print(f"✓ Pairing efficiency: {optimized_results.get('pairing_efficiency', 0):.1%}")
    
    return comparison_df


def main():
    parser = argparse.ArgumentParser(
        description="Compare original and optimized AMPA poses analysis methods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare methods with default parameters
  python scripts/compare_ampa_poses_methods.py --tomogram-path data/15F1/TOP_TOMOS/20241030_AMmilled12-1_15 --output-dir results/ampa_poses_comparison
  
  # Compare with custom parameters
  python scripts/compare_ampa_poses_methods.py --tomogram-path data/15F1/TOP_TOMOS/20241030_AMmilled12-1_15 --output-dir results/ampa_poses_comparison --aunp-min-distance 5 --aunp-max-distance 12 --membrane-min-distance 15 --membrane-max-distance 25 --steric-radius 6
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
        help="Directory to save comparison results"
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
        "--aunp-active-zones",
        nargs="+",
        type=int,
        help="Specific active zone indices to analyze (default: all active zones)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.no_aunp_distance_cutoff and args.aunp_min_distance >= args.aunp_max_distance:
        print("Error: AuNP minimum distance must be less than maximum distance")
        sys.exit(1)
    
    if not args.no_membrane_distance_cutoff and args.membrane_min_distance >= args.membrane_max_distance:
        print("Error: Membrane minimum distance must be less than maximum distance")
        sys.exit(1)
    
    if args.steric_radius <= 0:
        print("Error: Steric radius must be positive")
        sys.exit(1)
    
    # Set distance parameters
    if args.no_aunp_distance_cutoff:
        inter_aunp_distance = None
    else:
        inter_aunp_distance = (args.aunp_min_distance, args.aunp_max_distance)
        
    if args.no_membrane_distance_cutoff:
        aunp_membrane_distance = None
    else:
        aunp_membrane_distance = (args.membrane_min_distance, args.membrane_max_distance)
    
    try:
        print(f"Comparing AMPA poses analysis methods on {args.tomogram_path}")
        print(f"Output directory: {args.output_dir}")
        print()
        
        # Run greedy method
        print("=" * 60)
        print("RUNNING GREEDY METHOD")
        print("=" * 60)
        greedy_results = run_ampa_poses_analysis_optimized(
            tomo_path=args.tomogram_path,
            output_dir=Path(args.output_dir) / "greedy",
            aunp_active_zones=args.aunp_active_zones,
            inter_aunp_distance=inter_aunp_distance,
            aunp_membrane_distance=aunp_membrane_distance,
            ampa_steric_radius=args.steric_radius,
            method="greedy"
        )
        
        print("\n" + "=" * 60)
        print("RUNNING ILP METHOD")
        print("=" * 60)
        ilp_results = run_ampa_poses_analysis_optimized(
            tomo_path=args.tomogram_path,
            output_dir=Path(args.output_dir) / "ilp",
            aunp_active_zones=args.aunp_active_zones,
            inter_aunp_distance=inter_aunp_distance,
            aunp_membrane_distance=aunp_membrane_distance,
            ampa_steric_radius=args.steric_radius,
            method="ilp"
        )
        
        print("\n" + "=" * 60)
        print("COMPARISON RESULTS")
        print("=" * 60)
        comparison_df = compare_results(greedy_results, ilp_results, args.output_dir)
        
        # Additional analysis if both methods succeeded
        if (greedy_results.get("status") == "success" and 
            ilp_results.get("status") == "success"):
            
            print("\n" + "=" * 60)
            print("DETAILED ANALYSIS")
            print("=" * 60)
            
            # Load and compare the actual AMPA positions
            try:
                original_summary = pd.read_csv(Path(args.output_dir) / "original" / 
                                             f"{Path(args.tomogram_path).name}_ampa_poses_*_summary.csv")
                optimized_summary = pd.read_csv(Path(args.output_dir) / "optimized" / 
                                              f"{Path(args.tomogram_path).name}_ampa_poses_optimized_*_summary.csv")
                
                print(f"Original method found {len(original_summary)} AMPA poses")
                print(f"Optimized method found {len(optimized_summary)} AMPA poses")
                
                # Calculate average distances
                if len(original_summary) > 0:
                    orig_avg_sep = original_summary['AuNP_Separation_nm'].mean()
                    orig_avg_mem = original_summary['Membrane_Distance_nm'].mean()
                    print(f"Original - Average AuNP separation: {orig_avg_sep:.2f} nm")
                    print(f"Original - Average membrane distance: {orig_avg_mem:.2f} nm")
                
                if len(optimized_summary) > 0:
                    opt_avg_sep = optimized_summary['AuNP_Separation_nm'].mean()
                    opt_avg_mem = optimized_summary['Membrane_Distance_nm'].mean()
                    print(f"Optimized - Average AuNP separation: {opt_avg_sep:.2f} nm")
                    print(f"Optimized - Average membrane distance: {opt_avg_mem:.2f} nm")
                
            except Exception as e:
                print(f"Could not load detailed results for comparison: {e}")
        
        print(f"\nComparison complete! Results saved to {args.output_dir}")
        
    except Exception as e:
        print(f"Error comparing AMPA poses analysis methods: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
