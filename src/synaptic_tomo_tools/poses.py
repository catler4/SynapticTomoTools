"""
CLI entry point for AMPA receptor pose analysis.

Estimates AMPA poses from AuNP pair analysis (findingampa-style create-relion-starfile).
Core algorithms live in ``ampa_poses``; this module provides the command-line interface.

Examples:
  python -m synaptic_tomo_tools.poses --tomogram-path data/.../tomo --alignment-dir best_alignment --output-dir results/poses
  python -m src.synaptic_tomo_tools.poses --tomogram-path data/.../tomo --alignment-dir best_alignment --output-dir results/poses
"""

from __future__ import annotations

import argparse
import sys

from .ampa_poses import run_ampa_poses_analysis_optimized, run_ampa_poses_analysis_original


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run AMPA poses analysis on a tomogram",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m synaptic_tomo_tools.poses --tomogram-path data/15F1/TOP_TOMOS/20241030_AMmilled12-1_15 --alignment-dir best_alignment --output-dir results/poses
  python -m synaptic_tomo_tools.poses --tomogram-path data/15F1/TOP_TOMOS/20241030_AMmilled12-1_15 --alignment-dir best_alignment --output-dir results/poses --aunp-min-distance 5 --aunp-max-distance 12 --membrane-min-distance 15 --membrane-max-distance 25
        """,
    )

    parser.add_argument(
        "--tomogram-path",
        required=True,
        help="Path to the tomogram directory",
    )
    parser.add_argument(
        "--alignment-dir",
        required=True,
        help="Alignment subdirectory under the tomogram (e.g. best_alignment, liza_az0); must match CSV alignment_dir",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save results",
    )
    parser.add_argument(
        "--aunp-min-distance",
        type=float,
        default=6.0,
        help="Minimum distance between AuNPs in nm (default: 6.0)",
    )
    parser.add_argument(
        "--aunp-max-distance",
        type=float,
        default=12.0,
        help="Maximum distance between AuNPs in nm (default: 12.0)",
    )
    parser.add_argument(
        "--no-aunp-distance-cutoff",
        action="store_true",
        help="Disable AuNP distance cutoff (use all AuNP pairs)",
    )
    parser.add_argument(
        "--membrane-min-distance",
        type=float,
        default=17.0,
        help="Minimum distance from AuNP to membrane in nm (default: 17.0)",
    )
    parser.add_argument(
        "--membrane-max-distance",
        type=float,
        default=23.0,
        help="Maximum distance from AuNP to membrane in nm (default: 23.0)",
    )
    parser.add_argument(
        "--no-membrane-distance-cutoff",
        action="store_true",
        help="Disable membrane distance cutoff (use all pairs regardless of membrane distance)",
    )
    parser.add_argument(
        "--cleft-ids",
        nargs="+",
        type=int,
        help="Specific synaptic cleft indices to analyze (default: all synaptic clefts)",
    )
    parser.add_argument(
        "--method",
        choices=["original", "greedy", "ilp"],
        default="greedy",
        help=(
            "Analysis method: 'original' for all poses (no optimization), "
            "'greedy' for fast heuristic (saves to greedy/), "
            "'ilp' for exact ILP solution (saves to ilp/) (default: greedy)"
        ),
    )
    parser.add_argument(
        "--steric-radius",
        type=float,
        default=5.0,
        help="Minimum distance between particle positions in nm (default: 5.0)",
    )
    parser.add_argument(
        "--pdb-file",
        type=str,
        help=(
            "Path to PDB file for structure template. If provided, generates PDB files "
            "with AMPA structures at calculated poses. Leave empty to skip PDB generation."
        ),
    )
    parser.add_argument(
        "--aunp-pick-star-pattern",
        type=str,
        default=None,
        help=(
            "Per-synaptic-cleft AuNP pick STAR filename pattern; use '*' for the synaptic cleft index "
            "(default: aunp_tm_BP_active_zone_*_manual_refined.star)"
        ),
    )

    args = parser.parse_args(argv)

    if not args.no_aunp_distance_cutoff and args.aunp_min_distance >= args.aunp_max_distance:
        print("Error: AuNP minimum distance must be less than maximum distance")
        sys.exit(1)

    if (
        not args.no_membrane_distance_cutoff
        and args.membrane_min_distance >= args.membrane_max_distance
    ):
        print("Error: Membrane minimum distance must be less than maximum distance")
        sys.exit(1)

    try:
        print(f"Running AMPA poses analysis on {args.tomogram_path}")
        print(f"Alignment directory: {args.alignment_dir}")
        print(f"Output directory: {args.output_dir}")

        if args.no_aunp_distance_cutoff:
            print("AuNP distance range: No cutoff (using all AuNP pairs)")
        else:
            print(f"AuNP distance range: {args.aunp_min_distance}-{args.aunp_max_distance} nm")

        if args.no_membrane_distance_cutoff:
            print(
                "Membrane distance range: No cutoff "
                "(using all pairs regardless of membrane distance)"
            )
        else:
            print(
                f"Membrane distance range: "
                f"{args.membrane_min_distance}-{args.membrane_max_distance} nm"
            )

        if args.cleft_ids:
            print(f"Synaptic clefts: {args.cleft_ids}")
        else:
            print("Synaptic clefts: all")
        print(f"Method: {args.method}")
        print(f"Steric radius: {args.steric_radius} nm")
        if args.pdb_file:
            print(f"PDB file: {args.pdb_file}")
        else:
            print("PDB file: None (skipping PDB generation)")
        if args.aunp_pick_star_pattern:
            print(f"AuNP pick STAR pattern: {args.aunp_pick_star_pattern}")
        print()

        if args.no_aunp_distance_cutoff:
            inter_aunp_distance = None
        else:
            inter_aunp_distance = (args.aunp_min_distance, args.aunp_max_distance)

        if args.no_membrane_distance_cutoff:
            aunp_membrane_distance = None
        else:
            aunp_membrane_distance = (args.membrane_min_distance, args.membrane_max_distance)

        if args.method == "original":
            results = run_ampa_poses_analysis_original(
                tomo_path=args.tomogram_path,
                output_dir=args.output_dir,
                cleft_ids=args.cleft_ids,
                inter_aunp_distance=inter_aunp_distance,
                aunp_membrane_distance=aunp_membrane_distance,
                pdb_file=args.pdb_file,
                alignment_dir=args.alignment_dir,
                aunp_pick_star_pattern=args.aunp_pick_star_pattern,
            )
        else:
            results = run_ampa_poses_analysis_optimized(
                tomo_path=args.tomogram_path,
                output_dir=args.output_dir,
                cleft_ids=args.cleft_ids,
                inter_aunp_distance=inter_aunp_distance,
                aunp_membrane_distance=aunp_membrane_distance,
                ampa_steric_radius=args.steric_radius,
                method=args.method,
                pdb_file=args.pdb_file,
                alignment_dir=args.alignment_dir,
                aunp_pick_star_pattern=args.aunp_pick_star_pattern,
            )

        if results["status"] == "success":
            print("AMPA poses analysis completed successfully!")
            print(f"Found {results['pairs_found']} AMPA receptor poses")
            print(f"RELION star file: {results['star_file']}")
            print(f"AuNPs star file: {results['aunps_file']}")
            print(f"All AuNPs star file: {results['all_aunps_file']}")
            print(f"Summary CSV: {results['summary_file']}")
        elif results["status"] == "no_pairs":
            print("No AuNP pairs found within specified distance range")
        elif results["status"] == "no_pairs_after_filtering":
            print("No AuNP pairs found within specified membrane distance range")
        else:
            print(f"Analysis failed with status: {results['status']}")
            sys.exit(1)

    except Exception as e:
        print(f"Error running AMPA poses analysis: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
