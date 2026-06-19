#!/usr/bin/env python3
"""
CLI entry point for fusion-point vs AuNP density analysis.

  conda activate synaptictomo
  PYTHONPATH=src python scripts/fusion_point_vs_aunp_density_analysis.py \\
    --tomogram-path data/15F1/TOP_TOMOS/20240111_WaffleHipp_116 \\
    --alignment-dir best_alignment \\
    --output-dir results/fusion_point_vs_aunp_density/20240111_WaffleHipp_116
"""

from synaptic_tomo_tools.fusion_point_vs_aunp_density import main

if __name__ == "__main__":
    main()
