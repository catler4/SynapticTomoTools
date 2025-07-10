# src/synaptic_tomo_tools/__init__.py

"""
SynapticTomoTools
------------------
A Python toolkit for analyzing synaptic cryo-electron tomography (cryo-ET) data.

Core functionality includes:
- Active zone geometry analysis
- Vesicle detection and spatial quantification
- Gold nanoparticle (AuNP) proximity analysis
"""

__version__ = "0.1.0"

# Expose core API functions for top-level imports
from .activezone import define_active_zone, calculate_cleft_width
from .vesicles import detect_vesicles, measure_distances_to_az
from .aunps import analyze_aunps, compute_vesicle_aunp_distances

__all__ = [
    "define_active_zone",
    "calculate_cleft_width",
    "detect_vesicles",
    "measure_distances_to_az",
    "analyze_aunps",
    "compute_vesicle_aunp_distances",
]
