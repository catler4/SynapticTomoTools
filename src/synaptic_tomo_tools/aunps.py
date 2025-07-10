# src/synaptic_tomo_tools/aunps.py

from pathlib import Path

def analyze_aunps(tomogram_path):
    """
    Performs analysis of gold nanoparticles (AuNPs) in the tomogram.

    Parameters:
        tomogram_path (str or Path): Path to the tomogram file.
    """
    print(f"Analyzing AuNPs in {Path(tomogram_path).name}")
    # TODO: implement AuNP detection and analysis


def compute_vesicle_aunp_distances(tomogram_path):
    """
    Computes distances between vesicles and AuNPs.

    Parameters:
        tomogram_path (str or Path): Path to the tomogram file.
    """
    print(f"Computing distances between vesicles and AuNPs in {Path(tomogram_path).name}")
    # TODO: implement distance computation logic
