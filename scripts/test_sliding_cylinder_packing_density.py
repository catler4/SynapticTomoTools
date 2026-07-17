#!/usr/bin/env python3
"""
Test the sliding-cylinder AuNP packing density calculation on one tomogram.

For each active zone found in the tomogram, computes the packing density map
via ``synaptic_tomo_tools.aunps.calculate_packing_density_using_sliding_cylinder``
and saves a PNG overlaying the density map on the zonogram-transformed tomogram.

Example:

  python -u scripts/test_sliding_cylinder_packing_density.py \\
    --tomo-path /nrs/elferich/gouaux_tomo/15F1/TOP_TOMOS/20241030_AMmilled12-1_15 \\
    --output-dir results/sliding_cylinder_packing_density_test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import starfile
from matplotlib import pyplot as plt
from scipy.interpolate import griddata

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from synaptic_tomo_tools.activezone import (
    define_active_zonogram,
    extract_active_zonogram,
    find_active_zones_from_glb,
    import_membrane_segmentations_from_glb,
    transform_coordinates_to_active_zonogram,
)
from synaptic_tomo_tools.aunps import (
    calculate_packing_density_using_sliding_cylinder,
    concave_hull_grid_mask,
)


def main():
    parser = argparse.ArgumentParser(
        description="Test sliding cylinder AuNP packing density on one tomogram"
    )
    parser.add_argument(
        "--tomo-path",
        type=Path,
        default=Path("/nrs/elferich/gouaux_tomo/15F1/TOP_TOMOS/20241030_AMmilled12-1_15"),
        help="Tomogram directory",
    )
    parser.add_argument(
        "--alignment-dir",
        type=str,
        default="patch_tracking",
        help="Alignment subdirectory name (default: patch_tracking)",
    )
    parser.add_argument(
        "--tomo-type",
        type=str,
        default="ddw",
        help="Tomogram type used for zonogram extraction (default: ddw)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_REPO_ROOT / "results" / "sliding_cylinder_packing_density_test",
        help="Where to write the PNGs",
    )
    args = parser.parse_args()

    tomo_path = args.tomo_path
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Tomogram: {tomo_path}", flush=True)

    membranes_glb = import_membrane_segmentations_from_glb(
        tomo_path, alignment_dir=args.alignment_dir
    )
    active_zones_glb = find_active_zones_from_glb(membranes_glb)
    active_zonograms = define_active_zonogram(active_zones_glb)
    transformed_tomo = extract_active_zonogram(
        active_zonograms=active_zonograms,
        active_zones=active_zones_glb,
        tomo_path=tomo_path,
        tomo_type=args.tomo_type,
        alignment_dir=args.alignment_dir,
    )

    for i, az_name in enumerate(active_zones_glb["active_zones"]):
        active_zone = active_zones_glb["active_zones"][az_name]
        azonogram = active_zonograms["zonogram_data"][az_name]

        aunp_star = (
            Path(tomo_path)
            / "patch_tracking"
            / "aunps"
            / f"aunp_tm_BP_active_zone_{i}_manual_refined.star"
        )
        aunps = starfile.read(aunp_star)
        v_co, num_aunps_in_cylinder, aunp_density_per_nm2, density = (
            calculate_packing_density_using_sliding_cylinder(
                active_zone,
                azonogram,
                aunps[["faCoordinateX", "faCoordinateY", "faCoordinateZ"]].to_numpy(),
            )
        )
        t_v = transform_coordinates_to_active_zonogram(
            v_co, active_zonograms["zonogram_data"][az_name]
        )

        fig = plt.figure(figsize=(30, 5))
        rendered_tomo = transformed_tomo["rendered_zonograms"][az_name]["transformed_tomogram"]
        plt.imshow(np.min(rendered_tomo, axis=0), cmap="gray", origin="lower")

        # Interpolate every pixel on density_map between centers and assign packing_coefficient values
        tomo_shape = rendered_tomo.shape
        grid_x, grid_y = np.mgrid[0 : tomo_shape[1], 0 : tomo_shape[2]]
        density_map = griddata(
            t_v[:, :2], density, (grid_y, grid_x), method="linear", fill_value=0
        )
        # Restrict the overlay to a concave outline around the projected vertices,
        # instead of griddata's convex hull, which bridges straight across
        # concavities in the active zone footprint and produces edge artifacts.
        inside_mask, hull_polygons = concave_hull_grid_mask(t_v[:, :2], density_map.shape)
        density_map = np.where(inside_mask, density_map, np.nan)
        plt.imshow(density_map, cmap="inferno", alpha=0.6, origin="lower",interpolation="mitchell")
        #for poly in hull_polygons:
        #    plt.plot(poly[:, 0], poly[:, 1], color="white", linewidth=0.8, linestyle="--")
        cbar = plt.colorbar(label="nm² / AuNP")
        # density (packing coefficient) is linear in aunp_density_per_nm2, so convert the
        # colorbar's tick positions to the corresponding 1/aunp_density_per_nm2 (nm² per AuNP)
        valid = aunp_density_per_nm2 > 0
        density_to_aunp_density = np.nanmedian(density[valid] / aunp_density_per_nm2[valid])
        ticks = cbar.get_ticks()
        with np.errstate(divide="ignore"):
            nm2_per_aunp = density_to_aunp_density / ticks
        cbar.set_ticklabels([f"{v:.0f}" if np.isfinite(v) else "∞" for v in nm2_per_aunp])

        out_png = args.output_dir / f"{tomo_path.name}_{az_name}_packing_density.png"
        fig.savefig(out_png, dpi=450, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {out_png}", flush=True)


if __name__ == "__main__":
    main()
