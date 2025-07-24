import mrcfile
import matplotlib.pyplot as plt
import numpy as np
import imageio.v2 as imageio
import io
import sys
import os
import csv
from pathlib import Path


def mrc_to_mp4(mrc_path, fps=10, output_dir=None):
    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), 'all_snapshots')
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(mrc_path))[0]
    output_mp4_path = os.path.join(output_dir, base_name + '.mp4')
    output_png_path = os.path.join(output_dir, base_name + '.png')

    # Skip if both outputs already exist
    if os.path.exists(output_mp4_path) and os.path.exists(output_png_path):
        print(f"Skipping {mrc_path}: outputs already exist.")
        return

    try:
        with mrcfile.open(mrc_path, permissive=True) as mrc:
            data = mrc.data
            if data.ndim != 3:
                print(f"Error: {mrc_path} is not 3D.", file=sys.stderr)
                return
    except Exception as e:
        print(f"Error opening {mrc_path}: {e}", file=sys.stderr)
        return

    frames = []
    vmin, vmax = np.percentile(data, [2, 98])

    # Save center slice as PNG
    center_idx = data.shape[0] // 2
    fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
    ax.imshow(data[center_idx], cmap='gray', vmin=vmin, vmax=vmax)
    ax.axis('off')
    plt.savefig(output_png_path, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print(f"Center slice snapshot saved to {output_png_path}")

    # Create movie frames
    for i in range(data.shape[0]):
        fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
        ax.imshow(data[i], cmap='gray', vmin=vmin, vmax=vmax)
        ax.axis('off')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        frame = imageio.imread(buf)
        frames.append(frame)
        buf.close()

    imageio.mimsave(output_mp4_path, frames, fps=10, codec='libx264')
    print(f"Movie saved to {output_mp4_path}")

# SET_ROOTS mapping as in cli.py
SET_ROOTS = {
    "15F1": Path("/Users/andecath/Documents/GitHub/SynapticTomoTools/data/15F1_tomograms/TOP_TOMOS"),
    "unlabeled": Path("/Users/andecath/Documents/GitHub/SynapticTomoTools/data/unlabeled_tomograms/TOP_TOMOS"),
    # Add more sets here if needed
}

CSV_PATH = "data/tomograms-test.csv"
ALL_SNAPSHOTS = "results/all_snapshots"

def main():
    print("Batch processing started.")
    with open(CSV_PATH, newline='') as csvfile:
        reader = list(csv.reader(csvfile))
        print(f"Found {len(reader)} rows in CSV.")
        header = reader[0]
        # Find column indices
        try:
            tomoname_idx = header.index("tomoname")
            set_idx = header.index("set")
        except ValueError:
            print("CSV must have 'tomoname' and 'set' columns.", file=sys.stderr)
            return
        for i, row in enumerate(reader[1:], 1):
            print(f"Row: {row}")
            tomoname = row[tomoname_idx].strip()
            tomoset = row[set_idx].strip()
            root = SET_ROOTS.get(tomoset)
            if root is None:
                print(f"No root path defined for set: {tomoset}", file=sys.stderr)
                continue
            tomo_dir = root / tomoname / "best_alignment"
            mrc_files = list(tomo_dir.glob("*ddw.mrc"))
            if not mrc_files:
                print(f"No *ddw.mrc file found in {tomo_dir}", file=sys.stderr)
                continue
            mrc_path = str(mrc_files[0])
            output_dir = os.path.join(ALL_SNAPSHOTS, tomoset)
            print(f"Processing {mrc_path} into {output_dir} ...")
            mrc_to_mp4(mrc_path, fps=10, output_dir=output_dir)

if __name__ == "__main__":
    main() 