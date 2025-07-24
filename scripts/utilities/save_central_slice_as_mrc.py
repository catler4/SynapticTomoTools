import mrcfile
import sys
import os
import numpy as np

# Usage: python save_central_slice_as_mrc.py input_tomogram.mrc output_slice.mrc

def main():
    if len(sys.argv) != 3:
        print("Usage: python save_central_slice_as_mrc.py input_tomogram.mrc output_slice.mrc")
        sys.exit(1)
    input_path = sys.argv[1]
    output_path = sys.argv[2]

    # Load the tomogram
    with mrcfile.open(input_path, permissive=True) as mrc:
        data = mrc.data
        central_idx = data.shape[0] // 2
        central_slice = data[central_idx]

    # Save the central slice as a new .mrc file (single slice)
    with mrcfile.new(output_path, overwrite=True) as out_mrc:
        out_mrc.set_data(central_slice[np.newaxis, ...].astype('float32'))
        if hasattr(mrc, 'voxel_size') and mrc.voxel_size is not None:
            out_mrc.voxel_size = mrc.voxel_size

    print(f"Central slice saved to: {output_path}")

if __name__ == "__main__":
    main() 