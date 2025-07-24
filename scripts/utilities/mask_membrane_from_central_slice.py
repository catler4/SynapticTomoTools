import mrcfile
import numpy as np
import sys
import os

# Usage: python mask_membrane_from_central_slice.py input_tomogram.mrc membrane_mask.mrc output_slice_masked.mrc

def main():
    if len(sys.argv) != 4:
        print("Usage: python mask_membrane_from_central_slice.py input_tomogram.mrc membrane_mask.mrc output_slice_masked.mrc")
        sys.exit(1)
    input_path = sys.argv[1]
    mask_path = sys.argv[2]
    output_path = sys.argv[3]

    # Load the tomogram and get central slice
    with mrcfile.open(input_path, permissive=True) as mrc:
        data = mrc.data
        central_idx = data.shape[0] // 2
        central_slice = data[central_idx]

    # Load the membrane mask (assume same shape as central_slice or as the tomogram)
    with mrcfile.open(mask_path, permissive=True) as mask_mrc:
        mask_data = mask_mrc.data
        if mask_data.ndim == 3:
            mask_central = mask_data[central_idx]
        else:
            mask_central = mask_data

    # Binarize mask if needed
    mask_central = (mask_central > 0)

    # Mask out membrane pixels (set to 0)
    masked_slice = np.copy(central_slice)
    masked_slice[mask_central] = 0

    # Save the masked central slice as a new .mrc file
    with mrcfile.new(output_path, overwrite=True) as out_mrc:
        out_mrc.set_data(masked_slice[np.newaxis, ...].astype('float32'))

    print(f"Masked central slice saved to: {output_path}")

if __name__ == "__main__":
    main() 