import mrcfile
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
from segment_anything import sam_model_registry, SamPredictor
import os
from tqdm import tqdm

# --- CONFIGURATION ---
MRC_PATH = "data/unlabeled_tomograms/TOP_TOMOS/20250228_AMmilled25-2_Position_1_7/best_alignment/Position_1_7_full_rec_BP_3DCTF_BIN4_ddw.mrc"
SAM_CHECKPOINT = "sam_vit_h_4b8939.pth"  # Path to your downloaded SAM checkpoint
MODEL_TYPE = "vit_h"  # or "vit_l", "vit_h" if you have those checkpoints
OUTPUT_DIR = "results/segment_extracellular_sam/"

# --- LOAD CENTRAL SLICE ---
with mrcfile.open(MRC_PATH) as mrc:
    data = mrc.data
central_idx = data.shape[0] // 2
central_slice = data[central_idx]
# Normalize for display
img = ((central_slice - central_slice.min()) / np.ptp(central_slice) * 255).astype(np.uint8)
img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

# --- LOAD SAM ---
device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry[MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
sam.to(device)
predictor = SamPredictor(sam)
predictor.set_image(img_rgb)

# --- ITERATIVE SEGMENTATION ---
all_points = []
all_labels = []
current_mask = None

while True:
    print("INSTRUCTIONS:\n- Click inside the EXTRACELLULAR region(s) (label=1, green)\n- Then, right-click to select INTRACELLULAR region(s) (label=0, red) if desired.\n- Close the window when done.\n\nFor best results, select both extracellular and intracellular points.")
    # Show current overlay if available
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.imshow(img, cmap='gray')
    if current_mask is not None:
        overlay = img_rgb.copy()
        overlay[current_mask] = [0, 255, 0]
        alpha = 0.5
        vis = cv2.addWeighted(overlay, alpha, img_rgb, 1 - alpha, 0)
        ax.imshow(vis, alpha=0.5)
    ax.set_title("Click: extracellular=left, intracellular=right, then close window")
    print("Left click: extracellular (label=1, green). Right click: intracellular (label=0, red). Close window when done.")

    coords = []
    labels = []
    def onclick(event):
        if event.xdata is None or event.ydata is None:
            return
        if event.button == 1:  # Left click
            coords.append([event.xdata, event.ydata])
            labels.append(1)
            ax.plot(event.xdata, event.ydata, 'go')
            fig.canvas.draw()
        elif event.button == 3:  # Right click
            coords.append([event.xdata, event.ydata])
            labels.append(0)
            ax.plot(event.xdata, event.ydata, 'ro')
            fig.canvas.draw()

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    fig.canvas.mpl_disconnect(cid)

    if not coords:
        if not all_points:
            print("No points selected. Exiting.")
            exit()
        else:
            print("No new points selected. Finishing segmentation.")
            break

    all_points.extend(coords)
    all_labels.extend(labels)
    input_points = np.array(all_points)
    input_labels = np.array(all_labels)
    print(f"Selected {len(input_points)} total points. Running SAM prediction...")

    # --- RUN SAM PREDICTION WITH PROGRESS BAR ---
    num_masks = 3  # SAM default multimask_output=True returns 3 masks
    masks = []
    scores = []
    logits = []
    for i in tqdm(range(num_masks), desc="SAM prediction"):
        mask, score, logit = predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=True,
        )
        masks.append(mask[i])
        scores.append(score[i])
        logits.append(logit[i])

    masks = np.array(masks)
    scores = np.array(scores)
    logits = np.array(logits)
    current_mask = masks[np.argmax(scores)]

    # Show overlay after this round
    fig2, ax2 = plt.subplots(figsize=(7.5, 7.5))
    overlay = img_rgb.copy()
    overlay[current_mask] = [0, 255, 0]
    alpha = 0.5
    vis = cv2.addWeighted(overlay, alpha, img_rgb, 1 - alpha, 0)
    ax2.imshow(vis)
    ax2.set_title("Current segmentation overlay. Close to continue.")
    plt.show()

    # Ask user if they want to continue
    resp = input("Add more points? (y/n): ").strip().lower()
    if resp != 'y':
        break

# --- SAVE FINAL MASK AND OVERLAY ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
base = os.path.splitext(os.path.basename(MRC_PATH))[0]
mask_path = os.path.join(OUTPUT_DIR, f"{base}_extracellular_mask.png")
overlay_path = os.path.join(OUTPUT_DIR, f"{base}_extracellular_overlay.png")
cv2.imwrite(mask_path, (current_mask * 255).astype(np.uint8))
cv2.imwrite(overlay_path, vis)

# Also save the mask as a .mrc file for ML training
mrc_mask_path = os.path.join(OUTPUT_DIR, f"{base}_extracellular_mask.mrc")
import mrcfile
with mrcfile.new(mrc_mask_path, overwrite=True) as mrc_out:
    mrc_out.set_data(current_mask[np.newaxis, ...].astype('float32'))

# Report percent extracellular
extracellular_pixels = np.count_nonzero(current_mask)
total_pixels = current_mask.size
percent_extracellular = 100.0 * extracellular_pixels / total_pixels
print(f"Extracellular mask saved to: {mask_path}")
print(f"Overlay saved to: {overlay_path}")
print(f"MRC mask saved to: {mrc_mask_path}")
print(f"Percent of central slice segmented as extracellular: {percent_extracellular:.2f}%") 