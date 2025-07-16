"""
Tomo Viewer GUI

Requirements:
    pip install pillow imageio opencv-python

Usage:
    python scripts/tomo_viewer_gui.py
"""
import os
import glob
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import imageio
import threading
import time
import csv

# --- CONFIG ---
ALL_SNAPSHOTS = "results/all_snapshots"
VISUALIZATIONS = "results/visualizations"
MOVIE_EXT = ".mp4"
SNAPSHOT_EXT = ".png"
COMBINED_SUFFIX = "_combined.png"

class Tomogram:
    def __init__(self, set_name, base_name, movie_path, snapshot_path, viz_path):
        self.set_name = set_name
        self.base_name = base_name
        self.movie_path = movie_path
        self.snapshot_path = snapshot_path
        self.viz_path = viz_path


def get_tomoname_bases():
    bases = set()
    with open("data/tomograms.csv", newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            bases.add(row["tomoname"])
    return sorted(bases, key=len, reverse=True)  # longest first to avoid partial matches

def find_tomograms():
    tomos = []
    tomoname_bases = get_tomoname_bases()
    for set_dir in sorted(os.listdir(ALL_SNAPSHOTS)):
        set_path = os.path.join(ALL_SNAPSHOTS, set_dir)
        if not os.path.isdir(set_path):
            continue
        for root, dirs, files in os.walk(set_path):
            for file in sorted(files):
                if file.endswith(MOVIE_EXT):
                    base = file[:-len(MOVIE_EXT)]
                    movie_path = os.path.join(root, file)
                    snapshot_path = os.path.join(root, base + SNAPSHOT_EXT)
                    # Find the matching tomogram base name
                    matched_base = None
                    for tbase in tomoname_bases:
                        if tbase in movie_path:
                            matched_base = tbase
                            break
                    if matched_base is None:
                        print(f"WARNING: No tomogram base found in {movie_path}")
                        continue
                    viz_path = os.path.join(VISUALIZATIONS, matched_base + COMBINED_SUFFIX)
                    if os.path.exists(viz_path):
                        print(f"FOUND combined visualization: {viz_path}")
                    else:
                        print(f"NOT FOUND: {viz_path}")
                        viz_path = None
                    if not os.path.exists(snapshot_path):
                        snapshot_path = None
                    tomos.append(Tomogram(set_dir, base, movie_path, snapshot_path, viz_path))
    return tomos

class TomoViewer(tk.Tk):
    def __init__(self, tomograms):
        super().__init__()
        self.title("Tomogram Movie & Snapshot Viewer")
        self.tomograms = tomograms
        self.idx = 0
        # Remove playing state and movie_thread
        # self.playing = False
        # self.movie_thread = None
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.build_ui()
        self.show_tomogram(0)

    def build_ui(self):
        # Controls
        ctrl_frame = ttk.Frame(self)
        ctrl_frame.pack(side=tk.TOP, fill=tk.X)
        self.prev_btn = ttk.Button(ctrl_frame, text="Previous", command=self.prev_tomo)
        self.prev_btn.pack(side=tk.LEFT)
        self.next_btn = ttk.Button(ctrl_frame, text="Next", command=self.next_tomo)
        self.next_btn.pack(side=tk.LEFT)
        self.tomo_label = ttk.Label(ctrl_frame, text="")
        self.tomo_label.pack(side=tk.LEFT, padx=10)
        # Main display
        disp_frame = ttk.Frame(self)
        disp_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        # Snapshot (now on the left)
        snap_frame = ttk.LabelFrame(disp_frame, text="Center Slice Snapshot (PNG)")
        snap_frame.grid(row=0, column=0, padx=5, pady=5)
        self.snap_canvas = tk.Label(snap_frame)
        self.snap_canvas.pack()
        # Movie (now on the right)
        movie_frame = ttk.LabelFrame(disp_frame, text="Movie (MP4)")
        movie_frame.grid(row=0, column=1, padx=5, pady=5)
        self.movie_canvas = tk.Label(movie_frame)
        self.movie_canvas.pack()
        self.movie_slider = tk.Scale(movie_frame, from_=0, to=0, orient=tk.HORIZONTAL, command=self.on_slider_move, length=256)
        self.movie_slider.pack(fill=tk.X)
        # Visualization (remains below, larger)
        viz_frame = ttk.LabelFrame(disp_frame, text="Combined Visualization")
        viz_frame.grid(row=1, column=0, columnspan=2, padx=5, pady=5)
        self.viz_canvas = tk.Label(viz_frame)
        self.viz_canvas.pack()

    def show_tomogram(self, idx):
        if not self.tomograms:
            self.tomo_label.config(text="No tomograms found.")
            return
        self.idx = idx % len(self.tomograms)
        tomo = self.tomograms[self.idx]
        self.tomo_label.config(text=f"{tomo.set_name} / {tomo.base_name}")
        # Show snapshot
        if tomo.snapshot_path:
            img = Image.open(tomo.snapshot_path)
            img = img.resize((256, 256))
            self.snap_img = ImageTk.PhotoImage(img)
            self.snap_canvas.config(image=self.snap_img)
        else:
            self.snap_canvas.config(image="", text="No snapshot")
        # Show visualization (now larger)
        if tomo.viz_path:
            img = Image.open(tomo.viz_path)
            img = img.resize((512, 512))
            self.viz_img = ImageTk.PhotoImage(img)
            self.viz_canvas.config(image=self.viz_img)
        else:
            self.viz_canvas.config(image="", text="No visualization")
        # Prepare movie
        self.movie_reader = imageio.get_reader(tomo.movie_path)
        self.movie_frames = [ImageTk.PhotoImage(Image.fromarray(frame).resize((256, 256))) for frame in self.movie_reader]
        self.movie_canvas.config(image=self.movie_frames[0])
        self.movie_idx = 0
        # Update slider
        self.movie_slider.config(to=len(self.movie_frames)-1)
        self.movie_slider.set(0)

    def prev_tomo(self):
        # self.playing = False
        self.show_tomogram((self.idx - 1) % len(self.tomograms))

    def next_tomo(self):
        # self.playing = False
        self.show_tomogram((self.idx + 1) % len(self.tomograms))

    def on_slider_move(self, val):
        idx = int(float(val))
        if hasattr(self, 'movie_frames') and self.movie_frames:
            self.movie_idx = idx
            self.movie_canvas.config(image=self.movie_frames[self.movie_idx])

    def on_close(self):
        # self.playing = False
        self.destroy()

if __name__ == "__main__":
    tomos = find_tomograms()
    if not tomos:
        print("No tomograms found in results/all_snapshots.")
    else:
        app = TomoViewer(tomos)
        app.mainloop() 