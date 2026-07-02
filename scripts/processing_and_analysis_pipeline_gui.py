import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk
import subprocess
import threading
import os
import sys
import socket
import webbrowser
import pandas as pd
from pathlib import Path
import shutil
from datetime import datetime

# Get the repository root directory (parent of scripts/)
REPO_ROOT = Path(__file__).parent.parent
FIG_HOME = REPO_ROOT / "figures" / "synaptictomotools_fig_gui_home-01.png"
FIG_AZ = REPO_ROOT / "figures" / "synaptictomotools_fig_gui_AZ-01.png"
FIG_VESICLES = REPO_ROOT / "figures" / "synaptictomotools_fig_gui_vesicles-01.png"
FIG_AUNPS = REPO_ROOT / "figures" / "synaptictomotools_fig_gui_aunps-01.png"
FIG_POSES = REPO_ROOT / "figures" / "synaptictomotools_fig_gui_poses-01.png"

class ToolTip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tipwindow = None
        widget.bind("<Enter>", self.show_tip)
        widget.bind("<Leave>", self.hide_tip)
    def show_tip(self, event=None):
        if self.tipwindow or not self.text:
            return
        x, y, cx, cy = self.widget.bbox("insert") if hasattr(self.widget, 'bbox') else (0,0,0,0)
        x = x + self.widget.winfo_rootx() + 25
        y = y + self.widget.winfo_rooty() + 20
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(tw, text=self.text, justify=tk.LEFT,
                         background="#ffffe0", relief=tk.SOLID, borderwidth=1,
                         font=("tahoma", "10", "normal"))
        label.pack(ipadx=1)
    def hide_tip(self, event=None):
        tw = self.tipwindow
        self.tipwindow = None
        if tw:
            tw.destroy()

class AnalysisPipelineGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Synaptic TomoTools Analysis Pipeline GUI")
        self.geometry("900x900")
        self.csv_path = tk.StringVar()
        self.root_dir = tk.StringVar()
        self.start_tomogram = tk.StringVar()
        self.log_text = None
        self._session_log_fp = None
        self._session_log_path = None
        self._img_refs = []  # Keep references to PhotoImage objects
        self._current_process = None  # Track running process for stopping
        
        # Initialize parameter variables with defaults
        self.use_custom_params = tk.BooleanVar(value=False)
        # Active zone parameters
        self.az_distance_min = tk.StringVar(value="10.0")
        self.az_distance_max = tk.StringVar(value="40.0")
        # Vesicle analysis parameters
        self.vesicle_distance_threshold = tk.StringVar(value="20.0")
        # AuNP analysis parameters
        self.dbscan_eps = tk.StringVar(value="16.0")
        self.dbscan_min_samples = tk.StringVar(value="1")
        # Visualization parameters
        self.sphere_size = tk.StringVar(value="36")
        self.sphere_color = tk.StringVar(value="gold")
        self.aunp_distance_min = tk.StringVar(value="")
        self.aunp_distance_max = tk.StringVar(value="")
        self.aunp_distance_cutoff_direction = tk.StringVar(value="below")
        self.aunp_distance_cutoff_value = tk.StringVar(value="15")
        self.cylinder_radius = tk.StringVar(value="25.0")
        self.receptor_crosssection = tk.StringVar(value="122.0")
        self.aunps_per_receptor = tk.StringVar(value="2.0")
        self.vertex_sampling_step = tk.StringVar(value="50")
        self.synaptic_designation_cutoff = tk.StringVar(value="30.0")
        self.min_cluster_size = tk.StringVar(value="4")
        self.fusion_point_threshold = tk.StringVar(value="20.0")
        self.fusing_perimeter_threshold = tk.StringVar(value="1.0")
        self.aunp_pick_star_pattern = tk.StringVar(value="")
        self.run_fusion_point_aunp_analyses = tk.BooleanVar(value=False)
        self.run_aunp_vs_az_center_ripley = tk.BooleanVar(value=False)
        self.run_aunp_monomer_dimer_ripley = tk.BooleanVar(value=False)
        self.monomer_star_pattern = tk.StringVar(value="")
        self.dimer_star_pattern = tk.StringVar(value="")
        
        self._build_tabs()
        self.protocol("WM_DELETE_WINDOW", self._on_app_close)

    def _open_gui_session_log(self):
        """Create results/logs and open a timestamped session log file."""
        logs_dir = REPO_ROOT / "results" / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        pid = os.getpid()
        csv_part = ""
        try:
            csv_p = self.csv_path.get().strip()
            if csv_p:
                stem = Path(csv_p).stem.replace(" ", "_")
                for ch in '<>:"/\\|?*':
                    stem = stem.replace(ch, "_")
                csv_part = "_" + stem[:80]
        except Exception:
            pass
        name = f"gui_session_{ts}_pid{pid}{csv_part}.log"
        path = logs_dir / name
        self._session_log_path = path
        started = datetime.now().isoformat(timespec="seconds")
        self._session_log_fp = open(path, "w", encoding="utf-8", newline="\n")
        host = socket.gethostname()
        self._session_log_fp.write("=== Synaptic TomoTools Analysis Pipeline GUI ===\n")
        self._session_log_fp.write(f"Session started: {started}\n")
        self._session_log_fp.write(f"Host: {host}\n")
        self._session_log_fp.write(f"Python: {sys.version.split()[0]}  pid={pid}\n")
        self._session_log_fp.write(f"Repo root: {REPO_ROOT}\n")
        self._session_log_fp.write(f"Working directory: {os.getcwd()}\n")
        csv_now = self.csv_path.get().strip() if self.csv_path.get() else "(not set yet)"
        root_now = self.root_dir.get().strip() if self.root_dir.get() else "(not set yet)"
        self._session_log_fp.write(f"CSV path: {csv_now}\n")
        self._session_log_fp.write(f"Tomogram root: {root_now}\n")
        self._session_log_fp.write("--- Log output (same as GUI) ---\n")
        self._session_log_fp.flush()

    def _close_gui_session_log(self):
        if self._session_log_fp is not None:
            try:
                self._session_log_fp.write(
                    f"\n=== Session ended: {datetime.now().isoformat(timespec='seconds')} ===\n"
                )
                self._session_log_fp.flush()
                self._session_log_fp.close()
            except Exception:
                pass
            self._session_log_fp = None

    def _on_app_close(self):
        self._close_gui_session_log()
        self.destroy()

    def _build_home_tab_content(self, tab):
        # Create a scrollable frame
        # Create canvas and scrollbar
        canvas = tk.Canvas(tab, highlightthickness=0)
        self._home_canvas = canvas  # Store reference for scroll region updates
        scrollbar = ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        def configure_scroll_region(event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))
        
        scrollable_frame.bind("<Configure>", configure_scroll_region)
        
        canvas_window = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        
        def configure_canvas_width(event):
            canvas_width = event.width
            canvas.itemconfig(canvas_window, width=canvas_width)
        
        canvas.bind('<Configure>', configure_canvas_width)
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Bind mousewheel to canvas (works on Windows and Mac)
        def _on_mousewheel(event):
            # For Windows and Mac
            if event.delta:
                canvas.yview_scroll(int(-1*(event.delta/120)), "units")
            # For Linux
            elif event.num == 4:
                canvas.yview_scroll(-1, "units")
            elif event.num == 5:
                canvas.yview_scroll(1, "units")
        
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        canvas.bind_all("<Button-4>", _on_mousewheel)
        canvas.bind_all("<Button-5>", _on_mousewheel)
        
        # Use scrollable_frame as the main frame
        frame = scrollable_frame
        # Home figure
        img = self._load_and_display_image(FIG_HOME, frame, max_width=700, max_height=320)
        if img:
            img_label = ttk.Label(frame, image=img)
            img_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
            self._img_refs.append(img)
        # CSV selection
        ttk.Label(frame, text="Tomogram CSV file:").grid(row=1, column=0, sticky=tk.W)
        csv_entry = ttk.Entry(frame, textvariable=self.csv_path, width=40)
        csv_entry.grid(row=1, column=1, sticky=tk.W)
        ttk.Button(frame, text="Browse...", command=self._browse_csv).grid(row=1, column=2, padx=5)
        # Root directory selection
        ttk.Label(frame, text="Root directory for tomogram sets:").grid(row=2, column=0, sticky=tk.W)
        root_entry = ttk.Entry(frame, textvariable=self.root_dir, width=40)
        root_entry.grid(row=2, column=1, sticky=tk.W)
        ttk.Button(frame, text="Browse...", command=self._browse_root).grid(row=2, column=2, padx=5)
        
        # Starting tomogram selection (alignment_dir comes only from the CSV)
        ttk.Label(frame, text="Processing mode:").grid(row=3, column=0, sticky=tk.W)
        self.processing_mode = tk.StringVar(value="All tomograms")
        self.processing_mode_combo = ttk.Combobox(frame, textvariable=self.processing_mode, width=37, state="readonly")
        self.processing_mode_combo.grid(row=3, column=1, sticky=tk.W)
        ttk.Button(frame, text="Load CSV", command=self._load_tomograms_from_csv).grid(row=3, column=2, padx=5)
        
        # Starting tomogram selection (initially hidden)
        self.start_tomogram_label = ttk.Label(frame, text="Start from tomogram:")
        self.start_tomogram_combo = ttk.Combobox(frame, textvariable=self.start_tomogram, width=37, state="readonly")
        
        # Add tooltip for the dropdown
        ToolTip(self.processing_mode_combo, "Select processing mode: 'All tomograms' for entire CSV, 'Single tomogram' for one specific tomogram, or 'Start from' to process from a specific tomogram onwards")
        ToolTip(frame.grid_slaves(row=3, column=2)[0], "Load tomogram names from the selected CSV file into the dropdown")
        
        # Bind the mode combo to show/hide the starting tomogram selection
        self.processing_mode_combo.bind('<<ComboboxSelected>>', self._on_processing_mode_change)
        
        # Add a separator at a fixed position
        separator = ttk.Separator(frame, orient='horizontal')
        separator.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        # Analysis Parameters section
        ttk.Label(frame, text="Analysis Parameters:", font=('TkDefaultFont', 10, 'bold')).grid(row=6, column=0, columnspan=3, sticky=tk.W, pady=(10, 5))
        
        # Parameter mode switch
        params_mode_frame = ttk.Frame(frame)
        params_mode_frame.grid(row=7, column=0, columnspan=3, sticky=tk.W, pady=5)
        ttk.Label(params_mode_frame, text="Parameter mode:").pack(side=tk.LEFT, padx=(0, 10))
        ttk.Radiobutton(params_mode_frame, text="Default", variable=self.use_custom_params, value=False, 
                       command=self._toggle_custom_params).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(params_mode_frame, text="Custom", variable=self.use_custom_params, value=True,
                       command=self._toggle_custom_params).pack(side=tk.LEFT, padx=5)
        
        # Custom parameters frame (initially hidden)
        self.custom_params_frame = ttk.LabelFrame(frame, text="Custom Parameters", padding=10)
        self.custom_params_frame.grid(row=8, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5, padx=0)
        
        # Active zone parameters
        az_frame = ttk.LabelFrame(self.custom_params_frame, text="Active Zone Parameters", padding=5)
        az_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        ttk.Label(az_frame, text="Distance range (nm):").grid(row=0, column=0, sticky=tk.W, padx=5)
        az_min_entry = ttk.Entry(az_frame, textvariable=self.az_distance_min, width=8)
        az_min_entry.grid(row=0, column=1, padx=2)
        ttk.Label(az_frame, text="to").grid(row=0, column=2, padx=2)
        az_max_entry = ttk.Entry(az_frame, textvariable=self.az_distance_max, width=8)
        az_max_entry.grid(row=0, column=3, padx=2)
        ttk.Label(az_frame, text="(default: 10.0 - 40.0)").grid(row=0, column=4, padx=5, sticky=tk.W)
        
        # Vesicle analysis parameters
        vesicle_frame = ttk.LabelFrame(self.custom_params_frame, text="Vesicle Analysis Parameters", padding=5)
        vesicle_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        ttk.Label(vesicle_frame, text="Close vesicle membrane distance cutoff (nm):").grid(row=0, column=0, sticky=tk.W, padx=5)
        vesicle_threshold_entry = ttk.Entry(vesicle_frame, textvariable=self.vesicle_distance_threshold, width=8)
        vesicle_threshold_entry.grid(row=0, column=1, padx=5)
        ttk.Label(vesicle_frame, text="(default: 20.0)").grid(row=0, column=2, padx=5, sticky=tk.W)
        ttk.Label(vesicle_frame, text="Fusing vesicle membrane distance cutoff (nm):").grid(row=1, column=0, sticky=tk.W, padx=5)
        fusing_perimeter_threshold_entry = ttk.Entry(vesicle_frame, textvariable=self.fusing_perimeter_threshold, width=8)
        fusing_perimeter_threshold_entry.grid(row=1, column=1, padx=5)
        ttk.Label(vesicle_frame, text="(default: 1.0)").grid(row=1, column=2, padx=5, sticky=tk.W)
        
        # AuNP analysis parameters
        aunp_frame = ttk.LabelFrame(self.custom_params_frame, text="AuNP Analysis Parameters", padding=5)
        aunp_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        ttk.Label(aunp_frame, text="DBSCAN eps (nm):").grid(row=0, column=0, sticky=tk.W, padx=5)
        dbscan_eps_entry = ttk.Entry(aunp_frame, textvariable=self.dbscan_eps, width=8)
        dbscan_eps_entry.grid(row=0, column=1, padx=5)
        ttk.Label(aunp_frame, text="(default: 16.0)").grid(row=0, column=2, padx=5, sticky=tk.W)
        ttk.Label(aunp_frame, text="DBSCAN min_samples:").grid(row=1, column=0, sticky=tk.W, padx=5)
        dbscan_min_samples_entry = ttk.Entry(aunp_frame, textvariable=self.dbscan_min_samples, width=8)
        dbscan_min_samples_entry.grid(row=1, column=1, padx=5)
        ttk.Label(aunp_frame, text="(default: 1)").grid(row=1, column=2, padx=5, sticky=tk.W)
        ttk.Label(aunp_frame, text="Synaptic designation cutoff (nm from postsynaptic outer membrane):").grid(row=2, column=0, sticky=tk.W, padx=5)
        synaptic_cutoff_entry = ttk.Entry(aunp_frame, textvariable=self.synaptic_designation_cutoff, width=8)
        synaptic_cutoff_entry.grid(row=2, column=1, padx=5)
        ttk.Label(aunp_frame, text="(default: 30.0)").grid(row=2, column=2, padx=5, sticky=tk.W)
        ttk.Label(aunp_frame, text="Minimum cluster size (post-DBSCAN):").grid(row=3, column=0, sticky=tk.W, padx=5)
        min_cluster_size_entry = ttk.Entry(aunp_frame, textvariable=self.min_cluster_size, width=8)
        min_cluster_size_entry.grid(row=3, column=1, padx=5)
        ttk.Label(aunp_frame, text="(default: 4)").grid(row=3, column=2, padx=5, sticky=tk.W)
        ttk.Label(aunp_frame, text="Fusion point threshold (nm):").grid(row=4, column=0, sticky=tk.W, padx=5)
        fusion_point_threshold_entry = ttk.Entry(aunp_frame, textvariable=self.fusion_point_threshold, width=8)
        fusion_point_threshold_entry.grid(row=4, column=1, padx=5)
        ttk.Label(aunp_frame, text="(default: 20.0)").grid(row=4, column=2, padx=5, sticky=tk.W)
        ttk.Label(aunp_frame, text="AuNP pick STAR filename pattern:").grid(row=5, column=0, sticky=tk.W, padx=5)
        aunp_pick_star_pattern_entry = ttk.Entry(
            aunp_frame, textvariable=self.aunp_pick_star_pattern, width=42
        )
        aunp_pick_star_pattern_entry.grid(row=5, column=1, columnspan=2, padx=5, sticky=tk.W)
        ttk.Label(
            aunp_frame,
            text="(default: aunp_tm_BP_active_zone_*_manual_refined.star; * = AZ index)",
        ).grid(row=6, column=0, columnspan=3, padx=5, sticky=tk.W)
        fusion_aunp_cb = ttk.Checkbutton(
            aunp_frame,
            text="Run fusion-point vs monomer/dimer AuNP analyses (Ripley L₁₂ + distances)",
            variable=self.run_fusion_point_aunp_analyses,
            command=self._toggle_monomer_dimer_star_entries,
        )
        fusion_aunp_cb.grid(row=7, column=0, columnspan=3, sticky=tk.W, padx=5, pady=(4, 0))
        ttk.Label(aunp_frame, text="Monomer STAR filename pattern:").grid(
            row=8, column=0, sticky=tk.W, padx=5
        )
        self._monomer_star_entry = ttk.Entry(
            aunp_frame, textvariable=self.monomer_star_pattern, width=42
        )
        self._monomer_star_entry.grid(row=8, column=1, columnspan=2, padx=5, sticky=tk.W)
        ttk.Label(aunp_frame, text="Dimer STAR filename pattern:").grid(
            row=9, column=0, sticky=tk.W, padx=5
        )
        self._dimer_star_entry = ttk.Entry(
            aunp_frame, textvariable=self.dimer_star_pattern, width=42
        )
        self._dimer_star_entry.grid(row=9, column=1, columnspan=2, padx=5, sticky=tk.W)
        ttk.Label(
            aunp_frame,
            text=(
                "(defaults: *_manual_refined_monomer.star / *_manual_refined_dimer.star; "
                "files in {alignment}/aunps/)"
            ),
        ).grid(row=10, column=0, columnspan=3, padx=5, sticky=tk.W)
        aunp_az_center_cb = ttk.Checkbutton(
            aunp_frame,
            text="Run AuNP vs active zone center Ripley L₁₂",
            variable=self.run_aunp_vs_az_center_ripley,
        )
        aunp_az_center_cb.grid(row=11, column=0, columnspan=3, sticky=tk.W, padx=5, pady=(4, 0))
        monomer_dimer_cb = ttk.Checkbutton(
            aunp_frame,
            text="Run monomer vs dimer AuNP Ripley L₁₂ (label-permutation control)",
            variable=self.run_aunp_monomer_dimer_ripley,
            command=self._toggle_monomer_dimer_star_entries,
        )
        monomer_dimer_cb.grid(row=12, column=0, columnspan=3, sticky=tk.W, padx=5, pady=(4, 0))
        self._toggle_monomer_dimer_star_entries()
        
        # Visualization parameters
        viz_frame = ttk.LabelFrame(self.custom_params_frame, text="Visualization Parameters", padding=5)
        viz_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        ttk.Label(viz_frame, text="Sphere size for active zonogram overlays:").grid(row=0, column=0, sticky=tk.W, padx=5)
        sphere_size_entry = ttk.Entry(viz_frame, textvariable=self.sphere_size, width=8)
        sphere_size_entry.grid(row=0, column=1, padx=5)
        ttk.Label(viz_frame, text="(default: 36)").grid(row=0, column=2, padx=5, sticky=tk.W)
        ttk.Label(viz_frame, text="Sphere color:").grid(row=1, column=0, sticky=tk.W, padx=5)
        # Color dropdown with matplotlib-compatible colors
        color_options = ['gold', 'purple', 'blue', 'red', 'green', 'orange', 'yellow', 
                        'cyan', 'magenta', 'pink', 'brown', 'black', 'white', 'gray', 
                        'navy', 'lime', 'olive', 'teal', 'maroon', 'silver']
        self.sphere_color_combo = ttk.Combobox(viz_frame, textvariable=self.sphere_color, 
                                                values=color_options, width=6, state="readonly")
        self.sphere_color_combo.grid(row=1, column=1, padx=5)
        self.sphere_color_combo.set('gold')  # Set default
        ttk.Label(viz_frame, text="(default: gold)").grid(row=1, column=2, padx=5, sticky=tk.W)
        ttk.Label(viz_frame, text="AuNP distance range for color scale (nm):").grid(row=2, column=0, sticky=tk.W, padx=5)
        aunp_distance_min_entry = ttk.Entry(viz_frame, textvariable=self.aunp_distance_min, width=8)
        aunp_distance_min_entry.grid(row=2, column=1, padx=2)
        ttk.Label(viz_frame, text="to").grid(row=2, column=2, padx=2)
        aunp_distance_max_entry = ttk.Entry(viz_frame, textvariable=self.aunp_distance_max, width=8)
        aunp_distance_max_entry.grid(row=2, column=3, padx=2)
        ttk.Label(viz_frame, text="(default: auto from data)").grid(row=2, column=4, padx=5, sticky=tk.W)
        ttk.Label(viz_frame, text="AuNP distance cutoff filter:").grid(row=3, column=0, sticky=tk.W, padx=5)
        cutoff_direction_combo = ttk.Combobox(viz_frame, textvariable=self.aunp_distance_cutoff_direction, 
                                              values=['below', 'above'], width=6, state="readonly")
        cutoff_direction_combo.grid(row=3, column=1, padx=2)
        cutoff_direction_combo.set('below')
        ttk.Label(viz_frame, text="(nm)").grid(row=3, column=2, padx=2)
        aunp_distance_cutoff_value_entry = ttk.Entry(viz_frame, textvariable=self.aunp_distance_cutoff_value, width=8)
        aunp_distance_cutoff_value_entry.grid(row=3, column=3, padx=2)
        ttk.Label(viz_frame, text="(default: below 15 nm)").grid(row=3, column=4, padx=5, sticky=tk.W)
        ttk.Label(viz_frame, text="Sliding cylinder radius (nm) for packing density heat map:").grid(row=4, column=0, sticky=tk.W, padx=5)
        cylinder_radius_entry = ttk.Entry(viz_frame, textvariable=self.cylinder_radius, width=8)
        cylinder_radius_entry.grid(row=4, column=1, padx=5)
        ttk.Label(viz_frame, text="(default: 25.0)").grid(row=4, column=2, padx=5, sticky=tk.W)
        ttk.Label(viz_frame, text="Receptor cross-sectional area (nm²):").grid(row=5, column=0, sticky=tk.W, padx=5)
        receptor_crosssection_entry = ttk.Entry(viz_frame, textvariable=self.receptor_crosssection, width=8)
        receptor_crosssection_entry.grid(row=5, column=1, padx=5)
        ttk.Label(viz_frame, text="(default: 122.0)").grid(row=5, column=2, padx=5, sticky=tk.W)
        ttk.Label(viz_frame, text="AuNPs per receptor (e.g. cylinder < AuNP pair distance may need 1):").grid(row=6, column=0, sticky=tk.W, padx=5)
        aunps_per_receptor_entry = ttk.Entry(viz_frame, textvariable=self.aunps_per_receptor, width=8)
        aunps_per_receptor_entry.grid(row=6, column=1, padx=5)
        ttk.Label(viz_frame, text="(default: 2.0)").grid(row=6, column=2, padx=5, sticky=tk.W)
        ttk.Label(viz_frame, text="Vertex sampling step (1=all, 50=every 50th):").grid(row=7, column=0, sticky=tk.W, padx=5)
        vertex_sampling_step_entry = ttk.Entry(viz_frame, textvariable=self.vertex_sampling_step, width=8)
        vertex_sampling_step_entry.grid(row=7, column=1, padx=5)
        ttk.Label(viz_frame, text="(default: 50)").grid(row=7, column=2, padx=5, sticky=tk.W)

        # Tooltips for custom parameters
        ToolTip(az_min_entry, "Active-zone minimum pre/post membrane distance in nm (default 10.0).")
        ToolTip(az_max_entry, "Active-zone maximum pre/post membrane distance in nm (default 40.0).")
        ToolTip(vesicle_threshold_entry, "Close vesicle membrane distance cutoff in nm (default 20.0).")
        ToolTip(dbscan_eps_entry, "DBSCAN epsilon in nm for AuNP clustering neighborhood radius (default 16.0).")
        ToolTip(dbscan_min_samples_entry, "DBSCAN min_samples core-point threshold (default 1).")
        ToolTip(synaptic_cutoff_entry, "AuNP is labeled synaptic if distance to postsynaptic active outer membrane is <= this cutoff in nm (default 30.0).")
        ToolTip(min_cluster_size_entry, "Post-DBSCAN minimum retained cluster size; smaller clusters are reassigned to noise (default 4).")
        ToolTip(fusion_point_threshold_entry, "Radius in nm for active-zone points contributing to per-vesicle fusion point estimation (default 20.0).")
        ToolTip(fusing_perimeter_threshold_entry, "Vesicle is fusing if minimum distance from original vesicle segmentation points to presynaptic active-zone points is <= this threshold (default 1.0 nm).")
        ToolTip(
            aunp_pick_star_pattern_entry,
            "Filename pattern for per-active-zone AuNP pick STAR files under {alignment}/aunps/. "
            "Use exactly one * for the active zone number (e.g. aunps_other_name_*_etc.star). "
            "Leave empty for the default aunp_tm_BP_active_zone_*_manual_refined.star.",
        )
        ToolTip(
            fusion_aunp_cb,
            "3D fusion-site vs monomer/dimer AuNP distance tables and bivariate Ripley L₁₂. "
            "Disable for datasets without separate monomer/dimer STAR picks.",
        )
        ToolTip(
            self._monomer_star_entry,
            "Filename pattern for per-active-zone monomer AuNP STAR files under {alignment}/aunps/. "
            "Use one * for the active zone index. "
            "Leave empty for aunp_tm_BP_active_zone_*_manual_refined_monomer.star.",
        )
        ToolTip(
            self._dimer_star_entry,
            "Filename pattern for per-active-zone dimer AuNP STAR files under {alignment}/aunps/. "
            "Use one * for the active zone index. "
            "Leave empty for aunp_tm_BP_active_zone_*_manual_refined_dimer.star.",
        )
        ToolTip(
            aunp_az_center_cb,
            "3D bivariate Ripley L₁₂ of AuNP positions relative to each active zone center "
            "(uses existing pick STAR files; no null model).",
        )
        ToolTip(
            monomer_dimer_cb,
            "3D bivariate Ripley L₁₂ of monomer vs dimer AuNP positions with a label-permutation "
            "control (1000 replicates). Uses the monomer/dimer STAR patterns above.",
        )
        ToolTip(sphere_size_entry, "Marker size for visualization sphere overlays (default 36).")
        ToolTip(self.sphere_color_combo, "Marker color for visualization sphere overlays (default gold).")
        ToolTip(aunp_distance_min_entry, "Minimum distance (nm) for AuNP color scale; leave empty for auto.")
        ToolTip(aunp_distance_max_entry, "Maximum distance (nm) for AuNP color scale; leave empty for auto.")
        ToolTip(cutoff_direction_combo, "Direction of AuNP distance cutoff filter: keep below or above cutoff value.")
        ToolTip(aunp_distance_cutoff_value_entry, "Distance cutoff value in nm for AuNP visualization filtering.")
        ToolTip(cylinder_radius_entry, "Sliding-cylinder radius in nm for packing-density heat map (default 25.0).")
        ToolTip(receptor_crosssection_entry, "Receptor cross-sectional area in nm² used in packing-density conversion (default 122.0).")
        ToolTip(aunps_per_receptor_entry, "AuNPs per receptor scaling factor for packing density (2=dimer/full receptor, 1=monomer/half receptor).")
        ToolTip(vertex_sampling_step_entry, "Use every Nth mesh vertex for packing-density computation (1=all vertices; default 50).")
        
        # Initially hide custom params frame
        self.custom_params_frame.grid_remove()
        
        # Add separator before Results Management
        separator2 = ttk.Separator(frame, orient='horizontal')
        separator2.grid(row=9, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        # Results management section
        ttk.Label(frame, text="Results Management:", font=('TkDefaultFont', 10, 'bold')).grid(row=10, column=0, columnspan=3, sticky=tk.W, pady=(10, 5))
        
        # Archive current results
        archive_frame = ttk.Frame(frame)
        archive_frame.grid(row=11, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        ttk.Label(archive_frame, text="Archive note:").pack(side=tk.LEFT, padx=(0, 5))
        self.archive_note_var = tk.StringVar()
        archive_entry = ttk.Entry(archive_frame, textvariable=self.archive_note_var, width=30)
        archive_entry.pack(side=tk.LEFT, padx=(0, 5))
        archive_btn = ttk.Button(archive_frame, text="Archive current results", command=self._archive_results)
        archive_btn.pack(side=tk.LEFT)
        ToolTip(archive_btn, "Move all contents of results/ directory to a dated archive directory. Archived results are preserved and not deleted.")
        
        # Delete previous results
        delete_btn = ttk.Button(frame, text="Delete previous results", command=self._delete_previous_results)
        delete_btn.grid(row=12, column=0, columnspan=3, sticky=tk.W, pady=5)
        ToolTip(delete_btn, "Delete results from individual tomogram STT_results directories (for tomograms in CSV) and the results directory. Does NOT delete archived results.")

    def _load_and_display_image(self, path, parent, max_width=400, max_height=180):
        try:
            img = Image.open(path)
            iw, ih = img.size
            scale = min(max_width / iw, max_height / ih, 1.0)
            img = img.resize((int(iw * scale), int(ih * scale)), Image.LANCZOS)
            tk_img = ImageTk.PhotoImage(img)
            return tk_img
        except Exception as e:
            self._log(f"Could not load image {path}: {e}\n")
            return None

    def _build_tabs(self):
        for widget in self.winfo_children():
            widget.destroy()

        self._close_gui_session_log()
        self._open_gui_session_log()
        
        # Create a PanedWindow to allow resizing between tabs and log
        self.paned_window = ttk.PanedWindow(self, orient=tk.VERTICAL)
        self.paned_window.pack(fill=tk.BOTH, expand=True)
        
        # Create the notebook for tabs
        notebook = ttk.Notebook(self.paned_window)
        self.paned_window.add(notebook, weight=2)  # Give more weight to tabs initially
        
        # Home tab
        home_tab = ttk.Frame(notebook)
        notebook.add(home_tab, text="Home")
        self.tabs = {"Home": home_tab}
        self._build_home_tab_content(home_tab)
        
        # FindingAMPA Processing tab as the second tab
        findingampa_tab = ttk.Frame(notebook)
        notebook.add(findingampa_tab, text="FindingAMPA Processing")
        self.tabs["FindingAMPA Processing"] = findingampa_tab
        # Initialize DDW flag variables before use
        self.ddw_flag_var = tk.StringVar(value='k3')
        self.ddw_flag_options = ['k3', 'falcon', 'falconczi']
        # Move this label to the top
        ttk.Label(findingampa_tab, text="Use FindingAMPA processing commands here.").pack(anchor=tk.W, pady=(10, 0), padx=20)
        # --- Single tomogram mode UI ---
        self.findingampa_single_mode = tk.BooleanVar(value=False)
        self.findingampa_all_mode = tk.BooleanVar(value=False)
        self.findingampa_single_dir = tk.StringVar()
        single_frame = ttk.Frame(findingampa_tab)
        single_frame.pack(anchor=tk.W, fill=tk.X, padx=20, pady=(10, 0))
        single_cb = ttk.Checkbutton(single_frame, text="Run on single tomogram only", variable=self.findingampa_single_mode, command=self._toggle_findingampa_single_dir)
        single_cb.pack(side=tk.LEFT)
        # Add 'Run on all tomograms' checkbox below
        all_cb = ttk.Checkbutton(findingampa_tab, text="Run on all tomograms in provided Tomogram CSV file", variable=self.findingampa_all_mode, command=self._toggle_findingampa_all_mode)
        all_cb.pack(anchor=tk.W, padx=20, pady=(2, 10))
        # Directory input widgets (initially hidden)
        self.single_dir_label = ttk.Label(single_frame, text="Tomogram directory:")
        self.single_dir_entry = ttk.Entry(single_frame, textvariable=self.findingampa_single_dir, width=20)
        self.single_dir_browse = ttk.Button(single_frame, text="Browse...", command=self._browse_findingampa_single_dir)
        # --- End single tomogram mode UI ---
        # Add buttons for each FindingAMPA command
        self.findingampa_commands = [
            ("Create Tomograms (Etomo)", "create-tomograms"),
            ("Denoise Tomograms (DeepDeWedge)", "ddw"),
            ("Segment Membranes (membrain-seg)", "annotate-membranes"),
            ("Match AuNPs", "match-aunps"),
            ("Annotate Membranes (Blender plug-in)", "new-annotate-aunps --reset"),
            ("Render Active Zonograms", "render-active-zonograms"),
            ("Select AuNP Picks", "select-aunp-picks"),
        ]
        
        # Tooltips for each FindingAmPA command
        self.findingampa_tooltips = {
            "create-tomograms": "Create tomogram reconstructions using Etomo. The default setting uses weighted backprojection without any filtering to reconstruct tomograms.",
            "ddw": "Apply DeepDeWedge denoising to tomograms. This uses previously trained models to denoise tomograms defined in findingampa pipeline (choose model from dropdown on right).",
            "annotate-membranes": "Segment presynaptic and postsynaptic membranes using MemBrain-seg. This runs on both weighted backprojection and DDW denoised tomograms.",
            "match-aunps": "Run template-matching to identify AuNPs within tomogram.",
            "new-annotate-aunps": "Annotate membranes using the Blender plug-in. This step allows manual cleaning of membrane segmentations and assignment of pre/postsynaptic membranes and presynaptic vesicles.",
            "render-active-zonograms": "Generate active zonogram visualizations. This creates 2D projections showing active zone regions with AuNP distributions.",
            "select-aunp-picks": "Select and refine AuNP picks for analysis. This step runs an automated selection of AuNPs confined to each active zone for further processing."
        }
        
        self.findingampa_check_vars = []
        self.findingampa_btns = []
        for idx, (label, command) in enumerate(self.findingampa_commands):
            row_frame = ttk.Frame(findingampa_tab)
            row_frame.pack(anchor=tk.W, pady=2, padx=20, fill=tk.X)
            var = tk.BooleanVar()
            cb = ttk.Checkbutton(row_frame, variable=var)
            cb.pack(side=tk.LEFT)
            # Add DDW flag option menu next to DDW button
            if command == "ddw":
                btn = ttk.Button(row_frame, text=label, command=lambda c=command: self._run_findingampa_command(c))
                btn.pack(side=tk.LEFT, padx=(5, 0))
                ddw_flag_menu = ttk.OptionMenu(row_frame, self.ddw_flag_var, self.ddw_flag_var.get(), *self.ddw_flag_options)
                ddw_flag_menu.pack(side=tk.LEFT, padx=(5, 0))
                self.findingampa_check_vars.append(var)
                self.findingampa_btns.append(btn)
                # Add tooltip for DDW button
                ToolTip(btn, self.findingampa_tooltips.get(command, "No description available"))
                continue
            # Use specific method for new-annotate-aunps to show popup
            if command == "new-annotate-aunps --reset":
                btn = ttk.Button(row_frame, text=label, command=self._run_findingampa_new_annotate_aunps)
            else:
                btn = ttk.Button(row_frame, text=label, command=lambda c=command: self._run_findingampa_command(c))
            btn.pack(side=tk.LEFT, padx=(5, 0))
            self.findingampa_check_vars.append(var)
            self.findingampa_btns.append(btn)
            # Add tooltip for each button
            ToolTip(btn, self.findingampa_tooltips.get(command, "No description available"))
        
        # Add Run Checked button
        run_checked_btn = ttk.Button(findingampa_tab, text="Run Checked", command=self._run_findingampa_checked)
        run_checked_btn.pack(anchor=tk.W, pady=8, padx=20)
        # Add tooltip for Run Checked button
        ToolTip(run_checked_btn, "Run all checked FindingAmPA processing steps in order from top to bottom. This executes the selected workflow steps sequentially, waiting for each command to complete before starting the next one.")
        # Analysis tabs
        for step in ["Active Zone", "Vesicles", "AuNPs", "Visualization", "Full Pipeline"]:
            tab = ttk.Frame(notebook)
            notebook.add(tab, text=step)
            self.tabs[step] = tab
            self._build_tab_content(tab, step)
        
        # Pose Prediction tab (dedicated tab)
        ampa_poses_tab = ttk.Frame(notebook)
        notebook.add(ampa_poses_tab, text="Pose Prediction")
        self.tabs["Pose Prediction"] = ampa_poses_tab
        self._build_ampa_poses_tab_content(ampa_poses_tab)
        
        # Post-Analysis Tools tab (moved to the far right)
        post_analysis_tab = ttk.Frame(notebook)
        notebook.add(post_analysis_tab, text="Post-Analysis Tools")
        self.tabs["Post-Analysis Tools"] = post_analysis_tab
        self._build_post_analysis_tab_content(post_analysis_tab)
        
        # Log output area with resizable splitter
        self.log_frame = ttk.Frame(self.paned_window)
        ttk.Label(self.log_frame, text="Log Output:").pack(anchor=tk.W)
        if self.log_text is None:
            self.log_text = scrolledtext.ScrolledText(self.log_frame, height=12, state=tk.NORMAL, font=("Courier", 10))
        self.log_text.pack(fill=tk.BOTH, expand=True)

        self._log(f"Session log file: {self._session_log_path}\n")
        
        # Track if log frame is currently in the paned window
        self.log_frame_visible = False
        
        # Show/hide log output based on tab
        def on_tab_change(event):
            tab_text = notebook.tab(notebook.select(), "text")
            if tab_text == "Home":
                # Hide the log frame from the paned window
                if self.log_frame_visible:
                    self.paned_window.forget(self.log_frame)
                    self.log_frame_visible = False
            else:
                # Show the log frame in the paned window if it's not already there
                if not self.log_frame_visible:
                    self.paned_window.add(self.log_frame, weight=1)
                    self.log_frame_visible = True
        notebook.bind("<<NotebookTabChanged>>", on_tab_change)
        
        # Initially show log frame for non-Home tabs
        if notebook.tab(notebook.select(), "text") != "Home":
            self.paned_window.add(self.log_frame, weight=1)
            self.log_frame_visible = True

    # Placeholder methods for FindingAMPA commands
    def _run_findingampa_create_tomograms(self):
        self._run_findingampa_command("create-tomograms")
    def _run_findingampa_ddw(self):
        self._run_findingampa_command("ddw")
    def _run_findingampa_annotate_membranes(self):
        self._run_findingampa_command("annotate-membranes")
    def _run_findingampa_match_aunps(self):
        self._run_findingampa_command("match-aunps")
    def _run_findingampa_new_annotate_aunps(self):
        # Show instructions popup before running
        instructions = """Annotate Membranes (Blender plug-in) Instructions:

1. Blender will open with the tomogram and membrane segmentations (both from WBP and DDW)loaded
2. Choose between WBP and DDW membrane segmentations (hide other)
3. Use the Blender interface to:
   - Clean up membrane segmentations (erase connections and/or adjust tresholding if necessary)
   - Assign pre/postsynaptic membranes (move to relevant group)
   - Assign presynaptic vesicles (move to relevant group)
4. Save your work when complete (Save as new_aunp_template_CJS_ddw.blend)
5. Close Blender to finish the annotation process

Do you want to continue?"""
        
        result = messagebox.askyesno("Blender Annotation Instructions", instructions)
        if result:
            self._run_findingampa_command("new-annotate-aunps --reset")
    def _run_findingampa_render_active_zonograms(self):
        self._run_findingampa_command("render-active-zonograms")
    def _run_findingampa_select_aunp_picks(self):
        self._run_findingampa_command("select-aunp-picks")

    # Add this new method to toggle the directory input
    def _toggle_findingampa_all_mode(self):
        # Make checkboxes mutually exclusive
        if self.findingampa_all_mode.get():
            self.findingampa_single_mode.set(False)
            self._toggle_findingampa_single_dir()
        # If both are unchecked, default to all mode
        elif not self.findingampa_single_mode.get():
            self.findingampa_all_mode.set(True)

    def _toggle_findingampa_single_dir(self):
        if self.findingampa_single_mode.get():
            self.single_dir_label.pack(side=tk.LEFT, padx=(10, 0))
            self.single_dir_entry.pack(side=tk.LEFT, padx=(5, 0))
            self.single_dir_browse.pack(side=tk.LEFT, padx=5)
            self.findingampa_all_mode.set(False)
        else:
            self.single_dir_label.pack_forget()
            self.single_dir_entry.pack_forget()
            self.single_dir_browse.pack_forget()
            # If both are unchecked, default to all mode
            if not self.findingampa_all_mode.get():
                self.findingampa_all_mode.set(True)
    # Add this new method to browse for a directory
    def _browse_findingampa_single_dir(self):
        path = filedialog.askdirectory(title="Select tomogram directory", initialdir=".")
        if path:
            self.findingampa_single_dir.set(path)
    # Update _run_findingampa_command to use single mode if checked
    def _run_findingampa_command(self, command, wait_for_completion=False):
        # Split command into base command and arguments
        command_parts = command.split()
        base_command = command_parts[0]
        command_args = command_parts[1:] if len(command_parts) > 1 else []
        
        # For DDW, require a model selection and pass as positional argument
        extra_args = []
        if base_command == "ddw":
            model = self.ddw_flag_var.get()
            if not model:
                self._log("Please select a model (k3, falcon, or falconczi) for DDW.\n")
                return None
            extra_args.append(model)
        
        # Create a threading event to track completion
        completion_event = threading.Event()
        
        def run_command_with_completion():
            try:
                if self.findingampa_single_mode.get() and self.findingampa_single_dir.get():
                    # Run in the selected directory only
                    cli = ["finding_ampa", base_command] + command_args + extra_args
                    self._log(f"Running (single tomogram): {' '.join(cli)} in {self.findingampa_single_dir.get()}\n")
                    env = os.environ.copy()
                    self._run_subprocess(cli, env, self.findingampa_single_dir.get())
                elif self.findingampa_all_mode.get() or not self.findingampa_single_mode.get():
                    # Run for all tomograms in CSV; alignment subdirectory per row from CSV only
                    csv_path = self.csv_path.get()
                    root_dir = self.root_dir.get()
                    if not csv_path or not root_dir:
                        self._log("CSV and root directory must be set to run for all tomograms.\n")
                        return
                    import csv as _csv, os as _os
                    with open(csv_path, newline='') as f:
                        reader = _csv.DictReader(f)
                        for row in reader:
                            set_name = row.get('set')
                            tomo_name = row.get('tomogram')
                            row_align = (row.get('alignment_dir') or '').strip()
                            if not set_name or not tomo_name:
                                continue
                            if not row_align or row_align.lower() == 'nan':
                                self._log(f"Skipping {tomo_name}: missing alignment_dir in CSV row.\n")
                                continue
                            align_work_dir = _os.path.join(root_dir, set_name, "TOP_TOMOS", tomo_name, row_align)
                            if not _os.path.isdir(align_work_dir):
                                self._log(f"Skipping missing directory: {align_work_dir}\n")
                                continue
                            cli = ["finding_ampa", base_command] + command_args + extra_args
                            self._log(f"Running: {' '.join(cli)} in {align_work_dir}\n")
                            env = os.environ.copy()
                            # Run subprocess and wait for completion before moving to next tomogram
                            self._run_subprocess(cli, env, align_work_dir)
                            self._log(f"Completed processing for tomogram: {tomo_name}\n")
            finally:
                completion_event.set()
        
        if wait_for_completion:
            # Run in current thread and wait for completion
            run_command_with_completion()
            return None
        else:
            # Run in background thread
            threading.Thread(target=run_command_with_completion).start()
            return completion_event

    def _run_findingampa_checked(self):
        # Run all checked commands in order from top to bottom, waiting for each to complete
        def run_checked_commands():
            for (label, command), var in zip(self.findingampa_commands, self.findingampa_check_vars):
                if var.get():
                    self._log(f"\nStarting command: {label}\n")
                    # Special handling for new-annotate-aunps to show popup
                    if command == "new-annotate-aunps --reset":
                        # Show popup and wait for user confirmation
                        instructions = """Annotate Membranes (Blender plug-in) Instructions:

1. Blender will open with the tomogram and membrane segmentations (both from WBP and DDW)loaded
2. Choose between WBP and DDW membrane segmentations (hide other)
3. Use the Blender interface to:
   - Clean up membrane segmentations (erase connections and/or adjust tresholding if necessary)
   - Assign pre/postsynaptic membranes (move to relevant group)
   - Assign presynaptic vesicles (move to relevant group)
4. Save your work when complete (Save as new_aunp_template_CJS_ddw.blend)
5. Close Blender to finish the annotation process

Do you want to continue?"""
                        
                        result = messagebox.askyesno("Blender Annotation Instructions", instructions)
                        if result:
                            completion_event = self._run_findingampa_command(command, wait_for_completion=False)
                            if completion_event:
                                completion_event.wait()  # Wait for this command to complete
                        else:
                            self._log("Skipping new-annotate-aunps command (user cancelled)\n")
                            continue
                    else:
                        # Run command and wait for completion
                        completion_event = self._run_findingampa_command(command, wait_for_completion=False)
                        if completion_event:
                            completion_event.wait()  # Wait for this command to complete
                    self._log(f"Completed command: {label}\n")
        
        # Run the entire sequence in a background thread
        threading.Thread(target=run_checked_commands).start()

    def _build_tab_content(self, tab, step):
        # For all tabs after home, use a horizontal layout
        content_frame = ttk.Frame(tab)
        content_frame.pack(fill=tk.BOTH, expand=True)
        controls_frame = ttk.Frame(content_frame)
        controls_frame.pack(side=tk.LEFT, anchor=tk.N, padx=10, pady=10)
        img_frame = ttk.Frame(content_frame)
        img_frame.pack(side=tk.RIGHT, anchor=tk.N, padx=10, pady=10)
        # Add figure to the right, larger size
        if step == "Active Zone":
            img = self._load_and_display_image(FIG_AZ, img_frame, max_width=525, max_height=240)
        elif step == "Vesicles":
            img = self._load_and_display_image(FIG_VESICLES, img_frame, max_width=525, max_height=240)
        else:
            img = self._load_and_display_image(FIG_AUNPS, img_frame, max_width=525, max_height=240)
        if img:
            img_label = ttk.Label(img_frame, image=img)
            img_label.pack()
            self._img_refs.append(img)
        if step == "Full Pipeline":
            ttk.Label(controls_frame, text="Active Zone -> Vesicles -> AuNPs -> Visualization").pack(anchor=tk.W, pady=(0, 10))
            ttk.Label(controls_frame, text="Steps to run:").pack(anchor=tk.W, pady=(0, 2))
            # (cli_step_name, display_label)
            pipeline_step_defs = [
                ("activezone", "Active Zone"),
                ("vesicles", "Vesicles"),
                ("aunps", "AuNPs"),
                ("visualizations", "Visualization"),
            ]
            tab._pipeline_step_vars = {}
            for cli_name, label in pipeline_step_defs:
                var = tk.BooleanVar(value=True)
                cb = ttk.Checkbutton(controls_frame, text=label, variable=var)
                cb.pack(anchor=tk.W)
                ToolTip(cb, f"Include the {label} step when running the full pipeline. Uncheck to skip it (e.g. run all except Visualization).")
                tab._pipeline_step_vars[cli_name] = var
            ttk.Label(controls_frame, text="").pack(anchor=tk.W, pady=(0, 4))
        run_btn = ttk.Button(controls_frame, text=f"Run {step}", command=lambda s=step: self._run_analysis(s, tab))
        run_btn.pack(anchor=tk.W, pady=5)
        # Add checkboxes for rerun, check-files
        rerun_var = tk.BooleanVar()
        checkfiles_var = tk.BooleanVar()
        rerun_cb = ttk.Checkbutton(controls_frame, text="Rerun (overwrite existing results)", variable=rerun_var)
        rerun_cb.pack(anchor=tk.W)
        ToolTip(rerun_cb, "Rerun analysis on already completed steps and overwrite existing results.")
        checkfiles_cb = ttk.Checkbutton(controls_frame, text="Check files only (no analysis)", variable=checkfiles_var)
        checkfiles_cb.pack(anchor=tk.W)
        ToolTip(checkfiles_cb, "Check that all expected files for the tomograms listed in the CSV are present in the expected locations. No analysis is run.")
        # Store flag variables in the tab for access in _run_analysis
        tab._flag_vars = (rerun_var, checkfiles_var)
        
        pdf_frame = None
        if step in ["Visualization", "Full Pipeline"]:
            pdf_frame = ttk.Frame(controls_frame)
            pdf_frame.pack(anchor=tk.W, pady=10)
            # PDF generation now runs automatically, only keep the view button
            view_btn = ttk.Button(pdf_frame, text="View PDF Summary", command=self._view_pdf_summary)
            view_btn.pack(anchor=tk.W)
        if step == "Full Pipeline":
            run_btn.config(command=lambda: self._run_analysis(step, tab))
        # Add Stop button in bottom right of tab (not Home)
        if step != "Home":
            stop_btn = ttk.Button(tab, text="Stop", command=self._stop_current_process)
            stop_btn.place(relx=1.0, rely=1.0, anchor="se", x=-10, y=-10)

    def _browse_csv(self):
        path = filedialog.askopenfilename(title="Select tomogram CSV", filetypes=[("CSV files", "*.csv")])
        if path:
            self.csv_path.set(path)

    def _browse_root(self):
        path = filedialog.askdirectory(title="Select root directory for tomogram sets", initialdir=".")
        if path:
            self.root_dir.set(path)
    
    def _browse_cluster_csv(self):
        path = filedialog.askopenfilename(title="Select cluster selection CSV", filetypes=[("CSV files", "*.csv")])
        if path:
            self.cluster_csv_path.set(path)
    
    def _browse_ampa_pdb_file(self):
        """Browse for AMPA PDB file."""
        file_path = filedialog.askopenfilename(
            title="Select PDB file for AMPA structure",
            filetypes=[("PDB files", "*.pdb"), ("All files", "*.*")]
        )
        if file_path:
            self.ampa_pdb_file.set(file_path)
    
    def _load_tomograms_from_csv(self):
        """Load tomogram names from the selected CSV file and populate the dropdown."""
        csv_path = self.csv_path.get()
        if not csv_path:
            messagebox.showerror("Error", "Please select a CSV file first.")
            return
        
        try:
            df = pd.read_csv(csv_path)
            if 'tomoname' not in df.columns:
                messagebox.showerror("Error", "CSV file must contain a 'tomoname' column.")
                return
            
            # Store the full dataframe for later use
            self.csv_data = df
            
            # Get tomogram names in the order they appear in the CSV
            tomogram_names = df['tomoname'].tolist()
            
            # Update the processing mode combo
            mode_values = ["All tomograms", "Single tomogram", "Start from"]
            self.processing_mode_combo['values'] = mode_values
            
            # Update the tomogram combo
            self.start_tomogram_combo['values'] = tomogram_names
            self.start_tomogram.set(tomogram_names[0] if tomogram_names else "All tomograms")  # Default to first tomogram
            
            self._log(f"Loaded {len(tomogram_names)} tomograms from CSV")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load CSV file: {e}")
            self._log(f"Error loading CSV: {e}")
    
    def _on_processing_mode_change(self, event=None):
        """Handle processing mode change to show/hide starting tomogram selection."""
        mode = self.processing_mode.get()
        
        if mode in ["Single tomogram", "Start from"]:
            # Show the starting tomogram selection (row 4: below processing mode, above separator)
            self.start_tomogram_label.grid(row=4, column=0, sticky=tk.W)
            self.start_tomogram_combo.grid(row=4, column=1, sticky=tk.W)
        else:
            # Hide the starting tomogram selection
            self.start_tomogram_label.grid_remove()
            self.start_tomogram_combo.grid_remove()
    
    def _cli_aunp_pick_star_pattern_args(self):
        """CLI flags for custom AuNP pick STAR naming (Home → Custom Parameters)."""
        pat = self.aunp_pick_star_pattern.get().strip()
        if pat:
            return ["--aunp-pick-star-pattern", pat]
        return []

    def _cli_fusion_point_aunp_analyses_args(self):
        """CLI flags for optional fusion-point vs monomer/dimer AuNP analyses."""
        if not self.use_custom_params.get():
            return []
        args = []
        if self.run_fusion_point_aunp_analyses.get():
            args += ["--fusion-point-aunp-analyses"]
        mon = self.monomer_star_pattern.get().strip()
        dim = self.dimer_star_pattern.get().strip()
        if mon:
            args += ["--monomer-star-pattern", mon]
        if dim:
            args += ["--dimer-star-pattern", dim]
        return args

    def _cli_aunp_vs_az_center_ripley_args(self):
        """CLI flag for optional AuNP vs active zone center Ripley L₁₂."""
        if not self.use_custom_params.get():
            return []
        if self.run_aunp_vs_az_center_ripley.get():
            return ["--aunp-vs-az-center-ripley"]
        return []

    def _cli_aunp_monomer_dimer_ripley_args(self):
        """CLI flag for optional monomer vs dimer AuNP Ripley L₁₂."""
        if not self.use_custom_params.get():
            return []
        if self.run_aunp_monomer_dimer_ripley.get():
            return ["--aunp-monomer-dimer-ripley"]
        return []

    def _toggle_monomer_dimer_star_entries(self):
        """Enable monomer/dimer STAR fields when any analysis using them is enabled."""
        uses_star = (
            self.run_fusion_point_aunp_analyses.get()
            or self.run_aunp_monomer_dimer_ripley.get()
        )
        state = "normal" if uses_star else "disabled"
        if hasattr(self, "_monomer_star_entry"):
            self._monomer_star_entry.configure(state=state)
            self._dimer_star_entry.configure(state=state)

    def _toggle_custom_params(self):
        """Handle custom parameters toggle to show/hide custom parameters frame."""
        if self.use_custom_params.get():
            # Show custom parameters frame
            self.custom_params_frame.grid()
        else:
            # Hide custom parameters frame
            self.custom_params_frame.grid_remove()
        # Update scroll region after showing/hiding
        if hasattr(self, '_home_canvas'):
            self._home_canvas.update_idletasks()
            self._home_canvas.configure(scrollregion=self._home_canvas.bbox("all"))
    
    def _archive_results(self):
        """Archive current results directory to a dated directory with user note."""
        results_dir = Path("results")
        if not results_dir.exists():
            messagebox.showwarning("Warning", "Results directory does not exist. Nothing to archive.")
            return
        
        # Check if results directory is empty
        if not any(results_dir.iterdir()):
            messagebox.showwarning("Warning", "Results directory is empty. Nothing to archive.")
            return
        
        # Get user note
        note = self.archive_note_var.get().strip()
        if not note:
            if not messagebox.askyesno("No Note", "No archive note provided. Continue without note?"):
                return
            note = "no_note"
        else:
            # Sanitize note for filename (remove invalid characters)
            import re
            note = re.sub(r'[<>:"/\\|?*]', '_', note)
            note = note.replace(' ', '_')
        
        # Create archive directory name
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_dir_name = f"{date_str}_results_{note}"
        
        # Create archived_results directory if it doesn't exist
        archived_results_base = Path("archived_results")
        archived_results_base.mkdir(exist_ok=True)
        
        archive_dir = archived_results_base / archive_dir_name
        
        # Check if archive directory already exists
        if archive_dir.exists():
            if not messagebox.askyesno("Directory Exists", f"Archive directory {archive_dir_name} already exists. Overwrite?"):
                return
            shutil.rmtree(archive_dir)
        
        try:
            # Move entire results directory to archive
            shutil.move(str(results_dir), str(archive_dir))
            # Recreate empty results directory
            results_dir.mkdir(exist_ok=True)
            
            messagebox.showinfo("Success", f"Results archived to archived_results/{archive_dir_name}")
            self._log(f"Archived results to archived_results/{archive_dir_name}\n")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to archive results: {e}")
            self._log(f"Error archiving results: {e}\n")
    
    def _delete_previous_results(self):
        """Delete previous results from tomogram STT_results and results directory."""
        csv_path = self.csv_path.get()
        if not csv_path:
            messagebox.showerror("Error", "Please select a CSV file first to identify which tomograms to process.")
            return
        
        # Get selected tomograms based on processing mode (same logic as _run_analysis)
        processing_mode = self.processing_mode.get()
        selected_tomogram = self.start_tomogram.get() if hasattr(self, 'start_tomogram') else None
        
        # Determine which tomograms to delete based on processing mode
        try:
            df = pd.read_csv(csv_path)
            if 'tomoname' not in df.columns:
                messagebox.showerror("Error", "CSV file must contain a 'tomoname' column.")
                return
            
            tomogram_names = df['tomoname'].tolist()
            
            if processing_mode == "Single tomogram" and selected_tomogram:
                tomograms_to_delete = [selected_tomogram]
                mode_description = f"single tomogram: {selected_tomogram}"
            elif processing_mode == "Start from" and selected_tomogram:
                start_idx = tomogram_names.index(selected_tomogram) if selected_tomogram in tomogram_names else 0
                tomograms_to_delete = tomogram_names[start_idx:]
                mode_description = f"tomograms starting from: {selected_tomogram} ({len(tomograms_to_delete)} tomograms)"
            else:
                # "All tomograms" mode
                tomograms_to_delete = tomogram_names
                mode_description = f"all tomograms in CSV ({len(tomograms_to_delete)} tomograms)"
        except Exception as e:
            messagebox.showerror("Error", f"Failed to read CSV file: {e}")
            return
        
        # Confirm deletion with specific tomogram list
        if not messagebox.askyesno("Confirm Deletion", 
                                   f"This will delete results for {mode_description}:\n\n"
                                   "- STT_results directories for selected tomograms\n"
                                   "- Results entries from results/analysis_results.json\n"
                                   "- All contents from results/ directory\n\n"
                                   "Archived results will NOT be deleted.\n\n"
                                   "Continue?"):
            return
        
        try:
            # Create a temporary CSV with only the selected tomograms
            import tempfile
            temp_csv = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
            temp_df = df[df['tomoname'].isin(tomograms_to_delete)]
            temp_df.to_csv(temp_csv.name, index=False)
            temp_csv.close()
            
            # Import delete function - ensure project root is in path
            import sys
            if str(REPO_ROOT) not in sys.path:
                sys.path.insert(0, str(REPO_ROOT))
            from src.synaptic_tomo_tools.cli import delete_csv_tomogram_results
            
            # Delete results for selected tomograms only
            self._log(f"Deleting previous results for {mode_description}...\n")
            delete_csv_tomogram_results(temp_csv.name, results_dir="results", data_dir="data")
            
            # Clean up temporary CSV
            os.unlink(temp_csv.name)
            
            # Also delete the entire results directory contents
            # (Archived directories are already moved out, so safe to delete everything)
            results_dir = Path("results")
            if results_dir.exists():
                for item in results_dir.iterdir():
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()
                self._log("Deleted all contents from results/ directory\n")
            
            messagebox.showinfo("Success", f"Previous results deleted successfully for {mode_description}.")
            self._log("Previous results deletion completed.\n")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to delete previous results: {e}")
            self._log(f"Error deleting previous results: {e}\n")

    def _run_analysis(self, step, tab, generate_pdf=False):
        # Build CLI command
        cli = ["python", "-u", "-m", "src.synaptic_tomo_tools.cli"]
        # CSV
        if self.csv_path.get():
            cli += ["--csv", self.csv_path.get()]
        
        # Handle processing mode and starting tomogram selection
        processing_mode = self.processing_mode.get()
        selected_tomogram = self.start_tomogram.get()
        
        if processing_mode == "Single tomogram" and selected_tomogram:
            # Create a temporary CSV with only the selected tomogram
            try:
                df = pd.read_csv(self.csv_path.get())
                tomogram_row = df[df['tomoname'] == selected_tomogram]
                if not tomogram_row.empty:
                    # Create temporary CSV with only this tomogram
                    import tempfile
                    temp_csv = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
                    tomogram_row.to_csv(temp_csv.name, index=False)
                    temp_csv.close()
                    
                    # Use the temporary CSV instead of the original
                    cli += ["--csv", temp_csv.name]
                    self._log(f"Processing single tomogram: {selected_tomogram}\n")
                    
                    # Store the temp file path to clean up later
                    if not hasattr(self, '_temp_csv_files'):
                        self._temp_csv_files = []
                    self._temp_csv_files.append(temp_csv.name)
                else:
                    self._log(f"Warning: Tomogram {selected_tomogram} not found in CSV\n")
            except Exception as e:
                self._log(f"Error creating temporary CSV for tomogram {selected_tomogram}: {e}\n")
        
        elif processing_mode == "Start from" and selected_tomogram:
            # Create a temporary CSV with the selected tomogram and all tomograms after it
            try:
                df = pd.read_csv(self.csv_path.get())
                # Find the index of the selected tomogram
                tomogram_indices = df[df['tomoname'] == selected_tomogram].index
                if len(tomogram_indices) > 0:
                    start_index = tomogram_indices[0]
                    # Get all tomograms from the selected one onwards
                    subset_df = df.iloc[start_index:]
                    
                    # Create temporary CSV with this subset
                    import tempfile
                    temp_csv = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
                    subset_df.to_csv(temp_csv.name, index=False)
                    temp_csv.close()
                    
                    # Use the temporary CSV instead of the original
                    cli += ["--csv", temp_csv.name]
                    self._log(f"Processing from tomogram {selected_tomogram} onwards ({len(subset_df)} tomograms)\n")
                    
                    # Store the temp file path to clean up later
                    if not hasattr(self, '_temp_csv_files'):
                        self._temp_csv_files = []
                    self._temp_csv_files.append(temp_csv.name)
                else:
                    self._log(f"Warning: Tomogram {selected_tomogram} not found in CSV\n")
            except Exception as e:
                self._log(f"Error creating temporary CSV for starting from {selected_tomogram}: {e}\n")
        
        # Analysis step
        if step == "Active Zone":
            cli += ["--analysis", "activezone"]
        elif step == "Vesicles":
            cli += ["--analysis", "vesicles"]
        elif step == "AuNPs":
            cli += ["--analysis", "aunps"]
        elif step == "Visualization":
            cli += ["--analysis", "visualizations"]
        elif step == "Full Pipeline":
            cli += ["--analysis", "all"]
            step_vars = getattr(tab, "_pipeline_step_vars", None)
            if step_vars:
                selected_steps = [name for name, var in step_vars.items() if var.get()]
                if not selected_steps:
                    messagebox.showwarning(
                        "No steps selected",
                        "Please select at least one pipeline step to run.",
                    )
                    return
                # Only pass --steps when running a subset; omit to keep default (all).
                if len(selected_steps) < len(step_vars):
                    cli += ["--steps", ",".join(selected_steps)]
            if generate_pdf:
                cli += ["--generate-pdf-summary"]
        # Add custom parameters if custom mode is selected
        if self.use_custom_params.get():
            # Active zone parameters
            if self.az_distance_min.get():
                try:
                    cli += ["--az-distance-min", str(float(self.az_distance_min.get()))]
                except ValueError:
                    pass
            if self.az_distance_max.get():
                try:
                    cli += ["--az-distance-max", str(float(self.az_distance_max.get()))]
                except ValueError:
                    pass
            
            # Vesicle analysis parameters
            if self.vesicle_distance_threshold.get():
                try:
                    cli += ["--vesicle-distance-threshold", str(float(self.vesicle_distance_threshold.get()))]
                except ValueError:
                    pass
            
            # AuNP analysis parameters
            if self.dbscan_eps.get():
                try:
                    cli += ["--dbscan-eps", str(float(self.dbscan_eps.get()))]
                except ValueError:
                    pass
            if self.dbscan_min_samples.get():
                try:
                    cli += ["--dbscan-min-samples", str(int(self.dbscan_min_samples.get()))]
                except ValueError:
                    pass
            if self.synaptic_designation_cutoff.get():
                try:
                    cli += ["--synaptic-designation-cutoff", str(float(self.synaptic_designation_cutoff.get()))]
                except ValueError:
                    pass
            if self.min_cluster_size.get():
                try:
                    cli += ["--min-cluster-size", str(int(self.min_cluster_size.get()))]
                except ValueError:
                    pass
            if self.fusion_point_threshold.get():
                try:
                    cli += ["--fusion-point-threshold", str(float(self.fusion_point_threshold.get()))]
                except ValueError:
                    pass
            if self.fusing_perimeter_threshold.get():
                try:
                    cli += ["--fusing-perimeter-threshold", str(float(self.fusing_perimeter_threshold.get()))]
                except ValueError:
                    pass
            cli += self._cli_aunp_pick_star_pattern_args()
            cli += self._cli_fusion_point_aunp_analyses_args()
            cli += self._cli_aunp_vs_az_center_ripley_args()
            cli += self._cli_aunp_monomer_dimer_ripley_args()
            
            # Visualization parameters
            if self.sphere_size.get():
                try:
                    cli += ["--sphere-size", str(int(self.sphere_size.get()))]
                except ValueError:
                    pass
            if self.sphere_color.get():
                cli += ["--sphere-color", str(self.sphere_color.get())]
            # AuNP distance range for color scale
            if self.aunp_distance_min.get():
                try:
                    cli += ["--aunp-distance-min", str(float(self.aunp_distance_min.get()))]
                except ValueError:
                    pass
            if self.aunp_distance_max.get():
                try:
                    cli += ["--aunp-distance-max", str(float(self.aunp_distance_max.get()))]
                except ValueError:
                    pass
            # AuNP distance cutoff filter
            if self.aunp_distance_cutoff_direction.get():
                cli += ["--aunp-distance-cutoff-direction", str(self.aunp_distance_cutoff_direction.get())]
            if self.aunp_distance_cutoff_value.get():
                try:
                    cli += ["--aunp-distance-cutoff-value", str(float(self.aunp_distance_cutoff_value.get()))]
                except ValueError:
                    pass
            if self.cylinder_radius.get():
                try:
                    cli += ["--cylinder-radius", str(float(self.cylinder_radius.get()))]
                except ValueError:
                    pass
            if self.receptor_crosssection.get():
                try:
                    cli += ["--receptor-crosssection", str(float(self.receptor_crosssection.get()))]
                except ValueError:
                    pass
            if self.aunps_per_receptor.get():
                try:
                    cli += ["--aunps-per-receptor", str(float(self.aunps_per_receptor.get()))]
                except ValueError:
                    pass
            if self.vertex_sampling_step.get():
                try:
                    cli += ["--vertex-sampling-step", str(int(self.vertex_sampling_step.get()))]
                except ValueError:
                    pass
        
        # Add flags from checkboxes
        rerun_var, checkfiles_var = getattr(tab, '_flag_vars', (None, None))
        if rerun_var and rerun_var.get():
            cli += ["--rerun"]
        if checkfiles_var and checkfiles_var.get():
            cli += ["--check-files"]
        self._log(f"Running: {' '.join(cli)}\n")
        # Pass the single root dir as env var TOMO_ROOT_BASE if set
        env = os.environ.copy()
        if self.root_dir.get():
            env["TOMO_ROOT_BASE"] = self.root_dir.get()
        threading.Thread(target=self._run_subprocess, args=(cli, env, str(REPO_ROOT))).start()

    def _run_subprocess(self, cli, env, cwd=None):
        self._current_process = subprocess.Popen(cli, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env, cwd=cwd, bufsize=1, universal_newlines=True)
        try:
            # Read output character by character to handle progress bars with \r
            output_buffer = ""
            while True:
                char = self._current_process.stdout.read(1)
                if not char:
                    break
                
                if char == '\r':
                    # Carriage return - clear the current line and update with new content
                    if output_buffer.strip():
                        # Remove the last line from the log and replace it
                        self._clear_last_line()
                        self._log(output_buffer)
                    output_buffer = ""
                elif char == '\n':
                    # Newline - log the complete line
                    self._log(output_buffer + char)
                    output_buffer = ""
                else:
                    # Regular character - add to buffer
                    output_buffer += char
            
            # Log any remaining buffer content
            if output_buffer:
                self._log(output_buffer)
            
            self._current_process.wait()
            self._log(f"\n[Process exited with code {self._current_process.returncode}]\n\n")
        finally:
            self._current_process = None
            # Clean up temporary CSV files
            self._cleanup_temp_csv_files()
    
    def _cleanup_temp_csv_files(self):
        """Clean up temporary CSV files created for single tomogram analysis."""
        if hasattr(self, '_temp_csv_files'):
            for temp_file in self._temp_csv_files:
                try:
                    os.unlink(temp_file)
                except Exception as e:
                    self._log(f"Warning: Could not delete temporary file {temp_file}: {e}\n")
            self._temp_csv_files = []

    def _stop_current_process(self):
        if self._current_process and self._current_process.poll() is None:
            self._current_process.terminate()
            self._log("\n[Process terminated by user]\n\n")
        else:
            self._log("\n[No process is currently running]\n\n")

    def _log(self, msg):
        if self._session_log_fp is not None:
            try:
                self._session_log_fp.write(msg)
                self._session_log_fp.flush()
            except Exception:
                pass
        if self.log_text is None:
            return
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, msg)
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.NORMAL)

    def _clear_last_line(self):
        """Clear the last line in the log text widget."""
        if self.log_text is None:
            return
        self.log_text.config(state=tk.NORMAL)
        # Get the current content
        content = self.log_text.get("1.0", tk.END)
        lines = content.split('\n')
        
        # Remove the last line if it exists
        if len(lines) > 1:
            # Remove the last line (which is usually empty due to trailing newline)
            new_content = '\n'.join(lines[:-1])
            if new_content and not new_content.endswith('\n'):
                new_content += '\n'
            
            # Replace all content
            self.log_text.delete("1.0", tk.END)
            self.log_text.insert("1.0", new_content)
        
        self.log_text.config(state=tk.NORMAL)


    def _view_pdf_summary(self):
        pdf_path = os.path.abspath("results/visualizations/pdf_summaries/all_tomograms_summary.pdf")
        if not os.path.exists(pdf_path):
            messagebox.showerror("PDF Not Found", f"{pdf_path} does not exist. Please generate the PDF summary first.")
            return
        webbrowser.open(f"file://{pdf_path}")
    
    def _build_post_analysis_tab_content(self, tab):
        """Build the content for the post-analysis tools tab."""
        frame = ttk.Frame(tab)
        frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Title
        title_label = ttk.Label(frame, text="Post-Analysis Tools", font=("Helvetica", 16, "bold"))
        title_label.pack(pady=(0, 20))
        
        # Create notebook for sub-tabs
        notebook = ttk.Notebook(frame)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Cluster Analysis tab
        cluster_tab = ttk.Frame(notebook)
        notebook.add(cluster_tab, text="Cluster Analysis")
        self._build_cluster_tab_content(cluster_tab)
        
        # Vesicle Analysis tab
        vesicle_tab = ttk.Frame(notebook)
        notebook.add(vesicle_tab, text="Vesicle Analysis")
        self._build_vesicle_tab_content(vesicle_tab)
    
    def _build_vesicle_tab_content(self, tab):
        """Build the content for the vesicle analysis tab."""
        frame = ttk.Frame(tab)
        frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Vesicle Slice Extraction section
        vesicle_frame = ttk.LabelFrame(frame, text="Vesicle Slice Extraction", padding=10)
        vesicle_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Description
        desc_label = ttk.Label(vesicle_frame, text="Extract 120x120 pixel slices from vesicles within 20nm of active zone membrane.\nGenerates three slice types: regular slice, thick slice (20nm), and MinIP (20nm).\nSlices are oriented with the closest active zone point pointing down.\nCreates both individual PNG files and a comprehensive PDF summary.")
        desc_label.pack(pady=(0, 10))
        
        # Output directory
        self.vesicle_output_dir = tk.StringVar(value="results/vesicle_slices")
        ttk.Label(vesicle_frame, text="Output directory:").pack(anchor=tk.W)
        output_entry = ttk.Entry(vesicle_frame, textvariable=self.vesicle_output_dir, width=50)
        output_entry.pack(fill=tk.X, pady=(0, 10))
        
        # Run button
        vesicle_btn = ttk.Button(vesicle_frame, text="Extract Vesicle Slices", command=self._run_vesicle_extraction)
        vesicle_btn.pack(pady=(0, 10))
        
        # Add tooltip
        ToolTip(vesicle_btn, "Extract vesicle slices from tomograms with MinIP projection for dark signal visualization. Requires vesicle analysis to be completed first. Creates PNG files and PDF summary.")
        
        # View PDF button
        view_pdf_btn = ttk.Button(vesicle_frame, text="View Vesicle Slices PDF", command=self._view_vesicle_pdf)
        view_pdf_btn.pack(pady=(0, 10))
        
        # Add tooltip for PDF button
        ToolTip(view_pdf_btn, "Open the vesicle slices summary PDF in your default PDF viewer.")
        
        # View Close Vesicles PDF button
        view_close_pdf_btn = ttk.Button(vesicle_frame, text="View Close Vesicles PDF", command=self._view_close_vesicle_pdf)
        view_close_pdf_btn.pack(pady=(0, 10))
        
        # Add tooltip for close vesicles PDF button
        ToolTip(view_close_pdf_btn, "Open the close vesicles summary PDF (≤4nm from active zone) in your default PDF viewer.")
    
    def _build_ampa_poses_tab_content(self, tab):
        """Build the content for the dedicated AMPA poses analysis tab."""
        # Use the same horizontal layout as other tabs
        content_frame = ttk.Frame(tab)
        content_frame.pack(fill=tk.BOTH, expand=True)
        controls_frame = ttk.Frame(content_frame)
        controls_frame.pack(side=tk.LEFT, anchor=tk.N, padx=10, pady=10)
        img_frame = ttk.Frame(content_frame)
        img_frame.pack(side=tk.RIGHT, anchor=tk.N, padx=10, pady=10)
        
        # Add figure to the right, same size as other tabs
        img = self._load_and_display_image(FIG_POSES, img_frame, max_width=525, max_height=240)
        if img:
            img_label = ttk.Label(img_frame, image=img)
            img_label.pack()
            self._img_refs.append(img)
        
        # Run Analysis button at the top
        ampa_run_btn = ttk.Button(controls_frame, text="Run Pose Prediction", command=self._run_ampa_poses_analysis_with_selected_method)
        ampa_run_btn.pack(anchor=tk.W, pady=5)
        ToolTip(ampa_run_btn, "Run AMPA poses analysis using the selected method: All poses (no optimization), Greedy (fast heuristic), or ILP (exact optimal).")
        
        # AMPA Poses Analysis section
        ampa_frame = ttk.LabelFrame(controls_frame, text="Analysis Parameters", padding=15)
        ampa_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Parameters frame
        params_frame = ttk.Frame(ampa_frame)
        params_frame.pack(fill=tk.X, pady=(0, 15))
        
        # AuNP distance parameters
        ttk.Label(params_frame, text="AuNP Distance Range (nm):").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.ampa_aunp_min_dist = tk.StringVar(value="6")
        ttk.Entry(params_frame, textvariable=self.ampa_aunp_min_dist, width=10).grid(row=0, column=1, padx=(0, 5))
        ttk.Label(params_frame, text="to").grid(row=0, column=2, padx=5)
        self.ampa_aunp_max_dist = tk.StringVar(value="12")
        ttk.Entry(params_frame, textvariable=self.ampa_aunp_max_dist, width=10).grid(row=0, column=3, padx=(5, 0))
        
        # AuNP distance cutoff checkbox
        self.ampa_aunp_no_cutoff = tk.BooleanVar()
        ttk.Checkbutton(params_frame, text="No AuNP distance cutoff", variable=self.ampa_aunp_no_cutoff).grid(row=0, column=4, padx=(10, 0), sticky=tk.W)
        
        # Membrane distance parameters
        ttk.Label(params_frame, text="Membrane Distance Range (nm):").grid(row=1, column=0, sticky=tk.W, padx=(0, 10), pady=(10, 0))
        self.ampa_membrane_min_dist = tk.StringVar(value="17")
        ttk.Entry(params_frame, textvariable=self.ampa_membrane_min_dist, width=10).grid(row=1, column=1, padx=(0, 5), pady=(10, 0))
        ttk.Label(params_frame, text="to").grid(row=1, column=2, padx=5, pady=(10, 0))
        self.ampa_membrane_max_dist = tk.StringVar(value="23")
        ttk.Entry(params_frame, textvariable=self.ampa_membrane_max_dist, width=10).grid(row=1, column=3, padx=(5, 0), pady=(10, 0))
        
        # Membrane distance cutoff checkbox
        self.ampa_membrane_no_cutoff = tk.BooleanVar()
        ttk.Checkbutton(params_frame, text="No membrane distance cutoff", variable=self.ampa_membrane_no_cutoff).grid(row=1, column=4, padx=(10, 0), sticky=tk.W, pady=(10, 0))
        
        # Steric radius parameter (for optimized method)
        ttk.Label(params_frame, text="Steric Radius (nm):").grid(row=2, column=0, sticky=tk.W, padx=(0, 10), pady=(10, 0))
        self.ampa_steric_radius = tk.StringVar(value="5.0")
        ttk.Entry(params_frame, textvariable=self.ampa_steric_radius, width=10).grid(row=2, column=1, padx=(0, 5), pady=(10, 0))
        ttk.Label(params_frame, text="(minimum distance between particle positions)").grid(row=2, column=2, columnspan=3, sticky=tk.W, padx=(5, 0), pady=(10, 0))
        
        # Analysis method selection
        ttk.Label(params_frame, text="Analysis Method:").grid(row=3, column=0, sticky=tk.W, padx=(0, 10), pady=(10, 0))
        self.ampa_optimization_method = tk.StringVar(value="original")
        method_frame = ttk.Frame(params_frame)
        method_frame.grid(row=3, column=1, columnspan=4, sticky=tk.W, pady=(10, 0))
        
        ttk.Radiobutton(method_frame, text="All poses", variable=self.ampa_optimization_method, value="original").pack(side=tk.LEFT, padx=(0, 15))
        ttk.Radiobutton(method_frame, text="Greedy (fast, heuristic)", variable=self.ampa_optimization_method, value="greedy").pack(side=tk.LEFT, padx=(0, 15))
        ttk.Radiobutton(method_frame, text="ILP (exact, linear)", variable=self.ampa_optimization_method, value="ilp").pack(side=tk.LEFT)
        
        # PDB file parameter
        ttk.Label(params_frame, text="PDB File:").grid(row=4, column=0, sticky=tk.W, padx=(0, 10), pady=(10, 0))
        self.ampa_pdb_file = tk.StringVar()
        pdb_entry = ttk.Entry(params_frame, textvariable=self.ampa_pdb_file, width=15)
        pdb_entry.grid(row=4, column=1, padx=(0, 5), pady=(10, 0))
        pdb_browse_btn = ttk.Button(params_frame, text="Browse...", command=self._browse_ampa_pdb_file)
        pdb_browse_btn.grid(row=4, column=2, padx=(5, 0), pady=(10, 0))
        ttk.Label(params_frame, text="(leave empty to skip PDB generation)").grid(row=4, column=3, columnspan=2, sticky=tk.W, padx=(5, 0), pady=(10, 0))
        ToolTip(pdb_entry, "Select a PDB file to generate PDB files with AMPA structures at calculated poses. Leave empty to skip PDB generation.")
        
        
        # Add Stop button in bottom right of tab
        stop_btn = ttk.Button(tab, text="Stop", command=self._stop_current_process)
        stop_btn.place(relx=1.0, rely=1.0, anchor="se", x=-10, y=-10)
    
    
    def _build_cluster_tab_content(self, tab):
        """Build the content for the cluster analysis tab."""
        frame = ttk.Frame(tab)
        frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Cluster Coordinate Extraction section
        cluster_frame = ttk.LabelFrame(frame, text="Cluster Coordinate Extraction", padding=10)
        cluster_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Description
        cluster_desc_label = ttk.Label(cluster_frame, text="Extract XYZ coordinates for specific AuNP clusters.\nInput a CSV file with tomogram names, cluster numbers, and set names to extract coordinates.\nOutputs a text file with XYZ coordinates for each specified cluster.\nAlso generates a PDF summary showing mini zonogram images for the selected clusters.")
        cluster_desc_label.pack(pady=(0, 10))
        
        # Cluster selection CSV file
        ttk.Label(cluster_frame, text="Cluster selection CSV file:").pack(anchor=tk.W)
        self.cluster_csv_path = tk.StringVar()
        cluster_csv_frame = ttk.Frame(cluster_frame)
        cluster_csv_frame.pack(fill=tk.X, pady=(0, 10))
        cluster_csv_entry = ttk.Entry(cluster_csv_frame, textvariable=self.cluster_csv_path, width=40)
        cluster_csv_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(cluster_csv_frame, text="Browse...", command=self._browse_cluster_csv).pack(side=tk.RIGHT, padx=(5, 0))
        
        # Output directory
        self.cluster_output_dir = tk.StringVar(value="results/cluster_coordinates")
        ttk.Label(cluster_frame, text="Output directory:").pack(anchor=tk.W)
        cluster_output_entry = ttk.Entry(cluster_frame, textvariable=self.cluster_output_dir, width=50)
        cluster_output_entry.pack(fill=tk.X, pady=(0, 10))
        
        # Run button
        cluster_btn = ttk.Button(cluster_frame, text="Extract Cluster Coordinates", command=self._run_cluster_coordinate_extraction)
        cluster_btn.pack(pady=(0, 10))
        
        # Add tooltip
        ToolTip(cluster_btn, "Extract XYZ coordinates for specific AuNP clusters based on the input CSV file. Requires AuNP analysis to be completed first.")
    
    def _run_vesicle_extraction(self):
        """Run the vesicle slice extraction script."""
        if not self.csv_path.get():
            messagebox.showerror("Error", "Please select a CSV file first.")
            return
        
        # Build command
        cli = ["python", "-u", "scripts/extract_vesicle_slices.py"]
        cli += ["--csv", self.csv_path.get()]
        cli += ["--output-dir", self.vesicle_output_dir.get()]
        
        if self.root_dir.get():
            cli += ["--data-dir", self.root_dir.get()]
        
        # Add processing mode and starting tomogram if specified
        processing_mode = self.processing_mode.get()
        selected_tomogram = self.start_tomogram.get()
        
        if processing_mode in ["Single tomogram", "Start from"] and selected_tomogram:
            cli += ["--start-from", selected_tomogram]
            self._log(f"Vesicle extraction will start from tomogram: {selected_tomogram}\n")
        
        self._log(f"Running vesicle slice extraction: {' '.join(cli)}\n")
        self._log("Note: This will generate individual PNG files and create a comprehensive PDF summary.\n")
        threading.Thread(target=self._run_subprocess, args=(cli, os.environ.copy())).start()

    def _view_vesicle_pdf(self):
        """Open the vesicle slices summary PDF."""
        # Check if vesicle output directory exists
        vesicle_output_dir = self.vesicle_output_dir.get() if hasattr(self, 'vesicle_output_dir') else "results/vesicle_slices"
        pdf_path = os.path.abspath(f"{vesicle_output_dir}/vesicle_slices_summary.pdf")
        
        if not os.path.exists(pdf_path):
            messagebox.showerror("PDF Not Found", 
                               f"Vesicle slices summary PDF not found at:\n{pdf_path}\n\n"
                               "Please run the vesicle slice extraction first.")
            return
        
        try:
            webbrowser.open(f"file://{pdf_path}")
            self._log(f"Opening vesicle slices PDF: {pdf_path}\n")
        except Exception as e:
            messagebox.showerror("Error", f"Could not open PDF: {e}")

    def _view_close_vesicle_pdf(self):
        """Open the close vesicles summary PDF."""
        # Check if vesicle output directory exists
        vesicle_output_dir = self.vesicle_output_dir.get() if hasattr(self, 'vesicle_output_dir') else "results/vesicle_slices"
        pdf_path = os.path.abspath(f"{vesicle_output_dir}/close_vesicles_summary.pdf")
        
        if not os.path.exists(pdf_path):
            messagebox.showerror("PDF Not Found", 
                               f"Close vesicles summary PDF not found at:\n{pdf_path}\n\n"
                               "Please run the vesicle slice extraction first.")
            return
        
        try:
            webbrowser.open(f"file://{pdf_path}")
            self._log(f"Opening close vesicles PDF: {pdf_path}\n")
        except Exception as e:
            messagebox.showerror("Error", f"Could not open PDF: {e}")

    def _run_cluster_coordinate_extraction(self):
        """Run the cluster coordinate extraction script."""
        if not self.cluster_csv_path.get():
            messagebox.showerror("Error", "Please select a cluster selection CSV file first.")
            return
        
        if not self.root_dir.get():
            messagebox.showerror("Error", "Please specify the root directory for tomogram sets.")
            return
        
        # Create output directory
        output_dir = self.cluster_output_dir.get()
        os.makedirs(output_dir, exist_ok=True)
        
        # Build command
        cli = ["python", "-u", "scripts/extract_cluster_coordinates.py"]
        cli += ["--cluster-csv", self.cluster_csv_path.get()]
        cli += ["--data-dir", self.root_dir.get()]
        cli += ["--output-dir", output_dir]
        
        self._log(f"Running cluster coordinate extraction: {' '.join(cli)}\n")
        self._log("Note: This will extract XYZ coordinates for specified clusters and save them as text files.\n")
        threading.Thread(target=self._run_subprocess, args=(cli, os.environ.copy())).start()

    def _run_ampa_poses_analysis(self):
        """Run the AMPA poses analysis on all tomograms."""
        if not self.csv_path.get():
            messagebox.showerror("Error", "Please select a CSV file first.")
            return
        
        # Check if root directory is specified
        if not self.root_dir.get():
            messagebox.showerror("Error", "Please specify the root directory for tomogram sets.")
            return
        
        # Get parameters from GUI
        try:
            aunp_min_dist = float(self.ampa_aunp_min_dist.get())
            aunp_max_dist = float(self.ampa_aunp_max_dist.get())
            membrane_min_dist = float(self.ampa_membrane_min_dist.get())
            membrane_max_dist = float(self.ampa_membrane_max_dist.get())
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numeric values for distance parameters.")
            return
        
        # Check if cutoffs are disabled
        aunp_no_cutoff = self.ampa_aunp_no_cutoff.get()
        membrane_no_cutoff = self.ampa_membrane_no_cutoff.get()
        
        # Validate distance ranges only if cutoffs are enabled
        if not aunp_no_cutoff and aunp_min_dist >= aunp_max_dist:
            messagebox.showerror("Error", "AuNP minimum distance must be less than maximum distance.")
            return
        
        if not membrane_no_cutoff and membrane_min_dist >= membrane_max_dist:
            messagebox.showerror("Error", "Membrane minimum distance must be less than maximum distance.")
            return
        
        # Get output directory (will be constructed per tomogram)
        output_dir_relative = "STT_results/ampa_poses"
        
        # Read CSV to get tomogram names and sets
        try:
            df = pd.read_csv(self.csv_path.get())
            
            # Handle processing mode and starting tomogram selection
            processing_mode = self.processing_mode.get()
            selected_tomogram = self.start_tomogram.get()
            
            if processing_mode == "Single tomogram" and selected_tomogram:
                # Filter to only the selected tomogram
                tomogram_row = df[df['tomoname'] == selected_tomogram]
                if not tomogram_row.empty:
                    df = tomogram_row
                    self._log(f"Processing single tomogram: {selected_tomogram}\n")
                else:
                    self._log(f"Warning: Tomogram {selected_tomogram} not found in CSV\n")
                    
            elif processing_mode == "Start from" and selected_tomogram:
                # Filter to the selected tomogram and all tomograms after it
                tomogram_indices = df[df['tomoname'] == selected_tomogram].index
                if len(tomogram_indices) > 0:
                    start_index = tomogram_indices[0]
                    df = df.iloc[start_index:]
                    self._log(f"Processing from tomogram {selected_tomogram} onwards ({len(df)} tomograms)\n")
                else:
                    self._log(f"Warning: Tomogram {selected_tomogram} not found in CSV\n")
            
            # Check for different possible column names for tomogram names
            if 'tomogram_name' in df.columns:
                tomogram_names = df['tomogram_name'].tolist()
            elif 'tomoname' in df.columns:
                tomogram_names = df['tomoname'].tolist()
            elif 'tomogram' in df.columns:
                tomogram_names = df['tomogram'].tolist()
            else:
                # Try to find any column that might contain tomogram names
                possible_columns = [col for col in df.columns if 'tom' in col.lower() or 'name' in col.lower()]
                if possible_columns:
                    tomogram_names = df[possible_columns[0]].tolist()
                    self._log(f"Using column '{possible_columns[0]}' for tomogram names\n")
                else:
                    messagebox.showerror("Error", "Could not find tomogram name column in CSV. Expected 'tomogram_name', 'tomoname', or 'tomogram'")
                    return
            
            # Get the set column for path construction
            if 'set' in df.columns:
                tomogram_sets = df['set'].tolist()
            else:
                # Default to '15F1' if no set column found
                tomogram_sets = ['15F1'] * len(tomogram_names)
                self._log("No 'set' column found, defaulting to '15F1' for all tomograms\n")

            if 'alignment_dir' not in df.columns:
                messagebox.showerror(
                    "Error",
                    "CSV must include 'alignment_dir' for each tomogram row.",
                )
                return
            tomogram_alignments = []
            for x in df['alignment_dir'].tolist():
                s = str(x).strip()
                if not s or s.lower() == 'nan':
                    messagebox.showerror(
                        "Error",
                        "alignment_dir must be non-empty for every CSV row.",
                    )
                    return
                tomogram_alignments.append(s)
                    
        except Exception as e:
            messagebox.showerror("Error", f"Could not read CSV file: {e}")
            return
        
        self._log(f"Starting AMPA poses analysis for {len(tomogram_names)} tomograms...\n")
        self._log(f"Output directory (relative to each tomogram): {output_dir_relative}\n")
        
        if aunp_no_cutoff:
            self._log(f"AuNP distance range: No cutoff (using all AuNP pairs)\n")
        else:
            self._log(f"AuNP distance range: {aunp_min_dist}-{aunp_max_dist} nm\n")
            
        if membrane_no_cutoff:
            self._log(f"Membrane distance range: No cutoff (using all pairs regardless of membrane distance)\n")
        else:
            self._log(f"Membrane distance range: {membrane_min_dist}-{membrane_max_dist} nm\n")
        
        # Process all tomograms
        all_commands = []
        
        for i, (tomogram_name, tomogram_set, align_sub) in enumerate(
            zip(tomogram_names, tomogram_sets, tomogram_alignments), 1
        ):
            self._log(f"\nPreparing tomogram {i}/{len(tomogram_names)}: {tomogram_name} (set: {tomogram_set}, alignment: {align_sub})\n")
            
            # Build tomogram path using the correct structure
            tomogram_path = os.path.join(self.root_dir.get(), tomogram_set, 'TOP_TOMOS', tomogram_name)
            
            if not os.path.exists(tomogram_path):
                self._log(f"Warning: Tomogram directory not found: {tomogram_path}\n")
                continue
            
            # Get active zones for this tomogram from CSV
            tomogram_row = df[df['tomoname'] == tomogram_name] if 'tomoname' in df.columns else df[df.iloc[:, 0] == tomogram_name]
            aunp_active_zones = None
            
            if not tomogram_row.empty and 'aunp_active_zones' in df.columns:
                aunp_active_zones_str = tomogram_row['aunp_active_zones'].iloc[0]
                if pd.notna(aunp_active_zones_str) and str(aunp_active_zones_str).strip():
                    try:
                        # Parse active zones (can be comma-separated or space-separated)
                        aunp_active_zones_str = str(aunp_active_zones_str).strip()
                        if ',' in aunp_active_zones_str:
                            aunp_active_zones = []
                            for x in aunp_active_zones_str.split(','):
                                x = x.strip()
                                if x.isdigit():
                                    aunp_active_zones.append(int(x))
                                elif x.replace(".", "").isdigit():  # Handle floats like "0.0"
                                    aunp_active_zones.append(int(float(x)))
                        else:
                            aunp_active_zones = []
                            for x in aunp_active_zones_str.split():
                                x = x.strip()
                                if x.isdigit():
                                    aunp_active_zones.append(int(x))
                                elif x.replace(".", "").isdigit():  # Handle floats like "0.0"
                                    aunp_active_zones.append(int(float(x)))
                        self._log(f"Using active zones for {tomogram_name}: {aunp_active_zones}\n")
                    except (ValueError, AttributeError):
                        self._log(f"Warning: Could not parse active zones for {tomogram_name}: {aunp_active_zones_str}\n")
                        aunp_active_zones = None
                else:
                    self._log(f"No active zones specified for {tomogram_name}, using all active zones\n")
            else:
                self._log(f"No active zones specified for {tomogram_name}, using all active zones\n")
            
            # Build command for this tomogram
            # Construct full output directory path relative to tomogram
            full_output_dir = os.path.join(tomogram_path, align_sub, output_dir_relative)
            os.makedirs(full_output_dir, exist_ok=True)
            
            cli = ["python", "-u", "scripts/run_ampa_poses_analysis.py"]
            cli += ["--tomogram-path", tomogram_path]
            cli += ["--alignment-dir", align_sub]
            cli += ["--output-dir", full_output_dir]
            
            # Add distance parameters only if cutoffs are enabled
            if not aunp_no_cutoff:
                cli += ["--aunp-min-distance", str(aunp_min_dist)]
                cli += ["--aunp-max-distance", str(aunp_max_dist)]
            else:
                cli += ["--no-aunp-distance-cutoff"]
                
            if not membrane_no_cutoff:
                cli += ["--membrane-min-distance", str(membrane_min_dist)]
                cli += ["--membrane-max-distance", str(membrane_max_dist)]
            else:
                cli += ["--no-membrane-distance-cutoff"]
            
            # Add active zones if specified
            if aunp_active_zones is not None:
                cli += ["--aunp-active-zones"] + [str(az) for az in aunp_active_zones]
            
            cli += self._cli_aunp_pick_star_pattern_args()
            
            all_commands.append((tomogram_name, cli, align_sub))
        
        if not all_commands:
            self._log("No valid tomograms found to process.\n")
            return
        
        # Run all commands sequentially with real-time output
        self._log(f"\nStarting AMPA poses analysis for {len(all_commands)} tomograms...\n")
        
        # Use threading to run the entire sequence in background while maintaining real-time output
        def run_sequential_analysis():
            all_particles_data = []
            all_aunps_data = []
            all_aunps_all_data = []
            successful_tomograms = []
            
            for i, (tomogram_name, cli, align_sub) in enumerate(all_commands, 1):
                self._log(f"\n{'='*60}\n")
                self._log(f"Processing tomogram {i}/{len(all_commands)}: {tomogram_name}\n")
                self._log(f"Command: {' '.join(cli)}\n")
                self._log(f"{'='*60}\n")
                
                # Run the subprocess and wait for completion
                self._run_subprocess(cli, os.environ.copy())
            
                # Try to load the results for combining
                try:
                    import starfile
                    import pandas as pd
                    
                    # Load the individual results
                    tomogram_path = os.path.join(self.root_dir.get(), tomogram_sets[i-1], 'TOP_TOMOS', tomogram_name)
                    individual_output_dir = os.path.join(tomogram_path, align_sub, output_dir_relative)
                    
                    # Find the star file (it will have the distance parameters in the name)
                    star_files = [f for f in os.listdir(individual_output_dir) if f.endswith('.star') and 'ampa_poses' in f and '_aunps' not in f]
                    if star_files:
                        star_file_path = os.path.join(individual_output_dir, star_files[0])
                        star_data = starfile.read(star_file_path)
                        if 'particles' in star_data:
                            particles_df = star_data['particles'].copy()
                            particles_df['rlnTomoName'] = tomogram_name  # Ensure tomogram name is set
                            all_particles_data.append(particles_df)
                            successful_tomograms.append(tomogram_name)
                    
                    # Load the AuNPs file (used for AMPA poses)
                    aunps_files = [f for f in os.listdir(individual_output_dir) if f.endswith('.star') and 'ampa_poses' in f and '_aunps' in f and '_all_aunps' not in f]
                    if aunps_files:
                        aunps_file_path = os.path.join(individual_output_dir, aunps_files[0])
                        aunps_data = starfile.read(aunps_file_path)
                        if 'particles' in aunps_data:
                            aunps_df = aunps_data['particles'].copy()
                            aunps_df['rlnTomoName'] = tomogram_name  # Ensure tomogram name is set
                            all_aunps_data.append(aunps_df)
                    
                    # Load the all AuNPs file
                    all_aunps_files = [f for f in os.listdir(individual_output_dir) if f.endswith('.star') and 'ampa_poses' in f and '_all_aunps' in f]
                    if all_aunps_files:
                        all_aunps_file_path = os.path.join(individual_output_dir, all_aunps_files[0])
                        all_aunps_data = starfile.read(all_aunps_file_path)
                        if 'particles' in all_aunps_data:
                            all_aunps_df = all_aunps_data['particles'].copy()
                            all_aunps_df['rlnTomoName'] = tomogram_name  # Ensure tomogram name is set
                            all_aunps_all_data.append(all_aunps_df)
                
                except Exception as e:
                    self._log(f"Warning: Could not load results for {tomogram_name}: {e}\n")
            
            # Save combined star files
            if all_particles_data:
                try:
                    # Create results/ampa_poses directory
                    combined_output_dir = "results/ampa_poses"
                    os.makedirs(combined_output_dir, exist_ok=True)
                    
                    # Combine all particles data
                    combined_particles = pd.concat(all_particles_data, ignore_index=True)
                    
                    # Save combined AMPA poses star file
                    starfile.write({
                        'particles': combined_particles,
                        'optics': pd.DataFrame([{'rlnOpticsGroup': 1}])
                    }, os.path.join(combined_output_dir, "all_ampa_poses.star"))
                    
                    self._log(f"Saved combined AMPA poses to {combined_output_dir}/all_ampa_poses.star\n")
                    
                    # Save combined AuNPs star file (used for AMPA poses)
                    if all_aunps_data:
                        combined_aunps = pd.concat(all_aunps_data, ignore_index=True)
                        starfile.write({
                            'particles': combined_aunps,
                            'optics': pd.DataFrame([{'rlnOpticsGroup': 1}])
                        }, os.path.join(combined_output_dir, "all_ampa_poses_aunps.star"))
                        
                        self._log(f"Saved combined AuNPs to {combined_output_dir}/all_ampa_poses_aunps.star\n")
                    
                    # Save combined ALL AuNPs star file
                    if all_aunps_all_data:
                        combined_all_aunps = pd.concat(all_aunps_all_data, ignore_index=True)
                        starfile.write({
                            'particles': combined_all_aunps,
                            'optics': pd.DataFrame([{'rlnOpticsGroup': 1}])
                        }, os.path.join(combined_output_dir, "all_ampa_poses_all_aunps.star"))
                        
                        self._log(f"Saved combined ALL AuNPs to {combined_output_dir}/all_ampa_poses_all_aunps.star\n")
                    
                    self._log(f"Combined results from {len(successful_tomograms)} tomograms: {', '.join(successful_tomograms)}\n")
                    
                except Exception as e:
                    self._log(f"Warning: Could not save combined results: {e}\n")
            
            self._log(f"\nAMPA poses analysis completed for {len(all_commands)} tomograms.\n")
            self._log(f"Results saved to: {output_dir_relative} within each tomogram's STT_results directory\n")
            if all_particles_data:
                self._log(f"Combined results saved to: results/ampa_poses/\n")
        
        # Start the sequential analysis in a background thread
        threading.Thread(target=run_sequential_analysis).start()

    def _run_ampa_poses_analysis_with_selected_method(self):
        """Run AMPA poses analysis using the selected method from radio buttons."""
        selected_method = self.ampa_optimization_method.get()
        if selected_method == "original":
            self._run_ampa_poses_analysis_with_method("original")
        elif selected_method in ["greedy", "ilp"]:
            self._run_ampa_poses_analysis_with_method("optimized")
        else:
            messagebox.showerror("Error", f"Unknown method selected: {selected_method}")

    def _run_ampa_poses_analysis_original(self):
        """Run the original AMPA poses analysis method."""
        self._run_ampa_poses_analysis_with_method("original")

    def _run_ampa_poses_analysis_optimized(self):
        """Run the optimized AMPA poses analysis method."""
        self._run_ampa_poses_analysis_with_method("optimized")

    def _run_ampa_poses_analysis_both(self):
        """Run both original and optimized AMPA poses analysis methods."""
        self._run_ampa_poses_analysis_with_method("both")

    def _run_ampa_poses_analysis_with_method(self, method):
        """Run AMPA poses analysis with specified method (original, optimized, or both)."""
        if not self.csv_path.get():
            messagebox.showerror("Error", "Please select a CSV file first.")
            return
        
        # Check if root directory is specified
        if not self.root_dir.get():
            messagebox.showerror("Error", "Please specify the root directory for tomogram sets.")
            return
        
        # Get parameters from GUI
        try:
            aunp_min_dist = float(self.ampa_aunp_min_dist.get())
            aunp_max_dist = float(self.ampa_aunp_max_dist.get())
            membrane_min_dist = float(self.ampa_membrane_min_dist.get())
            membrane_max_dist = float(self.ampa_membrane_max_dist.get())
            steric_radius = float(self.ampa_steric_radius.get())
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numeric values for distance parameters.")
            return
        
        # Check if cutoffs are disabled
        aunp_no_cutoff = self.ampa_aunp_no_cutoff.get()
        membrane_no_cutoff = self.ampa_membrane_no_cutoff.get()
        
        # Validate distance ranges only if cutoffs are enabled
        if not aunp_no_cutoff and aunp_min_dist >= aunp_max_dist:
            messagebox.showerror("Error", "AuNP minimum distance must be less than maximum distance.")
            return
        
        if not membrane_no_cutoff and membrane_min_dist >= membrane_max_dist:
            messagebox.showerror("Error", "Membrane minimum distance must be less than maximum distance.")
            return
        
        if steric_radius <= 0:
            messagebox.showerror("Error", "Steric radius must be positive.")
            return
        
        # Get output directory (will be constructed per tomogram)
        output_dir_relative = "STT_results/ampa_poses"
        
        # Read CSV to get tomogram names and sets
        try:
            df = pd.read_csv(self.csv_path.get())
            
            # Handle processing mode and starting tomogram selection
            processing_mode = self.processing_mode.get()
            selected_tomogram = self.start_tomogram.get()
            
            if processing_mode == "Single tomogram" and selected_tomogram:
                # Filter to only the selected tomogram
                tomogram_row = df[df['tomoname'] == selected_tomogram]
                if not tomogram_row.empty:
                    df = tomogram_row
                    self._log(f"Processing single tomogram: {selected_tomogram}\n")
                else:
                    self._log(f"Warning: Tomogram {selected_tomogram} not found in CSV\n")
                    
            elif processing_mode == "Start from" and selected_tomogram:
                # Filter to the selected tomogram and all tomograms after it
                tomogram_indices = df[df['tomoname'] == selected_tomogram].index
                if len(tomogram_indices) > 0:
                    start_index = tomogram_indices[0]
                    df = df.iloc[start_index:]
                    self._log(f"Processing from tomogram {selected_tomogram} onwards ({len(df)} tomograms)\n")
                else:
                    self._log(f"Warning: Tomogram {selected_tomogram} not found in CSV\n")
            
            # Check for different possible column names for tomogram names
            if 'tomogram_name' in df.columns:
                tomogram_names = df['tomogram_name'].tolist()
            elif 'tomoname' in df.columns:
                tomogram_names = df['tomoname'].tolist()
            elif 'tomogram' in df.columns:
                tomogram_names = df['tomogram'].tolist()
            else:
                # Try to find any column that might contain tomogram names
                possible_columns = [col for col in df.columns if 'tom' in col.lower() or 'name' in col.lower()]
                if possible_columns:
                    tomogram_names = df[possible_columns[0]].tolist()
                    self._log(f"Using column '{possible_columns[0]}' for tomogram names\n")
                else:
                    messagebox.showerror("Error", "Could not find tomogram name column in CSV. Expected 'tomogram_name', 'tomoname', or 'tomogram'")
                    return
            
            # Get the set column for path construction
            if 'set' in df.columns:
                tomogram_sets = df['set'].tolist()
            else:
                # Default to '15F1' if no set column found
                tomogram_sets = ['15F1'] * len(tomogram_names)
                self._log("No 'set' column found, defaulting to '15F1' for all tomograms\n")

            if 'alignment_dir' not in df.columns:
                messagebox.showerror(
                    "Error",
                    "CSV must include 'alignment_dir' for each tomogram row.",
                )
                return
            tomogram_alignments = []
            for x in df['alignment_dir'].tolist():
                s = str(x).strip()
                if not s or s.lower() == 'nan':
                    messagebox.showerror(
                        "Error",
                        "alignment_dir must be non-empty for every CSV row.",
                    )
                    return
                tomogram_alignments.append(s)
                    
        except Exception as e:
            messagebox.showerror("Error", f"Could not read CSV file: {e}")
            return
        
        method_names = {
            "original": "All Poses",
            "optimized": "Optimized", 
            "both": "Both All Poses and Optimized"
        }
        
        self._log(f"Starting {method_names[method]} AMPA poses analysis for {len(tomogram_names)} tomograms...\n")
        self._log(f"Output directory (relative to each tomogram): {output_dir_relative}\n")
        
        if aunp_no_cutoff:
            self._log(f"AuNP distance range: No cutoff (using all AuNP pairs)\n")
        else:
            self._log(f"AuNP distance range: {aunp_min_dist}-{aunp_max_dist} nm\n")
            
        if membrane_no_cutoff:
            self._log(f"Membrane distance range: No cutoff (using all pairs regardless of membrane distance)\n")
        else:
            self._log(f"Membrane distance range: {membrane_min_dist}-{membrane_max_dist} nm\n")
        
        if method in ["optimized", "both"]:
            self._log(f"Steric radius: {steric_radius} nm\n")
        
        # Process all tomograms
        all_commands = []
        
        for i, (tomogram_name, tomogram_set, align_sub) in enumerate(
            zip(tomogram_names, tomogram_sets, tomogram_alignments), 1
        ):
            self._log(
                f"\nPreparing tomogram {i}/{len(tomogram_names)}: {tomogram_name} "
                f"(set: {tomogram_set}, alignment: {align_sub})\n"
            )
            
            # Build tomogram path using the correct structure
            tomogram_path = os.path.join(self.root_dir.get(), tomogram_set, 'TOP_TOMOS', tomogram_name)
            
            if not os.path.exists(tomogram_path):
                self._log(f"Warning: Tomogram directory not found: {tomogram_path}\n")
                continue
            
            # Get active zones for this tomogram from CSV
            tomogram_row = df[df['tomoname'] == tomogram_name] if 'tomoname' in df.columns else df[df.iloc[:, 0] == tomogram_name]
            aunp_active_zones = None
            
            if not tomogram_row.empty and 'aunp_active_zones' in df.columns:
                aunp_active_zones_str = tomogram_row['aunp_active_zones'].iloc[0]
                if pd.notna(aunp_active_zones_str) and str(aunp_active_zones_str).strip():
                    try:
                        # Parse active zones (can be comma-separated or space-separated)
                        aunp_active_zones_str = str(aunp_active_zones_str).strip()
                        if ',' in aunp_active_zones_str:
                            aunp_active_zones = []
                            for x in aunp_active_zones_str.split(','):
                                x = x.strip()
                                if x.isdigit():
                                    aunp_active_zones.append(int(x))
                                elif x.replace(".", "").isdigit():  # Handle floats like "0.0"
                                    aunp_active_zones.append(int(float(x)))
                        else:
                            aunp_active_zones = []
                            for x in aunp_active_zones_str.split():
                                x = x.strip()
                                if x.isdigit():
                                    aunp_active_zones.append(int(x))
                                elif x.replace(".", "").isdigit():  # Handle floats like "0.0"
                                    aunp_active_zones.append(int(float(x)))
                        self._log(f"Using active zones for {tomogram_name}: {aunp_active_zones}\n")
                    except (ValueError, AttributeError):
                        self._log(f"Warning: Could not parse active zones for {tomogram_name}: {aunp_active_zones_str}\n")
                        aunp_active_zones = None
                else:
                    self._log(f"No active zones specified for {tomogram_name}, using all active zones\n")
            else:
                self._log(f"No active zones specified for {tomogram_name}, using all active zones\n")
            
            # Build commands for this tomogram
            tomogram_commands = []
            
            if method in ["original", "both"]:
                # Original method - save to all_poses directory
                original_output_dir = os.path.join(tomogram_path, align_sub, output_dir_relative, "all_poses")
                os.makedirs(original_output_dir, exist_ok=True)
                
                cli_original = ["python", "-u", "scripts/run_ampa_poses_analysis.py"]
                cli_original += ["--tomogram-path", tomogram_path]
                cli_original += ["--alignment-dir", align_sub]
                cli_original += ["--output-dir", original_output_dir]
                cli_original += ["--method", "original"]  # Explicitly set method to original
                
                # Add distance parameters only if cutoffs are enabled
                if not aunp_no_cutoff:
                    cli_original += ["--aunp-min-distance", str(aunp_min_dist)]
                    cli_original += ["--aunp-max-distance", str(aunp_max_dist)]
                else:
                    cli_original += ["--no-aunp-distance-cutoff"]
                    
                if not membrane_no_cutoff:
                    cli_original += ["--membrane-min-distance", str(membrane_min_dist)]
                    cli_original += ["--membrane-max-distance", str(membrane_max_dist)]
                else:
                    cli_original += ["--no-membrane-distance-cutoff"]
                
                # Add active zones if specified
                if aunp_active_zones is not None:
                    cli_original += ["--aunp-active-zones"] + [str(az) for az in aunp_active_zones]
                
                cli_original += self._cli_aunp_pick_star_pattern_args()
                
                # Add PDB file if specified
                pdb_file = self.ampa_pdb_file.get().strip()
                if pdb_file:
                    cli_original += ["--pdb-file", pdb_file]
                
                tomogram_commands.append(("all_poses", cli_original))
            
            if method in ["optimized", "both"]:
                # Optimized method - determine output directory based on method
                optimization_method = self.ampa_optimization_method.get()
                if optimization_method == "ilp":
                    method_output_dir_name = "ilp"
                else:
                    method_output_dir_name = "greedy"
                
                optimized_output_dir = os.path.join(tomogram_path, align_sub, output_dir_relative, method_output_dir_name)
                os.makedirs(optimized_output_dir, exist_ok=True)
                
                cli_optimized = ["python", "-u", "scripts/run_ampa_poses_analysis.py"]
                cli_optimized += ["--tomogram-path", tomogram_path]
                cli_optimized += ["--alignment-dir", align_sub]
                cli_optimized += ["--output-dir", optimized_output_dir]
                cli_optimized += ["--steric-radius", str(steric_radius)]
                
                # Add distance parameters only if cutoffs are enabled
                if not aunp_no_cutoff:
                    cli_optimized += ["--aunp-min-distance", str(aunp_min_dist)]
                    cli_optimized += ["--aunp-max-distance", str(aunp_max_dist)]
                else:
                    cli_optimized += ["--no-aunp-distance-cutoff"]
                    
                if not membrane_no_cutoff:
                    cli_optimized += ["--membrane-min-distance", str(membrane_min_dist)]
                    cli_optimized += ["--membrane-max-distance", str(membrane_max_dist)]
                else:
                    cli_optimized += ["--no-membrane-distance-cutoff"]
                
                # Add active zones if specified
                if aunp_active_zones is not None:
                    cli_optimized += ["--aunp-active-zones"] + [str(az) for az in aunp_active_zones]
                
                cli_optimized += self._cli_aunp_pick_star_pattern_args()
                
                # Add optimization method
                optimization_method = self.ampa_optimization_method.get()
                cli_optimized += ["--method", optimization_method]
                
                # Add PDB file if specified
                pdb_file = self.ampa_pdb_file.get().strip()
                if pdb_file:
                    cli_optimized += ["--pdb-file", pdb_file]
                
                tomogram_commands.append((method_output_dir_name, cli_optimized))
            
            all_commands.append((tomogram_name, tomogram_set, align_sub, tomogram_commands))
        
        if not all_commands:
            self._log("No valid tomograms found to process.\n")
            return
        
        # Run all commands sequentially with real-time output
        self._log(f"\nStarting {method_names[method]} AMPA poses analysis for {len(all_commands)} tomograms...\n")
        
        # Use threading to run the entire sequence in background while maintaining real-time output
        def run_sequential_analysis():
            all_particles_data_original = []
            all_particles_data_optimized = []
            all_aunps_data_original = []
            all_aunps_data_optimized = []
            all_unpaired_data_original = []
            all_unpaired_data_optimized = []
            successful_tomograms = []
            
            for i, (tomogram_name, tomogram_set, align_sub, tomogram_commands) in enumerate(all_commands, 1):
                tomogram_path = os.path.join(
                    self.root_dir.get(), tomogram_set, "TOP_TOMOS", tomogram_name
                )
                self._log(f"\n{'='*60}\n")
                self._log(f"Processing tomogram {i}/{len(all_commands)}: {tomogram_name}\n")
                self._log(f"{'='*60}\n")
                
                for method_type, cli in tomogram_commands:
                    self._log(f"\nRunning {method_type} method for {tomogram_name}...\n")
                    self._log(f"Command: {' '.join(cli)}\n")
                    # Check if --method is in the command
                    if '--method' in cli:
                        method_idx = cli.index('--method')
                        if method_idx + 1 < len(cli):
                            self._log(f"Method specified in command: {cli[method_idx + 1]}\n")
                    else:
                        self._log(f"WARNING: --method flag not found in command!\n")
                    
                    # Run the subprocess and wait for completion
                    self._run_subprocess(cli, os.environ.copy())
                    
                    # Try to load the results for combining
                    try:
                        import starfile
                        import pandas as pd
                        
                        # Determine output directory based on method
                        if method_type == "all_poses":
                            # Try the direct all_poses directory first
                            individual_output_dir = os.path.join(tomogram_path, align_sub, output_dir_relative, "all_poses")
                            # If that doesn't exist, check for greedy subdirectory (legacy/fallback)
                            if not os.path.exists(individual_output_dir):
                                greedy_fallback = os.path.join(tomogram_path, align_sub, output_dir_relative, "all_poses", "greedy")
                                if os.path.exists(greedy_fallback):
                                    individual_output_dir = greedy_fallback
                                    self._log(f"Using fallback directory: {greedy_fallback}\n")
                        else:  # optimized method - use the actual method directory
                            optimization_method = self.ampa_optimization_method.get()
                            if optimization_method == "ilp":
                                method_dir = "ilp"
                            else:
                                method_dir = "greedy"
                            individual_output_dir = os.path.join(tomogram_path, align_sub, output_dir_relative, method_dir)
                        
                        # Load the particles file
                        if not os.path.exists(individual_output_dir):
                            self._log(f"Warning: Output directory does not exist: {individual_output_dir}\n")
                        else:
                            particles_files = [f for f in os.listdir(individual_output_dir) if f.endswith('.star') and 'ampa_poses' in f and '_aunps' not in f and '_unpaired' not in f and '_all_aunps' not in f]
                            if particles_files:
                                particles_file_path = os.path.join(individual_output_dir, particles_files[0])
                                self._log(f"Loading particles from: {particles_file_path}\n")
                                particles_data = starfile.read(particles_file_path)
                                if 'particles' in particles_data:
                                    particles_df = particles_data['particles'].copy()
                                    particles_df['rlnTomoName'] = tomogram_name  # Ensure tomogram name is set
                                    if method_type == "all_poses":
                                        all_particles_data_original.append(particles_df)
                                        self._log(f"Added {len(particles_df)} particles to all_poses collection (total: {len(all_particles_data_original)} tomograms)\n")
                                    else:
                                        all_particles_data_optimized.append(particles_df)
                                        self._log(f"Added {len(particles_df)} particles to optimized collection (total: {len(all_particles_data_optimized)} tomograms)\n")
                            else:
                                self._log(f"Warning: No particles files found in {individual_output_dir}\n")
                            
                            # Load the AuNPs file
                            aunps_files = [f for f in os.listdir(individual_output_dir) if f.endswith('.star') and 'ampa_poses' in f and '_paired_aunps' in f]
                            if aunps_files:
                                aunps_file_path = os.path.join(individual_output_dir, aunps_files[0])
                                aunps_data = starfile.read(aunps_file_path)
                                if 'particles' in aunps_data:
                                    aunps_df = aunps_data['particles'].copy()
                                    aunps_df['rlnTomoName'] = tomogram_name  # Ensure tomogram name is set
                                    if method_type == "all_poses":
                                        all_aunps_data_original.append(aunps_df)
                                    else:
                                        all_aunps_data_optimized.append(aunps_df)
                            
                            # Load unpaired AuNPs file (both methods now have this)
                            unpaired_files = [f for f in os.listdir(individual_output_dir) if f.endswith('.star') and '_unpaired_aunps' in f]
                            if unpaired_files:
                                unpaired_file_path = os.path.join(individual_output_dir, unpaired_files[0])
                                unpaired_data = starfile.read(unpaired_file_path)
                                if 'particles' in unpaired_data:
                                    unpaired_df = unpaired_data['particles'].copy()
                                    unpaired_df['rlnTomoName'] = tomogram_name  # Ensure tomogram name is set
                                    if method_type == "all_poses":
                                        all_unpaired_data_original.append(unpaired_df)
                                    else:
                                        all_unpaired_data_optimized.append(unpaired_df)
                    
                    except Exception as e:
                        self._log(f"Warning: Could not load results for {tomogram_name} ({method_type}): {e}\n")
                
                successful_tomograms.append(tomogram_name)
            
            # Save combined star files
            self._log(f"\n{'='*60}\n")
            self._log(f"Preparing to save combined results...\n")
            self._log(f"all_poses data: {len(all_particles_data_original)} tomograms\n")
            self._log(f"optimized data: {len(all_particles_data_optimized)} tomograms\n")
            self._log(f"{'='*60}\n")
            
            try:
                # Create results/ampa_poses directory
                combined_output_dir = "results/ampa_poses"
                os.makedirs(combined_output_dir, exist_ok=True)
                self._log(f"Created/verified directory: {combined_output_dir}\n")
                
                # Build parameter strings for filenames (same format as individual tomogram files)
                if aunp_no_cutoff:
                    aunp_str = "aunpNONE"
                else:
                    aunp_str = f"aunp{aunp_min_dist}-{aunp_max_dist}nm"
                
                if membrane_no_cutoff:
                    membrane_str = "memNONE"
                else:
                    membrane_str = f"mem{membrane_min_dist}-{membrane_max_dist}nm"
                
                # Save original method results (all poses)
                if all_particles_data_original:
                    self._log(f"Combining {len(all_particles_data_original)} all_poses datasets...\n")
                    combined_particles_original = pd.concat(all_particles_data_original, ignore_index=True)
                    original_filename = f"all_ampa_poses_all_poses_{aunp_str}_{membrane_str}.star"
                    starfile.write({
                        'particles': combined_particles_original,
                        'optics': pd.DataFrame([{'rlnOpticsGroup': 1}])
                    }, os.path.join(combined_output_dir, original_filename))
                    self._log(f"Saved combined all poses AMPA poses to {combined_output_dir}/{original_filename}\n")
                    
                    if all_aunps_data_original:
                        combined_aunps_original = pd.concat(all_aunps_data_original, ignore_index=True)
                        original_aunps_filename = f"all_ampa_poses_all_poses_{aunp_str}_{membrane_str}_paired_aunps.star"
                        starfile.write({
                            'particles': combined_aunps_original,
                            'optics': pd.DataFrame([{'rlnOpticsGroup': 1}])
                        }, os.path.join(combined_output_dir, original_aunps_filename))
                        self._log(f"Saved combined all poses paired AuNPs to {combined_output_dir}/{original_aunps_filename}\n")
                
                # Save optimized method results
                if all_particles_data_optimized:
                    combined_particles_optimized = pd.concat(all_particles_data_optimized, ignore_index=True)
                    # Determine the method name for file naming
                    optimization_method = self.ampa_optimization_method.get()
                    method_suffix = optimization_method if optimization_method in ["greedy", "ilp"] else "optimized"
                    
                    optimized_filename = f"all_ampa_poses_{method_suffix}_{aunp_str}_{membrane_str}_steric{steric_radius}nm.star"
                    starfile.write({
                        'particles': combined_particles_optimized,
                        'optics': pd.DataFrame([{'rlnOpticsGroup': 1}])
                    }, os.path.join(combined_output_dir, optimized_filename))
                    self._log(f"Saved combined {method_suffix} AMPA poses to {combined_output_dir}/{optimized_filename}\n")
                    
                    if all_aunps_data_optimized:
                        combined_aunps_optimized = pd.concat(all_aunps_data_optimized, ignore_index=True)
                        optimized_aunps_filename = f"all_ampa_poses_{method_suffix}_{aunp_str}_{membrane_str}_steric{steric_radius}nm_paired_aunps.star"
                        starfile.write({
                            'particles': combined_aunps_optimized,
                            'optics': pd.DataFrame([{'rlnOpticsGroup': 1}])
                        }, os.path.join(combined_output_dir, optimized_aunps_filename))
                        self._log(f"Saved combined {method_suffix} paired AuNPs to {combined_output_dir}/{optimized_aunps_filename}\n")
                    
                    # Save unpaired AuNPs for all poses method
                    if all_unpaired_data_original:
                        combined_unpaired_original = pd.concat(all_unpaired_data_original, ignore_index=True)
                        unpaired_original_filename = f"all_ampa_poses_all_poses_{aunp_str}_{membrane_str}_unpaired_aunps.star"
                        starfile.write({
                            'particles': combined_unpaired_original,
                            'optics': pd.DataFrame([{'rlnOpticsGroup': 1}])
                        }, os.path.join(combined_output_dir, unpaired_original_filename))
                        self._log(f"Saved combined all poses unpaired AuNPs to {combined_output_dir}/{unpaired_original_filename}\n")
                    
                    # Save unpaired AuNPs for optimized method
                    if all_unpaired_data_optimized:
                        combined_unpaired_optimized = pd.concat(all_unpaired_data_optimized, ignore_index=True)
                        unpaired_optimized_filename = f"all_ampa_poses_{method_suffix}_{aunp_str}_{membrane_str}_steric{steric_radius}nm_unpaired_aunps.star"
                        starfile.write({
                            'particles': combined_unpaired_optimized,
                            'optics': pd.DataFrame([{'rlnOpticsGroup': 1}])
                        }, os.path.join(combined_output_dir, unpaired_optimized_filename))
                        self._log(f"Saved combined {method_suffix} unpaired AuNPs to {combined_output_dir}/{unpaired_optimized_filename}\n")
                
                # Generate comparison report if both methods were run
                if method == "both" and all_particles_data_original and all_particles_data_optimized:
                    self._generate_ampa_comparison_report(
                        combined_output_dir, 
                        combined_particles_original, 
                        combined_particles_optimized,
                        combined_aunps_original if all_aunps_data_original else None,
                        combined_aunps_optimized if all_aunps_data_optimized else None,
                        combined_unpaired_original if all_unpaired_data_original else None,
                        combined_unpaired_optimized if all_unpaired_data_optimized else None
                    )
                    
            except Exception as e:
                import traceback
                self._log(f"Error saving combined results: {e}\n")
                self._log(f"Traceback: {traceback.format_exc()}\n")
            
            self._log(f"\n{method_names[method]} AMPA poses analysis completed for {len(all_commands)} tomograms.\n")
            self._log(f"Results saved to: {output_dir_relative} within each tomogram's STT_results directory\n")
            if all_particles_data_original or all_particles_data_optimized:
                self._log(f"Combined results saved to: results/ampa_poses/\n")
        
        # Start the sequential analysis in a background thread
        threading.Thread(target=run_sequential_analysis).start()

    def _generate_ampa_comparison_report(self, output_dir, original_particles, optimized_particles, 
                                       original_aunps=None, optimized_aunps=None, 
                                       original_unpaired_aunps=None, optimized_unpaired_aunps=None):
        """Generate a comparison report between original and optimized AMPA analysis methods."""
        try:
            import pandas as pd
            import numpy as np
            
            # Calculate steric clashes for all poses method using same criteria as optimized method
            all_poses_steric_clashes = 0
            if len(original_particles) > 1:
                # Extract AMPA positions from all poses method
                all_poses_positions = []
                for _, row in original_particles.iterrows():
                    all_poses_positions.append([row['rlnCoordinateX'], row['rlnCoordinateY'], row['rlnCoordinateZ']])
                
                if len(all_poses_positions) > 1:
                    all_poses_positions = np.array(all_poses_positions)
                    
                    # Check for steric clashes using same criteria as optimized method (5.0 nm minimum distance)
                    clashes = []
                    for i in range(len(all_poses_positions)):
                        for j in range(i + 1, len(all_poses_positions)):
                            distance = np.linalg.norm(all_poses_positions[i] - all_poses_positions[j])
                            if distance < 5.0:  # Same steric radius as optimized method
                                clashes.append((i, j))
                    
                    all_poses_steric_clashes = len(clashes)
            
            # Create comparison summary
            optimization_method = self.ampa_optimization_method.get()
            method_display_name = optimization_method.upper() if optimization_method in ["greedy", "ilp"] else "Optimized"
            comparison_data = {
                'Method': ['All Poses', method_display_name],
                'Total_AMPA_Poses': [
                    len(original_particles),
                    len(optimized_particles)
                ],
                'Total_AuNPs_Used': [
                    len(original_aunps) if original_aunps is not None else 0,
                    len(optimized_aunps) if optimized_aunps is not None else 0
                ],
                'Unpaired_AuNPs': [
                    len(original_unpaired_aunps) if original_unpaired_aunps is not None else 0,  # All poses method now tracks this
                    len(optimized_unpaired_aunps) if optimized_unpaired_aunps is not None else 0   # Optimized method tracks this
                ],
                'Pairing_Efficiency': [
                    'N/A',  # All poses method doesn't calculate this
                    len(optimized_aunps) / (len(optimized_aunps) + len(optimized_unpaired_aunps)) * 100 if optimized_aunps is not None and optimized_unpaired_aunps is not None else 'N/A'
                ],
                'Steric_Clashes': [
                    all_poses_steric_clashes,  # Actual calculated clashes in all poses method
                    0  # Optimized method eliminates clashes
                ],
                'Quality_Assessment': [
                    'Includes overpicked poses with potential steric clashes',
                    'High-quality poses with no steric clashes'
                ]
            }
            
            comparison_df = pd.DataFrame(comparison_data)
            comparison_file = os.path.join(output_dir, "ampa_poses_comparison.csv")
            comparison_df.to_csv(comparison_file, index=False)
            
            self._log(f"Generated comparison report: {comparison_file}\n")
            
            # Calculate quality vs quantity metrics
            if len(original_particles) > 0:
                poses_reduction = (len(original_particles) - len(optimized_particles)) / len(original_particles) * 100
                self._log(f"AMPA poses reduction (overpicking eliminated): {poses_reduction:.1f}%\n")
                self._log(f"All poses method: {len(original_particles)} poses with {all_poses_steric_clashes} steric clashes\n")
                # Get the actual method name for display
                optimization_method = self.ampa_optimization_method.get()
                method_display_name = optimization_method.upper() if optimization_method in ["greedy", "ilp"] else "Optimized"
                self._log(f"{method_display_name} method: {len(optimized_particles)} high-quality poses with 0 steric clashes\n")
                if all_poses_steric_clashes > 0:
                    self._log(f"Steric clashes eliminated: {all_poses_steric_clashes} (biologically impossible poses removed)\n")
            
            if optimized_aunps is not None and optimized_unpaired_aunps is not None:
                total_aunps = len(optimized_aunps) + len(optimized_unpaired_aunps)
                if total_aunps > 0:
                    pairing_efficiency = len(optimized_aunps) / total_aunps * 100
                    self._log(f"Optimized pairing efficiency: {pairing_efficiency:.1f}%\n")
            
        except Exception as e:
            self._log(f"Error generating comparison report: {e}\n")

        """Generate a comprehensive PDF showing all zonogram images from all tomograms."""
        if not self.csv_path.get():
            messagebox.showerror("Error", "Please select a CSV file first.")
            return
        
        # Check if root directory is specified
        if not self.root_dir.get():
            messagebox.showerror("Error", "Please specify the root directory for tomogram sets.")
            return
        
        # Use threading to run the PDF generation in background while maintaining real-time output
        def generate_pdf_background():
            try:
                self._log("Starting zonogram PDF generation...\n")
                
                # Read CSV to get tomogram names and sets
                df = pd.read_csv(self.csv_path.get())
                
                # Handle processing mode and starting tomogram selection
                processing_mode = self.processing_mode.get()
                selected_tomogram = self.start_tomogram.get()
                
                if processing_mode == "Single tomogram" and selected_tomogram:
                    tomogram_row = df[df['tomoname'] == selected_tomogram]
                    if not tomogram_row.empty:
                        df = tomogram_row
                        self._log(f"Generating PDF for single tomogram: {selected_tomogram}\n")
                    else:
                        self._log(f"Warning: Tomogram {selected_tomogram} not found in CSV\n")
                        return
                        
                elif processing_mode == "Start from" and selected_tomogram:
                    tomogram_indices = df[df['tomoname'] == selected_tomogram].index
                    if len(tomogram_indices) > 0:
                        start_index = tomogram_indices[0]
                        df = df.iloc[start_index:]
                        self._log(f"Generating PDF from tomogram {selected_tomogram} onwards ({len(df)} tomograms)\n")
                    else:
                        self._log(f"Warning: Tomogram {selected_tomogram} not found in CSV\n")
                        return
                
                # Check for different possible column names for tomogram names
                if 'tomogram_name' in df.columns:
                    tomogram_names = df['tomogram_name'].tolist()
                elif 'tomoname' in df.columns:
                    tomogram_names = df['tomoname'].tolist()
                elif 'tomogram' in df.columns:
                    tomogram_names = df['tomogram'].tolist()
                else:
                    possible_columns = [col for col in df.columns if 'tom' in col.lower() or 'name' in col.lower()]
                    if possible_columns:
                        tomogram_names = df[possible_columns[0]].tolist()
                        self._log(f"Using column '{possible_columns[0]}' for tomogram names\n")
                    else:
                        messagebox.showerror("Error", "Could not find tomogram name column in CSV.")
                        return
                
                # Get the set column for path construction
                if 'set' in df.columns:
                    tomogram_sets = df['set'].tolist()
                else:
                    tomogram_sets = ['15F1'] * len(tomogram_names)
                    self._log("No 'set' column found, defaulting to '15F1' for all tomograms\n")

                if 'alignment_dir' not in df.columns:
                    messagebox.showerror("Error", "CSV must include 'alignment_dir' for each tomogram row.")
                    return
                tomogram_alignments = []
                for x in df['alignment_dir'].tolist():
                    s = str(x).strip()
                    if not s or s.lower() == 'nan':
                        messagebox.showerror("Error", "alignment_dir must be non-empty for every CSV row.")
                        return
                    tomogram_alignments.append(s)
                
                # Create output directory
                output_dir = Path("results/visualizations/pdf_summaries")
                output_dir.mkdir(parents=True, exist_ok=True)
                pdf_path = output_dir / "all_zonograms_summary.pdf"
                
                self._log(f"Generating PDF: {pdf_path}\n")
                
                # Import required libraries
                from reportlab.lib.pagesizes import letter, A4
                from reportlab.platypus import SimpleDocTemplate, Image, PageBreak, Spacer, Paragraph
                from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
                from reportlab.lib.units import inch
                from reportlab.lib import colors
                
                # Create PDF document
                doc = SimpleDocTemplate(str(pdf_path), pagesize=A4)
                story = []
                styles = getSampleStyleSheet()
                
                # Create custom style for tomogram names
                title_style = ParagraphStyle(
                    'CustomTitle',
                    parent=styles['Heading1'],
                    fontSize=16,
                    spaceAfter=20,
                    textColor=colors.darkblue
                )
                
                # Process each tomogram
                for i, (tomogram_name, tomogram_set, align_sub) in enumerate(
                    zip(tomogram_names, tomogram_sets, tomogram_alignments), 1
                ):
                    self._log(f"Processing tomogram {i}/{len(tomogram_names)}: {tomogram_name} ({align_sub})\n")
                    
                    # Build tomogram path
                    tomogram_path = Path(self.root_dir.get()) / tomogram_set / "TOP_TOMOS" / tomogram_name
                    active_zonograms_dir = tomogram_path / align_sub / "STT_results" / "active_zonograms"
                    
                    if not active_zonograms_dir.exists():
                        self._log(f"Warning: Active zonograms directory not found: {active_zonograms_dir}\n")
                        continue
                    
                    # Add tomogram name as title
                    story.append(Paragraph(f"Tomogram: {tomogram_name}", title_style))
                    story.append(Spacer(1, 10))
                    
                    # Find regular active zonogram files (aunps_by_cluster.png)
                    regular_zonogram_files = list(active_zonograms_dir.glob("*_active_zonogram_*_selected_aunps_by_cluster.png"))
                    
                    # Add regular active zonograms first
                    for zonogram_file in sorted(regular_zonogram_files):
                        try:
                            # Get zone name from filename
                            zone_name = zonogram_file.stem.split('_active_zonogram_')[1].split('_selected_aunps_by_cluster')[0]
                            
                            # Add zone name as subtitle
                            zone_style = ParagraphStyle(
                                'ZoneTitle',
                                parent=styles['Heading2'],
                                fontSize=12,
                                spaceAfter=10,
                                textColor=colors.darkgreen
                            )
                            story.append(Paragraph(f"Active Zone: {zone_name}", zone_style))
                            
                            # Add the image (preserve aspect ratio but ensure it fits on page)
                            # First, get the original image dimensions
                            from PIL import Image as PILImage
                            pil_img = PILImage.open(str(zonogram_file))
                            orig_width, orig_height = pil_img.size
                            aspect_ratio = orig_width / orig_height
                            
                            # Calculate maximum dimensions that fit on page
                            max_width = 7 * inch
                            max_height = 600  # Leave some margin
                            
                            # Calculate dimensions that preserve aspect ratio and fit within limits
                            if max_width / aspect_ratio <= max_height:
                                # Width is the limiting factor
                                final_width = max_width
                                final_height = max_width / aspect_ratio
                            else:
                                # Height is the limiting factor
                                final_height = max_height
                                final_width = max_height * aspect_ratio
                            
                            img = Image(str(zonogram_file), width=final_width, height=final_height)
                            story.append(img)
                            story.append(Spacer(1, 10))
                            
                            self._log(f"  Added regular zonogram: {zone_name}\n")
                        except Exception as e:
                            self._log(f"  Error adding regular zonogram {zonogram_file}: {e}\n")
                    
                    # Find mini zonogram comparison files
                    mini_zonogram_files = list(active_zonograms_dir.glob("*_mini_zonogram_cluster_*_comparison.png"))
                    
                    if mini_zonogram_files:
                        # Add mini zonograms section title
                        mini_style = ParagraphStyle(
                            'MiniTitle',
                            parent=styles['Heading2'],
                            fontSize=12,
                            spaceAfter=10,
                            textColor=colors.darkred
                        )
                        story.append(Paragraph("Mini Zonograms (Small Clusters)", mini_style))
                        story.append(Spacer(1, 5))
                        
                        # Add mini zonograms in two columns
                        for j in range(0, len(mini_zonogram_files), 2):
                            # Create a table-like layout for two columns
                            from reportlab.platypus import Table, TableStyle
                            from reportlab.lib import colors
                            
                            row_data = []
                            for k in range(2):
                                if j + k < len(mini_zonogram_files):
                                    mini_file = mini_zonogram_files[j + k]
                                    try:
                                        # Get cluster number from filename
                                        cluster_num = mini_file.stem.split('_cluster_')[1].split('_comparison')[0]
                                        
                                        # Add image (smaller size for two columns, preserve aspect ratio but ensure it fits)
                                        # First, get the original image dimensions
                                        from PIL import Image as PILImage
                                        pil_img = PILImage.open(str(mini_file))
                                        orig_width, orig_height = pil_img.size
                                        aspect_ratio = orig_width / orig_height
                                        
                                        # Calculate maximum dimensions that fit in two columns
                                        max_width = 3.5 * inch
                                        max_height = 300  # Leave some margin for mini zonograms
                                        
                                        # Calculate dimensions that preserve aspect ratio and fit within limits
                                        if max_width / aspect_ratio <= max_height:
                                            # Width is the limiting factor
                                            final_width = max_width
                                            final_height = max_width / aspect_ratio
                                        else:
                                            # Height is the limiting factor
                                            final_height = max_height
                                            final_width = max_height * aspect_ratio
                                        
                                        img = Image(str(mini_file), width=final_width, height=final_height)
                                        
                                        row_data.append([img])
                                        self._log(f"  Added mini zonogram: Cluster {cluster_num}\n")
                                    except Exception as e:
                                        self._log(f"  Error adding mini zonogram {mini_file}: {e}\n")
                                        row_data.append([""])
                                else:
                                    row_data.append([""])
                            
                            if any(cell != [""] for cell in row_data):
                                # Create table for this row
                                table = Table(row_data, colWidths=[3.5*inch, 3.5*inch])
                                table.setStyle(TableStyle([
                                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                                    ('LEFTPADDING', (0, 0), (-1, -1), 5),
                                    ('RIGHTPADDING', (0, 0), (-1, -1), 5),
                                ]))
                                story.append(table)
                                story.append(Spacer(1, 10))
                    
                    # Add page break between tomograms
                    if i < len(tomogram_names):
                        story.append(PageBreak())
                
                # Build PDF
                self._log("Building PDF...\n")
                doc.build(story)
                
                self._log(f"PDF generation completed successfully!\n")
                self._log(f"PDF saved to: {pdf_path}\n")
                
                # Open the PDF
                try:
                    webbrowser.open(f"file://{pdf_path.absolute()}")
                    self._log("PDF opened in browser.\n")
                except Exception as e:
                    self._log(f"Could not open PDF automatically: {e}\n")
                    self._log(f"Please open manually: {pdf_path}\n")
                
            except Exception as e:
                self._log(f"Error generating zonogram PDF: {e}\n")
                import traceback
                self._log(f"Traceback: {traceback.format_exc()}\n")
        
        # Start the PDF generation in a background thread
        threading.Thread(target=generate_pdf_background).start()

    def _generate_mini_zonogram_pdf(self):
        """Generate a PDF showing only the mini zonogram comparison images from all tomograms."""
        if not self.csv_path.get():
            messagebox.showerror("Error", "Please select a CSV file first.")
            return
        
        # Check if root directory is specified
        if not self.root_dir.get():
            messagebox.showerror("Error", "Please specify the root directory for tomogram sets.")
            return
        
        # Use threading to run the PDF generation in background while maintaining real-time output
        def generate_pdf_background():
            try:
                self._log("Starting mini zonogram PDF generation...\n")
                
                # Read CSV to get tomogram names and sets
                df = pd.read_csv(self.csv_path.get())
                
                # Handle processing mode and starting tomogram selection
                processing_mode = self.processing_mode.get()
                selected_tomogram = self.start_tomogram.get()
                
                if processing_mode == "Single tomogram" and selected_tomogram:
                    tomogram_row = df[df['tomoname'] == selected_tomogram]
                    if not tomogram_row.empty:
                        df = tomogram_row
                        self._log(f"Generating mini zonogram PDF for single tomogram: {selected_tomogram}\n")
                    else:
                        self._log(f"Warning: Tomogram {selected_tomogram} not found in CSV\n")
                        return
                        
                elif processing_mode == "Start from" and selected_tomogram:
                    tomogram_indices = df[df['tomoname'] == selected_tomogram].index
                    if len(tomogram_indices) > 0:
                        start_index = tomogram_indices[0]
                        df = df.iloc[start_index:]
                        self._log(f"Generating mini zonogram PDF from tomogram {selected_tomogram} onwards ({len(df)} tomograms)\n")
                    else:
                        self._log(f"Warning: Tomogram {selected_tomogram} not found in CSV\n")
                        return
                
                # Check for different possible column names for tomogram names
                if 'tomogram_name' in df.columns:
                    tomogram_names = df['tomogram_name'].tolist()
                elif 'tomoname' in df.columns:
                    tomogram_names = df['tomoname'].tolist()
                elif 'tomogram' in df.columns:
                    tomogram_names = df['tomogram'].tolist()
                else:
                    possible_columns = [col for col in df.columns if 'tom' in col.lower() or 'name' in col.lower()]
                    if possible_columns:
                        tomogram_names = df[possible_columns[0]].tolist()
                        self._log(f"Using column '{possible_columns[0]}' for tomogram names\n")
                    else:
                        messagebox.showerror("Error", "Could not find tomogram name column in CSV.")
                        return
                
                # Get the set column for path construction
                if 'set' in df.columns:
                    tomogram_sets = df['set'].tolist()
                else:
                    tomogram_sets = ['15F1'] * len(tomogram_names)
                    self._log("No 'set' column found, defaulting to '15F1' for all tomograms\n")

                if 'alignment_dir' not in df.columns:
                    messagebox.showerror("Error", "CSV must include 'alignment_dir' for each tomogram row.")
                    return
                tomogram_alignments = []
                for x in df['alignment_dir'].tolist():
                    s = str(x).strip()
                    if not s or s.lower() == 'nan':
                        messagebox.showerror("Error", "alignment_dir must be non-empty for every CSV row.")
                        return
                    tomogram_alignments.append(s)
                
                # Create output directory
                output_dir = Path("results/visualizations/pdf_summaries")
                output_dir.mkdir(parents=True, exist_ok=True)
                pdf_path = output_dir / "mini_zonograms_summary.pdf"
                pdf_path_4aunps = output_dir / "mini_zonograms_4aunps_summary.pdf"
                
                self._log(f"Generating mini zonogram PDFs: {pdf_path} and {pdf_path_4aunps}\n")
                
                # Import required libraries
                from reportlab.lib.pagesizes import letter, A4
                from reportlab.platypus import SimpleDocTemplate, Image, PageBreak, Spacer, Paragraph
                from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
                from reportlab.lib.units import inch
                from reportlab.lib import colors
                
                # Create PDF documents
                doc = SimpleDocTemplate(str(pdf_path), pagesize=A4)
                doc_4aunps = SimpleDocTemplate(str(pdf_path_4aunps), pagesize=A4)
                story = []
                story_4aunps = []
                styles = getSampleStyleSheet()
                
                # Create custom style for tomogram names
                title_style = ParagraphStyle(
                    'CustomTitle',
                    parent=styles['Heading1'],
                    fontSize=16,
                    spaceAfter=20,
                    textColor=colors.darkblue
                )
                
                # Process each tomogram
                for i, (tomogram_name, tomogram_set, align_sub) in enumerate(
                    zip(tomogram_names, tomogram_sets, tomogram_alignments), 1
                ):
                    self._log(f"Processing tomogram {i}/{len(tomogram_names)}: {tomogram_name} ({align_sub})\n")
                    
                    # Build tomogram path
                    tomogram_path = Path(self.root_dir.get()) / tomogram_set / "TOP_TOMOS" / tomogram_name
                    active_zonograms_dir = tomogram_path / align_sub / "STT_results" / "active_zonograms"
                    
                    if not active_zonograms_dir.exists():
                        self._log(f"Warning: Active zonograms directory not found: {active_zonograms_dir}\n")
                        continue
                    
                    # Find mini zonogram comparison files
                    mini_zonogram_files = list(active_zonograms_dir.glob("*_mini_zonogram_cluster_*_comparison.png"))
                    
                    # Get cluster data to identify clusters with 4 AuNPs
                    cluster_data_path = tomogram_path / align_sub / "STT_results" / "aunps" / "aunp_clusters.star"
                    clusters_with_4_aunps = set()
                    
                    if cluster_data_path.exists():
                        try:
                            import starfile
                            cluster_df = starfile.read(cluster_data_path)
                            # Count AuNPs per cluster
                            cluster_counts = cluster_df['aunp_cluster'].value_counts()
                            # Get clusters with exactly 4 AuNPs
                            clusters_with_4_aunps = set(cluster_counts[cluster_counts == 4].index)
                            self._log(f"  Found {len(clusters_with_4_aunps)} clusters with 4 AuNPs: {sorted(clusters_with_4_aunps)}\n")
                        except Exception as e:
                            self._log(f"  Warning: Could not read cluster data: {e}\n")
                    
                    if mini_zonogram_files:
                        # Add tomogram name as title
                        story.append(Paragraph(f"Tomogram: {tomogram_name}", title_style))
                        story.append(Spacer(1, 10))
                        
                        # Also add to 4 AuNP PDF if this tomogram has any 4 AuNP clusters
                        has_4aunp_clusters = any(
                            int(f.stem.split('_cluster_')[1].split('_comparison')[0]) in clusters_with_4_aunps 
                            for f in mini_zonogram_files
                        )
                        if has_4aunp_clusters:
                            story_4aunps.append(Paragraph(f"Tomogram: {tomogram_name}", title_style))
                            story_4aunps.append(Spacer(1, 10))
                        
                        # Add mini zonograms in two columns
                        for j in range(0, len(mini_zonogram_files), 2):
                            # Create a table-like layout for two columns
                            from reportlab.platypus import Table, TableStyle
                            from reportlab.lib import colors
                            
                            row_data = []
                            for k in range(2):
                                if j + k < len(mini_zonogram_files):
                                    mini_file = mini_zonogram_files[j + k]
                                    try:
                                        # Get cluster number from filename
                                        cluster_num = mini_file.stem.split('_cluster_')[1].split('_comparison')[0]
                                        
                                        # Add image (preserve aspect ratio but ensure it fits)
                                        # First, get the original image dimensions
                                        from PIL import Image as PILImage
                                        pil_img = PILImage.open(str(mini_file))
                                        orig_width, orig_height = pil_img.size
                                        aspect_ratio = orig_width / orig_height
                                        
                                        # Calculate maximum dimensions that fit in two columns
                                        max_width = 3.5 * inch
                                        max_height = 300  # Leave some margin for mini zonograms
                                        
                                        # Calculate dimensions that preserve aspect ratio and fit within limits
                                        if max_width / aspect_ratio <= max_height:
                                            # Width is the limiting factor
                                            final_width = max_width
                                            final_height = max_width / aspect_ratio
                                        else:
                                            # Height is the limiting factor
                                            final_height = max_height
                                            final_width = max_height * aspect_ratio
                                        
                                        img = Image(str(mini_file), width=final_width, height=final_height)
                                        
                                        row_data.append([img])
                                        self._log(f"  Added mini zonogram: Cluster {cluster_num}\n")
                                    except Exception as e:
                                        self._log(f"  Error adding mini zonogram {mini_file}: {e}\n")
                                        row_data.append([""])
                                else:
                                    row_data.append([""])
                            
                            if any(cell != [""] for cell in row_data):
                                # Create table for this row
                                table = Table(row_data, colWidths=[3.5*inch, 3.5*inch])
                                table.setStyle(TableStyle([
                                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                                    ('LEFTPADDING', (0, 0), (-1, -1), 5),
                                    ('RIGHTPADDING', (0, 0), (-1, -1), 5),
                                ]))
                                story.append(table)
                                story.append(Spacer(1, 10))
                        
                        # Add 4 AuNP clusters to the separate PDF using same two-column layout
                        clusters_4aunps = [f for f in mini_zonogram_files 
                                         if int(f.stem.split('_cluster_')[1].split('_comparison')[0]) in clusters_with_4_aunps]
                        
                        if clusters_4aunps:
                            for j in range(0, len(clusters_4aunps), 2):
                                # Create a table-like layout for two columns (same as main PDF)
                                from reportlab.platypus import Table, TableStyle
                                from reportlab.lib import colors
                                
                                row_data = []
                                for k in range(2):
                                    if j + k < len(clusters_4aunps):
                                        mini_file = clusters_4aunps[j + k]
                                        try:
                                            # Get cluster number from filename
                                            cluster_num = mini_file.stem.split('_cluster_')[1].split('_comparison')[0]
                                            
                                            # Add image (preserve aspect ratio but ensure it fits) - same as main PDF
                                            from PIL import Image as PILImage
                                            pil_img = PILImage.open(str(mini_file))
                                            orig_width, orig_height = pil_img.size
                                            aspect_ratio = orig_width / orig_height
                                            
                                            # Calculate maximum dimensions that fit in two columns (same as main PDF)
                                            max_width = 3.5 * inch
                                            max_height = 300  # Leave some margin for mini zonograms
                                            
                                            # Calculate dimensions that preserve aspect ratio and fit within limits
                                            if max_width / aspect_ratio <= max_height:
                                                # Width is the limiting factor
                                                final_width = max_width
                                                final_height = max_width / aspect_ratio
                                            else:
                                                # Height is the limiting factor
                                                final_height = max_height
                                                final_width = max_height * aspect_ratio
                                            
                                            img = Image(str(mini_file), width=final_width, height=final_height)
                                            
                                            row_data.append([img])
                                            self._log(f"  Added to 4 AuNP PDF: Cluster {cluster_num}\n")
                                        except Exception as e:
                                            self._log(f"  Error adding 4 AuNP mini zonogram {mini_file}: {e}\n")
                                            row_data.append([""])
                                    else:
                                        row_data.append([""])
                                
                                if any(cell != [""] for cell in row_data):
                                    # Create table for this row (same as main PDF)
                                    table = Table(row_data, colWidths=[3.5*inch, 3.5*inch])
                                    table.setStyle(TableStyle([
                                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                                        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                                        ('LEFTPADDING', (0, 0), (-1, -1), 5),
                                        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
                                    ]))
                                    story_4aunps.append(table)
                                    story_4aunps.append(Spacer(1, 10))
                    else:
                        self._log(f"  No mini zonograms found for {tomogram_name}\n")
                    
                    # Add page break between tomograms
                    if i < len(tomogram_names):
                        story.append(PageBreak())
                
                # Build PDFs
                self._log("Building mini zonogram PDFs...\n")
                doc.build(story)
                
                # Build 4 AuNP PDF if there are any clusters with 4 AuNPs
                if story_4aunps:
                    doc_4aunps.build(story_4aunps)
                    self._log(f"4 AuNP mini zonogram PDF generation completed successfully!\n")
                    self._log(f"4 AuNP PDF saved to: {pdf_path_4aunps}\n")
                    
                    # Open the 4 AuNP PDF
                    try:
                        webbrowser.open(f"file://{pdf_path_4aunps.absolute()}")
                        self._log("4 AuNP PDF opened in browser.\n")
                    except Exception as e:
                        self._log(f"Could not open 4 AuNP PDF automatically: {e}\n")
                        self._log(f"Please open manually: {pdf_path_4aunps}\n")
                else:
                    self._log("No clusters with 4 AuNPs found, skipping 4 AuNP PDF.\n")
                
                self._log(f"Mini zonogram PDF generation completed successfully!\n")
                self._log(f"PDF saved to: {pdf_path}\n")
                
                # Open the main PDF
                try:
                    webbrowser.open(f"file://{pdf_path.absolute()}")
                    self._log("PDF opened in browser.\n")
                except Exception as e:
                    self._log(f"Could not open PDF automatically: {e}\n")
                    self._log(f"Please open manually: {pdf_path}\n")
                
            except Exception as e:
                self._log(f"Error generating mini zonogram PDF: {e}\n")
                import traceback
                self._log(f"Traceback: {traceback.format_exc()}\n")
        
        # Start the PDF generation in a background thread
        threading.Thread(target=generate_pdf_background).start()

if __name__ == "__main__":
    app = AnalysisPipelineGUI()
    app.mainloop() 