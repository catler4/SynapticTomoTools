import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk
import subprocess
import threading
import os
import webbrowser
import pandas as pd
from pathlib import Path

FIG_HOME = "figures/synaptictomotools_fig_gui_home-01.png"
FIG_AZ = "figures/synaptictomotools_fig_gui_AZ-01.png"
FIG_VESICLES = "figures/synaptictomotools_fig_gui_vesicles-01.png"
FIG_AUNPS = "figures/synaptictomotools_fig_gui_aunps-01.png"

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
        self.geometry("800x700")
        self.csv_path = tk.StringVar()
        self.root_dir = tk.StringVar()
        self.start_tomogram = tk.StringVar()
        self.log_text = None
        self._img_refs = []  # Keep references to PhotoImage objects
        self._current_process = None  # Track running process for stopping
        self._build_tabs()

    def _build_home_tab_content(self, tab):
        frame = ttk.Frame(tab)
        frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
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
        
        # Starting tomogram selection
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

    def _load_and_display_image(self, path, parent, max_width=400, max_height=180):
        try:
            img = Image.open(path)
            iw, ih = img.size
            scale = min(max_width / iw, max_height / ih, 1.0)
            img = img.resize((int(iw * scale), int(ih * scale)), Image.LANCZOS)
            tk_img = ImageTk.PhotoImage(img)
            return tk_img
        except Exception as e:
            print(f"Could not load image {path}: {e}")
            return None

    def _build_tabs(self):
        for widget in self.winfo_children():
            widget.destroy()
        
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
            ("Annotate Membranes (Blender plug-in)", "new-annotate-aunps"),
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
        self._run_findingampa_command("new-annotate-aunps")
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
        # For DDW, require a model selection and pass as positional argument
        extra_args = []
        if command == "ddw":
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
                    cli = ["finding_ampa", command] + extra_args
                    self._log(f"Running (single tomogram): {' '.join(cli)} in {self.findingampa_single_dir.get()}\n")
                    env = os.environ.copy()
                    self._run_subprocess(cli, env, self.findingampa_single_dir.get())
                elif self.findingampa_all_mode.get() or not self.findingampa_single_mode.get():
                    # Run for all tomograms in CSV, using best_alignment dir for each
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
                            if not set_name or not tomo_name:
                                continue
                            best_align_dir = _os.path.join(root_dir, set_name, "TOP_TOMOS", tomo_name, "best_alignment")
                            if not _os.path.isdir(best_align_dir):
                                self._log(f"Skipping missing directory: {best_align_dir}\n")
                                continue
                            cli = ["finding_ampa", command] + extra_args
                            self._log(f"Running: {' '.join(cli)} in {best_align_dir}\n")
                            env = os.environ.copy()
                            self._run_subprocess(cli, env, best_align_dir)
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
        for (label, command), var in zip(self.findingampa_commands, self.findingampa_check_vars):
            if var.get():
                self._log(f"\nStarting command: {label}\n")
                self._run_findingampa_command(command, wait_for_completion=True)
                self._log(f"Completed command: {label}\n")

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
        ttk.Label(controls_frame, text=f"Run {step} analysis").pack(anchor=tk.W, pady=10)
        if step == "Full Pipeline":
            ttk.Label(controls_frame, text="Active Zone -> Vesicles -> AuNPs -> Visualization").pack(anchor=tk.W, pady=(0, 10))
        run_btn = ttk.Button(controls_frame, text=f"Run {step}", command=lambda s=step: self._run_analysis(s, tab))
        run_btn.pack(anchor=tk.W, pady=5)
        # Add checkboxes for rerun, delete-results, check-files
        rerun_var = tk.BooleanVar()
        delres_var = tk.BooleanVar()
        checkfiles_var = tk.BooleanVar()
        rerun_cb = ttk.Checkbutton(controls_frame, text="Rerun (overwrite existing results)", variable=rerun_var)
        rerun_cb.pack(anchor=tk.W)
        ToolTip(rerun_cb, "Rerun analysis on already completed steps and overwrite existing results.")
        delres_cb = ttk.Checkbutton(controls_frame, text="Delete all results before running", variable=delres_var)
        delres_cb.pack(anchor=tk.W)
        ToolTip(delres_cb, "Delete all analysis results files before running analysis.")
        checkfiles_cb = ttk.Checkbutton(controls_frame, text="Check files only (no analysis)", variable=checkfiles_var)
        checkfiles_cb.pack(anchor=tk.W)
        ToolTip(checkfiles_cb, "Check that all expected files for the tomograms listed in the CSV are present in the expected locations. No analysis is run.")
        # Store flag variables in the tab for access in _run_analysis
        tab._flag_vars = (rerun_var, delres_var, checkfiles_var)
        
        # Add calculate signals checkbox for vesicles
        if step == "Vesicles":
            calculate_signals_var = tk.BooleanVar()
            calculate_signals_cb = ttk.Checkbutton(controls_frame, text="Calculate vesicle signals (slower)", variable=calculate_signals_var)
            calculate_signals_cb.pack(anchor=tk.W)
            ToolTip(calculate_signals_cb, "Calculate vesicle signal intensity (slower but provides signal data).")
            tab._calculate_signals_var = calculate_signals_var
        
        # Add calculate signals checkbox for full pipeline
        if step == "Full Pipeline":
            calculate_signals_var = tk.BooleanVar()
            calculate_signals_cb = ttk.Checkbutton(controls_frame, text="Calculate vesicle signals (slower)", variable=calculate_signals_var)
            calculate_signals_cb.pack(anchor=tk.W)
            ToolTip(calculate_signals_cb, "Calculate vesicle signal intensity (slower but provides signal data).")
            tab._calculate_signals_var = calculate_signals_var
        
        pdf_frame = None
        if step in ["Visualization", "Full Pipeline"]:
            pdf_frame = ttk.Frame(controls_frame)
            pdf_frame.pack(anchor=tk.W, pady=10)
            pdf_btn = ttk.Button(pdf_frame, text="Generate PDF Summary", command=self._run_pdf_summary)
            pdf_btn.pack(anchor=tk.W, pady=(0, 4))
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
            # Show the starting tomogram selection
            self.start_tomogram_label.grid(row=4, column=0, sticky=tk.W)
            self.start_tomogram_combo.grid(row=4, column=1, sticky=tk.W)
        else:
            # Hide the starting tomogram selection
            self.start_tomogram_label.grid_remove()
            self.start_tomogram_combo.grid_remove()

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
            if hasattr(tab, '_calculate_signals_var') and tab._calculate_signals_var.get():
                cli += ["--calculate-vesicle-signals"]
        elif step == "AuNPs":
            cli += ["--analysis", "aunps"]
        elif step == "Visualization":
            cli += ["--analysis", "visualizations"]
        elif step == "Full Pipeline":
            cli += ["--analysis", "all"]
            if generate_pdf:
                cli += ["--generate-pdf-summary"]
            if hasattr(tab, '_calculate_signals_var') and tab._calculate_signals_var.get():
                cli += ["--calculate-vesicle-signals"]
        # Add flags from checkboxes
        rerun_var, delres_var, checkfiles_var = getattr(tab, '_flag_vars', (None, None, None))
        if rerun_var and rerun_var.get():
            cli += ["--rerun"]
        if delres_var and delres_var.get():
            cli += ["--delete-results"]
        if checkfiles_var and checkfiles_var.get():
            cli += ["--check-files"]
        self._log(f"Running: {' '.join(cli)}\n")
        # Pass the single root dir as env var TOMO_ROOT_BASE if set
        env = os.environ.copy()
        if self.root_dir.get():
            env["TOMO_ROOT_BASE"] = self.root_dir.get()
        threading.Thread(target=self._run_subprocess, args=(cli, env)).start()

    def _run_subprocess(self, cli, env, cwd=None):
        self._current_process = subprocess.Popen(cli, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env, cwd=cwd)
        try:
            for line in self._current_process.stdout:
                self._log(line)
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
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, msg)
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.NORMAL)

    def _run_pdf_summary(self):
        cli = ["python", "-u", "scripts/generate_tomogram_summary_pdf.py"]
        if self.root_dir.get():
            cli += ["--data-dir", self.root_dir.get()]
        if self.csv_path.get():
            # Always use the original CSV for PDF generation to include all tomograms
            cli += ["--tomocsv", self.csv_path.get()]
            
            # Add starting tomogram if specified
            processing_mode = self.processing_mode.get()
            selected_tomogram = self.start_tomogram.get()
            
            if processing_mode in ["Single tomogram", "Start from"] and selected_tomogram:
                cli += ["--start-from", selected_tomogram]
                self._log(f"PDF generation will start from tomogram: {selected_tomogram}\n")
                self._log("Note: Final PDF will still include all tomograms from the original CSV file\n")
            else:
                self._log("Note: PDF generation will include all tomograms from the original CSV file\n")
                
        self._log(f"Running: {' '.join(cli)}\n")
        threading.Thread(target=self._run_subprocess, args=(cli, os.environ.copy())).start()

    def _view_pdf_summary(self):
        pdf_path = os.path.abspath("results/summary_pdfs/all_tomograms_summary.pdf")
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
        
        # Vesicle Slice Extraction section
        vesicle_frame = ttk.LabelFrame(frame, text="Vesicle Slice Extraction", padding=10)
        vesicle_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Description
        desc_label = ttk.Label(vesicle_frame, text="Extract 120x120 pixel slices from vesicles within 10nm of active zone membrane.\nGenerates three slice types: regular slice, thick slice (20nm), and MinIP (20nm).\nSlices are oriented with the closest active zone point pointing down.\nCreates both individual PNG files and a comprehensive PDF summary.")
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

if __name__ == "__main__":
    app = AnalysisPipelineGUI()
    app.mainloop() 