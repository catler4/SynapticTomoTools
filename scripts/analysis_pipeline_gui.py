import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk
import subprocess
import threading
import os
import webbrowser

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
        self.log_text = None
        self._img_refs = []  # Keep references to PhotoImage objects
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
        notebook = ttk.Notebook(self)
        notebook.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        # Home tab
        home_tab = ttk.Frame(notebook)
        notebook.add(home_tab, text="Home")
        self.tabs = {"Home": home_tab}
        self._build_home_tab_content(home_tab)
        # Tabs for each analysis step
        for step in ["Active Zone", "Vesicles", "AuNPs", "Visualization", "Full Pipeline"]:
            tab = ttk.Frame(notebook)
            notebook.add(tab, text=step)
            self.tabs[step] = tab
            self._build_tab_content(tab, step)
        # Log output
        self.log_frame = ttk.Frame(self)
        ttk.Label(self.log_frame, text="Log Output:").pack(anchor=tk.W)
        self.log_text = scrolledtext.ScrolledText(self.log_frame, height=12, state=tk.NORMAL, font=("Courier", 10))
        self.log_text.pack(fill=tk.BOTH, expand=True)
        # Show/hide log output based on tab
        def on_tab_change(event):
            tab_text = notebook.tab(notebook.select(), "text")
            if tab_text == "Home":
                self.log_frame.pack_forget()
            else:
                self.log_frame.pack(fill=tk.BOTH, expand=True, pady=(5, 0))
        notebook.bind("<<NotebookTabChanged>>", on_tab_change)
        # Initially hide log if Home is selected
        if notebook.tab(notebook.select(), "text") != "Home":
            self.log_frame.pack(fill=tk.BOTH, expand=True, pady=(5, 0))

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

    def _browse_csv(self):
        path = filedialog.askopenfilename(title="Select tomogram CSV", filetypes=[("CSV files", "*.csv")])
        if path:
            self.csv_path.set(path)

    def _browse_root(self):
        path = filedialog.askdirectory(title="Select root directory for tomogram sets", initialdir=".")
        if path:
            self.root_dir.set(path)

    def _run_analysis(self, step, tab, generate_pdf=False):
        # Build CLI command
        cli = ["python", "-u", "-m", "src.synaptic_tomo_tools.cli"]
        # CSV
        if self.csv_path.get():
            cli += ["--csv", self.csv_path.get()]
        # Analysis step
        if step == "Active Zone":
            cli += ["--analysis", "activezone"]
        elif step == "Vesicles":
            cli += ["--analysis", "vesicles"]
        elif step == "AuNPs":
            cli += ["--analysis", "aunps"]
        elif step == "Visualization":
            cli += ["--analysis", "all", "--generate-visualizations"]
        elif step == "Full Pipeline":
            cli += ["--analysis", "all"]
            if generate_pdf:
                cli += ["--generate-pdf-summary"]
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

    def _run_subprocess(self, cli, env):
        process = subprocess.Popen(cli, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
        for line in process.stdout:
            self._log(line)
        process.wait()
        self._log(f"\n[Process exited with code {process.returncode}]\n\n")

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
            cli += ["--tomocsv", self.csv_path.get()]
        self._log(f"Running: {' '.join(cli)}\n")
        threading.Thread(target=self._run_subprocess, args=(cli, os.environ.copy())).start()

    def _view_pdf_summary(self):
        pdf_path = os.path.abspath("results/summary_pdfs/all_tomograms_summary.pdf")
        if not os.path.exists(pdf_path):
            messagebox.showerror("PDF Not Found", f"{pdf_path} does not exist. Please generate the PDF summary first.")
            return
        webbrowser.open(f"file://{pdf_path}")

if __name__ == "__main__":
    app = AnalysisPipelineGUI()
    app.mainloop() 