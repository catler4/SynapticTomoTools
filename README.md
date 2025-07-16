# SynapticTomoTools

```text
╔═╗┬ ┬┌┐┌┌─┐┌─┐┌┬┐┬┌─┐╔╦╗┌─┐┌┬┐┌─┐╔╦╗┌─┐┌─┐┬  ┌─┐
╚═╗└┬┘│││├─┤├─┘ │ ││   ║ │ │││││ │ ║ │ ││ ││  └─┐
╚═╝ ┴ ┘└┘┴ ┴┴   ┴ ┴└─┘ ╩ └─┘┴ ┴└─┘ ╩ └─┘└─┘┴─┘└─┘
```
A Python toolkit for running various analyses on cryo-electron tomography (cryo-ET) data,  
with a focus on synaptic structures.

---

## 🚀 Features

- Quantitative measurements
    - Active zone identification
    - Synaptic cleft width measurement
    - Presynaptic vesicle volume, diameter, localization, enclosed signal analyses
    - Presynaptic vesicle fusion site estimation
    - AuNP label nearest neighbor distances
    - AuNP label to active zone membrane and vesicle fusion site distances
    - AuNP label cluster analyses
- Command-line interface for easy pipeline execution
- Automated visualization generation

## ⏯️ Workflow

Modules should be run in the following order for all analyses to be completed:

1. activezone
2. vesicles
3. aunps
4. visualization (optional)

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/catler4/SynapticTomoTools.git
cd SynapticTomoTools
pip install -r requirements.txt

```

---

## 📁 Data organization

### Required File Formats

Each tomogram directory must contain the following files in the specified locations and formats:

#### `presynatpticmembranes_*.txt` and `postsynapticmembranes_*.txt`
- **Location:** `best_alignment/aunps/`
- **Format:** Plain text file. Each line contains three whitespace-separated numbers representing the X, Y, Z coordinates of a point on the membrane segmentation.
- **Example:**
  ```text
  123.4 567.8 90.1
  124.0 568.2 90.3
  ...
  ```
- **Notes:**
  - The typo `presynatptic` (instead of `presynaptic`) is expected and required for now.

#### synapticvesicles_*.txt
- **Location:** `best_alignment/aunps/`
- **Format:** Plain text file. Each line contains three whitespace-separated numbers representing the X, Y, Z coordinates of a point on the vesicle surface or center.
- **Example:**
  ```text
  200.1 300.2 50.0
  201.0 301.1 50.2
  ...
  ```
- **Notes:**
  - Multiple files may exist (e.g., `synapticvesicles_1.txt`, `synapticvesicles_2.txt`, ...), each corresponding to a different vesicle segmentation. Inner and outer vesicle membranes can be segmented individually. If this is the case, the inner membrane will be removed from analysis in a later filtering step.

#### aunp_tm_BP_active_zone_*.star
- **Location:** `best_alignment/aunps/`
- **Format:** [STAR file](https://www.ccpem.ac.uk/download/starfile.php) (Self-defining Text Archival and Retrieval format), typically used in cryo-EM. Contains a table of AuNP coordinates and metadata.
- **Key columns:**
  - `faCoordinateX`, `faCoordinateY`, `faCoordinateZ`: Coordinates of each AuNP.
  - `active_zone`: Index of the active zone the AuNP is associated with.
- **Example:**
  ```
  data_
  loop_
  _faCoordinateX _faCoordinateY _faCoordinateZ _active_zone
  100.0 200.0 50.0 0
  101.2 201.5 50.1 0
  ...
  ```
- **Notes:**
  - There may be multiple files, one per active zone (e.g., `aunp_tm_BP_active_zone_0.star`, `aunp_tm_BP_active_zone_1.star`, ...).

#### *_ddw.mrc
- **Location:** `best_alignment/`
- **Format:** [MRC file](https://en.wikipedia.org/wiki/MRC_(file_format)), a standard format for electron density maps in cryo-EM.
- **Purpose:** Used as the tomogram volume for visualization overlays and analysis. Typically the best version of a tomogram for visualization (e.g. denoised or filtered) is used. We typically use the DeepDeWedge denoised (ddw) versions of tomograms.
- **Notes:**
  - The filename must end with `_ddw.mrc` and match the tomogram's name prefix.

### Required File Organization

Tomograms are grouped by sets based on experimental conditions and marked for certain analyses in /data/tomograms.csv

The cli.py must be updated with the root directories for each of the tomogram sets, in which each tomogram's data should be stored in a separate directory.

### **Example:**
```
SET_ROOTS = {
    "15F1": Path("/goliath/processing/Gouaux/CJS/BestTomo/ProcessingCJS/tomograms/15F1_tomograms/TOP_TOMOS"),
    "5F11": Path("/goliath/processing/Gouaux/CJS/BestTomo/ProcessingCJS/tomograms/5F11_tomograms/TOP_TOMOS"),
    "15F1and5F11": Path("/goliath/processing/Gouaux/CJS/BestTomo/ProcessingCJS/tomograms/15F1and5F11_tomograms/TOP_TOMOS"),
    "15F1and5F11dimer": Path("/goliath/processing/Gouaux/CJS/BestTomo/ProcessingCJS/tomograms/15F1and5F11dimer_tomograms/TOP_TOMOS"),
    "11B8": Path("/goliath/processing/Gouaux/CJS/BestTomo/ProcessingCJS/tomograms/11B8_tomograms/TOP_TOMOS"),
    "unlabeled": Path("/goliath/processing/Gouaux/CJS/BestTomo/ProcessingCJS/tomograms/unlabeled_tomograms/TOP_TOMOS"),
    # Add more sets here if needed
}
```

Within each tomogram subdirectory, a best_alignment/aunps subdirectory is expected that contains the pre- and postsynaptic membrane segmentations (presynatpticmembranes_1.txt, postsynapticmembranes_1.txt, ...), vesicle segmentations (synapticvesicles_1.txt, synapticvesicles_2.txt, ...), and aunp picks saved in individual .star files per active zone (aunp_tm_BP_active_zone_0.star, aunp_tm_BP_active_zone_1.star, ...).

```text
TOMOGRAM_SET_ROOT/
├── TOP_TOMOS/
│   ├── tomogram1/
│   │   ├── best_alignment/
│   │   │   ├── tomogram1_ddw.mrc (tomogram that will be used for visualizations)
│   │   │   ├── aunps/
│   │   │   │   ├── presynatpticmembranes_1.txt (NOTE THE TYPO presynaTptic, IT IS EXPECTED until this is fixed in findingampa code!)
│   │   │   │   ├── postsynapticmembranes_1.txt
│   │   │   │   ├── synapticvesicles_1.txt
│   │   │   │   ├── synapticvesicles_2.txt
│   │   │   │   └── aunp_tm_BP_active_zone_0.star
│   ├── tomogram2/
│   │   ├── best_alignment/
│   │   │   ├── tomogram2_ddw.mrc
│   │   │   ├── aunps/
│   │   │   │   ├── presynatpticmembranes_1.txt
│   │   │   │   ├── presynatpticmembranes_2.txt
│   │   │   │   ├── postsynapticmembranes_1.txt
│   │   │   │   ├── synapticvesicles_1.txt
│   │   │   │   ├── synapticvesicles_2.txt
│   │   │   │   ├── aunp_tm_BP_active_zone_0.star
│   │   │   │   └── aunp_tm_BP_active_zone_1.star
...
```


The `data/tomograms.csv` file controls which tomograms are analyzed and how. Each row corresponds to a tomogram and specifies which analyses to run and (optionally) which AuNP active zones to use.

### **Required columns:**
- `tomoname`: Name of the tomogram directory (matches folder name under your set root)
- `set`: Experimental set name (must match a key in your SET_ROOTS in cli.py)
- `activezone`: `True` or `False` — whether to run active zone analysis
- `vesicles`: `True` or `False` — whether to run vesicle analysis
- `aunps`: `True` or `False` — whether to run AuNP analysis
- `aunp_active_zones` (optional): Comma-separated list of active zone numbers (e.g., `"0,2"`).
    - If empty, all numbered `aunp_tm_BP_active_zone_*.star` files are used.
    - If set, only those indices are used (e.g., `2` → only `aunp_tm_BP_active_zone_2.star`).

### **Example:**
```csv
tomoname,set,activezone,vesicles,aunps,aunp_active_zones
20231017_EGmilled24-2_68,15F1,True,True,True,"2"
20231017_HippAu_141,15F1,True,True,True,"0,2"
20231026_HippAu_26,15F1,True,True,True,
```

---

## 🖥️ Usage

### Command Line Interface

Run from the project root:

```bash
python -m src.synaptic_tomo_tools.cli --analysis all
```

### Key CLI Flags

| Flag                        | Description                                                                                      |
|-----------------------------|--------------------------------------------------------------------------------------------------|
| `--analysis`                | **Required.** Which analysis to run. Choices: `activezone`, `vesicles`, `aunps`, `all`          |
| `--set`                     | (Optional) Filter tomograms by experimental set name (e.g., 15F1, unlabeled)                    |
| `--csv`                     | Path to CSV file listing tomograms and analysis flags (default: `data/tomograms.csv`)            |
| `--rerun`                   | Rerun analysis on already completed steps and overwrite existing results                                 |
| `--results-dir`             | Directory to store analysis results (default: `results`)                                         |
| `--generate-visualizations` | Generate visualization images for each tomogram after analysis completion                        |
| `--delete-results`          | **Delete all analysis results files before running analysis**                                    |
| `--test`                    | Use local test data roots and default to `data/tomograms-test.csv` unless `--csv` is specified. All test set roots are set relative to the repo root. Supported test sets: 15F1, 5F11, 15F1and5F11, 15F1and5F11dimer, 11B8, unlabeled. |

### Example: Active Zone Analysis

```bash
python -m src.synaptic_tomo_tools.cli --analysis activezone
```

### Example: Full Pipeline on non-default tomogram test set

```bash
python -m src.synaptic_tomo_tools.cli --analysis all --test
```

### Example: Full Pipeline with All Options

```bash
python -m src.synaptic_tomo_tools.cli --analysis all --csv data/tomograms.csv --rerun --generate-visualizations --delete-results
```

### Test Mode

The `--test` flag switches the pipeline to use local test data directories and defaults to `data/tomograms-test.csv` for the tomogram list (unless you specify `--csv`).

- All test set roots are set as relative paths from the repository root, so the code works on any machine where the repo is cloned.
- Supported test sets: `15F1`, `5F11`, `15F1and5F11`, `15F1and5F11dimer`, `11B8`, `unlabeled`.
- Example usage:

```bash
python -m src.synaptic_tomo_tools.cli --analysis all --test
```

You can override the CSV file with `--csv` if you want to use a different test list.

### See All Options

For the latest options and descriptions, run:
```bash
python -m src.synaptic_tomo_tools.cli --help
```

---

### Output Files

Analysis and visualization results are saved in the following locations:

- **Active zone results:**
  - `results/activezone_results.csv` — summary statistics for all tomograms
  - Per-tomogram results in each tomogram's `best_alignment/STT_results/active_zones/`
- **Vesicle results:**
  - `results/vesicle_results.csv` — summary statistics for all tomograms
  - Per-tomogram results in each tomogram's `best_alignment/STT_results/vesicles/` (e.g., `vesicle_results.json`)
- **AuNP results:**
  - `results/aunp_results.csv` — summary statistics for all tomograms
  - `results/all_aunp_distances.csv` — all per-AuNP distances for all tomograms
  - Per-tomogram results in each tomogram's `best_alignment/aunps/` (e.g., `aunp_nearest_neighbor_distances.csv`)
- **Combined results:**
  - `results/analysis_results.json` — all results for all tomograms in a single JSON
- **Visualizations:**
  - **Per-tomogram images:**
    - `best_alignment/STT_results/visualizations/` inside each tomogram directory:
      - `{tomo_name}_vesicles_active_zones.png`: Vesicles and active zones
      - `{tomo_name}_aunps.png`: Vesicles and AuNPs (filtered by aunp_active_zones)
      - `{tomo_name}_combined.png`: All elements together
      - `{tomo_name}_vesicles_signal.png`: Vesicles colored by average signal intensity (gradient fill)
  - **Combined images:**
    - `results/visualizations/` — copies of all per-tomogram images for easy access

**Notes:**
- All visualizations use the same AuNP filtering as the analysis step (based on `aunp_active_zones`).
- Only vesicles intersecting the central slice (±1 pixel) are shown by default in visualizations.

---
