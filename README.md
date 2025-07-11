# SynapticTomoTools

A Python toolkit for running various analyses on cryo-electron tomography (cryo-ET) data,  
with a focus on synaptic structures. Modular, command-line friendly, and designed for reproducible research workflows.

---

## 🚀 Features

- Quantitative measurements
    > Pre- and postsynaptic compartment volumes
    > Active zone area
    > Synaptic cleft width
    > Presynaptic vesicle volume, diameter, and localization
    > Presynaptic vesicle loading estimation
    > AuNP label nearest neighbor distances
    > AuNP label to vesicle fusion site distances
- Command-line interface for easy pipeline execution
- Automated visualization generation

## ⏯️ Workflow

Modules should be run in the following order for all analyses to be completed
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

## 🖥️ Usage

### Command Line Interface

The toolkit provides a command-line interface for running analyses. Navigate to the project root directory and use the following commands:

#### Running Individual Analyses

```bash
# Run active zone analysis
python -m src.synaptic_tomo_tools.cli activezone

# Run vesicle analysis
python -m src.synaptic_tomo_tools.cli vesicles

# Run AuNP analysis
python -m src.synaptic_tomo_tools.cli aunps

# Run visualization generation
python -m src.synaptic_tomo_tools.cli visualization
```

#### Running All Analyses Sequentially

```bash
# Run all analyses in the correct order
python -m src.synaptic_tomo_tools.cli all
```

#### Generating Visualizations

Visualizations can be generated in two ways:

1. **As part of the analysis pipeline** (recommended):
```bash
# Run all analyses including visualizations
python -m src.synaptic_tomo_tools.cli all --generate-visualizations
```

2. **Standalone visualization** (requires previous analysis results):
```bash
# Generate visualizations for existing analysis results
python -m src.synaptic_tomo_tools.cli visualization
```

#### Command Line Options

- `--tomogram-set`: Specify a particular tomogram set (default: all sets)
- `--generate-visualizations`: Include visualization generation in the pipeline
- `--help`: Show help information for any command

Example with specific tomogram set:
```bash
python -m src.synaptic_tomo_tools.cli all --tomogram-set 15F1 --generate-visualizations
```

### Output Files

Analysis results are saved in the following locations:

- **Active zone results**: `results/activezone_results.csv`
- **Vesicle results**: `results/vesicle_results.csv`
- **AuNP results**: `results/aunp_results.csv`
- **Combined results**: `results/analysis_results.json`
- **Visualizations**: 
  - Individual tomogram visualizations: `data/{tomogram_set}/TOP_TOMOS/{tomogram}/best_alignment/STT_results/visualizations/`
  - Combined visualizations: `results/visualizations/`

### Visualization Outputs

For each tomogram, three types of visualization images are generated:

1. **Active zone visualization** (`{tomo_name}_activezone.png`): Shows vesicles and active zones with fusion sites
2. **AuNP visualization** (`{tomo_name}_aunps.png`): Shows vesicles and AuNP labels
3. **Combined visualization** (`{tomo_name}_combined.png`): Shows all elements together

---

## 📁 Data organization

Tomograms are grouped by sets based on experimental conditions and marked for certain analyses in /data/tomograms.csv

The cli.py must be updated with the root directories for the tomogram sets, in which each tomogram's data should be stored in a separate directory.

Within each tomogram subdirectory, a best_alignment/aunps subdirectory is expected that contains the pre- and postsynaptic membrane segmentations (presynatpticmembranes_1.txt, postsynapticmembranes_1.txt, ...), vesicle segmentations (synapticvesicles_1.txt, synapticvesicles_2.txt, ...), and aunp picks (aunp_tm_BP_active_zone_all.star). 

TOMOGRAM_SET_ROOT/
├── TOP_TOMOS/
│   ├── tomogram1/
│   │   ├── best_alignment/
│   │   │   ├── aunps/
│   │   │   │   ├── presynatpticmembranes_1.txt
│   │   │   │   ├── postsynapticmembranes_1.txt
│   │   │   │   ├── synapticvesicles_1.txt
│   │   │   │   ├── synapticvesicles_2.txt
│   │   │   │   └── aunp_tm_BP_active_zone_all.star
│   ├── tomogram2/
│   │   ├── best_alignment/
│   │   │   ├── aunps/
│   │   │   │   ├── presynatpticmembranes_1.txt
│   │   │   │   ├── postsynapticmembranes_1.txt
│   │   │   │   ├── synapticvesicles_1.txt
│   │   │   │   ├── synapticvesicles_2.txt
│   │   │   │   └── aunp_tm_BP_active_zone_all.star
...

---
