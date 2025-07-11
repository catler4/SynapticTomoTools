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

## ⏯️ Workflow

Modules should be run in the following order for all analyses to be completed
    1. activezone
    2. vesicles
    3. aunps

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
