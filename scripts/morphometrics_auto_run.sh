#!/usr/bin/env bash
# Resume-friendly morphometrics + STT comparison pipeline.
#
# Edit DATASETS below (directories under DATA_DIR / morphometrics layout), then:
#   bash scripts/morphometrics_auto_run.sh
#
# Recompute everything:
#   FORCE=1 bash scripts/morphometrics_auto_run.sh
#
# Each DATASETS entry is:
#   <set_dir>|<csv_path>|<morpho_subdir>
# where:
#   set_dir       = directory name under DATA_DIR (documentation / progress label;
#                   tomogram paths still come from the CSV set/tomoname columns)
#   csv_path      = STT tomograms.csv relative to this repo (or absolute)
#   morpho_subdir = folder under MORPHO_ROOT for symlinks + make_meshes + results/

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# ---------------------------------------------------------------------------
# User-editable paths / dataset list
# ---------------------------------------------------------------------------
CONDA_SH="${CONDA_SH:-/home/users/andecath/miniconda3/etc/profile.d/conda.sh}"
DATA_DIR="${DATA_DIR:-/goliath/processing/Gouaux/CJS/BestTomo/ProcessingCJS/tomograms}"
MORPHO_ROOT="${MORPHO_ROOT:-/goliath/processing/Gouaux/CJS/BestTomo/ProcessingCJS/analyses/morphometrics}"
COMPARE_ROOT="${COMPARE_ROOT:-results}"

# Add / remove datasets here (one per line).
DATASETS=(
  "15F1|tomogram_csv_files/tomograms_15F1_FINAL.csv|15F1"
  "15F1-H4K2Cys|tomogram_csv_files/tomograms_15F1-H4K2Cys_FINAL.csv|H4K-2Cys"
)

FORCE="${FORCE:-0}"
# ---------------------------------------------------------------------------

RERUN_ARGS=()
FORCE_ARGS=()
if [[ "${FORCE}" == "1" ]]; then
  RERUN_ARGS=(--rerun)
  FORCE_ARGS=(--force)
fi

# conda.sh / conda activate reference unset vars (e.g. CONDA_BUILD); disable -u around them.
set +u
source "${CONDA_SH}"
set -u

conda_activate() {
  set +u
  conda activate "$1"
  set -u
}

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

N_DATASETS="${#DATASETS[@]}"
N_STEPS=5

progress_bar() {
  # progress_bar <current> <total> <label>
  local current="$1"
  local total="$2"
  local label="${3:-}"
  local width=28
  if (( total <= 0 )); then
    total=1
  fi
  local filled=$(( current * width / total ))
  if (( filled > width )); then
    filled="${width}"
  fi
  local empty=$(( width - filled ))
  local bar
  bar="$(printf "%${filled}s" | tr ' ' '#')$(printf "%${empty}s" | tr ' ' '-')"
  printf "\r[%s] %d/%d %s" "${bar}" "${current}" "${total}" "${label}"
  if (( current >= total )); then
    printf "\n"
  fi
}

step_banner() {
  local step="$1"
  local title="$2"
  echo ""
  progress_bar "${step}" "${N_STEPS}" "pipeline step"
  echo "=== Step ${step}/${N_STEPS}: ${title} ==="
}

dataset_progress() {
  local idx="$1"
  local set_dir="$2"
  local action="$3"
  progress_bar "${idx}" "${N_DATASETS}" "${action}: ${set_dir}"
  echo "  -> ${action}: ${set_dir} (${idx}/${N_DATASETS})"
}

parse_dataset() {
  # Sets globals: SET_DIR CSV_PATH MORPHO_SUBDIR MORPHO_DIR COMPARE_DIR
  local entry="$1"
  IFS='|' read -r SET_DIR CSV_PATH MORPHO_SUBDIR <<<"${entry}"
  if [[ -z "${SET_DIR}" || -z "${CSV_PATH}" || -z "${MORPHO_SUBDIR}" ]]; then
    echo "Invalid DATASETS entry (need set|csv|morpho_subdir): ${entry}" >&2
    exit 1
  fi
  if [[ "${CSV_PATH}" != /* ]]; then
    CSV_PATH="${REPO_ROOT}/${CSV_PATH}"
  fi
  MORPHO_DIR="${MORPHO_ROOT}/${MORPHO_SUBDIR}"
  COMPARE_DIR="${COMPARE_ROOT}/aunp_membrane_distance_stt_vs_morphometrics_${MORPHO_SUBDIR}"
}

all_cleft_plys_present() {
  local csv="$1"
  local results_dir="$2"
  python - "${csv}" "${results_dir}" <<'PY'
import sys
from pathlib import Path
import pandas as pd

csv_path, results_dir = Path(sys.argv[1]), Path(sys.argv[2])
if not csv_path.is_file():
    raise SystemExit(f"CSV not found: {csv_path}")
df = pd.read_csv(csv_path)
missing = []
for tomoname in df["tomoname"].astype(str):
    tomoname = tomoname.strip()
    for suffix in ("_cleft_pre.ply", "_cleft_post.ply"):
        path = results_dir / f"{tomoname}{suffix}"
        if not path.is_file() or path.stat().st_size == 0:
            missing.append(str(path))
if missing:
    print(f"Missing {len(missing)} morphometrics PLY(s); will run make_meshes")
    for p in missing[:8]:
        print(f"  {p}")
    if len(missing) > 8:
        print(f"  ... and {len(missing) - 8} more")
    raise SystemExit(1)
print(f"All cleft PLYs present under {results_dir} ({len(df)} tomograms)")
raise SystemExit(0)
PY
}

run_make_meshes_if_needed() {
  local work_dir="$1"
  local csv="$2"
  local results_dir="${work_dir}/results"

  if [[ "${FORCE}" != "1" ]] && all_cleft_plys_present "${csv}" "${results_dir}"; then
    echo "SKIP morphometrics make_meshes in ${work_dir} (PLY outputs already present)"
    return 0
  fi

  echo "Running morphometrics make_meshes in ${work_dir}"
  conda_activate morphometrics
  (
    cd "${work_dir}"
    morphometrics make_meshes config.yml
  )
  conda_activate synaptictomo
}

echo "========================================"
echo "Repo:       ${REPO_ROOT}"
echo "Data dir:   ${DATA_DIR}"
echo "Morpho:     ${MORPHO_ROOT}"
echo "Datasets:   ${N_DATASETS}"
for entry in "${DATASETS[@]}"; do
  parse_dataset "${entry}"
  echo "  - ${SET_DIR}  csv=$(basename "${CSV_PATH}")  morpho=${MORPHO_SUBDIR}"
done
if [[ "${FORCE}" == "1" ]]; then
  echo "Mode:       FORCE (recompute)"
else
  echo "Mode:       resume (skip completed)"
fi
echo "========================================"

conda_activate synaptictomo
echo "Active env: ${CONDA_DEFAULT_ENV}"

# --- 1) Relabel ---
step_banner 1 "Relabel MemBrain segmentations"
for i in "${!DATASETS[@]}"; do
  parse_dataset "${DATASETS[$i]}"
  dataset_progress "$((i + 1))" "${SET_DIR}" "relabel"
  python scripts/relabel_membrain_segmentation.py \
    --csv "${CSV_PATH}" \
    --data-dir "${DATA_DIR}" \
    "${RERUN_ARGS[@]+"${RERUN_ARGS[@]}"}"
done
echo "Done: relabel"

# --- 2) Symlinks ---
step_banner 2 "Symlink tomograms + labeled segmentations"
for i in "${!DATASETS[@]}"; do
  parse_dataset "${DATASETS[$i]}"
  dataset_progress "$((i + 1))" "${SET_DIR}" "symlink"
  mkdir -p "${MORPHO_DIR}"
  python scripts/symlink_tomograms_and_segmentations.py \
    --csv "${CSV_PATH}" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${MORPHO_DIR}" \
    "${FORCE_ARGS[@]+"${FORCE_ARGS[@]}"}"
done
echo "Done: symlink"

# --- 3) make_meshes ---
step_banner 3 "Surface Morphometrics make_meshes"
for i in "${!DATASETS[@]}"; do
  parse_dataset "${DATASETS[$i]}"
  dataset_progress "$((i + 1))" "${SET_DIR}" "make_meshes"
  run_make_meshes_if_needed "${MORPHO_DIR}" "${CSV_PATH}"
done
echo "Done: make_meshes"

# --- 4) Copy PLYs ---
step_banner 4 "Copy PLYs into each tomogram surface_morphometrics/"
conda_activate synaptictomo
cd "${REPO_ROOT}"
for i in "${!DATASETS[@]}"; do
  parse_dataset "${DATASETS[$i]}"
  dataset_progress "$((i + 1))" "${SET_DIR}" "copy_plys"
  python scripts/copy_surface_morphometrics_ply_to_tomograms.py \
    --csv "${CSV_PATH}" \
    --data-dir "${DATA_DIR}" \
    --source-dir "${MORPHO_DIR}/results" \
    "${FORCE_ARGS[@]+"${FORCE_ARGS[@]}"}"
done
echo "Done: copy PLYs"

# --- 5) Compare ---
step_banner 5 "Compare STT vs SM AuNP–membrane distances"
for i in "${!DATASETS[@]}"; do
  parse_dataset "${DATASETS[$i]}"
  dataset_progress "$((i + 1))" "${SET_DIR}" "compare"
  python scripts/compare_aunp_membrane_distance_stt_vs_morphometrics.py \
    --csv "${CSV_PATH}" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${COMPARE_DIR}" \
    "${RERUN_ARGS[@]+"${RERUN_ARGS[@]}"}"
done
echo "Done: compare"

echo ""
progress_bar "${N_STEPS}" "${N_STEPS}" "pipeline complete"
echo "Finished morphometrics_auto_run.sh"
