#!/usr/bin/env bash
#SBATCH --job-name=kan-fullanova
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/greatlakes/%x_%A_%a.out
#SBATCH --error=logs/greatlakes/%x_%A_%a.err

set -euo pipefail

# Great Lakes full-KAN ANOVA array job.
#
# Submit via:
#   sbatch --account jaabell0 --array 0-5 scripts/greatlakes_fullkan_anova_array.sbatch
#
# Required on the cluster before submitting:
#   git clone / update this repository
#   create a Python environment with torch, pykan, numpy, pandas, sklearn
#   export PYTHON_BIN=/path/to/python   # optional; defaults to python

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs/greatlakes

PY="${PYTHON_BIN:-python}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-2}"

# label|samples|noise|update_grid|seeds
ROWS=(
  "clean_w16_n512_60seed|512|0|0|0-59"
  "clean_w16_n640_60seed|640|0|0|0-59"
  "clean_w16_n768_60seed|768|0|0|0-59"
  "clean_w16_n896_60seed|896|0|0|0-59"
  "gridupdate_w16_n512_60seed|512|0|1|0-59"
  "gridupdate_w16_n640_60seed|640|0|1|0-59"
  "gridupdate_w16_n768_60seed|768|0|1|0-59"
  "gridupdate_w16_n896_60seed|896|0|1|0-59"
  "noise005_w16_n512_60seed|512|0.05|0|0-59"
  "noise005_w16_n768_60seed|768|0.05|0|0-59"
  "noise005_w16_n1024_60seed|1024|0.05|0|0-59"
  "noise010_w16_n896_60seed|896|0.10|0|0-59"
)

row="${ROWS[$SLURM_ARRAY_TASK_ID]}"
IFS='|' read -r label samples noise update_grid seeds <<< "$row"

out_dir="results/revision/fullkan_anova_boundary/${label}"
args=(
  scripts/run_full_kan_pair_anova_probe.py
  --function core_interaction_c025
  --samples "$samples"
  --test-samples 2048
  --dimension 100
  --noise "$noise"
  --seeds "$seeds"
  --width-hidden 16
  --grid 5
  --k 3
  --steps 75
  --lamb 0.001
  --pair-mode all
  --max-all-pairs 10000
  --anova-points 16
  --anova-background 16
  --batch-size 8192
  --pair-chunk-size 1000
  --device cuda
  --out-dir "$out_dir"
)

if [[ "$update_grid" == "1" ]]; then
  args+=(--update-grid --grid-update-num 5)
fi

echo "[$(date -Is)] running ${label} on ${SLURM_JOB_NODELIST:-unknown-node}"
"$PY" "${args[@]}"
echo "[$(date -Is)] done ${label}"

