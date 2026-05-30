#!/usr/bin/env bash
#SBATCH --job-name=kan-prunesym
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=logs/greatlakes/%x_%A_%a.out
#SBATCH --error=logs/greatlakes/%x_%A_%a.err

set -euo pipefail

# Great Lakes downstream pyKAN pruning/symbolic audit array.
#
# Submit after the current full-KAN / semisynthetic jobs free GPUs:
#   sbatch --account=jaabell0 --export=ALL,PYTHON_BIN=$PWD/.venv/bin/python --array=0-3%2 scripts/greatlakes_prune_symbolic_array.sbatch

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs/greatlakes

PY="${PYTHON_BIN:-python}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-2}"

# label|samples|noise|update_grid|seeds
ROWS=(
  "clean_w16_n512_30seed|512|0|0|0-29"
  "clean_w16_n1024_30seed|1024|0|0|0-29"
  "gridupdate_w16_n512_30seed|512|0|1|0-29"
  "noise005_w16_n512_30seed|512|0.05|0|0-29"
)

row="${ROWS[$SLURM_ARRAY_TASK_ID]}"
IFS='|' read -r label samples noise update_grid seeds <<< "$row"

out_dir="results/revision/greatlakes_prune_symbolic/${label}"
args=(
  scripts/run_pykan_prune_symbolic_smoke.py
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
  --thresholds 0.01 0.03 0.05 0.10
  --workflows prune_input prune
  --symbolic-smoke
  --batch-size 4096
  --device cuda
  --out-dir "$out_dir"
)

if [[ "$update_grid" == "1" ]]; then
  args+=(--update-grid --grid-update-num 5)
fi

echo "[$(date -Is)] running ${label} on ${SLURM_JOB_NODELIST:-unknown-node}"
"$PY" "${args[@]}"
echo "[$(date -Is)] done ${label}"
