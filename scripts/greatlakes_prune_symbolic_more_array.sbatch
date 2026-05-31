#!/usr/bin/env bash
#SBATCH --job-name=kan-prunesym2
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/greatlakes/%x_%A_%a.out
#SBATCH --error=logs/greatlakes/%x_%A_%a.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs/greatlakes

PY="${PYTHON_BIN:-python}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-2}"

# label|function|samples|dimension|noise|update_grid|width|steps|seeds
ROWS=(
  "gridupdate_w16_n1024_30seed|core_interaction_c025|1024|100|0|1|16|75|0-29"
  "noise010_w16_n512_30seed|core_interaction_c025|512|100|0.10|0|16|75|0-29"
  "noise010_w16_n1024_30seed|core_interaction_c025|1024|100|0.10|0|16|75|0-29"
  "clean_w32_n768_30seed|core_interaction_c025|768|100|0|0|32|75|0-29"
)

row="${ROWS[$SLURM_ARRAY_TASK_ID]}"
IFS='|' read -r label function samples dimension noise update_grid width steps seeds <<< "$row"

out_dir="results/revision/greatlakes_prune_symbolic_more/${label}"
args=(
  scripts/run_pykan_prune_symbolic_smoke.py
  --function "$function"
  --samples "$samples"
  --test-samples 2048
  --dimension "$dimension"
  --noise "$noise"
  --seeds "$seeds"
  --width-hidden "$width"
  --grid 5
  --k 3
  --steps "$steps"
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

echo "[$(date -Is)] running prune/symbolic more ${label} on ${SLURM_JOB_NODELIST:-unknown-node}"
"$PY" "${args[@]}"
echo "[$(date -Is)] done prune/symbolic more ${label}"

