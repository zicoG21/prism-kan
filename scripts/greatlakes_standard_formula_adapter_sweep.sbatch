#!/usr/bin/env bash
#SBATCH --job-name=kan-std-sweep
#SBATCH --partition=standard
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/greatlakes/%x_%A_%a.out
#SBATCH --error=logs/greatlakes/%x_%A_%a.err

set -euo pipefail

# Dependency-light standard-formula adapter sweep over the generated task-card
# suite.  Use STANDARD_AUGMENT=full for the 90-card v1 expansion.
#
# Example:
#   sbatch --account=jaabell0 --array=0-29 \
#     --export=ALL,PYTHON_BIN=$PWD/.venv/bin/python,SEED_BASE=0,SEEDS_PER_TASK=3,STANDARD_AUGMENT=full \
#     scripts/greatlakes_standard_formula_adapter_sweep.sbatch

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs/greatlakes

PYTHON_BIN="${PYTHON_BIN:-$PWD/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python)"
fi

STANDARD_AUGMENT="${STANDARD_AUGMENT:-none}"
"$PYTHON_BIN" scripts/generate_standard_formula_task_cards.py --augment "${STANDARD_AUGMENT}"

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
SEED_BASE="${SEED_BASE:-0}"
SEEDS_PER_TASK="${SEEDS_PER_TASK:-3}"
SEED_START=$(( SEED_BASE + TASK_ID * SEEDS_PER_TASK ))
SEED_STOP=$(( SEED_START + SEEDS_PER_TASK - 1 ))
LABEL_SUFFIX="${LABEL_SUFFIX:-s${SEED_START}_${SEED_STOP}_${STANDARD_AUGMENT}}"
OUT_DIR="results/revision/standard_formula_adapter_sweep_${LABEL_SUFFIX}"

echo "[$(date -Is)] host=$(hostname) task=${TASK_ID} seeds=${SEED_START}-${SEED_STOP} augment=${STANDARD_AUGMENT}"
"$PYTHON_BIN" scripts/run_standard_formula_adapter_sweep.py \
  --seeds "${SEED_START}-${SEED_STOP}" \
  --out-dir "${OUT_DIR}"
echo "[$(date -Is)] done standard formula adapter sweep -> ${OUT_DIR}"
