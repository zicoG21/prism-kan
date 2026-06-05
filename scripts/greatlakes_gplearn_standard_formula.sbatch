#!/bin/bash
#SBATCH --job-name=kan-gplearn-sr
#SBATCH --partition=standard
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/greatlakes/kan-gplearn-sr_%A_%a.out
#SBATCH --error=logs/greatlakes/kan-gplearn-sr_%A_%a.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs/greatlakes

PYTHON_BIN="${PYTHON_BIN:-$PWD/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python)"
fi

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
SEEDS_PER_TASK="${SEEDS_PER_TASK:-3}"
SEED_START=$(( TASK_ID * SEEDS_PER_TASK ))
SEED_STOP=$(( SEED_START + SEEDS_PER_TASK - 1 ))
POPULATION_SIZE="${POPULATION_SIZE:-600}"
GENERATIONS="${GENERATIONS:-20}"
LABEL_SUFFIX="${LABEL_SUFFIX:-s${SEED_START}_${SEED_STOP}}"
OUT_DIR="results/revision/gplearn_standard_formula_baseline_${LABEL_SUFFIX}"

echo "[$(date -Is)] host=$(hostname) task=${TASK_ID} seeds=${SEED_START}-${SEED_STOP} pop=${POPULATION_SIZE} gen=${GENERATIONS}"
"$PYTHON_BIN" scripts/run_gplearn_standard_formula_baseline.py \
  --seeds "${SEED_START}-${SEED_STOP}" \
  --population-size "${POPULATION_SIZE}" \
  --generations "${GENERATIONS}" \
  --n-jobs "${SLURM_CPUS_PER_TASK:-1}" \
  --out-dir "$OUT_DIR"
echo "[$(date -Is)] done gplearn standard formula baseline -> $OUT_DIR"
