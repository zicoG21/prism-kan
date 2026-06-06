#!/usr/bin/env bash
#SBATCH --job-name=kan-pysr-sr
#SBATCH --partition=standard
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=logs/greatlakes/kan-pysr-sr_%A_%a.out
#SBATCH --error=logs/greatlakes/kan-pysr-sr_%A_%a.err

set -euo pipefail

# PySR / SymbolicRegression.jl baseline over the standard-formula cards.
#
# Submit from Great Lakes after installing/validating PySR in the repo venv:
#
#   sbatch --account=jaabell0 --array=0-14 \
#     --export=ALL,PYTHON_BIN=$PWD/.venv/bin/python,SEEDS_PER_TASK=2,NITERATIONS=120,POPULATION_SIZE=80 \
#     scripts/greatlakes_pysr_standard_formula.sbatch
#
# This is CPU-only.  If PySR/JL is not installed, the job fails early with a
# clear import error and does not write partial adapter rows.

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs/greatlakes

PYTHON_BIN="${PYTHON_BIN:-$PWD/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python)"
fi

"$PYTHON_BIN" scripts/generate_standard_formula_task_cards.py

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
SEEDS_PER_TASK="${SEEDS_PER_TASK:-2}"
SEED_START=$(( TASK_ID * SEEDS_PER_TASK ))
SEED_STOP=$(( SEED_START + SEEDS_PER_TASK - 1 ))
NITERATIONS="${NITERATIONS:-100}"
POPULATIONS="${POPULATIONS:-12}"
POPULATION_SIZE="${POPULATION_SIZE:-64}"
MAXSIZE="${MAXSIZE:-20}"
PARSIMONY="${PARSIMONY:-0.003}"
PROCS="${PROCS:-${SLURM_CPUS_PER_TASK:-8}}"
LABEL_SUFFIX="${LABEL_SUFFIX:-s${SEED_START}_${SEED_STOP}}"
OUT_DIR="results/revision/pysr_standard_formula_baseline_${LABEL_SUFFIX}"

echo "[$(date -Is)] host=$(hostname) task=${TASK_ID} seeds=${SEED_START}-${SEED_STOP} iter=${NITERATIONS} pop=${POPULATION_SIZE} procs=${PROCS}"
"$PYTHON_BIN" scripts/run_pysr_standard_formula_baseline.py \
  --seeds "${SEED_START}-${SEED_STOP}" \
  --niterations "${NITERATIONS}" \
  --populations "${POPULATIONS}" \
  --population-size "${POPULATION_SIZE}" \
  --maxsize "${MAXSIZE}" \
  --parsimony "${PARSIMONY}" \
  --procs "${PROCS}" \
  --out-dir "$OUT_DIR"
echo "[$(date -Is)] done PySR standard formula baseline -> $OUT_DIR"
