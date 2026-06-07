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
#     --export=ALL,PYTHON_BIN=$PWD/.venv/bin/python,SEED_BASE=0,SEEDS_PER_TASK=2,NITERATIONS=120,POPULATION_SIZE=80,STANDARD_AUGMENT=full \
#     scripts/greatlakes_pysr_standard_formula.sbatch
#
# Continue with fresh seeds by changing SEED_BASE, for example SEED_BASE=30
# with --array=0-14 covers seeds 30-59.
#
# This is CPU-only.  If PySR/JL is not installed, the job fails early with a
# clear import error and does not write partial adapter rows.
# Some Great Lakes standard nodes intermittently fail PySR import with
# "No locks available" when Julia/Python caches lock on network filesystems.
# Keep writable caches in node-local scratch; read installed Julia packages
# from the user's depot as a fallback.

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs/greatlakes

JOB_TMP="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/claimtransfer_pysr_${SLURM_JOB_ID:-manual}_${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "$JOB_TMP"/{julia_depot,python_cache,compiled,mpl}
export TMPDIR="$JOB_TMP"
export PYTHONPYCACHEPREFIX="$JOB_TMP/python_cache"
export MPLCONFIGDIR="$JOB_TMP/mpl"
export JULIA_DEPOT_PATH="$JOB_TMP/julia_depot:${JULIA_DEPOT_PATH:-$HOME/.julia}"
export JULIA_PKG_PRECOMPILE_AUTO="${JULIA_PKG_PRECOMPILE_AUTO:-0}"
export JULIA_NUM_THREADS="${JULIA_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"

PYTHON_BIN="${PYTHON_BIN:-$PWD/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python)"
fi

STANDARD_AUGMENT="${STANDARD_AUGMENT:-none}"
"$PYTHON_BIN" scripts/generate_standard_formula_task_cards.py --augment "${STANDARD_AUGMENT}"

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
SEED_BASE="${SEED_BASE:-0}"
SEEDS_PER_TASK="${SEEDS_PER_TASK:-2}"
SEED_START=$(( SEED_BASE + TASK_ID * SEEDS_PER_TASK ))
SEED_STOP=$(( SEED_START + SEEDS_PER_TASK - 1 ))
NITERATIONS="${NITERATIONS:-100}"
POPULATIONS="${POPULATIONS:-12}"
POPULATION_SIZE="${POPULATION_SIZE:-64}"
MAXSIZE="${MAXSIZE:-20}"
PARSIMONY="${PARSIMONY:-0.003}"
PROCS="${PROCS:-${SLURM_CPUS_PER_TASK:-8}}"
LABEL_SUFFIX="${LABEL_SUFFIX:-s${SEED_START}_${SEED_STOP}}"
OUT_DIR="results/revision/pysr_standard_formula_baseline_${LABEL_SUFFIX}"

echo "[$(date -Is)] host=$(hostname) task=${TASK_ID} seeds=${SEED_START}-${SEED_STOP} iter=${NITERATIONS} pop=${POPULATION_SIZE} procs=${PROCS} augment=${STANDARD_AUGMENT}"
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
