#!/usr/bin/env bash
#SBATCH --job-name=kan-mlphess-sr
#SBATCH --partition=spgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/greatlakes/%x_%A_%a.out
#SBATCH --error=logs/greatlakes/%x_%A_%a.err

set -euo pipefail

# MLP + gradient/Hessian baseline over standard-formula cards.
#
# Submit from Great Lakes:
#
#   sbatch --account=jaabell0 --partition=spgpu --array=0-14 \
#     --export=ALL,PYTHON_BIN=$PWD/.venv/bin/python,SEEDS_PER_TASK=2,STEPS=1800 \
#     scripts/greatlakes_mlp_hessian_standard_formula_a40.sbatch
#
# No array throttle is set.  Override --partition=gpu for V100s if useful.

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
WIDTH="${WIDTH:-128}"
DEPTH="${DEPTH:-3}"
STEPS="${STEPS:-1800}"
BATCH_SIZE="${BATCH_SIZE:-512}"
EXPLAIN_POINTS="${EXPLAIN_POINTS:-128}"
HESSIAN_POINTS="${HESSIAN_POINTS:-32}"
LABEL_SUFFIX="${LABEL_SUFFIX:-s${SEED_START}_${SEED_STOP}}"
OUT_DIR="results/revision/mlp_hessian_standard_formula_baseline_${LABEL_SUFFIX}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"

echo "[$(date -Is)] host=$(hostname) task=${TASK_ID} seeds=${SEED_START}-${SEED_STOP} width=${WIDTH} depth=${DEPTH} steps=${STEPS}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
"$PYTHON_BIN" scripts/run_mlp_hessian_standard_formula_baseline.py \
  --seeds "${SEED_START}-${SEED_STOP}" \
  --width "${WIDTH}" \
  --depth "${DEPTH}" \
  --steps "${STEPS}" \
  --batch-size "${BATCH_SIZE}" \
  --explain-points "${EXPLAIN_POINTS}" \
  --hessian-points "${HESSIAN_POINTS}" \
  --device cuda \
  --out-dir "$OUT_DIR"
echo "[$(date -Is)] done MLP-Hessian standard formula baseline -> $OUT_DIR"
