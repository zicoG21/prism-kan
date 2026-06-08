#!/usr/bin/env bash
#SBATCH --job-name=kan-eql-sr
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

# EQL-style neural-symbolic adapter over the standard-formula suite.
#
# Full 90-seed run on A40:
#
#   sbatch --account=jaabell0 --partition=spgpu --array=0-44 \
#     --export=ALL,PYTHON_BIN=$PWD/.venv/bin/python,SEED_BASE=0,SEEDS_PER_TASK=2,STANDARD_AUGMENT=full \
#     scripts/greatlakes_eql_standard_formula_a40.sbatch
#
# Continue with fresh seeds by changing SEED_BASE. No array throttle is set.

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs/greatlakes

PYTHON_BIN="${PYTHON_BIN:-$PWD/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python)"
fi

STANDARD_AUGMENT="${STANDARD_AUGMENT:-full}"
TASK_CARDS="${TASK_CARDS:-task_cards/claimtransfer_v1_standard_formula_public.json}"
TASK_DOC="${TASK_DOC:-task_cards/standard_formula_task_card_map.md}"
TASK_SPLIT="${TASK_SPLIT:-public}"
TASK_REGISTRY_VERSION="${TASK_REGISTRY_VERSION:-claimtransfer_v1_standard_formula_public}"
GENERATE_TASK_CARDS="${GENERATE_TASK_CARDS:-1}"
if [[ "${GENERATE_TASK_CARDS}" == "1" ]]; then
  "$PYTHON_BIN" scripts/generate_standard_formula_task_cards.py \
    --augment "${STANDARD_AUGMENT}" \
    --out "${TASK_CARDS}" \
    --doc "${TASK_DOC}" \
    --split "${TASK_SPLIT}" \
    --registry-version "${TASK_REGISTRY_VERSION}"
fi

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
SEED_BASE="${SEED_BASE:-0}"
SEEDS_PER_TASK="${SEEDS_PER_TASK:-2}"
SEED_START=$(( SEED_BASE + TASK_ID * SEEDS_PER_TASK ))
SEED_STOP=$(( SEED_START + SEEDS_PER_TASK - 1 ))
UNITS="${UNITS:-48}"
STEPS="${STEPS:-2200}"
BATCH_SIZE="${BATCH_SIZE:-512}"
L1="${L1:-1e-4}"
EXPLAIN_POINTS="${EXPLAIN_POINTS:-128}"
HESSIAN_POINTS="${HESSIAN_POINTS:-32}"
LABEL_SUFFIX="${LABEL_SUFFIX:-s${SEED_START}_${SEED_STOP}}"
OUT_DIR="results/revision/eql_standard_formula_baseline_${LABEL_SUFFIX}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"

echo "[$(date -Is)] host=$(hostname) task=${TASK_ID} seeds=${SEED_START}-${SEED_STOP} units=${UNITS} steps=${STEPS} augment=${STANDARD_AUGMENT} task_cards=${TASK_CARDS}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
"$PYTHON_BIN" scripts/run_eql_standard_formula_baseline.py \
  --task-cards "${TASK_CARDS}" \
  --seeds "${SEED_START}-${SEED_STOP}" \
  --units "${UNITS}" \
  --steps "${STEPS}" \
  --batch-size "${BATCH_SIZE}" \
  --l1 "${L1}" \
  --explain-points "${EXPLAIN_POINTS}" \
  --hessian-points "${HESSIAN_POINTS}" \
  --device cuda \
  --out-dir "$OUT_DIR"
echo "[$(date -Is)] done EQL standard formula baseline -> $OUT_DIR"
