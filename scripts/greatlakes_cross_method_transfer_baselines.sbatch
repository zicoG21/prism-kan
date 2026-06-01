#!/bin/bash
#SBATCH --job-name=kan-xfer-base
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-19%8
#SBATCH --output=logs/greatlakes/kan-xfer-base_%A_%a.out
#SBATCH --error=logs/greatlakes/kan-xfer-base_%A_%a.err

set -euo pipefail

# Submit example:
#   mkdir -p logs/greatlakes
#   sbatch --account=engin1 --export=ALL,PYTHON_BIN=$PWD/.venv/bin/python \
#     scripts/greatlakes_cross_method_transfer_baselines.sbatch
#
# This is CPU-only: it does not request GPUs.  It expands the local pilot into
# transfer-compatible sparse-library, GBM-H, GA2M-style, and symbolic-library
# baseline rows.

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs/greatlakes

PY="${PYTHON_BIN:-python}"

FUNCTIONS=(
  core_interaction_c025
  formula_bilinear
  formula_mixed_sparse
  formula_division_mixed
)
METHODS=(
  sparse_spline_lasso
  sparse_lasso
  gbm_hstat
  ga2m_spline
  symbolic_lasso
)

NUM_METHODS=${#METHODS[@]}
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
FUNC_INDEX=$(( TASK_ID / NUM_METHODS ))
METHOD_INDEX=$(( TASK_ID % NUM_METHODS ))

FUNCTION="${FUNCTIONS[$FUNC_INDEX]}"
METHOD="${METHODS[$METHOD_INDEX]}"

SEED_STOP=29
CANDIDATE_PAIRS=160
EXTRA_ARGS=()
if [ "${METHOD}" = "ga2m_spline" ]; then
  # GA2M-style tensor-pair scoring is the strongest but most expensive CPU
  # baseline.  Use a smaller candidate set and 20 seeds for the first expansion;
  # the output records candidate-generation coverage so misses are auditable.
  SEED_STOP=19
  CANDIDATE_PAIRS=80
  EXTRA_ARGS=(--ga2m_prediction_pairs 8 --ga2m_pair_eval_fraction 0.2)
elif [ "${METHOD}" = "symbolic_lasso" ]; then
  # Fixed-library symbolic/SINDy-style regression is a strong positive-control
  # baseline but can be memory-heavy at d=100.  Start with 20 seeds and an
  # auditable prefiltered pair library.
  SEED_STOP=19
  EXTRA_ARGS=(--symbolic_candidate_pairs 500)
fi

SEEDS=$(seq 0 "${SEED_STOP}")
OUT_DIR="results/revision/cross_method_transfer_baselines/gl_${FUNCTION}_${METHOD}_s0_${SEED_STOP}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"

echo "[$(date -Is)] function=${FUNCTION} method=${METHOD} host=$(hostname)"

"${PY}" experiments/run_cross_method_transfer_baselines.py \
  --functions "${FUNCTION}" \
  --methods "${METHOD}" \
  --samples 1024 \
  --test_samples 2048 \
  --dimension 100 \
  --seeds ${SEEDS} \
  --cv 3 \
  --max_iter 3000 \
  --gbm_max_iter 160 \
  --h_background 64 \
  --h_grid_size 5 \
  --candidate_pairs "${CANDIDATE_PAIRS}" \
  --out_dir "${OUT_DIR}" \
  "${EXTRA_ARGS[@]}"

echo "[$(date -Is)] done ${OUT_DIR}"
