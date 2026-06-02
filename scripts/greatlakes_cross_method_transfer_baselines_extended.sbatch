#!/bin/bash
#SBATCH --job-name=kan-xfer-ext
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=36G
#SBATCH --array=0-49
#SBATCH --time=04:00:00
#SBATCH --output=logs/greatlakes/kan-xfer-ext_%A_%a.out
#SBATCH --error=logs/greatlakes/kan-xfer-ext_%A_%a.err

set -euo pipefail

# Extended cross-adapter coverage for ClaimTransfer-Bench.
#
# Purpose:
#   strengthen the method-agnostic benchmark claim without adding another
#   pyKAN-only experiment block. Each array task runs one task-card / adapter
#   combination and writes the same claim_record schema used by the main
#   cross-adapter matrix.
#
# Submit from Great Lakes:
#
#   sbatch --account=jaabell0 --export=ALL,PYTHON_BIN=$PWD/.venv/bin/python \
#     scripts/greatlakes_cross_method_transfer_baselines_extended.sbatch

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs/greatlakes

PY="${PYTHON_BIN:-python}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"

# label|function|noise
TASKS=(
  "core_clean|core_interaction_c025|0.00"
  "core_noise010|core_interaction_c025|0.10"
  "bilinear_clean|formula_bilinear|0.00"
  "weak_centered_clean|formula_weak_centered|0.00"
  "weak_centered_noise005|formula_weak_centered|0.05"
  "mixed_sparse_clean|formula_mixed_sparse|0.00"
  "mixed_sparse_noise005|formula_mixed_sparse|0.05"
  "division_mixed_clean|formula_division_mixed|0.00"
  "rational_product_clean|formula_rational_product|0.00"
  "nested_trig_clean|formula_nested_trig|0.00"
)

METHODS=(
  "sparse_spline_lasso"
  "sparse_lasso"
  "gbm_hstat"
  "ga2m_spline"
  "symbolic_lasso"
)

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
NUM_METHODS=${#METHODS[@]}
TASK_INDEX=$(( TASK_ID / NUM_METHODS ))
METHOD_INDEX=$(( TASK_ID % NUM_METHODS ))

IFS='|' read -r TASK_LABEL FUNCTION NOISE <<< "${TASKS[$TASK_INDEX]}"
METHOD="${METHODS[$METHOD_INDEX]}"

SEED_START="${SEED_START:-40}"
SEED_STOP="${SEED_STOP:-69}"
CANDIDATE_PAIRS=240
EXTRA_ARGS=()

if [[ "${METHOD}" == "ga2m_spline" ]]; then
  # GA2M-style pair evaluation is the slowest CPU adapter. Keep it broad but
  # not extravagant; the record includes candidate-generation coverage.
  SEED_STOP="${GA2M_SEED_STOP:-59}"
  CANDIDATE_PAIRS=120
  EXTRA_ARGS=(--ga2m_prediction_pairs 12 --ga2m_pair_eval_fraction 0.25)
elif [[ "${METHOD}" == "symbolic_lasso" ]]; then
  SEED_STOP="${SYMBOLIC_SEED_STOP:-59}"
  CANDIDATE_PAIRS=240
  EXTRA_ARGS=(--symbolic_candidate_pairs 800)
fi

SEEDS=$(seq "${SEED_START}" "${SEED_STOP}")
OUT_DIR="results/revision/cross_method_transfer_baselines_extended/${TASK_LABEL}_${METHOD}_s${SEED_START}_${SEED_STOP}"

echo "[$(date -Is)] host=$(hostname) task=${TASK_LABEL} function=${FUNCTION} noise=${NOISE} method=${METHOD} seeds=${SEED_START}-${SEED_STOP}"

"${PY}" experiments/run_cross_method_transfer_baselines.py \
  --functions "${FUNCTION}" \
  --methods "${METHOD}" \
  --samples 1024 \
  --test_samples 2048 \
  --dimension 100 \
  --noise "${NOISE}" \
  --seeds ${SEEDS} \
  --cv 3 \
  --max_iter 3000 \
  --gbm_max_iter 180 \
  --h_background 64 \
  --h_grid_size 5 \
  --candidate_pairs "${CANDIDATE_PAIRS}" \
  --out_dir "${OUT_DIR}" \
  "${EXTRA_ARGS[@]}"

echo "[$(date -Is)] done ${OUT_DIR}"

