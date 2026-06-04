#!/usr/bin/env bash
#SBATCH --job-name=kan-xfer-gap
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=36G
#SBATCH --array=0-49
#SBATCH --time=04:00:00
#SBATCH --output=logs/greatlakes/%x_%A_%a.out
#SBATCH --error=logs/greatlakes/%x_%A_%a.err

set -euo pipefail

# CPU-only ClaimTransfer coverage gap fill.
#
# This is not another broad seed sweep.  It targets public task families that
# the original cross-method sweep did not cover: exp/log/sqrt/trig/three-way,
# Feynman-style scientific templates, and correlated-covariate templates.

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs/greatlakes

PY="${PYTHON_BIN:-$PWD/.venv/bin/python}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"

# label|function|noise|rho|proxies|samples|dimension
TASKS=(
  "trig_product_gap|formula_trig_product|0.00|0.0|0|1024|100"
  "exp_product_gap|formula_exp_product|0.00|0.0|0|1024|100"
  "log_product_gap|formula_log_product|0.00|0.0|0|1024|100"
  "sqrt_energy_gap|formula_sqrt_energy|0.00|0.0|0|1024|100"
  "three_way_gap|formula_three_way_product|0.00|0.0|0|1024|100"
  "scientific_energy_gap|feynman_energy|0.00|0.0|0|2048|100"
  "correlated_core_gap|core_interaction_c025|0.00|0.75|8|2048|100"
  "semi_breast_cancer_gap|semisynthetic_breast_cancer|0.00|0.0|0|1024|30"
  "semi_diabetes_gap|semisynthetic_diabetes|0.00|0.0|0|1024|10"
  "semi_wine_gap|semisynthetic_wine|0.00|0.0|0|1024|13"
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

IFS='|' read -r TASK_LABEL FUNCTION NOISE RHO PROXIES SAMPLES DIMENSION <<< "${TASKS[$TASK_INDEX]}"
METHOD="${METHODS[$METHOD_INDEX]}"

SEED_START="${SEED_START:-160}"
SEED_STOP="${SEED_STOP:-189}"
CANDIDATE_PAIRS="${CANDIDATE_PAIRS:-240}"
EXTRA_ARGS=()

if [[ "${METHOD}" == "ga2m_spline" ]]; then
  SEED_STOP="${GA2M_SEED_STOP:-179}"
  CANDIDATE_PAIRS="${GA2M_CANDIDATE_PAIRS:-120}"
  EXTRA_ARGS=(--ga2m_prediction_pairs 12 --ga2m_pair_eval_fraction 0.25)
elif [[ "${METHOD}" == "symbolic_lasso" ]]; then
  SEED_STOP="${SYMBOLIC_SEED_STOP:-179}"
  CANDIDATE_PAIRS="${SYMBOLIC_CANDIDATE_PAIRS:-240}"
  EXTRA_ARGS=(--symbolic_candidate_pairs 800)
fi

SEEDS=$(seq "${SEED_START}" "${SEED_STOP}")
OUT_DIR="results/revision/cross_method_transfer_gapfill/${TASK_LABEL}_${METHOD}_s${SEED_START}_${SEED_STOP}"

echo "[$(date -Is)] host=$(hostname) task=${TASK_LABEL} function=${FUNCTION} noise=${NOISE} rho=${RHO} proxies=${PROXIES} method=${METHOD} seeds=${SEED_START}-${SEED_STOP}"

"${PY}" experiments/run_cross_method_transfer_baselines.py \
  --functions "${FUNCTION}" \
  --methods "${METHOD}" \
  --samples "${SAMPLES}" \
  --test_samples 2048 \
  --dimension "${DIMENSION}" \
  --noise "${NOISE}" \
  --nuisance_correlation "${RHO}" \
  --n_correlated_proxies "${PROXIES}" \
  --semisynthetic_c "${SEMISYNTHETIC_C:-0.25}" \
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
