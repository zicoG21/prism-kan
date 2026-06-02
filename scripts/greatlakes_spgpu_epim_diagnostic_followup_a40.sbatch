#!/usr/bin/env bash
#SBATCH --job-name=kan-a40-epim-diag
#SBATCH --partition=spgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/greatlakes/%x_%A_%a.out
#SBATCH --error=logs/greatlakes/%x_%A_%a.err

set -euo pipefail

# A40/spgpu EPIM diagnostic follow-up.
#
# This is lower priority than `greatlakes_spgpu_claimcard_followup_a40.sbatch`.
# It exists only to test whether the EPIM candidate/verifier split is stable
# under fresh seed blocks for the few rows that are most useful to the paper:
# rational/bilinear positives, mixed-sparse candidate recovery, division-mixed
# proposal-vs-verifier disagreement, and the core grid/noise boundary.
#
# Submit only when spgpu is genuinely open:
#
#   sbatch --account=jaabell0 --array=0-7%2 \
#     --export=ALL,PYTHON_BIN=$PWD/.venv/bin/python,SEED_BASE=30,LABEL_SUFFIX=s30_59_j \
#     scripts/greatlakes_spgpu_epim_diagnostic_followup_a40.sbatch
#
#   sbatch --account=engin1 --array=0-7%2 \
#     --export=ALL,PYTHON_BIN=$PWD/.venv/bin/python,SEED_BASE=60,LABEL_SUFFIX=s60_89_e \
#     scripts/greatlakes_spgpu_epim_diagnostic_followup_a40.sbatch

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs/greatlakes

PY="${PYTHON_BIN:-python}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-2}"

seed_base="${SEED_BASE:-30}"
seed_end="$((seed_base + 29))"
label_suffix="${LABEL_SUFFIX:-s${seed_base}_${seed_end}_a40}"
seeds="${seed_base}-${seed_end}"

echo "[$(date -Is)] node=${SLURM_JOB_NODELIST:-unknown-node}"
echo "[$(date -Is)] seeds=${seeds} label_suffix=${label_suffix}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

# label|function|samples|dimension|noise|update_grid|width|steps
ROWS=(
  "formula_bilinear|formula_bilinear|1024|100|0.00|0|16|90"
  "formula_rational_product|formula_rational_product|1024|100|0.00|0|16|90"
  "formula_division_mixed|formula_division_mixed|1024|100|0.00|0|16|90"
  "formula_mixed_sparse|formula_mixed_sparse|1024|100|0.00|0|16|90"
  "formula_mixed_sparse_noise005|formula_mixed_sparse|1024|100|0.05|0|16|90"
  "core_clean_n1024|core_interaction_c025|1024|100|0.00|0|16|75"
  "core_grid_n1024|core_interaction_c025|1024|100|0.00|1|16|75"
  "core_noise010_n1024|core_interaction_c025|1024|100|0.10|0|16|75"
)

row="${ROWS[$SLURM_ARRAY_TASK_ID]}"
IFS='|' read -r base_label function samples dimension noise update_grid width steps <<< "$row"

label="${base_label}_${label_suffix}"
out_dir="results/revision/epim_pairverify_diagnostic_a40/${label}"
args=(
  scripts/run_kan_epim_pairverify.py
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
  --proposal-qs 16,32,64,128
  --random-controls 128
  --endpoint-top-m 4
  --anova-points 24
  --anova-background 24
  --pair-chunk-size 256
  --batch-size 8192
  --score-true-pair-for-diagnostics
  --device cuda
  --out-dir "$out_dir"
)

if [[ "$update_grid" == "1" ]]; then
  args+=(--update-grid --grid-update-num 5)
fi

echo "[$(date -Is)] running A40 EPIM diagnostic follow-up ${label}"
"$PY" "${args[@]}"
echo "[$(date -Is)] done A40 EPIM diagnostic follow-up ${label}"

