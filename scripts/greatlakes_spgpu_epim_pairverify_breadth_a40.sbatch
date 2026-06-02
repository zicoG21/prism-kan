#!/usr/bin/env bash
#SBATCH --job-name=kan-a40-epim-breadth
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

# A40 / spgpu-only EPIM PairVerify breadth queue.
#
# This complements `greatlakes_spgpu_epim_pairverify_a40.sbatch` without
# duplicating its rows.  The goal is not another core-task sweep; it is to test
# whether EPIM-propose / ANOVA-verify behaves like a reusable claim-provenance
# primitive across formulas where the declared pair predicate is less obvious.
#
# Suggested split when both accounts have A40 quota:
#
#   sbatch --account=jaabell0 --array=0-3%2 \
#     --export=ALL,PYTHON_BIN=$PWD/.venv/bin/python \
#     scripts/greatlakes_spgpu_epim_pairverify_breadth_a40.sbatch
#
#   sbatch --account=engin1 --array=4-7%2 \
#     --export=ALL,PYTHON_BIN=$PWD/.venv/bin/python \
#     scripts/greatlakes_spgpu_epim_pairverify_breadth_a40.sbatch

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs/greatlakes

PY="${PYTHON_BIN:-python}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-2}"

echo "[$(date -Is)] node=${SLURM_JOB_NODELIST:-unknown-node}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

# label|function|samples|dimension|noise|update_grid|width|steps|proposal_qs|random_controls|endpoint_top_m|seeds
ROWS=(
  "formula_weak_centered_clean_qs_a40|formula_weak_centered|1024|100|0.00|0|16|90|8,16,32,64,128|128|4|0-29"
  "formula_weak_centered_noise005_qs_a40|formula_weak_centered|1024|100|0.05|0|16|90|8,16,32,64,128|128|4|0-29"
  "formula_nested_trig_clean_qs_a40|formula_nested_trig|1024|100|0.00|0|16|90|8,16,32,64,128|128|4|0-29"
  "formula_nested_trig_noise005_qs_a40|formula_nested_trig|1024|100|0.05|0|16|90|8,16,32,64,128|128|4|0-29"
  "formula_three_way_clean_qs_a40|formula_three_way_product|1024|100|0.00|0|16|90|8,16,32,64,128|128|4|0-29"
  "formula_three_way_noise005_qs_a40|formula_three_way_product|1024|100|0.05|0|16|90|8,16,32,64,128|128|4|0-29"
  "formula_sqrt_energy_clean_qs_a40|formula_sqrt_energy|1024|100|0.00|0|16|90|8,16,32,64,128|128|4|0-29"
  "formula_trig_product_clean_qs_a40|formula_trig_product|1024|100|0.00|0|16|90|8,16,32,64,128|128|4|0-29"
)

row="${ROWS[$SLURM_ARRAY_TASK_ID]}"
IFS='|' read -r label function samples dimension noise update_grid width steps proposal_qs random_controls endpoint_top_m seeds <<< "$row"

if [[ -n "${SEEDS:-}" ]]; then
  seeds="${SEEDS}"
fi
if [[ -n "${LABEL_SUFFIX:-}" ]]; then
  label="${label}_${LABEL_SUFFIX}"
fi

out_dir="results/revision/epim_pairverify_breadth_a40/${label}"
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
  --proposal-qs "$proposal_qs"
  --random-controls "$random_controls"
  --endpoint-top-m "$endpoint_top_m"
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

echo "[$(date -Is)] running A40 EPIM PairVerify breadth ${label}"
"$PY" "${args[@]}"
echo "[$(date -Is)] done A40 EPIM PairVerify breadth ${label}"
