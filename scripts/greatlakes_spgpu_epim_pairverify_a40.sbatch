#!/usr/bin/env bash
#SBATCH --job-name=kan-a40-epim
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

# A40 / spgpu-only EPIM PairVerify queue.
#
# Use this when spgpu A40 nodes are available and gpu/V100 is backed up.
# Array throttle is supplied at submission time so it can match the account:
#
#   # jaabell0: start with five concurrent jobs if the account has room.
#   sbatch --account=jaabell0 --array=0-9%5 \
#     --export=ALL,PYTHON_BIN=$PWD/.venv/bin/python \
#     scripts/greatlakes_spgpu_epim_pairverify_a40.sbatch
#
#   # engin1: quota is usually two GPUs, so keep %2.
#   sbatch --account=engin1 --array=0-9%2 \
#     --export=ALL,PYTHON_BIN=$PWD/.venv/bin/python \
#     scripts/greatlakes_spgpu_epim_pairverify_a40.sbatch
#
# This script is spgpu-only, so it should land on A40-class nodes.  If you want
# spgpu first but gpu fallback, use the non-A40 script with --partition=spgpu,gpu.

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
  "core_clean_n512_qs_a40|core_interaction_c025|512|100|0|0|16|75|8,16,32,64,128|128|4|0-29"
  "core_clean_n1024_qs_a40|core_interaction_c025|1024|100|0|0|16|75|8,16,32,64,128|128|4|0-29"
  "core_grid_n512_qs_a40|core_interaction_c025|512|100|0|1|16|75|8,16,32,64,128|128|4|0-29"
  "core_grid_n1024_qs_a40|core_interaction_c025|1024|100|0|1|16|75|8,16,32,64,128|128|4|0-29"
  "core_noise010_n512_qs_a40|core_interaction_c025|512|100|0.10|0|16|75|8,16,32,64,128|128|4|0-29"
  "core_noise010_n1024_qs_a40|core_interaction_c025|1024|100|0.10|0|16|75|8,16,32,64,128|128|4|0-29"
  "formula_bilinear_qs_a40|formula_bilinear|1024|100|0|0|16|90|8,16,32,64,128|128|4|0-29"
  "formula_mixed_sparse_qs_a40|formula_mixed_sparse|1024|100|0|0|16|90|8,16,32,64,128|128|4|0-29"
  "formula_division_mixed_qs_a40|formula_division_mixed|1024|100|0|0|16|90|8,16,32,64,128|128|4|0-29"
  "formula_rational_product_qs_a40|formula_rational_product|1024|100|0|0|16|90|8,16,32,64,128|128|4|0-29"
)

row="${ROWS[$SLURM_ARRAY_TASK_ID]}"
IFS='|' read -r label function samples dimension noise update_grid width steps proposal_qs random_controls endpoint_top_m seeds <<< "$row"

out_dir="results/revision/epim_pairverify_a40/${label}"
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

echo "[$(date -Is)] running A40 EPIM PairVerify ${label}"
"$PY" "${args[@]}"
echo "[$(date -Is)] done A40 EPIM PairVerify ${label}"

