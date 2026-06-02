#!/usr/bin/env bash
#SBATCH --job-name=kan-gpu-claimcard
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/greatlakes/%x_%A_%a.out
#SBATCH --error=logs/greatlakes/%x_%A_%a.err

set -euo pipefail

# V100/gpu follow-up for the same claim-card rows as the A40 script.
# Use this only when gpu/V100 capacity is open after the long full-capacity
# jobs.  Pick a SEED_BASE that does not overlap the A40 follow-up.
#
#   sbatch --account=jaabell0 --array=0-9%2 \
#     --export=ALL,PYTHON_BIN=$PWD/.venv/bin/python,SEED_BASE=332,LABEL_SUFFIX=s332_347_gpu \
#     scripts/greatlakes_gpu_claimcard_followup.sbatch

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs/greatlakes

PY="${PYTHON_BIN:-python}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-2}"

seed_base="${SEED_BASE:-332}"
seed_end="$((seed_base + 15))"
label_suffix="${LABEL_SUFFIX:-s${seed_base}_${seed_end}_gpu}"
seeds="${seed_base}-${seed_end}"

echo "[$(date -Is)] node=${SLURM_JOB_NODELIST:-unknown-node}"
echo "[$(date -Is)] seeds=${seeds} label_suffix=${label_suffix}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

ROWS=(
  "bilinear_clean|formula_bilinear|1024|100|0.00|0|16|90|24|24"
  "bilinear_noise005|formula_bilinear|1024|100|0.05|0|16|90|24|24"
  "sqrt_energy_clean|formula_sqrt_energy|1024|100|0.00|0|16|90|24|24"
  "rational_product_clean|formula_rational_product|1024|100|0.00|0|16|90|24|24"
  "rational_product_noise005|formula_rational_product|1024|100|0.05|0|16|90|24|24"
  "division_mixed_clean|formula_division_mixed|1024|100|0.00|0|16|90|24|24"
  "mixed_sparse_clean|formula_mixed_sparse|1024|100|0.00|0|16|90|24|24"
  "mixed_sparse_noise005|formula_mixed_sparse|1024|100|0.05|0|16|90|24|24"
  "weak_centered_clean|formula_weak_centered|1024|100|0.00|0|16|90|24|24"
  "weak_centered_noise005|formula_weak_centered|1024|100|0.05|0|16|90|24|24"
)

row="${ROWS[$SLURM_ARRAY_TASK_ID]}"
IFS='|' read -r base_label function samples dimension noise update_grid width steps anova_points anova_background <<< "$row"

label="${base_label}_${label_suffix}"
out_dir="results/revision/greatlakes_claimcard_followup/${label}"
setting="${label}|${function}|${samples}|${dimension}|${noise}|${update_grid}|${width}|${steps}|${seeds}|${anova_points}|${anova_background}"

echo "[$(date -Is)] running GPU claim-card follow-up ${label}"
"$PY" experiments/run_seed_aligned_stage_records.py \
  --settings "$setting" \
  --test-samples 2048 \
  --top-m 4 \
  --grid 5 \
  --k 3 \
  --lamb 0.001 \
  --grid-update-num 5 \
  --batch -1 \
  --batch-size 8192 \
  --pair-chunk-size 1000 \
  --refit-width-hidden 16 \
  --refit-steps 90 \
  --refit-anova-points 32 \
  --refit-anova-background 32 \
  --prune-workflow prune_input \
  --prune-threshold 0.03 \
  --symbolic-smoke \
  --mse-threshold 0.05 \
  --device cuda \
  --out-dir "$out_dir" \
  --max-table-rows 10
echo "[$(date -Is)] done GPU claim-card follow-up ${label}"

