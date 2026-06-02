#!/usr/bin/env bash
#SBATCH --job-name=kan-a40-formstage
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

# A40 / spgpu-only formula-family seed-aligned stage records.
#
# These rows are reviewer-facing evidence-transfer traces beyond the core
# weak-centered task.  They are intentionally spgpu-only so they can use the
# A40 nodes when gpu/V100 is backed up.
#
# Submit examples:
#
#   sbatch --account=jaabell0 --array=0-13%5 \
#     --export=ALL,PYTHON_BIN=$PWD/.venv/bin/python \
#     scripts/greatlakes_spgpu_formula_stage_marathon_a40.sbatch
#
#   sbatch --account=engin1 --array=0-13%2 \
#     --export=ALL,PYTHON_BIN=$PWD/.venv/bin/python \
#     scripts/greatlakes_spgpu_formula_stage_marathon_a40.sbatch

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs/greatlakes

PY="${PYTHON_BIN:-python}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-2}"

echo "[$(date -Is)] node=${SLURM_JOB_NODELIST:-unknown-node}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

# label|function|samples|dimension|noise|update_grid|width|steps|seeds|anova_points|anova_background
ROWS=(
  "bilinear_clean_d100_n1024_s200_215_a40|formula_bilinear|1024|100|0.00|0|16|90|200-215|24|24"
  "bilinear_noise005_d100_n1024_s200_215_a40|formula_bilinear|1024|100|0.05|0|16|90|200-215|24|24"
  "weak_centered_clean_d100_n1024_s200_215_a40|formula_weak_centered|1024|100|0.00|0|16|90|200-215|24|24"
  "weak_centered_noise005_d100_n1024_s200_215_a40|formula_weak_centered|1024|100|0.05|0|16|90|200-215|24|24"
  "nested_trig_clean_d100_n1024_s200_215_a40|formula_nested_trig|1024|100|0.00|0|16|90|200-215|24|24"
  "nested_trig_noise005_d100_n1024_s200_215_a40|formula_nested_trig|1024|100|0.05|0|16|90|200-215|24|24"
  "rational_product_clean_d100_n1024_s200_215_a40|formula_rational_product|1024|100|0.00|0|16|90|200-215|24|24"
  "rational_product_noise005_d100_n1024_s200_215_a40|formula_rational_product|1024|100|0.05|0|16|90|200-215|24|24"
  "three_way_clean_d100_n1024_s200_215_a40|formula_three_way_product|1024|100|0.00|0|16|90|200-215|24|24"
  "three_way_noise005_d100_n1024_s200_215_a40|formula_three_way_product|1024|100|0.05|0|16|90|200-215|24|24"
  "mixed_sparse_clean_d100_n1024_s200_215_a40|formula_mixed_sparse|1024|100|0.00|0|16|90|200-215|24|24"
  "mixed_sparse_noise005_d100_n1024_s200_215_a40|formula_mixed_sparse|1024|100|0.05|0|16|90|200-215|24|24"
  "division_mixed_clean_d100_n1024_s200_215_a40|formula_division_mixed|1024|100|0.00|0|16|90|200-215|24|24"
  "sqrt_energy_clean_d100_n1024_s200_215_a40|formula_sqrt_energy|1024|100|0.00|0|16|90|200-215|24|24"
)

row="${ROWS[$SLURM_ARRAY_TASK_ID]}"
IFS='|' read -r label function samples dimension noise update_grid width steps seeds anova_points anova_background <<< "$row"

out_dir="results/revision/spgpu_a40_formula_stage_marathon/${label}"
setting="${label}|${function}|${samples}|${dimension}|${noise}|${update_grid}|${width}|${steps}|${seeds}|${anova_points}|${anova_background}"

echo "[$(date -Is)] running A40 formula stage marathon ${label}"
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
echo "[$(date -Is)] done A40 formula stage marathon ${label}"

