#!/usr/bin/env bash
#SBATCH --job-name=kan-form-seed
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=28G
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/greatlakes/%x_%A_%a.out
#SBATCH --error=logs/greatlakes/%x_%A_%a.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs/greatlakes

PY="${PYTHON_BIN:-python}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-2}"

# Formula-family seed-aligned records for the engin1 side queue.
#
# These do not duplicate the jaabell0 formula-SDR array: the jaabell0 job
# produces aggregate full-reliance/readout rows, while this job carries the same
# seed/model through full-KAN ANOVA, exposed readout, refit, and pruning.
#
# label|function|samples|dimension|noise|update_grid|width|steps|seeds|anova_points|anova_background
ROWS=(
  "bilinear_d100_n1024|formula_bilinear|1024|100|0|0|16|90|20-27|24|24"
  "weak_centered_formula_d100_n1024|formula_weak_centered|1024|100|0|0|16|90|20-27|24|24"
  "trig_product_d100_n1024|formula_trig_product|1024|100|0|0|16|90|20-27|24|24"
  "nested_trig_d100_n1024|formula_nested_trig|1024|100|0|0|16|90|20-27|24|24"
  "log_product_d100_n1024|formula_log_product|1024|100|0|0|16|90|20-27|24|24"
  "exp_product_d100_n1024|formula_exp_product|1024|100|0|0|16|90|20-27|24|24"
  "rational_product_d100_n1024|formula_rational_product|1024|100|0|0|16|90|20-27|24|24"
  "three_way_product_d100_n1024|formula_three_way_product|1024|100|0|0|16|90|20-27|24|24"
)

row="${ROWS[$SLURM_ARRAY_TASK_ID]}"
IFS='|' read -r label function samples dimension noise update_grid width steps seeds anova_points anova_background <<< "$row"

out_dir="results/revision/engin1_formula_seed_aligned_stage_records/${label}"
setting="${label}|${function}|${samples}|${dimension}|${noise}|${update_grid}|${width}|${steps}|${seeds}|${anova_points}|${anova_background}"

echo "[$(date -Is)] running engin1 formula seed-aligned records ${label} on ${SLURM_JOB_NODELIST:-unknown-node}"
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
  --max-table-rows 8
echo "[$(date -Is)] done engin1 formula seed-aligned records ${label}"
