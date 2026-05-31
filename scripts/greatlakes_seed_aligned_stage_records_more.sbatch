#!/usr/bin/env bash
#SBATCH --job-name=kan-seed-stage2
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

# Extra seed-aligned stage-record rows that do not duplicate the first
# seed-stage wave.  These rows target transition settings that are useful for a
# compact reviewer-facing table: clean intermediate n, noise boundaries, and
# grid-update recovery between n=512 and n=1024.
#
# label|function|samples|dimension|noise|update_grid|width|steps|seeds|anova_points|anova_background
ROWS=(
  "clean_w16_n768|core_interaction_c025|768|100|0|0|16|75|10-15|24|24"
  "gridupdate_w16_n768|core_interaction_c025|768|100|0|1|16|75|10-15|24|24"
  "gridupdate_w16_n896|core_interaction_c025|896|100|0|1|16|75|10-15|24|24"
  "noise005_w16_n512|core_interaction_c025|512|100|0.05|0|16|75|10-15|24|24"
  "noise005_w16_n1024|core_interaction_c025|1024|100|0.05|0|16|75|10-15|24|24"
  "noise010_w16_n1024|core_interaction_c025|1024|100|0.10|0|16|75|10-15|24|24"
)

row="${ROWS[$SLURM_ARRAY_TASK_ID]}"
IFS='|' read -r label function samples dimension noise update_grid width steps seeds anova_points anova_background <<< "$row"

out_dir="results/revision/seed_aligned_stage_records_more/${label}"
setting="${label}|${function}|${samples}|${dimension}|${noise}|${update_grid}|${width}|${steps}|${seeds}|${anova_points}|${anova_background}"

echo "[$(date -Is)] running extra seed-aligned stage records ${label} on ${SLURM_JOB_NODELIST:-unknown-node}"
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
  --refit-steps 75 \
  --refit-anova-points 32 \
  --refit-anova-background 32 \
  --prune-workflow prune_input \
  --prune-threshold 0.03 \
  --symbolic-smoke \
  --mse-threshold 0.05 \
  --device cuda \
  --out-dir "$out_dir" \
  --max-table-rows 8
echo "[$(date -Is)] done extra seed-aligned stage records ${label}"
