#!/usr/bin/env bash
#SBATCH --job-name=kan-pykan-gap
#SBATCH --partition=spgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/greatlakes/%x_%A_%a.out
#SBATCH --error=logs/greatlakes/%x_%A_%a.err

set -euo pipefail

# Targeted pyKAN coverage gap fill for the full benchmark.
#
# This is intentionally tiny: it covers the remaining pyKAN public-template
# task families in the coverage report, not another broad scorergram sweep.
#
# Submit from Great Lakes:
#
#   sbatch --account=jaabell0 --array=0-1 \
#     --export=ALL,PYTHON_BIN=$PWD/.venv/bin/python,SEED_BASE=5200,SEED_COUNT=8,LABEL_SUFFIX=s5200_5207 \
#     scripts/greatlakes_pykan_gapfill_a40.sbatch

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs/greatlakes

PY="${PYTHON_BIN:-python}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"

batch_size="${BATCH_SIZE:-16384}"
pair_chunk_size="${PAIR_CHUNK_SIZE:-1500}"
seed_base="${SEED_BASE:-5200}"
seed_count="${SEED_COUNT:-8}"
seed_end="$((seed_base + seed_count - 1))"
seeds="${seed_base}-${seed_end}"
label_suffix="${LABEL_SUFFIX:-s${seed_base}_${seed_end}}"

echo "[$(date -Is)] node=${SLURM_JOB_NODELIST:-unknown-node}"
echo "[$(date -Is)] targeted pyKAN gapfill seeds=${seeds} suffix=${label_suffix}"
echo "[$(date -Is)] batch_size=${batch_size} pair_chunk_size=${pair_chunk_size}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

# label|function|samples|dimension|noise|update_grid|width|steps|anova_points|anova_background|rho|proxies
ROWS=(
  "gap_correlated_covariates|core_interaction_c025|2048|100|0.00|0|16|90|24|24|0.75|8"
  "gap_scientific_expression|feynman_energy|2048|100|0.00|0|16|90|24|24|0.00|0"
)

row="${ROWS[$SLURM_ARRAY_TASK_ID]}"
IFS='|' read -r base_label function samples dimension noise update_grid width steps anova_points anova_background rho proxies <<< "$row"

label="${base_label}_${label_suffix}"
out_dir="results/revision/pykan_gapfill_claimcards/${label}"
setting="${label}|${function}|${samples}|${dimension}|${noise}|${update_grid}|${width}|${steps}|${seeds}|${anova_points}|${anova_background}"

echo "[$(date -Is)] running targeted pyKAN gapfill ${label} function=${function} rho=${rho} proxies=${proxies}"
"$PY" experiments/run_seed_aligned_stage_records.py \
  --settings "$setting" \
  --test-samples 2048 \
  --nuisance-correlation "$rho" \
  --n-correlated-proxies "$proxies" \
  --top-m 4 \
  --grid 5 \
  --k 3 \
  --lamb 0.001 \
  --grid-update-num 5 \
  --batch -1 \
  --batch-size "$batch_size" \
  --pair-chunk-size "$pair_chunk_size" \
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

echo "[$(date -Is)] done targeted pyKAN gapfill ${label}"
