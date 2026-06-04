#!/usr/bin/env bash
#SBATCH --job-name=kan-a40-claimcard
#SBATCH --partition=spgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/greatlakes/%x_%A_%a.out
#SBATCH --error=logs/greatlakes/%x_%A_%a.err

set -euo pipefail

# A40/spgpu follow-up for the paper's main claim-card story.
#
# This is intentionally not a generic seed dump.  It expands the formula rows
# that currently carry the cleanest paper-facing evidence:
#   - aligned positives: bilinear, sqrt-energy
#   - pair-scorer provenance: division-mixed
#   - full-function -> readout/pruning transfer failure: mixed-sparse
#   - core boundary controls: weak-centered clean/noise
#
# Use distinct SEED_BASE/LABEL_SUFFIX values for different accounts so the
# resulting records are non-overlapping:
#
#   sbatch --account=jaabell0 --array=0-9%2 \
#     --export=ALL,PYTHON_BIN=$PWD/.venv/bin/python,SEED_BASE=300,LABEL_SUFFIX=s300_315_j \
#     scripts/greatlakes_spgpu_claimcard_followup_a40.sbatch
#
#   sbatch --account=engin1 --array=0-9%2 \
#     --export=ALL,PYTHON_BIN=$PWD/.venv/bin/python,SEED_BASE=316,LABEL_SUFFIX=s316_331_e \
#     scripts/greatlakes_spgpu_claimcard_followup_a40.sbatch

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs/greatlakes

PY="${PYTHON_BIN:-python}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"

batch_size="${BATCH_SIZE:-16384}"
pair_chunk_size="${PAIR_CHUNK_SIZE:-1500}"

seed_base="${SEED_BASE:-300}"
seed_end="$((seed_base + 15))"
label_suffix="${LABEL_SUFFIX:-s${seed_base}_${seed_end}_a40}"
seeds="${seed_base}-${seed_end}"

echo "[$(date -Is)] node=${SLURM_JOB_NODELIST:-unknown-node}"
echo "[$(date -Is)] seeds=${seeds} label_suffix=${label_suffix}"
echo "[$(date -Is)] batch_size=${batch_size} pair_chunk_size=${pair_chunk_size}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

# label|function|samples|dimension|noise|update_grid|width|steps|anova_points|anova_background
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
out_dir="results/revision/spgpu_a40_claimcard_followup/${label}"
setting="${label}|${function}|${samples}|${dimension}|${noise}|${update_grid}|${width}|${steps}|${seeds}|${anova_points}|${anova_background}"

echo "[$(date -Is)] running A40 claim-card follow-up ${label}"
"$PY" experiments/run_seed_aligned_stage_records.py \
  --settings "$setting" \
  --test-samples 2048 \
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
echo "[$(date -Is)] done A40 claim-card follow-up ${label}"
