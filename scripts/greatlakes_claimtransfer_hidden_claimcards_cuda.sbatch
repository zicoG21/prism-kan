#!/usr/bin/env bash
#SBATCH --job-name=kan-ct-hidden
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

# ClaimTransfer-Bench heldout/private-seed claim-card rows.
#
# Purpose:
#   This is the GPU/A40 training block for the fuller benchmark version.  It
#   runs the same official claim-card contract on fresh seed blocks and a small
#   set of heldout-style cards that are not used as main-text examples.  These
#   rows are meant for coverage/robustness reports, not for adding another
#   table to the workshop story.
#
# Submit examples from Great Lakes.  Command-line --partition can override the
# default gpu partition when A40/spgpu is available.
#
#   sbatch --account=jaabell0 --array=0-11 \
#     --export=ALL,PYTHON_BIN=$PWD/.venv/bin/python,SEED_BASE=2000,SEED_COUNT=12,LABEL_SUFFIX=s2000_2011_gpu \
#     scripts/greatlakes_claimtransfer_hidden_claimcards_cuda.sbatch
#
#   sbatch --account=jaabell0 --partition=spgpu --array=0-11 \
#     --export=ALL,PYTHON_BIN=$PWD/.venv/bin/python,SEED_BASE=2300,SEED_COUNT=12,LABEL_SUFFIX=s2300_2311_a40 \
#     scripts/greatlakes_claimtransfer_hidden_claimcards_cuda.sbatch
#
# No array throttle is set here.  Let account/partition GRES limits determine
# concurrency.

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs/greatlakes

PY="${PYTHON_BIN:-python}"

if [[ "${SLURM_JOB_PARTITION:-}" == *"spgpu"* ]]; then
  default_batch_size=16384
  default_pair_chunk_size=1500
  default_threads=4
else
  default_batch_size=8192
  default_pair_chunk_size=1000
  default_threads=2
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$default_threads}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$default_threads}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$default_threads}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-$default_threads}"

batch_size="${BATCH_SIZE:-$default_batch_size}"
pair_chunk_size="${PAIR_CHUNK_SIZE:-$default_pair_chunk_size}"

seed_base="${SEED_BASE:-2000}"
seed_count="${SEED_COUNT:-12}"
seed_end="$((seed_base + seed_count - 1))"
seeds="${seed_base}-${seed_end}"
label_suffix="${LABEL_SUFFIX:-s${seed_base}_${seed_end}}"

echo "[$(date -Is)] node=${SLURM_JOB_NODELIST:-unknown-node}"
echo "[$(date -Is)] seeds=${seeds} label_suffix=${label_suffix}"
echo "[$(date -Is)] partition=${SLURM_JOB_PARTITION:-unknown} batch_size=${batch_size} pair_chunk_size=${pair_chunk_size} threads=${OMP_NUM_THREADS}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

# label|function|samples|dimension|noise|update_grid|width|steps|anova_points|anova_background|prune_threshold
ROWS=(
  # Public-core fresh seeds: stability of the official score report.
  "public_weak_centered_clean|formula_weak_centered|1024|100|0.00|0|16|90|24|24|0.03"
  "public_weak_centered_noise005|formula_weak_centered|1024|100|0.05|0|16|90|24|24|0.03"
  "public_mixed_sparse_clean|formula_mixed_sparse|1024|100|0.00|0|16|90|24|24|0.03"
  "public_division_mixed_clean|formula_division_mixed|1024|100|0.00|0|16|90|24|24|0.03"

  # Heldout-style cards: same schema, not main-text anchors.
  "heldout_bilinear_noise010|formula_bilinear|1024|100|0.10|0|16|90|24|24|0.03"
  "heldout_trig_product_noise005|formula_trig_product|1024|100|0.05|0|16|90|24|24|0.03"
  "heldout_exp_product_clean|formula_exp_product|1024|100|0.00|0|16|90|24|24|0.03"
  "heldout_log_product_clean|formula_log_product|1024|100|0.00|0|16|90|24|24|0.03"
  "heldout_rational_noise005|formula_rational_product|1024|100|0.05|0|16|90|24|24|0.03"
  "heldout_sqrt_energy_noise005|formula_sqrt_energy|1024|100|0.05|0|16|90|24|24|0.03"

  # Protocol variation cards: grid-update and sparse-pruning boundary.
  "heldout_weak_centered_grid|formula_weak_centered|1024|100|0.00|1|16|90|24|24|0.03"
  "heldout_mixed_sparse_prune010|formula_mixed_sparse|1024|100|0.00|0|16|90|24|24|0.10"
)

row="${ROWS[$SLURM_ARRAY_TASK_ID]}"
IFS='|' read -r base_label function samples dimension noise update_grid width steps anova_points anova_background prune_threshold <<< "$row"

label="${base_label}_${label_suffix}"
out_dir="results/revision/claimtransfer_hidden_claimcards/${label}"
setting="${label}|${function}|${samples}|${dimension}|${noise}|${update_grid}|${width}|${steps}|${seeds}|${anova_points}|${anova_background}"

echo "[$(date -Is)] running ClaimTransfer hidden/fresh claimcard ${label}"
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
  --prune-threshold "$prune_threshold" \
  --symbolic-smoke \
  --mse-threshold 0.05 \
  --device cuda \
  --out-dir "$out_dir" \
  --max-table-rows 10
echo "[$(date -Is)] done ClaimTransfer hidden/fresh claimcard ${label}"
