#!/usr/bin/env bash
#SBATCH --job-name=kan-ct-pack
#SBATCH --partition=spgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/greatlakes/%x_%A_%a.out
#SBATCH --error=logs/greatlakes/%x_%A_%a.err

set -euo pipefail

# Packed A40 ClaimTransfer claim-card runner.
#
# Why this exists:
#   A single pyKAN claim-card process is too light for an A40: typical observed
#   usage is ~2GB memory and low GPU utilization.  This script packs several
#   independent seed sub-runs onto one allocated A40.  It is intended for A40
#   / spgpu only, not 16GB V100 nodes.
#
# Submit from Great Lakes:
#
#   sbatch --account=jaabell0 --array=0-11 \
#     --export=ALL,PYTHON_BIN=$PWD/.venv/bin/python,SEED_BASE=3000,PACK_FACTOR=4,SEEDS_PER_LANE=4,LABEL_SUFFIX=s3000_pack4 \
#     scripts/greatlakes_claimtransfer_hidden_claimcards_a40_packed.sbatch
#
# If nvidia-smi still shows low utilization, try PACK_FACTOR=5.  If OOM or
# instability appears, lower to PACK_FACTOR=3.  No array throttle is set here.

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs/greatlakes

PY="${PYTHON_BIN:-python}"

pack_factor="${PACK_FACTOR:-4}"
seeds_per_lane="${SEEDS_PER_LANE:-4}"
seed_base="${SEED_BASE:-3000}"
label_suffix="${LABEL_SUFFIX:-s${seed_base}_pack${pack_factor}}"

threads_per_lane="${THREADS_PER_LANE:-1}"
batch_size="${BATCH_SIZE:-12288}"
pair_chunk_size="${PAIR_CHUNK_SIZE:-1000}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$threads_per_lane}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$threads_per_lane}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$threads_per_lane}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-$threads_per_lane}"

echo "[$(date -Is)] node=${SLURM_JOB_NODELIST:-unknown-node}"
echo "[$(date -Is)] packed A40 run: pack_factor=${pack_factor} seeds_per_lane=${seeds_per_lane} seed_base=${seed_base}"
echo "[$(date -Is)] batch_size=${batch_size} pair_chunk_size=${pair_chunk_size} threads_per_lane=${threads_per_lane}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

# label|function|samples|dimension|noise|update_grid|width|steps|anova_points|anova_background|prune_threshold
ROWS=(
  "public_weak_centered_clean|formula_weak_centered|1024|100|0.00|0|16|90|24|24|0.03"
  "public_weak_centered_noise005|formula_weak_centered|1024|100|0.05|0|16|90|24|24|0.03"
  "public_mixed_sparse_clean|formula_mixed_sparse|1024|100|0.00|0|16|90|24|24|0.03"
  "public_division_mixed_clean|formula_division_mixed|1024|100|0.00|0|16|90|24|24|0.03"
  "heldout_bilinear_noise010|formula_bilinear|1024|100|0.10|0|16|90|24|24|0.03"
  "heldout_trig_product_noise005|formula_trig_product|1024|100|0.05|0|16|90|24|24|0.03"
  "heldout_exp_product_clean|formula_exp_product|1024|100|0.00|0|16|90|24|24|0.03"
  "heldout_log_product_clean|formula_log_product|1024|100|0.00|0|16|90|24|24|0.03"
  "heldout_rational_noise005|formula_rational_product|1024|100|0.05|0|16|90|24|24|0.03"
  "heldout_sqrt_energy_noise005|formula_sqrt_energy|1024|100|0.05|0|16|90|24|24|0.03"
  "heldout_weak_centered_grid|formula_weak_centered|1024|100|0.00|1|16|90|24|24|0.03"
  "heldout_mixed_sparse_prune010|formula_mixed_sparse|1024|100|0.00|0|16|90|24|24|0.10"
)

row="${ROWS[$SLURM_ARRAY_TASK_ID]}"
IFS='|' read -r base_label function samples dimension noise update_grid width steps anova_points anova_background prune_threshold <<< "$row"

pids=()
for lane in $(seq 0 "$((pack_factor - 1))"); do
  lane_seed_start="$((seed_base + lane * seeds_per_lane))"
  lane_seed_end="$((lane_seed_start + seeds_per_lane - 1))"
  seeds="${lane_seed_start}-${lane_seed_end}"
  label="${base_label}_${label_suffix}_lane${lane}_s${lane_seed_start}_${lane_seed_end}"
  out_dir="results/revision/claimtransfer_hidden_claimcards_a40_packed/${label}"
  setting="${label}|${function}|${samples}|${dimension}|${noise}|${update_grid}|${width}|${steps}|${seeds}|${anova_points}|${anova_background}"

  echo "[$(date -Is)] launch lane=${lane} label=${label} seeds=${seeds}"
  (
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
  ) &
  pids+=("$!")
done

fail=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    fail=1
  fi
done

if [[ "$fail" != "0" ]]; then
  echo "[$(date -Is)] one or more packed lanes failed" >&2
  exit 1
fi

echo "[$(date -Is)] done packed A40 row ${base_label}"
