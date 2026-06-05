#!/usr/bin/env bash
#SBATCH --job-name=kan-ct-repair
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

# Audit-guided repair rows for ClaimTransfer-Bench.
#
# Purpose:
#   Test whether the audit can do more than diagnose overclaims.  Each card is
#   run under a baseline protocol and simple repair protocols that target a
#   specific broken handoff:
#     - larger support budget for fitted-pair -> readout/refit failures;
#     - lower prune threshold for pruning/symbolic endpoint failures;
#     - combined budget + prune guard.
#
# Submit from Great Lakes:
#
#   sbatch --account=jaabell0 --array=0-24 \
#     --export=ALL,PYTHON_BIN=$PWD/.venv/bin/python,SEED_BASE=7600,SEED_COUNT=8,LABEL_SUFFIX=s7600_7607 \
#     scripts/greatlakes_audit_guided_repair_a40.sbatch
#
# No array throttle is set.  Override --partition=gpu at submission time for
# V100s if that queue is idle.

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
seed_base="${SEED_BASE:-7600}"
seed_count="${SEED_COUNT:-8}"
seed_end="$((seed_base + seed_count - 1))"
seeds="${seed_base}-${seed_end}"
label_suffix="${LABEL_SUFFIX:-s${seed_base}_${seed_end}}"

CARDS=(
  "weak_clean|core_interaction_c025|1024|100|0.00|0|0.00|0"
  "weak_noise010|core_interaction_c025|1024|100|0.10|0|0.00|0"
  "weak_proxy_rho09|core_interaction_c025|1024|100|0.05|0|0.90|8"
  "weak_gridupdate|core_interaction_c025|1024|100|0.00|1|0.00|0"
  "mixed_sparse_clean|formula_mixed_sparse|1024|100|0.00|0|0.00|0"
)

REPAIRS=(
  "baseline|4|0.03|90"
  "support_budget10|10|0.03|90"
  "support_budget20|20|0.03|90"
  "prune_guard|4|0.01|90"
  "combined_budget10_prune_guard|10|0.01|120"
)

ROWS=()
for card in "${CARDS[@]}"; do
  for repair in "${REPAIRS[@]}"; do
    ROWS+=("${card}|${repair}")
  done
done

task_id="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
if (( task_id < 0 || task_id >= ${#ROWS[@]} )); then
  echo "Task id ${task_id} outside row range 0-$(( ${#ROWS[@]} - 1 ))" >&2
  exit 2
fi

row="${ROWS[$task_id]}"
IFS='|' read -r card_label function samples dimension noise update_grid rho proxies repair_label top_m prune_threshold refit_steps <<< "$row"

base_label="repair_${repair_label}__${card_label}"
label="${base_label}__${label_suffix}"
out_dir="results/revision/audit_guided_repair_claimcards/${label}"
setting="${label}|${function}|${samples}|${dimension}|${noise}|${update_grid}|16|90|${seeds}|24|24"

echo "[$(date -Is)] node=${SLURM_JOB_NODELIST:-unknown-node}"
echo "[$(date -Is)] repair task=${task_id}/${#ROWS[@]} label=${label}"
echo "[$(date -Is)] card=${card_label} function=${function} repair=${repair_label} top_m=${top_m} prune=${prune_threshold} refit_steps=${refit_steps}"
echo "[$(date -Is)] seeds=${seeds} batch_size=${batch_size} pair_chunk_size=${pair_chunk_size} threads=${OMP_NUM_THREADS}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

"$PY" experiments/run_seed_aligned_stage_records.py \
  --settings "$setting" \
  --test-samples 2048 \
  --nuisance-correlation "$rho" \
  --n-correlated-proxies "$proxies" \
  --top-m "$top_m" \
  --grid 5 \
  --k 3 \
  --lamb 0.001 \
  --grid-update-num 5 \
  --batch -1 \
  --batch-size "$batch_size" \
  --pair-chunk-size "$pair_chunk_size" \
  --refit-width-hidden 16 \
  --refit-steps "$refit_steps" \
  --refit-anova-points 32 \
  --refit-anova-background 32 \
  --prune-workflow prune_input \
  --prune-threshold "$prune_threshold" \
  --symbolic-smoke \
  --mse-threshold 0.05 \
  --device cuda \
  --out-dir "$out_dir" \
  --max-table-rows 10

"$PY" scripts/summarize_audit_guided_repair.py \
  --root results/revision/audit_guided_repair_claimcards \
  --out-prefix results/revision/audit_guided_repair_claimcards/audit_guided_repair_summary || true

echo "[$(date -Is)] done audit-guided repair ${label}"
