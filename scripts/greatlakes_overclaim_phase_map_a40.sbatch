#!/usr/bin/env bash
#SBATCH --job-name=kan-overphase
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

# Overclaim-risk phase map for pyKAN stage records.
#
# Purpose:
#   Convert "overclaim risk exists" into a stronger empirical statement:
#   unsupported structural transfers occupy predictable regimes in signal
#   strength, sample size, noise, covariate proxies, and grid-update protocol.
#
# Submit from Great Lakes:
#
#   sbatch --account=jaabell0 --array=0-35 \
#     --export=ALL,PYTHON_BIN=$PWD/.venv/bin/python,SEED_BASE=7000,SEED_COUNT=8,LABEL_SUFFIX=s7000_7007 \
#     scripts/greatlakes_overclaim_phase_map_a40.sbatch
#
# The script sets no array throttle.  Let account/partition GRES limits decide
# concurrency.  Override --partition=gpu at submission time for V100s.

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
seed_base="${SEED_BASE:-7000}"
seed_count="${SEED_COUNT:-8}"
seed_end="$((seed_base + seed_count - 1))"
seeds="${seed_base}-${seed_end}"
label_suffix="${LABEL_SUFFIX:-s${seed_base}_${seed_end}}"

ROWS=()
STRENGTHS=(
  "c010|core_interaction_c01"
  "c025|core_interaction_c025"
  "c050|core_interaction_c05"
)
SAMPLES=(512 1024 2048)
CONDITIONS=(
  "clean|0.00|0|0.00|0"
  "noise010|0.10|0|0.00|0"
  "proxy_rho09|0.05|0|0.90|8"
  "gridupdate|0.00|1|0.00|0"
)

for strength in "${STRENGTHS[@]}"; do
  IFS='|' read -r strength_label function <<< "$strength"
  for samples in "${SAMPLES[@]}"; do
    for condition in "${CONDITIONS[@]}"; do
      IFS='|' read -r condition_label noise update_grid rho proxies <<< "$condition"
      ROWS+=("${strength_label}|${function}|${samples}|${condition_label}|${noise}|${update_grid}|${rho}|${proxies}")
    done
  done
done

task_id="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
if (( task_id < 0 || task_id >= ${#ROWS[@]} )); then
  echo "Task id ${task_id} outside row range 0-$(( ${#ROWS[@]} - 1 ))" >&2
  exit 2
fi

row="${ROWS[$task_id]}"
IFS='|' read -r strength_label function samples condition_label noise update_grid rho proxies <<< "$row"

base_label="phase_${strength_label}_${condition_label}_n${samples}"
label="${base_label}__${label_suffix}"
out_dir="results/revision/overclaim_phase_map/${label}"
setting="${label}|${function}|${samples}|100|${noise}|${update_grid}|16|90|${seeds}|24|24"

echo "[$(date -Is)] node=${SLURM_JOB_NODELIST:-unknown-node}"
echo "[$(date -Is)] overclaim phase task=${task_id}/${#ROWS[@]} label=${label}"
echo "[$(date -Is)] function=${function} samples=${samples} noise=${noise} update_grid=${update_grid} rho=${rho} proxies=${proxies}"
echo "[$(date -Is)] seeds=${seeds} batch_size=${batch_size} pair_chunk_size=${pair_chunk_size} threads=${OMP_NUM_THREADS}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

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

"$PY" scripts/summarize_overclaim_phase_map.py \
  --root results/revision/overclaim_phase_map \
  --out-prefix results/revision/overclaim_phase_map/overclaim_phase_map_summary || true

echo "[$(date -Is)] done overclaim phase ${label}"
