#!/usr/bin/env bash
set -euo pipefail

# High-value local GPU queue, ordered by current reviewer value.
#
# Policy:
# - Local machine runs only low-CPU, manuscript-relevant tasks.
# - Duration is a soft target, not a hard cutoff: after each case completes,
#   stop only if the queue has already exceeded TARGET_HOURS.
# - Do not duplicate currently running local work; wait for the lightweight
#   follow-up queue if it is still active.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DEFAULT_PY="/home/perzival/anaconda3/envs/prism/bin/python"
if [[ -n "${PYTHON_BIN:-}" ]]; then
  PY="$PYTHON_BIN"
elif [[ -x "$DEFAULT_PY" ]]; then
  PY="$DEFAULT_PY"
else
  PY="python"
fi

STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
BASE="results/revision/local_gpu_highvalue_10h_queue/${STAMP}"
mkdir -p "$BASE"

TARGET_HOURS="${TARGET_HOURS:-10}"
TARGET_SECONDS=$(( TARGET_HOURS * 3600 ))
START_TS="$(date +%s)"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

log() {
  echo "[$(date -Is)] $*" | tee -a "$BASE/progress.log"
}

elapsed_seconds() {
  echo $(( "$(date +%s)" - START_TS ))
}

wait_for_current_lowimpact_queue() {
  local pids
  pids="$(pgrep -f 'scripts/run_local_gpu_stage_record_followup_lowimpact.sh' || true)"
  if [[ -n "$pids" ]]; then
    log "waiting for existing low-impact follow-up queue: ${pids//$'\n'/ }"
  fi
  while pgrep -f 'scripts/run_local_gpu_stage_record_followup_lowimpact.sh' >/dev/null; do
    sleep 180
  done
}

run_stage_case() {
  local phase="$1"
  local label="$2"
  local setting="$3"
  local prune_threshold="$4"
  local out_dir="$BASE/${phase}/${label}"
  mkdir -p "$out_dir"

  local elapsed
  elapsed="$(elapsed_seconds)"
  if (( elapsed >= TARGET_SECONDS )); then
    log "soft target reached before ${phase}/${label}; elapsed=${elapsed}s target=${TARGET_SECONDS}s"
    return 2
  fi

  log "start ${phase}/${label} prune_threshold=${prune_threshold} setting=${setting}"
  set +e
  nice -n 15 "$PY" -u experiments/run_seed_aligned_stage_records.py \
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
    --prune-threshold "$prune_threshold" \
    --symbolic-smoke \
    --mse-threshold 0.05 \
    --device cuda \
    --out-dir "$out_dir" \
    --max-table-rows 16 \
    > "$out_dir/run.log" 2>&1
  local status=$?
  set -e

  log "exit ${phase}/${label} status=${status} elapsed=$(elapsed_seconds)s"
  if [[ "$status" -ne 0 ]]; then
    tail -80 "$out_dir/run.log" || true
  fi
  return 0
}

run_phase_rows() {
  local phase="$1"
  local threshold="$2"
  shift 2
  local row label
  for row in "$@"; do
    label="${row%%|*}"
    if ! run_stage_case "$phase" "$label" "$row" "$threshold"; then
      return 0
    fi
  done
}

wait_for_current_lowimpact_queue

log "high-value local queue started; soft_target_hours=${TARGET_HOURS}; base=${BASE}"

# Phase 1: seed-aligned CI expansion for the rows now used in the main table.
# Value: turns the new same-seed main-text table from a single illustrative
# block into a more stable paired provenance estimate.
PHASE1_ROWS=(
  "core_clean_w16_n512_s316_331|core_interaction_c025|512|100|0.00|0|16|75|316-331|24|24"
  "core_clean_w32_n768_s316_331|core_interaction_c025|768|100|0.00|0|32|75|316-331|24|24"
  "core_grid_w16_n1024_s316_331|core_interaction_c025|1024|100|0.00|1|16|75|316-331|24|24"
  "core_noise010_w16_n1024_s316_331|core_interaction_c025|1024|100|0.10|0|16|75|316-331|24|24"
  "core_clean_w16_n1024_s316_331|core_interaction_c025|1024|100|0.00|0|16|75|316-331|24|24"
  "core_grid_w16_n512_s316_331|core_interaction_c025|512|100|0.00|1|16|75|316-331|24|24"
  "core_noise010_w16_n512_s316_331|core_interaction_c025|512|100|0.10|0|16|75|316-331|24|24"
)
run_phase_rows "01_core_seed_aligned_ci" "0.03" "${PHASE1_ROWS[@]}"

# Phase 2: formula-family same-seed traces for breadth beyond the core pressure
# test. Value: addresses the "single synthetic task" critique without turning
# the local machine into a broad benchmark runner.
PHASE2_ROWS=(
  "formula_mixed_sparse_s316_331|formula_mixed_sparse|1024|100|0.00|0|16|90|316-331|24|24"
  "formula_division_mixed_s316_331|formula_division_mixed|1024|100|0.00|0|16|90|316-331|24|24"
  "formula_rational_product_s316_331|formula_rational_product|1024|100|0.00|0|16|90|316-331|24|24"
  "formula_bilinear_s316_331|formula_bilinear|1024|100|0.00|0|16|90|316-331|24|24"
  "formula_weak_centered_s316_331|formula_weak_centered|1024|100|0.00|0|16|90|316-331|24|24"
  "formula_exp_product_s316_331|formula_exp_product|1024|100|0.00|0|16|90|316-331|24|24"
)
run_phase_rows "02_formula_seed_aligned_breadth" "0.03" "${PHASE2_ROWS[@]}"

# Phase 3: pruning-threshold sensitivity on the same claim rows. Value:
# strengthens the claim-decision/pruning-provenance section without requiring a
# full symbolic benchmark.
PHASE3_ROWS=(
  "core_clean_w32_n768_prune010_s332_339|core_interaction_c025|768|100|0.00|0|32|75|332-339|24|24"
  "core_grid_w16_n1024_prune010_s332_339|core_interaction_c025|1024|100|0.00|1|16|75|332-339|24|24"
  "formula_mixed_sparse_prune010_s332_339|formula_mixed_sparse|1024|100|0.00|0|16|90|332-339|24|24"
  "formula_division_mixed_prune010_s332_339|formula_division_mixed|1024|100|0.00|0|16|90|332-339|24|24"
  "formula_rational_product_prune010_s332_339|formula_rational_product|1024|100|0.00|0|16|90|332-339|24|24"
)
run_phase_rows "03_prune_threshold_sensitivity" "0.10" "${PHASE3_ROWS[@]}"

# Phase 4: extra boundary seeds only if time remains. Value: paired
# uncertainty for the hardest n=512 rows.
PHASE4_ROWS=(
  "core_clean_w16_n512_s340_355|core_interaction_c025|512|100|0.00|0|16|75|340-355|24|24"
  "core_grid_w16_n512_s340_355|core_interaction_c025|512|100|0.00|1|16|75|340-355|24|24"
  "core_noise010_w16_n512_s340_355|core_interaction_c025|512|100|0.10|0|16|75|340-355|24|24"
)
run_phase_rows "04_extra_boundary_seeds" "0.03" "${PHASE4_ROWS[@]}"

log "high-value local queue complete; elapsed=$(elapsed_seconds)s; base=${BASE}"
