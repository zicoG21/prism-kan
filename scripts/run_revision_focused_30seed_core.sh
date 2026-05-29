#!/usr/bin/env bash
set -euo pipefail

# Focused reviewer-response run.
#
# This is intentionally narrower than the 12h stretch pack.  It targets the
# core statistical objections reviewers kept raising:
#   1. 10-seed boundary rows should be checked at 30 seeds;
#   2. the non-monotone c=0.10,d=20 cell should be explained by rank/margin
#      diagnostics, not only by success counts;
#   3. grid-update and noise rows should be reported under the same seed budget.
#
# It writes resumable per-setting outputs under:
#   results/revision/focused_30seed_core/

PY="${PYTHON:-/home/perzival/anaconda3/envs/prism/bin/python}"
DEVICE="${FOCUSED_DEVICE:-cpu}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT_ROOT="results/revision/focused_30seed_core"
mkdir -p "$OUT_ROOT"

date -Is > "$OUT_ROOT/STARTED_AT"
echo "$$" > "$OUT_ROOT/PID"

log() {
  echo "[$(date -Is)] $*" | tee -a "$OUT_ROOT/progress.log"
}

run_cmd() {
  local label="$1"
  shift
  log "BEGIN ${label}"
  set +e
  "$@" >> "$OUT_ROOT/${label}.log" 2>&1
  local status=$?
  set -e
  if [[ "$status" -ne 0 ]]; then
    log "FAILED ${label} status=${status}"
    printf '%s,%s,%s\n' "$(date -Is)" "$label" "$status" >> "$OUT_ROOT/FAILED_STAGES.csv"
    return 0
  fi
  log "END ${label}"
}

SEEDS=()
for s in $(seq 1200 1229); do
  SEEDS+=("$s")
done

run_kan_sens() {
  local label="$1"
  local fn="$2"
  local n="$3"
  local d="$4"
  local width="$5"
  local steps="$6"
  local noise="${7:-0}"
  local update_grid="${8:-0}"

  local common_args=(
    experiments/run_kan_probe_sensitivity.py
    --out_dir "$OUT_ROOT/${label}"
    --function "$fn"
    --samples "$n"
    --dimension "$d"
    --test_samples 2048
    --noise "$noise"
    --methods feature_stability_var feature_edge_hybrid
    --top_ms 4 6 10 20
    --width_hidden "$width"
    --grid 5
    --k 3
    --lamb 0.001
    --probe_steps "$steps"
    --probe_variable_points 512
    --pred_batch_size 4096
    --device "$DEVICE"
  )
  if [[ "$update_grid" == "1" ]]; then
    common_args+=(--update_grid --grid_update_num 5)
  fi
  for seed in "${SEEDS[@]}"; do
    set +e
    "$PY" "${common_args[@]}" --seeds "$seed"
    local status=$?
    set -e
    if [[ "$status" -ne 0 ]]; then
      printf '%s,%s,seed%s,%s\n' "$(date -Is)" "$label" "$seed" "$status" >> "$OUT_ROOT/FAILED_SEEDS.csv"
    fi
  done
  "$PY" "${common_args[@]}" --seeds "${SEEDS[@]}" --summarize_existing_only
}

# Core c=0.25,d=100 boundary rows from the main paper.
for n in 512 1024; do
  run_cmd "core_c025_d100_clean_w8_n${n}" run_kan_sens "core_c025_d100_clean_w8_n${n}" core_interaction_c025 "$n" 100 8 35 0 0
  run_cmd "core_c025_d100_clean_w16_n${n}" run_kan_sens "core_c025_d100_clean_w16_n${n}" core_interaction_c025 "$n" 100 16 75 0 0
  run_cmd "core_c025_d100_noise010_w16_n${n}" run_kan_sens "core_c025_d100_noise010_w16_n${n}" core_interaction_c025 "$n" 100 16 75 0.10 0
  run_cmd "core_c025_d100_gridupdate_w16_n${n}" run_kan_sens "core_c025_d100_gridupdate_w16_n${n}" core_interaction_c025 "$n" 100 16 75 0 1
done

# Non-monotone weak-interaction row, with the same 30-seed budget.
for n in 256 512 1024; do
  run_cmd "nonmonotone_c01_d20_w8_n${n}" run_kan_sens "nonmonotone_c01_d20_w8_n${n}" core_interaction_c01 "$n" 20 8 35 0 0
done

run_cmd "summarize_focused_30seed_core" "$PY" scripts/summarize_revision_focused_30seed_core.py --root "$OUT_ROOT"
