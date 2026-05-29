#!/usr/bin/env bash
set -euo pipefail

# Repair run for the mini-suite baseline stretch from the reviewer-boundary
# overnight pack. The original stretch used legacy function aliases; this uses
# the canonical formula_* names accepted by src.data.

PY="${PYTHON:-/home/perzival/anaconda3/envs/prism/bin/python}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT_ROOT="${OUT_ROOT:-results/revision/boundary_overnight_12h}"
mkdir -p "$OUT_ROOT"

log() {
  echo "[$(date -Is)] $*" | tee -a "$OUT_ROOT/minisuite_baseline_stretch_progress.log"
}

run_cmd() {
  local label="$1"
  shift
  log "BEGIN ${label}"
  set +e
  "$@" 2>&1 | tee -a "$OUT_ROOT/${label}.log"
  local status=${PIPESTATUS[0]}
  set -e
  if [[ "$status" -ne 0 ]]; then
    log "FAILED ${label} status=${status}"
    printf '%s,%s,%s\n' "$(date -Is)" "$label" "$status" >> "$OUT_ROOT/FAILED_MINISUITE_BASELINE_STAGES.csv"
    return 0
  fi
  log "END ${label}"
}

SEEDS12=(930 931 932 933 934 935 936 937 938 939 940 941)
FUNCTIONS=(
  formula_trig_product
  formula_rational_product
  formula_three_way_product
  formula_mixed_sparse
  formula_nested_trig
)

run_lasso() {
  local label="$1"
  local fn="$2"
  "$PY" experiments/run_sparse_interaction_lasso_baseline.py \
    --function "$fn" \
    --samples 512 1024 \
    --test_samples 2048 \
    --dimension 100 \
    --noise 0.00 \
    --nuisance_correlation 0 \
    --n_correlated_proxies 0 \
    --top_m 4 \
    --seeds "${SEEDS12[@]}" \
    --cv 5 \
    --max_iter 10000 \
    --out_dir "$OUT_ROOT/baselines/lasso_${label}"
}

run_hsic() {
  local label="$1"
  local fn="$2"
  "$PY" experiments/run_residual_hsic_pair_screen.py \
    --functions "$fn" \
    --samples 512 1024 \
    --test_samples 2048 \
    --dimension 100 \
    --noise 0.00 \
    --nuisance_correlation 0 \
    --n_correlated_proxies 0 \
    --crossfit_folds 5 \
    --rff_dim 64 \
    --y_rff_dim 32 \
    --top_pairs_for_support 1 \
    --seeds "${SEEDS12[@]}" \
    --out_dir "$OUT_ROOT/baselines/hsic_${label}"
}

date -Is > "$OUT_ROOT/MINISUITE_BASELINE_STRETCH_STARTED_AT"
for fn in "${FUNCTIONS[@]}"; do
  run_cmd "repair_lasso_${fn}" run_lasso "stretch_${fn}" "$fn"
  run_cmd "repair_hsic_${fn}" run_hsic "stretch_${fn}" "$fn"
done

run_cmd "repair_summary" "$PY" scripts/summarize_revision_boundary_overnight.py --root "$OUT_ROOT"
date -Is > "$OUT_ROOT/MINISUITE_BASELINE_STRETCH_FINISHED_AT"
log "DONE mini-suite baseline stretch repair"
