#!/usr/bin/env bash
set -euo pipefail

# Low-CPU local GPU follow-up for high-value seed-aligned evidence-transfer
# traces.  These rows use a disjoint seed block from the earlier local queues
# and focus on reviewer-visible gaps: formula breadth and same-seed workflow
# provenance.  This is useful local work while Great Lakes handles the larger
# grids.

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
BASE="results/revision/local_gpu_anchor_trace_followup/${STAMP}"
mkdir -p "$BASE"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

log() {
  echo "[$(date -Is)] $*" | tee -a "$BASE/progress.log"
}

run_case() {
  local label="$1"
  local function="$2"
  local samples="$3"
  local dimension="$4"
  local noise="$5"
  local update_grid="$6"
  local width_hidden="$7"
  local steps="$8"
  local seeds="$9"
  local prune_threshold="${10}"
  local out_dir="$BASE/${label}"
  local setting="${label}|${function}|${samples}|${dimension}|${noise}|${update_grid}|${width_hidden}|${steps}|${seeds}|24|24"

  mkdir -p "$out_dir"
  log "start ${label} function=${function} seeds=${seeds} prune=${prune_threshold}"
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
  log "exit ${label} status=${status}"
  if [[ "$status" -ne 0 ]]; then
    tail -80 "$out_dir/run.log" || true
  fi
}

log "local anchor trace follow-up started; base=${BASE}"

# Disjoint seed block after previous local stage queues:
# 300-307, 308-315, 316-331, 332-339, 340-355, 356-363, 364-371.
SEEDS="372-387"

run_case "formula_mixed_sparse_s372_387" "formula_mixed_sparse" 1024 100 0.00 0 16 90 "$SEEDS" 0.03
run_case "formula_division_mixed_s372_387" "formula_division_mixed" 1024 100 0.00 0 16 90 "$SEEDS" 0.03
run_case "formula_rational_product_s372_387" "formula_rational_product" 1024 100 0.00 0 16 90 "$SEEDS" 0.03
run_case "formula_bilinear_s372_387" "formula_bilinear" 1024 100 0.00 0 16 90 "$SEEDS" 0.03

# One positive-control KAN row helps anchor the seed-aligned table: when width
# is enough and the pressure-test is clean, all workflow objects should agree
# more often than in grid/noise boundary rows.
run_case "core_clean_w32_n768_s372_387" "core_interaction_c025" 768 100 0.00 0 32 75 "$SEEDS" 0.03

log "local anchor trace follow-up complete; base=${BASE}"
