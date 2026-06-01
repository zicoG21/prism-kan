#!/usr/bin/env bash
set -euo pipefail

# Local high-value queue for reviewer-facing seed-aligned stage records.
#
# This queue is intentionally not a time filler.  Each row trains/inspects the
# same-seed KAN workflow and records the stages reviewers keep asking for:
# prediction -> full-model pair reliance -> exposed readout endpoints ->
# support refit -> prune/symbolic provenance.

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
BASE="results/revision/local_gpu_highvalue_stage_queue/${STAMP}"
mkdir -p "$BASE"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

run_stage_case() {
  local label="$1"
  local setting="$2"
  local out_dir="$BASE/$label"
  mkdir -p "$out_dir"

  echo "[$(date -Is)] start ${label}" | tee -a "$BASE/progress.log"
  set +e
  nice -n 10 "$PY" -u experiments/run_seed_aligned_stage_records.py \
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
    --prune-threshold 0.03 \
    --symbolic-smoke \
    --mse-threshold 0.05 \
    --device cuda \
    --out-dir "$out_dir" \
    --max-table-rows 16 \
    > "$out_dir/run.log" 2>&1
  local status=$?
  set -e
  echo "[$(date -Is)] exit ${label} status=${status}" | tee -a "$BASE/progress.log"
  if [[ "$status" -ne 0 ]]; then
    tail -80 "$out_dir/run.log" || true
  fi
}

# label|function|samples|dimension|noise|update_grid|width|steps|seeds|anova_points|anova_background
CASES=(
  "core_clean_w16_n512_s300_307|core_interaction_c025|512|100|0.00|0|16|75|300-307|24|24"
  "core_clean_w16_n1024_s300_307|core_interaction_c025|1024|100|0.00|0|16|75|300-307|24|24"
  "core_grid_w16_n512_s300_307|core_interaction_c025|512|100|0.00|1|16|75|300-307|24|24"
  "core_grid_w16_n1024_s300_307|core_interaction_c025|1024|100|0.00|1|16|75|300-307|24|24"
  "core_noise010_w16_n512_s300_307|core_interaction_c025|512|100|0.10|0|16|75|300-307|24|24"
  "core_noise010_w16_n1024_s300_307|core_interaction_c025|1024|100|0.10|0|16|75|300-307|24|24"
  "core_clean_w32_n768_s300_307|core_interaction_c025|768|100|0.00|0|32|75|300-307|24|24"
  "core_grid_w32_n1024_s300_307|core_interaction_c025|1024|100|0.00|1|32|75|300-307|24|24"
  "formula_bilinear_s300_307|formula_bilinear|1024|100|0.00|0|16|90|300-307|24|24"
  "formula_weak_centered_s300_307|formula_weak_centered|1024|100|0.00|0|16|90|300-307|24|24"
  "formula_nested_trig_s300_307|formula_nested_trig|1024|100|0.00|0|16|90|300-307|24|24"
  "formula_rational_product_s300_307|formula_rational_product|1024|100|0.00|0|16|90|300-307|24|24"
  "formula_three_way_s300_307|formula_three_way_product|1024|100|0.00|0|16|90|300-307|24|24"
  "formula_mixed_sparse_s300_307|formula_mixed_sparse|1024|100|0.00|0|16|90|300-307|24|24"
  "formula_division_mixed_s300_307|formula_division_mixed|1024|100|0.00|0|16|90|300-307|24|24"
)

for row in "${CASES[@]}"; do
  label="${row%%|*}"
  run_stage_case "$label" "$row"
done

echo "[$(date -Is)] all local high-value stage jobs complete: $BASE" | tee -a "$BASE/progress.log"
