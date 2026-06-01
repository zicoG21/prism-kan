#!/usr/bin/env bash
set -euo pipefail

# Low-CPU local GPU queue for formula-family evidence-transfer breadth.
#
# This intentionally avoids the large Great Lakes queues.  It runs a small
# seed-aligned workflow trace for formula families that are useful as breadth
# checks and positive controls, while keeping BLAS thread counts low so the
# laptop remains usable.

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
BASE="results/revision/local_gpu_formula_breadth_lowcpu/${STAMP}"
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
  local seeds="$3"
  local out_dir="$BASE/${label}"
  local setting="${label}|${function}|1024|100|0.00|0|16|90|${seeds}|24|24"

  mkdir -p "$out_dir"
  log "start ${label} function=${function} seeds=${seeds}"
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
    --prune-threshold 0.03 \
    --symbolic-smoke \
    --mse-threshold 0.05 \
    --device cuda \
    --out-dir "$out_dir" \
    --max-table-rows 8 \
    > "$out_dir/run.log" 2>&1
  local status=$?
  set -e
  log "exit ${label} status=${status}"
  if [[ "$status" -ne 0 ]]; then
    tail -80 "$out_dir/run.log" || true
  fi
}

log "local formula-breadth low-CPU queue started; base=${BASE}"
run_case "formula_trig_product_s356_363" "formula_trig_product" "356-363"
run_case "formula_log_product_s356_363" "formula_log_product" "356-363"
run_case "formula_sqrt_energy_s356_363" "formula_sqrt_energy" "356-363"
run_case "formula_exp_product_s364_371" "formula_exp_product" "364-371"
log "local formula-breadth low-CPU queue complete"
