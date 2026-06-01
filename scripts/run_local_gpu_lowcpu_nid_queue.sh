#!/usr/bin/env bash
set -euo pipefail

# Local laptop GPU queue with low CPU pressure.
#
# Design:
#   - one Python process at a time;
#   - no Hessian scorer, only NID weight-based scoring;
#   - low thread counts for BLAS/OpenMP;
#   - no timeout: runtime is controlled by finite case list and epochs.

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
BASE="results/revision/local_gpu_lowcpu_nid_queue/${STAMP}"
mkdir -p "$BASE"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

run_case() {
  local label="$1"
  local function="$2"
  local samples="$3"
  local noise="$4"
  local hidden="$5"
  local depth="$6"
  local seeds="$7"

  local out_dir="$BASE/${label}"
  mkdir -p "$out_dir"
  echo "[$(date -Is)] start ${label}" | tee -a "$BASE/progress.log"
  set +e
  nice -n 10 "$PY" -u experiments/run_nid_interaction_baseline.py \
    --function "$function" \
    --samples "$samples" \
    --test_samples 2048 \
    --dimension 100 \
    --noise "$noise" \
    --seeds $seeds \
    --methods nid \
    --hidden "$hidden" \
    --depth "$depth" \
    --epochs 2200 \
    --patience 350 \
    --eval_every 25 \
    --batch_size 256 \
    --lr 0.001 \
    --weight_decay 1e-5 \
    --device cuda \
    --out_dir "$out_dir" \
    > "$out_dir/run.log" 2>&1
  local status=$?
  set -e
  echo "[$(date -Is)] exit ${label} status=${status}" | tee -a "$BASE/progress.log"
  if [[ "$status" -ne 0 ]]; then
    tail -80 "$out_dir/run.log" || true
    return "$status"
  fi
  echo "[$(date -Is)] done ${label}" | tee -a "$BASE/progress.log"
}

# Different seeds/width than the Great Lakes NID rows; NID-only keeps this
# local queue light on CPU.
SEEDS="$(seq -s ' ' 60 89)"
run_case "weak_c025_n512_h384d3_nidonly" "core_interaction_c025" 512 0.00 384 3 "$SEEDS"
run_case "weak_c025_n1024_h384d3_nidonly" "core_interaction_c025" 1024 0.00 384 3 "$SEEDS"
run_case "noise010_c025_n1024_h384d3_nidonly" "core_interaction_c025" 1024 0.10 384 3 "$SEEDS"
run_case "strong_c1_n1024_h384d3_nidonly" "core_interaction_c1" 1024 0.00 384 3 "$SEEDS"
run_case "nested_trig_h384d3_nidonly" "formula_nested_trig" 1024 0.00 384 3 "$SEEDS"
run_case "rational_product_h384d3_nidonly" "formula_rational_product" 1024 0.00 384 3 "$SEEDS"
run_case "three_way_product_h384d3_nidonly" "formula_three_way_product" 1024 0.00 384 3 "$SEEDS"

echo "[$(date -Is)] all local low-CPU NID jobs complete: $BASE" | tee -a "$BASE/progress.log"
