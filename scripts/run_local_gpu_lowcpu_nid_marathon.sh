#!/usr/bin/env bash
set -euo pipefail

# Long local laptop-GPU queue with low CPU pressure.
#
# Intended use:
#   setsid env RUN_STAMP=$(date +%Y%m%d_%H%M%S) \
#     bash scripts/run_local_gpu_lowcpu_nid_marathon.sh > .../nohup.log 2>&1 < /dev/null &
#
# This is intentionally a finite long queue, not a timeout. It runs one Python
# process at a time, keeps BLAS/OpenMP thread counts low, and uses NID-only
# scoring so the UI remains usable.

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
BASE="results/revision/local_gpu_lowcpu_nid_marathon/${STAMP}"
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
  local seed_start="$7"
  local seed_end="$8"

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
    --seeds $(seq -s ' ' "$seed_start" "$seed_end") \
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

# Chunked batches keep the queue resumable and make progress easier to inspect.
# This is still a finite work queue rather than a timeout.
for batch_start in $(seq 1000 50 2999); do
  batch_end=$((batch_start + 49))
  if [[ "$batch_end" -gt 2999 ]]; then
    batch_end=2999
  fi
  suffix="s${batch_start}_${batch_end}"

  run_case "weak_c025_n512_h384d3_nidonly_${suffix}" "core_interaction_c025" 512 0.00 384 3 "$batch_start" "$batch_end"
  run_case "weak_c025_n1024_h384d3_nidonly_${suffix}" "core_interaction_c025" 1024 0.00 384 3 "$batch_start" "$batch_end"
  run_case "noise010_c025_n1024_h384d3_nidonly_${suffix}" "core_interaction_c025" 1024 0.10 384 3 "$batch_start" "$batch_end"
  run_case "strong_c1_n1024_h384d3_nidonly_${suffix}" "core_interaction_c1" 1024 0.00 384 3 "$batch_start" "$batch_end"
  run_case "nested_trig_h384d3_nidonly_${suffix}" "formula_nested_trig" 1024 0.00 384 3 "$batch_start" "$batch_end"
  run_case "rational_product_h384d3_nidonly_${suffix}" "formula_rational_product" 1024 0.00 384 3 "$batch_start" "$batch_end"
  run_case "three_way_product_h384d3_nidonly_${suffix}" "formula_three_way_product" 1024 0.00 384 3 "$batch_start" "$batch_end"
  run_case "exp_product_h384d3_nidonly_${suffix}" "formula_exp_product" 1024 0.00 384 3 "$batch_start" "$batch_end"
  run_case "trig_product_h384d3_nidonly_${suffix}" "formula_trig_product" 1024 0.00 384 3 "$batch_start" "$batch_end"
  run_case "mixed_sparse_h384d3_nidonly_${suffix}" "formula_mixed_sparse" 1024 0.00 384 3 "$batch_start" "$batch_end"
done

echo "[$(date -Is)] all local NID marathon jobs complete: $BASE" | tee -a "$BASE/progress.log"
