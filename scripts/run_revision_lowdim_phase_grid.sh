#!/usr/bin/env bash
set -euo pipefail

PY="${PYTHON:-/home/perzival/anaconda3/envs/prism/bin/python}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ROOT_OUT="results/revision/lowdim_phase_grid"
SEEDS=(810 811 812 813 814 815 816 817)

COMMON=(
  --device auto
  --test_samples 2048
  --noise 0
  --top_ms 4 6 10 20
  --methods feature_stability_var feature_edge_hybrid
  --probe_variable_points 512
  --pred_batch_size 4096
  --seeds "${SEEDS[@]}"
  --width_hidden 8
  --grid 5
  --lamb 0.001
  --probe_steps 35
)

run_phase() {
  local fn="$1"
  local d="$2"
  local n="$3"
  local label="$ROOT_OUT/phase/${fn}/d${d}/n${n}"
  "$PY" experiments/run_kan_probe_sensitivity.py \
    --out_dir "$label" \
    --function "$fn" \
    --dimension "$d" \
    --samples "$n" \
    "${COMMON[@]}"
}

run_width() {
  local d="$1"
  local width="$2"
  local label="$ROOT_OUT/width_check/core_interaction_c025/d${d}/n512/w${width}"
  "$PY" experiments/run_kan_probe_sensitivity.py \
    --out_dir "$label" \
    --function core_interaction_c025 \
    --dimension "$d" \
    --samples 512 \
    --width_hidden "$width" \
    --grid 5 \
    --lamb 0.001 \
    --probe_steps 35 \
    --device auto \
    --test_samples 2048 \
    --noise 0 \
    --top_ms 4 6 10 20 \
    --methods feature_stability_var feature_edge_hybrid \
    --probe_variable_points 512 \
    --pred_batch_size 4096 \
    --seeds "${SEEDS[@]}"
}

# Low-/moderate-dimensional phase grid. This is the paper-relevant supplement:
# interaction strength x nuisance dimension x sample size, under one fixed
# inspected pyKAN readout protocol.
for fn in core_interaction_c01 core_interaction_c025 core_interaction_c05; do
  for d in 20 50 100; do
    for n in 256 512 1024; do
      run_phase "$fn" "$d" "$n"
    done
  done
done

# Small configuration check at the critical weak-interaction sample size.
for d in 20 50 100; do
  for width in 8 16 32; do
    run_width "$d" "$width"
  done
done

"$PY" scripts/summarize_revision_lowdim_phase_grid.py
