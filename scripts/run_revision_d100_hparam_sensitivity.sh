#!/usr/bin/env bash
set -euo pipefail

PY="${PYTHON:-/home/perzival/anaconda3/envs/prism/bin/python}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

COMMON=(
  --device auto
  --function core_interaction_c025
  --dimension 100
  --test_samples 2048
  --noise 0
  --top_ms 4 6 10 20
  --methods feature_stability_var feature_edge_hybrid
  --probe_variable_points 512
  --pred_batch_size 4096
  --seeds 730 731 732 733 734 735 736 737 738 739
)

ROOT_OUT="results/revision/d100_c025_hparam_sensitivity"

run_one() {
  local n="$1"
  local label="$2"
  local width="$3"
  local grid="$4"
  local lamb="$5"
  local steps="$6"

  "$PY" experiments/run_kan_probe_sensitivity.py \
    --out_dir "$ROOT_OUT/n${n}/${label}" \
    --samples "$n" \
    --width_hidden "$width" \
    --grid "$grid" \
    --lamb "$lamb" \
    --probe_steps "$steps" \
    "${COMMON[@]}"
}

for n in 512 896 1024; do
  run_one "$n" "default_w8_g5_l1e-3_s35" 8 5 0.001 35
  run_one "$n" "wide_w16_g5_l1e-3_s35" 16 5 0.001 35
  run_one "$n" "wide_w32_g5_l1e-3_s35" 32 5 0.001 35
  run_one "$n" "coarse_w8_g3_l1e-3_s35" 8 3 0.001 35
  run_one "$n" "fine_w8_g10_l1e-3_s35" 8 10 0.001 35
  run_one "$n" "lowreg_w8_g5_l1e-4_s35" 8 5 0.0001 35
  run_one "$n" "highreg_w8_g5_l1e-2_s35" 8 5 0.01 35
  run_one "$n" "longer_w8_g5_l1e-3_s100" 8 5 0.001 100
done

"$PY" scripts/summarize_revision_d100_hparam.py
