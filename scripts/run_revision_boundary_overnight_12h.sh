#!/usr/bin/env bash
set -euo pipefail

# Reviewer-boundary overnight pack.
#
# Goal: fill the most attackable boundary conditions without changing the
# paper's scope into a full benchmark:
#   1. critical-regime pyKAN capacity/training budget sensitivity;
#   2. non-oracle support-budget/rank behavior under noise/correlation;
#   3. semi-synthetic real-covariate checks with injected interactions;
#   4. non-KAN interaction baselines under the same noisy/correlated slices.
#   5. stretch queue if earlier stages finish before the timeout.
#
# The script is resumable at the level supported by the underlying runners.
# Run it in the background with:
#   nohup env PYTHON=/home/perzival/anaconda3/envs/prism/bin/python \
#     bash scripts/run_revision_boundary_overnight_12h.sh \
#     > results/revision/boundary_overnight_12h/master.log 2>&1 &

PY="${PYTHON:-/home/perzival/anaconda3/envs/prism/bin/python}"
RUN_DEVICE="${RUN_DEVICE:-auto}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT_ROOT="results/revision/boundary_overnight_12h"
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
  "$@" 2>&1 | tee -a "$OUT_ROOT/${label}.log"
  local status=${PIPESTATUS[0]}
  set -e
  if [[ "$status" -ne 0 ]]; then
    log "FAILED ${label} status=${status}"
    printf '%s,%s,%s\n' "$(date -Is)" "$label" "$status" >> "$OUT_ROOT/FAILED_STAGES.csv"
    return 0
  fi
  log "END ${label}"
}

SEEDS8=(910 911 912 913 914 915 916 917)
SEEDS10=(920 921 922 923 924 925 926 927 928 929)
SEEDS12=(930 931 932 933 934 935 936 937 938 939 940 941)

########################################
# 1. Critical regime: width x steps x sample size.
########################################

run_kan_sens() {
  local label="$1"
  local fn="$2"
  local n="$3"
  local d="$4"
  local width="$5"
  local grid="$6"
  local lamb="$7"
  local steps="$8"
  local noise="${9:-0}"
  local rho="${10:-0}"
  local proxies="${11:-0}"

  "$PY" experiments/run_kan_probe_sensitivity.py \
    --out_dir "$OUT_ROOT/kan_sensitivity/${label}" \
    --function "$fn" \
    --samples "$n" \
    --dimension "$d" \
    --test_samples 2048 \
    --noise "$noise" \
    --nuisance_correlation "$rho" \
    --n_correlated_proxies "$proxies" \
    --seeds "${SEEDS10[@]}" \
    --methods feature_stability_var feature_edge_hybrid \
    --top_ms 4 6 10 20 \
    --width_hidden "$width" \
    --grid "$grid" \
    --k 3 \
    --lamb "$lamb" \
    --probe_steps "$steps" \
    --probe_variable_points 512 \
    --pred_batch_size 4096 \
    --device "$RUN_DEVICE"
}

for n in 512 640 768 896 1024; do
  run_cmd "kan_default_n${n}" run_kan_sens "default_c025_d100_n${n}" core_interaction_c025 "$n" 100 8 5 0.001 35
  run_cmd "kan_width16_n${n}" run_kan_sens "width16_c025_d100_n${n}" core_interaction_c025 "$n" 100 16 5 0.001 35
  run_cmd "kan_width32_n${n}" run_kan_sens "width32_c025_d100_n${n}" core_interaction_c025 "$n" 100 32 5 0.001 35
done

for steps in 35 75 150; do
  run_cmd "kan_w8_steps${steps}_n512" run_kan_sens "w8_steps${steps}_c025_d100_n512" core_interaction_c025 512 100 8 5 0.001 "$steps"
  run_cmd "kan_w16_steps${steps}_n512" run_kan_sens "w16_steps${steps}_c025_d100_n512" core_interaction_c025 512 100 16 5 0.001 "$steps"
done

########################################
# 2. Noise/correlation boundary under synthetic covariates.
########################################

for n in 512 1024; do
  for noise in 0.00 0.05 0.10; do
    run_cmd "kan_noise${noise}_n${n}" run_kan_sens "noise${noise}_rho0_c025_d100_n${n}" core_interaction_c025 "$n" 100 16 5 0.001 75 "$noise" 0 0
  done
  for rho in 0.5 0.9; do
    run_cmd "kan_rho${rho}_n${n}" run_kan_sens "rho${rho}_prox8_c025_d100_n${n}" core_interaction_c025 "$n" 100 16 5 0.001 75 0.05 "$rho" 8
  done
done

########################################
# 3. Semi-synthetic covariates with injected interaction and noise.
########################################

run_semisynth() {
  local label="$1"
  local noise="$2"
  "$PY" experiments/run_semisynthetic_covariate_audit.py \
    --out-dir "$OUT_ROOT/semisynthetic/${label}" \
    --datasets diabetes breast_cancer \
    --coefficients 0.10 0.25 0.50 \
    --samples 128 256 384 \
    --test-samples 128 \
    --outer-seeds "${SEEDS10[@]}" \
    --noise "$noise" \
    --R 12 \
    --top-m 4 \
    --methods feature_stability_var feature_edge_hybrid \
    --width-hidden 8 \
    --grid 5 \
    --k 3 \
    --lamb 0.001 \
    --probe-steps 35 \
    --pred-batch-size 4096 \
    --device "$RUN_DEVICE"
}

run_cmd "semisynth_noise0" run_semisynth "noise0" 0.00
run_cmd "semisynth_noise005" run_semisynth "noise005" 0.05
run_cmd "semisynth_noise010" run_semisynth "noise010" 0.10

########################################
# 4. Lightweight interaction baselines on reviewer-sensitive slices.
########################################

run_lasso() {
  local label="$1"
  local fn="$2"
  local noise="$3"
  local rho="$4"
  local proxies="$5"
  "$PY" experiments/run_sparse_interaction_lasso_baseline.py \
    --function "$fn" \
    --samples 512 1024 \
    --test_samples 2048 \
    --dimension 100 \
    --noise "$noise" \
    --nuisance_correlation "$rho" \
    --n_correlated_proxies "$proxies" \
    --top_m 4 \
    --seeds "${SEEDS12[@]}" \
    --cv 5 \
    --max_iter 10000 \
    --out_dir "$OUT_ROOT/baselines/lasso_${label}"
}

run_hsic() {
  local label="$1"
  local fn="$2"
  local noise="$3"
  local rho="$4"
  local proxies="$5"
  "$PY" experiments/run_residual_hsic_pair_screen.py \
    --functions "$fn" \
    --samples 512 1024 \
    --test_samples 2048 \
    --dimension 100 \
    --noise "$noise" \
    --nuisance_correlation "$rho" \
    --n_correlated_proxies "$proxies" \
    --crossfit_folds 5 \
    --rff_dim 64 \
    --y_rff_dim 32 \
    --top_pairs_for_support 1 \
    --seeds "${SEEDS12[@]}" \
    --out_dir "$OUT_ROOT/baselines/hsic_${label}"
}

run_cmd "lasso_clean_c025" run_lasso "clean_c025" core_interaction_c025 0.00 0 0
run_cmd "lasso_noise010_c025" run_lasso "noise010_c025" core_interaction_c025 0.10 0 0
run_cmd "lasso_rho09_c025" run_lasso "rho09_c025" core_interaction_c025 0.05 0.9 8
run_cmd "hsic_clean_c025" run_hsic "clean_c025" core_interaction_c025 0.00 0 0
run_cmd "hsic_noise010_c025" run_hsic "noise010_c025" core_interaction_c025 0.10 0 0
run_cmd "hsic_rho09_c025" run_hsic "rho09_c025" core_interaction_c025 0.05 0.9 8

########################################
# 5. Stretch queue: only reached if the main pack finishes early.
########################################

# A. Grid-update and longer-step checks. These address the critique that the
# inspected readout failed only because the default no-grid-update training was
# under-tuned.
run_kan_sens_update_grid() {
  local label="$1"
  local n="$2"
  local width="$3"
  local steps="$4"
  "$PY" experiments/run_kan_probe_sensitivity.py \
    --out_dir "$OUT_ROOT/kan_sensitivity/${label}" \
    --function core_interaction_c025 \
    --samples "$n" \
    --dimension 100 \
    --test_samples 2048 \
    --noise 0 \
    --seeds "${SEEDS10[@]}" \
    --methods feature_stability_var feature_edge_hybrid \
    --top_ms 4 6 10 20 \
    --width_hidden "$width" \
    --grid 5 \
    --k 3 \
    --lamb 0.001 \
    --probe_steps "$steps" \
    --update_grid \
    --grid_update_num 5 \
    --probe_variable_points 512 \
    --pred_batch_size 4096 \
    --device "$RUN_DEVICE"
}

for n in 512 768 1024; do
  run_cmd "stretch_updategrid_w8_n${n}" run_kan_sens_update_grid "updategrid_w8_c025_d100_n${n}" "$n" 8 75
  run_cmd "stretch_updategrid_w16_n${n}" run_kan_sens_update_grid "updategrid_w16_c025_d100_n${n}" "$n" 16 75
done

# B. Add a third real covariate distribution. This is slower than a smoke test
# but useful if the earlier semi-synthetic checks finish too quickly.
run_semisynth_wine() {
  local label="$1"
  local noise="$2"
  "$PY" experiments/run_semisynthetic_covariate_audit.py \
    --out-dir "$OUT_ROOT/semisynthetic/${label}" \
    --datasets wine \
    --coefficients 0.10 0.25 0.50 \
    --samples 128 256 384 \
    --test-samples 128 \
    --outer-seeds "${SEEDS10[@]}" \
    --noise "$noise" \
    --R 12 \
    --top-m 4 \
    --methods feature_stability_var feature_edge_hybrid \
    --width-hidden 8 \
    --grid 5 \
    --k 3 \
    --lamb 0.001 \
    --probe-steps 35 \
    --pred-batch-size 4096 \
    --device "$RUN_DEVICE"
}

run_cmd "stretch_semisynth_wine_noise0" run_semisynth_wine "wine_noise0" 0.00
run_cmd "stretch_semisynth_wine_noise010" run_semisynth_wine "wine_noise010" 0.10

# C. Baselines on non-product and multi-interaction formulas. These are not
# intended to become a leaderboard, but they help decide whether the residual
# screens are only winning on raw bilinear products.
for fn in formula_trig_product formula_rational_product formula_three_way_product formula_mixed_sparse formula_nested_trig; do
  run_cmd "stretch_lasso_${fn}" run_lasso "stretch_${fn}" "$fn" 0.00 0 0
  run_cmd "stretch_hsic_${fn}" run_hsic "stretch_${fn}" "$fn" 0.00 0 0
done

########################################
# Final summary.
########################################

run_cmd "summary" "$PY" scripts/summarize_revision_boundary_overnight.py --root "$OUT_ROOT"

date -Is > "$OUT_ROOT/FINISHED_AT"
log "DONE boundary overnight pack"
