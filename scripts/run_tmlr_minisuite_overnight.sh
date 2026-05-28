#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PYTHON:-/home/perzival/anaconda3/envs/prism/bin/python}"
STAMP="${1:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="results/formula_fidelity_minisuite/overnight_tmlr_${STAMP}"
LOG_DIR="$RUN_ROOT/logs"
mkdir -p "$LOG_DIR"

exec > >(tee -a "$LOG_DIR/overnight.log") 2>&1

echo "[START] $(date)"
echo "[ROOT] $ROOT"
echo "[PY] $PY"
echo "[RUN_ROOT] $RUN_ROOT"

COMMON_FUNCTIONS=(
  formula_poly_additive
  formula_bilinear
  formula_weak_centered
  formula_trig_product
  formula_nested_trig
  formula_rational_product
  formula_division_mixed
  formula_exp_product
  formula_log_product
  formula_three_way_product
  formula_mixed_sparse
  formula_sqrt_energy
)

INTERACTION_FUNCTIONS=(
  formula_bilinear
  formula_weak_centered
  formula_trig_product
  formula_nested_trig
  formula_rational_product
  formula_division_mixed
  formula_exp_product
  formula_log_product
  formula_three_way_product
  formula_mixed_sparse
  formula_sqrt_energy
)

echo "[PHASE 1] Broad d=100 support/prediction ladder"
"$PY" experiments/run_formula_fidelity_minisuite.py \
  --out_dir "$RUN_ROOT/d100_support_ladder" \
  --functions "${COMMON_FUNCTIONS[@]}" \
  --samples 1024 \
  --test_samples 2048 \
  --dimension 100 \
  --noise 0.0 \
  --seeds 0 1 2 3 4 \
  --screen_modes raw rf oracle_support random exclude_interaction \
  --top_m 20 \
  --rf_trees 200 \
  --kan_steps 35 \
  --resume

echo "[PHASE 2] Interaction scoring on screened/oracle supports"
"$PY" experiments/run_formula_fidelity_minisuite.py \
  --out_dir "$RUN_ROOT/d100_interaction_scoring" \
  --functions "${INTERACTION_FUNCTIONS[@]}" \
  --samples 1024 \
  --test_samples 2048 \
  --dimension 100 \
  --noise 0.0 \
  --seeds 0 1 2 \
  --screen_modes rf oracle_support random exclude_interaction \
  --top_m 12 \
  --rf_trees 200 \
  --kan_steps 35 \
  --compute_interactions \
  --hessian_points 8 \
  --resume

echo "[PHASE 3] Correlated/noisy support audit on representative formulas"
"$PY" experiments/run_formula_fidelity_minisuite.py \
  --out_dir "$RUN_ROOT/d100_correlated_noisy_support" \
  --functions formula_weak_centered formula_division_mixed formula_mixed_sparse formula_trig_product \
  --samples 1024 \
  --test_samples 2048 \
  --dimension 100 \
  --noise 0.1 \
  --nuisance_correlation 0.9 \
  --n_correlated_proxies 8 \
  --seeds 0 1 2 \
  --screen_modes raw rf oracle_support random exclude_interaction \
  --top_m 20 \
  --rf_trees 200 \
  --kan_steps 35 \
  --resume

echo "[PHASE 4] Summarize"
"$PY" experiments/summarize_formula_fidelity_minisuite.py \
  --root "$RUN_ROOT" \
  --out "$RUN_ROOT/overnight_report.md"

echo "[DONE] $(date)"
echo "[REPORT] $RUN_ROOT/overnight_report.md"
