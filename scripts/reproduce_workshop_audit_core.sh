#!/usr/bin/env bash
set -euo pipefail

# Reproduce the core finite-data audit artifacts used by the workshop draft.
# These commands are intentionally explicit rather than hidden behind a private
# pipeline. They assume the local conda environment used during the experiments.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-/home/perzival/anaconda3/envs/prism/bin/python}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

cd "$ROOT"

echo "[1/6] Same-training-set KAN-FE bootstrap, d=100"
"$PY" experiments/run_same_data_kan_stability_probe.py \
  --function core_interaction_c025 \
  --samples 512 1024 \
  --dimension 100 \
  --test_samples 2048 \
  --outer_seeds 0 1 2 3 4 5 6 7 8 9 \
  --R 20 \
  --resample bootstrap \
  --probe_steps 35 \
  --refit_steps 50 \
  --anova_points 64 \
  --anova_background 64 \
  --out_dir results/workshop_review_tables/same_data_kan_stability_c025_d100_bootstrap_R20_10seed

echo "[2/6] Same-training-set KAN score ablation, d=100"
"$PY" experiments/run_same_data_kan_stability_probe.py \
  --function core_interaction_c025 \
  --samples 512 1024 \
  --dimension 100 \
  --test_samples 2048 \
  --outer_seeds 0 1 2 3 4 5 6 7 8 9 \
  --R 20 \
  --resample bootstrap \
  --probe_steps 35 \
  --refit_steps 1 \
  --skip_refit \
  --methods feature_stability_var edge_stability_var edge_endpoint_mass edge_pair_endpoint feature_edge_hybrid \
  --out_dir results/workshop_review_tables/same_data_kan_score_ablation_c025_d100_R20_10seed_skiprefit

echo "[3/6] Residual raw-product pair screen, d=100"
"$PY" experiments/run_residual_pair_screen_baseline.py \
  --functions core_interaction_c025 \
  --samples 512 1024 \
  --dimension 100 \
  --test_samples 2048 \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --fixed_alpha 1.0 \
  --out_dir results/interaction_baselines/residual_pair_screen_c025_d100_n512_1024

echo "[4/6] Residual raw-product pair screen, cross-fit check"
"$PY" experiments/run_residual_pair_screen_baseline.py \
  --functions core_interaction_c025 \
  --samples 512 1024 \
  --dimension 100 \
  --test_samples 2048 \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --fixed_alpha 1.0 \
  --crossfit_folds 5 \
  --out_dir results/interaction_baselines/residual_pair_screen_crossfit_c025_d100

echo "[5/6] Residual tensor-spline mini-suite"
"$PY" experiments/run_residual_tensor_spline_pair_baseline.py \
  --functions formula_bilinear formula_weak_centered formula_trig_product \
    formula_nested_trig formula_rational_product formula_division_mixed \
    formula_exp_product formula_log_product formula_three_way_product \
    formula_mixed_sparse formula_sqrt_energy \
  --samples 1024 \
  --dimension 100 \
  --test_samples 2048 \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --fixed_alpha 1.0 \
  --out_dir results/interaction_baselines/residual_tensor_spline_minisuite_trainresid_alpha1_d100_n1024_10seed

echo "[6/6] Regenerate main paper figure"
"$PY" experiments/plot_audit_benchmark_main_figure.py

echo "Done. See docs/artifact_checklist_workshop_20260527.md for the artifact map."
