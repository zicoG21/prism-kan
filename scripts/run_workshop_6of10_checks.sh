#!/usr/bin/env bash
set -euo pipefail

PY="${PYTHON:-/home/perzival/anaconda3/envs/prism/bin/python}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

COMMON=(--device auto --probe_steps 25 --probe_variable_points 512 --test_samples 2048)
SENS_ROOT="results/workshop_review_tables/kan_probe_hparam_sensitivity_c1_d1000_n1024"
ROBUST_ROOT="results/workshop_review_tables/kan_probe_noise_corr_c025_d100_n1024"
RESID_ROOT="results/workshop_review_tables/residual_pair_screen_noise_corr_c025_d100_n1024"

# One-factor KAN-native support-probe sensitivity around the paper default.
"$PY" experiments/run_kan_probe_sensitivity.py \
  --out_dir "$SENS_ROOT/width8_grid5_lamb1e-3" \
  --function core_interaction_c1 --samples 1024 --dimension 1000 \
  --seeds 730 731 732 733 \
  --width_hidden 8 --grid 5 --lamb 0.001 \
  --top_ms 6 20 50 100 250 500 1000 \
  "${COMMON[@]}"

"$PY" experiments/run_kan_probe_sensitivity.py \
  --out_dir "$SENS_ROOT/width16_grid5_lamb1e-3" \
  --function core_interaction_c1 --samples 1024 --dimension 1000 \
  --seeds 730 731 732 733 \
  --width_hidden 16 --grid 5 --lamb 0.001 \
  --top_ms 6 20 50 100 250 500 1000 \
  "${COMMON[@]}"

"$PY" experiments/run_kan_probe_sensitivity.py \
  --out_dir "$SENS_ROOT/width32_grid5_lamb1e-3" \
  --function core_interaction_c1 --samples 1024 --dimension 1000 \
  --seeds 730 731 732 733 \
  --width_hidden 32 --grid 5 --lamb 0.001 \
  --top_ms 6 20 50 100 250 500 1000 \
  "${COMMON[@]}"

"$PY" experiments/run_kan_probe_sensitivity.py \
  --out_dir "$SENS_ROOT/width8_grid3_lamb1e-3" \
  --function core_interaction_c1 --samples 1024 --dimension 1000 \
  --seeds 730 731 732 733 \
  --width_hidden 8 --grid 3 --lamb 0.001 \
  --top_ms 6 20 50 100 250 500 1000 \
  "${COMMON[@]}"

"$PY" experiments/run_kan_probe_sensitivity.py \
  --out_dir "$SENS_ROOT/width8_grid10_lamb1e-3" \
  --function core_interaction_c1 --samples 1024 --dimension 1000 \
  --seeds 730 731 732 733 \
  --width_hidden 8 --grid 10 --lamb 0.001 \
  --top_ms 6 20 50 100 250 500 1000 \
  "${COMMON[@]}"

"$PY" experiments/run_kan_probe_sensitivity.py \
  --out_dir "$SENS_ROOT/width8_grid5_lamb1e-4" \
  --function core_interaction_c1 --samples 1024 --dimension 1000 \
  --seeds 730 731 732 733 \
  --width_hidden 8 --grid 5 --lamb 0.0001 \
  --top_ms 6 20 50 100 250 500 1000 \
  "${COMMON[@]}"

"$PY" experiments/run_kan_probe_sensitivity.py \
  --out_dir "$SENS_ROOT/width8_grid5_lamb1e-2" \
  --function core_interaction_c1 --samples 1024 --dimension 1000 \
  --seeds 730 731 732 733 \
  --width_hidden 8 --grid 5 --lamb 0.01 \
  --top_ms 6 20 50 100 250 500 1000 \
  "${COMMON[@]}"

# Noise/correlated-nuisance KAN-native support-probe robustness at d=100.
for noise in 0 0.05 0.1; do
  for rho in 0 0.5 0.9; do
    "$PY" experiments/run_kan_probe_sensitivity.py \
      --out_dir "$ROBUST_ROOT/noise${noise}_rho${rho}" \
      --function core_interaction_c025 --samples 1024 --dimension 100 \
      --noise "$noise" --nuisance_correlation "$rho" --n_correlated_proxies 8 \
      --seeds 740 741 742 743 744 \
      --width_hidden 8 --grid 5 --lamb 0.001 \
      --top_ms 4 8 20 \
      "${COMMON[@]}"

    "$PY" experiments/run_residual_pair_screen_baseline.py \
      --out_dir "$RESID_ROOT/noise${noise}_rho${rho}" \
      --function core_interaction_c025 --samples 1024 --dimension 100 \
      --noise "$noise" --nuisance_correlation "$rho" --n_correlated_proxies 8 \
      --seeds 740 741 742 743 744
  done
done

"$PY" scripts/summarize_workshop_6of10_checks.py
