#!/usr/bin/env bash
set -euo pipefail

# Denser same-data KAN-FE sample-size grid for the workshop audit.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-/home/perzival/anaconda3/envs/prism/bin/python}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

cd "$ROOT"
mkdir -p logs

"$PY" experiments/run_same_data_kan_stability_probe.py \
  --function core_interaction_c025 \
  --samples 640 768 896 1280 \
  --dimension 100 \
  --test_samples 2048 \
  --outer_seeds 0 1 2 3 4 5 6 7 8 9 \
  --R 20 \
  --resample bootstrap \
  --probe_steps 35 \
  --refit_steps 50 \
  --anova_points 64 \
  --anova_background 64 \
  --method feature_edge_hybrid \
  --out_dir results/workshop_review_tables/same_data_kan_stability_c025_d100_n_grid_R20_10seed \
  > logs/same_data_kan_stability_c025_d100_n_grid_R20_10seed.log 2>&1
