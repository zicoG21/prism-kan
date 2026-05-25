#!/usr/bin/env bash
set -euo pipefail

mkdir -p results/hard_regime/logs
mkdir -p results/hard_regime/details
mkdir -p results/hard_regime/summaries
mkdir -p results/hard_regime/figs

FUNCTIONS=(
  core_interaction_c01
  core_interaction_c025
  core_interaction_c05
  core_interaction_c1
)

SAMPLES=(128 256 512 1024)
DIMS=(50 100)

SCREEN_MODES=(raw rf oracle_support random exclude_interaction)
SEEDS=(0 1 2 3 4 5 6 7 8 9)

for fn in "${FUNCTIONS[@]}"; do
  for n in "${SAMPLES[@]}"; do
    for d in "${DIMS[@]}"; do

      tag="${fn}_n${n}_d${d}"

      echo "============================================================"
      echo "Running ${tag}"
      echo "============================================================"

      python experiments/run_tuned_kan_recovery.py \
        --functions "${fn}" \
        --screen_modes "${SCREEN_MODES[@]}" \
        --samples "${n}" \
        --test_samples 4096 \
        --dimension "${d}" \
        --noise 0.0 \
        --seeds "${SEEDS[@]}" \
        --grid 5 \
        --k 3 \
        --width_hidden 8 \
        --lamb 0.001 \
        --steps 50 \
        --opt LBFGS \
        --no_update_grid \
        --top_m 4 \
        --rf_trees 500 \
        --variable_points 512 \
        --interaction_method fd \
        --fd_points 512 \
        --fd_h 0.01 \
        --pred_batch_size 4096 \
        --device auto \
        --out "results/hard_regime/details/${tag}_detail.csv" \
        --summary_out "results/hard_regime/summaries/${tag}_summary.csv" \
        --fig_dir "results/hard_regime/figs/${tag}" \
        2>&1 | tee "results/hard_regime/logs/${tag}.log"

    done
  done
done

echo "All hard-regime runs finished."
