#!/usr/bin/env bash
set -euo pipefail

# Next-round tests for KAN formula-level fidelity.
# Run from project root:
#   cd ~/kan_xai_benchmark
#   bash scripts/run_next_round_tests.sh
#
# Overnight:
#   nohup systemd-inhibit --what=sleep:idle --why="KAN next round tests" --mode=block \
#     bash scripts/run_next_round_tests.sh > results/next_round/logs/next_round.log 2>&1 &

source ~/anaconda3/etc/profile.d/conda.sh
conda activate prism

cd ~/kan_xai_benchmark
mkdir -p results/next_round/logs
mkdir -p results/next_round/feynman_interaction
mkdir -p results/next_round/dim_transition
mkdir -p results/next_round/tuned_screened

# If KAN grid update crashes with SGELSY/lstsq errors, rerun with:
#   KAN_GRID_FLAG="--no_update_grid" bash scripts/run_next_round_tests.sh
KAN_GRID_FLAG=${KAN_GRID_FLAG:-}

echo "============================================================"
echo "Stage 0: smoke test Feynman-style data functions"
echo "============================================================"
PYTHONPATH=. python - <<'PY'
from src.data import make_synthetic

for fn in ["feynman_energy", "feynman_gravity", "feynman_coulomb", "feynman_damped_wave"]:
    data = make_synthetic(function_name=fn, n_train=16, n_test=8, d=100, seed=0)
    gt = data["ground_truth"]
    print(fn, data["X_train"].shape, data["y_train"].shape, gt.active_variables, gt.interactions, gt.formula)
PY


echo "============================================================"
echo "Stage 1: Feynman interaction validation"
echo "Purpose: rerun Feynman formulas with real pairwise interaction scoring."
echo "Use only energy/gravity/coulomb first; damped_wave is currently a hard optimization case."
echo "============================================================"

PYTHONPATH=. python experiments/run_tuned_kan_recovery.py \
  --functions feynman_energy feynman_gravity feynman_coulomb \
  --screen_modes raw rf oracle_support random exclude_interaction \
  --samples 1024 \
  --test_samples 2048 \
  --dimension 100 \
  --noise 0.0 \
  --seeds 0 1 2 \
  --grid 5 \
  --k 3 \
  --width_hidden 8 \
  --lamb 0.0 \
  --steps 50 \
  ${KAN_GRID_FLAG} \
  --interaction_method fd \
  --fd_points 16 \
  --out results/next_round/feynman_interaction/feynman_interaction_eval.csv \
  --summary_out results/next_round/feynman_interaction/feynman_interaction_summary.csv \
  --fig_dir results/next_round/feynman_interaction/figures


echo "============================================================"
echo "Stage 2: dimension / nuisance-feature transition"
echo "Purpose: stop treating d=100 as the whole story; show where formula fidelity breaks."
echo "Functions: core_interaction and feynman_coulomb."
echo "Dimensions: d=5,20,50,100."
echo "============================================================"

for D in 5 20 50 100; do
  echo "---- dimension D=${D} ----"
  PYTHONPATH=. python experiments/run_tuned_kan_recovery.py \
    --functions core_interaction feynman_coulomb \
    --screen_modes raw rf oracle_support random exclude_interaction \
    --samples 1024 \
    --test_samples 2048 \
    --dimension ${D} \
    --noise 0.0 \
    --seeds 0 1 2 \
    --grid 5 \
    --k 3 \
    --width_hidden 8 \
    --lamb 0.0 \
    --steps 50 \
    ${KAN_GRID_FLAG} \
    --interaction_method fd \
    --fd_points 16 \
    --out results/next_round/dim_transition/dim_${D}_eval.csv \
    --summary_out results/next_round/dim_transition/dim_${D}_summary.csv \
    --fig_dir results/next_round/dim_transition/figures_dim_${D}
done


echo "============================================================"
echo "Stage 3: tuned KAN with support retained"
echo "Purpose: test whether Adrian-style tuned KAN succeeds once support is retained."
echo "This directly complements tuned raw KAN failure."
echo "============================================================"

PYTHONPATH=. python experiments/run_tuned_kan_recovery.py \
  --functions core_interaction core_interaction_c5 feynman_coulomb \
  --screen_modes raw rf oracle_support \
  --samples 1024 \
  --test_samples 2048 \
  --dimension 100 \
  --noise 0.0 \
  --seeds 0 1 2 \
  --grid 10 \
  --k 5 \
  --width_hidden 5 \
  --lamb 0.0022356972728751583 \
  --steps 200 \
  ${KAN_GRID_FLAG} \
  --interaction_method fd \
  --fd_points 16 \
  --out results/next_round/tuned_screened/tuned_screened_eval.csv \
  --summary_out results/next_round/tuned_screened/tuned_screened_summary.csv \
  --fig_dir results/next_round/tuned_screened/figures


echo "============================================================"
echo "Stage 4: compact summary"
echo "============================================================"
PYTHONPATH=. python scripts/summarize_next_round_tests.py

echo "DONE. Results are under results/next_round/"
