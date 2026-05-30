#!/usr/bin/env bash
#SBATCH --job-name=kan-anova-mc
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/greatlakes/%x_%A_%a.out
#SBATCH --error=logs/greatlakes/%x_%A_%a.err

set -euo pipefail

# Great Lakes full-KAN ANOVA estimator-stability array.
#
# Purpose:
#   Fix each trained full-KAN model and repeat the functional-ANOVA scorer with
#   different anchor/background subsamples. This directly checks whether the
#   all-pairs true-pair rank/margin is driven by Monte Carlo estimator noise.
#
# Suggested submission after current GPU chains:
#   sbatch --account=jaabell0 --dependency=afterany:<UPSTREAM_JOBID> \
#     --export=ALL,PYTHON_BIN=$PWD/.venv/bin/python \
#     --array=0-3%4 scripts/greatlakes_anova_estimator_stability_array.sbatch

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs/greatlakes

PY="${PYTHON_BIN:-python}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-2}"

# label|function|samples|dimension|noise|update_grid|width|steps|seeds|mc_repeats|anova_points|anova_background
ROWS=(
  "clean_w16_n768_mc10|core_interaction_c025|768|100|0|0|16|75|0-19|10|16|16"
  "clean_w16_n1024_mc10|core_interaction_c025|1024|100|0|0|16|75|0-19|10|16|16"
  "gridupdate_w16_n1024_mc10|core_interaction_c025|1024|100|0|1|16|75|0-19|10|16|16"
  "noise010_w16_n1024_mc10|core_interaction_c025|1024|100|0.10|0|16|75|0-19|10|16|16"
)

row="${ROWS[$SLURM_ARRAY_TASK_ID]}"
IFS='|' read -r label function samples dimension noise update_grid width steps seeds mc_repeats anova_points anova_background <<< "$row"

out_dir="results/revision/anova_estimator_stability/${label}"
args=(
  scripts/run_full_kan_anova_estimator_stability.py
  --function "$function"
  --samples "$samples"
  --test-samples 2048
  --dimension "$dimension"
  --noise "$noise"
  --seeds "$seeds"
  --mc-repeats "$mc_repeats"
  --width-hidden "$width"
  --grid 5
  --k 3
  --steps "$steps"
  --lamb 0.001
  --anova-points "$anova_points"
  --anova-background "$anova_background"
  --batch-size 8192
  --pair-chunk-size 1000
  --device cuda
  --out-dir "$out_dir"
)

if [[ "$update_grid" == "1" ]]; then
  args+=(--update-grid --grid-update-num 5)
fi

echo "[$(date -Is)] running ANOVA estimator stability ${label} on ${SLURM_JOB_NODELIST:-unknown-node}"
"$PY" "${args[@]}"
echo "[$(date -Is)] done ANOVA estimator stability ${label}"
