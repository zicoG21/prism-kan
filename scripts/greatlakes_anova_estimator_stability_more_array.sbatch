#!/usr/bin/env bash
#SBATCH --job-name=kan-anova-mc2
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

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs/greatlakes

PY="${PYTHON_BIN:-python}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-2}"

# label|function|samples|dimension|noise|update_grid|width|steps|seeds|mc_repeats|anova_points|anova_background
ROWS=(
  "clean_w16_n1024_32x32_s0_9_mc6|core_interaction_c025|1024|100|0|0|16|75|0-9|6|32|32"
  "gridupdate_w16_n1024_32x32_s0_9_mc6|core_interaction_c025|1024|100|0|1|16|75|0-9|6|32|32"
  "noise010_w16_n1024_32x32_s0_9_mc6|core_interaction_c025|1024|100|0.10|0|16|75|0-9|6|32|32"
  "clean_w32_n768_24x24_s0_9_mc6|core_interaction_c025|768|100|0|0|32|75|0-9|6|24|24"
)

row="${ROWS[$SLURM_ARRAY_TASK_ID]}"
IFS='|' read -r label function samples dimension noise update_grid width steps seeds mc_repeats anova_points anova_background <<< "$row"

out_dir="results/revision/anova_estimator_stability_more/${label}"
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

echo "[$(date -Is)] running ANOVA estimator stability more ${label} on ${SLURM_JOB_NODELIST:-unknown-node}"
"$PY" "${args[@]}"
echo "[$(date -Is)] done ANOVA estimator stability more ${label}"

