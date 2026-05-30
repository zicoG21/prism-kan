#!/usr/bin/env bash
#SBATCH --job-name=kan-full-follow
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

# Great Lakes full-KAN ANOVA follow-up rows.
#
# Purpose:
#   1. Fill the noise=0.10 full-model reliance curve.
#   2. Add width-32 capacity contrasts for the reviewer concern that failures
#      are under-capacity artifacts.
#   3. Check whether the c=0.10,d=20 non-monotone readout cell is also a
#      full-model reliance issue.
#
# Submit after the readout-taxonomy round:
#   sbatch --account=jaabell0 --dependency=afterany:<TAXONOMY_JOBID> \
#     --export=ALL,PYTHON_BIN=$PWD/.venv/bin/python \
#     --array=0-9%5 scripts/greatlakes_fullkan_followup_array.sbatch

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs/greatlakes

PY="${PYTHON_BIN:-python}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-2}"

# label|function|samples|dimension|noise|update_grid|width|steps|seeds|anova_points|anova_background
ROWS=(
  "noise010_w16_n640_60seed|core_interaction_c025|640|100|0.10|0|16|75|0-59|16|16"
  "noise010_w16_n768_60seed|core_interaction_c025|768|100|0.10|0|16|75|0-59|16|16"
  "clean_w32_n512_60seed|core_interaction_c025|512|100|0|0|32|75|0-59|16|16"
  "clean_w32_n768_60seed|core_interaction_c025|768|100|0|0|32|75|0-59|16|16"
  "gridupdate_w32_n512_60seed|core_interaction_c025|512|100|0|1|32|75|0-59|16|16"
  "gridupdate_w32_n1024_60seed|core_interaction_c025|1024|100|0|1|32|75|0-59|16|16"
  "c01_d20_w16_n512_60seed|core_interaction_c01|512|20|0|0|16|75|0-59|32|32"
  "c01_d20_w16_n1024_60seed|core_interaction_c01|1024|20|0|0|16|75|0-59|32|32"
  "c01_d20_w32_n512_60seed|core_interaction_c01|512|20|0|0|32|75|0-59|32|32"
  "c01_d20_w32_n1024_60seed|core_interaction_c01|1024|20|0|0|32|75|0-59|32|32"
)

row="${ROWS[$SLURM_ARRAY_TASK_ID]}"
IFS='|' read -r label function samples dimension noise update_grid width steps seeds anova_points anova_background <<< "$row"

out_dir="results/revision/fullkan_anova_followup/${label}"
args=(
  scripts/run_full_kan_pair_anova_probe.py
  --function "$function"
  --samples "$samples"
  --test-samples 2048
  --dimension "$dimension"
  --noise "$noise"
  --seeds "$seeds"
  --width-hidden "$width"
  --grid 5
  --k 3
  --steps "$steps"
  --lamb 0.001
  --pair-mode all
  --max-all-pairs 10000
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

echo "[$(date -Is)] running full-KAN follow-up ${label} on ${SLURM_JOB_NODELIST:-unknown-node}"
"$PY" "${args[@]}"
echo "[$(date -Is)] done full-KAN follow-up ${label}"
