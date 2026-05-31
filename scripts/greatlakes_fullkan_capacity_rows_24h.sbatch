#!/usr/bin/env bash
#SBATCH --job-name=kan-full-cap
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/greatlakes/%x_%A_%a.out
#SBATCH --error=logs/greatlakes/%x_%A_%a.err

set -euo pipefail

# Full-KAN all-pairs ANOVA capacity rows moved from local GPU to Great Lakes.
#
# These rows are CPU/ANOVA-heavy on the laptop and do not use the local GPU
# efficiently.  They close capacity/protocol questions for the stage-discordance
# story: width 32 clean, width 32 grid-update, and width 64 clean at n=512.
#
# Suggested submit command from Great Lakes:
#
#   sbatch --account=jaabell0 \
#     --export=ALL,PYTHON_BIN=$PWD/.venv/bin/python \
#     --array=0-2%3 scripts/greatlakes_fullkan_capacity_rows_24h.sbatch

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs/greatlakes

PY="${PYTHON_BIN:-python}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-2}"

# label|width|steps|update_grid|seed_start|seed_end
ROWS=(
  "clean_w32_n512_s4000_4059|32|120|0|4000|4059"
  "gridupdate_w32_n512_s4060_4119|32|120|1|4060|4119"
  "clean_w64_n512_s4120_4179|64|100|0|4120|4179"
)

row="${ROWS[$SLURM_ARRAY_TASK_ID]}"
IFS='|' read -r label width steps update_grid seed_start seed_end <<< "$row"

out_dir="results/revision/greatlakes_fullkan_capacity_rows/${label}"

args=(
  scripts/run_full_kan_pair_anova_probe.py
  --function core_interaction_c025
  --samples 512
  --test-samples 4096
  --dimension 100
  --noise 0.00
  --seeds "${seed_start}-${seed_end}"
  --width-hidden "$width"
  --grid 5
  --k 3
  --steps "$steps"
  --lamb 0.001
  --batch-size 2048
  --anova-points 32
  --anova-background 32
  --pair-mode all
  --device cuda
  --out-dir "$out_dir"
)

if [[ "$update_grid" == "1" ]]; then
  args+=(--update-grid --grid-update-num 5)
fi

echo "[$(date -Is)] running full-KAN capacity row ${label} on ${SLURM_JOB_NODELIST:-unknown-node}"
"$PY" "${args[@]}"
echo "[$(date -Is)] done full-KAN capacity row ${label}"
