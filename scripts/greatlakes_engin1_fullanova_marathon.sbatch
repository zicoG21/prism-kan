#!/usr/bin/env bash
#SBATCH --job-name=kan-e1-fullmar
#SBATCH --account=engin1
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=28G
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --array=0-17%2
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

# Long engin1 queue.  The 24h Slurm time is a safety allocation, not a target
# cutoff.  Each row uses a disjoint seed block from the earlier full-KAN ANOVA
# waves so we can improve uncertainty estimates without duplicating rows.
#
# label|function|samples|dimension|noise|update_grid|width|steps|seeds|anova_points|anova_background
ROWS=(
  "clean_w16_n512_s100_189|core_interaction_c025|512|100|0.00|0|16|75|100-189|24|24"
  "clean_w16_n768_s100_189|core_interaction_c025|768|100|0.00|0|16|75|100-189|24|24"
  "clean_w16_n1024_s100_189|core_interaction_c025|1024|100|0.00|0|16|75|100-189|24|24"
  "gridupdate_w16_n512_s100_189|core_interaction_c025|512|100|0.00|1|16|75|100-189|24|24"
  "gridupdate_w16_n768_s100_189|core_interaction_c025|768|100|0.00|1|16|75|100-189|24|24"
  "gridupdate_w16_n1024_s100_189|core_interaction_c025|1024|100|0.00|1|16|75|100-189|24|24"
  "noise005_w16_n512_s100_189|core_interaction_c025|512|100|0.05|0|16|75|100-189|24|24"
  "noise005_w16_n1024_s100_189|core_interaction_c025|1024|100|0.05|0|16|75|100-189|24|24"
  "noise010_w16_n512_s100_189|core_interaction_c025|512|100|0.10|0|16|75|100-189|24|24"
  "noise010_w16_n1024_s100_189|core_interaction_c025|1024|100|0.10|0|16|75|100-189|24|24"
  "clean_w32_n512_s100_189|core_interaction_c025|512|100|0.00|0|32|75|100-189|24|24"
  "clean_w32_n768_s100_189|core_interaction_c025|768|100|0.00|0|32|75|100-189|24|24"
  "gridupdate_w32_n512_s100_189|core_interaction_c025|512|100|0.00|1|32|75|100-189|24|24"
  "gridupdate_w32_n1024_s100_189|core_interaction_c025|1024|100|0.00|1|32|75|100-189|24|24"
  "c01_d20_w16_n512_s100_189|core_interaction_c01|512|20|0.00|0|16|75|100-189|24|24"
  "c01_d20_w16_n1024_s100_189|core_interaction_c01|1024|20|0.00|0|16|75|100-189|24|24"
  "c01_d20_w32_n512_s100_189|core_interaction_c01|512|20|0.00|0|32|75|100-189|24|24"
  "c01_d20_w32_n1024_s100_189|core_interaction_c01|1024|20|0.00|0|32|75|100-189|24|24"
)

row="${ROWS[$SLURM_ARRAY_TASK_ID]}"
IFS='|' read -r label function samples dimension noise update_grid width steps seeds anova_points anova_background <<< "$row"

out_dir="results/revision/engin1_fullanova_marathon/${label}"
args=()
if [[ "$update_grid" == "1" ]]; then
  args+=(--update-grid)
fi

echo "[$(date -Is)] running engin1 full-KAN ANOVA marathon ${label} on ${SLURM_JOB_NODELIST:-unknown-node}"
"$PY" scripts/run_full_kan_pair_anova_probe.py \
  --function "$function" \
  --samples "$samples" \
  --test-samples 2048 \
  --dimension "$dimension" \
  --noise "$noise" \
  --seeds "$seeds" \
  --width-hidden "$width" \
  --grid 5 \
  --k 3 \
  --steps "$steps" \
  --lamb 0.001 \
  --grid-update-num 5 \
  --batch -1 \
  --batch-size 8192 \
  --pair-mode all \
  --anova-points "$anova_points" \
  --anova-background "$anova_background" \
  --pair-chunk-size 1000 \
  --device cuda \
  --out-dir "$out_dir" \
  "${args[@]}"
echo "[$(date -Is)] done engin1 full-KAN ANOVA marathon ${label}"
