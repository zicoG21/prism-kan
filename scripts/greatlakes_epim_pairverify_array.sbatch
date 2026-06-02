#!/usr/bin/env bash
#SBATCH --job-name=kan-epim-verify
#SBATCH --partition=gpu,spgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/greatlakes/%x_%A_%a.out
#SBATCH --error=logs/greatlakes/%x_%A_%a.err

set -euo pipefail

# EPIM-propose / ANOVA-verify validation.
#
# Submit with an explicit account, for example:
#   sbatch --account=jaabell0 --array=0-9%5 \
#     --export=ALL,PYTHON_BIN=$PWD/.venv/bin/python \
#     scripts/greatlakes_epim_pairverify_array.sbatch
#
# Or, if jaabell0 is congested and engin1 has available GPU quota:
#   sbatch --account=engin1 --array=0-9%2 \
#     --export=ALL,PYTHON_BIN=$PWD/.venv/bin/python \
#     scripts/greatlakes_epim_pairverify_array.sbatch

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs/greatlakes

PY="${PYTHON_BIN:-python}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-2}"

# label|function|samples|dimension|noise|update_grid|width|steps|proposal_qs|random_controls|endpoint_top_m|seeds
ROWS=(
  "core_clean_n512_qs|core_interaction_c025|512|100|0|0|16|75|8,16,32,64,128|128|4|0-29"
  "core_clean_n1024_qs|core_interaction_c025|1024|100|0|0|16|75|8,16,32,64,128|128|4|0-29"
  "core_grid_n512_qs|core_interaction_c025|512|100|0|1|16|75|8,16,32,64,128|128|4|0-29"
  "core_grid_n1024_qs|core_interaction_c025|1024|100|0|1|16|75|8,16,32,64,128|128|4|0-29"
  "core_noise010_n512_qs|core_interaction_c025|512|100|0.10|0|16|75|8,16,32,64,128|128|4|0-29"
  "core_noise010_n1024_qs|core_interaction_c025|1024|100|0.10|0|16|75|8,16,32,64,128|128|4|0-29"
  "core_clean_d500_qs|core_interaction_c025|2048|500|0|0|16|75|16,32,64,128|256|6|0-19"
  "formula_bilinear_qs|formula_bilinear|1024|100|0|0|16|90|8,16,32,64,128|128|4|0-29"
  "formula_mixed_sparse_qs|formula_mixed_sparse|1024|100|0|0|16|90|8,16,32,64,128|128|4|0-29"
  "formula_division_mixed_qs|formula_division_mixed|1024|100|0|0|16|90|8,16,32,64,128|128|4|0-29"
)

row="${ROWS[$SLURM_ARRAY_TASK_ID]}"
IFS='|' read -r label function samples dimension noise update_grid width steps proposal_qs random_controls endpoint_top_m seeds <<< "$row"

out_dir="results/revision/epim_pairverify/${label}"
args=(
  scripts/run_kan_epim_pairverify.py
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
  --proposal-qs "$proposal_qs"
  --random-controls "$random_controls"
  --endpoint-top-m "$endpoint_top_m"
  --anova-points 24
  --anova-background 24
  --pair-chunk-size 256
  --batch-size 8192
  --score-true-pair-for-diagnostics
  --device cuda
  --out-dir "$out_dir"
)

if [[ "$update_grid" == "1" ]]; then
  args+=(--update-grid --grid-update-num 5)
fi

echo "[$(date -Is)] running EPIM PairVerify ${label} on ${SLURM_JOB_NODELIST:-unknown-node}"
"$PY" "${args[@]}"
echo "[$(date -Is)] done EPIM PairVerify ${label}"
