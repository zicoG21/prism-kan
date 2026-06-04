#!/usr/bin/env bash
#SBATCH --job-name=kan-a40-scorergram
#SBATCH --partition=spgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/greatlakes/%x_%A_%a.out
#SBATCH --error=logs/greatlakes/%x_%A_%a.err

set -euo pipefail

# A40 / spgpu scorer-indexed claim-grammar experiment.
#
# Purpose:
#   This is not another pyKAN seed dump.  It trains one full-dimensional KAN per
#   seed and evaluates the same declared interaction claim with multiple pair
#   scorers on the same fitted workflow:
#       EPIM/readout proposal, functional ANOVA, finite difference, and hybrid.
#   The paper use is to support "scorer-indexed structural claims": a task card
#   must declare both its claim grammar and the scorer that licenses a pair
#   recovery statement.
#
# Suggested submissions.  No array throttle is used here; Slurm/account GRES
# limits should decide concurrency.
#
#   sbatch --account=jaabell0 --array=0-9 \
#     --export=ALL,PYTHON_BIN=$PWD/.venv/bin/python,SEED_BASE=0,SEED_COUNT=12,LABEL_SUFFIX=j0 \
#     scripts/greatlakes_spgpu_pair_scorer_grammar_a40.sbatch
#
#   sbatch --account=engin1 --array=0-9 \
#     --export=ALL,PYTHON_BIN=$PWD/.venv/bin/python,SEED_BASE=200,SEED_COUNT=12,LABEL_SUFFIX=e0 \
#     scripts/greatlakes_spgpu_pair_scorer_grammar_a40.sbatch
#
# Optional:
#   export SCORERS=epim,anova_abs,fd,hessian,hybrid_epim_anova
# Hessian is intentionally off by default because it is slower at d=100.

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs/greatlakes

PY="${PYTHON_BIN:-python}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"

batch_size="${BATCH_SIZE:-16384}"
pair_chunk_size="${PAIR_CHUNK_SIZE:-1024}"
max_all_pairs="${MAX_ALL_PAIRS:-20000}"

seed_base="${SEED_BASE:-0}"
seed_count="${SEED_COUNT:-12}"
seed_end="$((seed_base + seed_count - 1))"
seeds="${seed_base}-${seed_end}"
label_suffix="${LABEL_SUFFIX:-s${seed_base}_${seed_end}}"
if [[ -n "${SCORERS_COLON:-}" ]]; then
  scorers="${SCORERS_COLON//:/,}"
else
  scorers="${SCORERS:-epim,anova_abs,fd,hybrid_epim_anova}"
fi

echo "[$(date -Is)] node=${SLURM_JOB_NODELIST:-unknown-node}"
echo "[$(date -Is)] seeds=${seeds} label_suffix=${label_suffix} scorers=${scorers}"
echo "[$(date -Is)] batch_size=${batch_size} pair_chunk_size=${pair_chunk_size} max_all_pairs=${max_all_pairs}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

# label|function|samples|dimension|noise|update_grid|width|steps|anova_points|anova_background|fd_points
ROWS=(
  "bilinear_clean|formula_bilinear|1024|100|0.00|0|16|90|16|16|16"
  "sqrt_energy_clean|formula_sqrt_energy|1024|100|0.00|0|16|90|16|16|16"
  "weak_centered_clean|formula_weak_centered|1024|100|0.00|0|16|90|16|16|16"
  "weak_centered_noise005|formula_weak_centered|1024|100|0.05|0|16|90|16|16|16"
  "mixed_sparse_clean|formula_mixed_sparse|1024|100|0.00|0|16|90|16|16|16"
  "mixed_sparse_noise005|formula_mixed_sparse|1024|100|0.05|0|16|90|16|16|16"
  "division_mixed_clean|formula_division_mixed|1024|100|0.00|0|16|90|16|16|16"
  "rational_product_clean|formula_rational_product|1024|100|0.00|0|16|90|16|16|16"
  "nested_trig_clean|formula_nested_trig|1024|100|0.00|0|16|90|16|16|16"
  "three_way_clean|formula_three_way_product|1024|100|0.00|0|16|90|16|16|16"
)

row="${ROWS[$SLURM_ARRAY_TASK_ID]}"
IFS='|' read -r base_label function samples dimension noise update_grid width steps anova_points anova_background fd_points <<< "$row"

label="${base_label}_${label_suffix}"
out_dir="results/revision/pair_scorer_claim_grammar_a40/${label}"

args=(
  experiments/run_pair_scorer_claim_grammar.py
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
  --scorers "$scorers"
  --anova-points "$anova_points"
  --anova-background "$anova_background"
  --pair-chunk-size "$pair_chunk_size"
  --fd-points "$fd_points"
  --fd-h 0.01
  --batch-size "$batch_size"
  --max-all-pairs "$max_all_pairs"
  --device cuda
  --out-dir "$out_dir"
)

if [[ "$update_grid" == "1" ]]; then
  args+=(--update-grid --grid-update-num 5)
fi

echo "[$(date -Is)] running scorer grammar ${label}"
"$PY" "${args[@]}"
echo "[$(date -Is)] done scorer grammar ${label}"
