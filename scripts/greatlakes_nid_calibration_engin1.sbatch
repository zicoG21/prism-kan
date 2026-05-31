#!/usr/bin/env bash
#SBATCH --job-name=kan-nid-cal
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

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs/greatlakes

PY="${PYTHON_BIN:-python}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-2}"

# Engin1 NID-style calibration rows.  This is not a leaderboard; it strengthens
# the paper's calibration object by contrasting weak centered interactions with
# product-like controls under a larger MLP and a fixed budget.
#
# label|function|samples|dimension|noise|hidden|depth|epochs|patience|seeds|methods|hessian_points
ROWS=(
  "weak_c025_n512_h512|core_interaction_c025|512|100|0.00|512|3|3000|500|30-37|nid hessian|64"
  "weak_c025_n1024_h512|core_interaction_c025|1024|100|0.00|512|3|3000|500|30-37|nid hessian|64"
  "noise010_c025_n1024_h512|core_interaction_c025|1024|100|0.10|512|3|3000|500|30-37|nid hessian|64"
  "strong_c1_n1024_h512|core_interaction_c1|1024|100|0.00|512|3|3000|500|30-37|nid hessian|64"
  "formula_bilinear_h512|formula_bilinear|1024|100|0.00|512|3|3000|500|30-37|nid hessian|64"
  "formula_log_product_h512|formula_log_product|1024|100|0.00|512|3|3000|500|30-37|nid hessian|64"
)

row="${ROWS[$SLURM_ARRAY_TASK_ID]}"
IFS='|' read -r label function samples dimension noise hidden depth epochs patience seeds methods hessian_points <<< "$row"

seed_args=()
IFS='-' read -r seed_lo seed_hi <<< "$seeds"
for s in $(seq "$seed_lo" "$seed_hi"); do
  seed_args+=("$s")
done

method_args=()
for m in $methods; do
  method_args+=("$m")
done

out_dir="results/revision/engin1_nid_calibration/${label}"

echo "[$(date -Is)] running engin1 NID calibration ${label} on ${SLURM_JOB_NODELIST:-unknown-node}"
"$PY" experiments/run_nid_interaction_baseline.py \
  --function "$function" \
  --samples "$samples" \
  --test_samples 2048 \
  --dimension "$dimension" \
  --noise "$noise" \
  --seeds "${seed_args[@]}" \
  --methods "${method_args[@]}" \
  --hidden "$hidden" \
  --depth "$depth" \
  --epochs "$epochs" \
  --patience "$patience" \
  --eval_every 25 \
  --batch_size 256 \
  --lr 0.001 \
  --weight_decay 1e-5 \
  --hessian_points "$hessian_points" \
  --device cuda \
  --out_dir "$out_dir"
echo "[$(date -Is)] done engin1 NID calibration ${label}"
