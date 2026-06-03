#!/usr/bin/env bash
#SBATCH --job-name=kan-treegate
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/greatlakes/%x_%A_%a.out
#SBATCH --error=logs/greatlakes/%x_%A_%a.err

set -euo pipefail

# CPU-only TreeGate pilot:
#   1. fit a tree ensemble;
#   2. screen "any-pair endpoints" by tree feature importance;
#   3. verify only gated candidate pairs by RF pair-permutation synergy.
#
# This job intentionally uses the standard partition, not GPU. It is a
# candidate-generator / verifier experiment for the ClaimTransfer-Bench
# "any-pair endpoint" claim, not another all-pairs scorer dump.

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs/greatlakes

PY="${PYTHON_BIN:-$PWD/.venv/bin/python}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-8}"

# label|functions_csv|samples|dimension|noise|seeds|gate_size|verify_points|rf_trees
ROWS=(
  "core_d100_clean|core_interaction_c025|1024|100|0.00|0-29|20|256|500"
  "core_d100_noise005|core_interaction_c025|1024|100|0.05|0-29|20|256|500"
  "core_d1000_clean|core_interaction_c025|2048|1000|0.00|0-19|30|192|700"
  "core_d1000_noise005|core_interaction_c025|2048|1000|0.05|0-19|30|192|700"
  "mixed_rational_d100_clean|formula_mixed_sparse,formula_rational_product,formula_division_mixed|1024|100|0.00|0-29|24|256|500"
  "formula_breadth_d100_clean|formula_bilinear,formula_weak_centered,formula_trig_product,formula_nested_trig,formula_three_way_product|1024|100|0.00|0-19|24|256|500"
  "formula_breadth_d100_noise005|formula_weak_centered,formula_mixed_sparse,formula_rational_product|1024|100|0.05|0-19|24|256|500"
  "core_gate_sensitivity|core_interaction_c025|1024|100|0.00|30-59|8|256|500"
  "core_gate_sensitivity_b16|core_interaction_c025|1024|100|0.00|30-59|16|256|500"
  "core_gate_sensitivity_b32|core_interaction_c025|1024|100|0.00|30-59|32|256|500"
)

row="${ROWS[$SLURM_ARRAY_TASK_ID]}"
IFS='|' read -r label functions_csv samples dimension noise seeds gate_size verify_points rf_trees <<< "$row"
IFS=',' read -r -a functions <<< "$functions_csv"

out_dir="results/revision/treegate_pair_screen/${label}"

echo "[$(date -Is)] host=$(hostname) label=${label} functions=${functions_csv} seeds=${seeds}"
echo "[$(date -Is)] TreeGate is CPU-only; no GPU requested."

"$PY" experiments/run_treegate_pair_screen.py \
  --functions "${functions[@]}" \
  --samples "$samples" \
  --test-samples 2048 \
  --dimension "$dimension" \
  --noise "$noise" \
  --seeds "$seeds" \
  --gate-size "$gate_size" \
  --verify-points "$verify_points" \
  --rf-trees "$rf_trees" \
  --rf-min-samples-leaf 2 \
  --n-jobs "${SLURM_CPUS_PER_TASK:-8}" \
  --batch-size 8192 \
  --pair-chunk-size 128 \
  --out-dir "$out_dir" \
  --label "$label" \
  --resume

echo "[$(date -Is)] done TreeGate ${label}"
