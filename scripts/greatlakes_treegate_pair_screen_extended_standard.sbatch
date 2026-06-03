#!/usr/bin/env bash
#SBATCH --job-name=kan-treegate2
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/greatlakes/%x_%A_%a.out
#SBATCH --error=logs/greatlakes/%x_%A_%a.err

set -euo pipefail

# Extended CPU-only TreeGate queue.
#
# Purpose:
#   Turn the TreeGate pilot into a usable method/benchmark story:
#     - any-pair endpoint screening;
#     - candidate-pair reduction;
#     - gated verifier recall;
#     - dimension and gate-size scaling.
#
# This uses the standard partition only. It does not request GPUs.

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs/greatlakes

PY="${PYTHON_BIN:-$PWD/.venv/bin/python}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-8}"

# label|functions_csv|samples|dimension|noise|seeds|gate_size|verify_points|rf_trees|min_leaf
ROWS=(
  # Dimension scaling for the weak centered card.
  "core_d100_b20_s60_119|core_interaction_c025|1024|100|0.00|60-119|20|256|500|2"
  "core_d500_b24_s0_39|core_interaction_c025|2048|500|0.00|0-39|24|192|700|2"
  "core_d1000_b30_s20_59|core_interaction_c025|2048|1000|0.00|20-59|30|192|700|2"
  "core_d2000_b40_s0_19|core_interaction_c025|3072|2000|0.00|0-19|40|128|900|2"

  # Gate-size curve at d=100: how strict can the endpoint gate be?
  "core_d100_b4_s120_179|core_interaction_c025|1024|100|0.00|120-179|4|256|500|2"
  "core_d100_b8_s120_179|core_interaction_c025|1024|100|0.00|120-179|8|256|500|2"
  "core_d100_b12_s120_179|core_interaction_c025|1024|100|0.00|120-179|12|256|500|2"
  "core_d100_b16_s120_179|core_interaction_c025|1024|100|0.00|120-179|16|256|500|2"
  "core_d100_b24_s120_179|core_interaction_c025|1024|100|0.00|120-179|24|256|500|2"
  "core_d100_b32_s120_179|core_interaction_c025|1024|100|0.00|120-179|32|256|500|2"

  # Noise and stronger interaction checks.
  "core_noise005_d100_b20_s60_119|core_interaction_c025|1024|100|0.05|60-119|20|256|500|2"
  "core_noise010_d100_b20_s60_119|core_interaction_c025|1024|100|0.10|60-119|20|256|500|2"
  "core_c05_d100_b20_s60_119|core_interaction_c05|1024|100|0.00|60-119|20|256|500|2"
  "core_c1_d100_b20_s60_119|core_interaction_c1|1024|100|0.00|60-119|20|256|500|2"

  # Formula breadth: product-like, rational, nested, and three-way cards.
  "breadth_products_clean_s40_79|formula_bilinear,formula_weak_centered,formula_trig_product|1024|100|0.00|40-79|24|256|500|2"
  "breadth_products_noise_s40_79|formula_bilinear,formula_weak_centered,formula_trig_product|1024|100|0.05|40-79|24|256|500|2"
  "breadth_rational_clean_s40_79|formula_rational_product,formula_division_mixed,formula_mixed_sparse|1024|100|0.00|40-79|24|256|500|2"
  "breadth_rational_noise_s40_79|formula_rational_product,formula_division_mixed,formula_mixed_sparse|1024|100|0.05|40-79|24|256|500|2"
  "breadth_stress_clean_s40_79|formula_nested_trig,formula_three_way_product,formula_sqrt_energy|1024|100|0.00|40-79|24|256|500|2"
  "breadth_stress_noise_s40_79|formula_nested_trig,formula_three_way_product,formula_sqrt_energy|1024|100|0.05|40-79|24|256|500|2"

  # RF capacity sensitivity: shallow vs deeper endpoint gates.
  "core_d100_b20_leaf1_s180_239|core_interaction_c025|1024|100|0.00|180-239|20|256|700|1"
  "core_d100_b20_leaf5_s180_239|core_interaction_c025|1024|100|0.00|180-239|20|256|700|5"
  "mixed_d100_b24_leaf1_s80_119|formula_mixed_sparse,formula_rational_product|1024|100|0.00|80-119|24|256|700|1"
  "mixed_d100_b24_leaf5_s80_119|formula_mixed_sparse,formula_rational_product|1024|100|0.00|80-119|24|256|700|5"
)

row="${ROWS[$SLURM_ARRAY_TASK_ID]}"
IFS='|' read -r label functions_csv samples dimension noise seeds gate_size verify_points rf_trees min_leaf <<< "$row"
IFS=',' read -r -a functions <<< "$functions_csv"

out_dir="results/revision/treegate_pair_screen_extended/${label}"

echo "[$(date -Is)] host=$(hostname) label=${label} functions=${functions_csv} seeds=${seeds}"
echo "[$(date -Is)] CPU-only TreeGate extended row; no GPU requested."

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
  --rf-min-samples-leaf "$min_leaf" \
  --n-jobs "${SLURM_CPUS_PER_TASK:-8}" \
  --batch-size 8192 \
  --pair-chunk-size 128 \
  --out-dir "$out_dir" \
  --label "$label" \
  --resume

echo "[$(date -Is)] done TreeGate extended ${label}"
