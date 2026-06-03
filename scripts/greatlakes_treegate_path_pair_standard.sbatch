#!/usr/bin/env bash
#SBATCH --job-name=kan-treepath
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/greatlakes/%x_%A_%a.out
#SBATCH --error=logs/greatlakes/%x_%A_%a.err

set -euo pipefail

# CPU-only TreePath gate:
#   tree path co-occurrence -> any-pair endpoint claim -> gated pair verifier.
#
# This follows the literature direction behind iRF/RIT/RuleFit more closely
# than ordinary RF feature importance: variables earn endpoint credit when they
# repeatedly co-occur on tree paths, and top path pairs can be admitted directly
# to the verifier candidate set.

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs/greatlakes

PY="${PYTHON_BIN:-$PWD/.venv/bin/python}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-8}"

# label|functions_csv|samples|dimension|noise|seeds|gate_size|verify_points|rf_trees|forest_type|gate_score|direct_pair_budget|min_leaf
ROWS=(
  # Directly compare RF vs ExtraTrees path gates on the weak centered pressure test.
  "core_rf_pathpair_b20_s0_59|core_interaction_c025|1024|100|0.00|0-59|20|256|700|rf|path_pair_endpoint|200|2"
  "core_extra_pathpair_b20_s0_59|core_interaction_c025|1024|100|0.00|0-59|20|256|700|extra|path_pair_endpoint|200|2"
  "core_extra_hybrid_b20_s0_59|core_interaction_c025|1024|100|0.00|0-59|20|256|700|extra|hybrid|200|2"
  "core_extra_pathpair_b40_s0_59|core_interaction_c025|1024|100|0.00|0-59|40|256|700|extra|path_pair_endpoint|400|2"

  # High-dimensional TreeGate: can path co-occurrence reduce all-pairs search?
  "core_d500_extra_path_b40_s0_39|core_interaction_c025|2048|500|0.00|0-39|40|192|900|extra|path_pair_endpoint|600|2"
  "core_d1000_extra_path_b50_s0_29|core_interaction_c025|2048|1000|0.00|0-29|50|192|900|extra|path_pair_endpoint|800|2"
  "core_d1000_extra_hybrid_b50_s0_29|core_interaction_c025|2048|1000|0.00|0-29|50|192|900|extra|hybrid|800|2"

  # Product/rational breadth where tree gates should have positive controls.
  "products_extra_path_s0_39|formula_bilinear,formula_weak_centered,formula_trig_product|1024|100|0.00|0-39|24|256|700|extra|path_pair_endpoint|300|2"
  "rational_extra_path_s0_39|formula_rational_product,formula_division_mixed,formula_mixed_sparse|1024|100|0.00|0-39|24|256|700|extra|path_pair_endpoint|300|2"
  "stress_extra_path_s0_39|formula_nested_trig,formula_three_way_product,formula_sqrt_energy|1024|100|0.00|0-39|24|256|700|extra|path_pair_endpoint|300|2"

  # Noise checks.
  "core_noise005_extra_path_b20_s0_59|core_interaction_c025|1024|100|0.05|0-59|20|256|700|extra|path_pair_endpoint|200|2"
  "rational_noise_extra_path_s0_39|formula_rational_product,formula_division_mixed,formula_mixed_sparse|1024|100|0.05|0-39|24|256|700|extra|path_pair_endpoint|300|2"
)

row="${ROWS[$SLURM_ARRAY_TASK_ID]}"
IFS='|' read -r label functions_csv samples dimension noise seeds gate_size verify_points rf_trees forest_type gate_score direct_pair_budget min_leaf <<< "$row"
IFS=',' read -r -a functions <<< "$functions_csv"

out_dir="results/revision/treegate_path_pair_screen/${label}"

echo "[$(date -Is)] host=$(hostname) label=${label} functions=${functions_csv} seeds=${seeds}"
echo "[$(date -Is)] CPU-only tree path co-occurrence gate; no GPU requested."

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
  --forest-type "$forest_type" \
  --gate-score "$gate_score" \
  --direct-pair-budget "$direct_pair_budget" \
  --rf-min-samples-leaf "$min_leaf" \
  --n-jobs "${SLURM_CPUS_PER_TASK:-8}" \
  --batch-size 8192 \
  --pair-chunk-size 128 \
  --out-dir "$out_dir" \
  --label "$label" \
  --resume

echo "[$(date -Is)] done TreePath gate ${label}"
