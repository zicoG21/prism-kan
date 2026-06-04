#!/usr/bin/env bash
#SBATCH --job-name=kan-treegap
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-6
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/greatlakes/%x_%A_%a.out
#SBATCH --error=logs/greatlakes/%x_%A_%a.err

set -euo pipefail

# CPU-only TreeGate gap fill for public task families absent from the original
# extended TreeGate queue.  This produces candidate_pair and verifier rows; it
# is intentionally separate from cross-method GBM-H rows.

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs/greatlakes

PY="${PYTHON_BIN:-$PWD/.venv/bin/python}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-8}"

# label|functions_csv|samples|dimension|noise|seeds|gate_size|verify_points|rf_trees|min_leaf
ROWS=(
  "gap_trig_exp_log_s160_199|formula_trig_product,formula_exp_product,formula_log_product|1024|100|0.00|160-199|24|256|500|2"
  "gap_sqrt_threeway_s160_199|formula_sqrt_energy,formula_three_way_product|1024|100|0.00|160-199|24|256|500|2"
  "gap_scientific_energy_s160_199|feynman_energy|2048|100|0.00|160-199|24|256|700|2"
  "gap_scientific_gravity_s160_179|feynman_gravity|2048|100|0.00|160-179|30|256|700|2"
  "gap_scientific_coulomb_s160_179|feynman_coulomb|2048|100|0.00|160-179|30|256|700|2"
  "gap_scientific_damped_wave_s160_179|feynman_damped_wave|2048|100|0.00|160-179|30|256|700|2"
  "gap_semisynthetic_s160_199|semisynthetic_breast_cancer,semisynthetic_diabetes,semisynthetic_wine|1024|30|0.00|160-199|24|256|700|2"
)

row="${ROWS[$SLURM_ARRAY_TASK_ID]}"
IFS='|' read -r label functions_csv samples dimension noise seeds gate_size verify_points rf_trees min_leaf <<< "$row"
IFS=',' read -r -a functions <<< "$functions_csv"

out_dir="results/revision/treegate_pair_screen_gapfill/${label}"

echo "[$(date -Is)] host=$(hostname) label=${label} functions=${functions_csv} seeds=${seeds}"
echo "[$(date -Is)] CPU-only TreeGate gap fill; no GPU requested."

"$PY" experiments/run_treegate_pair_screen.py \
  --functions "${functions[@]}" \
  --samples "$samples" \
  --test-samples 2048 \
  --dimension "$dimension" \
  --noise "$noise" \
  --semisynthetic-c "${SEMISYNTHETIC_C:-0.25}" \
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

echo "[$(date -Is)] done TreeGate gap fill ${label}"
