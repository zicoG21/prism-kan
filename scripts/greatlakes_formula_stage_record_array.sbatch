#!/usr/bin/env bash
#SBATCH --job-name=kan-formula-sdr
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

# Formula-family stage-discordance expansion.
#
# Purpose:
#   Move the paper from a core-task audit to a reusable stage-record benchmark.
#   Each array task runs the same evidence objects for one formula family:
#     1. full-KAN all-pairs functional-ANOVA reliance;
#     2. exposed pyKAN readout endpoint surfacing across readout families.
#
# Submit from the repo root, e.g.
#   sbatch --account=jaabell0 \
#     --export=ALL,PYTHON_BIN=$PWD/.venv/bin/python \
#     --array=0-6%5 scripts/greatlakes_formula_stage_record_array.sbatch
#
# After completion:
#   $PWD/.venv/bin/python experiments/make_formula_stage_record_summary.py

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs/greatlakes

PY="${PYTHON_BIN:-python}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-2}"

# label|function|ontology|dimension|samples|noise|update_grid|width|steps|seeds|anova_points|anova_background
ROWS=(
  "bilinear_d50_w16_n1024|formula_bilinear|simple_pair|50|1024|0|0|16|75|0-29|24|24"
  "weak_centered_d50_w16_n1024|formula_weak_centered|weak_centered_pair|50|1024|0|0|16|75|0-29|24|24"
  "trig_product_d50_w16_n1024|formula_trig_product|nonlinear_pair|50|1024|0|0|16|75|0-29|24|24"
  "exp_product_d50_w16_n1024|formula_exp_product|nonlinear_pair|50|1024|0|0|16|75|0-29|24|24"
  "log_product_d50_w16_n1024|formula_log_product|nonlinear_pair|50|1024|0|0|16|75|0-29|24|24"
  "sqrt_energy_d50_w16_n1024|formula_sqrt_energy|nonlinear_pair|50|1024|0|0|16|75|0-29|24|24"
  "nested_trig_d50_w16_n1024|formula_nested_trig|nested_pair_label|50|1024|0|0|16|75|0-29|24|24"
)

row="${ROWS[$SLURM_ARRAY_TASK_ID]}"
IFS='|' read -r label function ontology dimension samples noise update_grid width steps seeds anova_points anova_background <<< "$row"

root="results/revision/formula_stage_record/${label}"
full_dir="${root}/full"
readout_dir="${root}/readout"
manifest="${root}/stage_record_manifest.txt"
mkdir -p "$root"
{
  echo "label=${label}"
  echo "function=${function}"
  echo "ontology=${ontology}"
  echo "dimension=${dimension}"
  echo "samples=${samples}"
  echo "noise=${noise}"
  echo "update_grid=${update_grid}"
  echo "width=${width}"
  echo "steps=${steps}"
  echo "seeds=${seeds}"
  echo "anova_points=${anova_points}"
  echo "anova_background=${anova_background}"
} > "$manifest"

readout_seeds=()
IFS='-' read -r seed_lo seed_hi <<< "$seeds"
for s in $(seq "$seed_lo" "$seed_hi"); do
  readout_seeds+=("$s")
done

full_args=(
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
  --out-dir "$full_dir"
)

readout_args=(
  experiments/run_kan_probe_sensitivity.py
  --out_dir "$readout_dir"
  --function "$function"
  --samples "$samples"
  --dimension "$dimension"
  --test_samples 2048
  --noise "$noise"
  --seeds "${readout_seeds[@]}"
  --methods
    feature_stability_var
    edge_stability_var
    edge_endpoint_mass
    feature_edge_hybrid
    edge_pair_hybrid
  --top_ms 3 4 6 10
  --width_hidden "$width"
  --grid 5
  --k 3
  --lamb 0.001
  --probe_steps "$steps"
  --probe_variable_points 512
  --pred_batch_size 4096
  --device cuda
)

if [[ "$update_grid" == "1" ]]; then
  full_args+=(--update-grid --grid-update-num 5)
  readout_args+=(--update_grid --grid_update_num 5)
fi

echo "[$(date -Is)] running formula SDR full-KAN ${label} on ${SLURM_JOB_NODELIST:-unknown-node}"
"$PY" "${full_args[@]}"
echo "[$(date -Is)] running formula SDR readout ${label} on ${SLURM_JOB_NODELIST:-unknown-node}"
"$PY" "${readout_args[@]}"
echo "[$(date -Is)] done formula SDR ${label}"
