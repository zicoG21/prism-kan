#!/usr/bin/env bash
#SBATCH --job-name=kan-taxonomy
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/greatlakes/%x_%A_%a.out
#SBATCH --error=logs/greatlakes/%x_%A_%a.err

set -euo pipefail

# Great Lakes readout-taxonomy audit.
#
# Purpose: make the exposed pyKAN readout family explicit rather than relying
# only on the KAN-FE hybrid column.  Each row trains/resumes 30 probes and
# summarizes all simple readout families:
#   grad, feature, edge, endpoint mass, edge-pair endpoint, FE hybrid,
#   edge-pair hybrid.
#
# Submit after current jobs:
#   sbatch --account=jaabell0 --dependency=afterany:<DEP> \
#     --export=ALL,PYTHON_BIN=$PWD/.venv/bin/python \
#     --array=0-8%5 scripts/greatlakes_readout_taxonomy_array.sbatch

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs/greatlakes

PY="${PYTHON_BIN:-python}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-2}"

# label|samples|noise|update_grid|seed_start|seed_stop
ROWS=(
  "clean_w16_n512|512|0|0|1800|1829"
  "clean_w16_n768|768|0|0|1800|1829"
  "clean_w16_n1024|1024|0|0|1800|1829"
  "gridupdate_w16_n512|512|0|1|1800|1829"
  "gridupdate_w16_n768|768|0|1|1800|1829"
  "gridupdate_w16_n1024|1024|0|1|1800|1829"
  "noise005_w16_n512|512|0.05|0|1800|1829"
  "noise010_w16_n512|512|0.10|0|1800|1829"
  "noise010_w16_n1024|1024|0.10|0|1800|1829"
)

row="${ROWS[$SLURM_ARRAY_TASK_ID]}"
IFS='|' read -r label samples noise update_grid seed_start seed_stop <<< "$row"

seeds=()
for s in $(seq "$seed_start" "$seed_stop"); do
  seeds+=("$s")
done

out_dir="results/revision/greatlakes_readout_taxonomy/${label}"
args=(
  experiments/run_kan_probe_sensitivity.py
  --out_dir "$out_dir"
  --function core_interaction_c025
  --samples "$samples"
  --dimension 100
  --test_samples 2048
  --noise "$noise"
  --seeds "${seeds[@]}"
  --methods
    grad_stability_var
    feature_stability_var
    edge_stability_var
    edge_endpoint_mass
    edge_pair_endpoint
    feature_edge_hybrid
    edge_pair_hybrid
  --top_ms 4 6 10 20
  --width_hidden 16
  --grid 5
  --k 3
  --lamb 0.001
  --probe_steps 75
  --probe_variable_points 512
  --pred_batch_size 4096
  --device cuda
)

if [[ "$update_grid" == "1" ]]; then
  args+=(--update_grid --grid_update_num 5)
fi

echo "[$(date -Is)] running readout-taxonomy ${label} on ${SLURM_JOB_NODELIST:-unknown-node}"
"$PY" "${args[@]}"
echo "[$(date -Is)] done readout-taxonomy ${label}"
