#!/usr/bin/env bash
#SBATCH --job-name=kan-resamp
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --output=logs/greatlakes/%x_%A_%a.out
#SBATCH --error=logs/greatlakes/%x_%A_%a.err

set -euo pipefail

# Great Lakes decomposition of the same-training-set bootstrap readout.
#
# Purpose:
#   Split the readout-stability signal into three probe perturbation modes:
#   no row resampling, bootstrap row resampling, and subsampling.  This answers
#   whether endpoint surfacing boundaries are mainly optimizer/readout geometry
#   or bootstrap-data perturbation effects under the current width-16 protocol.
#
# Submit after the current ANOVA-MC wave:
#   sbatch --account=jaabell0 --dependency=afterany:<ANOVA_MC_JOBID> \
#     --export=ALL,PYTHON_BIN=$PWD/.venv/bin/python \
#     --array=0-11%5 scripts/greatlakes_resampling_decomp_array.sbatch

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs/greatlakes

PY="${PYTHON_BIN:-python}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-2}"

# label|samples|noise|update_grid|resample
ROWS=(
  "clean_n512_none|512|0|0|none"
  "clean_n512_bootstrap|512|0|0|bootstrap"
  "clean_n512_subsample|512|0|0|subsample"
  "gridupdate_n512_none|512|0|1|none"
  "gridupdate_n512_bootstrap|512|0|1|bootstrap"
  "gridupdate_n512_subsample|512|0|1|subsample"
  "noise010_n512_none|512|0.10|0|none"
  "noise010_n512_bootstrap|512|0.10|0|bootstrap"
  "noise010_n512_subsample|512|0.10|0|subsample"
  "clean_n1024_none|1024|0|0|none"
  "clean_n1024_bootstrap|1024|0|0|bootstrap"
  "clean_n1024_subsample|1024|0|0|subsample"
)

row="${ROWS[$SLURM_ARRAY_TASK_ID]}"
IFS='|' read -r label samples noise update_grid resample <<< "$row"

out_dir="results/revision/greatlakes_resampling_decomp/${label}"
args=(
  experiments/run_same_data_kan_stability_probe.py
  --function core_interaction_c025
  --samples "$samples"
  --test_samples 2048
  --dimension 100
  --noise "$noise"
  --outer_seeds 3000 3001 3002 3003 3004 3005 3006 3007
  --R 10
  --resample "$resample"
  --subsample_frac 0.8
  --methods
    feature_stability_var
    edge_stability_var
    edge_endpoint_mass
    feature_edge_hybrid
    edge_pair_hybrid
  --top_m 4
  --skip_refit
  --width_hidden 16
  --grid 5
  --k 3
  --lamb 0.001
  --probe_steps 75
  --refit_steps 75
  --pred_batch_size 4096
  --device cuda
  --out_dir "$out_dir"
)

if [[ "$update_grid" == "1" ]]; then
  args+=(--update_grid --grid_update_num 5)
fi

echo "[$(date -Is)] running resampling decomposition ${label} on ${SLURM_JOB_NODELIST:-unknown-node}"
"$PY" "${args[@]}"
echo "[$(date -Is)] done resampling decomposition ${label}"
