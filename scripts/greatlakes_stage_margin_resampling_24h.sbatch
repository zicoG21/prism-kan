#!/usr/bin/env bash
#SBATCH --job-name=kan-margin-resamp
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

# Extended resampling decomposition for the structural-margin claim.
#
# This is a finite job list.  The one-day SLURM walltime is intentionally
# generous; it is not a target cutoff.  The goal is to test whether endpoint
# margins are mostly fixed-data optimization/readout geometry or bootstrap row
# perturbation.
#
# Suggested submit command from Great Lakes:
#
#   sbatch --account=jaabell0 \
#     --export=ALL,PYTHON_BIN=$PWD/.venv/bin/python \
#     --array=0-14%5 scripts/greatlakes_stage_margin_resampling_24h.sbatch

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs/greatlakes

PY="${PYTHON_BIN:-python}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-2}"

# label|samples|width|noise|update_grid|resample|seed_offset
ROWS=(
  "clean_w16_n512_none|512|16|0.00|0|none|0"
  "clean_w16_n512_bootstrap|512|16|0.00|0|bootstrap|100"
  "clean_w16_n512_subsample|512|16|0.00|0|subsample|200"
  "clean_w16_n1024_none|1024|16|0.00|0|none|300"
  "clean_w16_n1024_bootstrap|1024|16|0.00|0|bootstrap|400"
  "grid_w16_n512_none|512|16|0.00|1|none|500"
  "grid_w16_n512_bootstrap|512|16|0.00|1|bootstrap|600"
  "grid_w16_n1024_none|1024|16|0.00|1|none|700"
  "grid_w16_n1024_bootstrap|1024|16|0.00|1|bootstrap|800"
  "noise010_w16_n512_none|512|16|0.10|0|none|900"
  "noise010_w16_n512_bootstrap|512|16|0.10|0|bootstrap|1000"
  "noise010_w16_n1024_none|1024|16|0.10|0|none|1100"
  "noise010_w16_n1024_bootstrap|1024|16|0.10|0|bootstrap|1200"
  "clean_w32_n768_none|768|32|0.00|0|none|1300"
  "clean_w32_n768_bootstrap|768|32|0.00|0|bootstrap|1400"
)

row="${ROWS[$SLURM_ARRAY_TASK_ID]}"
IFS='|' read -r label samples width noise update_grid resample seed_offset <<< "$row"

outer=()
for i in $(seq 0 11); do
  outer+=("$((seed_offset + i))")
done

out_dir="results/revision/greatlakes_stage_margin_resampling/${label}"

args=(
  experiments/run_same_data_kan_stability_probe.py
  --function core_interaction_c025
  --samples "$samples"
  --test_samples 2048
  --dimension 100
  --noise "$noise"
  --outer_seeds "${outer[@]}"
  --R 12
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
  --width_hidden "$width"
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

echo "[$(date -Is)] running stage-margin resampling ${label} on ${SLURM_JOB_NODELIST:-unknown-node}"
"$PY" "${args[@]}"
echo "[$(date -Is)] done stage-margin resampling ${label}"
