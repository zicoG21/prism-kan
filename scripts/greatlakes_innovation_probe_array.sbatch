#!/usr/bin/env bash
#SBATCH --job-name=kan-innov
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/greatlakes/%x_%A_%a.out
#SBATCH --error=logs/greatlakes/%x_%A_%a.err

set -euo pipefail

# Great Lakes exploratory innovation probe.
#
# Purpose:
#   Spend spare GPU cycles after the reviewer-facing closure jobs on searching
#   for a stronger contribution angle: readout-family disagreement patterns,
#   exploitable boundary regimes, and stress rows that create clearer stage
#   disagreement than the current core formula alone.
#
# Submit after the resampling-decomposition wave:
#   sbatch --account=jaabell0 --dependency=afterany:<RESAMP_JOBID> \
#     --export=ALL,PYTHON_BIN=$PWD/.venv/bin/python \
#     --array=0-2%3 scripts/greatlakes_innovation_probe_array.sbatch

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs/greatlakes

PY="${PYTHON_BIN:-python}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-2}"

# label|width|noise|update_grid|probe_steps|refit_steps|stress_dims|seed_offset
ROWS=(
  "clean_width16_family_search|16|0|0|75|100|500|0"
  "noise010_width16_family_search|16|0.10|0|75|100|500|100"
  "gridupdate_width16_family_search|16|0|1|75|100|500|200"
)

row="${ROWS[$SLURM_ARRAY_TASK_ID]}"
IFS='|' read -r label width noise update_grid probe_steps refit_steps stress_dims seed_offset <<< "$row"

out_dir="results/revision/greatlakes_innovation_probe/${label}"

broad_eval=()
broad_probe=()
exploit_eval=()
exploit_probe=()
stress_eval=()
stress_probe=()
for i in $(seq 0 7); do broad_eval+=("$((seed_offset + 100 + i))"); done
for i in $(seq 0 15); do broad_probe+=("$((seed_offset + 200 + i))"); done
for i in $(seq 0 7); do exploit_eval+=("$((seed_offset + 300 + i))"); done
for i in $(seq 0 15); do exploit_probe+=("$((seed_offset + 400 + i))"); done
for i in $(seq 0 5); do stress_eval+=("$((seed_offset + 500 + i))"); done
for i in $(seq 0 11); do stress_probe+=("$((seed_offset + 600 + i))"); done

args=(
  experiments/run_kan_native_innovation_loop.py
  --out_dir "$out_dir"
  --time_budget_hours 7.5
  --device cuda
  --noise "$noise"
  --methods
    grad_stability_var
    feature_stability_var
    edge_stability_var
    edge_endpoint_mass
    feature_edge_hybrid
    edge_pair_hybrid
  --broad_eval_seeds "${broad_eval[@]}"
  --broad_probe_seeds "${broad_probe[@]}"
  --exploit_eval_seeds "${exploit_eval[@]}"
  --exploit_probe_seeds "${exploit_probe[@]}"
  --stress_eval_seeds "${stress_eval[@]}"
  --stress_probe_seeds "${stress_probe[@]}"
  --stress_dimensions "$stress_dims"
  --width_hidden "$width"
  --grid 5
  --k 3
  --lamb 0.001
  --probe_steps "$probe_steps"
  --refit_steps "$refit_steps"
  --exploit_probe_steps "$probe_steps"
  --exploit_refit_steps "$refit_steps"
  --stress_probe_steps "$probe_steps"
  --fd_points 128
  --keep_top_pairs 160
)

if [[ "$update_grid" == "1" ]]; then
  args+=(--update_grid --grid_update_num 5)
else
  args+=(--no_update_grid)
fi

echo "[$(date -Is)] running innovation probe ${label} on ${SLURM_JOB_NODELIST:-unknown-node}"
"$PY" "${args[@]}"
echo "[$(date -Is)] done innovation probe ${label}"
