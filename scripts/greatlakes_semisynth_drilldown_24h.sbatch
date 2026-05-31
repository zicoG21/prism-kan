#!/usr/bin/env bash
#SBATCH --job-name=kan-semi-drill
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

# Semi-synthetic drilldown for the evidence-geometry claim.
#
# This is a finite job list; the one-day SLURM limit is a generous guardrail,
# not the intended runtime.  The point is to break the previous aggregate
# semi-synthetic result into dataset x noise rows so the paper can show which
# covariate geometries drive KAN/readout versus residual-screen reversals.
#
# Suggested submit command from Great Lakes:
#
#   sbatch --account=jaabell0 \
#     --export=ALL,PYTHON_BIN=$PWD/.venv/bin/python \
#     --array=0-8%5 scripts/greatlakes_semisynth_drilldown_24h.sbatch

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs/greatlakes

PY="${PYTHON_BIN:-python}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-2}"

# label|dataset|noise|seed_offset
ROWS=(
  "breast_cancer_clean|breast_cancer|0.00|0"
  "diabetes_clean|diabetes|0.00|100"
  "wine_clean|wine|0.00|200"
  "breast_cancer_noise005|breast_cancer|0.05|300"
  "diabetes_noise005|diabetes|0.05|400"
  "wine_noise005|wine|0.05|500"
  "breast_cancer_noise010|breast_cancer|0.10|600"
  "diabetes_noise010|diabetes|0.10|700"
  "wine_noise010|wine|0.10|800"
)

row="${ROWS[$SLURM_ARRAY_TASK_ID]}"
IFS='|' read -r label dataset noise seed_offset <<< "$row"

seeds=()
for i in $(seq 0 14); do
  seeds+=("$((seed_offset + i))")
done

out_dir="results/revision/greatlakes_semisynthetic_drilldown/${label}"

echo "[$(date -Is)] running semi-synthetic drilldown ${label} on ${SLURM_JOB_NODELIST:-unknown-node}"
"$PY" experiments/run_semisynthetic_covariate_audit.py \
  --datasets "$dataset" \
  --coefficients 0.10 0.25 0.50 \
  --samples 128 256 384 512 \
  --test-samples 256 \
  --outer-seeds "${seeds[@]}" \
  --noise "$noise" \
  --methods feature_stability_var feature_edge_hybrid \
  --top-m 4 \
  --R 14 \
  --resample bootstrap \
  --subsample-frac 0.8 \
  --width-hidden 16 \
  --grid 5 \
  --k 3 \
  --lamb 0.001 \
  --probe-steps 75 \
  --device cuda \
  --out-dir "$out_dir"
echo "[$(date -Is)] done semi-synthetic drilldown ${label}"
