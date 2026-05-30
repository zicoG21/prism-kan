#!/usr/bin/env bash
#SBATCH --job-name=kan-semisyn
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=logs/greatlakes/%x_%A_%a.out
#SBATCH --error=logs/greatlakes/%x_%A_%a.err

set -euo pipefail

# Great Lakes semi-synthetic covariate disaggregation array.
#
# Submit via:
#   sbatch --account jaabell0 --array 0-2 scripts/greatlakes_semisynthetic_array.sbatch

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs/greatlakes

PY="${PYTHON_BIN:-python}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-2}"

DATASETS=(breast_cancer diabetes wine)
dataset="${DATASETS[$SLURM_ARRAY_TASK_ID]}"

out_dir="results/revision/greatlakes_semisynthetic/${dataset}"

echo "[$(date -Is)] running semisynthetic ${dataset} on ${SLURM_JOB_NODELIST:-unknown-node}"
"$PY" experiments/run_semisynthetic_covariate_audit.py \
  --datasets "$dataset" \
  --coefficients 0.10 0.25 0.50 \
  --samples 128 256 384 \
  --test-samples 512 \
  --outer-seeds 2000 2001 2002 2003 2004 2005 2006 2007 2008 2009 \
  --noise 0.0 \
  --methods feature_stability_var feature_edge_hybrid \
  --top-m 4 \
  --R 20 \
  --resample bootstrap \
  --width-hidden 16 \
  --grid 5 \
  --k 3 \
  --lamb 0.001 \
  --probe-steps 75 \
  --device cuda \
  --out-dir "$out_dir"
echo "[$(date -Is)] done semisynthetic ${dataset}"

