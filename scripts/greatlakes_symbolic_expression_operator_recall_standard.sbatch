#!/usr/bin/env bash
#SBATCH --job-name=kan-symexpr
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/greatlakes/%x_%A_%a.out
#SBATCH --error=logs/greatlakes/%x_%A_%a.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs/greatlakes

PY="${PYTHON_BIN:-python}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

# label|case|variants|seeds
ROWS=(
  "energy_core|energy|oracle,missing_power,support_only,overcomplex|0-59"
  "gravity_core|gravity|oracle,missing_power,multiplicative_only,overcomplex|0-59"
  "coulomb_core|coulomb|oracle,missing_divide,variables_only,overcomplex|0-59"
  "damped_wave_core|damped_wave|oracle,missing_trig,support_only,overcomplex|0-59"
)

row="${ROWS[$SLURM_ARRAY_TASK_ID]}"
IFS='|' read -r label case_name variants_csv seeds <<< "$row"
IFS=',' read -r -a variants <<< "$variants_csv"

out_dir="results/revision/symbolic_expression_operator_recall/${label}"

echo "[$(date -Is)] running symbolic expression operator-recall ${label} on ${SLURM_JOB_NODELIST:-unknown-node}"
"$PY" scripts/run_symbolic_expression_operator_recall.py \
  --cases "$case_name" \
  --variants "${variants[@]}" \
  --seeds "$seeds" \
  --label "$label" \
  --out-dir "$out_dir"
echo "[$(date -Is)] done symbolic expression operator-recall ${label}"
