#!/usr/bin/env bash
set -euo pipefail

# Submit the remaining pyKAN P1 coverage gapfill and a dependent score refresh.
#
# Run from Great Lakes project root:
#
#   cd /home/zicong/prism-kan
#   git pull
#   bash scripts/submit_claimtransfer_pykan_gapfill_gl.sh

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs/greatlakes

PY="${PYTHON_BIN:-$PWD/.venv/bin/python}"
ACCOUNT="${ACCOUNT:-jaabell0}"
PARTITION="${PARTITION:-spgpu}"
SEED_BASE="${SEED_BASE:-5200}"
SEED_COUNT="${SEED_COUNT:-8}"
SEED_END="$((SEED_BASE + SEED_COUNT - 1))"
LABEL_SUFFIX="${LABEL_SUFFIX:-s${SEED_BASE}_${SEED_END}_pykangap}"

echo "[$(date -Is)] submitting pyKAN gapfill"
echo "[$(date -Is)] account=${ACCOUNT} partition=${PARTITION} seeds=${SEED_BASE}-${SEED_END}"

out="$(sbatch --account="${ACCOUNT}" --partition="${PARTITION}" --array=0-1 \
  --export=ALL,PYTHON_BIN="${PY}",SEED_BASE="${SEED_BASE}",SEED_COUNT="${SEED_COUNT}",LABEL_SUFFIX="${LABEL_SUFFIX}" \
  scripts/greatlakes_pykan_gapfill_a40.sbatch)"
echo "$out"
job_id="$(awk '/Submitted batch job/ {print $4}' <<< "$out" | tail -1)"

dep_args=()
if [[ -n "${job_id}" ]]; then
  dep_args=(--dependency="afterany:${job_id}")
fi

out2="$(sbatch --account="${ACCOUNT}" "${dep_args[@]}" \
  --export=ALL,PYTHON_BIN="${PY}",BUILD_FIGURE_SUMMARIES=1 \
  scripts/greatlakes_claimtransfer_score_refresh_standard.sbatch)"
echo "$out2"

echo 'Check with: squeue -u $USER -o "%.18i %.12P %.10a %.30j %.2t %.10M %.10l %R"'
