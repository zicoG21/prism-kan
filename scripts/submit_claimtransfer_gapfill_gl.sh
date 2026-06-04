#!/usr/bin/env bash
set -euo pipefail

# Submit only the coverage-gap work that is not already covered by the broad
# full-benchmark queue.  This is the preferred command after the older
# scorergram/claimcard/EPIM arrays are already in flight.
#
# Run from Great Lakes project root:
#
#   cd /home/zicong/prism-kan
#   git pull
#   bash scripts/submit_claimtransfer_gapfill_gl.sh

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs/greatlakes

PY="${PYTHON_BIN:-$PWD/.venv/bin/python}"
STANDARD_ACCOUNT="${STANDARD_ACCOUNT:-jaabell0}"
SUBMIT_XFER="${SUBMIT_XFER:-1}"
SUBMIT_TREEGATE="${SUBMIT_TREEGATE:-1}"
SUBMIT_SCORE_REFRESH="${SUBMIT_SCORE_REFRESH:-1}"

echo "[$(date -Is)] ClaimTransfer coverage-gap submit"
echo "[$(date -Is)] python=${PY}"
echo "[$(date -Is)] standard_account=${STANDARD_ACCOUNT}"
echo "[$(date -Is)] toggles: xfer=${SUBMIT_XFER} treegate=${SUBMIT_TREEGATE} refresh=${SUBMIT_SCORE_REFRESH}"

JOB_IDS=()

submit() {
  echo
  echo "+ $*"
  local output
  output="$("$@")"
  echo "$output"
  local job_id
  job_id="$(awk '/Submitted batch job/ {print $4}' <<< "$output" | tail -1)"
  if [[ -n "$job_id" ]]; then
    JOB_IDS+=("$job_id")
  fi
}

if [[ "${SUBMIT_XFER}" == "1" ]]; then
  submit sbatch --account="${STANDARD_ACCOUNT}" \
    --export=ALL,PYTHON_BIN="${PY}",SEED_START="${XFER_SEED_START:-160}",SEED_STOP="${XFER_SEED_STOP:-189}",GA2M_SEED_STOP="${XFER_GA2M_SEED_STOP:-179}",SYMBOLIC_SEED_STOP="${XFER_SYMBOLIC_SEED_STOP:-179}" \
    scripts/greatlakes_cross_method_gapfill_standard.sbatch
fi

if [[ "${SUBMIT_TREEGATE}" == "1" ]]; then
  submit sbatch --account="${STANDARD_ACCOUNT}" \
    --export=ALL,PYTHON_BIN="${PY}" \
    scripts/greatlakes_treegate_gapfill_standard.sbatch
fi

if [[ "${SUBMIT_SCORE_REFRESH}" == "1" ]]; then
  dep_args=()
  if [[ "${#JOB_IDS[@]}" -gt 0 ]]; then
    dep_args=(--dependency="afterany:$(IFS=:; echo "${JOB_IDS[*]}")")
  fi
  submit sbatch --account="${STANDARD_ACCOUNT}" \
    "${dep_args[@]}" \
    --export=ALL,PYTHON_BIN="${PY}",BUILD_FIGURE_SUMMARIES=1 \
    scripts/greatlakes_claimtransfer_score_refresh_standard.sbatch
fi

echo
echo "[$(date -Is)] submitted coverage-gap jobs."
if [[ "${#JOB_IDS[@]}" -gt 0 ]]; then
  echo "[$(date -Is)] job ids: ${JOB_IDS[*]}"
fi
echo 'Check with: squeue -u $USER -o "%.18i %.12P %.10a %.30j %.2t %.10M %.10l %R"'
