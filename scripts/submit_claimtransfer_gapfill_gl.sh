#!/usr/bin/env bash
set -euo pipefail

# Submit only the CPU gap-fill jobs referenced by coverage_gap_action_plan.csv.
#
# Run from the project root on Great Lakes:
#
#   ACCOUNT=jaabell0 bash scripts/submit_claimtransfer_gapfill_gl.sh
#
# Knobs:
#   SUBMIT_XFER=0       skip cross-method standard gapfill
#   SUBMIT_TREEGATE=0   skip TreeGate standard gapfill
#   SUBMIT_SYMEXPR=0    skip symbolic expression operator-recall diagnostic
#   SUBMIT_SCORE=0      skip dependent score refresh

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs/greatlakes

PY="${PYTHON_BIN:-$PWD/.venv/bin/python}"
ACCOUNT="${ACCOUNT:-${STANDARD_ACCOUNT:-jaabell0}}"
SUBMIT_XFER="${SUBMIT_XFER:-1}"
SUBMIT_TREEGATE="${SUBMIT_TREEGATE:-1}"
SUBMIT_SYMEXPR="${SUBMIT_SYMEXPR:-1}"
SUBMIT_SCORE="${SUBMIT_SCORE:-1}"

submit() {
  echo
  echo "+ $*"
  "$@"
}

echo "[$(date -Is)] ClaimTransfer gapfill submit account=${ACCOUNT} python=${PY}"

deps=()

if [[ "${SUBMIT_XFER}" == "1" ]]; then
  out=$(sbatch --parsable --account="${ACCOUNT}" \
    --export=ALL,PYTHON_BIN="${PY}",SEED_START="${GAP_XFER_SEED_START:-190}",SEED_STOP="${GAP_XFER_SEED_STOP:-219}",GA2M_SEED_STOP="${GAP_XFER_GA2M_SEED_STOP:-209}",SYMBOLIC_SEED_STOP="${GAP_XFER_SYMBOLIC_SEED_STOP:-209}" \
    scripts/greatlakes_cross_method_gapfill_standard.sbatch)
  echo "+ submitted xfer gapfill ${out}"
  deps+=("${out%%;*}")
fi

if [[ "${SUBMIT_TREEGATE}" == "1" ]]; then
  out=$(sbatch --parsable --account="${ACCOUNT}" --array=0-23 \
    --export=ALL,PYTHON_BIN="${PY}" \
    scripts/greatlakes_treegate_gapfill_standard.sbatch)
  echo "+ submitted treegate gapfill ${out}"
  deps+=("${out%%;*}")
fi

if [[ "${SUBMIT_SYMEXPR}" == "1" ]]; then
  out=$(sbatch --parsable --account="${ACCOUNT}" --array=0-3 \
    --export=ALL,PYTHON_BIN="${PY}" \
    scripts/greatlakes_symbolic_expression_operator_recall_standard.sbatch)
  echo "+ submitted symbolic expression diagnostic ${out}"
  deps+=("${out%%;*}")
fi

if [[ "${SUBMIT_SCORE}" == "1" ]]; then
  dep_arg=()
  if [[ "${#deps[@]}" -gt 0 ]]; then
    dep_arg=(--dependency="afterany:$(IFS=:; echo "${deps[*]}")")
  fi
  submit sbatch --account="${ACCOUNT}" "${dep_arg[@]}" \
    --export=ALL,PYTHON_BIN="${PY}",BUILD_FIGURE_SUMMARIES=1 \
    scripts/greatlakes_claimtransfer_score_refresh_standard.sbatch
fi

echo
echo "[$(date -Is)] gapfill submission done"
