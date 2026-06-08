#!/usr/bin/env bash
set -euo pipefail

# Focused Great Lakes queue for public-vs-hidden split consistency.
#
# This submits only the jobs needed to materialize fresh hidden/heldout-style
# pyKAN claim-card rows and an official score refresh.  It intentionally does
# not submit scorergram, EPIM breadth, or cross-method standard sweeps.
#
# Run from Great Lakes project root:
#
#   cd /home/zicong/prism-kan
#   ACCOUNT=jaabell0 bash scripts/submit_claimtransfer_hidden_split_gl.sh
#
# To use engin1:
#
#   ACCOUNT=engin1 bash scripts/submit_claimtransfer_hidden_split_gl.sh

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs/greatlakes

PY="${PYTHON_BIN:-$PWD/.venv/bin/python}"
ACCOUNT="${ACCOUNT:-jaabell0}"
STANDARD_ACCOUNT="${STANDARD_ACCOUNT:-$ACCOUNT}"
PARTITION="${PARTITION:-spgpu}"
SEED_BASE="${SEED_BASE:-4100}"
SEED_COUNT="${SEED_COUNT:-12}"
LABEL_SUFFIX="${LABEL_SUFFIX:-hidden_s${SEED_BASE}_$((SEED_BASE + SEED_COUNT - 1))}"
BATCH_SIZE="${BATCH_SIZE:-16384}"
PAIR_CHUNK_SIZE="${PAIR_CHUNK_SIZE:-1500}"

echo "[$(date -Is)] ClaimTransfer hidden split submit"
echo "[$(date -Is)] account=${ACCOUNT} standard_account=${STANDARD_ACCOUNT} partition=${PARTITION}"
echo "[$(date -Is)] python=${PY}"
echo "[$(date -Is)] seed_base=${SEED_BASE} seed_count=${SEED_COUNT} label_suffix=${LABEL_SUFFIX}"

submit() {
  echo
  echo "+ $*"
  "$@"
}

claim_job=$(sbatch --parsable --account="${ACCOUNT}" --partition="${PARTITION}" --array=0-11 \
  --export=ALL,PYTHON_BIN="${PY}",SEED_BASE="${SEED_BASE}",SEED_COUNT="${SEED_COUNT}",LABEL_SUFFIX="${LABEL_SUFFIX}",BATCH_SIZE="${BATCH_SIZE}",PAIR_CHUNK_SIZE="${PAIR_CHUNK_SIZE}" \
  scripts/greatlakes_claimtransfer_hidden_claimcards_cuda.sbatch)
echo "submitted hidden claimcards: ${claim_job}"

score_job=$(sbatch --parsable --dependency=afterok:"${claim_job}" --account="${STANDARD_ACCOUNT}" \
  --export=ALL,PYTHON_BIN="${PY}",BUILD_FIGURE_SUMMARIES=1 \
  scripts/greatlakes_claimtransfer_score_refresh_standard.sbatch)
echo "submitted dependent score refresh: ${score_job}"

echo
echo "Check with:"
echo 'squeue -u $USER -o "%.18i %.12P %.10a %.30j %.2t %.10M %.10l %R"'
