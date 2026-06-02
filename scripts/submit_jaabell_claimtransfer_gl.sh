#!/usr/bin/env bash
set -euo pipefail

# Submit the current high-value ClaimTransfer-Bench hard-evidence queue on
# Great Lakes under jaabell0.
#
# Design:
#   - no explicit array throttles; Slurm account/GRES limits decide concurrency
#   - non-overlapping seed blocks relative to earlier s300/s500/s700/s800 waves
#   - each job family targets a reviewer hard point:
#       * claimcard: seed-level statistical precision for typed claims
#       * epim: proposal-vs-verifier separation
#       * scorer grammar: scorer-indexed claim grammar
#       * cross-method: method-agnostic adapter coverage
#
# Run from the project root on Great Lakes:
#
#   cd /home/zicong/prism-kan
#   bash scripts/submit_jaabell_claimtransfer_gl.sh

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs/greatlakes

PY="${PYTHON_BIN:-$PWD/.venv/bin/python}"
ACCOUNT="${ACCOUNT:-jaabell0}"

echo "[$(date -Is)] submitting ClaimTransfer-Bench queue with account=${ACCOUNT}"
echo "[$(date -Is)] python=${PY}"

submit() {
  echo
  echo "+ $*"
  "$@"
}

# A40/spgpu: seed-level Claim Provenance Record expansion for pyKAN.
submit sbatch --account="${ACCOUNT}" \
  --array=0-9 \
  --export=ALL,PYTHON_BIN="${PY}",SEED_BASE=900,LABEL_SUFFIX=s900_915_jb \
  scripts/greatlakes_spgpu_claimcard_followup_a40.sbatch

# A40/spgpu: EPIM proposal-vs-verifier diagnostics.  This backs the
# "proposal evidence is not verifier evidence" claim.
submit sbatch --account="${ACCOUNT}" \
  --array=0-7 \
  --export=ALL,PYTHON_BIN="${PY}",SEED_BASE=270,LABEL_SUFFIX=s270_299_jb \
  scripts/greatlakes_spgpu_epim_diagnostic_followup_a40.sbatch

# A40/spgpu: scorer-indexed claim grammar.  This is the main novelty-facing
# follow-up: the same fitted workflow is evaluated by multiple declared pair
# scorers.
submit sbatch --account="${ACCOUNT}" \
  --array=0-9 \
  --export=ALL,PYTHON_BIN="${PY}",SEED_BASE=900,SEED_COUNT=12,LABEL_SUFFIX=s900_911_jb \
  scripts/greatlakes_spgpu_pair_scorer_grammar_a40.sbatch

# V100/gpu: same claimcard family on the gpu partition.  This is useful when
# V100s are open and gives a non-A40 seed block for the same claim grammar.
submit sbatch --account="${ACCOUNT}" \
  --array=0-9 \
  --export=ALL,PYTHON_BIN="${PY}",SEED_BASE=920,LABEL_SUFFIX=s920_935_gpu_jb \
  scripts/greatlakes_gpu_claimcard_followup.sbatch

# CPU/standard: method-agnostic adapter coverage.  This fills the reviewer
# concern that the benchmark is too pyKAN-specific.
submit sbatch --account="${ACCOUNT}" \
  --export=ALL,PYTHON_BIN="${PY}" \
  scripts/greatlakes_cross_method_transfer_baselines.sbatch

echo
echo "[$(date -Is)] submitted. Check with:"
echo 'squeue -u $USER -o "%.18i %.12P %.10a %.30j %.2t %.10M %.10l %R"'
