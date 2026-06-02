#!/usr/bin/env bash
set -euo pipefail

# Extra high-value Great Lakes queue after the base ClaimTransfer submission.
# This intentionally avoids more plain claimcard jobs. It targets the remaining
# reviewer hard points:
#   1. method-agnostic cross-adapter coverage,
#   2. scorer-indexed grammar sensitivity with Hessian enabled,
#   3. EPIM proposal-vs-verifier breadth on fresh seeds.
#
# Run from the project root on Great Lakes:
#
#   cd /home/zicong/prism-kan
#   bash scripts/submit_jaabell_claimtransfer_extra_gl.sh

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs/greatlakes

PY="${PYTHON_BIN:-$PWD/.venv/bin/python}"
ACCOUNT="${ACCOUNT:-jaabell0}"

echo "[$(date -Is)] submitting extra ClaimTransfer queue with account=${ACCOUNT}"
echo "[$(date -Is)] python=${PY}"

submit() {
  echo
  echo "+ $*"
  "$@"
}

# CPU/standard: larger method-agnostic adapter coverage with consistent schema.
submit sbatch --account="${ACCOUNT}" \
  --export=ALL,PYTHON_BIN="${PY}",SEED_START=40,SEED_STOP=69,GA2M_SEED_STOP=59,SYMBOLIC_SEED_STOP=59 \
  scripts/greatlakes_cross_method_transfer_baselines_extended.sbatch

# A40/spgpu: scorer-indexed grammar with Hessian included.  This is slower than
# the default scorergram job, but directly supports the "official scorer"
# novelty claim.
submit sbatch --account="${ACCOUNT}" \
  --array=0-9 \
  --export=ALL,PYTHON_BIN="${PY}",SEED_BASE=1200,SEED_COUNT=8,LABEL_SUFFIX=s1200_1207_hess_jb,SCORERS=epim,anova_abs,fd,hessian,hybrid_epim_anova \
  scripts/greatlakes_spgpu_pair_scorer_grammar_a40.sbatch

# A40/spgpu: EPIM breadth on fresh seeds.  Proposal-vs-verifier evidence is more
# novelty-relevant than another plain pyKAN claimcard seed block.
submit sbatch --account="${ACCOUNT}" \
  --array=0-7 \
  --export=ALL,PYTHON_BIN="${PY}",SEEDS=300-329,LABEL_SUFFIX=s300_329_jb \
  scripts/greatlakes_spgpu_epim_pairverify_breadth_a40.sbatch

echo
echo "[$(date -Is)] submitted extra queue. Check with:"
echo 'squeue -u $USER -o "%.18i %.12P %.10a %.30j %.2t %.10M %.10l %R"'
