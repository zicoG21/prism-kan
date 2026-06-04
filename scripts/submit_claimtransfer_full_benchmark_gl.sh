#!/usr/bin/env bash
set -euo pipefail

# Submit the higher-value ClaimTransfer-Bench queue on Great Lakes.
#
# This is the "full benchmark" queue, not the workshop-simple quick path.
# It prioritizes:
#   1. fresh/private-seed claim cards;
#   2. heldout-style task cards;
#   3. scorer-indexed sensitivity with Hessian;
#   4. EPIM/TreeGate candidate-vs-verifier coverage;
#   5. official score/coverage refresh.
#
# Run from the project root on Great Lakes:
#
#   cd /home/zicong/prism-kan
#   bash scripts/submit_claimtransfer_full_benchmark_gl.sh
#
# Defaults submit to jaabell0 and avoid Slurm array throttles.  Override with:
#
#   ACCOUNT=engin1 bash scripts/submit_claimtransfer_full_benchmark_gl.sh
#   SUBMIT_GPU=0 bash scripts/submit_claimtransfer_full_benchmark_gl.sh
#   SUBMIT_STANDARD=0 bash scripts/submit_claimtransfer_full_benchmark_gl.sh
#
# Throughput knobs for A40/spgpu jobs can be overridden without editing scripts:
#
#   A40_BATCH_SIZE=24576 A40_PAIR_CHUNK_SIZE=2048 \
#     bash scripts/submit_claimtransfer_full_benchmark_gl.sh

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs/greatlakes

PY="${PYTHON_BIN:-$PWD/.venv/bin/python}"
ACCOUNT="${ACCOUNT:-jaabell0}"

SUBMIT_GPU="${SUBMIT_GPU:-1}"
SUBMIT_SPGPU="${SUBMIT_SPGPU:-1}"
SUBMIT_STANDARD="${SUBMIT_STANDARD:-1}"
SUBMIT_SCORE_REFRESH="${SUBMIT_SCORE_REFRESH:-1}"

echo "[$(date -Is)] ClaimTransfer full benchmark submit"
echo "[$(date -Is)] account=${ACCOUNT}"
echo "[$(date -Is)] python=${PY}"
echo "[$(date -Is)] submit_gpu=${SUBMIT_GPU} submit_spgpu=${SUBMIT_SPGPU} submit_standard=${SUBMIT_STANDARD} submit_score_refresh=${SUBMIT_SCORE_REFRESH}"

submit() {
  echo
  echo "+ $*"
  "$@"
}

if [[ "${SUBMIT_GPU}" == "1" ]]; then
  # V100/gpu fresh private-seed claim cards.  Useful when gpu partition frees up.
  submit sbatch --account="${ACCOUNT}" --array=0-11 \
    --export=ALL,PYTHON_BIN="${PY}",SEED_BASE="${GPU_SEED_BASE:-2000}",SEED_COUNT="${GPU_SEED_COUNT:-12}",LABEL_SUFFIX="${GPU_LABEL_SUFFIX:-s2000_2011_gpu}" \
    scripts/greatlakes_claimtransfer_hidden_claimcards_cuda.sbatch
fi

if [[ "${SUBMIT_SPGPU}" == "1" ]]; then
  # A40/spgpu fresh private-seed claim cards.  Command-line partition overrides
  # the script's default gpu partition.
  submit sbatch --account="${ACCOUNT}" --partition=spgpu --array=0-11 \
    --export=ALL,PYTHON_BIN="${PY}",SEED_BASE="${A40_SEED_BASE:-2300}",SEED_COUNT="${A40_SEED_COUNT:-12}",LABEL_SUFFIX="${A40_LABEL_SUFFIX:-s2300_2311_a40}",BATCH_SIZE="${A40_BATCH_SIZE:-16384}",PAIR_CHUNK_SIZE="${A40_PAIR_CHUNK_SIZE:-1500}" \
    scripts/greatlakes_claimtransfer_hidden_claimcards_cuda.sbatch

  # Scorer-indexed grammar with Hessian included.  This is slow but directly
  # addresses the "scorer arbitrary" benchmark concern.
  submit sbatch --account="${ACCOUNT}" --array=0-9 \
    --export=ALL,PYTHON_BIN="${PY}",SEED_BASE="${SCORER_SEED_BASE:-1600}",SEED_COUNT="${SCORER_SEED_COUNT:-8}",LABEL_SUFFIX="${SCORER_LABEL_SUFFIX:-s1600_1607_hess_full}",SCORERS_COLON=epim:anova_abs:fd:hessian:hybrid_epim_anova,BATCH_SIZE="${A40_BATCH_SIZE:-16384}",PAIR_CHUNK_SIZE="${SCORER_PAIR_CHUNK_SIZE:-1024}" \
    scripts/greatlakes_spgpu_pair_scorer_grammar_a40.sbatch

  # EPIM propose / ANOVA verify breadth on fresh seeds.  This supports the
  # candidate-vs-verifier benchmark story.
  submit sbatch --account="${ACCOUNT}" --array=0-7 \
    --export=ALL,PYTHON_BIN="${PY}",SEEDS="${EPIM_SEEDS:-600-629}",LABEL_SUFFIX="${EPIM_LABEL_SUFFIX:-s600_629_full}",BATCH_SIZE="${A40_BATCH_SIZE:-16384}",PAIR_CHUNK_SIZE="${EPIM_PAIR_CHUNK_SIZE:-768}" \
    scripts/greatlakes_spgpu_epim_pairverify_breadth_a40.sbatch
fi

if [[ "${SUBMIT_STANDARD}" == "1" ]]; then
  # CPU-only cross-adapter coverage and TreeGate candidate screens.  These do
  # not consume GPU quota.
  submit sbatch --account="${ACCOUNT}" \
    --export=ALL,PYTHON_BIN="${PY}",SEED_START="${XFER_SEED_START:-70}",SEED_STOP="${XFER_SEED_STOP:-99}",GA2M_SEED_STOP="${XFER_GA2M_SEED_STOP:-89}",SYMBOLIC_SEED_STOP="${XFER_SYMBOLIC_SEED_STOP:-89}" \
    scripts/greatlakes_cross_method_transfer_baselines_extended.sbatch

  submit sbatch --account="${ACCOUNT}" --array=0-23 \
    --export=ALL,PYTHON_BIN="${PY}" \
    scripts/greatlakes_treegate_pair_screen_extended_standard.sbatch
fi

if [[ "${SUBMIT_SCORE_REFRESH}" == "1" ]]; then
  submit sbatch --account="${ACCOUNT}" \
    --export=ALL,PYTHON_BIN="${PY}",BUILD_FIGURE_SUMMARIES=1 \
    scripts/greatlakes_claimtransfer_score_refresh_standard.sbatch
fi

echo
echo "[$(date -Is)] submitted. Check with:"
echo 'squeue -u $USER -o "%.18i %.12P %.10a %.30j %.2t %.10M %.10l %R"'
