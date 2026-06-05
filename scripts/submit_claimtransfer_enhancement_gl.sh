#!/usr/bin/env bash
set -euo pipefail

# Submit optional ClaimTransfer-Bench enhancement jobs on Great Lakes.
#
# These jobs are not needed for P0/P1/P2 alpha readiness.  They are useful for
# strengthening the mature benchmark artifact:
#   - expression-level symbolic operator recall controls;
#   - scorer-sensitivity breadth;
#   - EPIM/TreeGate candidate-vs-verifier breadth;
#   - semi-synthetic covariate drilldowns;
#   - pruning/symbolic stress rows;
#   - readout taxonomy rows;
#   - official score refresh after results are merged.
#
# Run from the project root on Great Lakes:
#
#   cd /home/zicong/prism-kan
#   bash scripts/submit_claimtransfer_enhancement_gl.sh
#
# Common overrides:
#
#   ACCOUNT=jaabell0 STANDARD_ACCOUNT=jaabell0 bash scripts/submit_claimtransfer_enhancement_gl.sh
#   ACCOUNT=engin1 SUBMIT_GPU=0 SUBMIT_SPGPU=1 SUBMIT_STANDARD=0 bash scripts/submit_claimtransfer_enhancement_gl.sh
#   SUBMIT_PYKAN_GPU=0 SUBMIT_STANDARD=1 bash scripts/submit_claimtransfer_enhancement_gl.sh

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs/greatlakes

PY="${PYTHON_BIN:-$PWD/.venv/bin/python}"
ACCOUNT="${ACCOUNT:-jaabell0}"
STANDARD_ACCOUNT="${STANDARD_ACCOUNT:-jaabell0}"

SUBMIT_STANDARD="${SUBMIT_STANDARD:-1}"
SUBMIT_SPGPU="${SUBMIT_SPGPU:-1}"
SUBMIT_GPU="${SUBMIT_GPU:-1}"
SUBMIT_PYKAN_GPU="${SUBMIT_PYKAN_GPU:-1}"
SUBMIT_SCORE_REFRESH="${SUBMIT_SCORE_REFRESH:-1}"

echo "[$(date -Is)] ClaimTransfer enhancement submit"
echo "[$(date -Is)] account=${ACCOUNT}"
echo "[$(date -Is)] standard_account=${STANDARD_ACCOUNT}"
echo "[$(date -Is)] python=${PY}"
echo "[$(date -Is)] submit_standard=${SUBMIT_STANDARD} submit_spgpu=${SUBMIT_SPGPU} submit_gpu=${SUBMIT_GPU} submit_pykan_gpu=${SUBMIT_PYKAN_GPU} submit_score_refresh=${SUBMIT_SCORE_REFRESH}"

submit() {
  echo
  echo "+ $*"
  "$@"
}

if [[ "${SUBMIT_STANDARD}" == "1" ]]; then
  # Fills / strengthens the optional scientific-expression operator-recall
  # track.  CPU-only; cheap; should usually be submitted first.
  submit sbatch --account="${STANDARD_ACCOUNT}" --array=0-3 \
    --export=ALL,PYTHON_BIN="${PY}" \
    scripts/greatlakes_symbolic_expression_operator_recall_standard.sbatch

  # CPU-only non-KAN gap/portability rows.
  submit sbatch --account="${STANDARD_ACCOUNT}" \
    --export=ALL,PYTHON_BIN="${PY}",SEED_START="${XFER_SEED_START:-190}",SEED_STOP="${XFER_SEED_STOP:-219}",GA2M_SEED_STOP="${XFER_GA2M_SEED_STOP:-209}",SYMBOLIC_SEED_STOP="${XFER_SYMBOLIC_SEED_STOP:-209}" \
    scripts/greatlakes_cross_method_gapfill_standard.sbatch

  submit sbatch --account="${STANDARD_ACCOUNT}" --array=0-23 \
    --export=ALL,PYTHON_BIN="${PY}" \
    scripts/greatlakes_treegate_gapfill_standard.sbatch

  submit sbatch --account="${STANDARD_ACCOUNT}" --array=0-23 \
    --export=ALL,PYTHON_BIN="${PY}" \
    scripts/greatlakes_treegate_pair_screen_extended_standard.sbatch
fi

if [[ "${SUBMIT_SPGPU}" == "1" ]]; then
  # A40 scorer-sensitivity rows with Hessian included.
  submit sbatch --account="${ACCOUNT}" --partition=spgpu --array=0-9 \
    --export=ALL,PYTHON_BIN="${PY}",SEED_BASE="${SCORER_SEED_BASE:-2400}",SEED_COUNT="${SCORER_SEED_COUNT:-8}",LABEL_SUFFIX="${SCORER_LABEL_SUFFIX:-s2400_2407_enhance}",SCORERS_COLON=epim:anova_abs:fd:hessian:hybrid_epim_anova,BATCH_SIZE="${A40_BATCH_SIZE:-16384}",PAIR_CHUNK_SIZE="${SCORER_PAIR_CHUNK_SIZE:-1024}" \
    scripts/greatlakes_spgpu_pair_scorer_grammar_a40.sbatch

  # A40 EPIM propose / ANOVA verify breadth.
  submit sbatch --account="${ACCOUNT}" --partition=spgpu --array=0-7 \
    --export=ALL,PYTHON_BIN="${PY}",SEEDS="${EPIM_SEEDS:-900-929}",LABEL_SUFFIX="${EPIM_LABEL_SUFFIX:-s900_929_enhance}",BATCH_SIZE="${A40_BATCH_SIZE:-16384}",PAIR_CHUNK_SIZE="${EPIM_PAIR_CHUNK_SIZE:-768}" \
    scripts/greatlakes_spgpu_epim_pairverify_breadth_a40.sbatch
fi

if [[ "${SUBMIT_GPU}" == "1" && "${SUBMIT_PYKAN_GPU}" == "1" ]]; then
  # V100/gpu pyKAN robustness rows.  These are slowest but most useful for
  # stress-test appendices and stronger benchmark claims.
  submit sbatch --account="${ACCOUNT}" --array=0-8 \
    --export=ALL,PYTHON_BIN="${PY}" \
    scripts/greatlakes_semisynth_drilldown_24h.sbatch

  submit sbatch --account="${ACCOUNT}" --array=0-3 \
    --export=ALL,PYTHON_BIN="${PY}" \
    scripts/greatlakes_prune_symbolic_more_array.sbatch

  submit sbatch --account="${ACCOUNT}" --array=0-8 \
    --export=ALL,PYTHON_BIN="${PY}" \
    scripts/greatlakes_readout_taxonomy_array.sbatch
fi

if [[ "${SUBMIT_SCORE_REFRESH}" == "1" ]]; then
  submit sbatch --account="${STANDARD_ACCOUNT}" \
    --export=ALL,PYTHON_BIN="${PY}",BUILD_FIGURE_SUMMARIES=1 \
    scripts/greatlakes_claimtransfer_score_refresh_standard.sbatch
fi

echo
echo "[$(date -Is)] submitted enhancement jobs. Check with:"
echo 'squeue -u $USER -o "%.18i %.12P %.10a %.30j %.2t %.10M %.10l %R"'
