#!/usr/bin/env bash
set -euo pipefail

# Submit all remaining high-value ClaimTransfer full-benchmark work to Great
# Lakes.  This script is meant for "GL is open, do not waste it" moments.
#
# Run from the project root on Great Lakes:
#
#   cd /home/zicong/prism-kan
#   git pull
#   bash scripts/submit_claimtransfer_all_gl.sh
#
# Design:
#   - no array throttles; Slurm account/GRES limits decide concurrency
#   - standard/CPU jobs go to jaabell0 by default
#   - jaabell0 and engin1 GPU/A40 queues use disjoint seed blocks
#   - packed A40 mode is off by default because GL A40s often use exclusive
#     process compute mode
#
# Useful switches:
#
#   SUBMIT_JAABELL_GPU=0 bash scripts/submit_claimtransfer_all_gl.sh
#   SUBMIT_ENGIN1_SPGPU=0 bash scripts/submit_claimtransfer_all_gl.sh
#   SUBMIT_FRESH_SEEDS=1 SUBMIT_OPTIONAL=1 bash scripts/submit_claimtransfer_all_gl.sh
#   SUBMIT_SCORE_REFRESH=0 bash scripts/submit_claimtransfer_all_gl.sh

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs/greatlakes

PY="${PYTHON_BIN:-$PWD/.venv/bin/python}"
JAABELL_ACCOUNT="${JAABELL_ACCOUNT:-jaabell0}"
ENGIN_ACCOUNT="${ENGIN_ACCOUNT:-engin1}"
STANDARD_ACCOUNT="${STANDARD_ACCOUNT:-jaabell0}"

SUBMIT_FRESH_SEEDS="${SUBMIT_FRESH_SEEDS:-0}"
SUBMIT_JAABELL_GPU="${SUBMIT_JAABELL_GPU:-$SUBMIT_FRESH_SEEDS}"
SUBMIT_JAABELL_SPGPU="${SUBMIT_JAABELL_SPGPU:-$SUBMIT_FRESH_SEEDS}"
SUBMIT_ENGIN_GPU="${SUBMIT_ENGIN_GPU:-$SUBMIT_FRESH_SEEDS}"
SUBMIT_ENGIN_SPGPU="${SUBMIT_ENGIN_SPGPU:-$SUBMIT_FRESH_SEEDS}"
SUBMIT_STANDARD="${SUBMIT_STANDARD:-1}"
SUBMIT_OPTIONAL="${SUBMIT_OPTIONAL:-0}"
SUBMIT_SCORE_REFRESH="${SUBMIT_SCORE_REFRESH:-1}"

A40_BATCH_SIZE="${A40_BATCH_SIZE:-16384}"
A40_PAIR_CHUNK_SIZE="${A40_PAIR_CHUNK_SIZE:-1500}"

echo "[$(date -Is)] ClaimTransfer all-GL submit"
echo "[$(date -Is)] python=${PY}"
echo "[$(date -Is)] jaabell=${JAABELL_ACCOUNT} engin=${ENGIN_ACCOUNT} standard=${STANDARD_ACCOUNT}"
echo "[$(date -Is)] toggles: fresh_seeds=${SUBMIT_FRESH_SEEDS} jaabell_gpu=${SUBMIT_JAABELL_GPU} jaabell_spgpu=${SUBMIT_JAABELL_SPGPU} engin_gpu=${SUBMIT_ENGIN_GPU} engin_spgpu=${SUBMIT_ENGIN_SPGPU} standard=${SUBMIT_STANDARD} optional=${SUBMIT_OPTIONAL} refresh=${SUBMIT_SCORE_REFRESH}"

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

submit_hidden_gpu() {
  local account="$1"
  local partition="$2"
  local seed_base="$3"
  local label_suffix="$4"
  local batch_size="$5"
  local pair_chunk="$6"
  submit sbatch --account="${account}" --partition="${partition}" --array=0-11 \
    --export=ALL,PYTHON_BIN="${PY}",SEED_BASE="${seed_base}",SEED_COUNT=12,LABEL_SUFFIX="${label_suffix}",BATCH_SIZE="${batch_size}",PAIR_CHUNK_SIZE="${pair_chunk}" \
    scripts/greatlakes_claimtransfer_hidden_claimcards_cuda.sbatch
}

submit_scorergram_a40() {
  local account="$1"
  local seed_base="$2"
  local label_suffix="$3"
  submit sbatch --account="${account}" --partition=spgpu --array=0-9 \
    --export=ALL,PYTHON_BIN="${PY}",SEED_BASE="${seed_base}",SEED_COUNT=8,LABEL_SUFFIX="${label_suffix}",SCORERS_COLON=epim:anova_abs:fd:hessian:hybrid_epim_anova,BATCH_SIZE="${A40_BATCH_SIZE}",PAIR_CHUNK_SIZE=1024 \
    scripts/greatlakes_spgpu_pair_scorer_grammar_a40.sbatch
}

submit_epim_a40() {
  local account="$1"
  local seeds="$2"
  local label_suffix="$3"
  submit sbatch --account="${account}" --partition=spgpu --array=0-7 \
    --export=ALL,PYTHON_BIN="${PY}",SEEDS="${seeds}",LABEL_SUFFIX="${label_suffix}",BATCH_SIZE="${A40_BATCH_SIZE}",PAIR_CHUNK_SIZE=768 \
    scripts/greatlakes_spgpu_epim_pairverify_breadth_a40.sbatch
}

if [[ "${SUBMIT_JAABELL_GPU}" == "1" ]]; then
  submit_hidden_gpu "${JAABELL_ACCOUNT}" gpu 5200 s5200_5211_jb_gpu 8192 1000
fi

if [[ "${SUBMIT_JAABELL_SPGPU}" == "1" ]]; then
  submit_hidden_gpu "${JAABELL_ACCOUNT}" spgpu 5300 s5300_5311_jb_a40 "${A40_BATCH_SIZE}" "${A40_PAIR_CHUNK_SIZE}"
  submit_scorergram_a40 "${JAABELL_ACCOUNT}" 5400 s5400_5407_hess_jb
  submit_epim_a40 "${JAABELL_ACCOUNT}" 900-929 s900_929_jb
fi

if [[ "${SUBMIT_ENGIN_GPU}" == "1" ]]; then
  submit_hidden_gpu "${ENGIN_ACCOUNT}" gpu 6200 s6200_6211_e_gpu 8192 1000
fi

if [[ "${SUBMIT_ENGIN_SPGPU}" == "1" ]]; then
  submit_hidden_gpu "${ENGIN_ACCOUNT}" spgpu 6300 s6300_6311_e_a40 "${A40_BATCH_SIZE}" "${A40_PAIR_CHUNK_SIZE}"
  submit_scorergram_a40 "${ENGIN_ACCOUNT}" 6400 s6400_6407_hess_e
  submit_epim_a40 "${ENGIN_ACCOUNT}" 930-959 s930_959_e
fi

if [[ "${SUBMIT_STANDARD}" == "1" ]]; then
  # Cross-method extended adapter coverage.  Internal script is array=0-49.
  submit sbatch --account="${STANDARD_ACCOUNT}" \
    --export=ALL,PYTHON_BIN="${PY}",SEED_START=100,SEED_STOP=129,GA2M_SEED_STOP=119,SYMBOLIC_SEED_STOP=119 \
    scripts/greatlakes_cross_method_transfer_baselines_extended.sbatch

  # TreeGate candidate-vs-verifier coverage.  CPU only.
  submit sbatch --account="${STANDARD_ACCOUNT}" --array=0-23 \
    --export=ALL,PYTHON_BIN="${PY}" \
    scripts/greatlakes_treegate_pair_screen_extended_standard.sbatch
fi

if [[ "${SUBMIT_OPTIONAL}" == "1" ]]; then
  # These are not workshop-critical, but are useful for a full benchmark paper.
  submit sbatch --account="${JAABELL_ACCOUNT}" --partition=gpu --array=0-8 \
    --export=ALL,PYTHON_BIN="${PY}" \
    scripts/greatlakes_semisynth_drilldown_24h.sbatch

  submit sbatch --account="${JAABELL_ACCOUNT}" --partition=gpu --array=0-3 \
    --export=ALL,PYTHON_BIN="${PY}" \
    scripts/greatlakes_prune_symbolic_more_array.sbatch

  submit sbatch --account="${JAABELL_ACCOUNT}" --partition=gpu --array=0-8 \
    --export=ALL,PYTHON_BIN="${PY}" \
    scripts/greatlakes_readout_taxonomy_array.sbatch
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
echo "[$(date -Is)] submitted all requested queues."
if [[ "${#JOB_IDS[@]}" -gt 0 ]]; then
  echo "[$(date -Is)] job ids: ${JOB_IDS[*]}"
fi
echo 'Check with: squeue -u $USER -o "%.18i %.12P %.10a %.30j %.2t %.10M %.10l %R"'
