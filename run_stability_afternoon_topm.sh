#!/usr/bin/env bash
set -euo pipefail

# Afternoon robustness sweep for SS-KAN-V support budget.
#
# Default sweep:
#   d=100
#   n in {256,512,1024}
#   c in {0.25,0.5,1.0}
#   top_m in {4,6,8}
#   eval seeds 0..9
#
# Override examples:
#   TOP_MS="4 6" SAMPLES="512 1024" ./run_stability_afternoon_topm.sh
#   FUNCTIONS="core_interaction_c05 core_interaction_c1" SEEDS="0 1 2 3 4" ./run_stability_afternoon_topm.sh

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-results/stability_kan/afternoon_topm_${RUN_ID}}"

FUNCTIONS="${FUNCTIONS:-core_interaction_c025 core_interaction_c05 core_interaction_c1}"
SAMPLES="${SAMPLES:-256 512 1024}"
TOP_MS="${TOP_MS:-4 6 8}"
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"

mkdir -p "${OUT_DIR}/details" "${OUT_DIR}/summaries" "${OUT_DIR}/figures" "${OUT_DIR}/logs"

echo "[INFO] run_id=${RUN_ID}"
echo "[INFO] out_dir=${OUT_DIR}"
echo "[INFO] functions=${FUNCTIONS}"
echo "[INFO] samples=${SAMPLES}"
echo "[INFO] top_ms=${TOP_MS}"
echo "[INFO] seeds=${SEEDS}"

for top_m in ${TOP_MS}; do
  for n in ${SAMPLES}; do
    for fn in ${FUNCTIONS}; do
      tag="${fn}_n${n}_d100_topm${top_m}"
      echo "[RUN] ${tag}"
      python experiments/run_stability_selected_kan_quick.py \
        --functions "${fn}" \
        --samples "${n}" \
        --seeds ${SEEDS} \
        --stability_methods ss_kan_variable \
        --top_m "${top_m}" \
        --out "${OUT_DIR}/details/${tag}_detail.csv" \
        --summary_out "${OUT_DIR}/summaries/${tag}_summary.csv" \
        --fig_out "${OUT_DIR}/figures/${tag}_interaction_f1.png" \
        2>&1 | tee "${OUT_DIR}/logs/${tag}.log"
    done
  done
done

python experiments/summarize_stability_topm_sweep.py \
  --summary_dir "${OUT_DIR}/summaries" \
  --detail_dir "${OUT_DIR}/details" \
  --out_dir "${OUT_DIR}"

echo "[DONE] ${OUT_DIR}"
