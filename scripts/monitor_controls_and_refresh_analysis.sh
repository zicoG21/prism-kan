#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

RUN_DIR="${RUN_DIR:-results/innovation_loop/strict_screened_baseline_controls_20260526_104243}"
ONE_SHOT_DIR="${ONE_SHOT_DIR:-results/innovation_loop/one_shot_kan_screen_controls_after_screened_20260526_104933}"
NATIVE_DIR="${NATIVE_DIR:-results/innovation_loop/strict_validation_20260526_011917}"
OUT_DIR="${OUT_DIR:-${RUN_DIR}/analysis_min8_live}"
LOG_FILE="${LOG_FILE:-${OUT_DIR}/monitor.log}"
DURATION_SEC="${DURATION_SEC:-7200}"
POLL_SEC="${POLL_SEC:-300}"
MIN_RUNS="${MIN_RUNS:-8}"

mkdir -p "${OUT_DIR}"

if [ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "${HOME}/anaconda3/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV:-prism}"
fi

refresh_analysis() {
  local args=(
    --native_run_dir "${NATIVE_DIR}"
    --screened_run_dir "${RUN_DIR}"
    --out_dir "${OUT_DIR}"
    --min_runs "${MIN_RUNS}"
  )
  if compgen -G "${ONE_SHOT_DIR}/*_detail.csv" > /dev/null || [ -f "${ONE_SHOT_DIR}/detail.csv" ]; then
    args+=(--one_shot_dir "${ONE_SHOT_DIR}")
  fi
  PYTHONPATH=. python experiments/compare_native_with_screened_controls.py "${args[@]}" \
    > "${OUT_DIR}/compare_latest.log" 2>&1 || true
}

status_snapshot() {
  {
    echo "============================================================"
    date
    echo "RUN_DIR=${RUN_DIR}"
    echo "ONE_SHOT_DIR=${ONE_SHOT_DIR}"
    echo
    echo "[processes]"
    pgrep -af 'run_strict_screened_baseline_controls|run_tuned_kan_recovery.py|one_shot_after_screened|run_one_shot_kan_screen_controls|run_single_kan_screen_refit' || true
    echo
    echo "[screened eval row counts]"
    for f in "${RUN_DIR}"/*_screen_eval.csv; do
      [ -e "$f" ] || continue
      printf '%s %s\n' "$(basename "$f")" "$(wc -l < "$f")"
    done | sort
    echo
    echo "[screened summaries]"
    find "${RUN_DIR}" -maxdepth 1 -name '*_screen_summary.csv' -printf '%f\n' | sort || true
    echo
    echo "[one-shot row counts]"
    if [ -d "${ONE_SHOT_DIR}" ]; then
      for f in "${ONE_SHOT_DIR}"/*_detail.csv; do
        [ -e "$f" ] || continue
        printf '%s %s\n' "$(basename "$f")" "$(wc -l < "$f")"
      done | sort
    fi
  } >> "${LOG_FILE}"
}

end_time=$((SECONDS + DURATION_SEC))
echo "Starting monitor for ${DURATION_SEC}s, poll=${POLL_SEC}s" >> "${LOG_FILE}"
while [ "${SECONDS}" -lt "${end_time}" ]; do
  status_snapshot
  refresh_analysis
  sleep "${POLL_SEC}"
done
status_snapshot
refresh_analysis
echo "DONE monitor at $(date)" >> "${LOG_FILE}"
