#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-results/innovation_loop/${STAMP}}"
mkdir -p "${OUT_DIR}"

if [ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "${HOME}/anaconda3/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV:-prism}"
fi

PYTHONPATH=. python experiments/run_kan_native_innovation_loop.py \
  --out_dir "${OUT_DIR}" \
  --time_budget_hours "${TIME_BUDGET_HOURS:-10}" \
  --device "${DEVICE:-auto}" \
  --probe_steps "${PROBE_STEPS:-35}" \
  --refit_steps "${REFIT_STEPS:-50}" \
  --exploit_probe_steps "${EXPLOIT_PROBE_STEPS:-50}" \
  --exploit_refit_steps "${EXPLOIT_REFIT_STEPS:-80}" \
  --stress_probe_steps "${STRESS_PROBE_STEPS:-35}" \
  --fd_points "${FD_POINTS:-128}" \
  --keep_top_pairs "${KEEP_TOP_PAIRS:-120}" \
  "$@"
