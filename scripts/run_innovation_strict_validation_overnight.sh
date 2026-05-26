#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-results/innovation_loop/strict_validation_${STAMP}}"
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
  --methods \
    grad_stability_var \
    feature_stability_var \
    edge_stability_var \
    edge_endpoint_mass \
    feature_edge_hybrid \
    edge_pair_hybrid \
  --broad_eval_seeds 100 101 102 103 104 105 106 107 108 109 \
  --broad_probe_seeds 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 216 217 218 219 \
  --exploit_eval_seeds 110 111 112 113 114 115 116 117 118 119 \
  --exploit_probe_seeds 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 \
  --stress_eval_seeds 120 121 122 123 124 125 126 127 \
  --stress_probe_seeds 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 \
  --stress_dimensions 500 1000 \
  --probe_steps "${PROBE_STEPS:-80}" \
  --refit_steps "${REFIT_STEPS:-100}" \
  --exploit_probe_steps "${EXPLOIT_PROBE_STEPS:-100}" \
  --exploit_refit_steps "${EXPLOIT_REFIT_STEPS:-120}" \
  --stress_probe_steps "${STRESS_PROBE_STEPS:-80}" \
  --fd_points "${FD_POINTS:-128}" \
  --keep_top_pairs "${KEEP_TOP_PAIRS:-160}" \
  "$@"
