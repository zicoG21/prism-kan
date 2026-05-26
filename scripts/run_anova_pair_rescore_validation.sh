#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
SOURCE_DETAIL="${SOURCE_DETAIL:-results/innovation_loop/strict_validation_20260526_011917/innovation_detail.csv}"
OUT_DIR="${OUT_DIR:-results/innovation_loop/anova_pair_rescore_validation_${STAMP}}"
mkdir -p "${OUT_DIR}"

if [ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "${HOME}/anaconda3/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV:-prism}"
fi

PYTHONPATH=. python experiments/rescore_stability_supports_with_pair_methods.py \
  --input "${SOURCE_DETAIL}" \
  --out_dir "${OUT_DIR}" \
  --functions core_interaction_c025 \
  --methods feature_stability_var feature_edge_hybrid \
  --samples 512 1024 \
  --dimensions 100 \
  --top_m 4 5 6 \
  --test_samples "${TEST_SAMPLES:-4096}" \
  --pair_methods fd anova_abs anova_var fd_anova_hybrid \
  --variable_points "${VARIABLE_POINTS:-512}" \
  --fd_points "${FD_POINTS:-128}" \
  --anova_points "${ANOVA_POINTS:-64}" \
  --anova_background "${ANOVA_BACKGROUND:-64}" \
  --device "${DEVICE:-cpu}" \
  > "${OUT_DIR}/rescore.log" 2>&1

echo "DONE ANOVA pair rescore validation: ${OUT_DIR}"
