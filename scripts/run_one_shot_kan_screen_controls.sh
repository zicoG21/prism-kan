#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-results/innovation_loop/one_shot_kan_screen_controls_${STAMP}}"
mkdir -p "${OUT_DIR}/logs"
export OUT_DIR

if [ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "${HOME}/anaconda3/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV:-prism}"
fi

run_one_shot_setting() {
  local tag="$1"
  local fn="$2"
  local n="$3"
  local dim="$4"
  local top_m="$5"
  shift 5
  local seeds=("$@")

  echo "============================================================"
  echo "One-shot KAN screen: ${tag} function=${fn} n=${n} d=${dim} top_m=${top_m} seeds=${seeds[*]}"
  echo "============================================================"

  PYTHONPATH=. python experiments/run_single_kan_screen_refit.py \
    --functions "${fn}" \
    --samples "${n}" \
    --test_samples "${TEST_SAMPLES:-4096}" \
    --dimension "${dim}" \
    --noise 0.0 \
    --seeds "${seeds[@]}" \
    --top_m "${top_m}" \
    --methods single_grad_var single_feature_var single_edge_var single_feature_edge_hybrid \
    --screen_steps "${SCREEN_STEPS:-40}" \
    --refit_steps "${REFIT_STEPS:-80}" \
    --grid 5 \
    --k 3 \
    --width_hidden 8 \
    --lamb 0.001 \
    --no_update_grid \
    --screen_variable_points "${SCREEN_VAR_POINTS:-256}" \
    --refit_variable_points "${REFIT_VAR_POINTS:-256}" \
    --fd_points "${FD_POINTS:-96}" \
    --device "${DEVICE:-auto}" \
    --out "${OUT_DIR}/${tag}_detail.csv" \
    --summary_out "${OUT_DIR}/${tag}_summary.csv" \
    > "${OUT_DIR}/logs/${tag}.log" 2>&1
}

SEEDS_QUICK=(100 101 102 103 104)

# Keep this control intentionally small. Full-dimensional one-shot KAN
# screening is much more expensive than low-dimensional screened refits; these
# settings are enough to test whether stability beats a single KAN explanation
# pass without turning the job into a multi-day stress run.
run_one_shot_setting c025_n512_d100_top4 core_interaction_c025 512 100 4 "${SEEDS_QUICK[@]}"
run_one_shot_setting c025_n1024_d100_top4 core_interaction_c025 1024 100 4 "${SEEDS_QUICK[@]}"
run_one_shot_setting c05_n512_d100_top5 core_interaction_c05 512 100 5 "${SEEDS_QUICK[@]}"

PYTHONPATH=. python - <<'PY'
from pathlib import Path
import os

import pandas as pd

out = Path(os.environ["OUT_DIR"])
pieces = []
for path in sorted(out.glob("*_summary.csv")):
    try:
        df = pd.read_csv(path)
        df["source_file"] = path.name
        pieces.append(df)
    except Exception as exc:
        print(f"Could not read {path}: {exc}")

if pieces:
    combined = pd.concat(pieces, ignore_index=True, sort=False)
    combined.to_csv(out / "combined_one_shot_summary.csv", index=False)
    cols = [
        "source_file",
        "method",
        "function",
        "dimension",
        "samples",
        "top_m",
        "screen_model_test_mse_mean",
        "test_mse_mean",
        "screen_contains_true_interactions_mean",
        "screen_interaction_endpoint_recall_mean",
        "interaction_f1_mean",
        "num_runs",
    ]
    cols = [c for c in cols if c in combined.columns]
    print(combined[cols].to_string(index=False))
    print(f"Wrote {out / 'combined_one_shot_summary.csv'}")
else:
    print("No summary CSVs found.")
PY

echo "DONE one-shot KAN screen controls: ${OUT_DIR}"
