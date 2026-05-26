#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-results/innovation_loop/strict_baseline_controls_${STAMP}}"
mkdir -p "${OUT_DIR}/logs"
export OUT_DIR

if [ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "${HOME}/anaconda3/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV:-prism}"
fi

COMMON=(
  --test_samples 4096
  --dimension 100
  --noise 0.0
  --grid 5
  --k 3
  --width_hidden 8
  --lamb 0.001
  --steps "${STEPS:-100}"
  --no_update_grid
  --interaction_method fd
  --device "${DEVICE:-auto}"
)

run_setting() {
  local tag="$1"
  local fn="$2"
  local n="$3"
  local top_m="$4"
  shift 4
  local seeds=("$@")

  echo "============================================================"
  echo "Baseline setting: ${tag} function=${fn} n=${n} top_m=${top_m} seeds=${seeds[*]}"
  echo "============================================================"

  PYTHONPATH=. python experiments/run_tuned_kan_recovery.py \
    --functions "${fn}" \
    --screen_modes raw \
    --samples "${n}" \
    --seeds "${seeds[@]}" \
    --top_m "${top_m}" \
    --fd_points "${RAW_FD_POINTS:-32}" \
    "${COMMON[@]}" \
    --out "${OUT_DIR}/${tag}_raw_eval.csv" \
    --summary_out "${OUT_DIR}/${tag}_raw_summary.csv" \
    > "${OUT_DIR}/logs/${tag}_raw.log" 2>&1

  PYTHONPATH=. python experiments/run_tuned_kan_recovery.py \
    --functions "${fn}" \
    --screen_modes rf oracle_support random exclude_interaction \
    --samples "${n}" \
    --seeds "${seeds[@]}" \
    --top_m "${top_m}" \
    --fd_points "${SCREEN_FD_POINTS:-128}" \
    "${COMMON[@]}" \
    --out "${OUT_DIR}/${tag}_screen_eval.csv" \
    --summary_out "${OUT_DIR}/${tag}_screen_summary.csv" \
    > "${OUT_DIR}/logs/${tag}_screen.log" 2>&1
}

run_stress_screened() {
  local tag="$1"
  local fn="$2"
  local n="$3"
  local dim="$4"
  local top_m="$5"
  shift 5
  local seeds=("$@")

  echo "============================================================"
  echo "Screened stress baseline: ${tag} function=${fn} n=${n} d=${dim} top_m=${top_m}"
  echo "============================================================"

  PYTHONPATH=. python experiments/run_tuned_kan_recovery.py \
    --functions "${fn}" \
    --screen_modes rf oracle_support random exclude_interaction \
    --samples "${n}" \
    --test_samples 4096 \
    --dimension "${dim}" \
    --noise 0.0 \
    --seeds "${seeds[@]}" \
    --grid 5 \
    --k 3 \
    --width_hidden 8 \
    --lamb 0.001 \
    --steps "${STRESS_STEPS:-100}" \
    --no_update_grid \
    --top_m "${top_m}" \
    --interaction_method fd \
    --fd_points "${SCREEN_FD_POINTS:-128}" \
    --device "${DEVICE:-auto}" \
    --out "${OUT_DIR}/${tag}_screen_eval.csv" \
    --summary_out "${OUT_DIR}/${tag}_screen_summary.csv" \
    > "${OUT_DIR}/logs/${tag}_screen.log" 2>&1
}

SEEDS_A=(100 101 102 103 104 105 106 107 108 109)
SEEDS_B=(110 111 112 113 114 115 116 117 118 119)
SEEDS_STRESS=(120 121 122 123 124 125 126 127)

run_setting c025_n512_top4 core_interaction_c025 512 4 "${SEEDS_A[@]}"
run_setting c025_n1024_top4 core_interaction_c025 1024 4 "${SEEDS_A[@]}"
run_setting c025_n1024_top5 core_interaction_c025 1024 5 "${SEEDS_B[@]}"
run_setting c025_n1024_top6 core_interaction_c025 1024 6 "${SEEDS_B[@]}"
run_setting c025_n2048_top6 core_interaction_c025 2048 6 "${SEEDS_B[@]}"
run_setting c05_n512_top5 core_interaction_c05 512 5 "${SEEDS_B[@]}"
run_setting c05_n1024_top5 core_interaction_c05 1024 5 "${SEEDS_B[@]}"
run_setting c1_n512_top5 core_interaction_c1 512 5 "${SEEDS_B[@]}"

run_stress_screened c025_n2048_d500_top6 core_interaction_c025 2048 500 6 "${SEEDS_STRESS[@]}"
run_stress_screened c025_n2048_d1000_top6 core_interaction_c025 2048 1000 6 "${SEEDS_STRESS[@]}"
run_stress_screened c05_n1024_d500_top6 core_interaction_c05 1024 500 6 "${SEEDS_STRESS[@]}"
run_stress_screened c05_n1024_d1000_top6 core_interaction_c05 1024 1000 6 "${SEEDS_STRESS[@]}"
run_stress_screened c1_n1024_d500_top6 core_interaction_c1 1024 500 6 "${SEEDS_STRESS[@]}"
run_stress_screened c1_n1024_d1000_top6 core_interaction_c1 1024 1000 6 "${SEEDS_STRESS[@]}"

PYTHONPATH=. python - <<'PY'
from pathlib import Path
import pandas as pd
import os

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
    combined.to_csv(out / "combined_baseline_summary.csv", index=False)
    cols = [
        "source_file", "function", "screen_mode", "dimension", "samples",
        "test_mse_mean", "screen_contains_true_interactions_mean",
        "screen_interaction_endpoint_recall_mean", "interaction_f1_mean",
    ]
    cols = [c for c in cols if c in combined.columns]
    print(combined[cols].to_string(index=False))
    print(f"Wrote {out / 'combined_baseline_summary.csv'}")
else:
    print("No summary CSVs found.")
PY

echo "DONE baseline controls: ${OUT_DIR}"
