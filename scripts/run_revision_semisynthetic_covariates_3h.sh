#!/usr/bin/env bash
set -euo pipefail

PY="${PYTHON:-python}"
OUT="results/revision/semisynthetic_covariates_3h"
mkdir -p "$OUT"

"$PY" experiments/run_semisynthetic_covariate_audit.py \
  --out-dir "$OUT" \
  --datasets diabetes breast_cancer \
  --coefficients 0.10 0.25 0.50 \
  --samples 128 256 384 \
  --test-samples 128 \
  --outer-seeds 0 1 2 3 4 5 6 7 8 9 \
  --R 12 \
  --top-m 4 \
  --methods feature_stability_var feature_edge_hybrid \
  --width-hidden 8 \
  --grid 5 \
  --k 3 \
  --lamb 0.001 \
  --probe-steps 35 \
  --pred-batch-size 4096 \
  --device auto
