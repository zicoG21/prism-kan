#!/usr/bin/env bash
set -euo pipefail

# Pack only the non-KAN hidden/private standard-formula result slices.
#
# Run on Great Lakes from the project root after
# submit_claimtransfer_nonkan_hidden_gl.sh finishes:
#
#   bash scripts/greatlakes_pack_nonkan_hidden_results.sh
#
# Then from local:
#
#   scp zicong@greatlakes.arc-ts.umich.edu:/home/zicong/prism-kan/artifacts/greatlakes/nonkan_hidden_results_YYYYmmdd_HHMMSS.tar.gz artifacts/greatlakes/

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p artifacts/greatlakes

stamp="$(date +%Y%m%d_%H%M%S)"
out="artifacts/greatlakes/nonkan_hidden_results_${stamp}.tar.gz"

tar -czf "${out}" \
  results/revision/standard_formula_adapter_sweep_hidden_nonkan* \
  results/revision/gplearn_standard_formula_baseline_hidden_nonkan* \
  results/revision/pysr_standard_formula_baseline_hidden_nonkan* \
  results/revision/mlp_hessian_standard_formula_baseline_hidden_nonkan* \
  results/revision/eql_standard_formula_baseline_hidden_nonkan* \
  task_cards/claimtransfer_v1_standard_formula_hidden_private.json \
  task_cards/standard_formula_hidden_private_task_card_map.md \
  claim_records/released_adapter_outputs.csv \
  claim_records/released_claim_records.csv \
  score_reports/score_report.csv \
  score_reports/coverage_table.csv \
  score_reports/overclaim_risk_report.csv \
  score_reports/split_overclaim_consistency.csv \
  score_reports/public_hidden_split_readiness.csv \
  logs/greatlakes/kan-std-sweep_* \
  logs/greatlakes/kan-gplearn-sr_* \
  logs/greatlakes/kan-pysr-sr_* \
  logs/greatlakes/kan-mlphess-sr_* \
  logs/greatlakes/kan-eql-sr_* \
  logs/greatlakes/kan-ct-score_* \
  2>/dev/null

ls -lh "${out}"
