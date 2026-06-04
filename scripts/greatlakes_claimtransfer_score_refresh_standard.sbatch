#!/usr/bin/env bash
#SBATCH --job-name=kan-ct-score
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=logs/greatlakes/%x_%j.out
#SBATCH --error=logs/greatlakes/%x_%j.err

set -euo pipefail

# CPU-only official ClaimTransfer-Bench report refresh.
#
# Run this after result CSVs are present on Great Lakes.  It validates task
# cards, materializes standardized adapter-output rows, recomputes official
# claim records, and writes score/coverage reports.  It does not train models
# and does not request GPUs.

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs/greatlakes

PY="${PYTHON_BIN:-$PWD/.venv/bin/python}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"

echo "[$(date -Is)] node=$(hostname)"
echo "[$(date -Is)] refreshing ClaimTransfer official reports"
"$PY" scripts/validate_task_cards.py
"$PY" scripts/build_claim_records.py
"$PY" scripts/build_score_report.py
"$PY" scripts/validate_claim_records.py
"$PY" scripts/build_coverage_gap_report.py
"$PY" scripts/summarize_coverage_gaps.py
"$PY" scripts/build_benchmark_manifest.py
"$PY" scripts/build_full_benchmark_readiness_report.py
"$PY" scripts/build_typed_dashboard.py
"$PY" scripts/check_benchmark_artifact.py

if [[ "${BUILD_FIGURE_SUMMARIES:-1}" == "1" ]]; then
  "$PY" scripts/build_cross_method_transfer_matrix.py || true
  "$PY" scripts/build_epim_pairverify_summary.py || true
fi

echo "[$(date -Is)] outputs:"
ls -lh \
  claim_records/released_adapter_outputs.csv \
  claim_records/released_claim_records.csv \
  score_reports/score_report.csv \
  score_reports/coverage_table.csv \
  score_reports/coverage_gap_summary.csv \
  score_reports/full_benchmark_readiness.csv \
  score_reports/task_card_validation.csv
echo "[$(date -Is)] done ClaimTransfer official report refresh"
