#!/usr/bin/env bash
set -euo pipefail

# Build a reviewer-facing ClaimTransfer benchmark bundle.
#
# This script is intentionally separate from the Great Lakes result packer.  It
# creates a compact release artifact from the official-scored benchmark contract:
# task cards, schemas, released adapter outputs, official claim records, score
# reports, typed dashboards, examples, and benchmark documentation.
#
# Run from the repository root:
#
#   bash scripts/build_claimtransfer_release_bundle.sh
#
# Optional:
#
#   OUT_DIR=artifacts/release SKIP_QUICK=1 bash scripts/build_claimtransfer_release_bundle.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
OUT_DIR="${OUT_DIR:-artifacts/release}"
SKIP_QUICK="${SKIP_QUICK:-0}"
stamp="${STAMP:-$(date +%Y%m%d_%H%M%S_%N)}"

mkdir -p "$OUT_DIR"
filelist="$OUT_DIR/claimtransfer_release_filelist_${stamp}.txt"
manifest="$OUT_DIR/claimtransfer_release_manifest_${stamp}.txt"
tarball="$OUT_DIR/claimtransfer_release_${stamp}.tar.gz"

if [[ "$SKIP_QUICK" != "1" ]]; then
  "$PYTHON_BIN" scripts/run_benchmark.py --mode quick
fi

collect_files() {
  for root in "$@"; do
    [[ -e "$root" ]] || continue
    find "$root" -type f \
      \( -name '*.csv' -o -name '*.md' -o -name '*.json' -o -name '*.txt' \)
  done
}

collect_official_docs() {
  for f in \
    docs/adapter_fairness_and_budget_policy.md \
    docs/claimtransfer_full_benchmark_status_20260604.md \
    docs/claimtransfer_full_benchmark_todo_20260604.md \
    docs/hidden_evaluation_protocol.md \
    docs/release_checklist_full_benchmark.md \
    docs/reproducibility_checklist.md \
    docs/statistical_reporting_policy.md \
    docs/submission_format.md \
    docs/task_card_authoring_protocol.md
  do
    [[ -f "$f" ]] && printf '%s\n' "$f"
  done
}

{
  collect_files task_cards adapters scorers claim_records score_reports dashboards examples
  collect_official_docs
  for f in \
    BENCHMARK.md \
    benchmark_release.json \
    README_WORKSHOP.md \
    requirements.txt \
    scripts/run_benchmark.py \
    scripts/validate_release_contract.py \
    scripts/validate_task_cards.py \
    scripts/validate_adapter_registry.py \
    scripts/validate_adapter_outputs.py \
    scripts/validate_claim_records.py \
    scripts/validate_score_reports.py \
    scripts/validate_submission_metadata.py \
    scripts/build_claim_records.py \
    scripts/build_score_report.py \
    scripts/build_coverage_gap_report.py \
    scripts/build_benchmark_manifest.py \
    scripts/build_typed_dashboard.py \
    scripts/check_benchmark_artifact.py \
    scripts/score_submission.py \
    scripts/build_hidden_private_bundle.py \
    scripts/validate_hidden_bundle.py \
    scripts/check_release_bundle.sh \
    scripts/check_release_overlay_checkout.sh \
    scripts/print_artifact_env.py \
    scripts/build_claimtransfer_release_bundle.sh \
    examples/minimal_adapter.py \
    examples/minimal_submission_metadata.json
  do
    [[ -f "$f" ]] && printf '%s\n' "$f"
  done
} | sort -u > "$filelist"

{
  echo "# ClaimTransfer Release Bundle Manifest"
  echo "created_at=$(date -Is)"
  echo "git_commit=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
  echo
  echo "## Quick check"
  echo "Command: python scripts/run_benchmark.py --mode quick"
  echo
  echo "## Packed file count"
  wc -l < "$filelist"
  echo
  echo "## Packed roots"
  cut -d/ -f1 "$filelist" | sort | uniq -c
  echo
  echo "## Key generated outputs"
  for f in \
    claim_records/released_adapter_outputs.csv \
    claim_records/released_claim_records.csv \
    score_reports/score_report.csv \
    score_reports/coverage_table.csv \
    score_reports/missingness_report.csv \
    score_reports/benchmark_manifest.csv \
    dashboards/README.md
  do
    [[ -f "$f" ]] && printf '%s\t%s bytes\n' "$f" "$(stat -c '%s' "$f")"
  done
  echo
  echo "## Packed files"
  cat "$filelist"
} > "$manifest"

tar -czf "$tarball" "$manifest" "$filelist" --files-from "$filelist"

echo "Wrote manifest: $manifest"
echo "Wrote filelist: $filelist"
echo "Wrote tarball:  $tarball"
