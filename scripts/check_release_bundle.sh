#!/usr/bin/env bash
set -euo pipefail

# End-to-end release-bundle smoke test.
#
# The test builds a ClaimTransfer release bundle, extracts it to a temporary
# directory with no `results/` tree, and checks that the bundled released adapter
# outputs are sufficient to rebuild official claim records, score reports,
# coverage reports, dashboards, and hidden/private template files.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
"$PYTHON_BIN" scripts/run_benchmark.py --quick
bash scripts/build_claimtransfer_release_bundle.sh

latest="$(ls -t artifacts/release/claimtransfer_release_*.tar.gz | head -1)"
tmpdir="$(mktemp -d /tmp/claimtransfer_bundle_check_XXXXXX)"

tar -xzf "$latest" -C "$tmpdir"

(
  cd "$tmpdir"
  "$PYTHON_BIN" scripts/run_benchmark.py --quick
  "$PYTHON_BIN" scripts/build_hidden_private_bundle.py
  "$PYTHON_BIN" scripts/validate_hidden_bundle.py
  "$PYTHON_BIN" examples/minimal_adapter.py \
    --out examples/minimal_submission_generated.csv
  "$PYTHON_BIN" scripts/score_submission.py examples/minimal_submission_generated.csv \
    --out-dir score_reports/example_generated_check \
    --validate-task-cards \
    --metadata examples/minimal_submission_metadata.json
  "$PYTHON_BIN" scripts/score_submission.py examples/minimal_submission.csv \
    --out-dir score_reports/example_minimal_check \
    --validate-task-cards \
    --metadata examples/minimal_submission_metadata.json
)

echo "ClaimTransfer release bundle smoke test passed: $latest"
echo "Temporary check directory: $tmpdir"
