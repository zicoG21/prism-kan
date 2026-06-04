#!/usr/bin/env bash
set -euo pipefail

# Simulate the reviewer path for a fresh source checkout plus a release bundle.
#
# The generated benchmark CSVs are intentionally not tracked in git.  A clean
# source checkout becomes runnable when the reviewer overlays the release bundle
# built by `scripts/build_claimtransfer_release_bundle.sh`.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
tmpdir="$(mktemp -d /tmp/claimtransfer_overlay_check_XXXXXX)"
source_dir="$tmpdir/source"

"$PYTHON_BIN" scripts/run_benchmark.py --quick
bash scripts/build_claimtransfer_release_bundle.sh
latest="$(ls -t artifacts/release/claimtransfer_release_*.tar.gz | head -1)"

mkdir -p "$source_dir"
git archive --format=tar HEAD | tar -x -C "$source_dir"

tar -xzf "$latest" -C "$source_dir"

(
  cd "$source_dir"
  "$PYTHON_BIN" scripts/run_benchmark.py --quick
  "$PYTHON_BIN" examples/minimal_adapter.py \
    --out examples/minimal_submission_generated.csv
  "$PYTHON_BIN" scripts/score_submission.py examples/minimal_submission_generated.csv \
    --out-dir score_reports/example_generated_check \
    --validate-task-cards \
    --metadata examples/minimal_submission_metadata.json
  "$PYTHON_BIN" scripts/build_hidden_private_bundle.py
  "$PYTHON_BIN" scripts/validate_hidden_bundle.py
)

echo "ClaimTransfer clean-checkout overlay check passed."
echo "Source overlay directory: $source_dir"
echo "Release bundle: $latest"
