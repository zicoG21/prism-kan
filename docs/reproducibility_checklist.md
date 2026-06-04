# ClaimTransfer Reproducibility Checklist

Last updated: 2026-06-04

This checklist describes the reviewer-facing ClaimTransfer benchmark artifact.
The quick path does not retrain pyKAN models. It rebuilds official claim records,
score reports, coverage tables, coverage-gap reports, missingness reports,
dashboards, and manifests from released adapter-output CSVs.

## Quick Reviewer Path

Run from the repository root:

```bash
python scripts/run_benchmark.py --quick
```

Expected output includes:

```text
Validated 21 task cards.
Validated adapter outputs: claim_records/released_adapter_outputs.csv
rows: 117114
required columns: 12
Wrote claim_records/released_claim_records.csv (117114 claim rows)
Wrote score_reports/score_report.csv (694 aggregate rows)
Wrote score_reports/coverage_table.csv (255 coverage rows)
Wrote score_reports/coverage_gap_report.csv (464 expected cells)
Wrote score_reports/missingness_report.csv (377 missingness rows)
Wrote .../score_reports/benchmark_manifest.csv (47 files)
ClaimTransfer benchmark artifact check passed.
```

## Release Bundle Smoke Test

The release bundle contains task cards, schemas, released adapter outputs,
official claim records, score reports, dashboards, examples, and benchmark
documentation. It is designed to be runnable without `results/revision`.

```bash
bash scripts/check_release_bundle.sh
```

This test:

- runs the quick path;
- builds `artifacts/release/claimtransfer_release_*.tar.gz`;
- extracts the bundle to a temporary directory;
- reruns the quick path from the extracted bundle;
- builds hidden/private participant and scorer registries;
- validates that the participant hidden view withholds labels, targets, and
  private seed blocks;
- generates a minimal adapter submission;
- scores both generated and static minimal submissions.

## Clean Source + Bundle Overlay Test

Generated benchmark CSVs are not all meant to be tracked in git. This test
simulates a clean source checkout plus release-bundle overlay:

```bash
bash scripts/check_release_overlay_checkout.sh
```

It creates a temporary source tree from `git archive HEAD`, overlays the latest
release bundle, and verifies that the quick path, minimal adapter, offline
submission scorer, and hidden/private bundle generator all run.

## Submission Scoring

Participants submit normalized adapter-output rows. The official scorer
recomputes pass/fail, rank, margin, and aggregate reports.

```bash
python examples/minimal_adapter.py --out examples/minimal_submission_generated.csv
python scripts/score_submission.py examples/minimal_submission_generated.csv \
  --out-dir score_reports/example_generated_check \
  --validate-task-cards
```

Expected output:

```text
Wrote .../claim_records.csv (6 claim rows)
Wrote .../score_report.csv (6 or 8 aggregate rows, depending on static versus generated example)
Wrote .../coverage_table.csv (6 or 8 coverage rows, depending on static versus generated example)
Wrote .../missingness_report.csv (6 or 8 missingness rows, depending on static versus generated example)
Official submission score written to ...
```

## Hidden/Offline Evaluation

Maintainers can generate hidden participant and private scoring views:

```bash
python scripts/build_hidden_private_bundle.py
python scripts/validate_hidden_bundle.py
python scripts/run_benchmark.py --hidden --hidden-input claim_records/released_adapter_outputs.csv
```

The participant view withholds formulas, supports, pair targets, and private
seed ranges. The private scoring registry keeps those fields for offline
official scoring.

## Generated Outputs

Core generated outputs:

- `claim_records/released_adapter_outputs.csv`
- `claim_records/released_claim_records.csv`
- `score_reports/score_report.csv`
- `score_reports/coverage_table.csv`
- `score_reports/missingness_report.csv`
- `score_reports/benchmark_manifest.csv`
- `dashboards/*.csv` and `dashboards/*.md`
- `artifacts/release/claimtransfer_release_*.tar.gz`
- `artifacts/private_hidden/*.json`

These outputs may be rebuilt locally or packed into a release bundle. The
source of truth for verdicts is the official scorer, not adapter-provided
pass/fail labels.
