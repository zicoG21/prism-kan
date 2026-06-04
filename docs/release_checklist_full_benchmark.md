# Full Benchmark Release Checklist

This checklist is for turning the current alpha artifact into a mature public
benchmark release.

## Clean Checkout Check

Run from a fresh clone or temporary checkout:

```bash
python scripts/run_benchmark.py --quick
python scripts/check_benchmark_artifact.py --min-claim-rows 100000 --min-score-rows 600 --min-coverage-rows 200 --min-missingness-rows 200
python examples/minimal_adapter.py --out examples/minimal_submission_generated.csv
python scripts/score_submission.py examples/minimal_submission.csv --out-dir score_reports/example_minimal
bash scripts/check_release_bundle.sh
bash scripts/check_release_overlay_checkout.sh
```

Expected current scale:

- at least 21 task-card/template rows;
- at least 100,000 claim rows;
- at least 600 aggregate score-report rows;
- at least 200 coverage rows.
- at least 200 coverage-gap rows.
- at least 200 missingness-report rows.

## Files That Should Be Versioned

- `task_cards/*.json`
- `adapters/adapter_output_schema.json`
- `adapters/adapter_family_registry.json`
- `adapters/submission_metadata_schema.json`
- `claim_records/claim_record_schema.json`
- `claim_records/example_claim_record.csv`
- `scripts/validate_adapter_registry.py`
- `scripts/validate_adapter_outputs.py`
- `scripts/validate_submission_metadata.py`
- `scripts/build_claim_records.py`
- `scripts/build_score_report.py`
- `scripts/build_coverage_gap_report.py`
- `scripts/score_submission.py`
- `scripts/run_benchmark.py`
- `scripts/build_typed_dashboard.py`
- `scripts/build_claimtransfer_release_bundle.sh`
- `scripts/check_release_bundle.sh`
- `scripts/check_release_overlay_checkout.sh`
- `scripts/build_hidden_private_bundle.py`
- `scripts/validate_hidden_bundle.py`
- `examples/minimal_adapter.py`
- `examples/minimal_submission.csv`
- release documentation under `docs/`

## Files That Should Stay Generated

- `claim_records/released_adapter_outputs.csv`
- `claim_records/released_claim_records.csv`
- `claim_records/hidden_claim_records.csv`
- `score_reports/benchmark_manifest.*`
- `score_reports/missingness_report.*`
- `score_reports/hidden_*`
- `score_reports/submission_score/`
- `score_reports/example_*/`
- `dashboards/`
- `results/`
- `artifacts/`

## Before a Public Benchmark Paper Submission

- Confirm GL result packs are unpacked and indexed.
- Rebuild released adapter outputs and official score reports.
- Rebuild typed dashboards.
- Run the clean checkout and release-bundle overlay checks.
- Add adapter-family budget/hyperparameter notes.
- Add a hidden/offline demonstration or clearly label hidden support as
  template-only.
- Freeze task-card registry version.
- Tag the release.
