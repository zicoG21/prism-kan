# ClaimTransfer Examples

These examples are intentionally tiny.  They demonstrate the submission and
scoring contract without training a model.

## Score a Minimal Submission

```bash
python scripts/score_submission.py examples/minimal_submission.csv \
  --metadata examples/minimal_submission_metadata.json \
  --out-dir score_reports/example_minimal
```

Expected outputs:

- `score_reports/example_minimal/claim_records.csv`
- `score_reports/example_minimal/score_report.csv`
- `score_reports/example_minimal/coverage_table.csv`
- `score_reports/example_minimal/dashboard/`
- `score_reports/example_minimal/submission_metadata.json`

The example contains two adapters:

- `ExampleSparse` passes prediction, support, and pair claims on a bilinear card.
- `ExamplePredictor` passes prediction on a weak-centered card but fails support
  and pair claims.  This is the basic ClaimTransfer pattern: predictive evidence
  is not automatically structural evidence.

## Write a Minimal Adapter Output

```bash
python examples/minimal_adapter.py --out examples/minimal_submission_generated.csv
python scripts/score_submission.py examples/minimal_submission_generated.csv \
  --metadata examples/minimal_submission_metadata.json \
  --out-dir score_reports/example_generated
```

The adapter writes evidence rows only.  The benchmark scorer recomputes
pass/fail labels from `predicate`, `rank`, `margin`, `raw_value`, and
`selected_set`.

## Build the Typed Dashboard

```bash
python scripts/build_typed_dashboard.py
```

This writes markdown and CSV dashboard views under `dashboards/`.
