# ClaimTransfer Benchmark

ClaimTransfer is an official-scored benchmark contract for structural claim
transfer.  Methods expose raw or normalized evidence objects; official scorers
recompute claim records, score reports, coverage tables, missingness reports,
and artifact manifests.

## Quick Check

```bash
python scripts/run_benchmark.py --quick
```

This validates task cards, rebuilds released adapter-output rows, recomputes
official claim records, rebuilds score/coverage/gap/missingness reports, and
checks the artifact manifest.

The quick check does not retrain pyKAN models or require a GPU; it rebuilds
official score reports from released/generated adapter-output CSVs.

## Release Bundle

Generated CSVs and score reports are intentionally not committed to git.  Build
the reviewer-facing artifact bundle with:

```bash
bash scripts/build_claimtransfer_release_bundle.sh
```

This runs the quick check and packs task cards, schemas, released adapter
outputs, official claim records, score reports, typed dashboards, examples,
benchmark documentation, and the critical scoring scripts under
`artifacts/release/`.

Verify the packed bundle from a temporary directory:

```bash
bash scripts/check_release_bundle.sh
```

Simulate a clean source checkout plus release-bundle overlay:

```bash
bash scripts/check_release_overlay_checkout.sh
```

## Core Contract

- Task-card schema: `task_cards/task_card_schema.json`
- Public diagnostic registry: `task_cards/claimtransfer_v0_public.json`
- Hidden-template registry: `task_cards/claimtransfer_v0_hidden_template.json`
- Scientific/expression templates: `task_cards/claimtransfer_v1_scientific_templates.json`
- Adapter-output schema: `adapters/adapter_output_schema.json`
- Adapter-family registry: `adapters/adapter_family_registry.json`
- Submission metadata schema: `adapters/submission_metadata_schema.json`
- Claim-record schema: `claim_records/claim_record_schema.json`

## Official Scoring

```bash
python scripts/validate_adapter_registry.py
python scripts/validate_submission_metadata.py examples/minimal_submission_metadata.json
python scripts/validate_adapter_outputs.py claim_records/released_adapter_outputs.csv
python scripts/build_claim_records.py
python scripts/build_score_report.py
python scripts/check_benchmark_artifact.py
```

Generated outputs:

- `claim_records/released_adapter_outputs.csv`
- `claim_records/released_claim_records.csv`
- `score_reports/score_report.csv`
- `score_reports/coverage_table.csv`
- `score_reports/coverage_gap_report.csv`
- `score_reports/missingness_report.csv`
- `score_reports/benchmark_manifest.csv`

## Submission Scoring

A participant submits a normalized adapter-output CSV.  The scorer recomputes
claim records and aggregate reports:

```bash
python scripts/score_submission.py path/to/submission.csv --out-dir score_reports/submission_score
```

Minimal examples:

```bash
python examples/minimal_adapter.py --out examples/minimal_submission_generated.csv
python scripts/score_submission.py examples/minimal_submission.csv \
  --out-dir score_reports/example_minimal \
  --metadata examples/minimal_submission_metadata.json
```

Use hidden/offline mode for private evaluation:

```bash
python scripts/run_benchmark.py --hidden --hidden-input path/to/private_submission.csv
```

Generate a maintainer-side private hidden bundle:

```bash
python scripts/build_hidden_private_bundle.py
```

The participant view withholds formulas, labels, pair targets, and private seed
blocks; the private scoring registry and seed manifest should stay outside the
public repository until evaluation closes.

## Reporting Policy

ClaimTransfer does not merge prediction, support, endpoint, pair, pruning, and
symbolic claims into one scalar.  Reports are grouped by registry version,
split, task card, adapter, evidence object, claim type, scorer, and predicate.
Alternative scorers produce separate scorer-indexed records.

## Typed Dashboard

Generate benchmark views without collapsing typed claims into one leaderboard:

```bash
python scripts/build_typed_dashboard.py
```

Outputs are written under `dashboards/`:

- `adapter_by_claim.md`
- `task_by_claim.md`
- `object_by_claim.md`
- `scorer_by_claim.md`
- `missingness.md`
