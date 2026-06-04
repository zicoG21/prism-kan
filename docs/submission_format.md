# ClaimTransfer Submission Format

ClaimTransfer scores raw or normalized evidence objects.  A submission should
not provide trusted final pass/fail labels; official scorers recompute verdicts
from the submitted evidence fields.

## Submission File

Submit one CSV using the normalized adapter-output schema:

```text
registry_version, split, task_id, task_family, adapter, adapter_family,
source_kind, source_file, seed, evidence_object, claim_type, target, scorer,
predicate, threshold, budget, rank, margin, raw_value, selected_set,
candidate_set, missing_reason, runtime_seconds, protocol
```

Required columns:

- `registry_version`
- `split`
- `task_id`
- `task_family`
- `adapter`
- `adapter_family`
- `seed`
- `evidence_object`
- `claim_type`
- `target`
- `scorer`
- `predicate`

Optional evidence columns:

- `threshold`
- `budget`
- `rank`
- `margin`
- `raw_value`
- `selected_set`
- `candidate_set`
- `missing_reason`
- `runtime_seconds`
- `protocol`

Missing evidence objects should be omitted or marked with `missing_reason`.
They should not be converted into failures for claims a workflow does not
expose.

## Metadata File

Submit a JSON metadata file with:

- `submission_id`
- `method_name`
- `adapter_family`
- `method_description`
- `native_outputs`
- `tuning_policy`
- `compute_budget`
- `missing_field_policy`
- optional `model_artifacts`, `contact`, and `license`

The schema is `adapters/submission_metadata_schema.json`; a minimal example is
`examples/minimal_submission_metadata.json`.

## Official Scoring

Run:

```bash
python scripts/score_submission.py path/to/submission.csv \
  --metadata path/to/submission_metadata.json \
  --out-dir score_reports/submission_score
```

Outputs:

- `claim_records.csv`
- `score_report.csv`
- `coverage_table.csv`
- `dashboard/`
- `submission_metadata.json` when metadata is supplied

For private or held-out evaluation, use the same submission format.  Hidden task
cards may withhold formula labels or seeds, but the official scorer uses the
same row-level contract.

## Leaderboard Policy

ClaimTransfer should not collapse prediction, support, endpoint, pair, pruning,
and symbolic claims into one scalar.  Reports are grouped by task card, adapter,
evidence object, claim type, scorer, and predicate.  A leaderboard-style view,
if used, should be a dashboard of typed claims rather than a single model score.
