# ClaimTransfer Full Benchmark Status

Updated: 2026-06-04

## Completed Core Contract

- Task-card schema: `task_cards/task_card_schema.json`
- Public registry: `task_cards/claimtransfer_v0_public.json`
- Hidden-template registry: `task_cards/claimtransfer_v0_hidden_template.json`
- Scientific/expression templates: `task_cards/claimtransfer_v1_scientific_templates.json`
- Adapter-output schema: `adapters/adapter_output_schema.json`
- Claim-record schema: `claim_records/claim_record_schema.json`
- Offline submission scorer: `scripts/score_submission.py`
- Artifact manifest builder: `scripts/build_benchmark_manifest.py`
- Authoring protocol: `docs/task_card_authoring_protocol.md`
- Submission format: `docs/submission_format.md`
- Statistical reporting policy: `docs/statistical_reporting_policy.md`
- Adapter fairness/budget policy: `docs/adapter_fairness_and_budget_policy.md`
- Hidden evaluation protocol: `docs/hidden_evaluation_protocol.md`
- Full release checklist: `docs/release_checklist_full_benchmark.md`
- Minimal adapter example: `examples/minimal_adapter.py`
- Typed dashboard builder: `scripts/build_typed_dashboard.py`
- Release bundle builder: `scripts/build_claimtransfer_release_bundle.sh`
- Release bundle smoke test: `scripts/check_release_bundle.sh`
- Clean-checkout release-overlay smoke test:
  `scripts/check_release_overlay_checkout.sh`
- Hidden/private bundle generator: `scripts/build_hidden_private_bundle.py`
- Benchmark paper draft: `manuscripts/foundation_benchmark_dev/main.tex`

## Completed Verification

The following commands were run successfully:

```bash
python3 scripts/validate_task_cards.py
python3 scripts/build_claim_records.py
python3 scripts/build_score_report.py
python3 scripts/run_benchmark.py --quick
python3 scripts/run_benchmark.py --hidden
python3 scripts/run_benchmark.py --hidden --hidden-input claim_records/released_adapter_outputs.csv
python3 scripts/run_benchmark.py --mode public --skip-cross-method --skip-minisuite
python3 scripts/score_submission.py claim_records/released_adapter_outputs.csv --out-dir score_reports/submission_score --validate-task-cards
bash scripts/build_claimtransfer_release_bundle.sh
python3 scripts/build_hidden_private_bundle.py
bash scripts/check_release_bundle.sh
bash scripts/check_release_overlay_checkout.sh
```

Current generated scale:

- 21 validated task cards/templates.
- 117,114 normalized adapter-output rows.
- 117,114 official claim-record rows.
- 694 aggregate score-report rows.
- 255 coverage rows.
- Artifact manifest generated under `score_reports/benchmark_manifest.csv`
  with 57 official contract and report entries.

## P0 Status

Complete for an alpha benchmark artifact:

- task-card schema and validation;
- raw adapter outputs separated from official claim records;
- official pass/fail recomputed by scorer script;
- split and registry fields carried into score and coverage reports;
- quick/public/hidden runner modes;
- offline scoring harness.
- minimal participant adapter example;
- typed dashboard views by adapter, task, evidence object, scorer, and claim type.
- reviewer-facing release bundle under `artifacts/release/`.

## P1 Status

Implemented as benchmark infrastructure:

- adapter-family contract documented;
- scorer families documented;
- scorer sensitivity represented as scorer-indexed records;
- cross-method rows integrated into released adapter outputs when present;
- optional symbolic expression claim predicates added;
- task authoring and versioning rules documented.
- adapter fairness and compute-budget policy documented;
- hidden/offline evaluation protocol documented;
- full release checklist documented.
- release-bundle smoke test verifies that the bundled artifact can rebuild
  official reports without `results/revision`.
- release-overlay smoke test verifies that a clean source checkout becomes
  runnable after applying the generated benchmark bundle.

Still data-dependent:

- complete all GL cross-method and scorer-grammar cells;
- merge any newly pulled GL rows into `results/revision`;
- rerun `scripts/run_benchmark.py --quick` after each merge.
- confirm Hessian and TreeGate scorer rows after the remaining GL scorergram
  tasks finish.

## P2 Status

Implemented as offline benchmark scaffolding:

- submission format;
- offline submission scorer;
- hidden-template task cards;
- scientific/expression templates;
- statistical reporting policy;
- artifact manifest.
- release-bundle script for packaging ignored generated CSVs and documentation.
- hidden/private bundle generator that writes participant and private scoring
  registries plus a private seed manifest.
- clean release-bundle smoke test from a temporary directory without
  `results/revision`.
- clean source-checkout plus release-bundle overlay smoke test.

Still future work for a public benchmark release:

- optional hosted submission server;
- private hidden cards/seeds should be stored outside the public repository for
  a live evaluation, using the generated private bundle as the maintainer input.
- broader expression-equivalence scorers if symbolic-expression claims become a primary track.
- public release tag and frozen registry version.

## Paper Draft Status

The full benchmark draft is now compiled at:

```text
manuscripts/foundation_benchmark_dev/main.pdf
```

Current properties:

- 15 pages;
- no overfull boxes, undefined references, or citation warnings in the latest
  compile;
- current artifact scale synchronized in the draft:
  117,114 claim rows, 694 score rows, 255 coverage rows, 57 manifest entries;
- paper identity: official-scored benchmark contract, with pyKAN as the
  high-resolution case study and non-KAN rows as adapter-interface checks.
