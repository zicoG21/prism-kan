# ClaimTransfer Full Benchmark Status

Updated: 2026-06-04

## Completed Core Contract

- Release contract: `benchmark_release.json`
- Task-card schema: `task_cards/task_card_schema.json`
- Public registry: `task_cards/claimtransfer_v0_public.json`
- Hidden-template registry: `task_cards/claimtransfer_v0_hidden_template.json`
- Scientific/expression templates: `task_cards/claimtransfer_v1_scientific_templates.json`
- Adapter-output schema: `adapters/adapter_output_schema.json`
- Adapter-family registry: `adapters/adapter_family_registry.json`
- Submission metadata schema: `adapters/submission_metadata_schema.json`
- Claim-record schema: `claim_records/claim_record_schema.json`
- Adapter-output validator: `scripts/validate_adapter_outputs.py`
- Claim-record validator: `scripts/validate_claim_records.py`
- Score-report validator: `scripts/validate_score_reports.py`
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
- Coverage-gap report builder: `scripts/build_coverage_gap_report.py`
- Coverage-gap priority summarizer: `scripts/summarize_coverage_gaps.py`
- Coverage-gap action-plan builder:
  `scripts/build_coverage_gap_action_plan.py`
- Great Lakes result refresh helper:
  `scripts/refresh_from_greatlakes_results.py`
- Great Lakes coverage-gap submit helper:
  `scripts/submit_claimtransfer_gapfill_gl.sh`
- Great Lakes gap-fill jobs:
  `scripts/greatlakes_cross_method_gapfill_standard.sbatch` and
  `scripts/greatlakes_treegate_gapfill_standard.sbatch`
- Release bundle builder: `scripts/build_claimtransfer_release_bundle.sh`
- Release bundle smoke test: `scripts/check_release_bundle.sh`
- Clean-checkout release-overlay smoke test:
  `scripts/check_release_overlay_checkout.sh`
- Hidden/private bundle generator: `scripts/build_hidden_private_bundle.py`
- Hidden/private leakage validator: `scripts/validate_hidden_bundle.py`
- Release-contract validator: `scripts/validate_release_contract.py`
- Full benchmark readiness report:
  `scripts/build_full_benchmark_readiness_report.py`
- Release-candidate gate report:
  `scripts/build_release_candidate_report.py`
- Benchmark paper draft: `manuscripts/foundation_benchmark_dev/main.tex`

## Completed Verification

The following commands were run successfully:

```bash
python3 scripts/validate_release_contract.py
python3 scripts/validate_task_cards.py
python3 scripts/validate_adapter_registry.py
python3 scripts/validate_submission_metadata.py examples/minimal_submission_metadata.json
python3 scripts/validate_adapter_outputs.py claim_records/released_adapter_outputs.csv
python3 scripts/build_claim_records.py
python3 scripts/build_score_report.py
python3 scripts/validate_claim_records.py claim_records/released_claim_records.csv
python3 scripts/validate_score_reports.py
python3 scripts/summarize_coverage_gaps.py
python3 scripts/build_coverage_gap_action_plan.py
python3 scripts/build_full_benchmark_readiness_report.py
python3 scripts/build_release_candidate_report.py
python3 scripts/refresh_from_greatlakes_results.py --no-unpack
python3 scripts/run_benchmark.py --quick
python3 scripts/run_benchmark.py --hidden
python3 scripts/run_benchmark.py --hidden --hidden-input claim_records/released_adapter_outputs.csv
python3 scripts/run_benchmark.py --mode public --skip-cross-method --skip-minisuite
python3 scripts/score_submission.py claim_records/released_adapter_outputs.csv --out-dir score_reports/submission_score --validate-task-cards
bash scripts/build_claimtransfer_release_bundle.sh
python3 scripts/build_hidden_private_bundle.py
python3 scripts/validate_hidden_bundle.py
bash scripts/check_release_bundle.sh
bash scripts/check_release_overlay_checkout.sh
```

Current generated scale:

- 21 validated task cards/templates.
- Release-contract validation table generated from the frozen alpha contract: 33 checks.
- 6 validated adapter families.
- Raw adapter-output validation table generated from the schema: 12 required fields checked.
- 168,056 normalized adapter-output rows.
- 168,056 official claim-record rows.
- Claim-record validation table generated from the schema: 9 required fields
  plus pass-value validity checked.
- 1,149 aggregate score-report rows.
- 540 coverage rows.
- Score/coverage/gap/missingness report validation generated from reporting-policy checks: 36 checks.
- Claim-grammar-aware coverage-gap report generated from adapter-family
  contracts and public task families: 337 expected cells, 336 covered and 1
  missing.
- 744 missingness-report rows.
- Full benchmark readiness report generated under
  `score_reports/full_benchmark_readiness.csv` with 20 P0/P1/P2 checks:
  P0 and P1 are complete for the alpha artifact; P2 offline scaffolding is
  complete with public release/tag marked as future work.
- Release-candidate gate report generated under
  `score_reports/release_candidate_report.csv`; it separates alpha packaging
  readiness from P1 data blockers and the later public tag/server step.
- Coverage-gap priority summary generated under
  `score_reports/coverage_gap_summary.csv` for GL/merge planning.
- Coverage-gap action plan generated under
  `score_reports/coverage_gap_action_plan.csv`; it maps missing cells to the
  existing GL gap-fill scripts, pyKAN merge steps, or optional symbolic tracks.
- Artifact manifest generated under `score_reports/benchmark_manifest.csv`
  with 61 official contract and report entries.

Workshop-paper status:

- the workshop audit/protocol PDF is treated as temporarily frozen;
- full-benchmark work should not change the workshop story unless a future
  paper pass explicitly reopens it.

## P0 Status

Complete for an alpha benchmark artifact:

- task-card schema and validation;
- task-card governance checks for unique claim ids, recognized claim grammar,
  scorer/predicate declarations, support/endpoint/pair target consistency, and
  threshold/budget validity;
- raw adapter outputs separated from official claim records;
- official pass/fail recomputed by scorer script;
- official claim-record schema validation after scoring;
- split and registry fields carried into score and coverage reports;
- quick/public/hidden runner modes;
- offline scoring harness.
- minimal participant adapter example;
- typed dashboard views by adapter, task, evidence object, scorer, and claim type.
- coverage-gap report showing which adapter-family/task-family/claim-type cells
  are covered or missing.
- coverage-gap action plan mapping each remaining missing-cell group to a GL
  submit command, merge step, or explicit manual-triage bucket.
- score-report validation for required columns, Wilson intervals, unit
  intervals, and nonnegative counts.
- official missingness report for omitted or non-scorable evidence objects.
- readiness report showing 10/10 P0 checks complete.
- reviewer-facing release bundle under `artifacts/release/`.

## P1 Status

Implemented as benchmark infrastructure:

- adapter-family contract documented;
- scorer families documented;
- scorer sensitivity represented as scorer-indexed records;
- cross-method rows integrated into released adapter outputs when present;
- optional symbolic expression claim predicates implemented in the official
  scorer and exercised by the minimal adapter;
- task authoring and versioning rules documented.
- adapter fairness and compute-budget policy documented;
- hidden/offline evaluation protocol documented;
- full release checklist documented.
- release-bundle smoke test verifies that the bundled artifact can rebuild
  official reports without `results/revision`.
- release-overlay smoke test verifies that a clean source checkout becomes
  runnable after applying the generated benchmark bundle.

Still data-dependent:

- P1 has no remaining data-dependent coverage blocker.
- the only remaining expected-cell gap is the P2 optional
  `symbolic_library/scientific_expression/symbolic_operator_recall` track.
- public release/tag/server work remains explicitly marked as P2 future work.
- use `score_reports/coverage_gap_summary.csv` to choose the next GL/merge
  target by adapter family and claim type.
- use `score_reports/coverage_gap_action_plan.csv` to check whether each gap is
  already targeted by current GL scripts or still needs manual triage.
- targeted gap-fill scripts now exist for missing public task families:
  `bash scripts/submit_claimtransfer_gapfill_gl.sh` submits CPU-only
  cross-method and TreeGate rows for exp/log/sqrt/trig/three-way,
  scientific-expression, correlated-covariate, and semi-synthetic
  real-covariate gaps, with a dependent official score refresh.  This is
  preferred over re-running broad scorergram arrays when the missing cells are
  adapter-family coverage gaps.

## P2 Status

Implemented as offline benchmark scaffolding:

- submission format;
- submission metadata schema and validator;
- offline submission scorer;
- hidden-template task cards;
- scientific/expression templates;
- statistical reporting policy;
- artifact manifest.
- release-bundle script for packaging ignored generated CSVs and documentation.
- hidden/private bundle generator that writes participant and private scoring
  registries plus a private seed manifest.
- hidden/private leakage validator that checks participant cards withhold
  formulas, supports, claim targets, and private seed blocks.
- clean release-bundle smoke test from a temporary directory without
  `results/revision`.
- clean source-checkout plus release-bundle overlay smoke test.
- readiness report showing offline submission, hidden/private bundle, and
  release packaging checks complete.
- release-candidate gate report showing which pre-tag gates are green and which
  data-dependent gates still wait on GL result merges.

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
  168,056 claim rows, 1,149 score rows, 540 coverage rows, 744 missingness rows,
  61 manifest entries;
- paper identity: official-scored benchmark contract, with pyKAN as the
  high-resolution case study and non-KAN rows as adapter-interface checks.
