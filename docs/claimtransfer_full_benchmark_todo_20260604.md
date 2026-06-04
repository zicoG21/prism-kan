# ClaimTransfer Full Benchmark TODO

Updated: 2026-06-04

This list is for the post-workshop, full-benchmark version.  The goal is to
turn ClaimTransfer from a workshop audit protocol into an official-scored,
reproducible, extensible benchmark.

## Benchmark Identity

The full version should make one stronger claim than the workshop paper:

> ClaimTransfer is an official-scored benchmark for structural claim transfer:
> methods submit raw evidence objects, and benchmark scorers produce claim
> records, score reports, coverage reports, and held-out diagnostics.

This is different from the workshop identity:

> ClaimTransfer is an audit protocol, demonstrated through a pyKAN case study
> and supported by interface checks.

## P0: Required for a Credible Full Benchmark

### 1. Freeze the Machine-Readable Task-Card Schema

- Define a JSON schema for task cards.
- Validate required fields:
  - task id, family, formula/covariate generator;
  - dimensions, train/test sizes, seed policy;
  - active support;
  - legal claim grammar;
  - official scorers and derived predicates;
  - public/hidden split;
  - stress tags and intended overclaim mode.
- Make `scripts/validate_task_cards.py` fail loudly on missing or inconsistent fields.
- Add at least one complete example card for each task family.

Acceptance check:

```bash
python scripts/validate_task_cards.py
```

must produce a clean validation table in `score_reports/task_card_validation.csv`.

### 2. Separate Raw Adapter Outputs From Official Claim Records

- Treat adapter outputs as raw submissions.
- Ensure adapters do not set final pass/fail labels.
- Define a stable adapter-output schema:
  - predictions or prediction metrics;
  - selected support;
  - pair score vectors or candidate pair sets;
  - readout/endpoint scores;
  - retained support after pruning;
  - symbolic expression/status;
  - missing fields marked explicitly.
- Keep `claim_records/released_adapter_outputs.csv` as the public example.

Acceptance check:

```bash
python scripts/build_claim_records.py
```

must rebuild `claim_records/released_claim_records.csv` from raw adapter
outputs and task cards.

### 3. Make Official Scoring the Source of Truth

- Implement scorer modules for:
  - prediction adequacy;
  - support recovery;
  - endpoint ranks and margins;
  - pair ranks and margins;
  - pruning support size and endpoint retention;
  - symbolic status;
  - candidate-pair and verifier-pair claims.
- Ensure scorer-indexed rows are separate records, not overwritten verdicts.
- Add explicit support for fANOVA, finite difference, Hessian, EPIM, and TreeGate-style candidate/verifier rows.
- Add tests or sanity checks showing that changing scorer changes the claim record, not the task truth.

Acceptance check:

```bash
python scripts/build_score_report.py
```

must regenerate:

- `score_reports/score_report.csv`
- `score_reports/coverage_table.csv`
- Wilson intervals / quantiles / missingness fields

### 4. Produce a Full Coverage Report

- Report coverage by:
  - adapter family;
  - task family;
  - claim type;
  - evidence object;
  - scorer;
  - public vs hidden split.
- Include:
  - number of task cards;
  - number of seeds;
  - number of claim rows;
  - missing rows;
  - pass rate with CI;
  - median rank;
  - median margin;
  - runtime/compute budget if available.
- Avoid one scalar leaderboard that merges prediction/support/pair/symbolic claims.

Acceptance check:

`score_reports/coverage_table.csv` should be complete enough to generate both
paper tables and reviewer artifact summaries without manual relabeling.

### 5. Add Public and Hidden Splits

- Keep public diagnostic cards for debugging and workshop-style audit examples.
- Add hidden or held-out task cards/private seeds for benchmark-style evaluation.
- Decide what is hidden:
  - formula cards;
  - seeds;
  - covariate geometries;
  - scoring thresholds;
  - or some combination.
- Ensure official scoring works identically for public and hidden splits.

Acceptance check:

`task_cards/` should contain public registry plus hidden/private registry
templates or encrypted/private placeholders, and the score report should carry a
`split` field.

### 6. One-Command Reviewer Path

- Make `scripts/run_benchmark.py` the official quick path.
- It should:
  - print environment;
  - validate task cards;
  - rebuild claim records;
  - rebuild score reports;
  - rebuild coverage tables;
  - optionally rebuild selected paper figures/tables.
- Add runtime modes:
  - `--quick`: no retraining, rebuild reports from released outputs;
  - `--public`: rerun lightweight public adapters;
  - `--full`: use all available generated outputs;
  - `--hidden`: score hidden submissions if available.

Acceptance check:

```bash
python scripts/run_benchmark.py --quick
```

should complete on a reviewer machine without GPU.

## P1: Needed for a Stronger Benchmark Paper

### 7. Expand and Stabilize Adapter Families

Minimum adapter set:

- pyKAN full workflow;
- sparse linear / sparse polynomial;
- GA2M / EBM-style additive-pair model;
- tree interaction scorer;
- symbolic regression/library adapter;
- EPIM / TreeGate candidate-verifier adapter.

For each adapter:

- document native outputs;
- document hyperparameters;
- document tuning policy;
- document missing-field behavior;
- include at least one positive-control and one stress-card row.

Current local support:

- adapter-family policy: `docs/adapter_fairness_and_budget_policy.md`;
- minimal adapter example: `examples/minimal_adapter.py`;
- offline scorer: `scripts/score_submission.py`.

### 8. Improve Scorer Sensitivity Coverage

- Make scorer sensitivity a formal benchmark result.
- Run/aggregate:
  - fANOVA;
  - finite difference;
  - Hessian;
  - EPIM;
  - TreeGate candidate + verifier.
- Report when scorers agree, when they diverge, and which task grammar each scorer is allowed to license.

Potential GL jobs:

- `kan-a40-scorergram` variants for missing Hessian / TreeGate verifier cards.
- CPU `kan-treegate` standard jobs for candidate-generation sweeps.

### 9. Complete Cross-Method Standard Outputs

- Finish cross-method standard jobs for all public cards and core methods.
- Verify outputs have consistent columns and no partial timeouts.
- Merge extended rows into the official adapter-output file.
- Generate a full, not curated, cross-adapter coverage table.

Potential GL jobs:

- `greatlakes_cross_method_transfer_baselines_extended.sbatch`
- score-refresh standard job after merge.

### 10. Strengthen Symbolic Evaluation Layer

Workshop version only audits symbolic status and support retention.  Full
benchmark should add expression-level optional claims:

- expression tree equivalence;
- operator recovery;
- coefficient error;
- simplified support;
- expression complexity;
- extrapolation error;
- dimensional/unit consistency if task supports it.

Keep these as separate symbolic claim types, not replacements for support/pair
claims.

### 11. Add Task-Authoring Rules and Governance

- Write rules for:
  - support declaration;
  - pair declaration;
  - multi-pair formula cards;
  - nested/compositional stress cards;
  - symbolic claims;
  - correlated-covariate cards;
  - alternative valid grammars.
- Add versioning:
  - `claimtransfer-v0-public`;
  - `claimtransfer-v0-hidden`;
  - future `v1`.
- Make any grammar changes produce a new version, not silent edits.

## P2: Main-Conference-Level Additions

### 12. Submission Format and Leaderboard-Style Report

- Define what a participant submits:
  - adapter output CSV/JSON;
  - metadata;
  - runtime;
  - method description;
  - optional model artifacts.
- Official scorer generates:
  - claim records;
  - score report;
  - coverage report;
  - missingness report.
- If a leaderboard exists, report by claim type rather than one merged scalar.

### 13. Hidden Evaluation Server or Offline Harness

- Build a small offline harness first.
- Later convert to a submission server if needed.
- Enforce no access to hidden formula labels during adapter generation.

Current local support:

- hidden scoring mode: `python scripts/run_benchmark.py --hidden --hidden-input ...`;
- protocol document: `docs/hidden_evaluation_protocol.md`;
- remaining gap: real private cards/seeds held outside the public repo.

### 14. More Realistic Scientific Cards

- Add real-covariate and semi-synthetic cards with known planted structure.
- Add symbolic-regression benchmark cards with expression-level claims.
- Add correlated features, heteroskedastic noise, nuisance maxima, and high-dimensional stress cards.

### 15. Statistical Reporting Policy

- Standardize uncertainty:
  - Wilson intervals for binary summaries;
  - bootstrap intervals for ranks/margins;
  - missingness intervals;
  - separate training randomness, scorer Monte Carlo randomness, and data split randomness.
- Decide minimum seed counts per card family for public diagnostic and hidden evaluation.

Current local support:

- Wilson intervals and missingness in `scripts/build_score_report.py`;
- release check: `docs/release_checklist_full_benchmark.md`.

## GL Work Queue

### High Value

- Hidden/public claim-card completion on A40.
- Scorer-grammar completion with Hessian and TreeGate verifier rows.
- Cross-method extended standard jobs.
- Score-refresh jobs after every merge.

### Medium Value

- Semi-synthetic drilldown.
- Prune/symbolic expression-quality expansion.
- Readout taxonomy.
- Additional full-KAN capacity rows if they fill a specific missing coverage cell.

### Low Value Unless Needed for a Coverage Gap

- More pyKAN seeds for already saturated rows.
- Extra width/grid sweeps without a specific claim-grammar purpose.
- New plots before score reports are stable.

## Recommended Next Milestones

### Milestone A: Workshop-Safe Artifact

Target: 1 day.

- Freeze current workshop PDF.
- Ensure quick path runs.
- Ensure generated score reports match paper tables.
- Do not expand paper claims.

### Milestone B: Full Benchmark Alpha

Target: 3--5 focused days.

- JSON schema finalized.
- Official scorer rebuilds claim records from raw outputs.
- Coverage table complete for public cards.
- Public/hidden split fields exist.
- Adapter docs cover all released methods.

### Milestone C: Full Benchmark Beta

Target: 1--2 weeks.

- Hidden/private split operational.
- Cross-method rows complete enough for a benchmark paper.
- Symbolic expression metrics added as optional claim types.
- One-command run and artifact docs tested from a clean checkout.

### Milestone D: Strong Paper Version

Target: after Beta.

- Rewrite as official-scored benchmark paper.
- Keep workshop audit narrative as motivation.
- Main result is coverage/score-report maturity, not pyKAN failure alone.
