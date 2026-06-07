# Hidden Evaluation Protocol

ClaimTransfer-Bench 1.0 ships with both a public diagnostic suite and an
offline hidden/private evaluation mode.  The hidden mode uses the same adapter
submission format and official scorer as the public suite; only cards, seeds,
or labels are withheld by the maintainer until evaluation closes.

## Public vs Hidden Roles

| Split | Purpose | What is public | What can be hidden |
| --- | --- | --- | --- |
| Public diagnostic | debugging, paper figures, adapter development | task cards, formulas, seeds, labels, scorer predicates | nothing |
| Hidden offline | benchmark-style evaluation without a server | schema, adapter format, scoring binary/interface | private seeds, covariate draws, formula parameters, support labels |
| Private report / leaderboard | maintainer-run evaluation | submission contract and aggregate reports | task cards, labels, seeds, detailed row-level records until release |

## Offline Hidden Workflow

1. Maintainer keeps private task cards or private seed expansion outside the
   public repository.
2. Participant submits normalized adapter-output rows or a runnable adapter
   container.
3. Maintainer runs the same official scorer:

```bash
python scripts/run_benchmark.py --hidden --hidden-input path/to/private_submission.csv
```

4. The scorer emits:

- `claim_records/hidden_claim_records.csv`
- `score_reports/hidden_score_report.csv`
- `score_reports/hidden_coverage_table.csv`

5. Released public summaries report typed claim families, confidence intervals,
   missingness, and runtime metadata.  Row-level hidden records can be withheld
   until the evaluation closes.

Maintainers can check the generated participant/private split before release:

```bash
python scripts/build_hidden_private_bundle.py
python scripts/validate_hidden_bundle.py
```

## Leakage Rules

- A hidden card must not expose support labels or declared pairs to an adapter
  before evidence generation.
- If formula parameters are public but seeds are private, the split should be
  labeled `private_seed`.
- If formula family is public but formula parameters are private, the split
  should be labeled `private_card`.
- If labels are revealed for debugging, those rows become public diagnostic
  rows and should not be used for leaderboard claims.

## ClaimTransfer-Bench 1.0 Repository State

The repository contains:

- public cards: `task_cards/claimtransfer_v0_public.json`;
- hidden templates: `task_cards/claimtransfer_v0_hidden_template.json`;
- standard-formula cards: `task_cards/claimtransfer_v1_standard_formula_public.json`;
- hidden scoring path in `scripts/run_benchmark.py`;
- hidden/private bundle generator: `scripts/build_hidden_private_bundle.py`;
- leakage validator: `scripts/validate_hidden_bundle.py`;
- hidden score output paths ignored by git.

For a live benchmark event, the maintainer stores the real private-card or
private-seed bundle outside the public repository, runs the same hidden scoring
command, and releases only typed aggregate reports until the evaluation closes.
