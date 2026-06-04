# Scorers

Scorers define how raw adapter evidence becomes official claim records.

The core score fields are continuous:

- standardized prediction MSE;
- support size and support F1;
- endpoint ranks and endpoint-vs-nuisance margins;
- declared pair rank and true-minus-max-false pair margin;
- pruning retained support size and endpoint retention;
- symbolic status when the workflow exposes it.

Binary predicates are derived summaries for planted-control tasks.  Pair claims
are scorer-indexed: changing the scorer changes the claim record.
Alternative scorers should be materialized as separate scorer-indexed rows, not
used to overwrite the official task-card verdict.

The workshop quick path uses:

```bash
python scripts/build_claim_records.py
python scripts/build_score_report.py
```

The first command materializes standardized raw evidence rows; the second
recomputes `pass/fail`, Wilson intervals, score reports, and coverage tables.
