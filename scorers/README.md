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

## Official Scorer Families

| Scorer family | Licensed claim | Notes |
| --- | --- | --- |
| standardized MSE | prediction adequacy | adequacy gate only, not structural recovery |
| ranked support / endpoint score | support and endpoint surfacing | reports rank, margin, and selected set |
| functional ANOVA | fitted-function pair reliance | official scorer for product-like pair cards |
| finite difference / Hessian | sensitivity pair evidence | separate scorer-indexed records |
| EPIM | KAN-native candidate proposal | proposal evidence, not verified pair recovery |
| candidate functional ANOVA | candidate-pair verifier | distinguishes proposal from verification |
| pruning support retention | pruning/symbolic support status | reports retained support size and endpoints |
| symbolic expression quality | expression-level claims | optional full-benchmark layer |

## Optional Symbolic Expression Claims

The workshop paper treats symbolic status as a workflow output.  The full
benchmark can add expression-level claim types without merging them with
support or pair claims:

- exact expression match;
- operator recovery;
- coefficient error;
- expression complexity;
- extrapolation error;
- dimensional or unit consistency when available.
