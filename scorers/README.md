# Scorers

Scorers define how claim records are evaluated.

The core score fields are continuous:

- standardized prediction MSE;
- support size and support F1;
- endpoint ranks and endpoint-vs-nuisance margins;
- declared pair rank and true-minus-max-false pair margin;
- pruning retained support size and endpoint retention;
- symbolic status when the workflow exposes it.

Binary predicates are derived summaries for planted-control tasks.  Pair claims
are scorer-indexed: changing the scorer changes the claim record.

