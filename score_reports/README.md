# Score Reports

A score report aggregates claim records by:

- task card;
- claim type;
- evidence object;
- workflow adapter.

Continuous fields are primary.  Binary predicate rates are reported with
confidence intervals.  The benchmark intentionally avoids one scalar leaderboard
that merges prediction, support, pair, pruning, and symbolic claims.

Generated workshop outputs:

- `score_report.csv`: official aggregate by task card, adapter, evidence object,
  claim type, scorer, and predicate.
- `coverage_table.csv`: compact coverage by adapter family, task family, and
  claim type, including rows, seeds, missing pass rows, success rate, Wilson
  interval, median rank, and median margin.
- `task_card_validation.csv`: machine-checkable validation summary for task
  cards.
