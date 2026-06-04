# ClaimTransfer Statistical Reporting Policy

The benchmark reports typed claim evidence, not a single method-level success
number.  Statistical summaries must preserve that typing.

## Binary Predicate Summaries

- Report successes, trials, pass rate, and Wilson interval.
- Keep missing pass rows separate from failures.
- Do not pool heterogeneous claim types into one primary statistic.

## Continuous Evidence

Continuous fields are primary:

- standardized MSE for prediction;
- endpoint rank and endpoint-vs-nuisance margin;
- pair rank and true-minus-max-false margin;
- support size / selected support;
- retained support size;
- expression complexity or coefficient error for expression-level symbolic
  claims when available.

Report medians or quantiles for ranks and margins.  Binary thresholds are
derived summaries for planted controls, not the only evidence.

## Randomness Sources

When available, separate:

- training initialization randomness;
- train/test split randomness;
- scorer Monte Carlo randomness;
- bootstrap/subsample probe randomness;
- task-card hidden seed randomness.

## Coverage and Missingness

Every score report should include:

- rows;
- unique seeds;
- missing pass rows;
- successes/trials;
- pass rate with Wilson interval;
- median rank;
- median margin.

Coverage reports should be grouped by registry version, split, adapter family,
task family, and claim type.

## Scorer Dependence

Scorer disagreement is not a nuisance to hide.  Alternative scorers produce
separate scorer-indexed claim records.  Scorer sensitivity should be reported as
an audit result whenever it changes the structural claim verdict.
