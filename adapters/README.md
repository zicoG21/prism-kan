# Workflow Adapters

A workflow adapter exposes a method's native outputs as raw evidence objects.
Adapters do not need to mimic pyKAN, and they should not decide their own
pass/fail labels.  The official scorer maps these raw fields into benchmark
claim records.

Examples:

- pyKAN adapter: prediction, fitted-function pair scores, exposed readouts,
  selected-support refits, pruning support, symbolic status.
- Sparse library adapter: selected variables, nonzero interaction terms, term
  coefficients.
- GA2M-style adapter: selected univariate and bivariate components.
- Tree interaction adapter: fitted prediction plus H-statistic, residual pair
  scores, or gated candidate endpoint/pair sets.
- Symbolic library adapter: variables, operators, and pair terms present in the
  expression.

Missing evidence objects should be omitted or marked missing, not converted into
failures for a claim the workflow never exposed.  The quick reviewer path
materializes these adapter outputs in `claim_records/released_adapter_outputs.csv`.
