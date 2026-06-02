# Workflow Adapters

A workflow adapter maps a method's native outputs into benchmark claim records.
Adapters do not need to mimic pyKAN.

Examples:

- pyKAN adapter: prediction, fitted-function pair scores, exposed readouts,
  selected-support refits, pruning support, symbolic status.
- Sparse library adapter: selected variables, nonzero interaction terms, term
  coefficients.
- GA2M-style adapter: selected univariate and bivariate components.
- Tree interaction adapter: fitted prediction plus H-statistic or residual pair
  scores.
- Symbolic library adapter: variables, operators, and pair terms present in the
  expression.

Missing evidence objects should be omitted or marked missing, not converted into
failures for a claim the workflow never exposed.

