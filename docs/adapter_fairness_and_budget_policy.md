# Adapter Fairness and Budget Policy

ClaimTransfer compares structural claim records, not model leaderboard scores.
Even so, a full benchmark release needs a consistent policy for how adapters are
generated, tuned, and reported.

## Adapter Classes

| Adapter class | Native evidence | Tuning policy | Missing-field policy |
| --- | --- | --- | --- |
| pyKAN workflow | fitted function, readouts, support refit, pruning, symbolic status | fixed recipe per task-card family; public hyperparameters recorded in protocol field | omit objects that were not exposed; mark failed extraction separately |
| Sparse / spline library | selected terms, coefficients, prediction metrics | regularization chosen on training/validation prediction loss; structural thresholds fixed by task card | no readout or pruning fields unless the method exposes them |
| GA2M / EBM-style | univariate and bivariate components | component budget and interaction budget fixed before scoring | no symbolic fields unless exported expression terms exist |
| Tree / residual screens | prediction, H-statistic, residual pair ranks, candidate gates | tree depth, number of estimators, and gate size fixed per benchmark profile | candidate rows do not imply verifier rows |
| Symbolic library | expression variables, operators, pair terms, complexity | search budget recorded; expression constraints fixed by task family | expression-status rows are separate from support/pair rows |
| EPIM / TreeGate proposal | candidate endpoints/pairs and verifier scores | candidate budget grid fixed before inspection | proposal success does not count as verified pair success |

## Required Adapter Metadata

Every released adapter family should document:

- native output objects;
- training data and validation split;
- hyperparameter grid or fixed recipe;
- selection criterion;
- compute budget and hardware class;
- missing evidence objects;
- mapping from native outputs to normalized adapter-output rows;
- whether the row is a candidate, verifier, support, pair, symbolic, or
  prediction claim.

The official scorer recomputes `pass` from submitted evidence fields.  Adapter
code should not submit trusted final pass/fail labels.

## Budget Profiles

The benchmark uses profiles rather than one global compute budget:

- `quick`: rebuilds claim records and score reports from released outputs;
- `public-light`: lightweight public adapters or smoke-test rows;
- `public-full`: released public diagnostic rows with fixed seeds;
- `hidden-offline`: private seed/task rows scored with the same official scorer;
- `research-extended`: optional ablations such as scorer-grammar sweeps,
  readout taxonomies, or symbolic-expression metrics.

Coverage reports must identify the profile when rows from different budgets are
mixed.

## Fairness Rule

ClaimTransfer does not require all methods to expose the same objects.  It does
require that any reported structural claim name the object that licensed it.
For example, a tree candidate gate can pass a candidate-pair claim without
passing a verified-pair claim, and a symbolic library can pass expression-status
without passing sparse support.
