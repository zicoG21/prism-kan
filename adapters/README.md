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

## Adapter Families in the Full Benchmark

Each adapter family must document native outputs, mapping to normalized
evidence rows, hyperparameters, tuning policy, and missing-field behavior.

| Adapter family | Native output | Normalized evidence objects |
| --- | --- | --- |
| pyKAN | prediction, fitted function, readout scores, pruning, symbolic calls | prediction, full_function, exposed_readout, support_refit, pruning, symbolic |
| sparse library | selected variables, selected terms, coefficients | selected_support, pair_terms, symbolic_expression |
| GA2M / EBM-style | univariate and bivariate components | prediction, selected_support, component_pairs |
| tree interaction | prediction, H-statistic, residual screens, candidate gates | prediction, candidate_gate, candidate_pairs, verified_candidate_pairs |
| symbolic library | expression tree, variables, operators, term set | symbolic_expression, selected_support, pair_terms, expression_quality |
| EPIM / TreeGate | proposal endpoints/pairs plus verifier scores | candidate_pair, endpoints, pairverify_probe, pairverify_practical |

The full benchmark treats these as different workflow objects, not as a demand
that every method expose every field.

Contract files:

- `adapter_output_schema.json`: normalized adapter-output row schema.
- `docs/submission_format.md`: submission format and offline scoring path.
- `scripts/score_submission.py`: official offline scorer for a submitted CSV.
