# Standard-Formula Expression Equivalence Subset

Optional numerical expression-equivalence check for symbolic adapters that expose final expressions.  Near equivalence uses held-out MSE < 0.05; exact-like equivalence uses held-out MSE < 1e-8.

| adapter_family   | adapter                    | rows | evaluated_rows | near_equivalence_successes | near_equivalence_rate_mse_lt_005 | exact_like_successes_mse_lt_1e8 | exact_like_rate_mse_lt_1e8 | median_expression_mse |
| ---------------- | -------------------------- | ---- | -------------- | -------------------------- | -------------------------------- | ------------------------------- | -------------------------- | --------------------- |
| symbolic_library | gplearn_symbolic_regressor | 840  | 840            | 98                         | 11.7%                            | 0                               | 0.0%                       | 0.1991                |
| symbolic_library | pysr_symbolic_regressor    | 5190 | 5190           | 4891                       | 94.2%                            | 1624                            | 31.3%                      | 3.851e-05             |
