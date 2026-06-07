# Split Overclaim Consistency

This report separates public diagnostic rows from standard-formula rows and hidden/private rows when present.  It is a stability check for whether overclaim risks are tied only to custom diagnostic cards.

| suite_split       | transfer_id                           | transfer                              | eligible_rows | source_passes | target_failures_given_source_pass | overclaim_risk |
| ----------------- | ------------------------------------- | ------------------------------------- | ------------- | ------------- | --------------------------------- | -------------- |
| public diagnostic | candidate_to_pair                     | Candidate -> pair                     | 5630          | 3919          | 1424                              | 36.3%          |
| public diagnostic | fitted_pair_to_pruning                | Fitted pair -> pruning                | 2884          | 2211          | 460                               | 20.8%          |
| public diagnostic | fitted_pair_to_readout                | Fitted pair -> readout                | 2884          | 2211          | 566                               | 25.6%          |
| public diagnostic | prediction_to_pair                    | Prediction -> pair                    | 15893         | 9062          | 2471                              | 27.3%          |
| public diagnostic | support_to_prediction                 | Support -> prediction                 | 10129         | 5252          | 1479                              | 28.2%          |
| public diagnostic | symbolic_status_to_expression_quality | Symbolic status -> expression quality | 1982          | 1982          | 1742                              | 87.9%          |
| standard formula  | prediction_to_pair                    | Prediction -> pair                    | 10077         | 8978          | 1472                              | 16.4%          |
| standard formula  | support_to_prediction                 | Support -> prediction                 | 12734         | 12014         | 877                               | 7.3%           |
| standard formula  | symbolic_status_to_expression_quality | Symbolic status -> expression quality | 6150          | 6150          | 5211                              | 84.7%          |
