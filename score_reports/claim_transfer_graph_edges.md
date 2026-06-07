# Claim-Transfer Graph Edges

Each edge is weighted by pooled overclaim risk from the official claim records.

| source_node     | target_node        | transfer_id                           | transfer                              | source_passes | target_failures_given_source_pass | overclaim_risk | wilson_low | wilson_high | edge_label                                    |
| --------------- | ------------------ | ------------------------------------- | ------------------------------------- | ------------- | --------------------------------- | -------------- | ---------- | ----------- | --------------------------------------------- |
| symbolic status | expression quality | symbolic_status_to_expression_quality | symbolic-status -> expression quality | 8132          | 6953                              | 0.855          | 0.847      | 0.863       | symbolic-status -> expression quality (85.5%) |
| candidate       | pair               | candidate_to_pair                     | candidate -> pair                     | 3919          | 1424                              | 0.363          | 0.348      | 0.379       | candidate -> pair (36.3%)                     |
| fitted pair     | readout            | fitted_pair_to_readout                | fitted pair -> readout endpoints      | 2211          | 566                               | 0.256          | 0.238      | 0.275       | fitted pair -> readout endpoints (25.6%)      |
| prediction      | pair               | prediction_to_pair                    | prediction -> pair                    | 18040         | 3943                              | 0.219          | 0.213      | 0.225       | prediction -> pair (21.9%)                    |
| fitted pair     | pruning            | fitted_pair_to_pruning                | fitted pair -> pruning endpoints      | 2211          | 460                               | 0.208          | 0.192      | 0.225       | fitted pair -> pruning endpoints (20.8%)      |
| support         | prediction         | support_to_prediction                 | support -> prediction                 | 17266         | 2356                              | 0.136          | 0.131      | 0.142       | support -> prediction (13.6%)                 |
