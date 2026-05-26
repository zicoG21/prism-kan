# KAN-Native Stability Validation Findings

Source run: `results/innovation_loop/strict_validation_20260526_011917`

Analysis artifacts:

- `results/innovation_loop/strict_validation_20260526_011917/analysis/strict_validation_key_regimes.csv`
- `results/innovation_loop/strict_validation_20260526_011917/analysis/strict_validation_failure_modes.csv`
- `results/innovation_loop/strict_validation_20260526_011917/analysis/strict_validation_supports.csv`
- `results/innovation_loop/strict_validation_20260526_011917/analysis/c025_boundary_validation.pdf`
- `results/innovation_loop/strict_validation_20260526_011917/analysis/high_dim_stress_validation.pdf`

## Executive Judgment

The KAN-native stability direction is no longer just a speculative repair. Under disjoint probe/evaluation seeds, internal KAN feature stability and feature+edge hybrid selection reliably recover the true support in `d=100` weak-to-moderate regimes and improve interaction recovery over the earlier SS-KAN runs.

The result should still be positioned carefully: it shifts the recovery boundary, but it does not remove the high-dimensional support bottleneck. At `d=1000`, even strong interactions can fail because the stable support no longer retains the true interaction endpoints.

## Main Empirical Pattern

For `core_interaction_c025`:

| Regime | Best observed method | Interaction F1 | Endpoint/support status |
|---|---:|---:|---|
| `d=100,n=512,top_m=4` | `feature_edge_hybrid` / `grad_stability_var` | `0.80` | true pair support retained |
| `d=100,n=1024,top_m=5` | `feature_edge_hybrid` | `0.80` | true pair support retained |
| `d=100,n=1024,top_m=6` | `grad_stability_var` | `0.90` | true pair support retained |
| `d=100,n=2048,top_m=6` | `edge_stability_var` | `0.90` | true pair support retained |
| `d=500,n=2048,top_m=6` | `feature_edge_hybrid` | `0.50` | true pair support retained |
| `d=1000,n=2048,top_m=6` | `feature_stability_var` / `feature_edge_hybrid` | `0.00` | true pair support not retained |

For stronger interactions:

- `core_interaction_c05`, `d=100,n=1024`: `feature_edge_hybrid` reaches interaction F1 `1.0`.
- `core_interaction_c1`, `d=500,n=1024`: both `feature_stability_var` and `feature_edge_hybrid` reach interaction F1 `1.0`.
- `core_interaction_c1`, `d=1000,n=1024`: both drop to interaction F1 `0.0`, indicating a dimension-driven support failure.

For Feynman-style formulas embedded in `d=100`:

- `feynman_energy`, `feynman_coulomb`, and `feynman_gravity` all reach interaction F1 `1.0` for both `feature_stability_var` and `feature_edge_hybrid`.

## Mechanistic Reading

The strict validation reveals two distinct failure modes.

First, in `d=100`, the stability-selected support often already contains `[0,1,2,3]`. When interaction recovery still fails, the top-ranked pair is usually `(0,1)` rather than the true `(2,3)`. This is a pair-ranking failure after support recovery, not a support-retention failure.

Second, in `d=500/1000`, failures increasingly become support failures again. For example, at `c=0.25,d=1000,n=2048`, the selected supports omit the true interaction endpoints and interaction F1 is `0.0`.

This gives the paper a sharper ladder:

```text
prediction -> stable support -> endpoint retention -> pair ranking
```

## Method Positioning

The strongest defensible method name is:

```text
KAN Feature-Stability Selection
```

Main variant:

```text
feature_stability_var
```

Primary ablation:

```text
feature_edge_hybrid
```

The edge-only and edge-pair methods are useful ablations, but the strict run suggests the simplest KAN feature-stability variant is often as strong as the more elaborate hybrid. That is good news for paper clarity.

## Claims We Can Make

Safe claim:

> KAN-native feature-stability selection improves formula-level support and interaction recovery in moderate high-dimensional sparse-interaction regimes, especially at `d=100`, while still failing under extreme nuisance dimension or weak signal.

Stronger but still plausible claim after baseline controls finish:

> The method shifts the empirical formula-fidelity recovery boundary without relying on external tree-based screening.

Claims to avoid:

- It does not solve arbitrary high-dimensional KAN support discovery.
- It is not a symbolic extraction method.
- It is not a proof of KAN sample complexity.
- It should not be described as universally better than RF until same-seed controls are complete.

## Baseline/Control Runs Added

The original raw-heavy baseline run was stopped because full-dimensional finite-difference pair scoring is too slow for `d=100` and above:

```text
results/innovation_loop/strict_baseline_controls_20260526_103933
```

The replacement screened-control run completed and avoids raw full-dimensional pair scoring:

```text
results/innovation_loop/strict_screened_baseline_controls_20260526_104243
```

It runs RF, oracle-support, random-support, and exclude-endpoints controls on the same key regimes as the strict KAN-native validation. Partial results already show that at `core_interaction_c025,d=100,n=512,top_m=4`, RF-screened KAN can miss both interaction endpoints and produce interaction F1 `0.0`, while oracle-support KAN is nontrivial but not perfect because the refit pair ranking can still choose the wrong pair.

The first completed same-seed control block is:

| Regime | Method | Endpoint support | Interaction F1 | Test MSE |
|---|---|---:|---:|---:|
| `c=0.25,d=100,n=512,m=4` | KAN feature+edge stability | `1.0` | `0.80` | `2.48e-4` |
| `c=0.25,d=100,n=512,m=4` | KAN feature stability | `1.0` | `0.70` | `2.54e-4` |
| `c=0.25,d=100,n=512,m=4` | Oracle-support KAN | `1.0` | `0.60` | `2.57e-4` |
| `c=0.25,d=100,n=512,m=4` | RF-screened KAN | `0.0` | `0.00` | `1.30e-2` |
| `c=0.25,d=100,n=512,m=4` | Random-support KAN | `0.0` | `0.00` | `2.75` |
| `c=0.25,d=100,n=512,m=4` | Exclude-endpoints KAN | `0.0` | `0.00` | `2.79` |

Interpretation: this is the first clean evidence that KAN-native stability can beat RF screening in a weak-interaction sparse-support regime, while still revealing that even oracle support does not guarantee perfect pair ranking.

Caveat: do not claim that KAN-native stability is intrinsically better than oracle support. In these blocks, native stability and oracle support often refit on the same selected variables `[0,1,2,3]`; small differences in interaction F1 come from refit/pair-ranking stochasticity and numerical sensitivity. The paper-safe claim is that KAN-native stability reaches the oracle-support regime without external screening, while RF screening does not retain the interaction endpoints.

The second completed block confirms the same pattern at larger `n` with the same `top_m`:

| Regime | Method | Endpoint support | Interaction F1 | Test MSE |
|---|---|---:|---:|---:|
| `c=0.25,d=100,n=1024,m=4` | KAN feature+edge stability | `1.0` | `0.70` | `2.65e-4` |
| `c=0.25,d=100,n=1024,m=4` | KAN feature stability | `1.0` | `0.60` | `2.66e-4` |
| `c=0.25,d=100,n=1024,m=4` | Oracle-support KAN | `1.0` | `0.40` | `2.60e-4` |
| `c=0.25,d=100,n=1024,m=4` | RF-screened KAN | `0.0` true-pair retention, `0.05` endpoint recall | `0.00` | `1.22e-2` |
| `c=0.25,d=100,n=1024,m=4` | Random-support KAN | `0.0` | `0.00` | `1.60` |
| `c=0.25,d=100,n=1024,m=4` | Exclude-endpoints KAN | `0.0` | `0.00` | `1.60` |

This strengthens the claim that the weak interaction endpoints are nearly invisible to RF marginal screening even when prediction improves. In 10 RF seeds, only one selected a single interaction endpoint and none selected both endpoints.

The third completed block shows that simply increasing the screening budget to `top_m=5` is still not enough for RF:

| Regime | Method | Endpoint support | Interaction F1 | Test MSE |
|---|---|---:|---:|---:|
| `c=0.25,d=100,n=1024,m=5` | KAN feature+edge stability | `1.0` | `0.80` | `2.35e-4` |
| `c=0.25,d=100,n=1024,m=5` | KAN feature stability | `1.0` | `0.70` | `1.51e-3` |
| `c=0.25,d=100,n=1024,m=5` | Oracle-support KAN | `1.0` | `0.60` | `3.10e-3` |
| `c=0.25,d=100,n=1024,m=5` | RF-screened KAN | `0.0` | `0.00` | `1.23e-2` |
| `c=0.25,d=100,n=1024,m=5` | Random-support KAN | `0.0` pair retention, `0.15` endpoint recall | `0.00` | `1.89` |
| `c=0.25,d=100,n=1024,m=5` | Exclude-endpoints KAN | `0.0` | `0.00` | `1.88` |

Across 10 RF seeds at `top_m=5`, the selected support was always `[0,1]` plus three nuisance variables. No interaction endpoint was selected.

The fourth completed block shows the same conclusion at `top_m=6`:

| Regime | Method | Endpoint support | Interaction F1 | Test MSE |
|---|---|---:|---:|---:|
| `c=0.25,d=100,n=1024,m=6` | KAN feature stability | `1.0` | `0.75` | `1.46e-3` |
| `c=0.25,d=100,n=1024,m=6` | KAN feature+edge stability | `1.0` | `0.55` | `2.46e-4` |
| `c=0.25,d=100,n=1024,m=6` | Oracle-support KAN | `1.0` | `0.60` | `2.24e-4` |
| `c=0.25,d=100,n=1024,m=6` | RF-screened KAN | `0.0` true-pair retention, `0.05` endpoint recall | `0.00` | `1.24e-2` |

Even at `top_m=6`, RF selected only one endpoint in one out of ten seeds and never retained both interaction endpoints.

The fifth completed block at larger sample size shows that RF begins to recover occasionally but still lags far behind KAN-native stability:

| Regime | Method | Endpoint support | Interaction F1 | Test MSE |
|---|---|---:|---:|---:|
| `c=0.25,d=100,n=2048,m=6` | KAN edge stability | `1.0` | `0.90` | `3.73e-4` |
| `c=0.25,d=100,n=2048,m=6` | KAN feature stability | `1.0` | `0.80` | `2.03e-4` |
| `c=0.25,d=100,n=2048,m=6` | KAN feature+edge stability | `1.0` | `0.80` | `1.99e-4` |
| `c=0.25,d=100,n=2048,m=6` | Oracle-support KAN | `1.0` | `0.80` | `3.56e-4` |
| `c=0.25,d=100,n=2048,m=6` | RF-screened KAN | `0.10` true-pair retention, `0.25` endpoint recall | `0.10` | `1.07e-2` |
| `c=0.25,d=100,n=2048,m=6` | Random-support KAN | `0.0` pair retention, `0.20` endpoint recall | `0.00` | `1.28` |
| `c=0.25,d=100,n=2048,m=6` | Exclude-endpoints KAN | `0.0` | `0.00` | `1.34` |

Interpretation: RF is not incapable in principle, but the recovery boundary is much later than the KAN-native stability boundary in this centered weak-interaction regime.

The first `c=0.50` screened-control block shows that this RF endpoint issue persists beyond the weakest interaction:

| Regime | Method | Endpoint support | Interaction F1 | Test MSE |
|---|---|---:|---:|---:|
| `c=0.50,d=100,n=512,m=5` | Oracle-support KAN | `1.0` | `0.80` | `2.88e-4` |
| `c=0.50,d=100,n=512,m=5` | RF-screened KAN | `0.0` true-pair retention, `0.25` endpoint recall | `0.00` | `9.88e-2` |
| `c=0.50,d=100,n=512,m=5` | Random-support KAN | `0.0` pair retention, `0.15` endpoint recall | `0.00` | `3.41` |
| `c=0.50,d=100,n=512,m=5` | Exclude-endpoints KAN | `0.0` | `0.00` | `3.28` |

Interpretation: stronger interaction signal helps the oracle refit but does not make RF marginal screening retain both centered interaction endpoints at this sample size.

The `c=0.50,n=1024` block shows RF beginning to recover occasionally, but still far behind KAN-native stability:

| Regime | Method | Endpoint support | Interaction F1 | Test MSE |
|---|---|---:|---:|---:|
| `c=0.50,d=100,n=1024,m=5` | KAN feature+edge stability | `1.0` | `1.00` | `2.21e-4` |
| `c=0.50,d=100,n=1024,m=5` | KAN feature stability | `1.0` | `0.90` | `2.61e-4` |
| `c=0.50,d=100,n=1024,m=5` | Oracle-support KAN | `1.0` | `0.90` | `2.37e-4` |
| `c=0.50,d=100,n=1024,m=5` | RF-screened KAN | `0.10` true-pair retention, `0.35` endpoint recall | `0.10` | `4.97e-2` |
| `c=0.50,d=100,n=1024,m=5` | Random-support KAN | `0.0` pair retention, `0.15` endpoint recall | `0.00` | `1.82` |
| `c=0.50,d=100,n=1024,m=5` | Exclude-endpoints KAN | `0.0` | `0.00` | `1.90` |

Interpretation: RF is not categorically incapable; its recovery boundary is simply much later than the KAN-native stability boundary in these centered interaction tasks.

A one-shot KAN-native control has completed:

```text
results/innovation_loop/one_shot_kan_screen_controls_after_screened_20260526_104933
```

This control trains one full-dimensional KAN, selects support from its own gradient/feature/edge scores, and then refits a low-dimensional KAN. It is the cleanest test of whether stability selection improves over a single KAN explanation pass.

Final one-shot summary:

| Regime | One-shot feature+edge F1 | Stability feature+edge F1 | Interpretation |
|---|---:|---:|---|
| `c=0.25,d=100,n=512,m=4` | `0.00` | `0.80` | stability aggregation is essential in the weakest block |
| `c=0.25,d=100,n=1024,m=4` | `0.40` | `0.70` | one-shot begins to recover, but remains weaker |
| `c=0.50,d=100,n=512,m=5` | `0.40` | `0.80` | stability remains clearly better in the moderate block |

This is now included in the main paper figure and Appendix single-pass table.

Analysis script:

```text
experiments/compare_native_with_screened_controls.py
```

It can merge KAN-native stability, screened controls, and one-shot KAN controls into a single comparison table/figure.

Additional RF-only diagnostic:

```text
results/innovation_loop/rf_screening_diagnostic_20260526_110853
```

This checks whether RF endpoint omission persists when increasing the number of trees from 500 to 2000 and varying `top_m` from 4 to 6, without running any KAN refits. Its purpose is to separate the screening failure from downstream KAN optimization.

Final RF-only diagnostic result:

| Regime | Trees | `top_m` | True-pair retention | Endpoint recall | Mean endpoint rank | Runs |
|---|---:|---:|---:|---:|---:|---:|
| `c=0.25,d=100,n=512` | `500` | `4/5/6` | `0.00` | `0.00` | `35.25` | `10` |
| `c=0.25,d=100,n=512` | `2000` | `4/5/6` | `0.00` | `0.00` | `31.95` | `10` |
| `c=0.25,d=100,n=1024` | `500` | `4/5/6` | `0.00` | `0.05` | `33.55` | `10` |
| `c=0.25,d=100,n=1024` | `2000` | `4` | `0.00` | `0.05` | `31.80` | `10` |
| `c=0.25,d=100,n=1024` | `2000` | `5/6` | `0.00` | `0.10` | `31.80` | `10` |

Interpretation: increasing RF trees and screening budget does not recover the centered interaction endpoints. RF importance ranks the true endpoints around positions `32-35` on average, far below any small `top_m` support. This supports the paper's use of RF as an external diagnostic baseline rather than as the main repair.

Pair-ranking follow-up:

```text
experiments/evaluate_pair_scoring_methods.py
results/innovation_loop/pair_scoring_oracle_pilot_after_rf_20260526_111150
```

This tests whether the interaction pair-ranking failures are partly an artifact of finite-difference mixed-derivative scoring. The new `anova_abs` and `anova_var` scores estimate a functional-ANOVA interaction component by conditioning on a candidate pair and averaging over background samples.

The oracle-support pilot is complete on `c=0.25,d=100,n=512,top_m=4`, seeds `100-104`:

| Pair scorer | Interaction F1 | True pair mean rank | Mean margin | Runs |
|---|---:|---:|---:|---:|
| `fd` | `0.60` | `1.40` | `-5.82e-2` | `5` |
| `anova_abs` | `1.00` | `1.00` | `7.26e-2` | `5` |
| `anova_var` | `1.00` | `1.00` | `1.00e-2` | `5` |
| `fd_anova_hybrid` | `1.00` | `1.00` | `5.48e-1` | `5` |

Interpretation: the old FD mixed-derivative score can mis-rank `(0,1)` above the true pure-interaction pair `(2,3)` even when the support is exactly `[0,1,2,3]`. Functional-ANOVA pair scoring fixes this in the first pilot. This is now a second candidate contribution: formula-fidelity evaluation should separate support recovery from interaction scoring, and KAN-native stability should be paired with an interaction score that discounts additive main effects.

The stability-support rescore has now completed the most important weak-regime block for both KAN-native support variants:

```text
results/innovation_loop/anova_pair_rescore_validation_20260526_113620_retry
```

| Source support | Regime | Original FD F1 | Refit FD F1 | ANOVA-abs F1 | ANOVA-var F1 | Hybrid F1 | Runs |
|---|---|---:|---:|---:|---:|---:|---:|
| `feature_edge_hybrid` | `c=0.25,d=100,n=512,m=4` | `0.80` | `0.60` | `1.00` | `1.00` | `1.00` | `10` |
| `feature_stability_var` | `c=0.25,d=100,n=512,m=4` | `0.70` | `0.50` | `1.00` | `1.00` | `1.00` | `10` |
| `feature_edge_hybrid` | `c=0.25,d=100,n=1024,m=4` | `0.70` | `0.40` | `1.00` | `1.00` | `1.00` | `10` |
| `feature_stability_var` | `c=0.25,d=100,n=1024,m=4` | `0.60` | `0.40` | `1.00` | `1.00` | `1.00` | `10` |

Interpretation: after the KAN-native support contains the true endpoints, weak-regime interaction recovery is not primarily limited by KAN expressivity. The remaining discrepancy is largely a pair-scoring issue: local FD mixed derivatives can rank the learned `(0,1)` additive structure above the true `(2,3)` pure interaction, while functional-ANOVA scoring consistently recovers `(2,3)`.

Paper-safe wording:

> KAN-native stability selection reaches the oracle-support regime in the weak `d=100` setting; functional-ANOVA pair scoring is then needed to avoid under-counting recovered interactions when local derivative scores are dominated by learned additive components.

Reusable code added:

```text
experiments/rescore_stability_supports_with_pair_methods.py
scripts/run_anova_pair_rescore_validation.sh
experiments/plot_rf_and_pair_scoring_diagnostics.py
```

Active validation run:

```text
results/innovation_loop/anova_pair_rescore_validation_20260526_113620_retry
```

This run rescoring existing strict-validation stability supports with `fd`, `anova_abs`, `anova_var`, and `fd_anova_hybrid` will test whether the ANOVA improvement persists beyond oracle support and across the KAN-native methods.

First completed stability-support rescore block:

| Support source | Regime | Pair scorer | Interaction F1 | True pair rank | Runs |
|---|---|---|---:|---:|---:|
| `feature_edge_hybrid` | `c=0.25,d=100,n=512,m=4` | `fd` | `0.60` | `1.40` | `10` |
| `feature_edge_hybrid` | `c=0.25,d=100,n=512,m=4` | `anova_abs` | `1.00` | `1.00` | `10` |
| `feature_edge_hybrid` | `c=0.25,d=100,n=512,m=4` | `anova_var` | `1.00` | `1.00` | `10` |
| `feature_edge_hybrid` | `c=0.25,d=100,n=512,m=4` | `fd_anova_hybrid` | `1.00` | `1.00` | `10` |

This is stronger than the oracle-only pilot because it uses KAN-native stability-selected support. The early conclusion is that some of the apparent weak-regime pair failures are scoring failures of the local FD interaction metric, not failures of the stabilized support.

Diagnostic figure:

```text
results/innovation_loop/diagnostic_figures/rf_pair_scoring_diagnostic.pdf
results/innovation_loop/anova_pair_rescore_validation_20260526_113620_retry/figures/feature_edge_n512_top4_pair_rescore.pdf
```

These runs should answer:

1. How much better is KAN feature-stability than raw KAN under identical seeds?
2. Does it approach RF-screened and oracle-support KAN at `d=100`?
3. At `d=500/1000`, is failure due to KAN-native support scoring, or do RF/oracle controls reveal a broader pair-ranking/optimization boundary?
4. Does stability selection beat one-shot KAN-native support extraction under the same training budget family?

If baseline controls confirm the gap, the next paper revision should promote KAN Feature-Stability Selection from exploratory ablation to the main KAN-native method.
