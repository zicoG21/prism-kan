# Current Paper Evaluation

Date: 2026-05-26

## Latest Revision Status

The latest pass has been incorporated into `paper/main.tex` and compiles cleanly into `paper/main.pdf`. The main stabilization figure is now a three-panel figure:

```text
single-pass KAN control -> KAN-native stability vs RF/oracle -> FD vs ANOVA pair scoring
```

Two appendix tables were added: one for the single-pass KAN control and one for the `d=500/1000` stress tests. The paper now explicitly states that KAN-native stabilization shifts the recovery boundary but does not solve extreme nuisance-dimension support recovery.

## Overall Assessment

The current draft has moved from a purely negative diagnostic paper into a stronger empirical-methods paper. The central story is now coherent:

```text
KANs are formula-capable, but not automatically formula-faithful under nuisance dimensions.
```

The strongest contribution is the formula-fidelity ladder:

```text
prediction -> active variables -> interaction endpoints -> interaction pair
```

This ladder is more defensible than a single interaction-F1 number because it separates prediction success, support retention, endpoint retention, and final pair ranking. SS-KAN is also the right strategic addition: it directly addresses the criticism that RF-screened KAN is an external tree-based intervention rather than a KAN-native method.

Current likely level: no longer just a negative diagnostic paper. The strongest version is now an empirical-methods paper with two KAN-native pieces:

```text
KAN-native support stabilization + formula-aware interaction scoring
```

It is still not a theory paper, and it should not pretend to be one. But the contribution is now more publishable than the earlier "KAN fails under nuisance dimensions" framing because it diagnoses the failure and adds a native intervention.

## Main Strengths

1. The core claim is now appropriately scoped.

The draft no longer claims that KANs fundamentally fail. It claims a structure-retention bottleneck under nuisance dimensions, while preserving the positive point that KANs fit the low-dimensional formula well once support is available.

2. The evidence chain is clear.

Raw KAN shows prediction/formula mismatch. Oracle support separates expressivity from support discovery. Random and endpoint-exclusion controls show arbitrary dimension reduction is not enough. SS-KAN then becomes a KAN-native intervention rather than an afterthought.

3. The weak-interaction result is genuinely interesting.

The new same-seed controls show a clean boundary: KAN-native stability reaches the oracle-support regime while RF screening misses centered interaction endpoints through `n=1024` and only weakly recovers at `n=2048`.

4. Pair scoring is now a separate contribution candidate.

The functional-ANOVA pair scoring pilot shows that FD mixed-derivative scoring can understate interaction recovery even after the correct support is supplied. This sharpens the ladder into:

```text
support recovery -> endpoint retention -> formula-aware pair scoring
```

5. The theory section is safer than the earlier SNR-theorem direction.

The three propositions are modest and defensible. They give formal intuition without overclaiming KAN-specific sample complexity.

## Major Risks

1. The main remaining risk is generality.

The paper now has real appendix content rather than placeholder prose, but the central evidence is still mostly controlled synthetic recovery. This is defensible for formula-fidelity diagnostics, because ground-truth variables and interactions are required, but reviewers may still ask whether the same bottleneck appears beyond the core polynomial interaction family.

Recommended fix: keep the current synthetic story as the main evidence. Treat the Feynman-style `d=100` results as a small sanity check, not as a broad scientific benchmark claim, unless the full baseline/control suite is added.

2. The MLP baseline claim needs sharper calibration.

The table supports the claim that support retention is a broad bottleneck. However, the current MLP rows have high test MSE even under RF/oracle support, while interaction F1 reaches 1.0. This means the table supports "interaction ranking can recover when support is supplied", but not "MLP fits the low-dimensional formula well".

Recommended fix: phrase this as:

```text
MLP baselines show that support retention is not unique to KAN. Under supplied support, MLP interaction rankings can recover, but the early-stopped MLP baseline remains much less accurate than low-dimensional KAN in this configuration.
```

Avoid implying that MLP is a strong formula-fitting baseline unless a better-tuned MLP run is added.

3. The paper still leans heavily on one synthetic family.

This is acceptable for a diagnostic paper because formula ground truth is clean, but it will be the first experimental generality criticism. The limitation section admits this, but the paper would benefit from either a small appendix external validation or a more explicit statement that the goal is controlled mechanism diagnosis.

Recommended fix: keep the main paper synthetic and add only a compact Feynman-style appendix table if those experiments are promoted later with clear caveats.

4. The ANOVA pair scorer is now validated on KAN-native stability supports.

The stability-support rescore confirms the oracle-support pilot. On `c=0.25,d=100,n=512,top_m=4`, KAN-F/KAN-FE have FD interaction F1 `0.50/0.60`, while `anova_abs`, `anova_var`, and `fd_anova_hybrid` reach `1.00` for both support variants over 10 runs. The feature+edge support also confirms the pattern at `n=1024,top_m=4`.

5. SS-KAN is currently useful but not a universal method.

This is handled better now, but reviewers may still ask whether SS-KAN is simply ensembling seeds and increasing compute.

This is now addressed in the method section:

```text
SS-KAN trades additional full-dimensional probe runs for a lower-dimensional refit; we therefore interpret it as a support-stabilization diagnostic/intervention rather than a compute-matched predictor.
```

6. Some inline numeric claims need source anchoring.

Most numbers match the CSVs, but the final draft should avoid too many inline numbers unless they are in a nearby figure/table or explicitly sourced. The highest-value numbers are already in figures/tables; the rest can be softened.

## Claim Audit

| Claim | Status | Evidence |
| --- | --- | --- |
| Prediction can improve before interaction recovery. | Strong | `fig2_prediction_vs_formula_raw_d100.pdf`; `c=0.10,n=1024` raw KAN MSE `0.002494`, interaction F1 `0.0`. |
| Recovery depends on sample size and interaction strength. | Strong | `fig1_formula_phase_d100.pdf`; raw KAN at `n=1024`: F1 `0.0/0.6/0.9/0.8` for `c=0.10/0.25/0.50/1.00`. |
| Oracle support separates expressivity from support discovery. | Strong for moderate/strong interactions | Oracle rows recover strong/moderate regimes; weak regimes remain harder. |
| KAN-native stability improves formula recovery. | Strong | `stabilization_pair_scoring_summary.pdf`; same-seed screened controls. |
| Stability improves over a single KAN pass. | Strong | One-shot controls: single-pass KAN-FE `0.00/0.40/0.40` versus stability KAN-FE `0.80/0.70/0.80` in the three plotted settings. |
| Stable/oracle support is not always pair fidelity under FD scoring. | Strong and refined | ANOVA rescore shows FD can be the bottleneck after KAN-native support; use "FD-scored pair recovery" for old FD results. |
| RF screening misses centered weak-interaction endpoints. | Strong | RF-only diagnostic: true-pair retention `0.00` at `n=512/1024` for 500 and 2000 trees; mean endpoint ranks around `32-35`. |
| KAN-native stability reaches oracle-support behavior without RF screening. | Strong at `d=100` | Same-seed screened controls: KAN-native support retention `1.0` and interaction F1 `0.55-0.90`; RF F1 `0-0.1`. |
| Functional-ANOVA pair scoring improves recovered pair ranking. | Strong in the core weak regime | Oracle-support pilot and stability-support rescore both show ANOVA/hybrid F1 `1.00` where FD is lower. |
| Support bottleneck is not unique to KAN. | Moderate | MLP raw/support-controlled rows support this, but MLP tuning is not the main focus. |
| KAN is better for downstream symbolic inspection. | Plausible but should be softened | No symbolic extraction is evaluated. Keep this as motivation, not demonstrated result. |
| Extreme nuisance dimension is solved by stability. | False/avoid | `d=1000` stress tests still fail for KAN-FE even when oracle support works. Claim only a shifted recovery boundary. |

## Suggested Next Edits

1. Tighten the MLP baseline paragraph further if the paper is moved into a strict conference template.

Keep the broad-bottleneck point. Avoid claiming symbolic or formula extraction benefits that were not directly tested.

2. Decide whether to include the Feynman-style results in a small appendix table.

The local results are promising, but they are native-only rather than a full control suite.

3. Add a short compute-cost table if reviewers are likely to object to repeated KAN probes.

The method is already framed as a diagnostic/intervention rather than a compute-matched predictor, but a small runtime table would make the tradeoff transparent.

## Recommendation

The paper has enough evidence to use the upgraded framing now. The ANOVA rescore and one-shot KAN controls have finished and are reflected in the paper. The next decision is venue/template and whether to add a minimal external-validation appendix.
