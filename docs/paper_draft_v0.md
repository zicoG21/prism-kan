# Prediction is Not Formula Fidelity: Diagnosing Structure-Retention Failures in KAN Explanations

Draft v0, 2026-05-25

## Abstract

Kolmogorov-Arnold Networks (KANs) are often presented as formula-friendly models for scientific discovery, but predictive accuracy alone does not establish that a learned model has recovered the correct formula structure. We study this gap using synthetic regression problems with known active variables and known interaction structure. In high-dimensional sparse interaction tasks with many nuisance variables, KANs can achieve low test error while failing to recover the formula-critical interaction. We show that formula fidelity undergoes a recovery threshold controlled by sample size, interaction strength, and support discovery. Oracle-support experiments show that KANs can fit the correct low-dimensional formula once the relevant support is available, while random-support and endpoint-exclusion controls show that arbitrary dimension reduction is not sufficient. These results suggest a structure-retention bottleneck: KANs are formula-capable, but not automatically formula-faithful under nuisance dimensions.

## Core Claim

KAN is formula-capable but not automatically formula-faithful under nuisance dimensions. Formula fidelity emerges only after crossing a recovery threshold controlled by interaction strength, sample size, and support discovery.

## Contributions

1. We define a formula-fidelity diagnostic separating prediction fidelity, variable fidelity, endpoint retention, and interaction fidelity.

2. We introduce a controlled sparse-interaction benchmark:

```text
f(x) = sin(2*pi*x0) + x1^2 + c*x2*x3
```

with known active variables `{x0, x1, x2, x3}` and true interaction `(x2, x3)`.

3. We show a formula-fidelity recovery boundary over sample size `n`, interaction strength `c`, and dimension `d`.

4. We use diagnostic interventions, including raw KAN, RF-screened KAN, oracle-support KAN, random support, and interaction-endpoint exclusion, to separate representability from support discovery.

5. We show that variable recovery is not sufficient: endpoint retention and pairwise interaction recovery are distinct failure points.

## Problem Setup

The main benchmark family is:

```text
f_core(x) = sin(2*pi*x0) + x1^2 + c*x2*x3
```

We evaluate:

- `d in {50, 100}`
- `n in {128, 256, 512, 1024}`
- `c in {0.10, 0.25, 0.50, 1.00}`
- screen modes: raw, RF, oracle-support, random, exclude-interaction

The central setting for main figures is `d=100`.

## Metrics

Prediction fidelity:

- test MSE

Variable fidelity:

- top-k variable F1

Endpoint retention:

- whether the selected explanation variables contain the true interaction endpoints
- endpoint recall over `{x2, x3}`

Interaction fidelity:

- top-k pair interaction F1
- true-pair score margin: true interaction score minus strongest false-pair score
- for new runs, exact true-pair rank is saved; for old hard-regime runs, rank is recoverable as `1` or `>1` from the saved margin.

## Main Result 1: Prediction is Not Formula Fidelity

In weak-interaction settings, raw KAN prediction can improve while formula recovery remains poor. For `d=100, c=0.10, n=1024`, raw KAN reaches low test MSE around `0.0025`, but interaction F1 remains `0.0`.

This shows that low prediction error is not enough evidence for scientific formula recovery.

Figure:

```text
results/hard_regime/paper_figures/fig2_prediction_vs_formula_raw_d100.pdf
```

## Main Result 2: Formula-Fidelity Recovery Boundary

The hard-regime sweep shows a clear recovery boundary. We use "recovery boundary" rather than claiming a formal sharp phase transition: the empirical pattern is threshold-like for strong and moderate interactions, but it is not a proven discontinuous transition. For `d=100`, raw KAN interaction F1 increases with stronger interactions and larger sample size:

- `c=1.00, n=1024`: interaction F1 around `0.8`
- `c=0.50, n=1024`: interaction F1 around `0.9`
- `c=0.25, n=1024`: interaction F1 around `0.6`
- `c=0.10, n=1024`: interaction F1 remains `0.0`

Oracle-support KAN separates representability from full-space discovery: when the correct support is given, the strong-interaction regimes become recoverable, while the weakest interaction remains hard. This supports a sample-size / interaction-strength recovery threshold rather than an absolute claim that KAN cannot learn the structure.

Main figure:

```text
results/hard_regime/paper_figures/fig1_formula_phase_d100.pdf
```

Source data:

```text
results/hard_regime/paper_figures/paper_figure_source_data.csv
results/hard_regime/paper_figures/hard_regime_augmented_summary.csv
```

## Main Result 3: Variable Recovery is Not Enough

Formula fidelity has a ladder:

```text
prediction -> variables -> interaction endpoints -> interaction pair
```

For `d=100, c=1.00, n=1024`, raw KAN has variable F1 around `0.925`, endpoint recall around `0.85`, and interaction F1 around `0.8`. The failed seeds are diagnosed as variable-ranking misses of the interaction endpoints.

This supports the claim that formula-level explanations require more than active-variable recovery.

Figure:

```text
results/hard_regime/paper_figures/fig3_variable_vs_interaction_raw_d100.pdf
```

Diagnostic table:

```text
results/hard_regime/paper_figures/endpoint_ladder_d100.csv
```

## Negative Controls

Random support and exclude-interaction controls remain near zero interaction F1. This rules out the interpretation that any low-dimensional KAN is sufficient. The content of the selected support, especially the true endpoints, is necessary.

Figure:

```text
results/hard_regime/paper_figures/fig4_negative_controls_d100.pdf
```

## Baselines

Main baselines are fixed in:

```text
docs/baseline_protocol.md
results/paper_figures/final_baseline_table.csv
```

Use early-stopped MLP as the main MLP baseline. Keep aggressive full-refit MLP variants, RF predictor diagnostics, Feynman formulas, and path deletion in appendix.

The current core pattern is:

- screened/oracle KAN fits the low-dimensional formula with much lower MSE than screened/oracle MLP;
- MLP can recover interaction ranking once support is supplied, so the support-retention bottleneck is not unique to KAN;
- the formula-fitting advantage after support selection remains a KAN strength in these runs;
- MLP baselines should be used to show that support retention is a broad bottleneck, while KAN remains valuable because the architecture is more naturally suited to low-dimensional formula fitting and downstream symbolic inspection once the relevant support is available.

## Mechanistic Interpretation

For centered independent features, a pure interaction can be invisible to first-order marginal screening:

```text
E[x2*x3*g(x2)] = 0
```

when `x3` is centered and independent of `x2`. This supports the intuition that pure interactions require joint evidence and can be missed by main-effect style selection. This should be presented as mechanism intuition, not as a KAN-specific sample-complexity theorem.

Appendix direction: include only a clearly labeled heuristic SNR calculation under explicit simplifying assumptions if the derivation is internally consistent. Do not present the earlier local-spline `p_G` / dimension-scaling argument as a theorem unless every independence and variance step can be justified. A safe appendix title would be "Heuristic SNR intuition for pure-interaction discovery", not "Sample-complexity theorem".

## Limitations

- The strongest evidence is synthetic, because ground-truth formula structure is known.
- The theory is mechanistic rather than a formal sample-complexity result.
- RF screening is a diagnostic support intervention, not a final formula-faithful algorithm.
- Interaction metrics measure pairwise dependence in the learned function; they do not by themselves prove symbolic simplification.
- Feynman-style formulas are currently appendix validation rather than central mechanism evidence.

## Next Writing Tasks

1. Turn the abstract into a 150-200 word final abstract.
2. Write Introduction around the formula-fidelity gap.
3. Write Methods around the four-level metric ladder.
4. Convert hard-regime figure captions into main-result prose.
5. Move old exploratory runs into appendix or omit them.
