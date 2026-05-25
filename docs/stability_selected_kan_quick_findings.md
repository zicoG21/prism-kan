# Stability-Selected KAN Quick Findings

Date: 2026-05-25

## Setup

Quick KAN-native stabilization checks on the hard interaction benchmark:

```text
f(x) = sin(2*pi*x0) + x1^2 + c*x2*x3
d = 100
n in {256, 512, 1024}
c in {0.1, 0.25, 0.5, 1.0}
eval seeds = {0, 1, 2, 3, 4}
```

Method:

- Reuse full-dimensional raw KAN runs as stability probes.
- Leave out the current eval seed when estimating stability.
- Select a 4-variable support from KAN-derived variable stability.
- Refit a low-dimensional KAN and evaluate the formula-fidelity ladder.

Artifacts:

- `results/stability_kan/quick_variable_boundary_summary.csv`
- `results/stability_kan/figures/quick_variable_boundary_interaction_f1.png`
- `results/stability_kan/quick_d100_n256_variable_summary.csv`
- `results/stability_kan/quick_d100_n512_summary.csv`
- `results/stability_kan/quick_d100_n1024_summary.csv`

## Main Readout

Interaction F1 for raw KAN vs SS-KAN-V:

```text
c=0.10
n=256: raw 0.0, SS-KAN-V 0.0
n=512: raw 0.0, SS-KAN-V 0.0
n=1024: raw 0.0, SS-KAN-V 0.0

c=0.25
n=256: raw 0.0, SS-KAN-V 0.0
n=512: raw 0.0, SS-KAN-V 0.0
n=1024: raw 0.8, SS-KAN-V 0.4

c=0.50
n=256: raw 0.0, SS-KAN-V 0.0
n=512: raw 0.0, SS-KAN-V 1.0
n=1024: raw 1.0, SS-KAN-V 1.0

c=1.00
n=256: raw 0.0, SS-KAN-V 0.0
n=512: raw 0.2, SS-KAN-V 0.6
n=1024: raw 0.8, SS-KAN-V 1.0
```

The clearest positive regime is `n=512, c=0.5`: raw KAN has interaction F1 0.0, while SS-KAN-V reaches 1.0 and matches oracle-level prediction error. At `n=512, c=1.0`, SS-KAN-V improves interaction F1 from 0.2 to 0.6, but still has unstable support in two of five seeds. At `n=256`, SS-KAN-V does not recover the interaction even for `c=1.0`, which places the quick recovery boundary between 256 and 512 samples.

## Pair-First Ablation

The pair-first variant, `ss_kan_pair`, should not be treated as the main method in its current form.

At `n=512, c=1.0`, pair-first selection frequently selects supports such as:

```text
[0, 1, 2, 97]
[0, 1, 3, 97]
```

The stable pair list often contains the true pair `(2, 3)`, but after higher-ranked spurious pairs such as `(0, 1)` or `(0, 97)`. Because support size is fixed at 4, pair-first can keep one true endpoint and one nuisance endpoint, then miss the full interaction.

Decision:

- Use variable-first stability selection as the main KAN-native method.
- Keep pair-first only as an ablation or failure analysis.
- Use pair stability as a diagnostic candidate set, not as the primary support constructor.

## Support Capacity Robustness

Targeted robustness check:

```text
n = 512
d = 100
c = 1.0
method = SS-KAN-V
top_m = 6
eval seeds = {0, 1, 2, 3, 4}
```

Result:

```text
raw KAN:        test MSE 0.2435, interaction F1 0.2
SS-KAN-V m=4:  test MSE 0.1513, interaction F1 0.6
SS-KAN-V m=6:  test MSE 0.000665, interaction F1 1.0
oracle support: test MSE 0.000517, interaction F1 1.0
```

For `top_m=6`, every seed selected a support containing all true variables `{0,1,2,3}`, and the final refit selected the true interaction `(2,3)` in every seed. This suggests that the residual `top_m=4` failures at `n=512,c=1.0` are largely support-capacity failures: the stability ranking often places one nuisance variable above one true endpoint, but the true endpoint is still close enough to be recovered when the support budget is slightly relaxed.

Decision:

- Do not run `top_m=8` yet; `top_m=6` already answers the immediate question.
- Treat `top_m` as a method hyperparameter / support-budget axis in the paper.
- Use `top_m=4` as the strict formula-support setting and `top_m=6` as a targeted robustness check.

## Current Interpretation

SS-KAN-V is not a universal fix. It improves the recovery boundary in moderate-to-strong signal regimes but remains sample limited:

- `c=0.1`: endpoint/support evidence is too weak.
- `c=0.25`: endpoint recovery improves at larger `n`, but pair extraction remains brittle.
- `c=0.5`: SS-KAN-V gives the strongest positive result, recovering at `n=512` where raw KAN fails.
- `c=1.0`: SS-KAN-V improves over raw at `n>=512`, but `n=256` is still too small.

This supports the revised paper line:

> KANs are formula-capable but not automatically formula-faithful. Stability selection can improve KAN-native support recovery, but the interaction recovery boundary remains governed by signal strength and sample size.

## Next Steps

1. Promote SS-KAN-V, not RF-screened KAN, as the KAN-native intervention.
2. Retain RF as an external diagnostic baseline.
3. Add a concise method section for variable-first stability selection.
4. Add pair-first as an ablation showing why stable pair frequencies alone are insufficient.
5. Run a minimal robustness check with more stability repeats or `top_m=6` only if the current story needs support for `c=1.0,n=512`.
