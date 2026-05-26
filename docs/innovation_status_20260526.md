# Innovation Status, 2026-05-26

## Current Novelty Level

The project is no longer only a negative diagnostic of KAN failures. The strongest current version has two linked contributions:

```text
KAN-native support stabilization + functional-ANOVA pair scoring
```

This is enough to reframe the paper as an empirical methods/diagnostics paper:

```text
KANs are formula-capable, but not automatically formula-faithful under nuisance dimensions.
```

The most defensible novelty is not "KAN fails" and not "RF fixes KAN". It is the decomposition of formula fidelity into support recovery and pair scoring, plus a KAN-native stabilization procedure that improves the support part.

## What Is Genuinely New Here

1. Formula-fidelity ladder:

```text
prediction -> variables -> interaction endpoints -> interaction pair
```

This is a clean evaluation framing for formula-level explanation fidelity. It makes visible failures that test MSE or variable F1 hides.

2. KAN-native support stabilization:

Repeated full-dimensional KAN explanations are used to estimate stable variable support, then a low-dimensional KAN is refit. This directly answers the criticism that RF-screened KAN is an external tree-model intervention.

The one-shot KAN control strengthens this point: using one full-dimensional KAN to screen and refit is not enough in the weakest block (`0.00` interaction F1 at `c=0.25,n=512,m=4`), while stability KAN-FE reaches `0.80`.

3. RF as a negative diagnostic:

RF screening is not the method. It is useful because it fails on centered weak-interaction endpoints in exactly the regime where KAN-native stability reaches oracle-support behavior.

4. Functional-ANOVA pair scoring:

The completed rescore shows that local FD mixed derivatives under-count recovered weak interactions after correct/stable support is supplied. Functional-ANOVA pair scoring recovers the true pair in all completed weak-regime stability-support blocks.

## Strongest Evidence So Far

Completed weak-regime KAN-native support rescore:

| Regime | Support | FD F1 | ANOVA / Hybrid F1 | Runs |
|---|---|---:|---:|---:|
| `c=0.25,d=100,n=512,m=4` | KAN-F | `0.50` | `1.00` | `10` |
| `c=0.25,d=100,n=512,m=4` | KAN-FE | `0.60` | `1.00` | `10` |
| `c=0.25,d=100,n=1024,m=4` | KAN-F | `0.40` | `1.00` | `10` |
| `c=0.25,d=100,n=1024,m=4` | KAN-FE | `0.40` | `1.00` | `10` |

Screened-control evidence:

- At `c=0.25,d=100,n=512,m=4`, KAN-F/KAN-FE reach FD interaction F1 `0.70/0.80`; RF is `0.00`; oracle is `0.60`.
- At `c=0.25,d=100,n=2048,m=6`, KAN-F/KAN-FE and oracle are all `0.80`; RF is `0.10`.
- At `c=0.50,d=100,n=512,m=5`, KAN-F/KAN-FE are `0.90/0.80`; RF is `0.00`; oracle is `0.80`.
- At `c=0.50,d=100,n=1024,m=5`, KAN-F/KAN-FE are `0.90/1.00`; RF is `0.10`; oracle is `0.90`.

One-shot control evidence:

- At `c=0.25,d=100,n=512,m=4`, single-pass KAN-FE is `0.00` while stability KAN-FE is `0.80`.
- At `c=0.25,d=100,n=1024,m=4`, single-pass KAN-FE is `0.40` while stability KAN-FE is `0.70`.
- At `c=0.50,d=100,n=512,m=5`, single-pass KAN-FE is `0.40` while stability KAN-FE is `0.80`.

## Remaining Innovation Risks

1. The experiments are still centered on one controlled synthetic family.

This is acceptable for a diagnostic paper, but the limitation must stay explicit. The Feynman-style results can help only if they are clean and formula-ground-truth metrics are well defined.

2. Stability selection costs more compute.

The paper should present it as a support-stabilization diagnostic/intervention, not as a compute-matched predictor.

3. Functional-ANOVA scoring improves evaluation and ranking, but it is not symbolic extraction.

Do not claim exact symbolic formula recovery unless a separate symbolic extraction step is evaluated.

4. High-dimensional stress remains hard.

At `d=500/1000`, the main bottleneck can return to support retention. This is a limitation, not a contradiction.

## Current Best Paper Claim

Safe main claim:

> KAN-native stability selection can reach the oracle-support regime under weak centered interactions where RF screening misses formula-critical endpoints; after support recovery, functional-ANOVA pair scoring avoids under-counting interactions that local derivative scores can mis-rank.

Avoid:

```text
KAN solves high-dimensional formula discovery
ANOVA proves symbolic recovery
RF is a bad baseline in general
phase transition
formal KAN sample complexity
```

## Next Decisive Steps

1. Freeze the current main claim and avoid starting broad new sweeps unless a reviewer-driven gap is identified.
2. Decide whether to add a compact Feynman-style appendix table; keep it clearly labeled as a sanity check unless full controls are added.
3. Add a small runtime/compute table if the target venue is likely to object that stability selection uses repeated probe runs.
4. Port the current paper into the target conference template and then do a numeric claim audit in that layout.
