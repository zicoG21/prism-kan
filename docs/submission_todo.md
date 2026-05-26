# Submission TODO

## Highest Priority

1. Decide the target venue/template, then port `paper/main.tex` into that template.
2. Do a final numeric claim audit after template porting.
3. Decide whether to include a compact Feynman-style appendix table or leave those results in docs only.
4. Add a small runtime/compute table if repeated KAN probes are likely to be challenged.
5. Convert remaining inline numeric claims into table/figure-backed statements where the template makes space tight.

## Text Polishing

1. Keep the main claim as:

```text
KANs are formula-capable but not automatically formula-faithful under nuisance dimensions.
```

2. Use:

```text
recovery boundary
recovery threshold
threshold-like empirical pattern
```

Avoid:

```text
sharp phase transition
proved sample complexity
KAN uniquely fails
RF solves formula recovery
```

3. Make the Formula-Fidelity Ladder the center of the paper:

```text
prediction -> variables -> interaction endpoints -> interaction pair
```

4. Use the now-confirmed one-shot and rescore results as:

```text
KAN-native support stabilization + functional-ANOVA pair scoring
```

## Figures

Main text:

1. Figure 1: prediction vs formula fidelity.
2. Figure 2: recovery boundary.
3. Figure 3: KAN-native support stabilization + functional-ANOVA pair scoring.
4. Figure 4: formula-fidelity ladder.
5. Figure 5: negative controls.
6. Table 1: KAN vs MLP baseline, if kept.

Appendix candidates:

1. `d=50` hard-regime recovery boundary.
2. high-dimension stress tests.
3. one-shot KAN controls.
4. pair-scoring diagnostics.
5. optional external validation, only if the results are clean.

## Do Not Do Yet

- Do not add more datasets before the first full paper draft is readable.
- Do not promote the SNR argument to a theorem.
- Do not make RF-screened KAN sound like a new algorithm.
- Do not claim exact symbolic formula extraction until symbolic extraction is actually evaluated.
- Do not keep the old "FD-vs-Hessian proves no metric artifact" claim without reconciling the ANOVA result.
