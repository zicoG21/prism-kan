# Paper Reframe Outline, 2026-05-26

## New Main Claim

KANs are formula-capable but not automatically formula-faithful under nuisance dimensions. The bottleneck has two separable parts:

```text
support stabilization + formula-aware pair scoring
```

The paper should not be framed as "KAN fails" or "RF fixes KAN". RF is now best treated as an external diagnostic probe that fails on centered weak interactions, while KAN-native stability is the main support intervention.

## Contribution List To Use

1. Formula-fidelity ladder:

```text
prediction -> variables -> interaction endpoints -> interaction pair
```

2. Controlled sparse-interaction benchmark with nuisance dimensions and known formula structure.

3. Empirical recovery boundary showing that low test error can precede formula recovery.

4. KAN-native feature-stability support selection, with RF/oracle/random/exclude as controls.

5. Single-pass KAN controls showing that stability aggregation improves over a simple KAN screen-and-refit pipeline.

6. RF screening diagnostic showing that marginal/tree importances miss centered weak-interaction endpoints even with many trees.

7. Functional-ANOVA pair scoring diagnostic showing that finite-difference mixed derivatives can under-rank the true interaction after correct support is supplied.

## Paper Edits Needed

### Abstract

Replace the current final sentence about weak interactions being unresolved with a more precise statement:

```text
The remaining failures split into support-retention failures and pair-scoring failures; functional-ANOVA pair scores recover weak interactions that finite-difference mixed-derivative scores can miss.
```

This is now implemented in `paper/main.tex`.

### Metrics Section

Current text says finite-difference is the main pair score and Hessian/permutation are robustness checks. Replace with:

```text
We report local mixed-derivative scores and a functional-ANOVA interaction score. The former measures local curvature; the latter estimates whether varying a pair jointly explains function variation beyond the two separate main effects.
```

Then define:

```text
h_{ij}(a,b)=E_Z[f(X_i=a,X_j=b,Z_{-(i,j)})]
             -E_Z[f(X_i=a,Z_{-i})]
             -E_Z[f(X_j=b,Z_{-j})]
             +E_Z[f(Z)].
```

Score by either `mean |h_ij|` or `var(h_ij)`.

### Results Section

The current paper uses one compact main subsection plus a stress-test subsection:

1. `KAN-Native Stabilization and Pair Scoring`
   - Use `stabilization_pair_scoring_summary.pdf`.
   - Panel (a): single-pass KAN control versus stability KAN-FE.
   - Panel (b): KAN-F/KAN-FE versus RF/oracle.
   - Panel (c): FD versus ANOVA pair scoring after stable support.

2. `Nuisance-Dimension Stress Tests`
   - State explicitly that `d=500/1000` stress tests show a shifted boundary, not a solved high-dimensional problem.

### Appendix

Replace the old "Metric Robustness" appendix. Do not say Hessian proves the pair-ranking issue is not a scoring artifact. Instead:

```text
Local derivative scores and functional-ANOVA scores answer different questions. Hessian/FD diagnose local mixed curvature; ANOVA estimates global pair synergy after marginal effects are averaged out. The weak centered interaction exposes the difference.
```

## Figure Candidates

Main:

1. `fig2_prediction_vs_formula_raw_d100.pdf`
2. `fig1_formula_phase_d100.pdf`
3. `native_vs_screened_controls.pdf`
4. `rf_pair_scoring_diagnostic.pdf`
5. `stabilization_pair_scoring_summary.pdf`

Current best single-figure candidate:

```text
results/innovation_loop/final_candidate_figures/stabilization_pair_scoring_summary.pdf
```

It combines the KAN-native support-stabilization comparison with the FD-vs-ANOVA pair-scoring result. Use this if the paper needs one compact figure that states the upgraded contribution in a single glance.

Appendix:

1. Old ladder/raw figures.
2. High-dimensional stress table.
3. Single-pass KAN control table.
4. Full ANOVA rescore table.

## Claims To Avoid

- Do not claim a formal phase transition.
- Do not claim KAN-native stability solves high-dimensional formula recovery.
- Do not claim oracle support necessarily fails at weak interactions without specifying the pair scorer.
- Do not present RF-screened KAN as the method.
- Do not claim symbolic formula extraction.
