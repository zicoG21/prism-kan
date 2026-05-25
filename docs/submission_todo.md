# Submission TODO

## Highest Priority

1. Read `paper/main.tex` end to end and mark weak paragraphs.
2. Verify BibTeX metadata against official arXiv entries.
3. Decide the target venue/template, then port `paper/main.tex` into that template.
4. Convert inline numeric claims into table/figure-backed statements only.
5. Add appendix figures for `d=50` if the venue page limit permits.

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

## Figures

Main text:

1. Figure 1: prediction vs formula fidelity.
2. Figure 2: recovery boundary.
3. Figure 3: formula-fidelity ladder.
4. Figure 4: negative controls.
5. Table 1: KAN vs MLP baseline.

Appendix candidates:

1. `d=50` hard-regime figures.
2. metric robustness summary.
3. Feynman-style external validation.
4. heuristic SNR intuition.

## Do Not Do Yet

- Do not add more datasets before the first full paper draft is readable.
- Do not promote the SNR argument to a theorem.
- Do not make RF-screened KAN sound like a new algorithm.
- Do not claim exact symbolic formula extraction until symbolic extraction is actually evaluated.
