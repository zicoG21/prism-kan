# Response to Gemini Review

Date: 2026-05-25

## Accepted Points

### 1. Formula-Fidelity Ladder

Accepted. This is now the main conceptual frame:

```text
prediction -> variables -> interaction endpoints -> interaction pair
```

Decision:

- Keep this in the introduction.
- Use it to structure the metrics section.
- Make Figure 3 the visual proof that variable recovery is not enough.

### 2. Recovery Boundary

Accepted with revised wording.

The empirical pattern is threshold-like:

```text
raw KAN, d=100, interaction F1
c=1.00: 0.00, 0.00, 0.10, 0.80
c=0.50: 0.00, 0.00, 0.10, 0.90
c=0.25: 0.00, 0.00, 0.00, 0.60
c=0.10: 0.00, 0.00, 0.00, 0.00
```

Decision:

- Main text uses `recovery boundary` or `recovery threshold`.
- Avoid claiming a formal sharp `phase transition`.
- `phase-transition-like` is acceptable only as informal descriptive language.

### 3. Sharper Baseline Story

Accepted with caution.

Decision:

- Use MLP to show the support-retention bottleneck is not unique to KAN.
- Emphasize that screened/oracle KAN fits the low-dimensional formula much more accurately than screened/oracle MLP in current runs.
- Do not claim exact symbolic formula extraction unless symbolic extraction is actually run.

Safe wording:

> MLP baselines can recover interaction rankings when support is supplied, but screened KAN achieves substantially lower prediction error on the same low-dimensional formula, making it a stronger candidate for subsequent formula inspection and symbolic simplification.

## Modified / Rejected Points

### SNR Upper Bound as Formal Appendix Theorem

Modified. Do not include the earlier local-spline `p_G` / `O(1/d)` SNR argument as a theorem yet.

Reason:

- The earlier derivation relies on independence and variance-scaling assumptions that are not fully justified.
- A strict reviewer could attack this more easily than the empirical contribution.

Decision:

- Keep the main text theory as mechanistic intuition:

```text
E[x2*x3*g(x2)] = 0
```

- Add only a clearly labeled heuristic appendix if needed:

```text
Appendix: Heuristic SNR intuition for pure-interaction discovery
```

- Do not use words like theorem, proof, or sample-complexity bound unless the argument is rebuilt rigorously.

## Immediate Changes Made

- `docs/paper_draft_v0.md`: changed main wording from `phase transition` to `recovery boundary`.
- `experiments/plot_paper_hard_regime_figures.py`: updated Figure 1 title to use `recovery boundary`.
- `paper/main.tex`: created a full LaTeX draft using recovery-boundary wording.
- `docs/reproducibility_checklist.md`: added regeneration and compilation commands.

## Next Decision Point

Superseded by the SS-KAN-V quick experiments.

New decision:

1. Promote variable-first Stability-Selected KAN (`SS-KAN-V`) as the KAN-native intervention.
2. Keep RF-screened KAN as an external diagnostic baseline, not the main method.
3. Keep pair-first stability selection as an ablation/failure analysis, because stable spurious pairs can steal support capacity from the true interaction endpoints.
4. Use the current quick boundary result as the main new evidence: SS-KAN-V helps at `n=512,c=0.5` and `n>=512,c=1.0`, but does not solve `n=256` or `c=0.1`.

Current priority:

1. Polish the revised `paper/main.tex` around the new SS-KAN-V method/result.
2. Decide whether to spend one more run on `top_m=6` or more stability repeats for `n=512,c=1.0`.
3. Move Feynman-style validation into a small main-text or appendix robustness section only after the core SS-KAN story is stable.
