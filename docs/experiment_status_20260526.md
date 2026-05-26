# Experiment Status, 2026-05-26

## Completed Runs Snapshot

### Screened Controls

```text
results/innovation_loop/strict_screened_baseline_controls_20260526_104243
```

Purpose: RF, oracle-support, random-support, and exclude-endpoints controls under the same KAN refit/evaluation protocol.

Important completed blocks:

- `c=0.25,d=100,n=512,top_m=4`
- `c=0.25,d=100,n=1024,top_m=4/5/6`
- `c=0.25,d=100,n=2048,top_m=6`
- `c=0.5,d=100,n=512,top_m=5`
- `c=0.5,d=100,n=1024,top_m=5`

The screened-control run has completed its planned matrix, including the later `d=500/1000` stress rows.

### ANOVA Pair Rescore

```text
results/innovation_loop/anova_pair_rescore_validation_20260526_113620_retry
```

Purpose: take already selected KAN-native stability supports from the strict validation run, refit on the same support, and score interactions with:

```text
fd, anova_abs, anova_var, fd_anova_hybrid
```

Completed `n=512,m=4` blocks:

| Source support | Regime | FD F1 | ANOVA-abs F1 | ANOVA-var F1 | Hybrid F1 | Runs |
|---|---|---:|---:|---:|---:|---:|
| `feature_edge_hybrid` | `c=0.25,d=100,n=512,m=4` | `0.60` | `1.00` | `1.00` | `1.00` | `10` |
| `feature_stability_var` | `c=0.25,d=100,n=512,m=4` | `0.50` | `1.00` | `1.00` | `1.00` | `10` |

Interpretation so far: FD mixed-derivative scoring is underestimating pair recovery after KAN-native support stabilization. This now holds for both the simplest feature-stability support and the feature+edge hybrid support in the weakest completed same-seed regime.

Additional completed rescore evidence:

- `feature_edge_hybrid`, `n=1024,m=4`: FD F1 `0.40`, ANOVA-abs/ANOVA-var/hybrid F1 `1.00`, `10` runs.
- `feature_stability_var`, `n=1024,m=4`: FD F1 `0.40`, ANOVA-abs/ANOVA-var/hybrid F1 `1.00`, `10` runs.

### One-Shot KAN Control

```text
results/innovation_loop/one_shot_kan_screen_controls_after_screened_20260526_104933
```

Purpose: test whether stability selection improves over a single full-dimensional KAN explanation pass.

Status: completed. The narrowed feasible control covered:

- `c=0.25,d=100,n=512,top_m=4`, seeds `100-104`
- `c=0.25,d=100,n=1024,top_m=4`, seeds `100-104`
- `c=0.50,d=100,n=512,top_m=5`, seeds `100-104`

Key result: one-shot feature+edge screen-and-refit is weaker than stability aggregation in all three control blocks: `0.00` vs `0.80` at `c=0.25,n=512,m=4`; `0.40` vs `0.70` at `c=0.25,n=1024,m=4`; and `0.40` vs `0.80` at `c=0.50,n=512,m=5`.

## New Code Added

```text
experiments/rescore_stability_supports_with_pair_methods.py
experiments/plot_rf_and_pair_scoring_diagnostics.py
experiments/plot_pair_rescore_summary.py
experiments/plot_stabilization_pair_scoring_figure.py
scripts/run_anova_pair_rescore_validation.sh
scripts/run_one_shot_kan_screen_controls.sh
```

Updated:

```text
experiments/run_tuned_kan_recovery.py
experiments/run_kan_native_innovation_loop.py
experiments/compare_native_with_screened_controls.py
docs/kan_native_stability_findings.md
docs/current_paper_evaluation.md
docs/claim_audit.md
docs/submission_todo.md
docs/reproducibility_checklist.md
```

## New Figures

```text
results/innovation_loop/strict_screened_baseline_controls_20260526_104243/analysis_min8_live/native_vs_screened_controls.pdf
results/innovation_loop/diagnostic_figures/rf_pair_scoring_diagnostic.pdf
results/innovation_loop/anova_pair_rescore_validation_20260526_113620_retry/figures/feature_edge_n512_top4_pair_rescore.pdf
results/innovation_loop/anova_pair_rescore_validation_20260526_113620_retry/figures/feature_stability_n512_top4_pair_rescore.pdf
results/innovation_loop/anova_pair_rescore_validation_20260526_113620_retry/figures/feature_edge_n1024_top4_pair_rescore.pdf
results/innovation_loop/anova_pair_rescore_validation_20260526_113620_retry/figures/feature_stability_n1024_top4_pair_rescore.pdf
results/innovation_loop/final_candidate_figures/stabilization_pair_scoring_summary.pdf
```

## Current Paper Direction

Best current framing:

```text
KAN-native support stabilization + functional-ANOVA interaction scoring
```

Safe claim:

KAN-native stability reaches the oracle-support regime under weak centered interactions much earlier than RF screening, and functional-ANOVA scoring can recover interaction pairs that FD mixed-derivative scoring mis-ranks.

Additional safe refinement:

Single-pass KAN controls show that cross-seed stability aggregation is doing useful work beyond a simple KAN-native screen-and-refit pipeline.

Claim to avoid:

Do not say oracle/stable support necessarily fails at weak interactions without specifying the pair scorer.

Current innovation memo:

```text
docs/innovation_status_20260526.md
```
