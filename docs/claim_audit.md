# Claim Audit

Last updated: 2026-05-26

This file maps the main paper claims to figures, tables, and source CSVs. It is meant to keep inline numeric claims traceable during revision.

## Main Claims

| Claim in draft | Evidence in paper | Source artifact | Numeric check |
| --- | --- | --- | --- |
| Prediction error can be low while formula interaction recovery fails. | Figure 1 / `fig:prediction-vs-formula` | `results/hard_regime/paper_figures/hard_regime_augmented_summary.csv` | Raw KAN, `d=100,c=0.10,n=1024`: test MSE `0.002494`, interaction F1 `0.0`. |
| Raw KAN shows an empirical recovery boundary over sample size and interaction strength. | Figure 2 / `fig:recovery-boundary` | `results/hard_regime/paper_figures/hard_regime_augmented_summary.csv` | Raw KAN, `d=100,n=1024`: interaction F1 is `0.0`, `0.6`, `0.9`, `0.8` for `c=0.10`, `0.25`, `0.50`, `1.00`. |
| Oracle support separates low-dimensional representability from full-space support discovery. | Figure 2 and Table 1 | `results/hard_regime/paper_figures/hard_regime_augmented_summary.csv`; `results/stability_kan/paper_figures/sskan_d100_method_comparison.csv` | In moderate/strong settings, oracle-support KAN reaches endpoint recall `1.0` and high interaction F1; weak `c=0.25` remains imperfect. |
| KAN-native stability improves recovery and reaches oracle-support behavior in key weak regimes. | Figure 3 / `fig:stabilization-pair` | `results/innovation_loop/strict_screened_baseline_controls_20260526_104243/analysis_min8_live/native_screened_key_table.csv`; `results/innovation_loop/final_candidate_figures/stabilization_pair_scoring_summary.pdf` | At `c=0.25,d=100,n=512,m=4`, KAN-F/KAN-FE F1 `0.70/0.80`, oracle `0.60`, RF `0.00`; at `n=2048,m=6`, KAN-F/KAN-FE and oracle are all `0.80`, RF `0.10`. |
| Stability aggregation improves over a single KAN explanation pass. | Figure 3(a) and Appendix single-pass table | `results/innovation_loop/one_shot_kan_screen_controls_after_screened_20260526_104933/combined_one_shot_summary.csv`; `results/innovation_loop/strict_screened_baseline_controls_20260526_104243/analysis_min8_live/native_screened_combined_summary.csv` | Single-pass KAN-FE vs stability KAN-FE: `0.00` vs `0.80` at `c=0.25,n=512,m=4`; `0.40` vs `0.70` at `c=0.25,n=1024,m=4`; `0.40` vs `0.80` at `c=0.50,n=512,m=5`. |
| Weak interactions can be under-counted by FD-scored pair recovery even after stable support. | Figure 3 and Appendix pair-scoring table | `results/innovation_loop/anova_pair_rescore_validation_20260526_113620_retry/pair_rescore_summary.csv` | At `c=0.25,d=100,n=512,m=4`, KAN-F/KAN-FE FD F1 `0.50/0.60`; ANOVA-abs, ANOVA-var, and hybrid F1 `1.00` for both. |
| RF screening misses centered weak-interaction endpoints. | New diagnostic figure/table candidate | `results/innovation_loop/rf_screening_diagnostic_20260526_110853/rf_screening_summary.csv` | RF true-pair retention is `0.00` for `n=512/1024`, 500/2000 trees, `top_m=4/5/6`; mean endpoint rank is about `32-35`. |
| KAN-native stability reaches the oracle-support regime without external tree screening. | New main-method table/figure candidate | `results/innovation_loop/strict_screened_baseline_controls_20260526_104243/analysis_min8_live/native_screened_key_table.csv` | At `c=0.25,d=100,n=2048,top_m=6`, KAN feature/feature+edge stability F1 `0.80`, oracle F1 `0.80`, RF F1 `0.10`. |
| Functional-ANOVA pair scoring can correct FD pair mis-ranking after oracle or KAN-native support. | Figure 3 / `fig:stabilization-pair`; Appendix pair-scoring table | `results/innovation_loop/pair_scoring_oracle_pilot_after_rf_20260526_111150/pair_scoring_summary.csv`; `results/innovation_loop/anova_pair_rescore_validation_20260526_113620_retry/pair_rescore_summary.csv` | Oracle pilot: FD F1 `0.60`, ANOVA/hybrid F1 `1.00`; stability rescore: KAN-F/KAN-FE FD F1 `0.50/0.60`, ANOVA/hybrid F1 `1.00`. |
| Extreme nuisance dimension remains a support-recovery bottleneck. | Main stress-test paragraph and Appendix high-dimension table | `results/innovation_loop/strict_validation_20260526_011917/innovation_summary.csv`; `results/innovation_loop/strict_screened_baseline_controls_20260526_104243/combined_screened_baseline_summary.csv` | KAN-FE: `0.50` at `c=0.25,d=500,n=2048`, but `0.00` at `d=1000`; at `c=1.0,d=500,n=1024` it reaches `1.00`, but drops to `0.00` at `d=1000`. |
| Variable recovery, endpoint retention, and pair recovery are distinct levels. | Figure 4 / `fig:ladder` | `results/hard_regime/paper_figures/endpoint_ladder_d100.csv`; `results/hard_regime/paper_figures/hard_regime_augmented_summary.csv` | Raw KAN, `d=100,c=1.00,n=1024`: variable F1 `0.925`, endpoint recall `0.85`, interaction F1 `0.8`. |
| Random support and endpoint exclusion are negative controls. | Figure 5 / `fig:controls` | `results/hard_regime/paper_figures/paper_figure_source_data.csv` | Random-support and endpoint-exclusion rows remain near zero interaction F1 in the main control regimes. |
| Support retention is not unique to KAN, but low-dimensional KAN is more accurate in this setup. | Table `tab:baseline` | `results/paper_figures/final_baseline_table.csv` | MLP RF/oracle interaction F1 `1.0`, but test MSE `0.832/0.877`; KAN RF/oracle test MSE `0.000334/0.000288`. |

## Wording Boundaries

Use:

```text
empirical recovery boundary
support-retention bottleneck
pair-ranking boundary
KAN-native support stabilization
functional-ANOVA interaction scoring
```

Avoid:

```text
formal phase transition
proved KAN sample complexity
symbolic formula extraction
KAN uniquely fails
RF solves formula recovery
FD/Hessian robustness proves pair-scoring is not the issue
```

## Remaining Checks Before Submission

- Re-run this audit after any new experiment or figure regeneration.
- Keep inline numeric claims close to a figure/table or move them into captions/tables.
- If an external-validation appendix is added later, add each new claim to this file before changing the main text.
