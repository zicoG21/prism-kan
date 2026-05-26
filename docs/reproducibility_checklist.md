# Reproducibility and Artifact Map

Last updated: 2026-05-26

## Paper Draft

```text
paper/main.tex
paper/main.pdf
paper/references.bib
```

Compile:

```bash
cd paper
latexmk -pdf -interaction=nonstopmode main.tex
```

Fallback if `latexmk` is unavailable:

```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Claim-to-Artifact Map

| Paper claim | Main artifact | Source data | Regeneration script |
| --- | --- | --- | --- |
| Prediction error is not formula fidelity. | `results/hard_regime/paper_figures/fig2_prediction_vs_formula_raw_d100.pdf` | `results/hard_regime/paper_figures/paper_figure_source_data.csv` | `experiments/plot_paper_hard_regime_figures.py` |
| Raw KAN shows a recovery boundary over sample size and interaction strength. | `results/hard_regime/paper_figures/fig1_formula_phase_d100.pdf` | `results/hard_regime/paper_figures/hard_regime_augmented_summary.csv` | `experiments/augment_hard_regime_metrics.py`, `experiments/plot_paper_hard_regime_figures.py` |
| Variable recovery, endpoint retention, and pair recovery fail at different levels. | `results/hard_regime/paper_figures/fig3_variable_vs_interaction_raw_d100.pdf` | `results/hard_regime/paper_figures/endpoint_ladder_d100.csv` | `experiments/plot_paper_hard_regime_figures.py` |
| Random support and endpoint exclusion do not recover formula structure. | `results/hard_regime/paper_figures/fig4_negative_controls_d100.pdf` | `results/hard_regime/paper_figures/paper_figure_source_data.csv` | `experiments/plot_paper_hard_regime_figures.py` |
| KAN-native support stabilization reaches oracle-support behavior while RF screening misses weak endpoints. | `results/innovation_loop/final_candidate_figures/stabilization_pair_scoring_summary.pdf` | `results/innovation_loop/strict_screened_baseline_controls_20260526_104243/analysis_min8_live/native_screened_combined_summary.csv` | `experiments/compare_native_with_screened_controls.py`, `experiments/plot_stabilization_pair_scoring_figure.py` |
| Stability aggregation improves over a single KAN screen-and-refit pass. | `results/innovation_loop/final_candidate_figures/stabilization_pair_scoring_summary.pdf`; Appendix single-pass table | `results/innovation_loop/one_shot_kan_screen_controls_after_screened_20260526_104933/combined_one_shot_summary.csv` | `scripts/run_one_shot_kan_screen_controls.sh`, `experiments/run_single_kan_screen_refit.py`, `experiments/plot_stabilization_pair_scoring_figure.py` |
| Functional-ANOVA pair scoring recovers interactions that FD scoring under-ranks after stable support. | `results/innovation_loop/final_candidate_figures/stabilization_pair_scoring_summary.pdf`; Appendix table in `paper/main.tex` | `results/innovation_loop/anova_pair_rescore_validation_20260526_113620_retry/pair_rescore_summary.csv` | `experiments/rescore_stability_supports_with_pair_methods.py`, `experiments/plot_pair_rescore_summary.py`, `experiments/plot_stabilization_pair_scoring_figure.py` |
| Extreme nuisance dimension remains unsolved by stability alone. | Appendix high-dimension table | `results/innovation_loop/strict_validation_20260526_011917/innovation_summary.csv`; `results/innovation_loop/strict_screened_baseline_controls_20260526_104243/combined_screened_baseline_summary.csv` | `scripts/run_innovation_strict_validation_overnight.sh`, `scripts/run_strict_screened_baseline_controls.sh` |
| KAN and MLP share support-retention bottlenecks, while KAN fits low-dimensional formulas more accurately. | Table `tab:baseline` in `paper/main.tex` | `results/paper_figures/final_baseline_table.csv` | `experiments/make_paper_baseline_table.py` |

## Main Figure Files

```text
results/hard_regime/paper_figures/fig1_formula_phase_d100.pdf
results/hard_regime/paper_figures/fig2_prediction_vs_formula_raw_d100.pdf
results/hard_regime/paper_figures/fig3_variable_vs_interaction_raw_d100.pdf
results/hard_regime/paper_figures/fig4_negative_controls_d100.pdf
results/innovation_loop/final_candidate_figures/stabilization_pair_scoring_summary.pdf
results/innovation_loop/anova_pair_rescore_validation_20260526_113620_retry/figures/feature_edge_n1024_top4_pair_rescore.pdf
```

## Main Source Data

```text
results/hard_regime/paper_figures/hard_regime_augmented_summary.csv
results/hard_regime/paper_figures/paper_figure_source_data.csv
results/hard_regime/paper_figures/endpoint_ladder_d100.csv
results/paper_figures/final_baseline_table.csv
results/stability_kan/paper_figures/sskan_d100_method_comparison.csv
results/stability_kan/paper_figures/sskan_best_stability_settings.csv
results/stability_kan/boundary_c025_n2048_4096_combined/summary.csv
results/stability_kan/metric_check_c025_oracle_fd_vs_hessian/summary.csv
results/innovation_loop/strict_screened_baseline_controls_20260526_104243/analysis_min8_live/native_screened_key_table.csv
results/innovation_loop/strict_screened_baseline_controls_20260526_104243/analysis_min8_live/native_screened_combined_summary.csv
results/innovation_loop/rf_screening_diagnostic_20260526_110853/rf_screening_summary.csv
results/innovation_loop/pair_scoring_oracle_pilot_after_rf_20260526_111150/pair_scoring_summary.csv
results/innovation_loop/anova_pair_rescore_validation_20260526_113620_retry/pair_rescore_summary.csv
results/innovation_loop/one_shot_kan_screen_controls_after_screened_20260526_104933/combined_one_shot_summary.csv
results/innovation_loop/strict_screened_baseline_controls_20260526_104243/combined_screened_baseline_summary.csv
```

## Regenerate Hard-Regime Figures

The expensive KAN runs are saved under:

```text
results/hard_regime/details/
results/hard_regime/summaries/
```

Recompute endpoint and margin diagnostics:

```bash
python experiments/augment_hard_regime_metrics.py --write_details
```

Redraw the main `d=100` hard-regime figures:

```bash
python experiments/plot_paper_hard_regime_figures.py \
  --dims 100 50 \
  --out_dir results/hard_regime/paper_figures
```

The paper figure scripts use Matplotlib's LaTeX text rendering. The local TinyTeX environment needs `type1cm`, `cm-super`, `underscore`, and `dvipng` installed.

Regenerate the KAN-vs-MLP baseline table:

```bash
python experiments/make_paper_baseline_table.py
```

## Regenerate Stability-Selection Artifacts

Strict KAN-native stability validation:

```bash
bash scripts/run_innovation_strict_validation_overnight.sh
```

Paper-ready SS-KAN figures and table:

```bash
python experiments/make_stability_paper_figures.py
```

Weak-interaction `n=2048` report:

```bash
python experiments/make_boundary_2048_report.py \
  --baseline_summary results/stability_kan/boundary_2048_fast_c025_combined/summary.csv \
  --stability_summary results/stability_kan/boundary_2048_fast_c025_topm5/summary.csv \
  --out_dir results/stability_kan/boundary_2048_fast_c025_combined
```

## Regenerate Pair-Scoring Diagnostics

RF screening diagnostic:

```bash
python experiments/check_rf_screening_stability.py \
  --functions core_interaction_c025 \
  --samples 512 1024 \
  --dimension 100 \
  --seeds 100 101 102 103 104 105 106 107 108 109 \
  --top_m 4 5 6 \
  --trees 500 2000 \
  --out_dir results/innovation_loop/rf_screening_diagnostic
```

Oracle-support pair scorer diagnostic:

```bash
python experiments/evaluate_pair_scoring_methods.py \
  --functions core_interaction_c025 \
  --samples 512 \
  --dimension 100 \
  --seeds 100 101 102 103 104 \
  --top_m 4 \
  --out_dir results/innovation_loop/pair_scoring_oracle
```

Rescore existing KAN-native stability supports:

```bash
bash scripts/run_anova_pair_rescore_validation.sh
```

Plot the RF and pair-scoring diagnostic:

```bash
python experiments/plot_rf_and_pair_scoring_diagnostics.py \
  --rf_summary results/innovation_loop/rf_screening_diagnostic_20260526_110853/rf_screening_summary.csv \
  --pair_summary results/innovation_loop/pair_scoring_oracle_pilot_after_rf_20260526_111150/pair_scoring_summary.csv \
  --out results/innovation_loop/diagnostic_figures/rf_pair_scoring_diagnostic
```

Plot the compact main stabilization/pair-scoring figure:

```bash
NATIVE=results/innovation_loop/strict_screened_baseline_controls_20260526_104243
PAIR=results/innovation_loop/anova_pair_rescore_validation_20260526_113620_retry
ONE=results/innovation_loop/one_shot_kan_screen_controls_after_screened_20260526_104933
python experiments/plot_stabilization_pair_scoring_figure.py \
  --combined_summary $NATIVE/analysis_min8_live/native_screened_combined_summary.csv \
  --pair_summary $PAIR/pair_rescore_summary.csv \
  --one_shot_summary $ONE/combined_one_shot_summary.csv \
  --out results/innovation_loop/final_candidate_figures/stabilization_pair_scoring_summary
```

One-shot KAN screen-and-refit controls:

```bash
bash scripts/run_one_shot_kan_screen_controls.sh
```

## Legacy Metric Robustness Check

The finite-difference rows are in:

```text
results/stability_kan/boundary_2048_fast_c025_oracle/oracle_summary.csv
results/stability_kan/boundary_4096_fast_c025_oracle/oracle_summary.csv
```

The Hessian rows were generated with `experiments/run_tuned_kan_recovery.py` using:

```text
--functions core_interaction_c025
--screen_modes oracle_support
--dimension 100
--samples 2048 or 4096
--seeds 0 1 2 3 4
--interaction_method hessian
```

Combined metric summary:

```text
results/stability_kan/metric_check_c025_oracle_fd_vs_hessian/summary.csv
```

## Re-run the Full Hard-Regime Matrix

The old root-level `run_hard_regime_matrix.sh` script has been removed in the current workspace. Use the experiment driver directly or restore the historical script from version control if a full rerun is required. The saved matrix currently covers:

- `c in {0.10, 0.25, 0.50, 1.00}`
- `n in {128, 256, 512, 1024}`
- `d in {50, 100}`
- screen modes: raw, RF, oracle support, random, exclude interaction
- seeds: 0 through 9

## Submission Cleanup

- BibTeX metadata for the two KAN papers has been checked against the arXiv pages.
- The `d=50` recovery-boundary figure is included as an appendix auxiliary result.
- Keep `recovery boundary` language; avoid claiming a formal sharp phase transition.
- Keep local-spline SNR as heuristic unless fully proved.
- Do not claim symbolic formula extraction unless symbolic extraction is actually evaluated.
- Keep RF-screened KAN as an external diagnostic baseline, not the main method.
