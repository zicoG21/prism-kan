# Reproducibility Checklist

Last updated: 2026-05-25

## Main Artifact Locations

Paper draft:

```text
paper/main.tex
paper/references.bib
```

Main figures:

```text
results/hard_regime/paper_figures/fig1_formula_phase_d100.pdf
results/hard_regime/paper_figures/fig2_prediction_vs_formula_raw_d100.pdf
results/hard_regime/paper_figures/fig3_variable_vs_interaction_raw_d100.pdf
results/hard_regime/paper_figures/fig4_negative_controls_d100.pdf
```

Main source data:

```text
results/hard_regime/paper_figures/hard_regime_augmented_summary.csv
results/hard_regime/paper_figures/paper_figure_source_data.csv
results/hard_regime/paper_figures/endpoint_ladder_d100.csv
results/paper_figures/final_baseline_table.csv
```

## Regenerate Hard-Regime Summaries and Figures

The expensive KAN runs are already saved in:

```text
results/hard_regime/details/
```

To recompute endpoint/margin diagnostics and summaries from existing detail CSVs:

```bash
python experiments/augment_hard_regime_metrics.py --write_details
```

To redraw the main d=100 paper figures:

```bash
python experiments/plot_paper_hard_regime_figures.py \
  --dims 100 \
  --out_dir results/hard_regime/paper_figures
```

To regenerate the compact baseline table:

```bash
python experiments/make_paper_baseline_table.py
```

## Re-run the Full Hard-Regime Matrix

This is expensive:

```bash
./run_hard_regime_matrix.sh
```

The script runs:

- `c in {0.10, 0.25, 0.50, 1.00}`
- `n in {128, 256, 512, 1024}`
- `d in {50, 100}`
- screen modes: raw, RF, oracle support, random, exclude interaction
- seeds: 0 through 9

## Compile the Paper

```bash
cd paper
make
```

If `latexmk` is not installed:

```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Pre-Submission Cleanup

- Verify BibTeX metadata against official arXiv pages.
- Move exploratory generated files out of the main artifact path.
- Decide whether d=50 figures go to appendix or are omitted.
- Keep `phase transition` language out of the main claim unless changed to `phase-transition-like`.
- Keep SNR theory as heuristic unless fully proved.
- Confirm all figures can be regenerated from committed scripts and source CSVs.
