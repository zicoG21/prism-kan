# Paper Draft

This directory contains the working LaTeX draft for the KAN-XAI formula-fidelity paper.

## Build

```bash
cd paper
make
```

This requires `latexmk`. If `latexmk` is unavailable, use:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Main Figure Sources

The LaTeX draft references existing generated figures:

- `../results/hard_regime/paper_figures/fig1_formula_phase_d100.pdf`
- `../results/hard_regime/paper_figures/fig2_prediction_vs_formula_raw_d100.pdf`
- `../results/hard_regime/paper_figures/fig3_variable_vs_interaction_raw_d100.pdf`
- `../results/hard_regime/paper_figures/fig4_negative_controls_d100.pdf`

Source CSVs:

- `../results/hard_regime/paper_figures/paper_figure_source_data.csv`
- `../results/hard_regime/paper_figures/hard_regime_augmented_summary.csv`
- `../results/hard_regime/paper_figures/endpoint_ladder_d100.csv`
- `../results/paper_figures/final_baseline_table.csv`

## Current Writing Stance

Use `recovery boundary` or `recovery threshold` in the main text. Avoid claiming a formal sharp `phase transition` unless later theory supports it.

Keep the theory section as mechanistic intuition. Do not promote the local-spline SNR calculation to a theorem unless the variance and independence assumptions are fully justified.
