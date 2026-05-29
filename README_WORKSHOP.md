# pyKAN Support and Interaction Diagnostic Suite

This repository contains the paper-facing artifacts for:

**A Controlled Diagnostic Suite for pyKAN Support and Interaction Recovery**

The suite checks whether pyKAN-style explanations recover known active variables
and interaction pairs under nuisance features.  It reports four levels:

1. Prediction error.
2. Active-variable recovery.
3. Interaction-endpoint retention.
4. Top-ranked interaction-pair recovery.

The current paper focuses on a reproducible diagnostic suite for pyKAN support
and interaction recovery.

## Reviewer Quickstart

The fastest check does not retrain pyKAN models.  It verifies the schema,
Wilson intervals, and the mini-suite table from the checked-in CSV summaries:

```bash
python scripts/print_artifact_env.py
python scripts/run_standard_audit_protocol.py \
  --out-dir results/workshop_review_tables/standard_audit_protocol
python scripts/build_formal_minisuite_baseline_table.py
```

Expected outputs:

- `results/workshop_review_tables/standard_audit_protocol/audit_summary.csv`
- `results/workshop_review_tables/standard_audit_protocol/audit_summary.md`
- `results/workshop_review_tables/standard_audit_protocol/audit_schema.json`
- `results/workshop_review_tables/formal_minisuite/formal_minisuite_baseline_table.csv`
- `results/workshop_review_tables/formal_minisuite/formal_minisuite_baseline_table.tex`

On the machine used for the paper, these checks are minute-scale.  The pyKAN
retraining commands below are longer-running.

## Environment

The experiments were run with the local `prism` conda environment.  A minimal
setup is:

```bash
conda create -n prism python=3.10 -y
conda activate prism
pip install -r requirements.txt
```

If `pykan` is unavailable from PyPI, install it from the upstream repository:

```bash
pip install git+https://github.com/KindXiaoming/pykan.git
```

The environment used for the paper can be inspected with:

```bash
python scripts/print_artifact_env.py
```

Recorded local context:

- Python 3.10.20
- NumPy 2.2.6, SciPy 1.15.3, pandas 2.3.3, scikit-learn 1.7.2
- PyTorch 2.11.0+cu130
- pyKAN imported as `kan` from the active environment
- CPU: Intel i7-13700HX, 24 logical threads
- GPU available during runs: NVIDIA RTX 4050 Laptop GPU, 6 GB

## Core Finite-Data Diagnostic

```bash
bash scripts/reproduce_workshop_audit_core.sh
```

This regenerates the finite-data pyKAN bootstrap outputs, KAN-readout ablation,
residual raw-product baselines, residual tensor-spline mini-suite screen, and
the support-budget figure.  It writes into:

- `results/workshop_review_tables/same_data_kan_stability_c025_d100_bootstrap_R20_10seed/`
- `results/workshop_review_tables/same_data_kan_score_ablation_c025_d100_R20_10seed_skiprefit/`
- `results/interaction_baselines/residual_pair_screen_c025_d100_n512_1024/`
- `results/interaction_baselines/residual_pair_screen_crossfit_c025_d100/`
- `results/interaction_baselines/residual_tensor_spline_minisuite_trainresid_alpha1_d100_n1024_10seed/`
- `paper_neurips_workshop_6_8/figures/support_size_curves.pdf`

The script is intentionally explicit and can be edited to reduce seed counts for
a smoke test.

## Standardized Diagnostic Schema

The paper-facing schema uses:

- `prediction_mse`: test-set MSE on standardized targets.
- `active_variable_f1`: top-`k` active-variable F1 in controlled tasks.
- `endpoint_retention`: success if all true interaction endpoints are retained.
- `top1_pair`: success if the top-ranked candidate pair is a true pair.
- `exact_support`: success if selected support equals the true active set.

The `run_standard_audit_protocol.py` script can also consume a user-provided CSV
with columns:

```text
label,protocol,metric,successes,trials,notes
```

and emits a normalized CSV/Markdown summary with Wilson confidence intervals.

## Full-KAN All-Pairs ANOVA Check

This is the 30-seed, all-pairs result reported in the main paper.  It ranks all
`100 * 99 / 2 = 4950` pairs directly on the full-dimensional KAN.

```bash
python scripts/run_full_kan_pair_anova_probe.py \
  --function core_interaction_c025 \
  --samples 1024 \
  --test-samples 2048 \
  --dimension 100 \
  --seeds 0-29 \
  --width-hidden 8 \
  --grid 5 \
  --k 3 \
  --steps 50 \
  --lamb 1e-3 \
  --pair-mode all \
  --anova-points 16 \
  --anova-background 16 \
  --batch-size 4096 \
  --out-dir results/workshop_review_tables/full_kan_pair_anova_probe/c025_d100_n1024_allpairs_30seed
```

Expected summary:

- all fits: true pair rank-1 in `24/30` seeds;
- fit-clean subset with test MSE `< 5e-3`: true pair rank-1 in `24/24`
  seeds.

For a faster code-path smoke test, reduce `--seeds 0-29` to `--seeds 0-2` and
write to a temporary output directory.

## Noise/Correlation Appendix Checks

The paper includes a small appendix check at `c=0.25, d=100, n=1024` with
five support-evaluation seeds and eight nuisance proxies.  The checked-in
summaries are under:

- `results/workshop_review_tables/kan_probe_noise_corr_c025_d100_n1024/`
- `results/workshop_review_tables/residual_pair_screen_noise_corr_c025_d100_n1024/`
- `results/workshop_review_tables/workshop_6of10_checks/`

These rows are intended as robustness checks, not as a full correlated-feature
benchmark.

## Paper Draft

The current workshop draft is built from:

```bash
cd paper_neurips_workshop_6_8
latexmk -pdf -interaction=nonstopmode main.tex
```

The compiled PDF is:

```text
paper_neurips_workshop_6_8/workshop_6_8_submission.pdf
```

For the current reviewer-facing copy:

```text
paper_neurips_workshop_6_8/professor_reviewer_draft.pdf
```

## Artifact Map

Additional notes are in:

- `docs/artifact_checklist_workshop_20260527.md`
- `docs/artifact_manifest_20260528.md`

## Notes on GitHub Submission

The repository `.gitignore` ignores `paper*/` directories.  To include the
current paper source and PDF in a public push, add them explicitly:

```bash
git add README_WORKSHOP.md README.md requirements.txt
git add scripts/print_artifact_env.py \
  scripts/run_standard_audit_protocol.py \
  scripts/run_full_kan_pair_anova_probe.py \
  scripts/build_formal_minisuite_baseline_table.py \
  scripts/build_highdim_prediction_clean_case.py
git add -f paper_neurips_workshop_6_8/main.tex \
  paper_neurips_workshop_6_8/workshop_6_8_submission.pdf \
  paper_neurips_workshop_6_8/professor_reviewer_draft.pdf
```
