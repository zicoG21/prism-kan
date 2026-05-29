# pyKAN Support and Interaction Diagnostic Suite

This repository contains the paper-facing artifacts for:

**Auditing pyKAN Readouts for Support and Interaction Recovery**

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

- `results/workshop_review_tables/standard_audit_protocol/audit_protocol_counts_with_ci.csv`
- `results/workshop_review_tables/standard_audit_protocol/audit_protocol_summary.md`
- `results/workshop_review_tables/standard_audit_protocol/audit_protocol_schema.json`
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

The current revision also runs a boundary full-KAN check for the grid-update
failure row at `c=0.25,d=100,n=512,width=16`.  This distinguishes readout-only
surfacing failure from a broader model-reliance/fitting failure:

```bash
python scripts/run_full_kan_pair_anova_probe.py \
  --function core_interaction_c025 \
  --samples 512 \
  --test-samples 2048 \
  --dimension 100 \
  --seeds 0-29 \
  --width-hidden 16 \
  --grid 5 \
  --k 3 \
  --steps 75 \
  --lamb 1e-3 \
  --update-grid \
  --grid-update-num 5 \
  --pair-mode all \
  --max-all-pairs 10000 \
  --anova-points 16 \
  --anova-background 16 \
  --batch-size 4096 \
  --device cuda \
  --out-dir results/revision/fullkan_anova_boundary/gridupdate_w16_n512
```

Expected summary for the completed grid-update boundary row:

- all fits: true pair rank-1 in `0/30` seeds;
- mean / median test MSE: `0.119 / 0.0687`;
- mean / median true-pair rank among all 4950 pairs: `2546 / 2768`;
- mean / median true-minus-max-false ANOVA margin: `-0.0615 / -0.0479`;
- even the subset with test MSE `< 0.05` is `0/11`.

This row is therefore a fitting/reliance failure as well as a readout-surfacing
failure under the grid-update protocol.

## Critical-Regime pyKAN Readout Sensitivity

For the most reviewer-relevant 30-seed confirmation run, use the resumable
Python controller. It defaults to CPU to avoid losing an overnight run to
device-specific pyKAN/Torch failures; set `FOCUSED_DEVICE=auto` or
`FOCUSED_DEVICE=cuda` if you want GPU execution.

```bash
export PYTHON="${PYTHON:-python}"
FOCUSED_DEVICE=cpu "$PYTHON" scripts/run_revision_focused_30seed_core.py
```

This focused run is narrower than the 12-hour stretch pack.  It reruns the
paper's core boundary rows and the non-monotone `c=0.10,d=20` margin cell with
30 seeds, using the same rank/margin outputs as the manuscript.  Outputs:

- `results/revision/focused_30seed_core/*/support_sensitivity_summary.csv`
- `results/revision/focused_30seed_core/summary/focused_30seed_main_rows.csv`
- `results/revision/focused_30seed_core/summary/summary.md`

The reviewer-facing sensitivity check around the main transition
(`c=0.25, d=100`) is:

```bash
bash scripts/run_revision_d100_hparam_sensitivity.sh
```

It runs the inspected pyKAN readout under a one-factor grid around the paper
default:

- `n in {512, 896, 1024}`;
- width `8/16/32`;
- grid `3/5/10`;
- lambda `1e-4/1e-3/1e-2`;
- default 35 probe steps plus a 100-step default check;
- support budgets `m in {4, 6, 10, 20}`.

Outputs are written under:

- `results/revision/d100_c025_hparam_sensitivity/`
- `results/revision/d100_c025_hparam_sensitivity/summary/d100_c025_hparam_sensitivity_focus.csv`
- `results/revision/d100_c025_hparam_sensitivity/summary/summary.md`

This check is meant to test whether the finite-sample endpoint transition is a
single hyperparameter artifact. It is not a full KAN architecture search.

The current paper-facing compact result at `n=512` after the focused 30-seed
rerun is:

| configuration | endpoints@4 | worst endpoint rank | margin |
| --- | ---: | ---: | ---: |
| clean, width 8, no grid update | 30/30 | 4.0 | 0.006 |
| clean, width 16, no grid update | 30/30 | 4.0 | 0.020 |
| noise 0.10, width 16, no grid update | 29/30 | 4.03 | 0.002 |
| clean, width 16, grid update | 0/30 | 40.5 | -0.001 |

The same grid-update protocol recovers at `n=1024` (`30/30`, rank `4.0`,
margin `0.041`).  The paper therefore treats grid update as a finite-data
protocol boundary in this setting, not as a universal failure mode.

## pyKAN Pruning / Symbolic Smoke Test

This optional smoke test addresses whether the diagnostic can also touch
pyKAN's exposed pruning/symbolic workflow.  It is not a full symbolic-recovery
benchmark: it records pruned input support, endpoint containment, full/pruned
MSE, and whether `symbolic_formula()` can be called after pruning.

```bash
python scripts/run_pykan_prune_symbolic_smoke.py \
  --function core_interaction_c025 \
  --samples 512 \
  --test-samples 2048 \
  --dimension 100 \
  --seeds 0-9 \
  --width-hidden 16 \
  --grid 5 \
  --k 3 \
  --steps 75 \
  --lamb 1e-3 \
  --device cuda \
  --thresholds 0.01 0.03 0.05 0.10 \
  --workflows prune_input prune \
  --symbolic-smoke \
  --out-dir results/revision/pykan_prune_symbolic_smoke/core_c025_d100_n512_w16
```

The launcher `scripts/launch_pykan_prune_symbolic_smoke_now.sh` runs the same
smoke test for `n=512` and `n=1024`.  Expected outputs are
`pykan_prune_symbolic_detail.csv` and `pykan_prune_symbolic_summary.csv`.

## Low-/Moderate-Dimensional Phase Grid

The current draft also includes a lower-dimensional phase grid to keep the
main evidence away from only `d=1000` stress behavior:

```bash
export PYTHON="${PYTHON:-python}"
bash scripts/run_revision_lowdim_phase_grid.sh
```

This runs the fixed pyKAN readout protocol over:

- `c in {0.10, 0.25, 0.50}`;
- `d in {20, 50, 100}`;
- `n in {256, 512, 1024}`;
- support budgets `m in {4, 6, 10, 20}`;
- readouts `feature_stability_var` and `feature_edge_hybrid`.

It also runs a compact width check at `c=0.25, n=512` for
`d in {20, 50, 100}` and `width in {8, 16, 32}`.

Outputs are written under:

- `results/revision/lowdim_phase_grid/phase/`
- `results/revision/lowdim_phase_grid/width_check/`
- `results/revision/lowdim_phase_grid/summary/summary.md`
- `results/revision/lowdim_phase_grid/summary/lowdim_phase_grid_focus.csv`
- `results/revision/lowdim_phase_grid/summary/lowdim_width_check_focus.csv`

The exploratory phase-grid pattern is:

- very weak interactions (`c=0.10`) expose protocol sensitivity in small
  grids and should be followed up with focused reruns;
- moderate interactions (`c=0.25`) recover by `n=1024` for `d=20/50/100`;
- stronger interactions (`c=0.50`) are clean positive controls by `n=1024`;
- widening from 8 to 16 or 32 hidden units removes the `d=100,n=512`
  endpoint failure in this low-dimensional check.

The phase grid is reported as a coverage map rather than a precise hypothesis
test.  Representative Wilson intervals are included in the standardized audit
summary produced by `scripts/run_standard_audit_protocol.py`.

One retained diagnostic cell appeared non-monotone:
`c=0.10,d=20,m=4` succeeded at `n=256` but failed at larger `n` under an
earlier fixed script.  The 30-seed follow-up shows that this is a real
readout-margin boundary rather than a clean sample-size curve.  Width 8 mostly
misses (`2/30`, `0/30`, `0/30` for `n=256/512/1024`).  Width 16 misses at
`n=256` (`0/30`), succeeds at `n=512` (`30/30`), and misses at `n=1024`
(`0/30`); width 32 succeeds at `n=256/512` (`30/30`) but mostly misses at
`n=1024` (`1/30`).  We now report this as protocol-sensitive weak-interaction
behavior and rely on endpoint rank/margin, not monotone success counts.

## Noise/Correlation Appendix Checks

The paper includes a small appendix check at `c=0.25, d=100, n=1024` with
five support-evaluation seeds and eight nuisance proxies.  The checked-in
summaries are under:

- `results/workshop_review_tables/kan_probe_noise_corr_c025_d100_n1024/`
- `results/workshop_review_tables/residual_pair_screen_noise_corr_c025_d100_n1024/`
- `results/workshop_review_tables/workshop_6of10_checks/`

These rows are intended as robustness checks, not as a full correlated-feature
benchmark.

## Semi-Synthetic Real-Covariate Check

The current draft includes a small external-validity check using real tabular
covariate distributions with injected known interactions.  It uses the
scikit-learn `diabetes` and `breast_cancer` covariates, standardizes features,
applies a `tanh` compression, and injects
`sin(pi*x0) + x1^2 + c*x2*x3`.

Full run:

```bash
export PYTHON="${PYTHON:-python}"
bash scripts/run_revision_semisynthetic_covariates_3h.sh
```

This runs:

- datasets: `diabetes`, `breast_cancer`;
- coefficients: `c in {0.10, 0.25, 0.50}`;
- samples: `n in {128, 256, 384}`;
- 10 outer seeds;
- `R=12` pyKAN probes per outer seed;
- readouts: `feature_stability_var`, `feature_edge_hybrid`;
- support budget: `m=4`.

Outputs are written under:

- `results/revision/semisynthetic_covariates_3h/semisynthetic_covariate_audit_detail.csv`
- `results/revision/semisynthetic_covariates_3h/semisynthetic_covariate_audit_summary.csv`

To regenerate the compact paper-facing table from the checked-in summary:

```bash
python scripts/summarize_revision_semisynthetic_covariates.py
```

Expected compact outputs:

- `results/revision/semisynthetic_covariates_3h/summary/semisynthetic_covariate_compact_table.csv`
- `results/revision/semisynthetic_covariates_3h/summary/semisynthetic_covariate_compact_table.tex`
- `results/revision/semisynthetic_covariates_3h/summary/summary.md`

The check is intentionally not a real-data benchmark: the ground-truth target is
injected, but the nuisance/proxy geometry comes from real covariates.

## NID-Style Neural Interaction Baseline

The repository also includes a lightweight Neural Interaction Detection
(NID-style) MLP baseline.  This is a reviewer-facing calibration reference, not
a tuned neural interaction leaderboard.  It asks whether a standard weight-based
neural interaction score ranks the known pair highly under the same weak
centered task.

```bash
python experiments/run_nid_interaction_baseline.py \
  --function core_interaction_c025 \
  --samples 512 \
  --test_samples 2048 \
  --dimension 100 \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --methods nid \
  --hidden 128 \
  --depth 2 \
  --epochs 800 \
  --patience 120 \
  --device cuda \
  --out_dir results/interaction_baselines/nid/core_c025_d100_n512_10seed
```

For the current revision, the queue script waits for the full-KAN ANOVA jobs and
then runs 30-seed NID baselines at `n=512` and `n=1024`, writing to
`results/revision/nid_baseline_core/core_c025_d100_n*_fixed/`:

```bash
bash scripts/launch_revision_nid_after_fullkan.sh
tail -f results/revision/nid_baseline_core/queue.log
```

## Reviewer-Boundary Overnight Pack

For a longer reviewer-response run, use the 12-hour boundary pack:

```bash
export PYTHON="${PYTHON:-python}"
HOURS=12 RUN_DEVICE=cuda bash scripts/launch_revision_boundary_overnight_12h.sh
```

This launches `scripts/run_revision_boundary_overnight_12h.sh` under a timeout
and writes to:

```text
results/revision/boundary_overnight_12h/
```

It targets the boundary conditions reviewers asked for:

- critical-regime pyKAN width/training-step sensitivity at `c=0.25,d=100`;
- noise and correlated-proxy variants at `n=512/1024`;
- semi-synthetic real-covariate checks with injected interactions and noise;
- pair-feature Lasso, GBM H-statistic, and residual RFF-HSIC baselines on
  noisy/correlated slices.
- a stretch queue, reached only if earlier stages finish early, with grid-update
  pyKAN checks, a third real covariate distribution (`wine`), and non-product /
  multi-interaction baseline slices.

Monitor:

```bash
tail -f results/revision/boundary_overnight_12h/master.log
ps -p "$(cat results/revision/boundary_overnight_12h/LAUNCH_PID)" -o pid,etime,cmd
```

Summarize completed partial or full outputs:

```bash
python scripts/summarize_revision_boundary_overnight.py \
  --root results/revision/boundary_overnight_12h
```

Expected summary outputs:

- `results/revision/boundary_overnight_12h/summary/summary.md`
- `results/revision/boundary_overnight_12h/summary/kan_boundary_focus.csv`
- `results/revision/boundary_overnight_12h/summary/semisynthetic_focus.csv`
- `results/revision/boundary_overnight_12h/summary/lasso_focus.csv`
- `results/revision/boundary_overnight_12h/summary/hsic_focus.csv`

The pack is designed to preserve partial results if the timeout stops it before
all stages finish.

Baseline roles used in the paper:

- Pair-feature Lasso: standardized main effects plus all raw pair products,
  LassoCV over the regularization path, reported as selected-library endpoint
  retention and top-1 product recovery.
- GBM H-statistic: shallow scikit-learn `HistGradientBoostingRegressor`
  (`max_iter=100`, `max_leaf_nodes=31` in the current table) followed by an
  H-statistic score over a raw-product-prefiltered candidate set.  Candidate
  retention and top-1 pair recovery are intentionally reported separately.
- Residual RFF-HSIC: 5-fold additive spline residual, fixed RFF dimensions
  (`64` for the pair map, `32` for the residual map in the boundary pack), and
  direct pair ranking.  The paper table uses 12-trial noisy/correlated slices
  and notes the independent 50-seed clean check (`2/50` at `n=512`, `50/50` at
  `n=1024`).

## Paper Draft

The current reviewer-facing workshop draft is built from:

```bash
cd manuscripts/workshop_case_study
latexmk -pdf -interaction=nonstopmode main.tex
```

The compiled PDF is:

```text
manuscripts/workshop_case_study/main.pdf
```

Older `paper*` folders are archived; active manuscript directions are described
in `PAPER_FOLDERS.md`.

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
  scripts/build_highdim_prediction_clean_case.py \
  scripts/run_revision_d100_hparam_sensitivity.sh \
  scripts/summarize_revision_d100_hparam.py \
  scripts/run_revision_lowdim_phase_grid.sh \
  scripts/summarize_revision_lowdim_phase_grid.py \
  scripts/run_revision_semisynthetic_covariates_3h.sh \
  scripts/summarize_revision_semisynthetic_covariates.py \
  scripts/run_revision_boundary_overnight_12h.sh \
  scripts/launch_revision_boundary_overnight_12h.sh \
  scripts/summarize_revision_boundary_overnight.py \
  experiments/run_semisynthetic_covariate_audit.py
git add PAPER_FOLDERS.md manuscripts
git add -f docs/next_revision_minimal_additions.md
git add -f results/revision/d100_c025_hparam_sensitivity/summary
git add -f results/revision/lowdim_phase_grid/summary
git add -f results/revision/semisynthetic_covariates_3h/summary
```

If you only want to push source and not generated PDFs/logs, replace
`git add PAPER_FOLDERS.md manuscripts` with explicit source adds for
`manuscripts/*/main.tex`, `references.bib`, style files, and README files.
