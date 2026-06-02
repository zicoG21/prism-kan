# ClaimTransfer-Bench for Formula-Recovery Workflows

This repository contains the paper-facing artifacts for:

**ClaimTransfer-Bench: Typed Structural Claims for Formula-Recovery Workflows**

The suite is a typed structural-claim benchmark protocol.  It checks whether
structural evidence transfers across workflow objects in controlled
formula-recovery tasks.  It reports object-level evidence for:

1. Prediction error.
2. Active-variable recovery.
3. Interaction-endpoint retention.
4. Top-ranked interaction-pair recovery.
5. Downstream pruning/symbolic provenance when available.

The current paper uses pyKAN as the detailed neural reference implementation
and stress-test case study, but the benchmark also includes sparse-library,
GA2M-style, tree-interaction, and symbolic-library workflow families.  Handoff
summaries are descriptive; the primary artifact is the row-level Claim
Provenance Record.

## Benchmark Interface

ClaimTransfer-Bench is organized around four release objects:

- `task_card`: formula id, covariate generator, dimensions/sample size, noise,
  train/test seed, true support, claim specification, official scorer, and declared
  pair/symbolic labels.
- `workflow_adapter`: method id, hyperparameters, exposed evidence objects, and
  extraction rules for support/endpoint/pair/symbolic claims.
- `claim_record`: one row per seed and evidence object with task id, adapter id,
  predicate, scorer, rank, margin, pass/fail, and protocol fields.
- `score_report`: aggregate continuous evidence, derived predicate rates,
  confidence intervals, and handoff summaries.

The checked-in scripts below rebuild reference `claim_record` summaries for the
paper rows; long retraining jobs regenerate the raw records.

Repository layout:

- `task_cards/`: machine-readable task-card examples and task-card rules.
- `adapters/`: adapter contract for exposing native workflow outputs.
- `scorers/`: scorer and predicate definitions.
- `claim_records/`: row-level submission schema and example CSV rows.
- `score_reports/`: aggregate report convention.
- `scripts/run_benchmark.py`: quick reviewer runner for available summaries.

### Benchmark Contract

The benchmark is a row-level structural-claim contract.

- **Input:** a fixed `task_card` with formula/covariate generator, seeds, known
  support, legal structural claims, and official scorers.
- **Submission:** a workflow adapter that writes `claim_record.csv` rows.
- **Required row fields:** `task_id`, `adapter`, `seed`, `evidence_object`,
  `claim_type`, `target`, `scorer`, `rank`, `margin`, `predicate`, `pass`, and
  `protocol`.
- **Scoring:** continuous ranks, margins, support sizes, and MSEs are primary;
  binary predicates are derived summaries with confidence intervals.
- **Aggregate report:** macro summaries are grouped by task card, claim type,
  and evidence object.  The benchmark does not collapse prediction, support,
  pair, and symbolic claims into one scalar leaderboard.
- **Conflict rule:** if two claim specifications for the same formula are both
  meaningful, they are separate task cards, not post-hoc reinterpretations of a
  single result.

Example rows:

```csv
task_id,adapter,seed,evidence_object,claim_type,target,scorer,rank,margin,predicate,pass,protocol
weak_centered_n1024,pyKAN,0,full_function,pair,"(2,3)",fANOVA,1,0.061,rank1,true,w16_grid5_no_update
weak_centered_n1024,pyKAN,0,exposed_readout,endpoints,"(2,3)",KAN-FE,4,0.041,top4,true,w16_grid5_no_update
weak_centered_n1024,pyKAN,1,exposed_readout,endpoints,"(2,3)",KAN-FE,17,-0.006,top4,false,w16_grid5_no_update
```

### Claim Specification Authoring Rule

The claim specification is part of the task card and is fixed before running a
workflow adapter.

- Product cards declare the algebraic product pair(s) in the data-generating
  formula.
- Multi-term rational or mixed-sparse cards declare the finite set of pair
  claims exposed by the formula and the official scorer for each claim.
- Nested, three-way, and compositional cards are tagged as pairwise-stress cards
  when no unique pairwise ground truth is implied by the formula.
- Different specifications for the same formula should be represented as separate
  task cards, not as post-hoc reinterpretations of one result.

### Minimum Score Report

A valid score report keeps continuous evidence primary and derives binary
predicates from it.  At minimum it should include:

- standardized prediction MSE and its task-card calibration scale;
- support F1 or selected support size, plus active-variable ranks when
  available;
- endpoint ranks and endpoint-vs-nuisance margins;
- declared pair rank and true-minus-max-false pair margin under the official
  scorer;
- pruning retained support and symbolic status when the workflow exposes them;
- Wilson intervals for binary predicates and seed quantiles or bootstrap
  intervals for ranks, margins, support sizes, and MSEs.

### Submitting a New Adapter

To add a workflow, implement an adapter that writes `claim_record` rows with the
schema above.  The adapter should expose its native evidence object rather than
forcing all methods into a single importance score.  For example, a sparse
library adapter writes selected variables and pair-term coefficients; a GA2M
adapter writes selected univariate/bivariate components; a symbolic-library
adapter writes variables/operators present in the expression.  The benchmark
then scores those native objects against the task-card specification.

## Reviewer Quickstart

The fastest check does not retrain pyKAN models.  It verifies the schema,
Wilson intervals, and the mini-suite table from the checked-in CSV summaries:

```bash
python scripts/run_benchmark.py
```

The command above expands to:

```bash
python scripts/print_artifact_env.py
python scripts/run_standard_audit_protocol.py \
  --out-dir results/workshop_review_tables/standard_audit_protocol
python scripts/build_formal_minisuite_baseline_table.py
python scripts/build_cross_method_transfer_matrix.py \
  --output-prefix local_notes/generated/reviewer_cross_method_transfer
```

Expected outputs:

- `results/workshop_review_tables/standard_audit_protocol/audit_protocol_counts_with_ci.csv`
- `results/workshop_review_tables/standard_audit_protocol/audit_protocol_summary.md`
- `results/workshop_review_tables/standard_audit_protocol/audit_protocol_schema.json`
- `results/workshop_review_tables/formal_minisuite/formal_minisuite_baseline_table.csv`
- `results/workshop_review_tables/formal_minisuite/formal_minisuite_baseline_table.tex`
- `local_notes/generated/reviewer_cross_method_transfer_method_summary.csv`
- `local_notes/generated/reviewer_cross_method_transfer_transfer_long.csv`

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

### Great Lakes GPU Notes

On University of Michigan Great Lakes, do not test CUDA from a login node: login
nodes do not expose GPUs.  Request a GPU job first.  V100 nodes require a PyTorch
build that supports compute capability 7.0; the CUDA 13 wheels may detect the
device but warn that no matching kernel image is available.  The tested setup is
Python 3.11 with PyTorch 2.5.1 CUDA 12.1:

```bash
module purge
module load python/3.11.5
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
grep -vi '^torch' requirements.txt > /tmp/prism_requirements_no_torch.txt
pip install -r /tmp/prism_requirements_no_torch.txt
```

GPU smoke test:

```bash
sbatch --account=jaabell0 --partition=gpu --gres=gpu:1 \
  --cpus-per-task=4 --mem=16G --time=00:05:00 --wrap='
cd ~/prism-kan
module purge
module load python/3.11.5
source .venv/bin/activate
hostname
nvidia-smi
python - <<PY
import torch
print(torch.__version__)
print("cuda available:", torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
x = torch.randn(2048, 2048, device="cuda")
y = x @ x
torch.cuda.synchronize()
print(float(y.mean()))
PY
'
```

Great Lakes array entry points:

```bash
export PYTHON_BIN=$PWD/.venv/bin/python

sbatch --account=jaabell0 \
  --export=ALL,PYTHON_BIN=$PWD/.venv/bin/python \
  --array=0-11 \
  scripts/greatlakes_fullkan_anova_array.sbatch

sbatch --account=jaabell0 \
  --export=ALL,PYTHON_BIN=$PWD/.venv/bin/python \
  --array=0-9 \
  scripts/greatlakes_readout_sensitivity_array.sbatch

sbatch --account=jaabell0 \
  --export=ALL,PYTHON_BIN=$PWD/.venv/bin/python \
  --array=0-2 \
  scripts/greatlakes_semisynthetic_array.sbatch

sbatch --account=jaabell0 \
  --export=ALL,PYTHON_BIN=$PWD/.venv/bin/python \
  --array=0-3 \
  scripts/greatlakes_anova_estimator_stability_array.sbatch
```

Scorer-indexed pair-claim rows on A40/spgpu nodes:

```bash
sbatch --account=jaabell0 \
  --export=ALL,PYTHON_BIN=$PWD/.venv/bin/python,SEED_BASE=0,SEED_COUNT=12,LABEL_SUFFIX=j0 \
  --array=0-9 \
  scripts/greatlakes_spgpu_pair_scorer_grammar_a40.sbatch

sbatch --account=engin1 \
  --export=ALL,PYTHON_BIN=$PWD/.venv/bin/python,SEED_BASE=200,SEED_COUNT=12,LABEL_SUFFIX=e0 \
  --array=0-9 \
  scripts/greatlakes_spgpu_pair_scorer_grammar_a40.sbatch
```

These rows compare EPIM, functional ANOVA, finite-difference, and hybrid pair
scorers on the same fitted KAN.  They are the reference run for scorer-indexed
pair claims: a task card can support a pair claim only relative to the declared
scorer.

For queued downstream pruning/symbolic smoke checks, submit after the current
arrays finish or use a dependency:

```bash
sbatch --account=jaabell0 \
  --dependency=afterany:<FULLKAN_JOBID>:<SEMISYN_JOBID> \
  --export=ALL,PYTHON_BIN=$PWD/.venv/bin/python \
  --array=0-3 \
  scripts/greatlakes_prune_symbolic_array.sbatch
```

Monitor jobs:

```bash
squeue -u $USER -o "%.18i %.9P %.30j %.8u %.2t %.10M %.6D %R"
sacct -j <JOBID> --format=JobID,JobName%24,State,ExitCode,Elapsed,MaxRSS,NodeList%20
tail -n 80 logs/greatlakes/<log-file>.err
```

After jobs finish, pack CSV summaries and Slurm logs for transfer:

```bash
bash scripts/greatlakes_pack_revision_results.sh
```

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

## Score Layers

The paper-facing schema separates continuous evidence from derived predicates:

- `prediction_mse`: continuous test-set MSE on standardized targets; the
  derived planted-control predicate is a fixed threshold when needed.
- `active_variable_f1`: active-variable recovery under the declared support
  budget; support size and ranks are retained when available.
- `endpoint_retention`: endpoint ranks and endpoint-vs-nuisance margins; the
  derived predicate is whether all true interaction endpoints are retained.
- `top1_pair`: pair rank and pair margin under the declared scorer; the derived
  predicate is whether the top-ranked candidate pair is a true pair.
- `exact_support`: selected support and retained support size; the derived
  predicate is equality with the declared active set in controlled tasks.

The `run_standard_audit_protocol.py` script can also consume a user-provided CSV
with columns:

```text
label,protocol,metric,successes,trials,notes
```

and emits a normalized CSV/Markdown summary with Wilson confidence intervals.

## Full-KAN All-Pairs ANOVA Check

This is the 60-seed, all-pairs result reported in the main paper. It ranks all
`100 * 99 / 2 = 4950` pairs directly on the full-dimensional KAN before support
selection or refit.

```bash
python scripts/run_full_kan_pair_anova_probe.py \
  --function core_interaction_c025 \
  --samples 1024 \
  --test-samples 2048 \
  --dimension 100 \
  --seeds 0-59 \
  --width-hidden 16 \
  --grid 5 \
  --k 3 \
  --steps 75 \
  --lamb 1e-3 \
  --pair-mode all \
  --anova-points 16 \
  --anova-background 16 \
  --batch-size 8192 \
  --device cuda \
  --out-dir results/revision/fullkan_anova_boundary/clean_w16_n1024_60seed
```

Expected summary:

- all fits: true pair rank-1 in `50/60` seeds;
- mean test MSE near `0.0048`;
- mean true-minus-max-false ANOVA margin near `0.062`.

For a faster code-path smoke test, reduce `--seeds 0-59` to `--seeds 0-2` and
write to a temporary output directory.

The current revision also runs a boundary full-KAN check for the grid-update
failure rows at `c=0.25,d=100,width=16`. This distinguishes readout-only
surfacing failure from broader model-reliance/fitting failures:

```bash
python scripts/run_full_kan_pair_anova_probe.py \
  --function core_interaction_c025 \
  --samples 512 \
  --test-samples 2048 \
  --dimension 100 \
  --seeds 0-59 \
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
  --batch-size 8192 \
  --device cuda \
  --out-dir results/revision/fullkan_anova_boundary/gridupdate_w16_n512_60seed
```

Expected summary for the completed grid-update boundary row:

- all fits: true pair rank-1 in `0/60` seeds at `n=512`;
- mean test MSE near `0.179`;
- mean true-pair rank among all 4950 pairs near `2146`;
- mean true-minus-max-false ANOVA margin near `-0.063`.

This row is therefore a fitting/reliance failure as well as a readout-surfacing
failure under the grid-update protocol.

### ANOVA Estimator Stability

The all-pairs full-KAN rows use a finite anchor/background ANOVA estimator. To
separate estimator noise from fitted-model variation, fix each trained KAN and
repeat the pair scorer with fresh anchor/background draws:

```bash
python scripts/run_full_kan_anova_estimator_stability.py \
  --function core_interaction_c025 \
  --samples 1024 \
  --test-samples 2048 \
  --dimension 100 \
  --seeds 0-19 \
  --mc-repeats 10 \
  --width-hidden 16 \
  --grid 5 \
  --k 3 \
  --steps 75 \
  --lamb 1e-3 \
  --anova-points 16 \
  --anova-background 16 \
  --batch-size 8192 \
  --pair-chunk-size 1000 \
  --device cuda \
  --out-dir results/revision/anova_estimator_stability/clean_w16_n1024_mc10
```

Outputs:

- `anova_estimator_stability_detail.csv`: one row per seed and Monte Carlo draw;
- `anova_estimator_stability_per_seed.csv`: within-model rank and margin spread;
- `anova_estimator_stability_summary.csv`: aggregate rank-1 and margin stability.

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
