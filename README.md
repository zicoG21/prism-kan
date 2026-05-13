# KAN-XAI Structure Recovery Benchmark

Minimal code for:
1. Synthetic ground-truth datasets.
2. KAN vs MLP training.
3. Variable recovery and explanation stability.
4. UCI Energy Efficiency smoke test.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If `pip install pykan` fails, use:

```bash
pip install git+https://github.com/KindXiaoming/pykan.git
```

## Run synthetic benchmark

```bash
python experiments/run_synthetic.py \
  --function core_interaction \
  --samples 1024 \
  --dimension 20 \
  --noise 0.0 \
  --seeds 0 1 2 3 4 \
  --kan_steps 50 \
  --out results/core_interaction.csv
```

## Run noise/sample sweep

```bash
python experiments/run_sweep.py --out results/sweep.csv
```

## Run UCI Energy Efficiency

```bash
python experiments/run_energy.py --out results/energy.csv
```

## Plot a summary

```bash
python experiments/plot_sweep.py --csv results/sweep.csv --out results/fig_variable_f1.png
```

Notes:
- KAN training can be slow on CPU. Start with `--kan_steps 20`.
- For the first research result, focus on synthetic functions first.
- For structure-recovery metrics, synthetic data is better than real data because the ground-truth active variables and interactions are known.

## Update: non-oracle explanation metrics

`experiments/run_synthetic.py` now outputs full-ranking explanation metrics in addition to oracle top-k F1:

- `variable_auroc`: AUROC of the variable importance ranking.
- `variable_auprc`: AUPRC of the variable importance ranking.
- `importance_scores`: JSON list of per-variable scores.
- `score_x0`, `score_x1`, ...: per-variable score columns.
- `active_score_mean`, `inactive_score_mean`, etc.: active-vs-inactive score diagnostics.

You can also save a long-form per-variable score table with:

```bash
python experiments/run_synthetic.py \
  --function core_interaction \
  --samples 1024 \
  --dimension 100 \
  --noise 0.0 \
  --seeds 0 1 2 3 4 \
  --kan_steps 50 \
  --skip_mlp \
  --out results/core_d100_nonoracle.csv \
  --scores_out results/core_d100_nonoracle_scores.csv
```

New synthetic function names:

- `highdim_sparse`: alias of `core_interaction`, useful for clearer high-dimensional sparse experiments.
- `correlated_proxy`: same true formula as `core_interaction`, but with proxy variables `x4 ≈ x0` and `x5 ≈ x1` that are correlated but not truly active.
