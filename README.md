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
