from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch

from src.data import make_synthetic


def parse_pair(pair_text: str) -> Tuple[int, int]:
    parts = pair_text.replace("(", "").replace(")", "").split(",")
    if len(parts) != 2:
        raise ValueError(f"Invalid pair format: {pair_text!r}. Use format like '2,3'.")
    i, j = int(parts[0].strip()), int(parts[1].strip())
    if i == j:
        raise ValueError(f"Pair must contain two different variables: {pair_text!r}")
    return tuple(sorted((i, j)))


def mse_np(pred: np.ndarray, target: np.ndarray) -> float:
    pred = pred.reshape(-1, 1)
    target = target.reshape(-1, 1)
    return float(np.mean((pred - target) ** 2))


def to_numpy_prediction(y) -> np.ndarray:
    if isinstance(y, torch.Tensor):
        return y.detach().cpu().numpy()
    return np.asarray(y)


def predict_model(model, X_np: np.ndarray) -> np.ndarray:
    X = torch.tensor(X_np, dtype=torch.float32)
    with torch.no_grad():
        y = model(X)
    return to_numpy_prediction(y).reshape(-1, 1)


def train_pykan(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    width_hidden: int,
    grid: int,
    k: int,
    steps: int,
    lamb: float,
    seed: int,
):
    try:
        from kan import KAN
    except Exception as exc:
        raise ImportError(
            "Could not import pykan. Install with `pip install pykan` "
            "or `pip install git+https://github.com/KindXiaoming/pykan.git`."
        ) from exc

    torch.manual_seed(seed)
    np.random.seed(seed)

    d = X_train.shape[1]

    model = KAN(
        width=[d, width_hidden, 1],
        grid=grid,
        k=k,
        seed=seed,
    )

    dataset = {
        "train_input": torch.tensor(X_train, dtype=torch.float32),
        "train_label": torch.tensor(y_train, dtype=torch.float32),
        "test_input": torch.tensor(X_test, dtype=torch.float32),
        "test_label": torch.tensor(y_test, dtype=torch.float32),
    }

    if hasattr(model, "fit"):
        model.fit(
            dataset,
            opt="LBFGS",
            steps=steps,
            lamb=lamb,
        )
    else:
        raise RuntimeError(
            "This pykan version does not expose model.fit(...). "
            "Check your installed pykan API."
        )

    return model


def apply_single_intervention(
    X: np.ndarray,
    variable: int,
    method: str,
    rng: np.random.Generator,
) -> np.ndarray:
    X_mod = X.copy()

    if method == "zero":
        X_mod[:, variable] = 0.0
    elif method == "permute":
        perm = rng.permutation(X_mod.shape[0])
        X_mod[:, variable] = X_mod[perm, variable]
    else:
        raise ValueError(f"Unknown intervention method: {method}")

    return X_mod


def apply_pair_intervention(
    X: np.ndarray,
    pair: Tuple[int, int],
    method: str,
    rng: np.random.Generator,
) -> np.ndarray:
    X_mod = X.copy()
    i, j = pair

    if method == "zero":
        X_mod[:, i] = 0.0
        X_mod[:, j] = 0.0
    elif method == "permute":
        perm_i = rng.permutation(X_mod.shape[0])
        perm_j = rng.permutation(X_mod.shape[0])
        X_mod[:, i] = X_mod[perm_i, i]
        X_mod[:, j] = X_mod[perm_j, j]
    else:
        raise ValueError(f"Unknown intervention method: {method}")

    return X_mod


def run_interventions_for_seed(args, seed: int) -> List[Dict]:
    print(f"Training KAN for seed={seed}")

    data = make_synthetic(
        function_name=args.function,
        n_train=args.samples,
        n_test=args.test_samples,
        d=args.dimension,
        noise=args.noise,
        seed=seed,
        standardize_target=True,
    )

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]
    gt = data["ground_truth"]

    model = train_pykan(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        width_hidden=args.kan_width,
        grid=args.kan_grid,
        k=args.kan_k,
        steps=args.kan_steps,
        lamb=args.kan_lamb,
        seed=seed,
    )

    baseline_pred = predict_model(model, X_test)
    baseline_mse = mse_np(baseline_pred, y_test)

    print(f"seed={seed}, baseline test MSE={baseline_mse:.6g}")

    rows: List[Dict] = []
    rng = np.random.default_rng(seed + 12345)

    variables = [v for v in args.variables if 0 <= v < args.dimension]

    single_delta: Dict[Tuple[str, int], float] = {}

    for method in args.methods:
        for variable in variables:
            X_mod = apply_single_intervention(
                X=X_test,
                variable=variable,
                method=method,
                rng=rng,
            )
            pred_mod = predict_model(model, X_mod)
            ablated_mse = mse_np(pred_mod, y_test)
            delta = ablated_mse - baseline_mse
            single_delta[(method, variable)] = delta

            rows.append(
                {
                    "row_type": "single_variable",
                    "model": "KAN",
                    "function": args.function,
                    "seed": seed,
                    "samples": args.samples,
                    "test_samples": args.test_samples,
                    "dimension": args.dimension,
                    "noise": args.noise,
                    "kan_width": args.kan_width,
                    "kan_grid": args.kan_grid,
                    "kan_k": args.kan_k,
                    "kan_steps": args.kan_steps,
                    "kan_lamb": args.kan_lamb,
                    "intervention_method": method,
                    "variable": variable,
                    "pair": "",
                    "baseline_mse": baseline_mse,
                    "ablated_mse": ablated_mse,
                    "delta_mse": delta,
                    "synergy": np.nan,
                    "is_true_active": int(variable in gt.active_variables),
                    "true_variables": list(gt.active_variables),
                    "true_interactions": list(gt.interactions),
                    "formula": gt.formula,
                }
            )

    pairs = [pair for pair in args.pairs if pair[0] < args.dimension and pair[1] < args.dimension]

    for method in args.methods:
        for pair in pairs:
            i, j = pair
            X_mod = apply_pair_intervention(
                X=X_test,
                pair=pair,
                method=method,
                rng=rng,
            )
            pred_mod = predict_model(model, X_mod)
            ablated_mse = mse_np(pred_mod, y_test)
            delta = ablated_mse - baseline_mse

            delta_i = single_delta.get((method, i), np.nan)
            delta_j = single_delta.get((method, j), np.nan)
            synergy = delta - delta_i - delta_j

            true_interaction_set = {tuple(sorted(p)) for p in gt.interactions}

            rows.append(
                {
                    "row_type": "pair",
                    "model": "KAN",
                    "function": args.function,
                    "seed": seed,
                    "samples": args.samples,
                    "test_samples": args.test_samples,
                    "dimension": args.dimension,
                    "noise": args.noise,
                    "kan_width": args.kan_width,
                    "kan_grid": args.kan_grid,
                    "kan_k": args.kan_k,
                    "kan_steps": args.kan_steps,
                    "kan_lamb": args.kan_lamb,
                    "intervention_method": method,
                    "variable": "",
                    "pair": str(pair),
                    "baseline_mse": baseline_mse,
                    "ablated_mse": ablated_mse,
                    "delta_mse": delta,
                    "synergy": synergy,
                    "is_true_active": "",
                    "is_true_interaction": int(pair in true_interaction_set),
                    "true_variables": list(gt.active_variables),
                    "true_interactions": list(gt.interactions),
                    "formula": gt.formula,
                }
            )

    if args.save_model_dir is not None:
        save_dir = Path(args.save_model_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"kan_{args.function}_d{args.dimension}_n{args.samples}_noise{args.noise}_seed{seed}"

        try:
            model.saveckpt(str(save_path))
            print(f"Saved pykan checkpoint to {save_path}")
        except Exception as exc:
            print(f"[WARN] Could not save pykan checkpoint with saveckpt: {exc}")

    return rows


def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "function",
        "dimension",
        "noise",
        "row_type",
        "intervention_method",
        "variable",
        "pair",
    ]

    metric_cols = ["baseline_mse", "ablated_mse", "delta_mse", "synergy"]

    summary = (
        df.groupby(group_cols, dropna=False)[metric_cols]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )

    summary.columns = [
        "_".join(str(x) for x in col if x != "").rstrip("_")
        for col in summary.columns
    ]

    return summary


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--function", type=str, required=True)
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--test_samples", type=int, default=2048)
    parser.add_argument("--dimension", type=int, default=20)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])

    parser.add_argument("--kan_width", type=int, default=8)
    parser.add_argument("--kan_grid", type=int, default=5)
    parser.add_argument("--kan_k", type=int, default=3)
    parser.add_argument("--kan_steps", type=int, default=50)
    parser.add_argument("--kan_lamb", type=float, default=0.0)

    parser.add_argument(
        "--variables",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4, 5],
        help="Variables to intervene on.",
    )

    parser.add_argument(
        "--pairs",
        type=str,
        nargs="+",
        default=["2,3", "0,1", "0,4", "1,5"],
        help="Pairs to intervene on. Use format like '2,3'.",
    )

    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=["zero", "permute"],
        choices=["zero", "permute"],
    )

    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--summary_out", type=str, default=None)
    parser.add_argument("--save_model_dir", type=str, default=None)

    args = parser.parse_args()

    args.pairs = [parse_pair(p) for p in args.pairs]

    all_rows: List[Dict] = []

    for seed in args.seeds:
        rows = run_interventions_for_seed(args, seed)
        all_rows.extend(rows)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(all_rows)
    df.to_csv(out_path, index=False)
    print(f"Wrote intervention rows to {out_path}")

    if args.summary_out is not None:
        summary = summarize_results(df)
        summary_path = Path(args.summary_out)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(summary_path, index=False)
        print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()