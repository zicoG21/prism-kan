from __future__ import annotations

import argparse
import itertools
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, roc_auc_score

from src.data import make_synthetic


def mse_np(pred: np.ndarray, target: np.ndarray) -> float:
    pred = pred.reshape(-1, 1)
    target = target.reshape(-1, 1)
    return float(np.mean((pred - target) ** 2))


def f1_from_sets(pred: set, true: set) -> Tuple[float, float, float]:
    if len(pred) == 0 and len(true) == 0:
        return 1.0, 1.0, 1.0
    if len(pred) == 0:
        return 0.0, 0.0, 0.0

    tp = len(pred & true)
    precision = tp / len(pred) if len(pred) > 0 else 0.0
    recall = tp / len(true) if len(true) > 0 else 0.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def safe_auroc_auprc(y_true: np.ndarray, scores: np.ndarray) -> Tuple[float, float]:
    try:
        auroc = float(roc_auc_score(y_true, scores))
    except ValueError:
        auroc = np.nan

    try:
        auprc = float(average_precision_score(y_true, scores))
    except ValueError:
        auprc = np.nan

    return auroc, auprc


def predict_model(model, X_np: np.ndarray) -> np.ndarray:
    X = torch.tensor(X_np, dtype=torch.float32)
    with torch.no_grad():
        y = model(X)

    if isinstance(y, torch.Tensor):
        return y.detach().cpu().numpy().reshape(-1, 1)
    return np.asarray(y).reshape(-1, 1)


def call_fit_with_optional_args(model, dataset, opt: str, steps: int, lamb: float, lr: float | None, update_grid_mode: str):
    kwargs = {
        "opt": opt,
        "steps": steps,
        "lamb": lamb,
    }

    if lr is not None:
        kwargs["lr"] = lr

    if update_grid_mode == "false":
        kwargs["update_grid"] = False
    elif update_grid_mode == "true":
        kwargs["update_grid"] = True
    elif update_grid_mode == "default":
        pass
    else:
        raise ValueError(f"Unknown update_grid_mode: {update_grid_mode}")

    try:
        return model.fit(dataset, **kwargs)
    except TypeError:
        # Some pykan versions do not accept lr or update_grid.
        kwargs.pop("lr", None)
        if "update_grid" in kwargs:
            try:
                return model.fit(dataset, **kwargs)
            except TypeError:
                kwargs.pop("update_grid", None)
                return model.fit(dataset, **kwargs)
        return model.fit(dataset, **kwargs)


def train_pykan(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    width_hidden: int,
    grid: int,
    k: int,
    opt: str,
    steps: int,
    lamb: float,
    lr: float | None,
    update_grid_mode: str,
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
    model = KAN(width=[d, width_hidden, 1], grid=grid, k=k, seed=seed)

    dataset = {
        "train_input": torch.tensor(X_train, dtype=torch.float32),
        "train_label": torch.tensor(y_train, dtype=torch.float32),
        "test_input": torch.tensor(X_test, dtype=torch.float32),
        "test_label": torch.tensor(y_test, dtype=torch.float32),
    }

    if not hasattr(model, "fit"):
        raise RuntimeError("This pykan version does not expose model.fit(...).")

    call_fit_with_optional_args(
        model=model,
        dataset=dataset,
        opt=opt,
        steps=steps,
        lamb=lamb,
        lr=lr,
        update_grid_mode=update_grid_mode,
    )

    return model


def gradient_importance(model, X_np: np.ndarray) -> np.ndarray:
    X = torch.tensor(X_np, dtype=torch.float32, requires_grad=True)
    y = model(X).sum()
    grad = torch.autograd.grad(y, X, create_graph=False)[0]
    return grad.abs().mean(dim=0).detach().cpu().numpy()


def permutation_importance(model, X_np: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 123456)
    baseline = predict_model(model, X_np).reshape(-1)
    scores = []

    for j in range(X_np.shape[1]):
        X_perm = X_np.copy()
        perm = rng.permutation(X_perm.shape[0])
        X_perm[:, j] = X_perm[perm, j]
        pred_perm = predict_model(model, X_perm).reshape(-1)
        scores.append(float(np.mean((pred_perm - baseline) ** 2)))

    return np.asarray(scores, dtype=float)


def hessian_interaction_scores(model, X_np: np.ndarray, hessian_points: int) -> Dict[Tuple[int, int], float]:
    d = X_np.shape[1]
    n = min(hessian_points, X_np.shape[0])
    X_sub = torch.tensor(X_np[:n], dtype=torch.float32)

    score_mat = torch.zeros((d, d), dtype=torch.float32)

    def scalar_func(z: torch.Tensor) -> torch.Tensor:
        return model(z.unsqueeze(0)).sum()

    for idx in range(n):
        z = X_sub[idx].clone().detach().requires_grad_(True)
        H = torch.autograd.functional.hessian(scalar_func, z)
        score_mat += H.abs().detach()

    score_mat = score_mat / max(n, 1)

    pair_scores = {}
    for i, j in itertools.combinations(range(d), 2):
        pair_scores[(i, j)] = float(score_mat[i, j].item())

    return pair_scores


def evaluate_variable_recovery(scores: np.ndarray, true_vars: Sequence[int]) -> Dict:
    true_set = set(int(v) for v in true_vars)
    k = len(true_set)
    selected = set(int(i) for i in np.argsort(-scores)[:k])

    precision, recall, f1 = f1_from_sets(selected, true_set)

    y_true = np.array([1 if i in true_set else 0 for i in range(len(scores))], dtype=int)
    auroc, auprc = safe_auroc_auprc(y_true, scores)

    active_scores = scores[y_true == 1]
    inactive_scores = scores[y_true == 0]

    return {
        "selected_variables": sorted(selected),
        "variable_precision": precision,
        "variable_recall": recall,
        "variable_f1": f1,
        "variable_auroc": auroc,
        "variable_auprc": auprc,
        "active_score_mean": float(active_scores.mean()) if len(active_scores) else np.nan,
        "inactive_score_mean": float(inactive_scores.mean()) if len(inactive_scores) else np.nan,
        "active_score_min": float(active_scores.min()) if len(active_scores) else np.nan,
        "inactive_score_max": float(inactive_scores.max()) if len(inactive_scores) else np.nan,
    }


def evaluate_interaction_recovery(pair_scores: Dict[Tuple[int, int], float], true_interactions: Sequence[Tuple[int, int]]) -> Dict:
    true_set = {tuple(sorted((int(i), int(j)))) for i, j in true_interactions}
    k = max(len(true_set), 1)

    ranked = sorted(pair_scores.items(), key=lambda kv: kv[1], reverse=True)
    selected = {pair for pair, _ in ranked[:k]}

    precision, recall, f1 = f1_from_sets(selected, true_set)

    return {
        "selected_interactions": sorted(selected),
        "interaction_precision": precision,
        "interaction_recall": recall,
        "interaction_f1": f1,
    }


def parse_variant_name(name: str) -> Dict:
    """Parse built-in variant names.

    Supported examples:
      lbfgs50
      lbfgs100
      lbfgs200
      lbfgs_nogrid50
      lbfgs_nogrid100
      lbfgs_nogrid200
      lbfgs_lamb1e-3_100
      adam200
      adam500
    """
    if name == "default":
        return {
            "variant": name,
            "opt": "LBFGS",
            "steps": 50,
            "lamb": 0.0,
            "lr": None,
            "update_grid_mode": "default",
        }

    if name.startswith("lbfgs_nogrid"):
        steps = int(name.replace("lbfgs_nogrid", ""))
        return {
            "variant": name,
            "opt": "LBFGS",
            "steps": steps,
            "lamb": 0.0,
            "lr": None,
            "update_grid_mode": "false",
        }

    if name.startswith("lbfgs_lamb"):
        # Format: lbfgs_lamb1e-3_100
        tail = name.replace("lbfgs_lamb", "")
        lamb_text, steps_text = tail.split("_")
        return {
            "variant": name,
            "opt": "LBFGS",
            "steps": int(steps_text),
            "lamb": float(lamb_text),
            "lr": None,
            "update_grid_mode": "default",
        }

    if name.startswith("lbfgs"):
        steps = int(name.replace("lbfgs", ""))
        return {
            "variant": name,
            "opt": "LBFGS",
            "steps": steps,
            "lamb": 0.0,
            "lr": None,
            "update_grid_mode": "default",
        }

    if name.startswith("adamw"):
        steps = int(name.replace("adamw", ""))
        return {
            "variant": name,
            "opt": "Adam",
            "steps": steps,
            "lamb": 0.0,
            "lr": 1e-3,
            "update_grid_mode": "false",
        }

    if name.startswith("adam"):
        steps = int(name.replace("adam", ""))
        return {
            "variant": name,
            "opt": "Adam",
            "steps": steps,
            "lamb": 0.0,
            "lr": 1e-3,
            "update_grid_mode": "false",
        }

    raise ValueError(f"Unknown variant name: {name}")


def run_one(args, function_name: str, seed: int, variant_cfg: Dict) -> List[Dict]:
    data = make_synthetic(
        function_name=function_name,
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

    base_row = {
        "model": "KAN_ROBUSTNESS",
        "function": function_name,
        "seed": seed,
        "samples": args.samples,
        "test_samples": args.test_samples,
        "dimension": args.dimension,
        "noise": args.noise,
        "variant": variant_cfg["variant"],
        "opt": variant_cfg["opt"],
        "steps": variant_cfg["steps"],
        "lamb": variant_cfg["lamb"],
        "lr": variant_cfg["lr"],
        "update_grid_mode": variant_cfg["update_grid_mode"],
        "kan_width": args.kan_width,
        "kan_grid": args.kan_grid,
        "kan_k": args.kan_k,
        "true_variables": list(gt.active_variables),
        "true_interactions": list(gt.interactions),
        "formula": gt.formula,
    }

    try:
        model = train_pykan(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            width_hidden=args.kan_width,
            grid=args.kan_grid,
            k=args.kan_k,
            opt=variant_cfg["opt"],
            steps=variant_cfg["steps"],
            lamb=variant_cfg["lamb"],
            lr=variant_cfg["lr"],
            update_grid_mode=variant_cfg["update_grid_mode"],
            seed=seed,
        )

        train_pred = predict_model(model, X_train)
        test_pred = predict_model(model, X_test)

        train_mse = mse_np(train_pred, y_train)
        test_mse = mse_np(test_pred, y_test)

        rows = []

        explain_methods = []
        if "grad" in args.explain_methods:
            explain_methods.append(("grad", gradient_importance(model, X_test)))
        if "perm" in args.explain_methods:
            explain_methods.append(("perm", permutation_importance(model, X_test, seed)))

        interaction_eval = {}
        if args.compute_interactions:
            pair_scores = hessian_interaction_scores(model, X_test, args.hessian_points)
            interaction_eval = evaluate_interaction_recovery(pair_scores, gt.interactions)

        for method, scores in explain_methods:
            row = dict(base_row)
            row.update({
                "status": "ok",
                "error": "",
                "explain_method": method,
                "train_mse": train_mse,
                "test_mse": test_mse,
                "importance_scores": scores.tolist(),
            })
            row.update(evaluate_variable_recovery(scores, gt.active_variables))
            if args.compute_interactions:
                row.update(interaction_eval)
            rows.append(row)

        return rows

    except Exception as exc:
        row = dict(base_row)
        row.update({
            "status": "failed",
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "explain_method": "failed",
            "train_mse": np.nan,
            "test_mse": np.nan,
            "selected_variables": [],
            "variable_precision": np.nan,
            "variable_recall": np.nan,
            "variable_f1": np.nan,
            "variable_auroc": np.nan,
            "variable_auprc": np.nan,
            "selected_interactions": [],
            "interaction_precision": np.nan,
            "interaction_recall": np.nan,
            "interaction_f1": np.nan,
        })
        print(f"[WARN] failed function={function_name}, seed={seed}, variant={variant_cfg['variant']}: {exc}")
        return [row]


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    ok = df[df["status"] == "ok"].copy()
    if ok.empty:
        return pd.DataFrame()

    group_cols = ["function", "dimension", "noise", "variant", "opt", "steps", "lamb", "update_grid_mode", "explain_method"]

    agg = {
        "train_mse": ["mean", "std", "min", "max"],
        "test_mse": ["mean", "std", "min", "max"],
        "variable_f1": ["mean", "std"],
        "variable_auroc": ["mean", "std"],
        "variable_auprc": ["mean", "std"],
    }

    if "interaction_f1" in ok.columns:
        agg["interaction_f1"] = ["mean", "std"]

    summary = ok.groupby(group_cols, dropna=False).agg(agg).reset_index()
    summary.columns = ["_".join(str(x) for x in col if x != "").rstrip("_") for col in summary.columns]

    counts = df.groupby(["function", "variant", "explain_method"], dropna=False).agg(
        num_runs=("status", "size"),
        num_failed=("status", lambda s: int((s != "ok").sum())),
    ).reset_index()

    summary = summary.merge(counts, on=["function", "variant", "explain_method"], how="left")
    return summary


def plot_summary(summary: pd.DataFrame, out_path: Path):
    if summary.empty:
        return

    plot_df = summary.copy()
    if "grad" in set(plot_df["explain_method"].astype(str)):
        plot_df = plot_df[plot_df["explain_method"].astype(str) == "grad"].copy()

    plot_df = plot_df.sort_values(["function", "variant"])
    plot_df["label"] = plot_df.apply(lambda r: f"{r['function']}\n{r['variant']}", axis=1)

    x = np.arange(len(plot_df))
    width = 0.28

    plt.figure(figsize=(max(10, len(plot_df) * 0.7), 5.5))
    plt.bar(x - width, plot_df["variable_f1_mean"], width=width, label="Variable F1")
    if "interaction_f1_mean" in plot_df.columns:
        plt.bar(x, plot_df["interaction_f1_mean"], width=width, label="Interaction F1")
    mse_scaled = plot_df["test_mse_mean"].to_numpy(dtype=float)
    if np.nanmax(mse_scaled) > 0:
        mse_scaled = mse_scaled / np.nanmax(mse_scaled)
        plt.bar(x + width, mse_scaled, width=width, label="Test MSE / max")
    plt.ylim(0, 1.08)
    plt.ylabel("F1 or normalized MSE")
    plt.xticks(x, plot_df["label"], rotation=55, ha="right")
    plt.title("KAN optimization robustness ablation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=250)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--functions", type=str, nargs="+", default=["core_interaction", "core_interaction_c5"])
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--test_samples", type=int, default=2048)
    parser.add_argument("--dimension", type=int, default=100)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])

    parser.add_argument(
        "--variants",
        type=str,
        nargs="+",
        default=[
            "default",
            "lbfgs100",
            "lbfgs200",
            "lbfgs_nogrid50",
            "lbfgs_nogrid100",
            "lbfgs_nogrid200",
        ],
        help=(
            "Built-in variants: default, lbfgs50/100/200, "
            "lbfgs_nogrid50/100/200, lbfgs_lamb1e-3_100, adam200, adam500."
        ),
    )

    parser.add_argument("--kan_width", type=int, default=8)
    parser.add_argument("--kan_grid", type=int, default=5)
    parser.add_argument("--kan_k", type=int, default=3)

    parser.add_argument("--explain_methods", type=str, nargs="+", default=["grad", "perm"])
    parser.add_argument("--compute_interactions", action="store_true")
    parser.add_argument("--hessian_points", type=int, default=64)

    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--summary_out", type=str, default=None)
    parser.add_argument("--plot_out", type=str, default=None)

    args = parser.parse_args()

    variant_cfgs = [parse_variant_name(v) for v in args.variants]

    all_rows: List[Dict] = []

    for function_name in args.functions:
        for seed in args.seeds:
            for cfg in variant_cfgs:
                print(f"Running function={function_name}, seed={seed}, variant={cfg['variant']}")
                rows = run_one(args, function_name=function_name, seed=seed, variant_cfg=cfg)
                all_rows.extend(rows)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(all_rows)
    df.to_csv(out_path, index=False)
    print(f"Wrote rows to {out_path}")

    summary = summarize(df)

    if args.summary_out is not None:
        summary_path = Path(args.summary_out)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(summary_path, index=False)
        print(f"Wrote summary to {summary_path}")

    if args.plot_out is not None:
        plot_path = Path(args.plot_out)
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plot_summary(summary, plot_path)
        print(f"Wrote plot to {plot_path}")


if __name__ == "__main__":
    main()
