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
from sklearn.ensemble import RandomForestRegressor
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
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(true) if true else 0.0
    f1 = 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)
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


def call_fit(model, dataset, opt: str, steps: int, lamb: float, lr: float | None, update_grid_mode: str):
    kwargs = {"opt": opt, "steps": steps, "lamb": lamb}
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
        # pykan versions differ in whether they accept lr/update_grid.
        kwargs.pop("lr", None)
        try:
            return model.fit(dataset, **kwargs)
        except TypeError:
            kwargs.pop("update_grid", None)
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

    call_fit(
        model=model,
        dataset=dataset,
        opt=opt,
        steps=steps,
        lamb=lamb,
        lr=lr,
        update_grid_mode=update_grid_mode,
    )
    return model


def get_interaction_endpoints(true_interactions: Sequence[Tuple[int, int]]) -> Tuple[int, ...]:
    s = set()
    for i, j in true_interactions:
        s.add(int(i))
        s.add(int(j))
    return tuple(sorted(s))


def rf_scores(X: np.ndarray, y: np.ndarray, seed: int, n_estimators: int = 300) -> np.ndarray:
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=seed,
        n_jobs=-1,
        min_samples_leaf=2,
    )
    rf.fit(X, y.reshape(-1))
    return np.asarray(rf.feature_importances_, dtype=float)


def fill_support(
    must_include: Sequence[int],
    candidate_pool: Sequence[int],
    top_m: int,
    rng: np.random.Generator,
    scores: np.ndarray | None = None,
) -> np.ndarray:
    selected = []
    seen = set()

    for v in must_include:
        v = int(v)
        if v not in seen:
            selected.append(v)
            seen.add(v)

    remaining = [int(v) for v in candidate_pool if int(v) not in seen]
    need = max(0, top_m - len(selected))

    if need > 0:
        if scores is None:
            rng.shuffle(remaining)
            selected.extend(remaining[:need])
        else:
            remaining = sorted(remaining, key=lambda v: float(scores[v]), reverse=True)
            selected.extend(remaining[:need])

    return np.array(sorted(selected[:top_m]), dtype=int)


def select_features(
    mode: str,
    X: np.ndarray,
    y: np.ndarray,
    top_m: int,
    true_vars: Sequence[int],
    true_interactions: Sequence[Tuple[int, int]],
    seed: int,
    rf_trees: int,
) -> Tuple[np.ndarray, np.ndarray, str]:
    d = X.shape[1]
    rng = np.random.default_rng(seed + 773)
    all_vars = list(range(d))
    true_vars = tuple(int(v) for v in true_vars)
    endpoints = tuple(get_interaction_endpoints(true_interactions))
    endpoints_set = set(endpoints)

    scores = np.zeros(d, dtype=float)

    if mode == "raw":
        return np.arange(d, dtype=int), scores, "none"

    if mode == "random":
        selected = np.array(sorted(rng.choice(d, size=min(top_m, d), replace=False).astype(int)), dtype=int)
        return selected, scores, "random"

    if mode == "oracle_support":
        selected = fill_support(true_vars, all_vars, top_m, rng, scores=None)
        return selected, scores, "forced_true_support_random_fill"

    if mode == "exclude_interaction":
        pool = [v for v in all_vars if v not in endpoints_set]
        selected = np.array(sorted(rng.choice(pool, size=min(top_m, len(pool)), replace=False).astype(int)), dtype=int)
        return selected, scores, "random_excluding_interaction_endpoints"

    if mode == "rf":
        scores = rf_scores(X, y, seed=seed, n_estimators=rf_trees)
        selected = np.array(sorted(np.argsort(-scores)[:top_m].astype(int)), dtype=int)
        return selected, scores, "rf"

    if mode == "rf_exclude_interaction":
        scores = rf_scores(X, y, seed=seed, n_estimators=rf_trees)
        pool = [v for v in all_vars if v not in endpoints_set]
        selected = np.array(sorted(sorted(pool, key=lambda v: float(scores[v]), reverse=True)[:top_m]), dtype=int)
        return selected, scores, "rf_excluding_interaction_endpoints"

    raise ValueError(f"Unknown screen mode: {mode}")


def support_stats(selected_features: np.ndarray, true_vars: Sequence[int], true_interactions: Sequence[Tuple[int, int]]) -> Dict:
    selected = set(int(v) for v in selected_features)
    true_var_set = set(int(v) for v in true_vars)
    endpoints = set(get_interaction_endpoints(true_interactions))

    return {
        "screen_contains_all_true_vars": int(true_var_set.issubset(selected)),
        "screen_true_var_recall": len(true_var_set & selected) / len(true_var_set) if true_var_set else np.nan,
        "screen_contains_all_interaction_endpoints": int(endpoints.issubset(selected)) if endpoints else np.nan,
        "screen_interaction_endpoint_recall": len(endpoints & selected) / len(endpoints) if endpoints else np.nan,
        "screen_contains_true_interactions": int(
            all(int(i) in selected and int(j) in selected for i, j in true_interactions)
        ) if true_interactions else np.nan,
    }


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

    return np.array(scores, dtype=float)


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


def local_to_full_scores(local_scores: np.ndarray, selected_features: np.ndarray, d_full: int) -> np.ndarray:
    full_scores = np.zeros(d_full, dtype=float)
    for local_idx, original_idx in enumerate(selected_features):
        full_scores[int(original_idx)] = float(local_scores[local_idx])
    return full_scores


def local_to_full_pair_scores(
    local_pair_scores: Dict[Tuple[int, int], float],
    selected_features: np.ndarray,
    d_full: int,
) -> Dict[Tuple[int, int], float]:
    full_pair_scores = {}
    selected_features = np.asarray(selected_features, dtype=int)

    for (i_local, j_local), score in local_pair_scores.items():
        i = int(selected_features[i_local])
        j = int(selected_features[j_local])
        full_pair_scores[tuple(sorted((i, j)))] = float(score)

    # Add zero scores for pairs not present in the screened model.
    for i, j in itertools.combinations(range(d_full), 2):
        full_pair_scores.setdefault((i, j), 0.0)

    return full_pair_scores


def evaluate_variable_recovery(full_scores: np.ndarray, true_vars: Sequence[int]) -> Dict:
    true_set = set(int(v) for v in true_vars)
    k = len(true_set)
    selected = set(int(i) for i in np.argsort(-full_scores)[:k])

    precision, recall, f1 = f1_from_sets(selected, true_set)
    y_true = np.array([1 if i in true_set else 0 for i in range(len(full_scores))], dtype=int)
    auroc, auprc = safe_auroc_auprc(y_true, full_scores)

    active_scores = full_scores[y_true == 1]
    inactive_scores = full_scores[y_true == 0]

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


def evaluate_interaction_recovery(full_pair_scores: Dict[Tuple[int, int], float], true_interactions: Sequence[Tuple[int, int]]) -> Dict:
    true_set = {tuple(sorted((int(i), int(j)))) for i, j in true_interactions}
    k = max(len(true_set), 1)

    ranked = sorted(full_pair_scores.items(), key=lambda kv: kv[1], reverse=True)
    selected = {pair for pair, _ in ranked[:k]}

    precision, recall, f1 = f1_from_sets(selected, true_set)

    return {
        "selected_interactions": sorted(selected),
        "interaction_precision": precision,
        "interaction_recall": recall,
        "interaction_f1": f1,
    }


def append_rows(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    header = not path.exists()
    df.to_csv(path, mode="a", header=header, index=False)


def run_one(args, function_name: str, n_train: int, seed: int, mode: str, top_m: int) -> List[Dict]:
    data = make_synthetic(
        function_name=function_name,
        n_train=n_train,
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

    true_vars = tuple(int(v) for v in gt.active_variables)
    true_interactions = tuple(tuple(sorted((int(i), int(j)))) for i, j in gt.interactions)

    selected_features, screen_scores, score_desc = select_features(
        mode=mode,
        X=X_train,
        y=y_train,
        top_m=top_m,
        true_vars=true_vars,
        true_interactions=true_interactions,
        seed=seed,
        rf_trees=args.rf_trees,
    )

    X_train_s = X_train[:, selected_features]
    X_test_s = X_test[:, selected_features]

    base = {
        "model": "KAN_SAMPLE_SIZE_ABLATION",
        "function": function_name,
        "seed": seed,
        "samples": n_train,
        "test_samples": args.test_samples,
        "dimension": args.dimension,
        "noise": args.noise,
        "screen_mode": mode,
        "screen_score_type": score_desc,
        "top_m": top_m if mode != "raw" else args.dimension,
        "selected_screen_features": selected_features.tolist(),
        "screen_scores": screen_scores.tolist(),
        "kan_width": args.kan_width,
        "kan_grid": args.kan_grid,
        "kan_k": args.kan_k,
        "kan_steps": args.kan_steps,
        "kan_lamb": args.kan_lamb,
        "opt": args.opt,
        "lr": args.lr,
        "update_grid_mode": args.update_grid_mode,
        "true_variables": list(true_vars),
        "true_interactions": list(true_interactions),
        "formula": gt.formula,
    }
    base.update(support_stats(selected_features, true_vars, true_interactions))

    try:
        model = train_pykan(
            X_train=X_train_s,
            y_train=y_train,
            X_test=X_test_s,
            y_test=y_test,
            width_hidden=args.kan_width,
            grid=args.kan_grid,
            k=args.kan_k,
            opt=args.opt,
            steps=args.kan_steps,
            lamb=args.kan_lamb,
            lr=args.lr,
            update_grid_mode=args.update_grid_mode,
            seed=seed,
        )

        train_mse = mse_np(predict_model(model, X_train_s), y_train)
        test_mse = mse_np(predict_model(model, X_test_s), y_test)

        explain_methods = []
        if "grad" in args.explain_methods:
            explain_methods.append(("grad", gradient_importance(model, X_test_s)))
        if "perm" in args.explain_methods:
            explain_methods.append(("perm", permutation_importance(model, X_test_s, seed)))

        interaction_eval = {}
        if args.compute_interactions:
            local_pair_scores = hessian_interaction_scores(model, X_test_s, args.hessian_points)
            full_pair_scores = local_to_full_pair_scores(local_pair_scores, selected_features, args.dimension)
            interaction_eval = evaluate_interaction_recovery(full_pair_scores, true_interactions)

        rows: List[Dict] = []
        for explain_method, local_scores in explain_methods:
            full_scores = local_to_full_scores(local_scores, selected_features, args.dimension)
            row = dict(base)
            row.update({
                "status": "ok",
                "error": "",
                "explain_method": explain_method,
                "train_mse": train_mse,
                "test_mse": test_mse,
                "importance_scores": full_scores.tolist(),
            })
            row.update(evaluate_variable_recovery(full_scores, true_vars))
            if args.compute_interactions:
                row.update(interaction_eval)
            rows.append(row)

        return rows

    except Exception as exc:
        row = dict(base)
        row.update({
            "status": "failed",
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "explain_method": "failed",
            "train_mse": np.nan,
            "test_mse": np.nan,
            "importance_scores": [],
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
        print(f"[WARN] failed function={function_name}, n={n_train}, seed={seed}, mode={mode}, M={top_m}: {exc}")
        return [row]


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    ok = df[df["status"].astype(str) == "ok"].copy()
    if ok.empty:
        return pd.DataFrame()

    group_cols = ["function", "dimension", "noise", "samples", "screen_mode", "top_m", "explain_method"]

    metric_cols = [
        "train_mse",
        "test_mse",
        "screen_contains_all_true_vars",
        "screen_true_var_recall",
        "screen_contains_true_interactions",
        "variable_f1",
        "variable_auroc",
        "variable_auprc",
        "interaction_f1",
    ]

    # Important: because some columns contain Python-list strings, pandas may infer
    # nearby metric columns as object dtype. Force only true metric columns to numeric
    # before groupby aggregation, otherwise pandas may concatenate strings like
    # "1e-128e-13..." and then fail when computing mean.
    for col in metric_cols:
        if col in ok.columns:
            ok[col] = pd.to_numeric(ok[col], errors="coerce")

    agg = {}
    for col in [
        "train_mse",
        "test_mse",
        "screen_contains_all_true_vars",
        "screen_true_var_recall",
        "screen_contains_true_interactions",
        "variable_f1",
        "variable_auroc",
        "variable_auprc",
    ]:
        if col in ok.columns:
            if col in {"screen_contains_all_true_vars", "screen_true_var_recall", "screen_contains_true_interactions"}:
                agg[col] = ["mean"]
            else:
                agg[col] = ["mean", "std"]

    if "interaction_f1" in ok.columns:
        agg["interaction_f1"] = ["mean", "std"]

    summary = ok.groupby(group_cols, dropna=False).agg(agg).reset_index()
    summary.columns = ["_".join(str(x) for x in col if x != "").rstrip("_") for col in summary.columns]

    count_cols = ["function", "samples", "screen_mode", "top_m", "explain_method"]
    counts = df.copy()
    for c in ["samples", "top_m"]:
        if c in counts.columns:
            counts[c] = pd.to_numeric(counts[c], errors="coerce")
    counts = counts.groupby(count_cols, dropna=False).agg(
        num_runs=("status", "size"),
        num_failed=("status", lambda s: int((s.astype(str) != "ok").sum())),
    ).reset_index()

    summary = summary.merge(counts, on=count_cols, how="left")
    return summary



def plot_metric(summary: pd.DataFrame, metric: str, ylabel: str, title: str, out_path: Path):
    if summary.empty or metric not in summary.columns:
        return

    plot_df = summary.copy()
    if "grad" in set(plot_df["explain_method"].astype(str)):
        plot_df = plot_df[plot_df["explain_method"].astype(str) == "grad"].copy()

    functions = plot_df["function"].drop_duplicates().tolist()
    fig, axes = plt.subplots(1, len(functions), figsize=(6 * len(functions), 4.8), sharey=True)
    if len(functions) == 1:
        axes = [axes]

    for ax, function_name in zip(axes, functions):
        sub_f = plot_df[plot_df["function"] == function_name].copy()
        for mode in sub_f["screen_mode"].drop_duplicates().tolist():
            sub_m = sub_f[sub_f["screen_mode"] == mode].sort_values("samples")
            if sub_m.empty:
                continue
            ax.plot(sub_m["samples"], sub_m[metric], marker="o", label=mode)
        ax.set_title(function_name)
        ax.set_xlabel("n_train")
        ax.set_xscale("log", base=2)
        ax.set_ylim(0, 1.08 if "f1" in metric else None)
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel(ylabel)
    axes[-1].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=250)
    plt.close(fig)


def plot_summary(summary: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_metric(
        summary,
        metric="interaction_f1_mean",
        ylabel="Interaction F1",
        title="Interaction recovery across sample sizes",
        out_path=out_dir / "sample_size_interaction_f1.png",
    )
    plot_metric(
        summary,
        metric="variable_f1_mean",
        ylabel="Variable F1",
        title="Variable recovery across sample sizes",
        out_path=out_dir / "sample_size_variable_f1.png",
    )

    plot_df = summary.copy()
    if "grad" in set(plot_df["explain_method"].astype(str)):
        plot_df = plot_df[plot_df["explain_method"].astype(str) == "grad"].copy()

    functions = plot_df["function"].drop_duplicates().tolist()
    fig, axes = plt.subplots(1, len(functions), figsize=(6 * len(functions), 4.8), sharey=False)
    if len(functions) == 1:
        axes = [axes]

    for ax, function_name in zip(axes, functions):
        sub_f = plot_df[plot_df["function"] == function_name].copy()
        for mode in sub_f["screen_mode"].drop_duplicates().tolist():
            sub_m = sub_f[sub_f["screen_mode"] == mode].sort_values("samples")
            if sub_m.empty:
                continue
            ax.plot(sub_m["samples"], sub_m["test_mse_mean"], marker="o", label=mode)
        ax.set_title(function_name)
        ax.set_xlabel("n_train")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("Test MSE")
    axes[-1].legend(fontsize=8)
    fig.suptitle("Test MSE across sample sizes")
    fig.tight_layout()
    fig.savefig(out_dir / "sample_size_test_mse.png", dpi=250)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--functions", type=str, nargs="+", default=["core_interaction"])
    parser.add_argument("--sample_sizes", type=int, nargs="+", default=[256, 512, 1024, 2048, 4096, 8192])
    parser.add_argument("--test_samples", type=int, default=2048)
    parser.add_argument("--dimension", type=int, default=100)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--screen_modes", type=str, nargs="+", default=["raw", "random", "oracle_support", "rf", "exclude_interaction"])
    parser.add_argument("--top_m", type=int, default=20)
    parser.add_argument("--rf_trees", type=int, default=300)

    parser.add_argument("--kan_width", type=int, default=8)
    parser.add_argument("--kan_grid", type=int, default=5)
    parser.add_argument("--kan_k", type=int, default=3)
    parser.add_argument("--kan_steps", type=int, default=50)
    parser.add_argument("--kan_lamb", type=float, default=0.0)
    parser.add_argument("--opt", type=str, default="LBFGS")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--update_grid_mode", type=str, default="default", choices=["default", "true", "false"])

    parser.add_argument("--explain_methods", type=str, nargs="+", default=["grad"])
    parser.add_argument("--compute_interactions", action="store_true")
    parser.add_argument("--hessian_points", type=int, default=32)

    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--summary_out", type=str, default=None)
    parser.add_argument("--fig_dir", type=str, default=None)
    parser.add_argument("--resume", action="store_true", help="Skip exact configurations already present in --out.")

    args = parser.parse_args()

    out_path = Path(args.out)
    existing_keys = set()

    if args.resume and out_path.exists():
        try:
            old = pd.read_csv(out_path)
            for r in old[["function", "samples", "seed", "screen_mode", "top_m"]].drop_duplicates().itertuples(index=False):
                existing_keys.add((r.function, int(r.samples), int(r.seed), r.screen_mode, int(r.top_m)))
            print(f"Resume mode: found {len(existing_keys)} completed configurations.")
        except Exception as exc:
            print(f"[WARN] Could not read existing output for resume: {exc}")

    for function_name in args.functions:
        for n_train in args.sample_sizes:
            for seed in args.seeds:
                for mode in args.screen_modes:
                    effective_top_m = args.dimension if mode == "raw" else args.top_m
                    key = (function_name, int(n_train), int(seed), mode, int(effective_top_m))
                    if key in existing_keys:
                        print(f"Skipping completed {key}")
                        continue

                    print(f"Running function={function_name}, n_train={n_train}, seed={seed}, mode={mode}, M={effective_top_m}")
                    rows = run_one(
                        args=args,
                        function_name=function_name,
                        n_train=n_train,
                        seed=seed,
                        mode=mode,
                        top_m=effective_top_m,
                    )
                    append_rows(out_path, rows)

    df = pd.read_csv(out_path)
    summary = summarize(df)

    if args.summary_out is not None:
        summary_path = Path(args.summary_out)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(summary_path, index=False)
        print(f"Wrote summary to {summary_path}")

    if args.fig_dir is not None:
        fig_dir = Path(args.fig_dir)
        plot_summary(summary, fig_dir)
        print(f"Wrote figures to {fig_dir}")

    print(f"Wrote rows to {out_path}")


if __name__ == "__main__":
    main()
