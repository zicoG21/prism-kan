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


# -----------------------------
# Basic metrics
# -----------------------------

def mse_np(pred: np.ndarray, target: np.ndarray) -> float:
    pred = pred.reshape(-1, 1)
    target = target.reshape(-1, 1)
    return float(np.mean((pred - target) ** 2))


def f1_from_sets(pred: set, true: set) -> Tuple[float, float, float]:
    if len(true) == 0:
        return np.nan, np.nan, np.nan
    if len(pred) == 0:
        return 0.0, 0.0, 0.0
    tp = len(pred & true)
    precision = tp / len(pred)
    recall = tp / len(true)
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


# -----------------------------
# Training / prediction
# -----------------------------

def predict_model(model, X_np: np.ndarray) -> np.ndarray:
    X = torch.tensor(X_np, dtype=torch.float32)
    with torch.no_grad():
        y = model(X)
    if isinstance(y, torch.Tensor):
        return y.detach().cpu().numpy().reshape(-1, 1)
    return np.asarray(y).reshape(-1, 1)


def fit_kan(model, dataset, opt: str, steps: int, lamb: float, lr: float | None, update_grid_mode: str):
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
        raise ValueError(f"Unknown update_grid_mode={update_grid_mode}")

    # pykan versions differ in accepted fit kwargs.
    try:
        return model.fit(dataset, **kwargs)
    except TypeError:
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
            "Could not import pykan. Install pykan or run in the same env used for previous KAN experiments."
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

    fit_kan(
        model=model,
        dataset=dataset,
        opt=opt,
        steps=steps,
        lamb=lamb,
        lr=lr,
        update_grid_mode=update_grid_mode,
    )
    return model


# -----------------------------
# Screening
# -----------------------------

def interaction_endpoints(true_interactions: Sequence[Tuple[int, int]]) -> Tuple[int, ...]:
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


def fill_with_support(
    must_include: Sequence[int],
    candidate_pool: Sequence[int],
    top_m: int,
    rng: np.random.Generator,
    scores: np.ndarray | None = None,
) -> np.ndarray:
    selected: List[int] = []
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

    # Important: do not truncate away true variables if |S*| > top_m.
    return np.array(sorted(selected), dtype=int)


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
    all_vars = list(range(d))
    rng = np.random.default_rng(seed * 1009 + 17)

    true_vars = tuple(int(v) for v in true_vars)
    endpoints = interaction_endpoints(true_interactions)
    endpoints_set = set(endpoints)

    scores = np.zeros(d, dtype=float)

    if mode == "raw":
        return np.arange(d, dtype=int), scores, "none"

    if mode == "random":
        m = min(top_m, d)
        selected = np.array(sorted(rng.choice(d, size=m, replace=False).astype(int)), dtype=int)
        return selected, scores, "random"

    if mode == "oracle_support":
        selected = fill_with_support(true_vars, all_vars, top_m, rng, scores=None)
        return selected, scores, "forced_true_support_random_fill"

    if mode == "oracle_interaction":
        selected = fill_with_support(endpoints, all_vars, top_m, rng, scores=None)
        return selected, scores, "forced_interaction_endpoints_random_fill"

    if mode == "exclude_interaction":
        pool = [v for v in all_vars if v not in endpoints_set]
        if len(pool) == 0:
            selected = np.array([], dtype=int)
        else:
            m = min(top_m, len(pool))
            selected = np.array(sorted(rng.choice(pool, size=m, replace=False).astype(int)), dtype=int)
        return selected, scores, "random_excluding_interaction_endpoints"

    if mode == "rf":
        scores = rf_scores(X, y, seed=seed, n_estimators=rf_trees)
        selected = np.array(sorted(np.argsort(-scores)[: min(top_m, d)].astype(int)), dtype=int)
        return selected, scores, "rf"

    if mode == "rf_contains_support":
        scores = rf_scores(X, y, seed=seed, n_estimators=rf_trees)
        selected = fill_with_support(true_vars, all_vars, top_m, rng, scores=scores)
        return selected, scores, "forced_true_support_rf_fill"

    if mode == "rf_exclude_interaction":
        scores = rf_scores(X, y, seed=seed, n_estimators=rf_trees)
        pool = [v for v in all_vars if v not in endpoints_set]
        selected = np.array(sorted(sorted(pool, key=lambda v: float(scores[v]), reverse=True)[:top_m]), dtype=int)
        return selected, scores, "rf_excluding_interaction_endpoints"

    raise ValueError(f"Unknown screen mode={mode}")


def support_stats(selected_features: np.ndarray, true_vars: Sequence[int], true_interactions: Sequence[Tuple[int, int]]) -> Dict:
    selected = set(int(v) for v in selected_features)
    true_var_set = set(int(v) for v in true_vars)
    endpoints = set(interaction_endpoints(true_interactions))

    return {
        "effective_dim": len(selected_features),
        "screen_contains_all_true_vars": int(true_var_set.issubset(selected)) if true_var_set else np.nan,
        "screen_true_var_recall": len(true_var_set & selected) / len(true_var_set) if true_var_set else np.nan,
        "screen_contains_all_interaction_endpoints": int(endpoints.issubset(selected)) if endpoints else np.nan,
        "screen_interaction_endpoint_recall": len(endpoints & selected) / len(endpoints) if endpoints else np.nan,
        "screen_contains_true_interactions": int(
            all(int(i) in selected and int(j) in selected for i, j in true_interactions)
        ) if true_interactions else np.nan,
    }


# -----------------------------
# Explanation metrics
# -----------------------------

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


def local_to_full_scores(local_scores: np.ndarray, selected_features: np.ndarray, d_full: int) -> np.ndarray:
    full_scores = np.zeros(d_full, dtype=float)
    for local_idx, original_idx in enumerate(selected_features):
        full_scores[int(original_idx)] = float(local_scores[local_idx])
    return full_scores


def local_to_full_pair_scores(local_pair_scores: Dict[Tuple[int, int], float], selected_features: np.ndarray, d_full: int) -> Dict[Tuple[int, int], float]:
    full_pair_scores = {}
    selected_features = np.asarray(selected_features, dtype=int)

    for (i_local, j_local), score in local_pair_scores.items():
        if i_local >= len(selected_features) or j_local >= len(selected_features):
            continue
        i = int(selected_features[i_local])
        j = int(selected_features[j_local])
        full_pair_scores[tuple(sorted((i, j)))] = float(score)

    # Pairs not represented by the screened model get score 0.
    for i, j in itertools.combinations(range(d_full), 2):
        full_pair_scores.setdefault((i, j), 0.0)

    return full_pair_scores


def evaluate_variable_recovery(full_scores: np.ndarray, true_vars: Sequence[int]) -> Dict:
    true_set = set(int(v) for v in true_vars)

    if len(true_set) == 0:
        return {
            "selected_variables": [],
            "variable_precision": np.nan,
            "variable_recall": np.nan,
            "variable_f1": np.nan,
            "variable_auroc": np.nan,
            "variable_auprc": np.nan,
        }

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
        "active_score_mean": float(np.mean(active_scores)) if len(active_scores) else np.nan,
        "inactive_score_mean": float(np.mean(inactive_scores)) if len(inactive_scores) else np.nan,
        "active_score_min": float(np.min(active_scores)) if len(active_scores) else np.nan,
        "inactive_score_max": float(np.max(inactive_scores)) if len(inactive_scores) else np.nan,
    }


def evaluate_interaction_recovery(full_pair_scores: Dict[Tuple[int, int], float], true_interactions: Sequence[Tuple[int, int]]) -> Dict:
    true_set = {tuple(sorted((int(i), int(j)))) for i, j in true_interactions}

    if len(true_set) == 0:
        return {
            "selected_interactions": [],
            "interaction_precision": np.nan,
            "interaction_recall": np.nan,
            "interaction_f1": np.nan,
        }

    k = len(true_set)
    ranked = sorted(full_pair_scores.items(), key=lambda kv: kv[1], reverse=True)
    selected = {pair for pair, _ in ranked[:k]}

    precision, recall, f1 = f1_from_sets(selected, true_set)

    return {
        "selected_interactions": sorted(selected),
        "interaction_precision": precision,
        "interaction_recall": recall,
        "interaction_f1": f1,
    }


# -----------------------------
# Run / save / summarize
# -----------------------------

def append_rows(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    header = not path.exists()
    df.to_csv(path, mode="a", header=header, index=False)


def normalize_function_alias(name: str) -> str:
    aliases = {
        "core_c5": "core_interaction_c5",
        "proxy": "correlated_proxy",
    }
    return aliases.get(name, name)


def run_one(args, function_name: str, seed: int, mode: str) -> List[Dict]:
    function_name = normalize_function_alias(function_name)

    data = make_synthetic(
        function_name=function_name,
        n_train=args.samples,
        n_test=args.test_samples,
        d=args.dimension,
        noise=args.noise,
        seed=seed,
        standardize_target=True,
        nuisance_correlation=args.nuisance_correlation,
        n_correlated_proxies=args.n_correlated_proxies,
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
        top_m=args.top_m,
        true_vars=true_vars,
        true_interactions=true_interactions,
        seed=seed,
        rf_trees=args.rf_trees,
    )

    X_train_s = X_train[:, selected_features]
    X_test_s = X_test[:, selected_features]

    base = {
        "model": "KAN_BENCHMARK_SUITE",
        "function": function_name,
        "seed": seed,
        "samples": args.samples,
        "test_samples": args.test_samples,
        "dimension": args.dimension,
        "noise": args.noise,
        "nuisance_correlation": args.nuisance_correlation,
        "n_correlated_proxies": args.n_correlated_proxies,
        "screen_mode": mode,
        "screen_score_type": score_desc,
        "top_m_requested": args.top_m,
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
        "num_true_variables": len(true_vars),
        "num_true_interactions": len(true_interactions),
        "formula": gt.formula,
    }
    base.update(support_stats(selected_features, true_vars, true_interactions))

    try:
        if len(selected_features) == 0:
            raise RuntimeError("No selected features; cannot train KAN.")

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

        if args.compute_interactions and len(true_interactions) > 0:
            local_pair_scores = hessian_interaction_scores(model, X_test_s, args.hessian_points)
            full_pair_scores = local_to_full_pair_scores(local_pair_scores, selected_features, args.dimension)
            interaction_eval = evaluate_interaction_recovery(full_pair_scores, true_interactions)
            interaction_eval["interaction_scoring_computed"] = 1
        else:
            interaction_eval = {
                "selected_interactions": [],
                "interaction_precision": np.nan,
                "interaction_recall": np.nan,
                "interaction_f1": np.nan,
                "interaction_scoring_computed": 0,
            }

        rows: List[Dict] = []
        for explain_method, local_scores in explain_methods:
            full_scores = local_to_full_scores(local_scores, selected_features, args.dimension)
            row = dict(base)
            row.update({
                "status": "ok",
                "error": "",
                "traceback": "",
                "explain_method": explain_method,
                "train_mse": train_mse,
                "test_mse": test_mse,
                "importance_scores": full_scores.tolist(),
            })
            row.update(evaluate_variable_recovery(full_scores, true_vars))
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
            "interaction_scoring_computed": 0,
        })
        print(f"[WARN] failed function={function_name}, seed={seed}, mode={mode}: {exc}")
        return [row]


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    ok = df[df["status"].astype(str) == "ok"].copy()
    if ok.empty:
        return pd.DataFrame()

    group_cols = [
        "function", "dimension", "samples", "test_samples", "noise",
        "nuisance_correlation", "n_correlated_proxies",
        "screen_mode", "explain_method",
    ]

    numeric_cols = [
        "train_mse", "test_mse", "effective_dim",
        "screen_contains_all_true_vars", "screen_true_var_recall",
        "screen_contains_all_interaction_endpoints",
        "screen_interaction_endpoint_recall",
        "screen_contains_true_interactions",
        "num_true_variables", "num_true_interactions",
        "variable_f1", "variable_auroc", "variable_auprc",
        "interaction_f1", "interaction_scoring_computed",
    ]

    for col in numeric_cols:
        if col in ok.columns:
            ok[col] = pd.to_numeric(ok[col], errors="coerce")

    agg = {}
    for col in [
        "train_mse", "test_mse", "effective_dim",
        "screen_contains_all_true_vars", "screen_true_var_recall",
        "screen_contains_all_interaction_endpoints",
        "screen_interaction_endpoint_recall",
        "screen_contains_true_interactions",
        "num_true_variables", "num_true_interactions",
        "variable_f1", "variable_auroc", "variable_auprc",
        "interaction_f1", "interaction_scoring_computed",
    ]:
        if col in ok.columns:
            if col in {"train_mse", "test_mse", "variable_f1", "variable_auroc", "variable_auprc", "interaction_f1"}:
                agg[col] = ["mean", "std"]
            else:
                agg[col] = ["mean"]

    summary = ok.groupby(group_cols, dropna=False).agg(agg).reset_index()
    summary.columns = ["_".join(str(x) for x in col if x != "").rstrip("_") for col in summary.columns]

    count_cols = ["function", "screen_mode", "explain_method"]
    counts = df.groupby(count_cols, dropna=False).agg(
        num_rows=("status", "size"),
        num_failed=("status", lambda s: int((s.astype(str) != "ok").sum())),
    ).reset_index()

    summary = summary.merge(counts, on=count_cols, how="left")
    return summary


def plot_summary(summary: pd.DataFrame, out_dir: Path):
    if summary.empty:
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    plot_df = summary.copy()

    if "grad" in set(plot_df["explain_method"].astype(str)):
        plot_df = plot_df[plot_df["explain_method"].astype(str) == "grad"].copy()

    method_order = ["raw", "rf", "oracle_support", "random", "exclude_interaction", "rf_exclude_interaction"]
    method_labels = {
        "raw": "Raw",
        "rf": "RF-screened",
        "oracle_support": "Oracle-support",
        "random": "Random",
        "exclude_interaction": "Exclude-interaction",
        "rf_exclude_interaction": "RF excl. interaction",
    }

    functions = plot_df["function"].drop_duplicates().tolist()

    def plot_metric(metric_col: str, ylabel: str, title: str, filename: str):
        if metric_col not in plot_df.columns:
            return

        x = np.arange(len(functions))
        width = 0.13

        plt.figure(figsize=(max(10, len(functions) * 1.1), 5.5))
        for idx, method in enumerate(method_order):
            vals = []
            for fn in functions:
                hit = plot_df[(plot_df["function"] == fn) & (plot_df["screen_mode"] == method)]
                if hit.empty:
                    vals.append(np.nan)
                else:
                    vals.append(float(hit[metric_col].iloc[0]))
            plt.bar(x + (idx - (len(method_order)-1)/2) * width, vals, width=width, label=method_labels.get(method, method))
        plt.ylim(0, 1.08)
        plt.ylabel(ylabel)
        plt.xticks(x, functions, rotation=35, ha="right")
        plt.title(title)
        plt.legend(ncol=2, fontsize=8)
        plt.tight_layout()
        plt.savefig(out_dir / filename, dpi=250)
        plt.close()

    plot_metric(
        metric_col="variable_f1_mean",
        ylabel="Variable F1",
        title="Variable recovery across benchmark suite",
        filename="benchmark_variable_f1.png",
    )
    plot_metric(
        metric_col="interaction_f1_mean",
        ylabel="Interaction F1",
        title="Interaction recovery across benchmark suite",
        filename="benchmark_interaction_f1.png",
    )

    # Test MSE plot uses log scale and groups by function/mode.
    x = np.arange(len(functions))
    width = 0.13
    plt.figure(figsize=(max(10, len(functions) * 1.1), 5.5))
    for idx, method in enumerate(method_order):
        vals = []
        for fn in functions:
            hit = plot_df[(plot_df["function"] == fn) & (plot_df["screen_mode"] == method)]
            vals.append(float(hit["test_mse_mean"].iloc[0]) if not hit.empty else np.nan)
        plt.bar(x + (idx - (len(method_order)-1)/2) * width, vals, width=width, label=method_labels.get(method, method))
    plt.yscale("log")
    plt.ylabel("Test MSE")
    plt.xticks(x, functions, rotation=35, ha="right")
    plt.title("Prediction error across benchmark suite")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "benchmark_test_mse.png", dpi=250)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--functions",
        type=str,
        nargs="+",
        default=[
            "additive_sparse",
            "core_interaction",
            "core_interaction_c5",
            "pairwise_interaction",
            "compositional",
            "dense_quadratic",
            "correlated_proxy",
        ],
    )
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--test_samples", type=int, default=2048)
    parser.add_argument("--dimension", type=int, default=100)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--nuisance_correlation", type=float, default=0.0)
    parser.add_argument("--n_correlated_proxies", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])

    parser.add_argument(
        "--screen_modes",
        type=str,
        nargs="+",
        default=["raw", "rf", "oracle_support", "random", "exclude_interaction"],
    )
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
    parser.add_argument("--hessian_points", type=int, default=16)

    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--summary_out", type=str, default=None)
    parser.add_argument("--fig_dir", type=str, default=None)
    parser.add_argument("--resume", action="store_true")

    args = parser.parse_args()

    out_path = Path(args.out)
    existing_keys = set()

    if args.resume and out_path.exists():
        try:
            old = pd.read_csv(out_path)
            for r in old[["function", "seed", "screen_mode"]].drop_duplicates().itertuples(index=False):
                existing_keys.add((str(r.function), int(r.seed), str(r.screen_mode)))
            print(f"Resume mode: found {len(existing_keys)} completed function/seed/mode configs.")
        except Exception as exc:
            print(f"[WARN] Could not read existing output for resume: {exc}")

    for function_name in args.functions:
        function_name = normalize_function_alias(function_name)
        for seed in args.seeds:
            for mode in args.screen_modes:
                key = (function_name, int(seed), mode)
                if key in existing_keys:
                    print(f"Skipping completed {key}")
                    continue

                print(f"Running function={function_name}, seed={seed}, mode={mode}")
                try:
                    rows = run_one(args, function_name=function_name, seed=seed, mode=mode)
                except Exception as exc:
                    rows = [{
                        "model": "KAN_BENCHMARK_SUITE",
                        "function": function_name,
                        "seed": seed,
                        "screen_mode": mode,
                        "status": "failed",
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                    }]
                    print(f"[WARN] failed before row construction: {function_name}, seed={seed}, mode={mode}: {exc}")

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
