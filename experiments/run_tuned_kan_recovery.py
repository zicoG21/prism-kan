from __future__ import annotations

import argparse
import itertools
import json
import random
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

# pykan import. Different pykan versions expose KAN slightly differently.
try:
    from kan import KAN
except Exception:
    from kan.MultKAN import MultKAN as KAN


Pair = Tuple[int, int]


# ============================================================
# Basic utilities
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def canonical_pairs(pairs: Sequence[Tuple[int, int]]) -> Tuple[Pair, ...]:
    return tuple(tuple(sorted((int(i), int(j)))) for i, j in pairs)


def interaction_endpoints(pairs: Sequence[Pair]) -> Tuple[int, ...]:
    s = set()
    for i, j in pairs:
        s.add(int(i))
        s.add(int(j))
    return tuple(sorted(s))


def endpoint_recovery(selected_variables: Sequence[int], true_interactions: Sequence[Pair], prefix: str) -> Dict:
    endpoints = set(interaction_endpoints(true_interactions))
    selected = set(int(v) for v in selected_variables)
    if not endpoints:
        return {
            f"{prefix}_contains_all_interaction_endpoints": np.nan,
            f"{prefix}_interaction_endpoint_recall": np.nan,
        }

    return {
        f"{prefix}_contains_all_interaction_endpoints": int(endpoints.issubset(selected)),
        f"{prefix}_interaction_endpoint_recall": len(endpoints & selected) / len(endpoints),
    }


def f1_from_sets(pred: set, true: set):
    if len(true) == 0:
        return np.nan, np.nan, np.nan
    if len(pred) == 0:
        return 0.0, 0.0, 0.0
    tp = len(pred & true)
    precision = tp / len(pred)
    recall = tp / len(true)
    f1 = 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)
    return precision, recall, f1


def mse_np(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((pred.reshape(-1, 1) - target.reshape(-1, 1)) ** 2))


def safe_auroc_auprc(y_true: np.ndarray, scores: np.ndarray):
    try:
        auroc = float(roc_auc_score(y_true, scores))
    except Exception:
        auroc = np.nan
    try:
        auprc = float(average_precision_score(y_true, scores))
    except Exception:
        auprc = np.nan
    return auroc, auprc


def normalize_function_alias(name: str) -> str:
    aliases = {
        "core_c5": "core_interaction_c5",
        "proxy": "correlated_proxy",
    }
    return aliases.get(name, name)


def batch_predict(model, X_np: np.ndarray, device: str, batch_size: int = 4096) -> np.ndarray:
    model.eval()
    outs = []
    with torch.no_grad():
        for start in range(0, len(X_np), batch_size):
            xb = torch.tensor(X_np[start:start + batch_size], dtype=torch.float32, device=device)
            pred = model(xb)
            outs.append(pred.detach().cpu().numpy().reshape(-1, 1))
    return np.vstack(outs)


# ============================================================
# Screening
# ============================================================

def rf_importance_scores(X: np.ndarray, y: np.ndarray, seed: int, trees: int = 500) -> np.ndarray:
    rf = RandomForestRegressor(
        n_estimators=trees,
        random_state=seed,
        n_jobs=-1,
        min_samples_leaf=2,
    )
    rf.fit(X, y.reshape(-1))
    return np.asarray(rf.feature_importances_, dtype=float)


def fill_support(must_include: Sequence[int], d: int, top_m: int, rng: np.random.Generator) -> np.ndarray:
    selected = []
    seen = set()
    for v in must_include:
        v = int(v)
        if v not in seen:
            selected.append(v)
            seen.add(v)
    rest = [i for i in range(d) if i not in seen]
    rng.shuffle(rest)
    selected.extend(rest[:max(0, top_m - len(selected))])
    return np.array(sorted(selected[:top_m]), dtype=int)


def select_features(
    mode: str,
    X: np.ndarray,
    y: np.ndarray,
    top_m: int,
    true_vars: Sequence[int],
    true_interactions: Sequence[Pair],
    seed: int,
    rf_trees: int,
):
    d = X.shape[1]
    rng = np.random.default_rng(seed * 1009 + 17)
    scores = np.zeros(d, dtype=float)

    if mode == "raw":
        return np.arange(d, dtype=int), scores, "none"

    if mode == "rf":
        scores = rf_importance_scores(X, y, seed=seed, trees=rf_trees)
        selected = np.array(sorted(np.argsort(-scores)[:min(top_m, d)]), dtype=int)
        return selected, scores, "rf"

    if mode == "oracle_support":
        selected = fill_support(true_vars, d, top_m, rng)
        return selected, scores, "oracle_support_random_fill"

    if mode == "random":
        selected = np.array(sorted(rng.choice(d, size=min(top_m, d), replace=False)), dtype=int)
        return selected, scores, "random"

    if mode == "exclude_interaction":
        endpoints = set(interaction_endpoints(true_interactions))
        pool = [i for i in range(d) if i not in endpoints]
        selected = np.array(sorted(rng.choice(pool, size=min(top_m, len(pool)), replace=False)), dtype=int)
        return selected, scores, "random_excluding_interaction_endpoints"

    raise ValueError(f"Unknown screen_mode={mode}")


def support_stats(selected_features: np.ndarray, true_vars: Sequence[int], true_interactions: Sequence[Pair]) -> Dict:
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


# ============================================================
# KAN training
# ============================================================

def make_kan(input_dim: int, width_hidden: int, grid: int, k: int, seed: int, device: str):
    set_seed(seed)
    # Standard pykan constructor.
    try:
        model = KAN(width=[input_dim, width_hidden, 1], grid=grid, k=k, seed=seed, device=device)
    except TypeError:
        # Some versions do not accept device in constructor.
        model = KAN(width=[input_dim, width_hidden, 1], grid=grid, k=k, seed=seed)
        try:
            model = model.to(device)
        except Exception:
            pass
    return model


def train_kan(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    args,
    seed: int,
    device: str,
):
    model = make_kan(
        input_dim=X_train.shape[1],
        width_hidden=args.width_hidden,
        grid=args.grid,
        k=args.k,
        seed=seed,
        device=device,
    )

    dataset = {
        "train_input": torch.tensor(X_train, dtype=torch.float32, device=device),
        "train_label": torch.tensor(y_train, dtype=torch.float32, device=device),
        "test_input": torch.tensor(X_test, dtype=torch.float32, device=device),
        "test_label": torch.tensor(y_test, dtype=torch.float32, device=device),
    }

    fit_kwargs = {
        "opt": args.opt,
        "steps": args.steps,
        "lamb": args.lamb,
    }

    # These kwargs exist in common pykan versions. If a local version does not
    # support them, the fallback below removes them.
    fit_kwargs["update_grid"] = bool(args.update_grid)
    fit_kwargs["grid_update_num"] = int(args.grid_update_num)

    if args.batch and args.batch > 0:
        fit_kwargs["batch"] = int(args.batch)

    try:
        model.fit(dataset, **fit_kwargs)
    except TypeError:
        # Fallback for older pykan signatures.
        fit_kwargs.pop("update_grid", None)
        fit_kwargs.pop("grid_update_num", None)
        fit_kwargs.pop("batch", None)
        model.fit(dataset, **fit_kwargs)

    return model


# ============================================================
# Explanation scores
# ============================================================

def local_to_full_scores(local_scores: np.ndarray, selected_features: np.ndarray, d_full: int) -> np.ndarray:
    full = np.zeros(d_full, dtype=float)
    for local_idx, orig_idx in enumerate(selected_features):
        full[int(orig_idx)] = float(local_scores[local_idx])
    return full


def local_to_full_pair_scores(local_pair_scores: Dict[Pair, float], selected_features: np.ndarray, d_full: int):
    selected_features = np.asarray(selected_features, dtype=int)
    full: Dict[Pair, float] = {}
    for (i_local, j_local), score in local_pair_scores.items():
        if i_local >= len(selected_features) or j_local >= len(selected_features):
            continue
        i = int(selected_features[i_local])
        j = int(selected_features[j_local])
        full[tuple(sorted((i, j)))] = float(score)
    for i, j in itertools.combinations(range(d_full), 2):
        full.setdefault((i, j), 0.0)
    return full


def gradient_importance(model, X_np: np.ndarray, device: str, points: int) -> np.ndarray:
    model.eval()
    n = min(points, len(X_np))
    X = torch.tensor(X_np[:n], dtype=torch.float32, device=device)
    X.requires_grad_(True)
    y = model(X).sum()
    grad = torch.autograd.grad(y, X, create_graph=False)[0]
    return grad.abs().mean(dim=0).detach().cpu().numpy()


def finite_difference_pair_scores(model, X_np: np.ndarray, device: str, points: int, h: float, batch_size: int):
    d = X_np.shape[1]
    n = min(points, len(X_np))
    X = X_np[:n].copy()
    f0 = batch_predict(model, X, device=device, batch_size=batch_size).reshape(-1)

    scores: Dict[Pair, float] = {}
    for i, j in itertools.combinations(range(d), 2):
        Xi = X.copy()
        Xj = X.copy()
        Xij = X.copy()
        Xi[:, i] += h
        Xj[:, j] += h
        Xij[:, i] += h
        Xij[:, j] += h

        fi = batch_predict(model, Xi, device=device, batch_size=batch_size).reshape(-1)
        fj = batch_predict(model, Xj, device=device, batch_size=batch_size).reshape(-1)
        fij = batch_predict(model, Xij, device=device, batch_size=batch_size).reshape(-1)

        mixed = (fij - fi - fj + f0) / (h ** 2)
        scores[(i, j)] = float(np.mean(np.abs(mixed)))

    return scores


def hessian_pair_scores(model, X_np: np.ndarray, device: str, points: int):
    d = X_np.shape[1]
    n = min(points, len(X_np))
    X = torch.tensor(X_np[:n], dtype=torch.float32, device=device)
    score_mat = torch.zeros((d, d), dtype=torch.float32, device=device)

    def scalar_func(z):
        return model(z.unsqueeze(0)).sum()

    model.eval()
    for idx in range(n):
        z = X[idx].clone().detach().requires_grad_(True)
        H = torch.autograd.functional.hessian(scalar_func, z)
        score_mat += H.abs().detach()

    score_mat = score_mat / max(n, 1)
    A = score_mat.detach().cpu().numpy()
    return {(i, j): float(A[i, j]) for i, j in itertools.combinations(range(d), 2)}


def evaluate_variable_recovery(scores: np.ndarray, true_vars: Sequence[int]) -> Dict:
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
    selected = set(int(i) for i in np.argsort(-scores)[:k])
    p, r, f1 = f1_from_sets(selected, true_set)

    y_true = np.array([1 if i in true_set else 0 for i in range(len(scores))], dtype=int)
    auroc, auprc = safe_auroc_auprc(y_true, scores)

    return {
        "selected_variables": sorted(selected),
        "variable_precision": p,
        "variable_recall": r,
        "variable_f1": f1,
        "variable_auroc": auroc,
        "variable_auprc": auprc,
    }


def evaluate_interaction_recovery(pair_scores: Dict[Pair, float], true_interactions: Sequence[Pair]) -> Dict:
    true_set = set(canonical_pairs(true_interactions))
    if len(true_set) == 0:
        return {
            "selected_interactions": [],
            "interaction_precision": np.nan,
            "interaction_recall": np.nan,
            "interaction_f1": np.nan,
            "selected_interaction_endpoint_recall": np.nan,
            "selected_interaction_contains_all_endpoints": np.nan,
            "true_interaction_best_rank": np.nan,
            "true_interaction_worst_rank": np.nan,
            "true_interaction_rank_mean": np.nan,
            "true_interaction_score_mean": np.nan,
            "max_nontrue_interaction_score": np.nan,
            "true_interaction_mean_score_margin": np.nan,
            "true_interaction_beats_all_false": np.nan,
        }

    k = len(true_set)
    ranked = sorted(pair_scores.items(), key=lambda kv: kv[1], reverse=True)
    selected = {pair for pair, _ in ranked[:k]}
    rank_lookup = {pair: idx + 1 for idx, (pair, _) in enumerate(ranked)}

    p, r, f1 = f1_from_sets(selected, true_set)

    true_scores = [float(pair_scores.get(pair, 0.0)) for pair in true_set]
    nontrue_scores = [float(v) for pair, v in pair_scores.items() if pair not in true_set]
    true_ranks = [float(rank_lookup.get(pair, len(ranked) + 1)) for pair in true_set]
    max_nontrue = float(np.max(nontrue_scores)) if nontrue_scores else np.nan
    true_mean = float(np.mean(true_scores)) if true_scores else np.nan

    true_endpoints = set(interaction_endpoints(true_set))
    selected_endpoints = set(interaction_endpoints(selected))

    return {
        "selected_interactions": sorted(selected),
        "interaction_precision": p,
        "interaction_recall": r,
        "interaction_f1": f1,
        "selected_interaction_endpoint_recall": (
            len(true_endpoints & selected_endpoints) / len(true_endpoints) if true_endpoints else np.nan
        ),
        "selected_interaction_contains_all_endpoints": (
            int(true_endpoints.issubset(selected_endpoints)) if true_endpoints else np.nan
        ),
        "true_interaction_best_rank": float(np.min(true_ranks)) if true_ranks else np.nan,
        "true_interaction_worst_rank": float(np.max(true_ranks)) if true_ranks else np.nan,
        "true_interaction_rank_mean": float(np.mean(true_ranks)) if true_ranks else np.nan,
        "true_interaction_score_mean": true_mean,
        "max_nontrue_interaction_score": max_nontrue,
        "true_interaction_mean_score_margin": true_mean - max_nontrue if np.isfinite(max_nontrue) else np.nan,
        "true_interaction_beats_all_false": (
            int(min(true_scores) > max_nontrue) if true_scores and np.isfinite(max_nontrue) else np.nan
        ),
    }


# ============================================================
# Main run
# ============================================================

def run_one(args, function_name: str, screen_mode: str, seed: int, device: str) -> Dict:
    function_name = normalize_function_alias(function_name)

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

    true_vars = tuple(int(v) for v in gt.active_variables)
    true_interactions = canonical_pairs(gt.interactions)

    selected_features, screen_scores, screen_desc = select_features(
        mode=screen_mode,
        X=X_train,
        y=y_train,
        top_m=args.top_m,
        true_vars=true_vars,
        true_interactions=true_interactions,
        seed=seed,
        rf_trees=args.rf_trees,
    )

    Xtr = X_train[:, selected_features]
    Xte = X_test[:, selected_features]

    base = {
        "model": "KAN_tuned_fixed",
        "function": function_name,
        "screen_mode": screen_mode,
        "seed": seed,
        "samples": args.samples,
        "test_samples": args.test_samples,
        "dimension": args.dimension,
        "noise": args.noise,
        "top_m": args.top_m,
        "selected_screen_features": selected_features.tolist(),
        "screen_score_type": screen_desc,
        "screen_scores": screen_scores.tolist(),
        "true_variables": list(true_vars),
        "true_interactions": list(true_interactions),
        "num_true_variables": len(true_vars),
        "num_true_interactions": len(true_interactions),
        "formula": gt.formula,
        "grid": args.grid,
        "k": args.k,
        "width_hidden": args.width_hidden,
        "lamb": args.lamb,
        "steps": args.steps,
        "opt": args.opt,
        "update_grid": int(args.update_grid),
        "grid_update_num": args.grid_update_num,
    }
    base.update(support_stats(selected_features, true_vars, true_interactions))

    try:
        model = train_kan(Xtr, y_train, Xte, y_test, args, seed=seed, device=device)

        train_pred = batch_predict(model, Xtr, device=device, batch_size=args.pred_batch_size)
        test_pred = batch_predict(model, Xte, device=device, batch_size=args.pred_batch_size)
        train_mse = mse_np(train_pred, y_train)
        test_mse = mse_np(test_pred, y_test)

        local_var_scores = gradient_importance(model, Xte, device=device, points=args.variable_points)
        full_var_scores = local_to_full_scores(local_var_scores, selected_features, args.dimension)

        if len(true_interactions) > 0:
            if args.interaction_method == "fd":
                local_pair_scores = finite_difference_pair_scores(
                    model, Xte, device=device, points=args.fd_points, h=args.fd_h, batch_size=args.pred_batch_size
                )
            elif args.interaction_method == "hessian":
                local_pair_scores = hessian_pair_scores(
                    model, Xte, device=device, points=args.hessian_points
                )
            else:
                raise ValueError(f"Unknown interaction_method={args.interaction_method}")

            full_pair_scores = local_to_full_pair_scores(local_pair_scores, selected_features, args.dimension)
        else:
            full_pair_scores = {}

        row = dict(base)
        row.update({
            "status": "ok",
            "error": "",
            "traceback": "",
            "train_mse": train_mse,
            "test_mse": test_mse,
            "variable_method": "grad",
            "interaction_method": args.interaction_method,
            "importance_scores": full_var_scores.tolist(),
        })
        var_eval = evaluate_variable_recovery(full_var_scores, true_vars)
        row.update(var_eval)
        row.update(endpoint_recovery(var_eval["selected_variables"], true_interactions, "explain"))
        row.update(evaluate_interaction_recovery(full_pair_scores, true_interactions))
        return row

    except Exception as exc:
        row = dict(base)
        row.update({
            "status": "failed",
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "train_mse": np.nan,
            "test_mse": np.nan,
            "variable_f1": np.nan,
            "interaction_f1": np.nan,
            "selected_variables": [],
            "selected_interactions": [],
            "explain_contains_all_interaction_endpoints": np.nan,
            "explain_interaction_endpoint_recall": np.nan,
            "selected_interaction_endpoint_recall": np.nan,
            "selected_interaction_contains_all_endpoints": np.nan,
            "true_interaction_best_rank": np.nan,
            "true_interaction_worst_rank": np.nan,
            "true_interaction_rank_mean": np.nan,
            "true_interaction_mean_score_margin": np.nan,
            "true_interaction_beats_all_false": np.nan,
        })
        print(f"[WARN] failed function={function_name}, screen={screen_mode}, seed={seed}: {exc}")
        return row


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    ok = df[df["status"].astype(str) == "ok"].copy()
    if ok.empty:
        return pd.DataFrame()

    group_cols = [
        "model", "function", "screen_mode", "dimension", "samples", "noise",
        "grid", "k", "width_hidden", "lamb", "steps", "opt", "update_grid",
        "interaction_method",
    ]

    numeric_cols = [
        "train_mse", "test_mse", "effective_dim",
        "screen_contains_all_true_vars", "screen_true_var_recall",
        "screen_contains_all_interaction_endpoints", "screen_interaction_endpoint_recall",
        "screen_contains_true_interactions",
        "explain_contains_all_interaction_endpoints", "explain_interaction_endpoint_recall",
        "selected_interaction_endpoint_recall", "selected_interaction_contains_all_endpoints",
        "variable_f1", "variable_auroc", "variable_auprc",
        "interaction_f1", "true_interaction_best_rank", "true_interaction_worst_rank",
        "true_interaction_rank_mean", "true_interaction_score_mean",
        "max_nontrue_interaction_score", "true_interaction_mean_score_margin",
        "true_interaction_beats_all_false",
    ]

    for col in numeric_cols:
        if col in ok.columns:
            ok[col] = pd.to_numeric(ok[col], errors="coerce")

    agg = {}
    for col in numeric_cols:
        if col in ok.columns:
            if col in {"train_mse", "test_mse", "variable_f1", "variable_auroc", "variable_auprc",
                       "interaction_f1", "true_interaction_best_rank", "true_interaction_worst_rank",
                       "true_interaction_rank_mean", "true_interaction_score_mean",
                       "max_nontrue_interaction_score", "true_interaction_mean_score_margin"}:
                agg[col] = ["mean", "std"]
            else:
                agg[col] = ["mean"]

    summary = ok.groupby(group_cols, dropna=False).agg(agg).reset_index()
    summary.columns = ["_".join(str(x) for x in col if x != "").rstrip("_") for col in summary.columns]
    return summary


def plot_summary(summary: pd.DataFrame, fig_dir: Path):
    if summary.empty:
        return

    fig_dir.mkdir(parents=True, exist_ok=True)

    funcs = summary["function"].drop_duplicates().tolist()
    modes = ["raw", "rf", "oracle_support", "random", "exclude_interaction"]
    labels = {
        "raw": "Raw tuned KAN",
        "rf": "RF-screened tuned KAN",
        "oracle_support": "Oracle-support tuned KAN",
        "random": "Random-screened tuned KAN",
        "exclude_interaction": "Exclude-interaction tuned KAN",
    }

    def plot_metric(col: str, ylabel: str, title: str, fname: str, log: bool = False):
        x = np.arange(len(funcs))
        width = 0.15
        plt.figure(figsize=(max(10, len(funcs) * 1.1), 5.5))
        for idx, mode in enumerate(modes):
            vals = []
            for fn in funcs:
                hit = summary[(summary["function"] == fn) & (summary["screen_mode"] == mode)]
                vals.append(float(hit[col].iloc[0]) if not hit.empty and col in hit.columns else np.nan)
            plt.bar(x + (idx - (len(modes)-1)/2) * width, vals, width=width, label=labels.get(mode, mode))
        if log:
            plt.yscale("log")
        else:
            plt.ylim(0, 1.08)
        plt.ylabel(ylabel)
        plt.xticks(x, funcs, rotation=35, ha="right")
        plt.title(title)
        plt.legend(ncol=2, fontsize=8)
        plt.tight_layout()
        plt.savefig(fig_dir / fname, dpi=250)
        plt.close()

    plot_metric("interaction_f1_mean", "Interaction F1", "Tuned KAN interaction recovery", "tuned_kan_interaction_f1.png", False)
    plot_metric("variable_f1_mean", "Variable F1", "Tuned KAN variable recovery", "tuned_kan_variable_f1.png", False)
    plot_metric("test_mse_mean", "Test MSE", "Tuned KAN prediction error", "tuned_kan_test_mse.png", True)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--functions", nargs="+", default=[
        "core_interaction", "core_interaction_c5", "correlated_proxy", "pairwise_interaction"
    ])
    parser.add_argument("--screen_modes", nargs="+", default=["raw", "rf", "oracle_support"])
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--test_samples", type=int, default=2048)
    parser.add_argument("--dimension", type=int, default=100)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])

    # Adrian-style tuned KAN params.
    parser.add_argument("--grid", type=int, default=10)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--width_hidden", type=int, default=5)
    parser.add_argument("--lamb", type=float, default=0.0022356972728751583)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--opt", default="LBFGS")

    # KAN grid-update controls. If SGELSY/grid-update crashes, use --no_update_grid.
    parser.add_argument("--update_grid", action="store_true")
    parser.add_argument("--no_update_grid", dest="update_grid", action="store_false")
    parser.set_defaults(update_grid=True)
    parser.add_argument("--grid_update_num", type=int, default=10)
    parser.add_argument("--batch", type=int, default=-1)

    parser.add_argument("--top_m", type=int, default=20)
    parser.add_argument("--rf_trees", type=int, default=500)

    parser.add_argument("--variable_points", type=int, default=2048)
    parser.add_argument("--interaction_method", choices=["fd", "hessian"], default="fd")
    parser.add_argument("--fd_points", type=int, default=32)
    parser.add_argument("--fd_h", type=float, default=1e-2)
    parser.add_argument("--hessian_points", type=int, default=16)
    parser.add_argument("--pred_batch_size", type=int, default=4096)

    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")

    parser.add_argument("--out", required=True)
    parser.add_argument("--summary_out", required=True)
    parser.add_argument("--fig_dir", default=None)

    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device={device}")

    rows = []
    for fn in args.functions:
        fn = normalize_function_alias(fn)
        for mode in args.screen_modes:
            for seed in args.seeds:
                print(f"[RUN] function={fn} screen={mode} seed={seed}")
                rows.append(run_one(args, fn, mode, seed, device))

                # Save incrementally so partial results survive crashes.
                out_path = Path(args.out)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(rows).to_csv(out_path, index=False)

    df = pd.DataFrame(rows)
    summary = summarize(df)

    summary_path = Path(args.summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)

    if args.fig_dir is not None:
        plot_summary(summary, Path(args.fig_dir))

    print(f"Wrote rows to {args.out}")
    print(f"Wrote summary to {args.summary_out}")
    if args.fig_dir is not None:
        print(f"Wrote figures to {args.fig_dir}")


if __name__ == "__main__":
    main()
