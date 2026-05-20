from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
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
    model = KAN(width=[d, width_hidden, 1], grid=grid, k=k, seed=seed)

    dataset = {
        "train_input": torch.tensor(X_train, dtype=torch.float32),
        "train_label": torch.tensor(y_train, dtype=torch.float32),
        "test_input": torch.tensor(X_test, dtype=torch.float32),
        "test_label": torch.tensor(y_test, dtype=torch.float32),
    }

    if hasattr(model, "fit"):
        model.fit(dataset, opt="LBFGS", steps=steps, lamb=lamb)
    else:
        raise RuntimeError("This pykan version does not expose model.fit(...).")

    return model


def correlation_scores(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    y_flat = y.reshape(-1)
    y_center = y_flat - y_flat.mean()
    y_norm = np.linalg.norm(y_center)
    scores = np.zeros(X.shape[1], dtype=float)
    if y_norm == 0:
        return scores

    for j in range(X.shape[1]):
        x = X[:, j]
        x_center = x - x.mean()
        denom = np.linalg.norm(x_center) * y_norm
        scores[j] = 0.0 if denom == 0 else abs(float(np.dot(x_center, y_center) / denom))
    return scores


def mutual_info_scores(X: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    return np.asarray(mutual_info_regression(X, y.reshape(-1), random_state=seed), dtype=float)


def rf_scores(X: np.ndarray, y: np.ndarray, seed: int, n_estimators: int = 300) -> np.ndarray:
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=seed,
        n_jobs=-1,
        min_samples_leaf=2,
    )
    rf.fit(X, y.reshape(-1))
    return np.asarray(rf.feature_importances_, dtype=float)


def rank_fill(
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
            remaining_sorted = sorted(remaining, key=lambda v: float(scores[v]), reverse=True)
            selected.extend(remaining_sorted[:need])

    return np.array(sorted(selected[:top_m]), dtype=int)


def get_interaction_endpoints(true_interactions: Sequence[Tuple[int, int]]) -> Tuple[int, ...]:
    s = set()
    for i, j in true_interactions:
        s.add(int(i))
        s.add(int(j))
    return tuple(sorted(s))


def controlled_screen(
    X: np.ndarray,
    y: np.ndarray,
    mode: str,
    top_m: int,
    seed: int,
    repeat: int,
    true_vars: Sequence[int],
    true_interactions: Sequence[Tuple[int, int]],
    rf_trees: int,
) -> Tuple[np.ndarray, np.ndarray, str]:
    d = X.shape[1]
    rng = np.random.default_rng(seed * 1009 + repeat * 9176 + 17)
    all_vars = list(range(d))
    true_vars = tuple(int(v) for v in true_vars)
    interaction_endpoints = get_interaction_endpoints(true_interactions)

    score_desc = "none"
    scores = np.zeros(d, dtype=float)

    if mode == "random":
        selected = np.array(sorted(rng.choice(d, size=min(top_m, d), replace=False).astype(int)), dtype=int)
        return selected, scores, score_desc

    if mode in {"oracle_support", "random_contains_support", "support_random"}:
        selected = rank_fill(
            must_include=true_vars,
            candidate_pool=all_vars,
            top_m=top_m,
            rng=rng,
            scores=None,
        )
        return selected, scores, "forced_true_support_random_fill"

    if mode in {"oracle_interaction", "interaction_random"}:
        selected = rank_fill(
            must_include=interaction_endpoints,
            candidate_pool=all_vars,
            top_m=top_m,
            rng=rng,
            scores=None,
        )
        return selected, scores, "forced_interaction_endpoints_random_fill"

    if mode in {"exclude_interaction", "no_interaction"}:
        pool = [v for v in all_vars if v not in set(interaction_endpoints)]
        selected = np.array(sorted(rng.choice(pool, size=min(top_m, len(pool)), replace=False).astype(int)), dtype=int)
        return selected, scores, "random_excluding_interaction_endpoints"

    if mode in {"exclude_support", "no_support"}:
        pool = [v for v in all_vars if v not in set(true_vars)]
        selected = np.array(sorted(rng.choice(pool, size=min(top_m, len(pool)), replace=False).astype(int)), dtype=int)
        return selected, scores, "random_excluding_true_support"

    if mode == "rf":
        scores = rf_scores(X, y, seed=seed + repeat * 131, n_estimators=rf_trees)
        selected = np.array(sorted(np.argsort(-scores)[:top_m].astype(int)), dtype=int)
        return selected, scores, "rf"

    if mode == "rf_contains_support":
        scores = rf_scores(X, y, seed=seed + repeat * 131, n_estimators=rf_trees)
        selected = rank_fill(true_vars, all_vars, top_m, rng, scores=scores)
        return selected, scores, "forced_true_support_rf_fill"

    if mode == "rf_exclude_interaction":
        scores = rf_scores(X, y, seed=seed + repeat * 131, n_estimators=rf_trees)
        pool = [v for v in all_vars if v not in set(interaction_endpoints)]
        selected = np.array(sorted(sorted(pool, key=lambda v: float(scores[v]), reverse=True)[:top_m]), dtype=int)
        return selected, scores, "rf_excluding_interaction_endpoints"

    if mode == "mutual_info":
        scores = mutual_info_scores(X, y, seed=seed + repeat * 131)
        selected = np.array(sorted(np.argsort(-scores)[:top_m].astype(int)), dtype=int)
        return selected, scores, "mutual_info"

    if mode == "mi_contains_support":
        scores = mutual_info_scores(X, y, seed=seed + repeat * 131)
        selected = rank_fill(true_vars, all_vars, top_m, rng, scores=scores)
        return selected, scores, "forced_true_support_mi_fill"

    if mode == "correlation":
        scores = correlation_scores(X, y)
        selected = np.array(sorted(np.argsort(-scores)[:top_m].astype(int)), dtype=int)
        return selected, scores, "correlation"

    raise ValueError(f"Unknown screen mode: {mode}")


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


def evaluate_interaction_recovery(
    full_pair_scores: Dict[Tuple[int, int], float],
    true_interactions: Sequence[Tuple[int, int]],
) -> Dict:
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


def support_stats(selected_features: np.ndarray, true_vars: Sequence[int], true_interactions: Sequence[Tuple[int, int]]) -> Dict:
    selected = set(int(v) for v in selected_features)
    true_var_set = set(int(v) for v in true_vars)
    interaction_endpoints = set(get_interaction_endpoints(true_interactions))

    return {
        "screen_contains_all_true_vars": int(true_var_set.issubset(selected)),
        "screen_true_var_recall": len(true_var_set & selected) / len(true_var_set) if true_var_set else np.nan,
        "screen_contains_all_interaction_endpoints": int(interaction_endpoints.issubset(selected)) if interaction_endpoints else np.nan,
        "screen_interaction_endpoint_recall": len(interaction_endpoints & selected) / len(interaction_endpoints) if interaction_endpoints else np.nan,
        "screen_contains_true_interactions": int(
            all(int(i) in selected and int(j) in selected for i, j in true_interactions)
        ) if true_interactions else np.nan,
    }


def run_one(args, seed: int, repeat: int, screen_mode: str, top_m: int) -> List[Dict]:
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

    true_vars = tuple(int(v) for v in gt.active_variables)
    true_interactions = tuple(tuple(sorted((int(i), int(j)))) for i, j in gt.interactions)

    selected_features, screen_scores, score_desc = controlled_screen(
        X=X_train,
        y=y_train,
        mode=screen_mode,
        top_m=top_m,
        seed=seed,
        repeat=repeat,
        true_vars=true_vars,
        true_interactions=true_interactions,
        rf_trees=args.rf_trees,
    )

    X_train_s = X_train[:, selected_features]
    X_test_s = X_test[:, selected_features]

    model = train_pykan(
        X_train=X_train_s,
        y_train=y_train,
        X_test=X_test_s,
        y_test=y_test,
        width_hidden=args.kan_width,
        grid=args.kan_grid,
        k=args.kan_k,
        steps=args.kan_steps,
        lamb=args.kan_lamb,
        seed=seed + repeat * 10000,
    )

    train_mse = mse_np(predict_model(model, X_train_s), y_train)
    test_mse = mse_np(predict_model(model, X_test_s), y_test)

    stats = support_stats(selected_features, true_vars, true_interactions)

    explain_methods = []
    if "grad" in args.explain_methods:
        explain_methods.append(("grad", gradient_importance(model, X_test_s)))
    if "perm" in args.explain_methods:
        explain_methods.append(("perm", permutation_importance(model, X_test_s, seed + repeat * 10000)))

    interaction_eval = {}
    if args.compute_interactions:
        local_pair_scores = hessian_interaction_scores(model, X_test_s, args.hessian_points)
        full_pair_scores = local_to_full_pair_scores(local_pair_scores, selected_features, args.dimension)
        interaction_eval = evaluate_interaction_recovery(full_pair_scores, true_interactions)

    rows: List[Dict] = []

    for explain_method, local_scores in explain_methods:
        full_scores = local_to_full_scores(local_scores, selected_features, args.dimension)
        var_eval = evaluate_variable_recovery(full_scores, true_vars)

        row = {
            "model": "SUPPORT_CONTROLLED_SCREENED_KAN",
            "function": args.function,
            "seed": seed,
            "screen_repeat": repeat,
            "samples": args.samples,
            "test_samples": args.test_samples,
            "dimension": args.dimension,
            "noise": args.noise,
            "screen_mode": screen_mode,
            "screen_score_type": score_desc,
            "top_m": top_m,
            "selected_screen_features": selected_features.tolist(),
            "screen_scores": screen_scores.tolist(),
            "kan_width": args.kan_width,
            "kan_grid": args.kan_grid,
            "kan_k": args.kan_k,
            "kan_steps": args.kan_steps,
            "kan_lamb": args.kan_lamb,
            "explain_method": explain_method,
            "train_mse": train_mse,
            "test_mse": test_mse,
            "true_variables": list(true_vars),
            "true_interactions": list(true_interactions),
            "formula": gt.formula,
            "importance_scores": full_scores.tolist(),
        }
        row.update(stats)
        row.update(var_eval)
        if args.compute_interactions:
            row.update(interaction_eval)
        rows.append(row)

    return rows


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["function", "dimension", "noise", "screen_mode", "top_m", "explain_method"]

    agg = {
        "train_mse": ["mean", "std"],
        "test_mse": ["mean", "std"],
        "screen_contains_all_true_vars": ["mean"],
        "screen_true_var_recall": ["mean"],
        "screen_contains_all_interaction_endpoints": ["mean"],
        "screen_interaction_endpoint_recall": ["mean"],
        "screen_contains_true_interactions": ["mean"],
        "variable_f1": ["mean", "std"],
        "variable_auroc": ["mean", "std"],
        "variable_auprc": ["mean", "std"],
    }

    if "interaction_f1" in df.columns:
        agg["interaction_f1"] = ["mean", "std"]

    summary = df.groupby(group_cols, dropna=False).agg(agg).reset_index()
    summary.columns = ["_".join(str(x) for x in col if x != "").rstrip("_") for col in summary.columns]
    return summary


def plot_summary(summary: pd.DataFrame, out_path: Path):
    if summary.empty:
        return

    plot_df = summary.copy()
    if "grad" in set(plot_df["explain_method"].astype(str)):
        plot_df = plot_df[plot_df["explain_method"].astype(str) == "grad"].copy()

    order = {
        "random": 0,
        "oracle_support": 1,
        "random_contains_support": 1,
        "oracle_interaction": 2,
        "exclude_interaction": 3,
        "exclude_support": 4,
        "rf": 5,
        "rf_contains_support": 6,
        "rf_exclude_interaction": 7,
        "mutual_info": 8,
        "mi_contains_support": 9,
        "correlation": 10,
    }

    plot_df["mode_order"] = plot_df["screen_mode"].map(order).fillna(99)
    plot_df = plot_df.sort_values(["mode_order", "screen_mode", "top_m"])

    plot_df["label"] = plot_df.apply(lambda r: f"{r['screen_mode']}\nM={int(r['top_m'])}", axis=1)

    x = np.arange(len(plot_df))
    width = 0.35

    plt.figure(figsize=(max(10, len(plot_df) * 0.75), 5.5))
    plt.bar(x - width / 2, plot_df["variable_f1_mean"], width=width, label="Variable F1")
    if "interaction_f1_mean" in plot_df.columns:
        plt.bar(x + width / 2, plot_df["interaction_f1_mean"], width=width, label="Interaction F1")

    plt.ylim(0, 1.08)
    plt.ylabel("Mean F1")
    plt.xticks(x, plot_df["label"], rotation=55, ha="right")
    plt.title("Support-controlled screening ablation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=250)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--function", type=str, required=True)
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--test_samples", type=int, default=2048)
    parser.add_argument("--dimension", type=int, default=100)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--screen_repeats", type=int, default=1)

    parser.add_argument(
        "--screen_modes",
        type=str,
        nargs="+",
        default=["random", "oracle_support", "exclude_interaction", "rf"],
    )
    parser.add_argument("--top_m", type=int, nargs="+", default=[10, 20, 30])
    parser.add_argument("--rf_trees", type=int, default=300)

    parser.add_argument("--kan_width", type=int, default=8)
    parser.add_argument("--kan_grid", type=int, default=5)
    parser.add_argument("--kan_k", type=int, default=3)
    parser.add_argument("--kan_steps", type=int, default=50)
    parser.add_argument("--kan_lamb", type=float, default=0.0)

    parser.add_argument("--explain_methods", type=str, nargs="+", default=["grad", "perm"])
    parser.add_argument("--compute_interactions", action="store_true")
    parser.add_argument("--hessian_points", type=int, default=64)

    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--summary_out", type=str, default=None)
    parser.add_argument("--plot_out", type=str, default=None)

    args = parser.parse_args()

    all_rows: List[Dict] = []

    for seed in args.seeds:
        for repeat in range(args.screen_repeats):
            for mode in args.screen_modes:
                for m in args.top_m:
                    print(f"Running seed={seed}, repeat={repeat}, mode={mode}, top_m={m}")
                    rows = run_one(args, seed=seed, repeat=repeat, screen_mode=mode, top_m=m)
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
