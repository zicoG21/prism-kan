from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

from src.data import make_synthetic


def canonical_pairs(pairs):
    return {tuple(sorted((int(i), int(j)))) for i, j in pairs}


def pair_design(X: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]]]:
    pairs = list(itertools.combinations(range(X.shape[1]), 2))
    Z = np.empty((X.shape[0], len(pairs)), dtype=np.float32)
    for k, (i, j) in enumerate(pairs):
        Z[:, k] = X[:, i] * X[:, j]
    return Z, pairs


def f1_from_sets(pred: set, true: set) -> tuple[float, float, float]:
    if not true:
        return np.nan, np.nan, np.nan
    if not pred:
        return 0.0, 0.0, 0.0
    tp = len(pred & true)
    precision = tp / len(pred)
    recall = tp / len(true)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def run_one(args: argparse.Namespace, seed: int) -> dict:
    data = make_synthetic(
        function_name=args.function,
        n_train=args.samples,
        n_test=args.test_samples,
        d=args.dimension,
        noise=args.noise,
        seed=seed,
        standardize_target=True,
        nuisance_correlation=args.nuisance_correlation,
        n_correlated_proxies=args.n_correlated_proxies,
    )
    X = data["X_train"].astype(np.float32)
    y = data["y_train"].reshape(-1)
    gt = data["ground_truth"]
    true_vars = set(int(v) for v in gt.active_variables)
    true_pairs = canonical_pairs(gt.interactions)
    endpoints = {v for pair in true_pairs for v in pair}

    scaler_raw = StandardScaler()
    Xz = scaler_raw.fit_transform(X).astype(np.float32)
    Z, pairs = pair_design(Xz)
    design = np.concatenate([Xz, Z], axis=1)
    design = StandardScaler().fit_transform(design)

    model = LassoCV(cv=args.cv, random_state=seed, max_iter=args.max_iter, n_jobs=-1)
    model.fit(design, y)
    coef = np.asarray(model.coef_, dtype=float)
    raw_scores = np.abs(coef[: args.dimension])
    pair_scores = np.abs(coef[args.dimension :])

    selected_vars = set(int(i) for i in np.argsort(-raw_scores)[: len(true_vars)])
    selected_support = set(int(i) for i in np.argsort(-raw_scores)[: args.top_m])
    top_pairs = [pairs[int(i)] for i in np.argsort(-pair_scores)[: max(1, len(true_pairs))]]
    selected_pairs = set(tuple(sorted(p)) for p in top_pairs[: len(true_pairs)])
    precision, recall, f1 = f1_from_sets(selected_pairs, true_pairs)
    var_precision, var_recall, var_f1 = f1_from_sets(selected_vars, true_vars)

    return {
        "function": args.function,
        "seed": seed,
        "samples": args.samples,
        "test_samples": args.test_samples,
        "dimension": args.dimension,
        "noise": args.noise,
        "nuisance_correlation": args.nuisance_correlation,
        "n_correlated_proxies": args.n_correlated_proxies,
        "top_m": args.top_m,
        "method": "pair_feature_lasso",
        "alpha": float(model.alpha_),
        "num_pair_features": len(pairs),
        "selected_variables": sorted(selected_vars),
        "selected_support": sorted(selected_support),
        "selected_interactions": sorted(selected_pairs),
        "variable_precision": var_precision,
        "variable_recall": var_recall,
        "variable_f1": var_f1,
        "endpoint_recall_at_m": len(selected_support & endpoints) / len(endpoints) if endpoints else np.nan,
        "pair_retained_at_m": int(all(i in selected_support and j in selected_support for i, j in true_pairs)) if true_pairs else np.nan,
        "top1_pair_accuracy": f1,
        "interaction_precision": precision,
        "interaction_recall": recall,
        "interaction_f1": f1,
        "top_pair_score": float(np.max(pair_scores)) if len(pair_scores) else np.nan,
        "true_pair_score_mean": float(np.mean([pair_scores[pairs.index(p)] for p in true_pairs])) if true_pairs else np.nan,
    }


def summarize(detail: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "function",
        "samples",
        "test_samples",
        "dimension",
        "noise",
        "nuisance_correlation",
        "n_correlated_proxies",
        "top_m",
        "method",
    ]
    numeric_cols = [
        "variable_f1",
        "endpoint_recall_at_m",
        "pair_retained_at_m",
        "top1_pair_accuracy",
        "interaction_f1",
        "alpha",
        "num_pair_features",
        "top_pair_score",
        "true_pair_score_mean",
    ]
    out = detail.groupby(group_cols, dropna=False)[numeric_cols].agg(["mean", "std"]).reset_index()
    out.columns = ["_".join(str(v) for v in col if v != "").rstrip("_") for col in out.columns]
    counts = detail.groupby(group_cols, dropna=False).size().reset_index(name="num_runs")
    return out.merge(counts, on=group_cols, how="left")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--function", default="core_interaction_c025")
    parser.add_argument("--samples", type=int, nargs="+", default=[512, 1024])
    parser.add_argument("--test_samples", type=int, default=2048)
    parser.add_argument("--dimension", type=int, default=100)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--nuisance_correlation", type=float, default=0.0)
    parser.add_argument("--n_correlated_proxies", type=int, default=0)
    parser.add_argument("--top_m", type=int, default=4)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--max_iter", type=int, default=5000)
    parser.add_argument("--out_dir", default="results/interaction_baselines/pair_feature_lasso_c025_d100")
    args = parser.parse_args()

    rows = []
    for n in args.samples:
        args.samples = int(n)
        for seed in args.seeds:
            print(f"Running n={n}, seed={seed}")
            rows.append(run_one(args, seed))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    detail = pd.DataFrame(rows)
    detail.to_csv(out_dir / "pair_feature_lasso_detail.csv", index=False)
    summary = summarize(detail)
    summary.to_csv(out_dir / "pair_feature_lasso_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
