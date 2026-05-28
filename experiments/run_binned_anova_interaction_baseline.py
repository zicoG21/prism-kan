from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.data import make_synthetic


def canonical_pairs(pairs):
    return {tuple(sorted((int(i), int(j)))) for i, j in pairs}


def bin_indices(x: np.ndarray, num_bins: int) -> np.ndarray:
    edges = np.quantile(x, np.linspace(0.0, 1.0, num_bins + 1))
    edges = np.unique(edges)
    if len(edges) <= 2:
        return np.zeros_like(x, dtype=np.int64)
    return np.clip(np.searchsorted(edges[1:-1], x, side="right"), 0, len(edges) - 2)


def binned_pair_score(xi: np.ndarray, xj: np.ndarray, y: np.ndarray, num_bins: int) -> float:
    bi = bin_indices(xi, num_bins)
    bj = bin_indices(xj, num_bins)
    ni = int(bi.max()) + 1
    nj = int(bj.max()) + 1
    global_mean = float(y.mean())

    count_i = np.bincount(bi, minlength=ni).astype(float)
    count_j = np.bincount(bj, minlength=nj).astype(float)
    sum_i = np.bincount(bi, weights=y, minlength=ni).astype(float)
    sum_j = np.bincount(bj, weights=y, minlength=nj).astype(float)
    mean_i = np.divide(sum_i, count_i, out=np.full_like(sum_i, global_mean), where=count_i > 0)
    mean_j = np.divide(sum_j, count_j, out=np.full_like(sum_j, global_mean), where=count_j > 0)

    flat = bi * nj + bj
    count_ij = np.bincount(flat, minlength=ni * nj).astype(float).reshape(ni, nj)
    sum_ij = np.bincount(flat, weights=y, minlength=ni * nj).astype(float).reshape(ni, nj)
    mean_ij = np.divide(sum_ij, count_ij, out=np.full_like(sum_ij, global_mean), where=count_ij > 0)
    h = mean_ij - mean_i[:, None] - mean_j[None, :] + global_mean
    weights = count_ij / max(float(len(y)), 1.0)
    return float(np.sum(weights * h * h))


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
    X = data["X_train"].astype(np.float64)
    y = data["y_train"].reshape(-1).astype(np.float64)
    gt = data["ground_truth"]
    true_pairs = canonical_pairs(gt.interactions)
    pairs = list(itertools.combinations(range(args.dimension), 2))
    scores = np.empty(len(pairs), dtype=np.float64)
    for k, (i, j) in enumerate(pairs):
        scores[k] = binned_pair_score(X[:, i], X[:, j], y, args.num_bins)

    order = np.argsort(-scores)
    top_pair = tuple(sorted(pairs[int(order[0])])) if len(order) else ()
    top_pairs = {tuple(sorted(pairs[int(i)])) for i in order[: max(1, len(true_pairs))]}
    true_pair_score = float(np.mean([scores[pairs.index(p)] for p in true_pairs])) if true_pairs else np.nan
    max_false_score = float(np.max([scores[k] for k, p in enumerate(pairs) if tuple(sorted(p)) not in true_pairs]))
    true_rank = np.nan
    if true_pairs:
        true_pair = next(iter(true_pairs))
        true_rank = int(np.where(order == pairs.index(true_pair))[0][0]) + 1

    return {
        "function": args.function,
        "method": "binned_functional_anova",
        "seed": seed,
        "samples": args.samples,
        "test_samples": args.test_samples,
        "dimension": args.dimension,
        "noise": args.noise,
        "nuisance_correlation": args.nuisance_correlation,
        "n_correlated_proxies": args.n_correlated_proxies,
        "num_bins": args.num_bins,
        "top_pair": str(top_pair),
        "selected_interactions": str(sorted(top_pairs)),
        "top1_pair_accuracy": int(top_pairs == true_pairs),
        "interaction_f1": int(top_pairs == true_pairs),
        "true_interaction_rank": true_rank,
        "true_pair_score": true_pair_score,
        "max_false_score": max_false_score,
        "true_pair_margin": true_pair_score - max_false_score,
    }


def summarize(detail: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "function",
        "method",
        "samples",
        "test_samples",
        "dimension",
        "noise",
        "nuisance_correlation",
        "n_correlated_proxies",
        "num_bins",
    ]
    numeric_cols = [
        "top1_pair_accuracy",
        "interaction_f1",
        "true_interaction_rank",
        "true_pair_score",
        "max_false_score",
        "true_pair_margin",
    ]
    out = detail.groupby(group_cols, dropna=False)[numeric_cols].agg(["mean", "std"]).reset_index()
    out.columns = ["_".join(str(v) for v in col if v != "").rstrip("_") for col in out.columns]
    counts = detail.groupby(group_cols, dropna=False).size().reset_index(name="num_runs")
    successes = detail.groupby(group_cols, dropna=False)["top1_pair_accuracy"].sum().reset_index(name="top1_successes")
    return out.merge(counts, on=group_cols, how="left").merge(successes, on=group_cols, how="left")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--function", default="core_interaction_c025")
    parser.add_argument("--samples", type=int, nargs="+", default=[512, 1024])
    parser.add_argument("--test_samples", type=int, default=2048)
    parser.add_argument("--dimension", type=int, default=100)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--nuisance_correlation", type=float, default=0.0)
    parser.add_argument("--n_correlated_proxies", type=int, default=0)
    parser.add_argument("--num_bins", type=int, default=8)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--out_dir", default="results/interaction_baselines/binned_anova_c025_d100")
    args = parser.parse_args()

    rows = []
    for n in args.samples:
        for seed in args.seeds:
            local_args = argparse.Namespace(**vars(args))
            local_args.samples = int(n)
            print(f"Running n={n}, seed={seed}")
            rows.append(run_one(local_args, int(seed)))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    detail = pd.DataFrame(rows)
    detail.to_csv(out_dir / "binned_anova_interaction_detail.csv", index=False)
    summary = summarize(detail)
    summary.to_csv(out_dir / "binned_anova_interaction_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
