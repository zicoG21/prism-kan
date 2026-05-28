from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

from src.data import make_synthetic


def canonical_pairs(pairs):
    return {tuple(sorted((int(i), int(j)))) for i, j in pairs}


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


def feature_grid(x: np.ndarray, grid_size: int) -> np.ndarray:
    qs = np.linspace(0.08, 0.92, int(grid_size))
    vals = np.quantile(x, qs)
    vals = np.unique(vals)
    if len(vals) < 2:
        lo, hi = float(np.min(x)), float(np.max(x))
        vals = np.linspace(lo, hi, int(grid_size))
    return vals.astype(np.float32)


def h_pair_score(
    model: HistGradientBoostingRegressor,
    background: np.ndarray,
    i: int,
    j: int,
    grid_i: np.ndarray,
    grid_j: np.ndarray,
    main_cache: dict[tuple[int, float], float],
    f0: float,
) -> float:
    gi = np.asarray(grid_i, dtype=np.float32)
    gj = np.asarray(grid_j, dtype=np.float32)
    fij = np.empty((len(gi), len(gj)), dtype=np.float64)
    for a_idx, a in enumerate(gi):
        for b_idx, b in enumerate(gj):
            Xp = background.copy()
            Xp[:, i] = float(a)
            Xp[:, j] = float(b)
            fij[a_idx, b_idx] = float(np.mean(model.predict(Xp)))
    fi = np.asarray([main_cache[(i, float(a))] for a in gi], dtype=np.float64)
    fj = np.asarray([main_cache[(j, float(b))] for b in gj], dtype=np.float64)
    h = fij - fi[:, None] - fj[None, :] + f0
    return float(np.mean(h * h))


def raw_product_corr_candidates(
    X: np.ndarray,
    y: np.ndarray,
    pairs: list[tuple[int, int]],
    top_k: int,
) -> tuple[list[tuple[int, int]], np.ndarray]:
    Xz = StandardScaler().fit_transform(X).astype(np.float32)
    yz = y.reshape(-1).astype(np.float64)
    yz = yz - yz.mean()
    yz_norm = float(np.linalg.norm(yz)) + 1e-12
    scores = np.empty(len(pairs), dtype=np.float64)
    for k, (i, j) in enumerate(pairs):
        z = (Xz[:, i] * Xz[:, j]).astype(np.float64)
        z = z - z.mean()
        scores[k] = abs(float(z @ yz)) / ((float(np.linalg.norm(z)) + 1e-12) * yz_norm)
    if top_k <= 0 or top_k >= len(pairs):
        return pairs, scores
    order = np.argsort(-scores)[: int(top_k)]
    return [pairs[int(idx)] for idx in order], scores


def run_one(args: argparse.Namespace, seed: int) -> dict:
    t0 = time.time()
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
    X_train = data["X_train"].astype(np.float32)
    y_train = data["y_train"].reshape(-1).astype(np.float32)
    X_score = data["X_test"].astype(np.float32)
    gt = data["ground_truth"]
    true_pairs = canonical_pairs(gt.interactions)
    endpoints = {v for pair in true_pairs for v in pair}

    model = HistGradientBoostingRegressor(
        max_iter=args.max_iter,
        learning_rate=args.learning_rate,
        max_leaf_nodes=args.max_leaf_nodes,
        l2_regularization=args.l2_regularization,
        random_state=seed,
    )
    model.fit(X_train, y_train)

    rng = np.random.default_rng(seed + 991)
    bg_idx = rng.choice(len(X_score), size=min(args.background, len(X_score)), replace=False)
    background = X_score[bg_idx].copy()
    f0 = float(np.mean(model.predict(background)))

    grids = [feature_grid(X_score[:, j], args.grid_size) for j in range(args.dimension)]
    main_cache: dict[tuple[int, float], float] = {}
    for j, grid in enumerate(grids):
        for val in grid:
            Xp = background.copy()
            Xp[:, j] = float(val)
            main_cache[(j, float(val))] = float(np.mean(model.predict(Xp)))

    all_pairs = list(itertools.combinations(range(args.dimension), 2))
    pairs, candidate_scores = raw_product_corr_candidates(
        X_train,
        y_train,
        all_pairs,
        args.candidate_pairs,
    )
    scores = np.empty(len(pairs), dtype=np.float64)
    for k, (i, j) in enumerate(pairs):
        scores[k] = h_pair_score(model, background, i, j, grids[i], grids[j], main_cache, f0)

    order = np.argsort(-scores)
    top_q = max(1, len(true_pairs))
    top_pairs = [tuple(sorted(pairs[int(idx)])) for idx in order[:top_q]]
    selected_pairs = set(top_pairs[: len(true_pairs)])
    precision, recall, f1 = f1_from_sets(selected_pairs, true_pairs)
    top_support = {v for pair in top_pairs[: args.top_pairs_for_support] for v in pair}
    true_pair_ranks = []
    true_pair_scores = []
    for pair in true_pairs:
        pair = tuple(sorted(pair))
        if pair in pairs:
            pair_idx = pairs.index(pair)
            true_pair_scores.append(scores[pair_idx])
            true_pair_ranks.append(int(np.where(order == pair_idx)[0][0]) + 1)
        else:
            true_pair_scores.append(np.nan)
            true_pair_ranks.append(np.inf)
    false_scores = [scores[k] for k, pair in enumerate(pairs) if tuple(sorted(pair)) not in true_pairs]

    return {
        "function": args.function,
        "method": "gbm_h_statistic",
        "seed": int(seed),
        "samples": int(args.samples),
        "test_samples": int(args.test_samples),
        "dimension": int(args.dimension),
        "noise": float(args.noise),
        "nuisance_correlation": float(args.nuisance_correlation),
        "n_correlated_proxies": int(args.n_correlated_proxies),
        "grid_size": int(args.grid_size),
        "background": int(len(background)),
        "max_iter": int(args.max_iter),
        "max_leaf_nodes": int(args.max_leaf_nodes),
        "num_pairs": len(pairs),
        "num_all_pairs": len(all_pairs),
        "candidate_pairs": int(args.candidate_pairs),
        "candidate_contains_true_pair": int(all(pair in set(pairs) for pair in true_pairs)) if true_pairs else np.nan,
        "selected_interactions": sorted(selected_pairs),
        "top_pair": top_pairs[0] if top_pairs else None,
        "endpoint_recall_at_top_pairs": len(top_support & endpoints) / len(endpoints) if endpoints else np.nan,
        "pair_retained_at_top_pairs": int(any(pair in set(top_pairs[: args.top_pairs_for_support]) for pair in true_pairs)) if true_pairs else np.nan,
        "top1_pair_accuracy": f1,
        "interaction_precision": precision,
        "interaction_recall": recall,
        "interaction_f1": f1,
        "true_interaction_rank_mean": float(np.mean(true_pair_ranks)) if true_pair_ranks else np.nan,
        "true_interaction_rank_worst": float(np.max(true_pair_ranks)) if true_pair_ranks else np.nan,
        "true_pair_score_mean": float(np.nanmean(true_pair_scores)) if true_pair_scores else np.nan,
        "max_false_pair_score": float(np.max(false_scores)) if false_scores else np.nan,
        "runtime_sec": float(time.time() - t0),
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
        "grid_size",
        "background",
        "max_iter",
        "max_leaf_nodes",
        "candidate_pairs",
    ]
    numeric_cols = [
        "endpoint_recall_at_top_pairs",
        "pair_retained_at_top_pairs",
        "top1_pair_accuracy",
        "interaction_f1",
        "true_interaction_rank_mean",
        "true_interaction_rank_worst",
        "true_pair_score_mean",
        "max_false_pair_score",
        "runtime_sec",
        "candidate_contains_true_pair",
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
    parser.add_argument("--grid_size", type=int, default=6)
    parser.add_argument("--background", type=int, default=96)
    parser.add_argument("--top_pairs_for_support", type=int, default=1)
    parser.add_argument("--max_iter", type=int, default=180)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--max_leaf_nodes", type=int, default=31)
    parser.add_argument("--l2_regularization", type=float, default=0.0)
    parser.add_argument("--candidate_pairs", type=int, default=250)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--out_dir", default="results/interaction_baselines/gbm_h_statistic_c025_d100")
    args = parser.parse_args()

    rows = []
    for n in args.samples:
        for seed in args.seeds:
            local_args = argparse.Namespace(**vars(args))
            local_args.samples = int(n)
            print(f"Running n={n}, seed={seed}", flush=True)
            rows.append(run_one(local_args, int(seed)))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    detail = pd.DataFrame(rows)
    detail.to_csv(out_dir / "gbm_h_statistic_detail.csv", index=False)
    summary = summarize(detail)
    summary.to_csv(out_dir / "gbm_h_statistic_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
