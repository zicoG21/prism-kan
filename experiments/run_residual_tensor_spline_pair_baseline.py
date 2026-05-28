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
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import SplineTransformer, StandardScaler

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


def fit_spline_blocks(X: np.ndarray, n_knots: int, degree: int) -> list[SplineTransformer]:
    blocks = []
    for j in range(X.shape[1]):
        spline = SplineTransformer(
            n_knots=int(n_knots),
            degree=int(degree),
            include_bias=False,
            extrapolation="continue",
        )
        spline.fit(X[:, [j]])
        blocks.append(spline)
    return blocks


def transform_spline_blocks(X: np.ndarray, blocks: list[SplineTransformer]) -> list[np.ndarray]:
    return [spline.transform(X[:, [j]]).astype(np.float32) for j, spline in enumerate(blocks)]


def concatenate_blocks(blocks: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(blocks, axis=1).astype(np.float32)


def crossfit_additive_residual(
    Xz: np.ndarray,
    y: np.ndarray,
    n_knots: int,
    degree: int,
    alphas: list[float],
    fixed_alpha: float | None,
    folds: int,
    seed: int,
) -> tuple[np.ndarray, float]:
    residual = np.empty_like(y, dtype=np.float32)
    selected_alphas = []
    splitter = KFold(n_splits=int(folds), shuffle=True, random_state=int(seed) + 1701)
    for train_idx, score_idx in splitter.split(Xz):
        train_blocks = fit_spline_blocks(Xz[train_idx], n_knots, degree)
        X_train_spline = concatenate_blocks(transform_spline_blocks(Xz[train_idx], train_blocks))
        X_score_spline = concatenate_blocks(transform_spline_blocks(Xz[score_idx], train_blocks))
        model = Ridge(alpha=float(fixed_alpha)) if fixed_alpha is not None else RidgeCV(alphas=np.asarray(alphas, dtype=float))
        model.fit(X_train_spline, y[train_idx])
        residual[score_idx] = (y[score_idx] - model.predict(X_score_spline).reshape(-1)).astype(np.float32)
        selected_alphas.append(float(fixed_alpha) if fixed_alpha is not None else float(model.alpha_))
    return residual, float(np.median(selected_alphas))


def additive_residual(
    Xz: np.ndarray,
    y: np.ndarray,
    n_knots: int,
    degree: int,
    alphas: list[float],
    fixed_alpha: float | None,
) -> tuple[np.ndarray, float, list[SplineTransformer]]:
    blocks = fit_spline_blocks(Xz, n_knots, degree)
    design = concatenate_blocks(transform_spline_blocks(Xz, blocks))
    model = Ridge(alpha=float(fixed_alpha)) if fixed_alpha is not None else RidgeCV(alphas=np.asarray(alphas, dtype=float))
    model.fit(design, y)
    residual = (y - model.predict(design).reshape(-1)).astype(np.float32)
    alpha = float(fixed_alpha) if fixed_alpha is not None else float(model.alpha_)
    return residual, alpha, blocks


def normalized_basis_blocks(Xz: np.ndarray, blocks: list[SplineTransformer]) -> list[np.ndarray]:
    raw_blocks = transform_spline_blocks(Xz, blocks)
    out = []
    for block in raw_blocks:
        z = block.astype(np.float64)
        z = z - z.mean(axis=0, keepdims=True)
        norm = np.linalg.norm(z, axis=0, keepdims=True)
        z = z / (norm + 1e-12)
        out.append(z.astype(np.float32))
    return out


def tensor_spline_pair_scores(
    basis_blocks: list[np.ndarray],
    residual: np.ndarray,
    pairs: list[tuple[int, int]],
) -> np.ndarray:
    r = residual.reshape(-1).astype(np.float64)
    r = r - r.mean()
    r = r / (np.linalg.norm(r) + 1e-12)
    scores = np.empty(len(pairs), dtype=np.float64)
    for k, (i, j) in enumerate(pairs):
        bi = basis_blocks[i].astype(np.float64)
        bj = basis_blocks[j].astype(np.float64)
        # Frobenius norm of the residual-weighted tensor-product projection.
        cross = bi.T @ (r[:, None] * bj)
        scores[k] = float(np.linalg.norm(cross, ord="fro"))
    return scores


def run_one(args: argparse.Namespace, function: str, samples: int, seed: int) -> dict:
    t0 = time.time()
    data = make_synthetic(
        function_name=function,
        n_train=samples,
        n_test=args.test_samples,
        d=args.dimension,
        noise=args.noise,
        seed=seed,
        standardize_target=True,
        nuisance_correlation=args.nuisance_correlation,
        n_correlated_proxies=args.n_correlated_proxies,
    )
    X = data["X_train"].astype(np.float32)
    y = data["y_train"].reshape(-1).astype(np.float32)
    gt = data["ground_truth"]
    true_pairs = canonical_pairs(gt.interactions)
    endpoints = {v for pair in true_pairs for v in pair}
    Xz = StandardScaler().fit_transform(X).astype(np.float32)

    if args.crossfit_folds > 1:
        residual, ridge_alpha = crossfit_additive_residual(
            Xz,
            y,
            args.n_knots,
            args.degree,
            args.alphas,
            args.fixed_alpha,
            args.crossfit_folds,
            seed,
        )
        blocks = fit_spline_blocks(Xz, args.n_knots, args.degree)
        residual_source = f"{int(args.crossfit_folds)}fold_crossfit"
    else:
        residual, ridge_alpha, blocks = additive_residual(
            Xz,
            y,
            args.n_knots,
            args.degree,
            args.alphas,
            args.fixed_alpha,
        )
        residual_source = "training_residual"

    basis_blocks = normalized_basis_blocks(Xz, blocks)
    pairs = list(itertools.combinations(range(args.dimension), 2))
    pair_scores = tensor_spline_pair_scores(basis_blocks, residual, pairs)
    order = np.argsort(-pair_scores)
    top_q = max(1, len(true_pairs))
    top_pairs = [tuple(sorted(pairs[int(i)])) for i in order[:top_q]]
    selected_pairs = set(top_pairs[: len(true_pairs)])
    precision, recall, f1 = f1_from_sets(selected_pairs, true_pairs)

    true_pair_ranks = []
    true_pair_scores = []
    pair_index = {pair: idx for idx, pair in enumerate(pairs)}
    rank_index = {int(pair_idx): rank + 1 for rank, pair_idx in enumerate(order)}
    for pair in true_pairs:
        idx = pair_index[pair]
        true_pair_scores.append(float(pair_scores[idx]))
        true_pair_ranks.append(int(rank_index[idx]))
    top_support = {v for pair in top_pairs[: args.top_pairs_for_support] for v in pair}
    false_scores = [pair_scores[k] for k, p in enumerate(pairs) if tuple(sorted(p)) not in true_pairs]

    return {
        "function": function,
        "method": "residual_tensor_spline_pair_screen",
        "seed": int(seed),
        "samples": int(samples),
        "test_samples": int(args.test_samples),
        "dimension": int(args.dimension),
        "noise": float(args.noise),
        "nuisance_correlation": float(args.nuisance_correlation),
        "n_correlated_proxies": int(args.n_correlated_proxies),
        "n_knots": int(args.n_knots),
        "degree": int(args.degree),
        "crossfit_folds": int(args.crossfit_folds),
        "residual_source": residual_source,
        "ridge_alpha": float(ridge_alpha),
        "num_pair_features": len(pairs),
        "selected_interactions": sorted(selected_pairs),
        "top_pair": top_pairs[0] if top_pairs else None,
        "top_pairs_for_support": int(args.top_pairs_for_support),
        "endpoint_recall_at_top_pairs": len(top_support & endpoints) / len(endpoints) if endpoints else np.nan,
        "pair_retained_at_top_pairs": int(any(pair in set(top_pairs[: args.top_pairs_for_support]) for pair in true_pairs)) if true_pairs else np.nan,
        "top1_pair_accuracy": f1,
        "interaction_precision": precision,
        "interaction_recall": recall,
        "interaction_f1": f1,
        "true_interaction_rank_mean": float(np.mean(true_pair_ranks)) if true_pair_ranks else np.nan,
        "true_interaction_rank_worst": float(np.max(true_pair_ranks)) if true_pair_ranks else np.nan,
        "true_pair_score_mean": float(np.mean(true_pair_scores)) if true_pair_scores else np.nan,
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
        "n_knots",
        "degree",
        "crossfit_folds",
        "residual_source",
        "top_pairs_for_support",
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
    ]
    out = detail.groupby(group_cols, dropna=False)[numeric_cols].agg(["mean", "std"]).reset_index()
    out.columns = ["_".join(str(v) for v in col if v != "").rstrip("_") for col in out.columns]
    counts = detail.groupby(group_cols, dropna=False).size().reset_index(name="num_runs")
    successes = detail.groupby(group_cols, dropna=False)["top1_pair_accuracy"].sum().reset_index(name="top1_successes")
    return out.merge(counts, on=group_cols, how="left").merge(successes, on=group_cols, how="left")


def main() -> None:
    parser = argparse.ArgumentParser(description="Residual tensor-spline interaction screen.")
    parser.add_argument("--function", default=None, help="Single function name. Deprecated in favor of --functions.")
    parser.add_argument("--functions", nargs="+", default=None, help="One or more synthetic function names.")
    parser.add_argument("--samples", type=int, nargs="+", default=[512, 1024])
    parser.add_argument("--test_samples", type=int, default=2048)
    parser.add_argument("--dimension", type=int, default=100)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--nuisance_correlation", type=float, default=0.0)
    parser.add_argument("--n_correlated_proxies", type=int, default=0)
    parser.add_argument("--n_knots", type=int, default=5)
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--crossfit_folds", type=int, default=1)
    parser.add_argument("--top_pairs_for_support", type=int, default=1)
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.01, 0.1, 1.0, 10.0, 100.0])
    parser.add_argument("--fixed_alpha", type=float, default=None, help="Use fixed Ridge alpha instead of RidgeCV.")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--out_dir", default="results/interaction_baselines/residual_tensor_spline_pair_screen")
    args = parser.parse_args()

    functions = args.functions if args.functions is not None else [args.function or "core_interaction_c025"]
    rows = []
    for fn in functions:
        for n in args.samples:
            for seed in args.seeds:
                print(f"Running function={fn}, n={n}, seed={seed}", flush=True)
                rows.append(run_one(args, str(fn), int(n), int(seed)))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    detail = pd.DataFrame(rows)
    detail.to_csv(out_dir / "residual_tensor_spline_pair_screen_detail.csv", index=False)
    summary = summarize(detail)
    summary.to_csv(out_dir / "residual_tensor_spline_pair_screen_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
