#!/usr/bin/env python3
"""Monte Carlo stability check for full-KAN functional-ANOVA pair ranking.

The main paper uses direct full-model functional ANOVA to ask whether a trained
full-dimensional KAN relies on the true interaction. Reviewers may worry that
all-pairs ranks at d=100 are sensitive to the particular anchor/background rows
used by the estimator. This script trains each full KAN once, then repeats the
ANOVA pair scorer with different anchor/background subsamples for the same
trained model.

The output answers a narrow question: conditional on the fitted model, does the
true-pair rank/margin change materially under Monte Carlo resampling of the
ANOVA estimator?
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import make_synthetic
from experiments.run_tuned_kan_recovery import batch_predict, canonical_pairs, mse_np, train_kan
from scripts.run_full_kan_pair_anova_probe import parse_seeds


Pair = tuple[int, int]


def train_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        width_hidden=int(args.width_hidden),
        grid=int(args.grid),
        k=int(args.k),
        steps=int(args.steps),
        lamb=float(args.lamb),
        opt=args.opt,
        update_grid=bool(args.update_grid),
        grid_update_num=int(args.grid_update_num),
        batch=int(args.batch),
    )


def rank_true_pair(pair_scores: dict[Pair, float], true_pairs: list[Pair]) -> dict:
    ranked = sorted(pair_scores.items(), key=lambda kv: (-float(kv[1]), kv[0]))
    true_pair = true_pairs[0]
    true_score = float(pair_scores[true_pair])
    false_scores = np.asarray([score for pair, score in ranked if pair not in true_pairs], dtype=float)
    max_false = float(np.max(false_scores)) if len(false_scores) else np.nan
    rank = int([pair for pair, _ in ranked].index(true_pair) + 1)
    top_pair, top_score = ranked[0]
    return {
        "true_pair_score": true_score,
        "max_false_pair_score": max_false,
        "true_minus_max_false": float(true_score - max_false),
        "true_pair_rank": rank,
        "true_pair_rank1": int(rank == 1),
        "top_pair": str(top_pair),
        "top_pair_score": float(top_score),
    }


def all_pair_anova_scores_resampled(
    model,
    X_np: np.ndarray,
    device: str,
    *,
    anchor_idx: np.ndarray,
    background_idx: np.ndarray,
    batch_size: int,
    pair_chunk_size: int,
) -> dict[Pair, float]:
    anchors = X_np[anchor_idx].copy()
    bg = X_np[background_idx].copy()
    n_points = int(len(anchors))
    n_bg = int(len(bg))
    d = int(bg.shape[1])
    pairs = list(itertools.combinations(range(d), 2))
    pair_arr = np.asarray(pairs, dtype=int)

    f_mean = float(np.mean(batch_predict(model, bg, device=device, batch_size=batch_size).reshape(-1)))
    accum = np.zeros(len(pairs), dtype=np.float64)
    if pair_chunk_size <= 0:
        pair_chunk_size = len(pairs)

    for row in anchors:
        main_rows_3d = np.broadcast_to(bg[None, :, :], (d, n_bg, d)).copy()
        main_idx = np.arange(d)
        main_rows_3d[main_idx, :, main_idx] = row[main_idx, None]
        main_rows = main_rows_3d.reshape(d * n_bg, d)
        main_pred = batch_predict(model, main_rows, device=device, batch_size=batch_size).reshape(d, n_bg)
        main_mean = np.mean(main_pred, axis=1)

        for start_pair in range(0, len(pairs), pair_chunk_size):
            stop_pair = min(start_pair + pair_chunk_size, len(pairs))
            chunk_arr = pair_arr[start_pair:stop_pair]
            chunk_len = len(chunk_arr)
            pair_rows_3d = np.broadcast_to(bg[None, :, :], (chunk_len, n_bg, d)).copy()
            chunk_idx = np.arange(chunk_len)
            pair_rows_3d[chunk_idx, :, chunk_arr[:, 0]] = row[chunk_arr[:, 0], None]
            pair_rows_3d[chunk_idx, :, chunk_arr[:, 1]] = row[chunk_arr[:, 1], None]
            pair_rows = pair_rows_3d.reshape(chunk_len * n_bg, d)
            pair_pred = batch_predict(model, pair_rows, device=device, batch_size=batch_size).reshape(chunk_len, n_bg)
            pair_mean = np.mean(pair_pred, axis=1)
            comps = pair_mean - main_mean[chunk_arr[:, 0]] - main_mean[chunk_arr[:, 1]] + f_mean
            accum[start_pair:stop_pair] += np.abs(comps)

    scores = accum / float(n_points)
    return {pair: float(score) for pair, score in zip(pairs, scores)}


def run_seed(args: argparse.Namespace, seed: int) -> list[dict]:
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
    y_train = data["y_train"].astype(np.float32)
    X_test = data["X_test"].astype(np.float32)
    y_test = data["y_test"].astype(np.float32)
    true_pairs = canonical_pairs(data["ground_truth"].interactions)

    t0 = time.time()
    model = train_kan(
        X_train,
        y_train,
        X_test,
        y_test,
        train_args(args),
        seed=seed,
        device=args.device,
    )
    train_pred = batch_predict(model, X_train, device=args.device, batch_size=args.batch_size)
    test_pred = batch_predict(model, X_test, device=args.device, batch_size=args.batch_size)
    train_mse = mse_np(train_pred, y_train)
    test_mse = mse_np(test_pred, y_test)
    train_runtime = time.time() - t0

    rows: list[dict] = []
    rng = np.random.default_rng(args.mc_seed_base + seed * 1009)
    for rep in range(args.mc_repeats):
        if args.replace:
            anchor_idx = rng.choice(len(X_test), size=min(args.anova_points, len(X_test)), replace=True)
            bg_idx = rng.choice(len(X_test), size=min(args.anova_background, len(X_test)), replace=True)
        else:
            anchor_idx = rng.choice(len(X_test), size=min(args.anova_points, len(X_test)), replace=False)
            bg_idx = rng.choice(len(X_test), size=min(args.anova_background, len(X_test)), replace=False)
        t_rep = time.time()
        scores = all_pair_anova_scores_resampled(
            model,
            X_test,
            device=args.device,
            anchor_idx=np.asarray(anchor_idx, dtype=int),
            background_idx=np.asarray(bg_idx, dtype=int),
            batch_size=args.batch_size,
            pair_chunk_size=args.pair_chunk_size,
        )
        stat = rank_true_pair(scores, true_pairs)
        rows.append(
            {
                "function": args.function,
                "seed": int(seed),
                "mc_rep": int(rep),
                "samples": int(args.samples),
                "test_samples": int(args.test_samples),
                "dimension": int(args.dimension),
                "noise": float(args.noise),
                "update_grid": int(bool(args.update_grid)),
                "width_hidden": int(args.width_hidden),
                "grid": int(args.grid),
                "lamb": float(args.lamb),
                "steps": int(args.steps),
                "anova_points": int(min(args.anova_points, len(X_test))),
                "anova_background": int(min(args.anova_background, len(X_test))),
                "candidate_pairs": int(args.dimension * (args.dimension - 1) // 2),
                "train_mse": float(train_mse),
                "test_mse": float(test_mse),
                "train_runtime_sec": float(train_runtime),
                "score_runtime_sec": float(time.time() - t_rep),
                **stat,
            }
        )
    return rows


def summarize(detail: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_seed = (
        detail.groupby(["function", "samples", "dimension", "noise", "update_grid", "width_hidden", "seed"], dropna=False)
        .agg(
            test_mse=("test_mse", "first"),
            rank1_mean=("true_pair_rank1", "mean"),
            rank_mean=("true_pair_rank", "mean"),
            rank_std=("true_pair_rank", "std"),
            margin_mean=("true_minus_max_false", "mean"),
            margin_std=("true_minus_max_false", "std"),
            score_repeats=("mc_rep", "count"),
        )
        .reset_index()
    )
    overall = (
        per_seed.groupby(["function", "samples", "dimension", "noise", "update_grid", "width_hidden"], dropna=False)
        .agg(
            seeds=("seed", "count"),
            mean_test_mse=("test_mse", "mean"),
            mean_rank1_over_mc=("rank1_mean", "mean"),
            seeds_always_rank1=("rank1_mean", lambda s: int(np.sum(np.asarray(s) == 1.0))),
            seeds_never_rank1=("rank1_mean", lambda s: int(np.sum(np.asarray(s) == 0.0))),
            mean_rank=("rank_mean", "mean"),
            mean_rank_std_within_model=("rank_std", "mean"),
            mean_margin=("margin_mean", "mean"),
            mean_margin_std_within_model=("margin_std", "mean"),
        )
        .reset_index()
    )
    return per_seed, overall


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--function", default="core_interaction_c025")
    parser.add_argument("--samples", type=int, default=768)
    parser.add_argument("--test-samples", type=int, default=2048)
    parser.add_argument("--dimension", type=int, default=100)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--nuisance-correlation", type=float, default=0.0)
    parser.add_argument("--n-correlated-proxies", type=int, default=0)
    parser.add_argument("--seeds", default="0-9")
    parser.add_argument("--mc-repeats", type=int, default=5)
    parser.add_argument("--mc-seed-base", type=int, default=99173)
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--width-hidden", type=int, default=16)
    parser.add_argument("--grid", type=int, default=5)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--steps", type=int, default=75)
    parser.add_argument("--lamb", type=float, default=1e-3)
    parser.add_argument("--opt", default="LBFGS")
    parser.add_argument("--update-grid", action="store_true")
    parser.add_argument("--grid-update-num", type=int, default=5)
    parser.add_argument("--batch", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--pair-chunk-size", type=int, default=1000)
    parser.add_argument("--anova-points", type=int, default=16)
    parser.add_argument("--anova-background", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-dir", type=Path, default=Path("results/revision/anova_estimator_stability/smoke"))
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    detail_path = args.out_dir / "anova_estimator_stability_detail.csv"
    per_seed_path = args.out_dir / "anova_estimator_stability_per_seed.csv"
    summary_path = args.out_dir / "anova_estimator_stability_summary.csv"

    rows: list[dict] = []
    completed: set[int] = set()
    if detail_path.exists() and not args.no_resume:
        existing = pd.read_csv(detail_path)
        if len(existing):
            rows = existing.to_dict("records")
            reps = existing.groupby("seed")["mc_rep"].nunique()
            completed = {int(seed) for seed, count in reps.items() if int(count) >= int(args.mc_repeats)}
            print(f"[resume] loaded {len(existing)} rows; completed seeds={sorted(completed)}", flush=True)

    for seed in parse_seeds(args.seeds):
        if int(seed) in completed:
            print(f"[resume] skip completed seed={seed}", flush=True)
            continue
        print(f"Running ANOVA estimator stability seed={seed}", flush=True)
        rows.extend(run_seed(args, seed))
        detail = pd.DataFrame(rows)
        per_seed, overall = summarize(detail)
        detail.to_csv(detail_path, index=False)
        per_seed.to_csv(per_seed_path, index=False)
        overall.to_csv(summary_path, index=False)

    detail = pd.DataFrame(rows)
    per_seed, overall = summarize(detail)
    detail.to_csv(detail_path, index=False)
    per_seed.to_csv(per_seed_path, index=False)
    overall.to_csv(summary_path, index=False)
    print(overall.to_string(index=False))


if __name__ == "__main__":
    main()
