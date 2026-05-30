#!/usr/bin/env python3
"""Probe whether a full-dimensional KAN functionally contains the true pair.

The paper's support audit asks whether KAN-native explanation scores surface
the interaction endpoints.  Reviewers also asked a distinct question: did the
full-dimensional KAN learn the interaction at all?  This script trains full KANs
and scores a small candidate set of pairs by functional ANOVA directly on the
full model, without refitting to a selected support.

By default the candidate set contains the true interaction, the additive-main
effect pair (0, 1), and pairs among top feature/edge-score variables.  For
moderate dimensions, ``--pair-mode all`` scores every pair and is the stronger
review-facing check.
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import make_synthetic
from experiments.run_kan_native_innovation_loop import safe_feature_score, safe_edge_path_scores
from experiments.run_tuned_kan_recovery import batch_predict, canonical_pairs, mse_np, train_kan


Pair = tuple[int, int]


def parse_seeds(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return sorted(dict.fromkeys(out))


def top_indices(scores: np.ndarray, k: int) -> list[int]:
    order = sorted(range(len(scores)), key=lambda j: (-float(scores[j]), int(j)))
    return [int(j) for j in order[: min(k, len(order))]]


def rank_desc(scores: np.ndarray, idx: int) -> int:
    order = sorted(range(len(scores)), key=lambda j: (-float(scores[j]), int(j)))
    return int(order.index(int(idx)) + 1)


def canonical_pair(pair: Sequence[int]) -> Pair:
    i, j = int(pair[0]), int(pair[1])
    return (i, j) if i < j else (j, i)


def candidate_pairs_from_scores(
    feature_scores: np.ndarray,
    edge_scores: np.ndarray,
    true_pairs: Sequence[Pair],
    top_k: int,
    top_edge_k: int,
) -> list[Pair]:
    pairs: set[Pair] = {canonical_pair(pair) for pair in true_pairs}
    if len(feature_scores) >= 2:
        pairs.add((0, 1))

    feature_vars = top_indices(feature_scores, top_k)
    edge_vars = top_indices(edge_scores, top_edge_k)
    for i, j in itertools.combinations(sorted(set(feature_vars + edge_vars)), 2):
        pairs.add(canonical_pair((i, j)))
    return sorted(pairs)


def candidate_anova_pair_scores(
    model,
    X_np: np.ndarray,
    pairs: Iterable[Pair],
    device: str,
    *,
    points: int,
    background: int,
    batch_size: int,
) -> dict[Pair, float]:
    n_points = min(points, len(X_np))
    n_bg = min(background, len(X_np))
    anchors = X_np[:n_points].copy()
    bg = X_np[:n_bg].copy()
    f_mean = float(np.mean(batch_predict(model, bg, device=device, batch_size=batch_size).reshape(-1)))

    out: dict[Pair, float] = {}
    for i, j in pairs:
        comps = []
        for row in anchors:
            Xij = bg.copy()
            Xi = bg.copy()
            Xj = bg.copy()
            Xij[:, i] = row[i]
            Xij[:, j] = row[j]
            Xi[:, i] = row[i]
            Xj[:, j] = row[j]
            fij = float(np.mean(batch_predict(model, Xij, device=device, batch_size=batch_size).reshape(-1)))
            fi = float(np.mean(batch_predict(model, Xi, device=device, batch_size=batch_size).reshape(-1)))
            fj = float(np.mean(batch_predict(model, Xj, device=device, batch_size=batch_size).reshape(-1)))
            comps.append(fij - fi - fj + f_mean)
        out[(int(i), int(j))] = float(np.mean(np.abs(np.asarray(comps, dtype=float))))
    return out


def all_pair_anova_pair_scores(
    model,
    X_np: np.ndarray,
    device: str,
    *,
    points: int,
    background: int,
    batch_size: int,
    pair_chunk_size: int,
) -> dict[Pair, float]:
    """Functional-ANOVA pair scores for all pairs using batched predictions.

    The naive estimator calls the model separately for each pair and anchor.
    This version batches pair interventions for one anchor at a time and
    vectorizes the intervention construction.  It is intended for d=100-style
    checks where all d(d-1)/2 pairs are feasible.
    """

    n_points = min(points, len(X_np))
    n_bg = min(background, len(X_np))
    anchors = X_np[:n_points].copy()
    bg = X_np[:n_bg].copy()
    d = int(bg.shape[1])
    pairs = list(itertools.combinations(range(d), 2))
    pair_arr = np.asarray(pairs, dtype=int)

    f_mean = float(np.mean(batch_predict(model, bg, device=device, batch_size=batch_size).reshape(-1)))
    accum = np.zeros(len(pairs), dtype=np.float64)
    pair_chunk_size = int(pair_chunk_size)
    if pair_chunk_size <= 0:
        pair_chunk_size = len(pairs)

    for row in anchors:
        # Main effects for all variables under the same anchor row.
        main_rows_3d = np.broadcast_to(bg[None, :, :], (d, n_bg, d)).copy()
        main_idx = np.arange(d)
        main_rows_3d[main_idx, :, main_idx] = row[main_idx, None]
        main_rows = main_rows_3d.reshape(d * n_bg, d)
        main_pred = batch_predict(model, main_rows, device=device, batch_size=batch_size).reshape(d, n_bg)
        main_mean = np.mean(main_pred, axis=1)

        # Pair effects for all pairs under this anchor row.
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
    X_train = data["X_train"].astype(np.float32)
    y_train = data["y_train"].astype(np.float32)
    X_test = data["X_test"].astype(np.float32)
    y_test = data["y_test"].astype(np.float32)
    true_pairs = canonical_pairs(data["ground_truth"].interactions)
    true_pair = true_pairs[0] if true_pairs else None

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

    feature_scores = safe_feature_score(model, args.dimension)
    edge_scores, _, endpoint_mass = safe_edge_path_scores(model, args.dimension)
    hybrid_scores = feature_scores + edge_scores + endpoint_mass

    if args.pair_mode == "all":
        num_all_pairs = args.dimension * (args.dimension - 1) // 2
        if num_all_pairs > args.max_all_pairs:
            raise ValueError(
                f"--pair-mode all would score {num_all_pairs} pairs; "
                f"increase --max-all-pairs if this is intentional."
            )
        pairs = list(itertools.combinations(range(args.dimension), 2))
        pair_scores = all_pair_anova_pair_scores(
            model,
            X_test,
            device=args.device,
            points=args.anova_points,
            background=args.anova_background,
            batch_size=args.batch_size,
            pair_chunk_size=args.pair_chunk_size,
        )
    else:
        pairs = candidate_pairs_from_scores(
            feature_scores=feature_scores,
            edge_scores=edge_scores,
            true_pairs=true_pairs,
            top_k=args.top_feature_vars,
            top_edge_k=args.top_edge_vars,
        )
        pair_scores = candidate_anova_pair_scores(
            model,
            X_test,
            pairs,
            device=args.device,
            points=args.anova_points,
            background=args.anova_background,
            batch_size=args.batch_size,
        )
    ranked_pairs = sorted(pair_scores.items(), key=lambda kv: (-float(kv[1]), kv[0]))

    true_score = float(pair_scores.get(true_pair, np.nan)) if true_pair is not None else np.nan
    true_rank = (
        int([p for p, _ in ranked_pairs].index(true_pair) + 1)
        if true_pair is not None and true_pair in dict(ranked_pairs)
        else -1
    )
    top_pair, top_score = ranked_pairs[0] if ranked_pairs else ((-1, -1), np.nan)
    false_scores = np.asarray([score for pair, score in ranked_pairs if pair not in true_pairs], dtype=float)
    max_false = float(np.max(false_scores)) if len(false_scores) else np.nan
    false_p95 = float(np.quantile(false_scores, 0.95)) if len(false_scores) else np.nan
    false_p99 = float(np.quantile(false_scores, 0.99)) if len(false_scores) else np.nan
    endpoints = sorted({v for pair in true_pairs for v in pair})

    return {
        "function": args.function,
        "seed": int(seed),
        "samples": int(args.samples),
        "test_samples": int(args.test_samples),
        "dimension": int(args.dimension),
        "noise": float(args.noise),
        "width_hidden": int(args.width_hidden),
        "grid": int(args.grid),
        "k": int(args.k),
        "lamb": float(args.lamb),
        "steps": int(args.steps),
        "pair_mode": args.pair_mode,
        "train_mse": mse_np(train_pred, y_train),
        "test_mse": mse_np(test_pred, y_test),
        "candidate_pairs": len(pairs),
        "true_pair": str(true_pair),
        "top_pair": str(top_pair),
        "top_pair_score": float(top_score),
        "true_pair_score": true_score,
        "max_false_pair_score": float(max_false),
        "false_pair_score_p95": false_p95,
        "false_pair_score_p99": false_p99,
        "true_minus_max_false": float(true_score - max_false) if np.isfinite(true_score) and np.isfinite(max_false) else np.nan,
        "true_pair_rank": true_rank,
        "true_pair_beats_candidates": int(true_rank == 1),
        "endpoint_rank_worst_feature": max((rank_desc(feature_scores, v) for v in endpoints), default=-1),
        "endpoint_rank_worst_hybrid": max((rank_desc(hybrid_scores, v) for v in endpoints), default=-1),
        "runtime_sec": float(time.time() - t0),
    }


def summarize(detail: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["function", "samples", "dimension", "width_hidden", "grid", "lamb", "steps", "pair_mode"]
    numeric = [
        "train_mse",
        "test_mse",
        "candidate_pairs",
        "true_pair_score",
        "max_false_pair_score",
        "false_pair_score_p95",
        "false_pair_score_p99",
        "true_minus_max_false",
        "true_pair_rank",
        "true_pair_beats_candidates",
        "endpoint_rank_worst_feature",
        "endpoint_rank_worst_hybrid",
        "runtime_sec",
    ]
    out = detail.groupby(group_cols, dropna=False)[numeric].agg(["mean", "std"]).reset_index()
    out.columns = ["_".join(str(v) for v in col if v != "").rstrip("_") for col in out.columns]
    counts = detail.groupby(group_cols, dropna=False).size().reset_index(name="num_runs")
    return out.merge(counts, on=group_cols)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--function", default="core_interaction_c025")
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--test-samples", type=int, default=2048)
    parser.add_argument("--dimension", type=int, default=1000)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--nuisance-correlation", type=float, default=0.0)
    parser.add_argument("--n-correlated-proxies", type=int, default=0)
    parser.add_argument("--seeds", default="0-2")
    parser.add_argument("--width-hidden", type=int, default=8)
    parser.add_argument("--grid", type=int, default=5)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--steps", type=int, default=35)
    parser.add_argument("--lamb", type=float, default=1e-3)
    parser.add_argument("--opt", default="LBFGS")
    parser.add_argument("--update-grid", action="store_true")
    parser.add_argument("--grid-update-num", type=int, default=5)
    parser.add_argument("--batch", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument(
        "--pair-chunk-size",
        type=int,
        default=0,
        help="Number of pair interventions to construct per anchor for --pair-mode all; 0 means all pairs.",
    )
    parser.add_argument("--anova-points", type=int, default=24)
    parser.add_argument("--anova-background", type=int, default=24)
    parser.add_argument("--pair-mode", choices=["candidate", "all"], default="candidate")
    parser.add_argument("--max-all-pairs", type=int, default=10000)
    parser.add_argument("--top-feature-vars", type=int, default=10)
    parser.add_argument("--top-edge-vars", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-dir", type=Path, default=Path("results/workshop_review_tables/full_kan_pair_anova_probe"))
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore an existing detail CSV in --out-dir and rerun all requested seeds.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    detail_path = args.out_dir / "full_kan_pair_anova_detail.csv"
    summary_path = args.out_dir / "full_kan_pair_anova_summary.csv"

    rows = []
    completed_seeds: set[int] = set()
    if detail_path.exists() and not args.no_resume:
        existing = pd.read_csv(detail_path)
        if len(existing):
            rows = existing.to_dict("records")
            completed_seeds = {int(v) for v in existing["seed"].dropna().astype(int).tolist()}
            print(f"[resume] loaded {len(existing)} existing rows from {detail_path}", flush=True)

    for seed in parse_seeds(args.seeds):
        if int(seed) in completed_seeds:
            print(f"[resume] skip completed seed={seed}", flush=True)
            continue
        print(f"Running full-KAN pair ANOVA probe seed={seed}", flush=True)
        rows.append(run_one(args, seed))
        detail = pd.DataFrame(rows)
        detail.to_csv(detail_path, index=False)
        summarize(detail).to_csv(summary_path, index=False)

    detail = pd.DataFrame(rows)
    summary = summarize(detail)
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
