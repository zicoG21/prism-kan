from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.data import make_synthetic
from experiments.run_tuned_kan_recovery import (
    anova_pair_scores,
    batch_predict,
    canonical_pairs,
    evaluate_interaction_recovery,
    finite_difference_pair_scores,
    local_to_full_pair_scores,
    mse_np,
    support_stats,
    train_kan,
)


def train_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        width_hidden=args.width_hidden,
        grid=args.grid,
        k=args.k,
        steps=args.steps,
        lamb=args.lamb,
        opt=args.opt,
        update_grid=args.update_grid,
        grid_update_num=args.grid_update_num,
        batch=args.batch,
    )


def parse_refit_seeds(args: argparse.Namespace, data_seed: int) -> list[int]:
    if args.refit_seeds:
        return [int(v) for v in args.refit_seeds]
    if args.refit_offsets:
        return [int(data_seed) + int(offset) for offset in args.refit_offsets]
    return [int(data_seed)]


def score_pairs(model, X_test_s: np.ndarray, args: argparse.Namespace) -> dict:
    if args.pair_score == "fd":
        return finite_difference_pair_scores(
            model,
            X_test_s,
            device=args.device,
            points=args.fd_points,
            h=args.fd_h,
            batch_size=args.pred_batch_size,
        )
    if args.pair_score == "anova_abs":
        return anova_pair_scores(
            model,
            X_test_s,
            device=args.device,
            points=args.anova_points,
            background=args.anova_background,
            batch_size=args.pred_batch_size,
            score="abs",
        )
    if args.pair_score == "anova_var":
        return anova_pair_scores(
            model,
            X_test_s,
            device=args.device,
            points=args.anova_points,
            background=args.anova_background,
            batch_size=args.pred_batch_size,
            score="var",
        )
    raise ValueError(f"Unknown pair_score={args.pair_score!r}")


def run_one(args: argparse.Namespace, n: int, data_seed: int, refit_seed: int) -> dict:
    t0 = time.time()
    data = make_synthetic(
        function_name=args.function,
        n_train=n,
        n_test=args.test_samples,
        d=args.dimension,
        noise=args.noise,
        seed=data_seed,
        standardize_target=True,
        nuisance_correlation=args.nuisance_correlation,
        n_correlated_proxies=args.n_correlated_proxies,
    )
    gt = data["ground_truth"]
    true_vars = tuple(int(v) for v in gt.active_variables)
    true_interactions = canonical_pairs(gt.interactions)
    support = np.asarray(true_vars if args.support == "true" else args.support, dtype=int)
    support = np.asarray(sorted(int(v) for v in support), dtype=int)

    X_train_s = data["X_train"][:, support]
    X_test_s = data["X_test"][:, support]

    model = train_kan(
        X_train_s,
        data["y_train"],
        X_test_s,
        data["y_test"],
        train_args(args),
        seed=int(refit_seed),
        device=args.device,
    )
    train_pred = batch_predict(model, X_train_s, device=args.device, batch_size=args.pred_batch_size)
    test_pred = batch_predict(model, X_test_s, device=args.device, batch_size=args.pred_batch_size)
    local_pair_scores = score_pairs(model, X_test_s, args)
    full_pair_scores = local_to_full_pair_scores(local_pair_scores, support, args.dimension)
    interaction_eval = evaluate_interaction_recovery(full_pair_scores, true_interactions)
    ranked = sorted(full_pair_scores.items(), key=lambda kv: kv[1], reverse=True)

    row = {
        "function": args.function,
        "samples": int(n),
        "test_samples": int(args.test_samples),
        "dimension": int(args.dimension),
        "noise": float(args.noise),
        "data_seed": int(data_seed),
        "refit_seed": int(refit_seed),
        "support_mode": str(args.support),
        "selected_screen_features": support.tolist(),
        "true_variables": list(true_vars),
        "true_interactions": list(true_interactions),
        "formula": gt.formula,
        "width_hidden": int(args.width_hidden),
        "grid": int(args.grid),
        "k": int(args.k),
        "steps": int(args.steps),
        "lamb": float(args.lamb),
        "opt": args.opt,
        "update_grid": int(bool(args.update_grid)),
        "pair_score": args.pair_score,
        "train_mse": mse_np(train_pred, data["y_train"]),
        "test_mse": mse_np(test_pred, data["y_test"]),
        "top_pair": ranked[0][0] if ranked else None,
        "top_pair_score": float(ranked[0][1]) if ranked else np.nan,
        "runtime_sec": float(time.time() - t0),
    }
    row.update(support_stats(support, true_vars, true_interactions))
    row.update(interaction_eval)
    return row


def summarize(detail: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "function",
        "samples",
        "dimension",
        "support_mode",
        "pair_score",
        "width_hidden",
        "grid",
        "k",
        "steps",
        "lamb",
    ]
    numeric_cols = [
        "train_mse",
        "test_mse",
        "screen_contains_all_true_vars",
        "screen_interaction_endpoint_recall",
        "interaction_f1",
        "true_interaction_rank_mean",
        "true_interaction_score_mean",
        "max_nontrue_interaction_score",
        "runtime_sec",
    ]
    out = detail.groupby(group_cols, dropna=False)[numeric_cols].agg(["mean", "std"]).reset_index()
    out.columns = ["_".join(str(v) for v in col if v != "").rstrip("_") for col in out.columns]
    counts = detail.groupby(group_cols, dropna=False).size().reset_index(name="num_runs")
    successes = detail.groupby(group_cols, dropna=False)["interaction_f1"].sum().reset_index(name="top1_successes")
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
    parser.add_argument("--data_seeds", type=int, nargs="+", default=list(range(100, 110)))
    parser.add_argument("--refit_seeds", type=int, nargs="+", default=None)
    parser.add_argument("--refit_offsets", type=int, nargs="+", default=None)
    parser.add_argument("--support", default="true")
    parser.add_argument("--pair_score", choices=["fd", "anova_abs", "anova_var"], default="fd")
    parser.add_argument("--width_hidden", type=int, default=8)
    parser.add_argument("--grid", type=int, default=5)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lamb", type=float, default=1e-3)
    parser.add_argument("--opt", default="LBFGS")
    parser.add_argument("--update_grid", action="store_true")
    parser.add_argument("--grid_update_num", type=int, default=10)
    parser.add_argument("--batch", type=int, default=-1)
    parser.add_argument("--fd_points", type=int, default=128)
    parser.add_argument("--fd_h", type=float, default=1e-2)
    parser.add_argument("--anova_points", type=int, default=64)
    parser.add_argument("--anova_background", type=int, default=64)
    parser.add_argument("--pred_batch_size", type=int, default=4096)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out_dir", default="results/workshop_review_tables/paired_true_support_refit_sanity")
    args = parser.parse_args()

    rows = []
    for n in args.samples:
        for data_seed in args.data_seeds:
            for refit_seed in parse_refit_seeds(args, int(data_seed)):
                print(f"Running n={n}, data_seed={data_seed}, refit_seed={refit_seed}", flush=True)
                rows.append(run_one(args, int(n), int(data_seed), int(refit_seed)))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    detail = pd.DataFrame(rows)
    summary = summarize(detail)
    detail.to_csv(out_dir / "paired_true_support_refit_detail.csv", index=False)
    summary.to_csv(out_dir / "paired_true_support_refit_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
