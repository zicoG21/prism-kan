from __future__ import annotations

import argparse
import itertools
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Dict

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import make_synthetic
from experiments.run_stability_selected_kan_quick import function_to_c
from experiments.run_tuned_kan_recovery import (
    Pair,
    batch_predict,
    canonical_pairs,
    evaluate_interaction_recovery,
    finite_difference_pair_scores,
    local_to_full_pair_scores,
    mse_np,
    support_stats,
    train_kan,
)


def make_train_args(args) -> SimpleNamespace:
    return SimpleNamespace(
        width_hidden=args.width_hidden,
        grid=args.grid,
        k=args.k,
        steps=args.steps,
        lamb=args.lamb,
        opt=args.opt,
        update_grid=not args.no_update_grid,
        grid_update_num=args.grid_update_num,
        batch=args.batch,
    )


def normalize_pair_scores(scores: Dict[Pair, float]) -> Dict[Pair, float]:
    if not scores:
        return {}
    vals = np.array([max(float(v), 0.0) for v in scores.values()], dtype=float)
    mx = float(np.max(vals)) if vals.size else 0.0
    if mx <= 0:
        return {pair: 0.0 for pair in scores}
    return {pair: max(float(v), 0.0) / mx for pair, v in scores.items()}


def anova_pair_scores(
    model,
    X_np: np.ndarray,
    device: str,
    *,
    points: int,
    background: int,
    batch_size: int,
    score: str,
) -> Dict[Pair, float]:
    d = X_np.shape[1]
    n_points = min(points, len(X_np))
    n_bg = min(background, len(X_np))
    base = X_np[:n_points].copy()
    bg = X_np[:n_bg].copy()
    f_mean = float(np.mean(batch_predict(model, bg, device=device, batch_size=batch_size).reshape(-1)))

    out: Dict[Pair, float] = {}
    for i, j in itertools.combinations(range(d), 2):
        comps = []
        for row in base:
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
        comps_arr = np.asarray(comps, dtype=float)
        if score == "abs":
            out[(i, j)] = float(np.mean(np.abs(comps_arr)))
        elif score == "var":
            out[(i, j)] = float(np.var(comps_arr))
        else:
            raise ValueError(f"Unknown ANOVA score={score}")
    return out


def hybrid_scores(*score_dicts: Dict[Pair, float]) -> Dict[Pair, float]:
    keys = sorted(set().union(*(d.keys() for d in score_dicts)))
    normalized = [normalize_pair_scores(d) for d in score_dicts]
    return {pair: float(np.mean([d.get(pair, 0.0) for d in normalized])) for pair in keys}


def fill_support(true_vars, dimension: int, top_m: int) -> np.ndarray:
    selected = []
    seen = set()
    for v in true_vars:
        v = int(v)
        if v not in seen and len(selected) < top_m:
            selected.append(v)
            seen.add(v)
    for v in range(dimension):
        if len(selected) >= top_m:
            break
        if v not in seen:
            selected.append(v)
            seen.add(v)
    return np.array(sorted(selected), dtype=int)


def run_one(args, function_name: str, seed: int, device: str) -> list[dict]:
    data = make_synthetic(
        function_name=function_name,
        n_train=args.samples,
        n_test=args.test_samples,
        d=args.dimension,
        noise=args.noise,
        seed=seed,
        standardize_target=True,
    )
    gt = data["ground_truth"]
    true_vars = tuple(int(v) for v in gt.active_variables)
    true_interactions = canonical_pairs(gt.interactions)
    support = fill_support(true_vars, args.dimension, args.top_m)
    Xtr = data["X_train"][:, support]
    Xte = data["X_test"][:, support]
    base = {
        "function": function_name,
        "interaction_strength": function_to_c(function_name),
        "seed": seed,
        "samples": args.samples,
        "dimension": args.dimension,
        "top_m": args.top_m,
        "support": support.tolist(),
        "true_variables": list(true_vars),
        "true_interactions": list(true_interactions),
        "formula": gt.formula,
        "steps": args.steps,
        "width_hidden": args.width_hidden,
        "grid": args.grid,
        "k": args.k,
        "lamb": args.lamb,
    }
    base.update(support_stats(support, true_vars, true_interactions))
    rows = []
    t0 = time.time()
    try:
        model = train_kan(Xtr, data["y_train"], Xte, data["y_test"], make_train_args(args), seed=seed, device=device)
        test_pred = batch_predict(model, Xte, device=device, batch_size=args.pred_batch_size)
        test_mse = mse_np(test_pred, data["y_test"])
        local_fd = finite_difference_pair_scores(
            model,
            Xte,
            device=device,
            points=args.fd_points,
            h=args.fd_h,
            batch_size=args.pred_batch_size,
        )
        local_anova_abs = anova_pair_scores(
            model,
            Xte,
            device=device,
            points=args.anova_points,
            background=args.anova_background,
            batch_size=args.pred_batch_size,
            score="abs",
        )
        local_anova_var = anova_pair_scores(
            model,
            Xte,
            device=device,
            points=args.anova_points,
            background=args.anova_background,
            batch_size=args.pred_batch_size,
            score="var",
        )
        local_hybrid = hybrid_scores(local_fd, local_anova_abs)
        methods = {
            "fd": local_fd,
            "anova_abs": local_anova_abs,
            "anova_var": local_anova_var,
            "fd_anova_hybrid": local_hybrid,
        }
        for method, local_scores in methods.items():
            full_scores = local_to_full_pair_scores(local_scores, support, args.dimension)
            row = dict(base)
            row.update(
                {
                    "pair_score_method": method,
                    "status": "ok",
                    "error": "",
                    "traceback": "",
                    "test_mse": test_mse,
                    "runtime_sec": float(time.time() - t0),
                }
            )
            row.update(evaluate_interaction_recovery(full_scores, true_interactions))
            rows.append(row)
    except Exception as exc:
        row = dict(base)
        row.update(
            {
                "pair_score_method": "all",
                "status": "failed",
                "error": repr(exc),
                "traceback": traceback.format_exc(),
                "test_mse": np.nan,
                "runtime_sec": float(time.time() - t0),
            }
        )
        rows.append(row)
    return rows


def summarize(detail: pd.DataFrame) -> pd.DataFrame:
    ok = detail[detail["status"].astype(str).eq("ok")].copy()
    if ok.empty:
        return pd.DataFrame()
    group_cols = ["pair_score_method", "function", "samples", "dimension", "top_m"]
    numeric_cols = [
        "test_mse",
        "interaction_f1",
        "true_interaction_rank_mean",
        "true_interaction_mean_score_margin",
        "true_interaction_beats_all_false",
    ]
    for col in numeric_cols:
        if col in ok.columns:
            ok[col] = pd.to_numeric(ok[col], errors="coerce")
    agg = {col: ["mean", "std"] for col in numeric_cols if col in ok.columns}
    out = ok.groupby(group_cols, dropna=False).agg(agg).reset_index()
    out.columns = ["_".join(str(x) for x in col if x != "").rstrip("_") for col in out.columns]
    counts = ok.groupby(group_cols, dropna=False).size().reset_index(name="num_runs")
    return out.merge(counts, on=group_cols, how="left")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare pair-scoring methods after oracle support refit.")
    parser.add_argument("--functions", nargs="+", default=["core_interaction_c025"])
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--test_samples", type=int, default=4096)
    parser.add_argument("--dimension", type=int, default=100)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(100, 110)))
    parser.add_argument("--top_m", type=int, default=4)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--grid", type=int, default=5)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--width_hidden", type=int, default=8)
    parser.add_argument("--lamb", type=float, default=0.001)
    parser.add_argument("--opt", default="LBFGS")
    parser.add_argument("--no_update_grid", action="store_true", default=True)
    parser.add_argument("--grid_update_num", type=int, default=10)
    parser.add_argument("--batch", type=int, default=-1)
    parser.add_argument("--fd_points", type=int, default=128)
    parser.add_argument("--fd_h", type=float, default=1e-2)
    parser.add_argument("--anova_points", type=int, default=64)
    parser.add_argument("--anova_background", type=int, default=64)
    parser.add_argument("--pred_batch_size", type=int, default=4096)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--out_dir", default="results/innovation_loop/pair_scoring_oracle")
    args = parser.parse_args()

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    print(f"Using device={device}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for fn in args.functions:
        for seed in args.seeds:
            print(f"[PAIR] function={fn} n={args.samples} d={args.dimension} seed={seed}", flush=True)
            rows.extend(run_one(args, fn, int(seed), device))
            pd.DataFrame(rows).to_csv(out_dir / "pair_scoring_detail.csv", index=False)
    detail = pd.DataFrame(rows)
    summary = summarize(detail)
    detail.to_csv(out_dir / "pair_scoring_detail.csv", index=False)
    summary.to_csv(out_dir / "pair_scoring_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"Wrote {out_dir / 'pair_scoring_detail.csv'}")
    print(f"Wrote {out_dir / 'pair_scoring_summary.csv'}")


if __name__ == "__main__":
    main()
