from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch

from src.data import make_synthetic
from experiments.run_stability_selected_kan_quick import function_to_c
from experiments.run_tuned_kan_recovery import (
    batch_predict,
    canonical_pairs,
    endpoint_recovery,
    evaluate_variable_recovery,
    gradient_importance,
    mse_np,
    train_kan,
)


def make_train_args(args):
    return SimpleNamespace(
        width_hidden=args.width_hidden,
        grid=args.grid,
        k=args.k,
        opt=args.opt,
        steps=args.steps,
        lamb=args.lamb,
        update_grid=not args.no_update_grid,
        grid_update_num=args.grid_update_num,
        batch=args.batch,
        pred_batch_size=args.pred_batch_size,
    )


def run_one(args, function_name: str, seed: int, device: str) -> dict:
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

    row = {
        "model": "KAN_raw_variable_pool",
        "function": function_name,
        "interaction_strength": function_to_c(function_name),
        "screen_mode": "raw",
        "seed": seed,
        "samples": args.samples,
        "test_samples": args.test_samples,
        "dimension": args.dimension,
        "noise": args.noise,
        "effective_dim": args.dimension,
        "selected_screen_features": list(range(args.dimension)),
        "screen_score_type": "none",
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
        "update_grid": int(not args.no_update_grid),
        "grid_update_num": args.grid_update_num,
        "interaction_method": "not_computed",
        "selected_interactions": [],
        "interaction_f1": np.nan,
    }

    try:
        model = train_kan(
            data["X_train"],
            data["y_train"],
            data["X_test"],
            data["y_test"],
            make_train_args(args),
            seed=seed,
            device=device,
        )
        train_pred = batch_predict(model, data["X_train"], device=device, batch_size=args.pred_batch_size)
        test_pred = batch_predict(model, data["X_test"], device=device, batch_size=args.pred_batch_size)
        scores = gradient_importance(model, data["X_test"], device=device, points=args.variable_points)

        row.update(
            {
                "status": "ok",
                "error": "",
                "train_mse": mse_np(data["y_train"], train_pred),
                "test_mse": mse_np(data["y_test"], test_pred),
                "importance_scores": scores.tolist(),
            }
        )
        row.update(evaluate_variable_recovery(scores, true_vars))
        row.update(endpoint_recovery(row["selected_variables"], true_interactions, "explain"))
    except Exception as exc:
        row.update(
            {
                "status": "failed",
                "error": repr(exc),
                "train_mse": np.nan,
                "test_mse": np.nan,
                "importance_scores": [],
                "selected_variables": [],
                "variable_f1": np.nan,
            }
        )
        print(f"[WARN] failed {function_name} seed={seed}: {exc}")

    return row


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    ok = df[df["status"].astype(str) == "ok"].copy()
    numeric_cols = [
        "train_mse",
        "test_mse",
        "variable_f1",
        "variable_auroc",
        "variable_auprc",
        "explain_interaction_endpoint_recall",
    ]
    for col in numeric_cols:
        if col in ok.columns:
            ok[col] = pd.to_numeric(ok[col], errors="coerce")
    group_cols = ["function", "interaction_strength", "screen_mode", "dimension", "samples"]
    summary = ok.groupby(group_cols, dropna=False)[numeric_cols].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join(str(x) for x in col if x != "").rstrip("_") for col in summary.columns]
    counts = ok.groupby(group_cols, dropna=False).size().reset_index(name="num_runs")
    return summary.merge(counts, on=group_cols, how="left")


def main():
    parser = argparse.ArgumentParser(description="Fast raw KAN pool for variable-stability selection.")
    parser.add_argument("--functions", nargs="+", default=["core_interaction_c025"])
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--test_samples", type=int, default=4096)
    parser.add_argument("--dimension", type=int, default=100)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--grid", type=int, default=5)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--width_hidden", type=int, default=8)
    parser.add_argument("--lamb", type=float, default=0.001)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--opt", default="LBFGS")
    parser.add_argument("--no_update_grid", action="store_true", default=True)
    parser.add_argument("--grid_update_num", type=int, default=10)
    parser.add_argument("--batch", type=int, default=-1)
    parser.add_argument("--variable_points", type=int, default=256)
    parser.add_argument("--pred_batch_size", type=int, default=4096)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--out_dir", default="results/stability_kan/raw_variable_pool")
    args = parser.parse_args()

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    print(f"Using device={device}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for fn in args.functions:
        rows = []
        out_path = out_dir / f"{fn}_n{args.samples}_d{args.dimension}_detail.csv"
        for seed in args.seeds:
            print(f"[RUN] function={fn} screen=raw-variable-only seed={seed}")
            row = run_one(args, fn, seed, device)
            rows.append(row)
            all_rows.append(row)
            pd.DataFrame(rows).to_csv(out_path, index=False)
        print(f"Wrote {out_path}")

    summary = summarize(pd.DataFrame(all_rows))
    summary_path = out_dir / f"summary_n{args.samples}_d{args.dimension}.csv"
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False))
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
