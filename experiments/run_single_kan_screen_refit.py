from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Sequence

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import make_synthetic
from experiments.run_kan_native_innovation_loop import (
    combine_scores,
    normalize_score,
    safe_edge_path_scores,
    safe_feature_score,
    support_from_pairs,
    top_vars,
)
from experiments.run_stability_selected_kan_quick import function_to_c
from experiments.run_tuned_kan_recovery import (
    Pair,
    batch_predict,
    canonical_pairs,
    endpoint_recovery,
    evaluate_interaction_recovery,
    evaluate_variable_recovery,
    finite_difference_pair_scores,
    gradient_importance,
    local_to_full_pair_scores,
    local_to_full_scores,
    mse_np,
    support_stats,
    train_kan,
)


METHODS = [
    "single_grad_var",
    "single_feature_var",
    "single_edge_var",
    "single_feature_edge_hybrid",
    "single_edge_pair_hybrid",
]


def dumps(obj) -> str:
    return json.dumps(obj, sort_keys=True)


def make_args(args, steps: int) -> SimpleNamespace:
    return SimpleNamespace(
        width_hidden=args.width_hidden,
        grid=args.grid,
        k=args.k,
        steps=steps,
        lamb=args.lamb,
        opt=args.opt,
        update_grid=not args.no_update_grid,
        grid_update_num=args.grid_update_num,
        batch=args.batch,
    )


def select_support(method: str, scores: Dict[str, object], top_m: int, d: int):
    grad = np.asarray(scores["grad"], dtype=float)
    feature = np.asarray(scores["feature"], dtype=float)
    edge = np.asarray(scores["edge"], dtype=float)
    endpoint_mass = np.asarray(scores["edge_endpoint_mass"], dtype=float)
    edge_pairs = dict(scores["edge_pairs"])

    if method == "single_grad_var":
        selection_score = grad
        support = top_vars(selection_score, top_m)
    elif method == "single_feature_var":
        selection_score = feature
        support = top_vars(selection_score, top_m)
    elif method == "single_edge_var":
        selection_score = edge
        support = top_vars(selection_score, top_m)
    elif method == "single_feature_edge_hybrid":
        selection_score = combine_scores(feature, edge, endpoint_mass)
        support = top_vars(selection_score, top_m)
    elif method == "single_edge_pair_hybrid":
        selection_score = combine_scores(feature, edge, endpoint_mass)
        support = support_from_pairs(edge_pairs, selection_score, top_m)
    else:
        raise ValueError(f"Unknown method={method}")

    ranked_pairs = sorted(edge_pairs.items(), key=lambda kv: -float(kv[1]))[:10]
    return np.array(sorted(support), dtype=int), {
        "selection_score": np.asarray(selection_score, dtype=float).tolist(),
        "top_selection_variables": top_vars(selection_score, min(12, d)),
        "top_edge_pairs": [(int(i), int(j), float(v)) for (i, j), v in ranked_pairs],
    }


def score_screen_model(model, X_test: np.ndarray, d: int, args, device: str) -> Dict[str, object]:
    grad = normalize_score(gradient_importance(model, X_test, device=device, points=args.screen_variable_points))
    feature = safe_feature_score(model, d)
    edge, edge_pairs, endpoint_mass = safe_edge_path_scores(model, d)
    return {
        "grad": grad,
        "feature": feature,
        "edge": edge,
        "edge_pairs": edge_pairs,
        "edge_endpoint_mass": endpoint_mass,
    }


def run_refit(
    *,
    args,
    data: Dict,
    function_name: str,
    seed: int,
    method: str,
    support: np.ndarray,
    support_meta: Dict[str, object],
    screen_test_mse: float,
    device: str,
) -> Dict:
    gt = data["ground_truth"]
    true_vars = tuple(int(v) for v in gt.active_variables)
    true_interactions = canonical_pairs(gt.interactions)
    Xtr = data["X_train"][:, support]
    Xte = data["X_test"][:, support]
    row = {
        "wave": "one_shot",
        "method": method,
        "function": function_name,
        "interaction_strength": function_to_c(function_name),
        "samples": args.samples,
        "dimension": args.dimension,
        "noise": args.noise,
        "seed": seed,
        "top_m": args.top_m,
        "selected_screen_features": support.tolist(),
        "true_variables": list(true_vars),
        "true_interactions": list(true_interactions),
        "num_true_variables": len(true_vars),
        "num_true_interactions": len(true_interactions),
        "formula": gt.formula,
        "screen_steps": args.screen_steps,
        "refit_steps": args.refit_steps,
        "width_hidden": args.width_hidden,
        "grid": args.grid,
        "k": args.k,
        "lamb": args.lamb,
        "fd_points": args.fd_points,
        "screen_model_test_mse": screen_test_mse,
        "status": "ok",
        "error": "",
        "traceback": "",
        "runtime_sec": np.nan,
        "selection_meta": dumps(support_meta),
        **support_meta,
    }
    row.update(support_stats(support, true_vars, true_interactions))

    t0 = time.time()
    try:
        refit_args = make_args(args, args.refit_steps)
        model = train_kan(Xtr, data["y_train"], Xte, data["y_test"], refit_args, seed=seed, device=device)
        train_pred = batch_predict(model, Xtr, device=device, batch_size=args.pred_batch_size)
        test_pred = batch_predict(model, Xte, device=device, batch_size=args.pred_batch_size)
        local_var_scores = gradient_importance(model, Xte, device=device, points=args.refit_variable_points)
        full_var_scores = local_to_full_scores(local_var_scores, support, args.dimension)
        if len(true_interactions) > 0:
            local_pair_scores = finite_difference_pair_scores(
                model,
                Xte,
                device=device,
                points=args.fd_points,
                h=args.fd_h,
                batch_size=args.pred_batch_size,
            )
            full_pair_scores = local_to_full_pair_scores(local_pair_scores, support, args.dimension)
        else:
            full_pair_scores = {}
        row.update(
            {
                "train_mse": mse_np(train_pred, data["y_train"]),
                "test_mse": mse_np(test_pred, data["y_test"]),
                "importance_scores": full_var_scores.tolist(),
                "interaction_method": "fd",
            }
        )
        var_eval = evaluate_variable_recovery(full_var_scores, true_vars)
        row.update(var_eval)
        row.update(endpoint_recovery(var_eval["selected_variables"], true_interactions, "explain"))
        row.update(evaluate_interaction_recovery(full_pair_scores, true_interactions))
    except Exception as exc:
        row.update(
            {
                "status": "failed",
                "error": repr(exc),
                "traceback": traceback.format_exc(),
                "train_mse": np.nan,
                "test_mse": np.nan,
                "variable_f1": np.nan,
                "interaction_f1": np.nan,
                "selected_variables": [],
                "selected_interactions": [],
            }
        )
    row["runtime_sec"] = float(time.time() - t0)
    return row


def summarize(detail: pd.DataFrame) -> pd.DataFrame:
    ok = detail[detail["status"].astype(str).eq("ok")].copy()
    if ok.empty:
        return pd.DataFrame()
    group_cols = ["wave", "method", "function", "samples", "dimension", "top_m"]
    numeric_cols = [
        "screen_model_test_mse",
        "train_mse",
        "test_mse",
        "screen_contains_all_true_vars",
        "screen_true_var_recall",
        "screen_contains_true_interactions",
        "screen_interaction_endpoint_recall",
        "variable_f1",
        "variable_auroc",
        "variable_auprc",
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


def run_function(args, function_name: str, device: str) -> list[Dict]:
    rows = []
    for seed in args.seeds:
        print(f"[SCREEN] function={function_name} seed={seed}", flush=True)
        data = make_synthetic(
            function_name=function_name,
            n_train=args.samples,
            n_test=args.test_samples,
            d=args.dimension,
            noise=args.noise,
            seed=int(seed),
            standardize_target=True,
        )
        try:
            screen_args = make_args(args, args.screen_steps)
            screen_model = train_kan(
                data["X_train"],
                data["y_train"],
                data["X_test"],
                data["y_test"],
                screen_args,
                seed=int(seed),
                device=device,
            )
            screen_pred = batch_predict(screen_model, data["X_test"], device=device, batch_size=args.pred_batch_size)
            screen_test_mse = mse_np(screen_pred, data["y_test"])
            screen_scores = score_screen_model(screen_model, data["X_test"], args.dimension, args, device)
        except Exception as exc:
            print(f"[WARN] screen failed function={function_name} seed={seed}: {exc}", flush=True)
            for method in args.methods:
                rows.append(
                    {
                        "wave": "one_shot",
                        "method": method,
                        "function": function_name,
                        "samples": args.samples,
                        "dimension": args.dimension,
                        "seed": seed,
                        "top_m": args.top_m,
                        "status": "failed",
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
            continue

        for method in args.methods:
            print(f"[REFIT] function={function_name} seed={seed} method={method}", flush=True)
            support, meta = select_support(method, screen_scores, args.top_m, args.dimension)
            row = run_refit(
                args=args,
                data=data,
                function_name=function_name,
                seed=int(seed),
                method=method,
                support=support,
                support_meta=meta,
                screen_test_mse=screen_test_mse,
                device=device,
            )
            rows.append(row)
            pd.DataFrame(rows).to_csv(args.out, index=False)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="One-shot KAN-native screening followed by low-dimensional KAN refit.")
    parser.add_argument("--functions", nargs="+", default=["core_interaction_c025"])
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--test_samples", type=int, default=4096)
    parser.add_argument("--dimension", type=int, default=100)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--top_m", type=int, default=4)
    parser.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    parser.add_argument("--screen_steps", type=int, default=80)
    parser.add_argument("--refit_steps", type=int, default=100)
    parser.add_argument("--grid", type=int, default=5)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--width_hidden", type=int, default=8)
    parser.add_argument("--lamb", type=float, default=0.001)
    parser.add_argument("--opt", default="LBFGS")
    parser.add_argument("--no_update_grid", action="store_true", default=True)
    parser.add_argument("--grid_update_num", type=int, default=10)
    parser.add_argument("--batch", type=int, default=-1)
    parser.add_argument("--screen_variable_points", type=int, default=256)
    parser.add_argument("--refit_variable_points", type=int, default=256)
    parser.add_argument("--fd_points", type=int, default=128)
    parser.add_argument("--fd_h", type=float, default=1e-2)
    parser.add_argument("--pred_batch_size", type=int, default=4096)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--out", default="results/innovation_loop/single_kan_screen_refit/detail.csv")
    parser.add_argument("--summary_out", default="results/innovation_loop/single_kan_screen_refit/summary.csv")
    args = parser.parse_args()

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    print(f"Using device={device}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for function_name in args.functions:
        all_rows.extend(run_function(args, function_name, device))
    detail = pd.DataFrame(all_rows)
    detail.to_csv(args.out, index=False)
    summary = summarize(detail)
    summary.to_csv(args.summary_out, index=False)
    print(summary.to_string(index=False))
    print(f"Wrote {args.out}")
    print(f"Wrote {args.summary_out}")


if __name__ == "__main__":
    main()
