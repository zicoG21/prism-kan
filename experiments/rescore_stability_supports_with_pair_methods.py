from __future__ import annotations

import argparse
import ast
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import make_synthetic
from experiments.run_stability_selected_kan_quick import function_to_c
from experiments.run_tuned_kan_recovery import (
    Pair,
    anova_pair_scores,
    batch_predict,
    canonical_pairs,
    endpoint_recovery,
    evaluate_interaction_recovery,
    evaluate_variable_recovery,
    finite_difference_pair_scores,
    gradient_importance,
    hybrid_pair_scores,
    local_to_full_pair_scores,
    local_to_full_scores,
    mse_np,
    support_stats,
    train_kan,
)


def parse_literal_list(value) -> List[int]:
    if isinstance(value, list):
        return [int(v) for v in value]
    if value is None:
        return []
    try:
        if isinstance(value, float) and np.isnan(value):
            return []
    except TypeError:
        pass
    parsed = ast.literal_eval(str(value))
    return [int(v) for v in parsed]


def make_train_args(row: pd.Series, args) -> SimpleNamespace:
    return SimpleNamespace(
        width_hidden=int(args.width_hidden or row.get("width_hidden", 8)),
        grid=int(args.grid or row.get("grid", 5)),
        k=int(args.k or row.get("k", 3)),
        steps=int(args.steps or row.get("refit_steps", row.get("steps", 100))),
        lamb=float(args.lamb if args.lamb is not None else row.get("lamb", 0.001)),
        opt=args.opt,
        update_grid=not args.no_update_grid,
        grid_update_num=int(args.grid_update_num),
        batch=int(args.batch),
    )


def score_pairs(model, Xte: np.ndarray, args, device: str) -> Dict[str, Dict[Pair, float]]:
    scores: Dict[str, Dict[Pair, float]] = {}
    need_fd = any(method in {"fd", "fd_anova_hybrid"} for method in args.pair_methods)
    need_anova_abs = any(method in {"anova_abs", "fd_anova_hybrid"} for method in args.pair_methods)
    need_anova_var = "anova_var" in args.pair_methods
    fd_scores = None
    anova_abs_scores = None

    if need_fd:
        fd_scores = finite_difference_pair_scores(
            model,
            Xte,
            device=device,
            points=args.fd_points,
            h=args.fd_h,
            batch_size=args.pred_batch_size,
        )
    if need_anova_abs:
        anova_abs_scores = anova_pair_scores(
            model,
            Xte,
            device=device,
            points=args.anova_points,
            background=args.anova_background,
            batch_size=args.pred_batch_size,
            score="abs",
        )
    if need_anova_var:
        scores["anova_var"] = anova_pair_scores(
            model,
            Xte,
            device=device,
            points=args.anova_points,
            background=args.anova_background,
            batch_size=args.pred_batch_size,
            score="var",
        )
    if "fd" in args.pair_methods:
        scores["fd"] = fd_scores or {}
    if "anova_abs" in args.pair_methods:
        scores["anova_abs"] = anova_abs_scores or {}
    if "fd_anova_hybrid" in args.pair_methods:
        scores["fd_anova_hybrid"] = hybrid_pair_scores(fd_scores or {}, anova_abs_scores or {})
    return scores


def filter_rows(df: pd.DataFrame, args) -> pd.DataFrame:
    out = df[df["status"].astype(str).eq("ok")].copy() if "status" in df.columns else df.copy()
    if args.functions:
        out = out[out["function"].astype(str).isin(args.functions)]
    if args.methods:
        out = out[out["method"].astype(str).isin(args.methods)]
    if args.samples:
        out = out[pd.to_numeric(out["samples"], errors="coerce").isin([int(v) for v in args.samples])]
    if args.dimensions:
        out = out[pd.to_numeric(out["dimension"], errors="coerce").isin([int(v) for v in args.dimensions])]
    if args.top_m:
        out = out[pd.to_numeric(out["top_m"], errors="coerce").isin([int(v) for v in args.top_m])]
    if args.seeds:
        out = out[pd.to_numeric(out["seed"], errors="coerce").isin([int(v) for v in args.seeds])]
    out = out.sort_values(["function", "samples", "dimension", "top_m", "method", "seed"])
    if args.max_rows:
        out = out.head(int(args.max_rows))
    return out


def run_one(source_row: pd.Series, args, device: str) -> List[dict]:
    function_name = str(source_row["function"])
    seed = int(source_row["seed"])
    samples = int(source_row["samples"])
    dimension = int(source_row["dimension"])
    top_m = int(source_row["top_m"])
    support = np.asarray(parse_literal_list(source_row["selected_screen_features"]), dtype=int)

    data = make_synthetic(
        function_name=function_name,
        n_train=samples,
        n_test=args.test_samples,
        d=dimension,
        noise=float(source_row.get("noise", args.noise)),
        seed=seed,
        standardize_target=True,
    )
    gt = data["ground_truth"]
    true_vars = tuple(int(v) for v in gt.active_variables)
    true_interactions = canonical_pairs(gt.interactions)
    Xtr = data["X_train"][:, support]
    Xte = data["X_test"][:, support]

    base = {
        "source_file": str(args.input),
        "source_wave": source_row.get("wave", ""),
        "source_method": source_row.get("method", ""),
        "source_interaction_method": source_row.get("interaction_method", ""),
        "source_interaction_f1": source_row.get("interaction_f1", np.nan),
        "function": function_name,
        "interaction_strength": function_to_c(function_name),
        "samples": samples,
        "test_samples": args.test_samples,
        "dimension": dimension,
        "seed": seed,
        "top_m": top_m,
        "selected_screen_features": support.tolist(),
        "true_variables": list(true_vars),
        "true_interactions": list(true_interactions),
        "formula": gt.formula,
    }
    base.update(support_stats(support, true_vars, true_interactions))

    t0 = time.time()
    rows: List[dict] = []
    try:
        train_args = make_train_args(source_row, args)
        model = train_kan(Xtr, data["y_train"], Xte, data["y_test"], train_args, seed=seed, device=device)
        train_pred = batch_predict(model, Xtr, device=device, batch_size=args.pred_batch_size)
        test_pred = batch_predict(model, Xte, device=device, batch_size=args.pred_batch_size)
        local_var_scores = gradient_importance(model, Xte, device=device, points=args.variable_points)
        full_var_scores = local_to_full_scores(local_var_scores, support, dimension)
        pair_scores_by_method = score_pairs(model, Xte, args, device) if true_interactions else {}
        var_eval = evaluate_variable_recovery(full_var_scores, true_vars)
        common = dict(base)
        common.update(
            {
                "status": "ok",
                "error": "",
                "traceback": "",
                "runtime_sec": float(time.time() - t0),
                "train_mse": mse_np(train_pred, data["y_train"]),
                "test_mse": mse_np(test_pred, data["y_test"]),
                "variable_method": "grad",
                "importance_scores": full_var_scores.tolist(),
            }
        )
        common.update(var_eval)
        common.update(endpoint_recovery(var_eval["selected_variables"], true_interactions, "explain"))
        for method in args.pair_methods:
            row = dict(common)
            row["pair_score_method"] = method
            full_pair_scores = local_to_full_pair_scores(pair_scores_by_method.get(method, {}), support, dimension)
            row.update(evaluate_interaction_recovery(full_pair_scores, true_interactions))
            rows.append(row)
    except Exception as exc:
        row = dict(base)
        row.update(
            {
                "status": "failed",
                "error": repr(exc),
                "traceback": traceback.format_exc(),
                "runtime_sec": float(time.time() - t0),
                "pair_score_method": "all",
            }
        )
        rows.append(row)
    return rows


def summarize(detail: pd.DataFrame) -> pd.DataFrame:
    ok = detail[detail["status"].astype(str).eq("ok")].copy()
    if ok.empty:
        return pd.DataFrame()
    group_cols = ["source_method", "pair_score_method", "function", "samples", "dimension", "top_m"]
    numeric_cols = [
        "source_interaction_f1",
        "test_mse",
        "screen_contains_true_interactions",
        "screen_interaction_endpoint_recall",
        "interaction_f1",
        "true_interaction_rank_mean",
        "true_interaction_mean_score_margin",
        "true_interaction_beats_all_false",
    ]
    for col in numeric_cols:
        ok[col] = pd.to_numeric(ok[col], errors="coerce")
    agg = {col: ["mean", "std"] for col in numeric_cols if col in ok.columns}
    out = ok.groupby(group_cols, dropna=False).agg(agg).reset_index()
    out.columns = ["_".join(str(v) for v in col if v != "").rstrip("_") for col in out.columns]
    counts = ok.groupby(group_cols, dropna=False).size().reset_index(name="num_runs")
    return out.merge(counts, on=group_cols, how="left")


def main() -> None:
    parser = argparse.ArgumentParser(description="Rescore existing stability-selected supports with multiple pair scorers.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--functions", nargs="+", default=["core_interaction_c025"])
    parser.add_argument("--methods", nargs="+", default=["feature_stability_var", "feature_edge_hybrid"])
    parser.add_argument("--samples", nargs="+", type=int, default=[512, 1024])
    parser.add_argument("--dimensions", nargs="+", type=int, default=[100])
    parser.add_argument("--top_m", nargs="+", type=int, default=[4])
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--max_rows", type=int, default=0)
    parser.add_argument("--test_samples", type=int, default=4096)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--pair_methods", nargs="+", default=["fd", "anova_abs", "anova_var", "fd_anova_hybrid"])
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--grid", type=int, default=0)
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--width_hidden", type=int, default=0)
    parser.add_argument("--lamb", type=float, default=None)
    parser.add_argument("--opt", default="LBFGS")
    parser.add_argument("--no_update_grid", action="store_true", default=True)
    parser.add_argument("--grid_update_num", type=int, default=10)
    parser.add_argument("--batch", type=int, default=-1)
    parser.add_argument("--variable_points", type=int, default=512)
    parser.add_argument("--fd_points", type=int, default=128)
    parser.add_argument("--fd_h", type=float, default=1e-2)
    parser.add_argument("--anova_points", type=int, default=64)
    parser.add_argument("--anova_background", type=int, default=64)
    parser.add_argument("--pred_batch_size", type=int, default=4096)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(args.input)
    selected = filter_rows(source, args)
    print(f"Using device={device}")
    print(f"Selected {len(selected)} source rows")

    rows: List[dict] = []
    detail_path = out_dir / "pair_rescore_detail.csv"
    summary_path = out_dir / "pair_rescore_summary.csv"
    for _, source_row in selected.iterrows():
        print(
            f"[RESCORE] method={source_row.get('method')} fn={source_row['function']} "
            f"n={int(source_row['samples'])} d={int(source_row['dimension'])} "
            f"top_m={int(source_row['top_m'])} seed={int(source_row['seed'])}",
            flush=True,
        )
        rows.extend(run_one(source_row, args, device))
        detail = pd.DataFrame(rows)
        detail.to_csv(detail_path, index=False)
        summarize(detail).to_csv(summary_path, index=False)

    detail = pd.DataFrame(rows)
    summary = summarize(detail)
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    if not summary.empty:
        print(summary.to_string(index=False))
    print(f"Wrote {detail_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
