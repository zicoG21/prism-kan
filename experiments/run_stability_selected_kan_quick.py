from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.data import make_synthetic
from experiments.run_tuned_kan_recovery import (
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
    train_kan,
)


Pair = Tuple[int, int]


METHOD_LABELS = {
    "raw": "Raw",
    "ss_kan_variable": "SS-KAN-V",
    "ss_kan_pair": "SS-KAN-P",
    "rf": "RF",
    "oracle_support": "Oracle",
    "random": "Random",
    "exclude_interaction": "Exclude",
}


def parse_literal(value, default):
    if isinstance(value, (list, tuple)):
        return value
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default
    text = str(value).strip()
    if text == "" or text.lower() == "nan":
        return default
    try:
        return ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return default


def parse_pairs(value) -> List[Pair]:
    pairs = []
    for item in parse_literal(value, []):
        if isinstance(item, (list, tuple)) and len(item) == 2:
            pairs.append(tuple(sorted((int(item[0]), int(item[1])))))
    return pairs


def function_to_c(function_name: str) -> float:
    mapping = {
        "core_interaction_c01": 0.10,
        "core_interaction_c025": 0.25,
        "core_interaction_c05": 0.50,
        "core_interaction_c1": 1.00,
    }
    return mapping.get(function_name, np.nan)


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        return scores
    mx = float(np.max(np.abs(scores)))
    if mx <= 0 or not np.isfinite(mx):
        return np.zeros_like(scores, dtype=float)
    return scores / mx


def load_raw_stability_rows(args, function_name: str, eval_seed: int) -> pd.DataFrame:
    detail_path = Path(args.existing_detail_dir) / f"{function_name}_n{args.samples}_d{args.dimension}_detail.csv"
    if not detail_path.exists():
        raise FileNotFoundError(f"Missing raw detail CSV: {detail_path}")

    df = pd.read_csv(detail_path)
    raw = df[(df["screen_mode"].astype(str) == "raw") & (df["status"].astype(str) == "ok")].copy()
    raw = raw.sort_values("seed")
    if args.leave_one_out:
        raw = raw[raw["seed"].astype(int) != int(eval_seed)].copy()
    if args.stability_repeats > 0:
        raw = raw.head(args.stability_repeats)
    if raw.empty:
        raise RuntimeError(f"No raw stability rows for {function_name}, eval_seed={eval_seed}")
    return raw


def variable_stability(raw: pd.DataFrame, d: int) -> tuple[np.ndarray, np.ndarray]:
    counts = np.zeros(d, dtype=float)
    mean_scores = np.zeros(d, dtype=float)
    score_runs = 0

    for _, row in raw.iterrows():
        selected = parse_literal(row.get("selected_variables"), [])
        for v in selected:
            v = int(v)
            if 0 <= v < d:
                counts[v] += 1.0

        scores = parse_literal(row.get("importance_scores"), [])
        if len(scores) == d:
            mean_scores += normalize_scores(np.asarray(scores, dtype=float))
            score_runs += 1

    freq = counts / max(len(raw), 1)
    if score_runs > 0:
        mean_scores = mean_scores / score_runs
    return freq, mean_scores


def pair_stability(raw: pd.DataFrame) -> Dict[Pair, float]:
    counts: Dict[Pair, float] = {}
    for _, row in raw.iterrows():
        pairs = parse_pairs(row.get("selected_interactions"))
        for pair in pairs:
            counts[pair] = counts.get(pair, 0.0) + 1.0
    return {pair: count / max(len(raw), 1) for pair, count in counts.items()}


def ranked_variables(freq: np.ndarray, mean_scores: np.ndarray) -> List[int]:
    return sorted(
        range(len(freq)),
        key=lambda v: (-float(freq[v]), -float(mean_scores[v]), int(v)),
    )


def select_support(raw: pd.DataFrame, method: str, d: int, top_m: int) -> tuple[np.ndarray, dict]:
    var_freq, mean_scores = variable_stability(raw, d)
    var_rank = ranked_variables(var_freq, mean_scores)
    pair_freq = pair_stability(raw)
    pair_rank = sorted(pair_freq, key=lambda p: (-float(pair_freq[p]), int(p[0]), int(p[1])))

    selected: List[int] = []
    seen = set()

    if method == "ss_kan_pair":
        for i, j in pair_rank:
            for v in (i, j):
                if v not in seen and len(selected) < top_m:
                    selected.append(v)
                    seen.add(v)
            if len(selected) >= top_m:
                break
    elif method != "ss_kan_variable":
        raise ValueError(f"Unknown stability method: {method}")

    for v in var_rank:
        if len(selected) >= top_m:
            break
        if v not in seen:
            selected.append(int(v))
            seen.add(int(v))

    support = np.array(sorted(selected[:top_m]), dtype=int)
    meta = {
        "selected_support": support.tolist(),
        "variable_stability_scores": var_freq.tolist(),
        "variable_mean_importance_scores": mean_scores.tolist(),
        "top_variables_by_stability": var_rank[:top_m],
        "pair_stability_scores": {str(k): float(v) for k, v in pair_freq.items()},
        "top_pairs_by_stability": [str(p) for p in pair_rank[: max(1, top_m)]],
        "num_stability_rows": int(len(raw)),
        "stability_seed_pool": [int(s) for s in raw["seed"].tolist()],
    }
    return support, meta


def make_train_args(args) -> SimpleNamespace:
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


def support_stats(selected_features: Sequence[int], true_vars: Sequence[int], true_interactions: Sequence[Pair]) -> dict:
    selected = set(int(v) for v in selected_features)
    true_var_set = set(int(v) for v in true_vars)
    endpoints = set()
    for i, j in true_interactions:
        endpoints.add(int(i))
        endpoints.add(int(j))
    return {
        "effective_dim": len(selected_features),
        "screen_contains_all_true_vars": int(true_var_set.issubset(selected)) if true_var_set else np.nan,
        "screen_true_var_recall": len(true_var_set & selected) / len(true_var_set) if true_var_set else np.nan,
        "screen_contains_all_interaction_endpoints": int(endpoints.issubset(selected)) if endpoints else np.nan,
        "screen_interaction_endpoint_recall": len(endpoints & selected) / len(endpoints) if endpoints else np.nan,
        "screen_contains_true_interactions": int(
            all(int(i) in selected and int(j) in selected for i, j in true_interactions)
        ) if true_interactions else np.nan,
    }


def run_stability_refit(args, function_name: str, method: str, seed: int, device: str) -> dict:
    raw = load_raw_stability_rows(args, function_name, eval_seed=seed)
    support, meta = select_support(raw, method=method, d=args.dimension, top_m=args.top_m)

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

    Xtr = data["X_train"][:, support]
    Xte = data["X_test"][:, support]
    train_args = make_train_args(args)

    row = {
        "model": "KAN_stability_selected",
        "function": function_name,
        "interaction_strength": function_to_c(function_name),
        "screen_mode": method,
        "seed": seed,
        "samples": args.samples,
        "test_samples": args.test_samples,
        "dimension": args.dimension,
        "noise": args.noise,
        "top_m": args.top_m,
        "selected_screen_features": support.tolist(),
        "screen_score_type": "kan_stability",
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
        **meta,
    }
    row.update(support_stats(support, true_vars, true_interactions))

    try:
        model = train_kan(Xtr, data["y_train"], Xte, data["y_test"], train_args, seed=seed, device=device)
        train_pred = batch_predict(model, Xtr, device=device, batch_size=args.pred_batch_size)
        test_pred = batch_predict(model, Xte, device=device, batch_size=args.pred_batch_size)

        local_var_scores = gradient_importance(model, Xte, device=device, points=args.variable_points)
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

        row.update({
            "status": "ok",
            "error": "",
            "train_mse": mse_np(train_pred, data["y_train"]),
            "test_mse": mse_np(test_pred, data["y_test"]),
            "variable_method": "grad",
            "interaction_method": "fd",
            "importance_scores": full_var_scores.tolist(),
        })
        var_eval = evaluate_variable_recovery(full_var_scores, true_vars)
        row.update(var_eval)
        row.update(endpoint_recovery(var_eval["selected_variables"], true_interactions, "explain"))
        row.update(evaluate_interaction_recovery(full_pair_scores, true_interactions))
    except Exception as exc:
        row.update({
            "status": "failed",
            "error": repr(exc),
            "train_mse": np.nan,
            "test_mse": np.nan,
            "variable_f1": np.nan,
            "interaction_f1": np.nan,
            "selected_variables": [],
            "selected_interactions": [],
        })
        print(f"[WARN] failed {function_name} {method} seed={seed}: {exc}")
    return row


def load_baseline_rows(args) -> pd.DataFrame:
    frames = []
    for function_name in args.functions:
        path = Path(args.existing_detail_dir) / f"{function_name}_n{args.samples}_d{args.dimension}_detail.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df = df[df["screen_mode"].isin(args.baseline_methods)].copy()
        df = df[df["seed"].astype(int).isin([int(s) for s in args.seeds])].copy()
        df["interaction_strength"] = function_to_c(function_name)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    ok = df[df["status"].astype(str) == "ok"].copy()
    numeric_cols = [
        "train_mse", "test_mse", "effective_dim",
        "screen_contains_all_true_vars", "screen_true_var_recall",
        "screen_contains_all_interaction_endpoints", "screen_interaction_endpoint_recall",
        "screen_contains_true_interactions",
        "explain_contains_all_interaction_endpoints", "explain_interaction_endpoint_recall",
        "selected_interaction_endpoint_recall", "selected_interaction_contains_all_endpoints",
        "variable_f1", "variable_auroc", "variable_auprc",
        "interaction_f1", "true_interaction_score_mean", "max_nontrue_interaction_score",
        "true_interaction_mean_score_margin", "true_interaction_beats_all_false",
    ]
    for col in numeric_cols:
        if col in ok.columns:
            ok[col] = pd.to_numeric(ok[col], errors="coerce")

    group_cols = ["function", "interaction_strength", "screen_mode", "dimension", "samples"]
    agg = {}
    for col in numeric_cols:
        if col in ok.columns:
            if col in {
                "train_mse", "test_mse", "variable_f1", "variable_auroc",
                "variable_auprc", "interaction_f1", "true_interaction_score_mean",
                "max_nontrue_interaction_score", "true_interaction_mean_score_margin",
            }:
                agg[col] = ["mean", "std"]
            else:
                agg[col] = ["mean"]
    summary = ok.groupby(group_cols, dropna=False).agg(agg).reset_index()
    summary.columns = ["_".join(str(x) for x in col if x != "").rstrip("_") for col in summary.columns]
    counts = ok.groupby(group_cols, dropna=False).size().reset_index(name="num_runs")
    return summary.merge(counts, on=group_cols, how="left")


def plot_summary(summary: pd.DataFrame, out_path: Path) -> None:
    plot_methods = ["raw", "ss_kan_variable", "ss_kan_pair", "rf", "oracle_support"]
    funcs = sorted(summary["function"].unique(), key=function_to_c)
    x = np.arange(len(funcs))
    width = 0.15

    plt.figure(figsize=(10.5, 4.8))
    for idx, method in enumerate(plot_methods):
        vals = []
        for fn in funcs:
            hit = summary[(summary["function"] == fn) & (summary["screen_mode"] == method)]
            vals.append(float(hit["interaction_f1_mean"].iloc[0]) if not hit.empty else np.nan)
        plt.bar(
            x + (idx - (len(plot_methods) - 1) / 2) * width,
            vals,
            width=width,
            label=METHOD_LABELS.get(method, method),
        )

    labels = [f"c={function_to_c(fn):g}" for fn in funcs]
    plt.xticks(x, labels)
    plt.ylim(0, 1.05)
    plt.ylabel("Interaction F1")
    plt.xlabel("Interaction strength")
    plt.title("Quick stability-selected KAN check (d=100, n=1024)")
    plt.legend(ncol=3, fontsize=8)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=250)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--functions", nargs="+", default=[
        "core_interaction_c01",
        "core_interaction_c025",
        "core_interaction_c05",
        "core_interaction_c1",
    ])
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--test_samples", type=int, default=4096)
    parser.add_argument("--dimension", type=int, default=100)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--stability_repeats", type=int, default=10)
    parser.add_argument("--leave_one_out", action="store_true", default=True)
    parser.add_argument("--no_leave_one_out", dest="leave_one_out", action="store_false")
    parser.add_argument("--stability_methods", nargs="+", default=["ss_kan_variable", "ss_kan_pair"])
    parser.add_argument("--baseline_methods", nargs="+", default=[
        "raw", "rf", "oracle_support", "random", "exclude_interaction"
    ])
    parser.add_argument("--existing_detail_dir", default="results/hard_regime/details")

    parser.add_argument("--grid", type=int, default=5)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--width_hidden", type=int, default=8)
    parser.add_argument("--lamb", type=float, default=0.001)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--opt", default="LBFGS")
    parser.add_argument("--no_update_grid", action="store_true", default=True)
    parser.add_argument("--grid_update_num", type=int, default=10)
    parser.add_argument("--batch", type=int, default=-1)
    parser.add_argument("--top_m", type=int, default=4)
    parser.add_argument("--variable_points", type=int, default=512)
    parser.add_argument("--fd_points", type=int, default=512)
    parser.add_argument("--fd_h", type=float, default=0.01)
    parser.add_argument("--pred_batch_size", type=int, default=4096)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")

    parser.add_argument("--out", default="results/stability_kan/quick_d100_n1024_detail.csv")
    parser.add_argument("--summary_out", default="results/stability_kan/quick_d100_n1024_summary.csv")
    parser.add_argument("--fig_out", default="results/stability_kan/figures/quick_d100_n1024_interaction_f1.png")
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device={device}")

    rows = []
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for fn in args.functions:
        for method in args.stability_methods:
            for seed in args.seeds:
                print(f"[RUN] function={fn} method={method} seed={seed}")
                rows.append(run_stability_refit(args, fn, method, seed, device))
                pd.DataFrame(rows).to_csv(out_path, index=False)

    stability_df = pd.DataFrame(rows)
    baseline_df = load_baseline_rows(args)
    combined = pd.concat([baseline_df, stability_df], ignore_index=True, sort=False)
    combined.to_csv(out_path, index=False)

    summary = summarize(combined)
    summary_path = Path(args.summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    plot_summary(summary, Path(args.fig_out))

    cols = [
        "function", "screen_mode", "test_mse_mean", "variable_f1_mean",
        "explain_interaction_endpoint_recall_mean", "interaction_f1_mean",
        "true_interaction_mean_score_margin_mean", "num_runs",
    ]
    print(summary[[c for c in cols if c in summary.columns]].to_string(index=False))
    print(f"Saved detail: {out_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved figure: {args.fig_out}")


if __name__ == "__main__":
    main()
