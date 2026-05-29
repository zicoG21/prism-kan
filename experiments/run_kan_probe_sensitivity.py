from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import make_synthetic
from experiments.run_kan_native_innovation_loop import (
    aggregate_probe_scores,
    json_dumps,
    safe_edge_path_scores,
    safe_feature_score,
    select_support,
    top_vars,
    train_args,
)
from experiments.run_tuned_kan_recovery import (
    batch_predict,
    canonical_pairs,
    gradient_importance,
    mse_np,
    support_stats,
    train_kan,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="KAN-native support-probe sensitivity with cache keys that include all data/training hyperparameters."
    )
    parser.add_argument("--out_dir", default="results/workshop_review_tables/kan_probe_sensitivity")
    parser.add_argument("--function", default="core_interaction_c1")
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--test_samples", type=int, default=2048)
    parser.add_argument("--dimension", type=int, default=1000)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--nuisance_correlation", type=float, default=0.0)
    parser.add_argument("--n_correlated_proxies", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[730, 731, 732, 733])
    parser.add_argument("--methods", nargs="+", default=["feature_stability_var", "feature_edge_hybrid"])
    parser.add_argument("--top_ms", type=int, nargs="+", default=[6, 20, 50, 100, 250, 500, 1000])
    parser.add_argument("--width_hidden", type=int, default=8)
    parser.add_argument("--grid", type=int, default=5)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--lamb", type=float, default=1e-3)
    parser.add_argument("--probe_steps", type=int, default=35)
    parser.add_argument("--opt", default="LBFGS")
    parser.add_argument("--update_grid", action="store_true")
    parser.add_argument("--grid_update_num", type=int, default=10)
    parser.add_argument("--batch", type=int, default=-1)
    parser.add_argument("--probe_variable_points", type=int, default=512)
    parser.add_argument("--pred_batch_size", type=int, default=4096)
    parser.add_argument("--keep_top_pairs", type=int, default=160)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--force_probe", action="store_true")
    parser.add_argument(
        "--summarize_existing_only",
        action="store_true",
        help="Do not train missing probes; summarize the cache entries matching --seeds.",
    )
    return parser.parse_args()


def device_from_arg(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


def probe_key(args: argparse.Namespace, seed: int) -> str:
    parts = [
        args.function,
        f"n{args.samples}",
        f"d{args.dimension}",
        f"noise{args.noise:g}",
        f"rho{args.nuisance_correlation:g}",
        f"prox{args.n_correlated_proxies}",
        f"seed{int(seed)}",
        f"w{args.width_hidden}",
        f"grid{args.grid}",
        f"k{args.k}",
        f"lamb{args.lamb:g}",
        f"steps{args.probe_steps}",
        f"opt{args.opt}",
        f"ug{int(bool(args.update_grid))}",
    ]
    return "__".join(parts)


def true_structure(args: argparse.Namespace) -> tuple[tuple[int, ...], tuple[tuple[int, int], ...]]:
    data = make_synthetic(
        function_name=args.function,
        n_train=args.samples,
        n_test=args.test_samples,
        d=args.dimension,
        noise=args.noise,
        seed=0,
        standardize_target=True,
        nuisance_correlation=args.nuisance_correlation,
        n_correlated_proxies=args.n_correlated_proxies,
    )
    gt = data["ground_truth"]
    return tuple(int(v) for v in gt.active_variables), canonical_pairs(gt.interactions)


def load_cache(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def train_probe(args: argparse.Namespace, seed: int, device: str) -> dict:
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
    t0 = time.time()
    row = {
        "probe_key": probe_key(args, seed),
        "function": args.function,
        "samples": int(args.samples),
        "test_samples": int(args.test_samples),
        "dimension": int(args.dimension),
        "noise": float(args.noise),
        "nuisance_correlation": float(args.nuisance_correlation),
        "n_correlated_proxies": int(args.n_correlated_proxies),
        "seed": int(seed),
        "width_hidden": int(args.width_hidden),
        "grid": int(args.grid),
        "k": int(args.k),
        "lamb": float(args.lamb),
        "probe_steps": int(args.probe_steps),
        "opt": args.opt,
        "update_grid": bool(args.update_grid),
        "status": "ok",
        "error": "",
        "traceback": "",
        "runtime_sec": np.nan,
    }
    try:
        p_args = train_args(
            width_hidden=args.width_hidden,
            grid=args.grid,
            k=args.k,
            steps=args.probe_steps,
            lamb=args.lamb,
            opt=args.opt,
            update_grid=args.update_grid,
            grid_update_num=args.grid_update_num,
            batch=args.batch,
        )
        model = train_kan(
            data["X_train"],
            data["y_train"],
            data["X_test"],
            data["y_test"],
            p_args,
            seed=int(seed),
            device=device,
        )
        pred = batch_predict(model, data["X_test"], device=device, batch_size=args.pred_batch_size)
        grad_scores = gradient_importance(
            model,
            data["X_test"],
            device=device,
            points=args.probe_variable_points,
        )
        grad_scores = np.asarray(grad_scores, dtype=float)
        feature_scores = safe_feature_score(model, args.dimension)
        edge_scores, pair_scores, endpoint_mass = safe_edge_path_scores(model, args.dimension)
        top_pairs = sorted(pair_scores.items(), key=lambda kv: -float(kv[1]))[: args.keep_top_pairs]
        row.update(
            {
                "test_mse": mse_np(pred, data["y_test"]),
                "grad_scores": json_dumps(grad_scores.tolist()),
                "feature_scores": json_dumps(feature_scores.tolist()),
                "edge_var_scores": json_dumps(edge_scores.tolist()),
                "edge_endpoint_mass": json_dumps(endpoint_mass.tolist()),
                "edge_pair_scores_top": json_dumps([(int(i), int(j), float(v)) for (i, j), v in top_pairs]),
                "top_grad_variables": json_dumps(top_vars(grad_scores, min(12, args.dimension))),
                "top_feature_variables": json_dumps(top_vars(feature_scores, min(12, args.dimension))),
                "top_edge_variables": json_dumps(top_vars(edge_scores, min(12, args.dimension))),
            }
        )
    except Exception as exc:
        row.update({"status": "failed", "error": repr(exc), "traceback": traceback.format_exc()})
    row["runtime_sec"] = float(time.time() - t0)
    return row


def ensure_probes(args: argparse.Namespace, probe_path: Path, device: str) -> pd.DataFrame:
    existing = load_cache(probe_path)
    rows = existing.to_dict("records") if not existing.empty else []
    existing_keys = {str(r.get("probe_key")) for r in rows}
    for seed in args.seeds:
        key = probe_key(args, int(seed))
        if key in existing_keys and not args.force_probe:
            continue
        if getattr(args, "summarize_existing_only", False):
            continue
        print(f"[PROBE] {key}", flush=True)
        rows.append(train_probe(args, int(seed), device))
        pd.DataFrame(rows).to_csv(probe_path, index=False)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    keys = {probe_key(args, int(seed)) for seed in args.seeds}
    mask = df["probe_key"].astype(str).isin(keys) & (df["status"].astype(str) == "ok")
    return df[mask].copy()


def score_rank_diagnostics(score: np.ndarray, true_vars: tuple[int, ...], true_pairs: tuple[tuple[int, int], ...]) -> dict:
    score = np.asarray(score, dtype=float).reshape(-1)
    order = sorted(range(len(score)), key=lambda j: (-float(score[j]), int(j)))
    rank = {int(v): i + 1 for i, v in enumerate(order)}
    endpoints = sorted({v for pair in true_pairs for v in pair})
    nuisance = [j for j in range(len(score)) if j not in set(true_vars)]
    endpoint_scores = [float(score[j]) for j in endpoints] if endpoints else []
    endpoint_ranks = [int(rank[j]) for j in endpoints] if endpoints else []
    return {
        "true_endpoint_rank_mean": float(np.mean(endpoint_ranks)) if endpoint_ranks else np.nan,
        "true_endpoint_rank_worst": float(np.max(endpoint_ranks)) if endpoint_ranks else np.nan,
        "true_endpoint_score_mean": float(np.mean(endpoint_scores)) if endpoint_scores else np.nan,
        "max_nuisance_score": float(np.max(score[nuisance])) if nuisance else np.nan,
        "endpoint_minus_max_nuisance": (
            float(np.mean(endpoint_scores) - np.max(score[nuisance]))
            if endpoint_scores and nuisance
            else np.nan
        ),
    }


def summarize(detail: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "function",
        "samples",
        "dimension",
        "noise",
        "nuisance_correlation",
        "n_correlated_proxies",
        "width_hidden",
        "grid",
        "k",
        "lamb",
        "probe_steps",
        "method",
        "top_m",
    ]
    numeric = [
        "screen_contains_all_true_vars",
        "screen_true_var_recall",
        "screen_contains_all_interaction_endpoints",
        "screen_interaction_endpoint_recall",
        "screen_contains_true_interactions",
        "true_endpoint_rank_mean",
        "true_endpoint_rank_worst",
        "true_endpoint_score_mean",
        "max_nuisance_score",
        "endpoint_minus_max_nuisance",
    ]
    out = detail.groupby(group_cols, dropna=False)[numeric].agg(["mean", "std"]).reset_index()
    out.columns = ["_".join(str(x) for x in col if x != "").rstrip("_") for col in out.columns]
    counts = detail.groupby(group_cols, dropna=False).size().reset_index(name="num_support_evals")
    return out.merge(counts, on=group_cols, how="left")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = device_from_arg(args.device)
    print(f"Using device={device}")
    print(f"Writing under {out_dir}")

    probe_path = out_dir / "probe_cache.csv"
    probes = ensure_probes(args, probe_path, device)
    true_vars, true_pairs = true_structure(args)

    rows = []
    if not probes.empty:
        for heldout in [None, *args.seeds]:
            if heldout is None:
                stable = probes.copy()
                heldout_label = "all"
            else:
                stable = probes[pd.to_numeric(probes["seed"], errors="coerce") != int(heldout)].copy()
                heldout_label = str(int(heldout))
            if stable.empty:
                continue
            agg = aggregate_probe_scores(stable, args.dimension)
            for method, top_m in itertools.product(args.methods, args.top_ms):
                support, meta = select_support(method, agg, int(top_m), args.dimension)
                stats = support_stats(np.asarray(support, dtype=int), true_vars, true_pairs)
                diag = score_rank_diagnostics(
                    np.asarray(meta.get("selection_score", np.zeros(args.dimension)), dtype=float),
                    true_vars,
                    true_pairs,
                )
                rows.append(
                    {
                        "function": args.function,
                        "samples": int(args.samples),
                        "dimension": int(args.dimension),
                        "noise": float(args.noise),
                        "nuisance_correlation": float(args.nuisance_correlation),
                        "n_correlated_proxies": int(args.n_correlated_proxies),
                        "width_hidden": int(args.width_hidden),
                        "grid": int(args.grid),
                        "k": int(args.k),
                        "lamb": float(args.lamb),
                        "probe_steps": int(args.probe_steps),
                        "method": method,
                        "top_m": int(top_m),
                        "heldout_seed": heldout_label,
                        "num_probe_rows": int(len(stable)),
                        "selected_screen_features": json.dumps([int(v) for v in support]),
                        "top_selection_variables": json.dumps([int(v) for v in meta.get("top_selection_variables", [])]),
                        "top_edge_pairs": json.dumps(meta.get("top_edge_pairs", [])),
                        **stats,
                        **diag,
                    }
                )

    detail = pd.DataFrame(rows)
    detail.to_csv(out_dir / "support_sensitivity_detail.csv", index=False)
    if detail.empty:
        summary = pd.DataFrame()
    else:
        source = detail[detail["heldout_seed"].astype(str).ne("all")].copy()
        if source.empty:
            source = detail
        summary = summarize(source)
    summary.to_csv(out_dir / "support_sensitivity_summary.csv", index=False)
    print(summary.to_string(index=False) if not summary.empty else "No support rows.")


if __name__ == "__main__":
    main()
