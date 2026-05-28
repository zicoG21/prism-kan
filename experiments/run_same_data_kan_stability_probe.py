from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import make_synthetic
from experiments.run_kan_native_innovation_loop import (
    aggregate_probe_scores,
    safe_edge_path_scores,
    safe_feature_score,
    select_support,
    train_args,
)
from experiments.run_tuned_kan_recovery import (
    anova_pair_scores,
    batch_predict,
    canonical_pairs,
    evaluate_interaction_recovery,
    mse_np,
    support_stats,
    train_kan,
    local_to_full_pair_scores,
)


def json_dumps(obj) -> str:
    return json.dumps(obj, sort_keys=True)


def run_probe_on_fixed_data(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    probe_seed: int,
    outer_seed: int,
    args: argparse.Namespace,
    device: str,
) -> dict:
    rng = np.random.default_rng(outer_seed * 1000003 + probe_seed)
    if args.resample == "bootstrap":
        idx = rng.choice(len(X_train), size=len(X_train), replace=True)
    elif args.resample == "subsample":
        size = max(4, int(round(float(args.subsample_frac) * len(X_train))))
        idx = rng.choice(len(X_train), size=size, replace=False)
    elif args.resample == "none":
        idx = np.arange(len(X_train))
    else:
        raise ValueError(f"Unknown resample={args.resample}")

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
        X_train[idx],
        y_train[idx],
        X_test,
        y_test,
        p_args,
        seed=int(probe_seed),
        device=device,
    )
    pred = batch_predict(model, X_test, device=device, batch_size=args.pred_batch_size)
    feature_scores = safe_feature_score(model, X_train.shape[1])
    edge_scores, pair_scores, endpoint_mass = safe_edge_path_scores(model, X_train.shape[1])
    top_pairs = sorted(pair_scores.items(), key=lambda kv: -float(kv[1]))[: args.keep_top_pairs]
    return {
        "seed": int(probe_seed),
        "status": "ok",
        "test_mse": mse_np(pred, y_test),
        "feature_scores": json_dumps(feature_scores.tolist()),
        "edge_var_scores": json_dumps(edge_scores.tolist()),
        "edge_endpoint_mass": json_dumps(endpoint_mass.tolist()),
        "edge_pair_scores_top": json_dumps([(int(i), int(j), float(v)) for (i, j), v in top_pairs]),
    }


def run_one_outer(args: argparse.Namespace, outer_seed: int, device: str):
    t0 = time.time()
    data = make_synthetic(
        function_name=args.function,
        n_train=args.samples,
        n_test=args.test_samples,
        d=args.dimension,
        noise=args.noise,
        seed=int(outer_seed),
        standardize_target=True,
        nuisance_correlation=args.nuisance_correlation,
        n_correlated_proxies=args.n_correlated_proxies,
    )
    X_train = data["X_train"].astype(np.float32)
    y_train = data["y_train"].astype(np.float32)
    X_test = data["X_test"].astype(np.float32)
    y_test = data["y_test"].astype(np.float32)
    gt = data["ground_truth"]
    true_vars = tuple(int(v) for v in gt.active_variables)
    true_pairs = canonical_pairs(gt.interactions)

    probe_rows = []
    for r in range(int(args.R)):
        probe_seed = int(args.probe_seed_offset + outer_seed * 1000 + r)
        try:
            probe_rows.append(
                run_probe_on_fixed_data(
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    probe_seed=probe_seed,
                    outer_seed=int(outer_seed),
                    args=args,
                    device=device,
                )
            )
        except Exception as exc:
            probe_rows.append({"seed": int(probe_seed), "status": "failed", "error": repr(exc)})

    probes = pd.DataFrame([r for r in probe_rows if r.get("status") == "ok"])
    agg = aggregate_probe_scores(probes, int(args.dimension))
    probe_mse = pd.to_numeric(probes.get("test_mse", pd.Series(dtype=float)), errors="coerce")
    base_row = {
        "function": args.function,
        "resample": args.resample,
        "outer_seed": int(outer_seed),
        "samples": int(args.samples),
        "test_samples": int(args.test_samples),
        "dimension": int(args.dimension),
        "noise": float(args.noise),
        "nuisance_correlation": float(args.nuisance_correlation),
        "n_correlated_proxies": int(args.n_correlated_proxies),
        "R": int(args.R),
        "top_m": int(args.top_m),
        "probe_steps": int(args.probe_steps),
        "refit_steps": int(args.refit_steps),
        "num_ok_probes": int(len(probes)),
        "probe_test_mse_mean": float(probe_mse.mean()) if len(probe_mse) else np.nan,
        "probe_test_mse_std": float(probe_mse.std()) if len(probe_mse) else np.nan,
        "probe_test_mse_min": float(probe_mse.min()) if len(probe_mse) else np.nan,
        "probe_test_mse_max": float(probe_mse.max()) if len(probe_mse) else np.nan,
        "true_variables": list(true_vars),
        "true_interactions": list(true_pairs),
    }

    methods = list(getattr(args, "methods", None) or [args.method])
    rows = []
    for method in methods:
        support, meta = select_support(method, agg, int(args.top_m), int(args.dimension))
        support = np.asarray(sorted(support), dtype=int)
        row = dict(base_row)
        row.update(
            {
                "method": method,
                "selected_screen_features": support.tolist(),
                "top_selection_variables": meta.get("top_selection_variables", []),
                "runtime_sec": float(time.time() - t0),
            }
        )
        row.update(support_stats(support, true_vars, true_pairs))

        if args.skip_refit or len(probes) == 0:
            row.update({"refit_test_mse": np.nan, "interaction_f1": np.nan})
            rows.append(row)
            continue

        try:
            r_args = SimpleNamespace(
                width_hidden=args.width_hidden,
                grid=args.grid,
                k=args.k,
                steps=args.refit_steps,
                lamb=args.lamb,
                opt=args.opt,
                update_grid=args.update_grid,
                grid_update_num=args.grid_update_num,
                batch=args.batch,
            )
            model = train_kan(
                X_train[:, support],
                y_train,
                X_test[:, support],
                y_test,
                r_args,
                seed=int(args.refit_seed_offset + outer_seed),
                device=device,
            )
            pred = batch_predict(model, X_test[:, support], device=device, batch_size=args.pred_batch_size)
            local_pair_scores = anova_pair_scores(
                model,
                X_test[:, support],
                device=device,
                points=args.anova_points,
                background=args.anova_background,
                batch_size=args.pred_batch_size,
                score="abs",
            )
            full_pair_scores = local_to_full_pair_scores(local_pair_scores, support, int(args.dimension))
            row.update({"refit_test_mse": mse_np(pred, y_test), "pair_score": "anova_abs"})
            row.update(evaluate_interaction_recovery(full_pair_scores, true_pairs))
        except Exception as exc:
            row.update({"refit_error": repr(exc), "refit_test_mse": np.nan, "interaction_f1": np.nan})
        row["runtime_sec"] = float(time.time() - t0)
        rows.append(row)
    return rows if len(rows) != 1 else rows[0]


def summarize(detail: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "function",
        "method",
        "resample",
        "samples",
        "dimension",
        "noise",
        "nuisance_correlation",
        "R",
        "top_m",
    ]
    numeric = [
        "screen_contains_all_true_vars",
        "screen_true_var_recall",
        "screen_contains_all_interaction_endpoints",
        "screen_interaction_endpoint_recall",
        "screen_contains_true_interactions",
        "interaction_f1",
        "true_interaction_rank_mean",
        "probe_test_mse_mean",
        "probe_test_mse_min",
        "refit_test_mse",
        "runtime_sec",
    ]
    numeric = [col for col in numeric if col in detail.columns]
    out = detail.groupby(group_cols, dropna=False)[numeric].agg(["mean", "std"]).reset_index()
    out.columns = ["_".join(str(v) for v in col if v != "").rstrip("_") for col in out.columns]
    counts = detail.groupby(group_cols, dropna=False).size().reset_index(name="num_runs")
    if "interaction_f1" in detail.columns:
        successes = (
            detail.groupby(group_cols, dropna=False)["interaction_f1"]
            .sum()
            .reset_index(name="top1_successes")
        )
    else:
        successes = detail.groupby(group_cols, dropna=False).size().reset_index(name="top1_successes")
        successes["top1_successes"] = np.nan
    exact = (
        detail.groupby(group_cols, dropna=False)["screen_contains_all_true_vars"]
        .sum()
        .reset_index(name="exact_support_successes")
    )
    endpoints = (
        detail.groupby(group_cols, dropna=False)["screen_contains_all_interaction_endpoints"]
        .sum()
        .reset_index(name="endpoint_successes")
    )
    return out.merge(counts, on=group_cols).merge(successes, on=group_cols).merge(exact, on=group_cols).merge(endpoints, on=group_cols)


def main() -> None:
    parser = argparse.ArgumentParser(description="Same-dataset KAN stability probe for fair-data-budget checks.")
    parser.add_argument("--function", default="core_interaction_c025")
    parser.add_argument("--samples", type=int, nargs="+", default=[512, 1024])
    parser.add_argument("--test_samples", type=int, default=2048)
    parser.add_argument("--dimension", type=int, default=100)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--nuisance_correlation", type=float, default=0.0)
    parser.add_argument("--n_correlated_proxies", type=int, default=0)
    parser.add_argument("--method", default="feature_edge_hybrid")
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--top_m", type=int, default=4)
    parser.add_argument("--R", type=int, default=8)
    parser.add_argument("--resample", choices=["none", "bootstrap", "subsample"], default="bootstrap")
    parser.add_argument("--subsample_frac", type=float, default=0.8)
    parser.add_argument("--outer_seeds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--probe_seed_offset", type=int, default=50000)
    parser.add_argument("--refit_seed_offset", type=int, default=90000)
    parser.add_argument("--width_hidden", type=int, default=8)
    parser.add_argument("--grid", type=int, default=5)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--lamb", type=float, default=1e-3)
    parser.add_argument("--probe_steps", type=int, default=35)
    parser.add_argument("--refit_steps", type=int, default=50)
    parser.add_argument("--opt", default="LBFGS")
    parser.add_argument("--update_grid", action="store_true")
    parser.add_argument("--grid_update_num", type=int, default=5)
    parser.add_argument("--batch", type=int, default=-1)
    parser.add_argument("--pred_batch_size", type=int, default=4096)
    parser.add_argument("--keep_top_pairs", type=int, default=300)
    parser.add_argument("--anova_points", type=int, default=64)
    parser.add_argument("--anova_background", type=int, default=64)
    parser.add_argument("--skip_refit", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--out_dir", default="results/workshop_review_tables/same_data_kan_stability_probe")
    parser.add_argument(
        "--no_resume",
        action="store_true",
        help="Do not reuse existing detail CSV rows in out_dir.",
    )
    args = parser.parse_args()

    if args.device == "auto":
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
    else:
        device = args.device

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    detail_path = out_dir / "same_data_kan_stability_detail.csv"
    summary_path = out_dir / "same_data_kan_stability_summary.csv"

    rows = []
    completed = set()
    if detail_path.exists() and not args.no_resume:
        existing = pd.read_csv(detail_path)
        if len(existing):
            rows = existing.to_dict("records")
            for _, row in existing.iterrows():
                completed.add((int(row["samples"]), int(row["outer_seed"]), str(row["method"])))

    for n in args.samples:
        for seed in args.outer_seeds:
            local_args = argparse.Namespace(**vars(args))
            local_args.samples = int(n)
            methods = list(getattr(local_args, "methods", None) or [local_args.method])
            if all((int(n), int(seed), str(method)) in completed for method in methods):
                print(f"[SKIP] n={n} outer_seed={seed} methods={','.join(map(str, methods))}", flush=True)
                continue
            print(f"[RUN] n={n} outer_seed={seed} R={args.R} resample={args.resample}", flush=True)
            result = run_one_outer(local_args, int(seed), device)
            result_rows = result if isinstance(result, list) else [result]
            filtered_rows = []
            for row in result_rows:
                key = (int(row["samples"]), int(row["outer_seed"]), str(row["method"]))
                if key in completed:
                    print(f"[SKIP-WRITE] n={key[0]} outer_seed={key[1]} method={key[2]}", flush=True)
                    continue
                completed.add(key)
                filtered_rows.append(row)
            rows.extend(filtered_rows)

            # Checkpoint after every outer seed so interrupted long sweeps can resume.
            detail = pd.DataFrame(rows)
            detail.to_csv(detail_path, index=False)
            if len(detail):
                summarize(detail).to_csv(summary_path, index=False)
        else:
            continue

    detail = pd.DataFrame(rows)
    detail.to_csv(detail_path, index=False)
    summary = summarize(detail)
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
