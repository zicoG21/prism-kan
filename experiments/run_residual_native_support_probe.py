from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import make_synthetic
from experiments.run_kan_native_innovation_loop import (
    METHODS,
    Setting,
    aggregate_probe_scores,
    combine_scores,
    json_dumps,
    normalize_score,
    safe_edge_path_scores,
    safe_feature_score,
    select_support,
    top_vars,
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
    parser = argparse.ArgumentParser(description="Residual KAN-native support probe after subtracting stable additive variables.")
    parser.add_argument("--out_dir", default="results/formula_aware_pair_scoring/residual_native_probe")
    parser.add_argument("--base_probe_cache", default="")
    parser.add_argument("--functions", nargs="+", default=["core_interaction_c05"])
    parser.add_argument("--samples", nargs="+", type=int, default=[1024])
    parser.add_argument("--dimensions", nargs="+", type=int, default=[500])
    parser.add_argument("--primary_method", default="feature_edge_hybrid")
    parser.add_argument("--primary_m", type=int, default=2)
    parser.add_argument("--top_ms", nargs="+", type=int, default=[4, 6, 8])
    parser.add_argument("--residual_methods", nargs="+", default=["feature_stability_var", "feature_edge_hybrid", "grad_stability_var"])
    parser.add_argument("--probe_seeds", nargs="+", type=int, default=[320, 321, 322, 323])
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--nuisance_correlation", type=float, default=0.0)
    parser.add_argument("--n_correlated_proxies", type=int, default=0)
    parser.add_argument("--test_samples", type=int, default=2048)
    parser.add_argument("--grid", type=int, default=5)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--width_hidden", type=int, default=8)
    parser.add_argument("--lamb", type=float, default=0.001)
    parser.add_argument("--additive_steps", type=int, default=80)
    parser.add_argument("--residual_probe_steps", type=int, default=25)
    parser.add_argument("--opt", default="LBFGS")
    parser.add_argument("--update_grid", action="store_true")
    parser.add_argument("--no_update_grid", dest="update_grid", action="store_false")
    parser.set_defaults(update_grid=False)
    parser.add_argument("--grid_update_num", type=int, default=10)
    parser.add_argument("--batch", type=int, default=-1)
    parser.add_argument("--probe_variable_points", type=int, default=512)
    parser.add_argument("--pred_batch_size", type=int, default=4096)
    parser.add_argument("--keep_top_pairs", type=int, default=120)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def device_from_arg(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


def train_args(args: argparse.Namespace, steps: int) -> SimpleNamespace:
    return SimpleNamespace(
        width_hidden=args.width_hidden,
        grid=args.grid,
        k=args.k,
        steps=int(steps),
        lamb=args.lamb,
        opt=args.opt,
        update_grid=args.update_grid,
        grid_update_num=args.grid_update_num,
        batch=args.batch,
    )


def load_base_probes(path: str, setting: Setting, seeds: Sequence[int] | None = None) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    mask = (
        df["function"].astype(str).eq(setting.function)
        & pd.to_numeric(df["samples"], errors="coerce").eq(setting.samples)
        & pd.to_numeric(df["dimension"], errors="coerce").eq(setting.dimension)
        & df["status"].astype(str).eq("ok")
    )
    if seeds:
        mask &= pd.to_numeric(df["seed"], errors="coerce").isin([int(s) for s in seeds])
    return df[mask].copy()


def select_primary_support(args: argparse.Namespace, setting: Setting) -> tuple[list[int], dict]:
    probes = load_base_probes(args.base_probe_cache, setting)
    if probes.empty:
        return list(range(args.primary_m)), {
            "primary_source": "fallback_first_variables",
            "primary_num_probe_rows": 0,
            "primary_top_variables": list(range(args.primary_m)),
        }
    agg = aggregate_probe_scores(probes, setting.dimension)
    support, meta = select_support(args.primary_method, agg, args.primary_m, setting.dimension)
    return sorted(int(v) for v in support), {
        "primary_source": args.primary_method,
        "primary_num_probe_rows": int(len(probes)),
        "primary_top_variables": [int(v) for v in meta.get("top_selection_variables", [])],
    }


def residual_probe_key(
    function_name: str,
    samples: int,
    dimension: int,
    primary: Sequence[int],
    seed: int,
    noise: float = 0.0,
    nuisance_correlation: float = 0.0,
    n_correlated_proxies: int = 0,
) -> str:
    primary_key = "-".join(str(int(v)) for v in primary)
    return (
        f"{function_name}__n{samples}__d{dimension}__noise{noise:g}"
        f"__rho{nuisance_correlation:g}__prox{int(n_correlated_proxies)}"
        f"__primary{primary_key}__seed{seed}"
    )


def train_residual_probe(
    *,
    setting: Setting,
    primary_support: Sequence[int],
    seed: int,
    args: argparse.Namespace,
    device: str,
) -> dict:
    data = make_synthetic(
        function_name=setting.function,
        n_train=setting.samples,
        n_test=args.test_samples,
        d=setting.dimension,
        noise=args.noise,
        seed=int(seed),
        standardize_target=True,
        nuisance_correlation=args.nuisance_correlation,
        n_correlated_proxies=args.n_correlated_proxies,
    )
    primary = np.asarray(sorted(int(v) for v in primary_support), dtype=int)
    t0 = time.time()
    row = {
        "probe_key": residual_probe_key(
            setting.function,
            setting.samples,
            setting.dimension,
            primary,
            int(seed),
            args.noise,
            args.nuisance_correlation,
            args.n_correlated_proxies,
        ),
        "function": setting.function,
        "samples": setting.samples,
        "dimension": setting.dimension,
        "noise": args.noise,
        "nuisance_correlation": args.nuisance_correlation,
        "n_correlated_proxies": int(args.n_correlated_proxies),
        "seed": int(seed),
        "primary_support": json_dumps(primary.tolist()),
        "probe_steps": int(args.residual_probe_steps),
        "status": "ok",
        "error": "",
        "traceback": "",
        "runtime_sec": np.nan,
    }
    try:
        additive_model = train_kan(
            data["X_train"][:, primary],
            data["y_train"],
            data["X_test"][:, primary],
            data["y_test"],
            train_args(args, args.additive_steps),
            seed=int(seed),
            device=device,
        )
        additive_train = batch_predict(additive_model, data["X_train"][:, primary], device=device, batch_size=args.pred_batch_size)
        additive_test = batch_predict(additive_model, data["X_test"][:, primary], device=device, batch_size=args.pred_batch_size)
        residual_train = data["y_train"] - additive_train
        residual_test = data["y_test"] - additive_test
        residual_mean = float(residual_train.mean())
        residual_std = float(residual_train.std())
        if residual_std < 1e-8:
            residual_std = 1.0
        residual_train_z = (residual_train - residual_mean) / residual_std
        residual_test_z = (residual_test - residual_mean) / residual_std

        residual_model = train_kan(
            data["X_train"],
            residual_train_z,
            data["X_test"],
            residual_test_z,
            train_args(args, args.residual_probe_steps),
            seed=int(seed) + 100_000,
            device=device,
        )
        residual_pred = batch_predict(residual_model, data["X_test"], device=device, batch_size=args.pred_batch_size)
        grad_scores = normalize_score(
            gradient_importance(residual_model, data["X_test"], device=device, points=args.probe_variable_points)
        )
        feature_scores = safe_feature_score(residual_model, setting.dimension)
        edge_scores, pair_scores, endpoint_mass = safe_edge_path_scores(residual_model, setting.dimension)
        top_pairs = sorted(pair_scores.items(), key=lambda kv: -float(kv[1]))[: args.keep_top_pairs]
        row.update(
            {
                "additive_test_mse": mse_np(additive_test, data["y_test"]),
                "residual_test_mse_z": mse_np(residual_pred, residual_test_z),
                "residual_train_std": residual_std,
                "test_mse": mse_np(residual_pred, residual_test_z),
                "grad_scores": json_dumps(grad_scores.tolist()),
                "feature_scores": json_dumps(feature_scores.tolist()),
                "edge_var_scores": json_dumps(edge_scores.tolist()),
                "edge_endpoint_mass": json_dumps(endpoint_mass.tolist()),
                "edge_pair_scores_top": json_dumps([(int(i), int(j), float(v)) for (i, j), v in top_pairs]),
                "top_grad_variables": json_dumps(top_vars(grad_scores, min(12, setting.dimension))),
                "top_feature_variables": json_dumps(top_vars(feature_scores, min(12, setting.dimension))),
                "top_edge_variables": json_dumps(top_vars(edge_scores, min(12, setting.dimension))),
            }
        )
    except Exception as exc:
        row.update({"status": "failed", "error": repr(exc), "traceback": traceback.format_exc()})
    row["runtime_sec"] = float(time.time() - t0)
    return row


def complete_support(primary: Sequence[int], residual_rank: Sequence[int], top_m: int, d: int) -> list[int]:
    selected: list[int] = []
    seen: set[int] = set()
    for v in primary:
        v = int(v)
        if v not in seen and len(selected) < top_m:
            selected.append(v)
            seen.add(v)
    for v in residual_rank:
        v = int(v)
        if v not in seen and len(selected) < top_m:
            selected.append(v)
            seen.add(v)
    for v in range(d):
        if len(selected) >= top_m:
            break
        if v not in seen:
            selected.append(v)
            seen.add(v)
    return selected[:top_m]


def score_residual_supports(
    probes: pd.DataFrame,
    setting: Setting,
    primary_support: Sequence[int],
    primary_meta: dict,
    args: argparse.Namespace,
) -> pd.DataFrame:
    data = make_synthetic(
        function_name=setting.function,
        n_train=setting.samples,
        n_test=args.test_samples,
        d=setting.dimension,
        noise=args.noise,
        seed=0,
        standardize_target=True,
        nuisance_correlation=args.nuisance_correlation,
        n_correlated_proxies=args.n_correlated_proxies,
    )
    true_vars = tuple(int(v) for v in data["ground_truth"].active_variables)
    true_interactions = canonical_pairs(data["ground_truth"].interactions)
    agg = aggregate_probe_scores(probes, setting.dimension)
    rows: list[dict] = []
    for method in args.residual_methods:
        if method not in METHODS:
            raise ValueError(f"Unknown residual method: {method}")
        for top_m in args.top_ms:
            _, meta = select_support(method, agg, setting.dimension, setting.dimension)
            residual_rank = [int(v) for v in meta.get("top_selection_variables", [])]
            if len(residual_rank) < setting.dimension:
                score = np.asarray(meta.get("selection_score", []), dtype=float)
                if len(score) == setting.dimension:
                    residual_rank = top_vars(score, setting.dimension)
            support = complete_support(primary_support, residual_rank, int(top_m), setting.dimension)
            stats = support_stats(np.asarray(support, dtype=int), true_vars, true_interactions)
            rows.append(
                {
                    "function": setting.function,
                    "samples": setting.samples,
                    "dimension": setting.dimension,
                    "noise": args.noise,
                    "nuisance_correlation": args.nuisance_correlation,
                    "n_correlated_proxies": int(args.n_correlated_proxies),
                    "primary_method": args.primary_method,
                    "primary_m": int(args.primary_m),
                    "primary_support": json_dumps([int(v) for v in primary_support]),
                    **primary_meta,
                    "method": f"residual_{method}",
                    "top_m": int(top_m),
                    "num_probe_rows": int(len(probes)),
                    "selected_screen_features": json_dumps([int(v) for v in support]),
                    "top_residual_variables": json_dumps(residual_rank[:12]),
                    **stats,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = device_from_arg(args.device)
    print(f"Using device={device}")
    print(f"Writing results under {out_dir}")

    probe_path = out_dir / "residual_probe_cache.csv"
    existing = pd.read_csv(probe_path) if probe_path.exists() else pd.DataFrame()
    rows = existing.to_dict("records") if not existing.empty else []
    existing_keys = {str(row.get("probe_key")) for row in rows}

    detail_tables: list[pd.DataFrame] = []
    for function_name in args.functions:
        for samples in args.samples:
            for dimension in args.dimensions:
                setting = Setting(str(function_name), int(samples), int(dimension), max(args.top_ms))
                primary_support, primary_meta = select_primary_support(args, setting)
                print(f"[PRIMARY] fn={function_name} n={samples} d={dimension} support={primary_support}", flush=True)
                for seed in args.probe_seeds:
                    key = residual_probe_key(
                        setting.function,
                        setting.samples,
                        setting.dimension,
                        primary_support,
                        int(seed),
                        args.noise,
                        args.nuisance_correlation,
                        args.n_correlated_proxies,
                    )
                    if key in existing_keys and not args.force:
                        continue
                    print(f"[RESIDUAL PROBE] {key}", flush=True)
                    row = train_residual_probe(
                        setting=setting,
                        primary_support=primary_support,
                        seed=int(seed),
                        args=args,
                        device=device,
                    )
                    rows.append(row)
                    existing_keys.add(key)
                    pd.DataFrame(rows).to_csv(probe_path, index=False)

                probe_df = pd.DataFrame(rows)
                mask = (
                    probe_df["function"].astype(str).eq(setting.function)
                    & pd.to_numeric(probe_df["samples"], errors="coerce").eq(setting.samples)
                    & pd.to_numeric(probe_df["dimension"], errors="coerce").eq(setting.dimension)
                    & probe_df["status"].astype(str).eq("ok")
                    & probe_df["primary_support"].astype(str).eq(json_dumps(primary_support))
                    & pd.to_numeric(probe_df["seed"], errors="coerce").isin([int(s) for s in args.probe_seeds])
                )
                support_detail = score_residual_supports(
                    probe_df[mask].copy(),
                    setting,
                    primary_support,
                    primary_meta,
                    args,
                )
                detail_tables.append(support_detail)

    detail = pd.concat(detail_tables, ignore_index=True, sort=False) if detail_tables else pd.DataFrame()
    detail_path = out_dir / "residual_support_detail.csv"
    summary_path = out_dir / "residual_support_summary.csv"
    detail.to_csv(detail_path, index=False)
    if detail.empty:
        summary = pd.DataFrame()
    else:
        numeric = [
            "screen_contains_all_true_vars",
            "screen_true_var_recall",
            "screen_contains_all_interaction_endpoints",
            "screen_interaction_endpoint_recall",
            "screen_contains_true_interactions",
        ]
        for col in numeric:
            detail[col] = pd.to_numeric(detail[col], errors="coerce")
        group_cols = [
            "function",
            "samples",
            "dimension",
            "noise",
            "nuisance_correlation",
            "n_correlated_proxies",
            "method",
            "top_m",
        ]
        summary = detail.groupby(group_cols, dropna=False)[numeric].mean().reset_index()
        counts = detail.groupby(group_cols, dropna=False).size().reset_index(name="num_support_evals")
        summary = summary.merge(counts, on=group_cols, how="left")
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False) if not summary.empty else "No rows.")
    print(f"Wrote {detail_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
