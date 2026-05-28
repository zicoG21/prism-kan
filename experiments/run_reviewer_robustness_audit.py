from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, Iterable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

from src.data import make_synthetic
from experiments.paper_plot_style import OKABE_ITO, clean_axis, configure_paper_plots, save_figure


def canonical_pairs(pairs: Sequence[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    return tuple(tuple(sorted((int(i), int(j)))) for i, j in pairs)


def endpoints(pairs: Iterable[tuple[int, int]]) -> set[int]:
    out: set[int] = set()
    for i, j in pairs:
        out.add(int(i))
        out.add(int(j))
    return out


def top_m(scores: np.ndarray, m: int) -> list[int]:
    scores = np.nan_to_num(np.asarray(scores, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    return [int(i) for i in np.argsort(-scores)[: min(int(m), len(scores))]]


def rf_scores(X: np.ndarray, y: np.ndarray, seed: int, trees: int) -> np.ndarray:
    model = RandomForestRegressor(
        n_estimators=int(trees),
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=int(seed),
    )
    model.fit(X, y.reshape(-1))
    return np.asarray(model.feature_importances_, dtype=float)


def mi_scores(X: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    return np.asarray(mutual_info_regression(X, y.reshape(-1), random_state=int(seed)), dtype=float)


def lasso_scores(X: np.ndarray, y: np.ndarray, seed: int, cv: int) -> np.ndarray:
    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)
    model = LassoCV(cv=int(cv), random_state=int(seed), max_iter=5000, n_jobs=-1)
    model.fit(Xz, y.reshape(-1))
    return np.abs(np.asarray(model.coef_, dtype=float))


def lassonet_scores(
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
    *,
    hidden_dim: int,
    n_iters_init: int,
    n_iters_path: int,
    path_multiplier: float,
    device: str | None,
) -> np.ndarray:
    from lassonet import LassoNetRegressor

    scaler = StandardScaler()
    Xz = scaler.fit_transform(X).astype(np.float32)
    y1 = y.reshape(-1).astype(np.float32)
    model = LassoNetRegressor(
        hidden_dims=(int(hidden_dim),),
        n_iters=(int(n_iters_init), int(n_iters_path)),
        patience=(max(10, int(n_iters_init) // 4), max(5, int(n_iters_path) // 3)),
        path_multiplier=float(path_multiplier),
        val_size=0.2,
        verbose=0,
        random_state=int(seed),
        torch_seed=int(seed),
        device=device,
    )
    path = model.path(Xz, y1, disable_lambda_warning=True)
    scores = np.zeros(X.shape[1], dtype=float)
    for item in path:
        selected = item.selected.detach().cpu().numpy().astype(bool)
        scores[selected] = np.maximum(scores[selected], float(item.lambda_))
    if np.all(scores == 0):
        dense = path[0].selected.detach().cpu().numpy().astype(bool)
        scores[dense] = 1.0
    return scores


def group_map_for_exact_and_proxy(active: Sequence[int], proxy_groups: Dict[int, int], d: int) -> dict[int, int]:
    groups = {int(j): int(j) for j in range(d)}
    for j in active:
        groups[int(j)] = int(j)
    for proxy, active_idx in proxy_groups.items():
        groups[int(proxy)] = int(active_idx)
    return groups


def support_metrics(selected: Sequence[int], active: Sequence[int], true_pairs: Sequence[tuple[int, int]], proxy_groups: Dict[int, int], d: int) -> dict:
    selected_set = {int(v) for v in selected}
    active_set = {int(v) for v in active}
    pair_set = set(canonical_pairs(true_pairs))
    endpoint_set = endpoints(pair_set)
    group_map = group_map_for_exact_and_proxy(active, proxy_groups, d)
    selected_groups = {group_map[int(v)] for v in selected_set}
    active_groups = {int(v) for v in active}
    endpoint_groups = {int(v) for v in endpoint_set}
    exact_false = selected_set - active_set
    group_false = {int(v) for v in selected_set if group_map[int(v)] not in active_groups}
    return {
        "selected_features": json.dumps(sorted(selected_set)),
        "selected_groups": json.dumps(sorted(selected_groups)),
        "exact_active_recall": len(selected_set & active_set) / len(active_set) if active_set else np.nan,
        "exact_active_precision": len(selected_set & active_set) / len(selected_set) if selected_set else np.nan,
        "exact_endpoint_recall": len(selected_set & endpoint_set) / len(endpoint_set) if endpoint_set else np.nan,
        "exact_pair_retained": int(all(i in selected_set and j in selected_set for i, j in pair_set)) if pair_set else np.nan,
        "exact_false_discovery_rate": len(exact_false) / len(selected_set) if selected_set else np.nan,
        "group_active_recall": len(selected_groups & active_groups) / len(active_groups) if active_groups else np.nan,
        "group_endpoint_recall": len(selected_groups & endpoint_groups) / len(endpoint_groups) if endpoint_groups else np.nan,
        "group_pair_retained": int(all(i in selected_groups and j in selected_groups for i, j in pair_set)) if pair_set else np.nan,
        "group_false_discovery_rate": len(group_false) / len(selected_set) if selected_set else np.nan,
    }


def run_one(args: argparse.Namespace, method: str, rho: float, noise: float, seed: int) -> dict:
    data = make_synthetic(
        function_name=args.function,
        n_train=args.samples,
        n_test=args.test_samples,
        d=args.dimension,
        noise=float(noise),
        seed=int(seed),
        standardize_target=True,
        nuisance_correlation=float(rho),
        n_correlated_proxies=int(args.n_correlated_proxies),
    )
    X = data["X_train"]
    y = data["y_train"]
    gt = data["ground_truth"]
    t0 = time.time()
    row = {
        "function": args.function,
        "method": method,
        "samples": int(args.samples),
        "dimension": int(args.dimension),
        "top_m": int(args.top_m),
        "rho": float(rho),
        "noise": float(noise),
        "seed": int(seed),
        "n_correlated_proxies": int(args.n_correlated_proxies),
        "status": "ok",
        "error": "",
        "traceback": "",
    }
    try:
        if method == "rf":
            scores = rf_scores(X, y, seed, args.rf_trees)
        elif method == "mi":
            scores = mi_scores(X, y, seed)
        elif method == "lasso":
            scores = lasso_scores(X, y, seed, args.lasso_cv)
        elif method == "lassonet":
            scores = lassonet_scores(
                X,
                y,
                seed,
                hidden_dim=args.lassonet_hidden,
                n_iters_init=args.lassonet_init_iters,
                n_iters_path=args.lassonet_path_iters,
                path_multiplier=args.lassonet_path_multiplier,
                device=None if args.device == "auto" else args.device,
            )
        else:
            raise ValueError(f"Unknown method={method}")
        selected = top_m(scores, args.top_m)
        row.update(
            {
                "screen_runtime_sec": float(time.time() - t0),
                "score_top12": json.dumps(top_m(scores, min(12, args.dimension))),
            }
        )
        row.update(
            support_metrics(
                selected,
                gt.active_variables,
                canonical_pairs(gt.interactions),
                data.get("proxy_groups", {}),
                args.dimension,
            )
        )
    except Exception as exc:
        row.update(
            {
                "status": "failed",
                "error": repr(exc),
                "traceback": traceback.format_exc(),
                "screen_runtime_sec": float(time.time() - t0),
            }
        )
    return row


def summarize(detail: pd.DataFrame) -> pd.DataFrame:
    ok = detail[detail["status"].astype(str).eq("ok")].copy()
    if ok.empty:
        return pd.DataFrame()
    group_cols = ["function", "method", "dimension", "samples", "top_m", "rho", "noise", "n_correlated_proxies"]
    numeric_cols = [
        "screen_runtime_sec",
        "exact_active_recall",
        "exact_active_precision",
        "exact_endpoint_recall",
        "exact_pair_retained",
        "exact_false_discovery_rate",
        "group_active_recall",
        "group_endpoint_recall",
        "group_pair_retained",
        "group_false_discovery_rate",
    ]
    for col in numeric_cols:
        ok[col] = pd.to_numeric(ok[col], errors="coerce")
    out = ok.groupby(group_cols, dropna=False)[numeric_cols].agg(["mean", "std"]).reset_index()
    out.columns = ["_".join(str(v) for v in col if v != "").rstrip("_") for col in out.columns]
    counts = ok.groupby(group_cols, dropna=False).size().reset_index(name="num_runs")
    return out.merge(counts, on=group_cols, how="left")


def plot_summary(summary: pd.DataFrame, out_base: Path) -> None:
    if summary.empty:
        return
    configure_paper_plots(usetex=False)
    methods = [m for m in ["rf", "mi", "lasso", "lassonet"] if m in set(summary["method"])]
    colors = {
        "rf": OKABE_ITO["gray"],
        "mi": OKABE_ITO["sky"],
        "lasso": OKABE_ITO["orange"],
        "lassonet": OKABE_ITO["blue"],
    }
    labels = {"rf": "RF", "mi": "MI", "lasso": "Lasso", "lassonet": "LassoNet"}
    noises = sorted(summary["noise"].unique())
    rhos = sorted(summary["rho"].unique())
    fig, axes = plt.subplots(1, len(noises), figsize=(3.25 * len(noises), 2.35), sharey=True)
    if len(noises) == 1:
        axes = [axes]
    for ax, noise in zip(axes, noises):
        subset = summary[summary["noise"] == noise]
        for method in methods:
            vals = []
            errs = []
            for rho in rhos:
                hit = subset[(subset["method"] == method) & (subset["rho"] == rho)]
                vals.append(float(hit["exact_pair_retained_mean"].iloc[0]) if not hit.empty else np.nan)
                errs.append(float(hit["exact_pair_retained_std"].fillna(0.0).iloc[0]) if not hit.empty else 0.0)
            ax.errorbar(
                rhos,
                vals,
                yerr=errs,
                marker="o",
                markersize=3.5,
                linewidth=1.2,
                capsize=2,
                color=colors.get(method, "#111827"),
                label=labels.get(method, method),
            )
        ax.set_title(f"noise={noise:g}", loc="left", pad=3)
        ax.set_xlabel("Proxy correlation $\\rho$")
        ax.set_ylim(-0.05, 1.05)
        ax.set_yticks([0, 0.5, 1.0])
        clean_axis(ax, grid=True)
    axes[0].set_ylabel("Exact pair retained")
    axes[-1].legend(frameon=False, loc="lower left", ncol=2)
    fig.suptitle("External support baselines under correlated nuisance proxies", y=1.02, fontsize=9.5)
    save_figure(fig, out_base)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reviewer-facing robustness audit for correlated/noisy support selection baselines.")
    parser.add_argument("--function", default="core_interaction_c025")
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--test_samples", type=int, default=2048)
    parser.add_argument("--dimension", type=int, default=100)
    parser.add_argument("--top_m", type=int, default=8)
    parser.add_argument("--rhos", nargs="+", type=float, default=[0.0, 0.5, 0.9])
    parser.add_argument("--noises", nargs="+", type=float, default=[0.0, 0.1])
    parser.add_argument("--n_correlated_proxies", type=int, default=8)
    parser.add_argument("--methods", nargs="+", default=["rf", "mi", "lasso", "lassonet"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--rf_trees", type=int, default=300)
    parser.add_argument("--lasso_cv", type=int, default=3)
    parser.add_argument("--lassonet_hidden", type=int, default=32)
    parser.add_argument("--lassonet_init_iters", type=int, default=200)
    parser.add_argument("--lassonet_path_iters", type=int, default=30)
    parser.add_argument("--lassonet_path_multiplier", type=float, default=1.15)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--out_dir", default="results/reviewer_robustness_audit")
    parser.add_argument("--tag", default="quick")
    args = parser.parse_args()

    if args.device == "auto" and torch.cuda.is_available():
        args.device = "cuda"

    out_dir = Path(args.out_dir) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    detail_path = out_dir / "support_baseline_detail.csv"
    for rho in args.rhos:
        for noise in args.noises:
            for seed in args.seeds:
                for method in args.methods:
                    print(f"[RUN] method={method} rho={rho} noise={noise} seed={seed}", flush=True)
                    row = run_one(args, method, rho, noise, seed)
                    rows.append(row)
                    pd.DataFrame(rows).to_csv(detail_path, index=False)
    detail = pd.DataFrame(rows)
    summary = summarize(detail)
    summary_path = out_dir / "support_baseline_summary.csv"
    summary.to_csv(summary_path, index=False)
    plot_summary(summary, out_dir / "support_baseline_pair_retention")
    print(f"Wrote {detail_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
