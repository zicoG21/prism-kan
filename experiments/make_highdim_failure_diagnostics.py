from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.paper_plot_style import OKABE_ITO, clean_axis, configure_paper_plots, save_figure


TRUE_ENDPOINTS = (2, 3)
ACTIVE = {0, 1, 2, 3}


def parse_list(value):
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return []
    try:
        return ast.literal_eval(str(value))
    except Exception:
        return []


def rank_desc(scores: list[float], index: int) -> int:
    arr = np.asarray(scores, dtype=float)
    order = np.argsort(-np.nan_to_num(arr, nan=-np.inf))
    where = np.where(order == int(index))[0]
    return int(where[0]) + 1 if len(where) else len(arr) + 1


def score_at(scores: list[float], index: int) -> float:
    if 0 <= int(index) < len(scores):
        return float(scores[int(index)])
    return np.nan


def max_nuisance_score(scores: list[float]) -> float:
    vals = [float(v) for i, v in enumerate(scores) if i not in ACTIVE]
    return float(np.nanmax(vals)) if vals else np.nan


def build_rows(detail: pd.DataFrame, method: str) -> pd.DataFrame:
    rows = []
    sub = detail[detail["method"].astype(str).eq(method)].copy()
    for _, row in sub.iterrows():
        scores = parse_list(row.get("selection_score"))
        selected = set(int(v) for v in parse_list(row.get("selected_screen_features")))
        if not scores:
            continue
        endpoint_ranks = [rank_desc(scores, idx) for idx in TRUE_ENDPOINTS]
        endpoint_scores = [score_at(scores, idx) for idx in TRUE_ENDPOINTS]
        nuisance_max = max_nuisance_score(scores)
        rows.append(
            {
                "method": method,
                "function": row["function"],
                "c": float(row["interaction_strength"]),
                "samples": int(row["samples"]),
                "dimension": int(row["dimension"]),
                "top_m": int(row["top_m"]),
                "seed": int(row["seed"]),
                "endpoint_recall": float(row["screen_interaction_endpoint_recall"]),
                "pair_retained": int(row["screen_contains_true_interactions"]),
                "top1_pair_accuracy": float(row["interaction_f1"]),
                "endpoint_rank_mean": float(np.mean(endpoint_ranks)),
                "endpoint_rank_worst": float(np.max(endpoint_ranks)),
                "endpoint_score_mean": float(np.mean(endpoint_scores)),
                "max_nuisance_score": nuisance_max,
                "support_score_margin": float(np.mean(endpoint_scores) - nuisance_max),
                "selected_support": str(sorted(selected)),
            }
        )
    return pd.DataFrame(rows)


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["method", "function", "c", "samples", "dimension", "top_m"]
    numeric = [
        "endpoint_recall",
        "pair_retained",
        "top1_pair_accuracy",
        "endpoint_rank_mean",
        "endpoint_rank_worst",
        "endpoint_score_mean",
        "max_nuisance_score",
        "support_score_margin",
    ]
    out = rows.groupby(group_cols, dropna=False)[numeric].agg(["mean", "std"]).reset_index()
    out.columns = ["_".join(str(v) for v in col if v != "").rstrip("_") for col in out.columns]
    counts = rows.groupby(group_cols, dropna=False).size().reset_index(name="num_runs")
    return out.merge(counts, on=group_cols, how="left")


def plot(summary: pd.DataFrame, out_dir: Path) -> None:
    configure_paper_plots(usetex=False)
    summary = summary.sort_values(["c", "dimension", "samples"])
    labels = [f"c={r.c:g}\nd={int(r.dimension)}" for r in summary.itertuples()]
    x = np.arange(len(summary))
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.05), constrained_layout=True)
    color = OKABE_ITO["blue"]
    fail_color = OKABE_ITO["vermillion"]

    axes[0].bar(x, summary["endpoint_recall_mean"], color=color, width=0.72)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Endpoint recall")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=35, ha="right")
    clean_axis(axes[0])

    axes[1].bar(x, summary["endpoint_rank_worst_mean"], color=fail_color, width=0.72)
    axes[1].set_ylabel("Worst endpoint rank")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=35, ha="right")
    clean_axis(axes[1])

    axes[2].bar(x, summary["support_score_margin_mean"], color=color, width=0.72)
    axes[2].axhline(0, color="#4b5563", linewidth=0.8)
    axes[2].set_ylabel("Endpoint - max nuisance")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=35, ha="right")
    clean_axis(axes[2])
    save_figure(fig, out_dir / "highdim_support_failure_diagnostics")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--detail",
        default="results/innovation_loop/strict_validation_20260526_011917/innovation_detail.csv",
    )
    parser.add_argument("--method", default="feature_edge_hybrid")
    parser.add_argument("--out_dir", default="results/innovation_loop/final_candidate_figures")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    detail = pd.read_csv(args.detail)
    rows = build_rows(detail, args.method)
    rows.to_csv(out_dir / "highdim_support_failure_diagnostics_detail.csv", index=False)
    summary = summarize(rows)
    focus = summary[
        (summary["dimension"].isin([500, 1000]))
        & (
            ((summary["c"].eq(0.25)) & (summary["samples"].eq(2048)))
            | ((summary["c"].isin([0.5, 1.0])) & (summary["samples"].eq(1024)))
        )
    ].copy()
    focus.to_csv(out_dir / "highdim_support_failure_diagnostics_summary.csv", index=False)
    plot(focus, out_dir)
    print(focus.to_string(index=False))


if __name__ == "__main__":
    main()
