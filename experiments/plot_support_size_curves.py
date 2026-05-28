from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from experiments.paper_plot_style import OKABE_ITO, clean_axis, configure_paper_plots, save_figure
except Exception:  # pragma: no cover
    from paper_plot_style import OKABE_ITO, clean_axis, configure_paper_plots, save_figure


METHODS = ["feature_edge_hybrid", "feature_stability_var", "rf", "random"]
METHOD_LABELS = {
    "feature_edge_hybrid": "KAN-FE",
    "feature_stability_var": "KAN-F",
    "rf": "RF",
    "random": "Random",
}
METHOD_COLORS = {
    "feature_edge_hybrid": OKABE_ITO["green"],
    "feature_stability_var": OKABE_ITO["blue"],
    "rf": OKABE_ITO["orange"],
    "random": OKABE_ITO["gray"],
}


def numericize(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def support_curve_source(combined: pd.DataFrame) -> pd.DataFrame:
    sub = combined[
        combined["function"].astype(str).eq("core_interaction_c025")
        & combined["dimension"].eq(100)
        & combined["samples"].eq(1024)
        & combined["top_m"].isin([4, 5, 6])
        & combined["method"].astype(str).isin(METHODS)
    ].copy()
    keep = [
        "method",
        "top_m",
        "screen_interaction_endpoint_recall_mean",
        "screen_interaction_endpoint_recall_std",
        "interaction_f1_mean",
        "interaction_f1_std",
        "num_runs",
    ]
    return sub[keep].sort_values(["method", "top_m"])


def highdim_budget_source(highdim: pd.DataFrame) -> pd.DataFrame:
    sub = highdim[
        highdim["method"].astype(str).eq("feature_edge_hybrid")
        & highdim["dimension"].isin([500, 1000])
    ].copy()
    sub["label"] = sub.apply(
        lambda r: rf"$c={float(r['c']):.2g},d={int(r['dimension'])}$", axis=1
    )
    keep = [
        "function",
        "c",
        "samples",
        "dimension",
        "top_m",
        "endpoint_rank_worst_mean",
        "endpoint_recall_mean",
        "pair_retained_mean",
        "top1_pair_accuracy_mean",
        "label",
    ]
    return sub[keep].sort_values(["c", "dimension"])


def budget_retention_curve(rows: pd.DataFrame, budgets: list[int]) -> pd.DataFrame:
    records = []
    for _, row in rows.iterrows():
        worst_rank = float(row["endpoint_rank_worst_mean"])
        for budget in budgets:
            records.append(
                {
                    "function": row["function"],
                    "c": float(row["c"]),
                    "samples": int(row["samples"]),
                    "dimension": int(row["dimension"]),
                    "label": row["label"],
                    "budget": int(budget),
                    "endpoint_pair_retained_from_rank": float(budget >= worst_rank),
                    "endpoint_rank_worst_mean": worst_rank,
                }
            )
    return pd.DataFrame(records)


def plot_curves(support: pd.DataFrame, budget: pd.DataFrame, out: Path) -> None:
    configure_paper_plots(usetex=False)
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.7), gridspec_kw={"width_ratios": [1.02, 1.2]})

    ax = axes[0]
    for method in METHODS:
        g = support[support["method"].eq(method)].sort_values("top_m")
        if g.empty:
            continue
        x = g["top_m"].to_numpy(dtype=float)
        endpoint = g["screen_interaction_endpoint_recall_mean"].to_numpy(dtype=float)
        pair = g["interaction_f1_mean"].to_numpy(dtype=float)
        color = METHOD_COLORS[method]
        ax.plot(
            x,
            endpoint,
            marker="o",
            markersize=3.4,
            linewidth=1.25,
            color=color,
            label=METHOD_LABELS[method],
        )
        ax.plot(
            x,
            pair,
            marker="s",
            markersize=3.0,
            linewidth=1.05,
            linestyle=(0, (2.2, 2.0)),
            color=color,
            alpha=0.88,
        )
        if method in {"feature_edge_hybrid", "feature_stability_var", "rf", "random"}:
            y = endpoint[-1] if method in {"feature_edge_hybrid", "feature_stability_var"} else max(endpoint[-1], pair[-1])
            dy = {
                "feature_edge_hybrid": -0.055,
                "feature_stability_var": 0.018,
                "rf": 0.035,
                "random": -0.035,
            }[method]
            ax.text(6.08, y + dy, METHOD_LABELS[method], color=color, fontsize=7.0, va="center")
    ax.set_title(r"(a) Refit curve, $c=0.25,d=100,n=1024$")
    ax.set_xlabel("support budget $m$")
    ax.set_ylabel("success rate")
    ax.set_ylim(-0.04, 1.04)
    ax.set_xticks([4, 5, 6])
    ax.text(
        4.05,
        0.36,
        "solid: endpoints\n dashed: top-1 pair",
        fontsize=7.0,
        color="#374151",
        linespacing=1.05,
    )
    clean_axis(ax, grid=True)

    ax = axes[1]
    marker_cycle = ["s", "D", "P", "o", "^", "v"]
    plot_budget = budget[
        (budget["dimension"].eq(1000))
        | ((budget["dimension"].eq(500)) & (budget["c"].isin([0.25, 1.00])))
    ].copy()
    labels = list(dict.fromkeys(plot_budget["label"].astype(str).tolist()))
    for idx, label in enumerate(labels):
        g = plot_budget[plot_budget["label"].astype(str).eq(label)].sort_values("budget")
        if g.empty:
            continue
        dim = int(g["dimension"].iloc[0])
        color = OKABE_ITO["vermillion"] if dim == 1000 else OKABE_ITO["blue"]
        if float(g["c"].iloc[0]) == 0.50:
            color = OKABE_ITO["orange"] if dim == 1000 else OKABE_ITO["sky"]
        if float(g["c"].iloc[0]) == 1.00:
            color = OKABE_ITO["green"] if dim == 1000 else OKABE_ITO["purple"]
        ax.step(
            g["budget"],
            g["endpoint_pair_retained_from_rank"],
            where="post",
            linewidth=1.25,
            marker=marker_cycle[idx % len(marker_cycle)],
            markersize=3.0,
            color=color,
            label=label,
        )
    ax.set_title(r"(b) KAN-FE endpoint budget from ranks")
    ax.set_xlabel("support budget $m$")
    ax.set_ylabel("both endpoints retained")
    ax.set_xscale("log")
    ax.set_ylim(-0.04, 1.04)
    ax.set_xticks([4, 6, 10, 20, 50, 100, 250, 500, 1000])
    ax.set_xticklabels(["4", "6", "10", "20", "50", "100", "250", "500", "1000"], rotation=35)
    clean_axis(ax, grid=True)
    axes[1].text(
        0.03,
        0.08,
        r"omitted $c=0.5,d=500$ jumps at $m=334$",
        transform=axes[1].transAxes,
        fontsize=6.8,
        color="#374151",
    )
    axes[1].legend(
        loc="upper center",
        bbox_to_anchor=(0.52, -0.30),
        ncol=2,
        frameon=False,
        handlelength=1.6,
        columnspacing=1.0,
    )

    fig.tight_layout(w_pad=1.0)
    fig.subplots_adjust(bottom=0.33)
    save_figure(fig, out)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--combined_summary",
        default="results/innovation_loop/strict_screened_baseline_controls_20260526_104243/analysis_min8_live/native_screened_combined_summary.csv",
    )
    parser.add_argument(
        "--highdim_summary",
        default="results/innovation_loop/final_candidate_figures/highdim_support_failure_diagnostics_summary.csv",
    )
    parser.add_argument("--out_dir", default="results/innovation_loop/final_candidate_figures")
    args = parser.parse_args()

    combined = numericize(
        pd.read_csv(args.combined_summary),
        [
            "samples",
            "dimension",
            "top_m",
            "interaction_f1_mean",
            "interaction_f1_std",
            "screen_interaction_endpoint_recall_mean",
            "screen_interaction_endpoint_recall_std",
            "num_runs",
        ],
    )
    highdim = numericize(
        pd.read_csv(args.highdim_summary),
        [
            "c",
            "samples",
            "dimension",
            "top_m",
            "endpoint_rank_worst_mean",
            "endpoint_recall_mean",
            "pair_retained_mean",
            "top1_pair_accuracy_mean",
        ],
    )

    support = support_curve_source(combined)
    highdim_rows = highdim_budget_source(highdim)
    budgets = [4, 6, 10, 20, 50, 100, 250, 500, 1000]
    budget = budget_retention_curve(highdim_rows, budgets)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    support.to_csv(out_dir / "support_size_refit_curve_source.csv", index=False)
    highdim_rows.to_csv(out_dir / "support_size_highdim_rank_source.csv", index=False)
    budget.to_csv(out_dir / "support_size_highdim_budget_curve_source.csv", index=False)
    plot_curves(support, budget, out_dir / "support_size_curves")
    print(support.to_string(index=False))
    print(highdim_rows.to_string(index=False))


if __name__ == "__main__":
    main()
