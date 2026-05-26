from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from experiments.paper_plot_style import OKABE_ITO, clean_axis, configure_paper_plots, save_figure
except Exception:  # pragma: no cover
    OKABE_ITO = {
        "blue": "#0072B2",
        "green": "#009E73",
        "purple": "#CC79A7",
        "orange": "#E69F00",
        "gray": "#6B7280",
        "vermillion": "#D55E00",
    }
    clean_axis = None
    configure_paper_plots = None
    save_figure = None


METHODS = [
    ("feature_edge_hybrid", "KAN-FE", OKABE_ITO["green"]),
    ("feature_stability_var", "KAN-F", OKABE_ITO["blue"]),
    ("oracle_support", "Oracle", OKABE_ITO["purple"]),
    ("rf", "RF", OKABE_ITO["orange"]),
    ("random", "Rand", OKABE_ITO["gray"]),
    ("exclude_interaction", "Excl", OKABE_ITO["vermillion"]),
]


def metric_value(row: pd.Series, name: str) -> float:
    for col in [f"{name}_mean", name]:
        if col in row:
            return float(row[col])
    return np.nan


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot a formula-fidelity case study for one regime.")
    parser.add_argument("--combined_summary", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--function", default="core_interaction_c025")
    parser.add_argument("--dimension", type=int, default=100)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--top_m", type=int, default=4)
    parser.add_argument("--min_runs", type=int, default=8)
    args = parser.parse_args()

    if configure_paper_plots is not None:
        configure_paper_plots(usetex=False)

    df = pd.read_csv(args.combined_summary)
    for col in ["dimension", "samples", "top_m", "num_runs"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    sub = df[
        df["function"].astype(str).eq(args.function)
        & df["dimension"].eq(args.dimension)
        & df["samples"].eq(args.samples)
        & df["top_m"].eq(args.top_m)
        & (pd.to_numeric(df["num_runs"], errors="coerce") >= args.min_runs)
    ].copy()
    if sub.empty:
        raise SystemExit("No rows match the requested case study setting.")

    rows = []
    for method, label, color in METHODS:
        hit = sub[sub["method"].astype(str).eq(method)]
        if hit.empty:
            continue
        row = hit.iloc[0]
        rows.append(
            {
                "method": method,
                "label": label,
                "color": color,
                "endpoint": metric_value(row, "screen_contains_true_interactions"),
                "interaction": metric_value(row, "interaction_f1"),
                "mse": metric_value(row, "test_mse"),
            }
        )
    plot_df = pd.DataFrame(rows)
    if plot_df.empty:
        raise SystemExit("Requested methods were not present.")

    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.25), gridspec_kw={"width_ratios": [1.0, 1.0, 1.15]})
    x = np.arange(len(plot_df))

    axes[0].bar(x, plot_df["endpoint"], color=plot_df["color"], width=0.68)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Rate")
    axes[0].set_title("Endpoint Support")

    axes[1].bar(x, plot_df["interaction"], color=plot_df["color"], width=0.68)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Interaction F1")

    mse_vals = np.maximum(plot_df["mse"].astype(float).to_numpy(), 1e-6)
    axes[2].bar(x, mse_vals, color=plot_df["color"], width=0.68)
    axes[2].set_yscale("log")
    axes[2].set_title("Test MSE")

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(plot_df["label"], rotation=25, ha="right", fontsize=7.0)
        ax.tick_params(axis="x", pad=2)
        if clean_axis is not None:
            clean_axis(ax, grid=True)

    c_label = "0.25"
    if str(args.function).endswith("c05"):
        c_label = "0.5"
    elif str(args.function).endswith("c1"):
        c_label = "1.0"
    fig.suptitle(
        rf"$c={c_label},\ d={args.dimension},\ n={args.samples},\ m={args.top_m}$",
        y=1.015,
    )
    fig.tight_layout(w_pad=0.9)
    out = Path(args.out)
    if save_figure is not None:
        save_figure(fig, out)
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
        fig.savefig(out.with_suffix(".png"), dpi=450, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
