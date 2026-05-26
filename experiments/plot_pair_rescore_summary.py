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


METHOD_ORDER = ["fd", "anova_abs", "anova_var", "fd_anova_hybrid"]
METHOD_LABELS = {
    "fd": "FD",
    "anova_abs": "ANOVA-abs",
    "anova_var": "ANOVA-var",
    "fd_anova_hybrid": "Hybrid",
}
METHOD_COLORS = {
    "fd": OKABE_ITO["gray"],
    "anova_abs": OKABE_ITO["blue"],
    "anova_var": OKABE_ITO["green"],
    "fd_anova_hybrid": OKABE_ITO["purple"],
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot pair-rescoring results for stability-selected supports.")
    parser.add_argument("--summary", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--source_method", default="feature_edge_hybrid")
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--dimension", type=int, default=100)
    parser.add_argument("--top_m", type=int, default=4)
    parser.add_argument("--min_runs", type=int, default=1)
    args = parser.parse_args()

    configure_paper_plots(usetex=False)
    df = pd.read_csv(args.summary)
    for col in ["samples", "dimension", "top_m", "num_runs", "interaction_f1_mean", "source_interaction_f1_mean"]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    sub = df[
        df["source_method"].astype(str).eq(args.source_method)
        & df["samples"].eq(args.samples)
        & df["dimension"].eq(args.dimension)
        & df["top_m"].eq(args.top_m)
        & (df["num_runs"] >= args.min_runs)
        & df["pair_score_method"].astype(str).isin(METHOD_ORDER)
    ].copy()
    if sub.empty:
        raise SystemExit("No matching rescore rows.")
    sub["pair_score_method"] = pd.Categorical(sub["pair_score_method"], METHOD_ORDER, ordered=True)
    sub = sub.sort_values("pair_score_method")

    x = np.arange(len(sub))
    fig, ax = plt.subplots(figsize=(3.35, 2.25))
    ax.bar(
        x,
        sub["interaction_f1_mean"],
        color=[METHOD_COLORS[str(m)] for m in sub["pair_score_method"]],
        width=0.65,
        edgecolor="white",
        linewidth=0.45,
    )
    source_f1 = float(sub["source_interaction_f1_mean"].dropna().iloc[0])
    ax.axhline(source_f1, color=OKABE_ITO["black"], linewidth=0.75, linestyle=(0, (2.0, 2.0)))
    ax.text(len(x) - 0.55, source_f1 + 0.03, "original FD", ha="right", va="bottom", fontsize=7.0)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Interaction F1")
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[str(m)] for m in sub["pair_score_method"]], rotation=18, ha="right")
    ax.text(0.02, 0.98, rf"$n={args.samples},\ m={args.top_m}$", transform=ax.transAxes, ha="left", va="top", fontsize=7.8)
    clean_axis(ax, grid=True)
    fig.tight_layout()
    save_figure(fig, Path(args.out))
    plt.close(fig)


if __name__ == "__main__":
    main()
