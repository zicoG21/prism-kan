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


PAIR_LABELS = {
    "fd": "FD",
    "anova_abs": "ANOVA-abs",
    "anova_var": "ANOVA-var",
    "fd_anova_hybrid": "Hybrid",
}

PAIR_COLORS = {
    "fd": OKABE_ITO["gray"],
    "anova_abs": OKABE_ITO["blue"],
    "anova_var": OKABE_ITO["green"],
    "fd_anova_hybrid": OKABE_ITO["purple"],
}


def mean_col(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df:
        return pd.to_numeric(df[col], errors="coerce")
    raise KeyError(col)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot RF screening and pair-scoring diagnostics.")
    parser.add_argument("--rf_summary", required=True)
    parser.add_argument("--pair_summary", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    configure_paper_plots(usetex=False)
    rf = pd.read_csv(args.rf_summary)
    pair = pd.read_csv(args.pair_summary)

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.55), gridspec_kw={"width_ratios": [1.15, 1.0]})

    # RF screening panel: true endpoints rank far outside a small support.
    rf = rf.copy()
    for col in ["samples", "trees", "top_m", "endpoint_mean_rank_mean", "screen_interaction_endpoint_recall_mean"]:
        rf[col] = pd.to_numeric(rf[col], errors="coerce")
    rf_rank = (
        rf[rf["top_m"].eq(6)]
        .sort_values(["samples", "trees"])
        .reset_index(drop=True)
    )
    x = np.arange(len(rf_rank))
    labels = [rf"$n={int(r.samples)}$" + "\n" + rf"{int(r.trees)} trees" for _, r in rf_rank.iterrows()]
    axes[0].bar(
        x,
        rf_rank["endpoint_mean_rank_mean"],
        color=[OKABE_ITO["orange"] if int(t) == 500 else OKABE_ITO["vermillion"] for t in rf_rank["trees"]],
        width=0.65,
        edgecolor="white",
        linewidth=0.45,
    )
    axes[0].axhline(6, color=OKABE_ITO["black"], linewidth=0.85, linestyle=(0, (2.0, 2.0)))
    axes[0].text(len(x) - 0.5, 6.7, r"$m=6$ cutoff", ha="right", va="bottom", fontsize=7.2)
    axes[0].set_ylabel("Mean Endpoint Rank")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_title("RF Marginal Screening")
    clean_axis(axes[0], grid=True)

    # Pair scoring panel: same oracle support, different interaction scorer.
    order = ["fd", "anova_abs", "anova_var", "fd_anova_hybrid"]
    pair = pair[pair["pair_score_method"].isin(order)].copy()
    pair["pair_score_method"] = pd.Categorical(pair["pair_score_method"], order, ordered=True)
    pair = pair.sort_values("pair_score_method")
    x2 = np.arange(len(pair))
    vals = mean_col(pair, "interaction_f1_mean").to_numpy(dtype=float)
    err = mean_col(pair, "interaction_f1_std").fillna(0.0).to_numpy(dtype=float)
    axes[1].bar(
        x2,
        vals,
        yerr=err,
        capsize=2.0,
        color=[PAIR_COLORS[str(m)] for m in pair["pair_score_method"]],
        width=0.65,
        edgecolor="white",
        linewidth=0.45,
        error_kw={"elinewidth": 0.6, "capthick": 0.6, "ecolor": "#374151"},
    )
    axes[1].set_ylim(0, 1.08)
    axes[1].set_ylabel("Interaction F1")
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels([PAIR_LABELS[str(m)] for m in pair["pair_score_method"]], rotation=18, ha="right")
    axes[1].set_title("Pair Scoring After Oracle Support")
    clean_axis(axes[1], grid=True)

    fig.tight_layout(w_pad=1.4)
    save_figure(fig, Path(args.out))
    plt.close(fig)


if __name__ == "__main__":
    main()
