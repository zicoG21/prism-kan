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


LEFT_METHODS = ["feature_stability_var", "feature_edge_hybrid", "rf", "oracle_support"]
LEFT_LABELS = {
    "feature_stability_var": "KAN-F",
    "feature_edge_hybrid": "KAN-FE",
    "rf": "RF",
    "oracle_support": "Oracle",
}
LEFT_COLORS = {
    "feature_stability_var": OKABE_ITO["blue"],
    "feature_edge_hybrid": OKABE_ITO["green"],
    "rf": OKABE_ITO["orange"],
    "oracle_support": OKABE_ITO["purple"],
}

PAIR_METHODS = ["fd", "anova_abs", "anova_var", "fd_anova_hybrid"]
PAIR_LABELS = {
    "fd": "FD",
    "anova_abs": "ANOVA-abs",
    "anova_var": "ANOVA-var",
    "fd_anova_hybrid": "Hybrid",
}
SUPPORT_METHODS = ["feature_stability_var", "feature_edge_hybrid"]
SUPPORT_LABELS = {
    "feature_stability_var": "KAN-F support",
    "feature_edge_hybrid": "KAN-FE support",
}
SUPPORT_COLORS = {
    "feature_stability_var": OKABE_ITO["blue"],
    "feature_edge_hybrid": OKABE_ITO["green"],
}

SINGLE_METHOD = "single_feature_edge_hybrid"


def numericize(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def stability_value_at(df: pd.DataFrame, function: str, method: str, n: int, top_m: int, col: str) -> float:
    hit = df[
        df["method"].astype(str).eq(method)
        & df["function"].astype(str).eq(function)
        & df["dimension"].eq(100)
        & df["samples"].eq(n)
        & df["top_m"].eq(top_m)
    ]
    if hit.empty:
        return np.nan
    return float(hit[col].iloc[0])


def one_shot_value_at(df: pd.DataFrame | None, function: str, n: int, top_m: int, col: str) -> float:
    if df is None:
        return np.nan
    hit = df[
        df["method"].astype(str).eq(SINGLE_METHOD)
        & df["function"].astype(str).eq(function)
        & df["dimension"].eq(100)
        & df["samples"].eq(n)
        & df["top_m"].eq(top_m)
    ]
    if hit.empty:
        return np.nan
    return float(hit[col].iloc[0])


def value_at(df: pd.DataFrame, method: str, n: int, top_m: int, col: str) -> float:
    hit = df[
        df["method"].astype(str).eq(method)
        & df["function"].astype(str).eq("core_interaction_c025")
        & df["dimension"].eq(100)
        & df["samples"].eq(n)
        & df["top_m"].eq(top_m)
    ]
    if hit.empty:
        return np.nan
    return float(hit[col].iloc[0])


def pair_value_at(df: pd.DataFrame, support_method: str, pair_method: str, col: str) -> float:
    hit = df[
        df["source_method"].astype(str).eq(support_method)
        & df["pair_score_method"].astype(str).eq(pair_method)
        & df["function"].astype(str).eq("core_interaction_c025")
        & df["dimension"].eq(100)
        & df["samples"].eq(512)
        & df["top_m"].eq(4)
        & (df["num_runs"] >= 10)
    ]
    if hit.empty:
        return np.nan
    return float(hit[col].iloc[0])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Make a compact paper figure for support stabilization and pair scoring."
    )
    parser.add_argument("--combined_summary", required=True)
    parser.add_argument("--pair_summary", required=True)
    parser.add_argument("--one_shot_summary")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    configure_paper_plots(usetex=False)

    combined = numericize(
        pd.read_csv(args.combined_summary),
        ["samples", "dimension", "top_m", "interaction_f1_mean", "interaction_f1_std", "num_runs"],
    )
    pair = numericize(
        pd.read_csv(args.pair_summary),
        ["samples", "dimension", "top_m", "interaction_f1_mean", "interaction_f1_std", "num_runs"],
    )
    one_shot = None
    if args.one_shot_summary:
        one_shot = numericize(
            pd.read_csv(args.one_shot_summary),
            ["samples", "dimension", "top_m", "interaction_f1_mean", "interaction_f1_std", "num_runs"],
        )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.05, 2.62),
        gridspec_kw={"width_ratios": [1.02, 1.25, 1.0]},
    )

    single_settings = [
        ("core_interaction_c025", 512, 4, r"$c=.25$" + "\n" + r"$n=512$"),
        ("core_interaction_c025", 1024, 4, r"$c=.25$" + "\n" + r"$n=1024$"),
        ("core_interaction_c05", 512, 5, r"$c=.50$" + "\n" + r"$n=512$"),
    ]
    x0 = np.arange(len(single_settings))
    single_vals = [
        one_shot_value_at(one_shot, function, n, m, "interaction_f1_mean")
        for function, n, m, _ in single_settings
    ]
    stable_vals = [
        stability_value_at(combined, function, "feature_edge_hybrid", n, m, "interaction_f1_mean")
        for function, n, m, _ in single_settings
    ]
    width0 = 0.34
    axes[0].bar(
        x0 - width0 / 2,
        single_vals,
        width=width0,
        label="Single KAN",
        color=OKABE_ITO["gray"],
        edgecolor="white",
        linewidth=0.45,
    )
    axes[0].bar(
        x0 + width0 / 2,
        stable_vals,
        width=width0,
        label="Stability KAN-FE",
        color=OKABE_ITO["green"],
        edgecolor="white",
        linewidth=0.45,
    )
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Interaction F1")
    axes[0].set_xticks(x0)
    axes[0].set_xticklabels([label for *_, label in single_settings])
    axes[0].text(
        0.02,
        0.98,
        r"(a) Single pass",
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        fontsize=7.8,
    )
    axes[0].legend(frameon=False, loc="upper center", bbox_to_anchor=(0.55, 1.20), ncol=1)
    clean_axis(axes[0], grid=True)

    settings = [(512, 4), (1024, 4), (2048, 6)]
    x = np.arange(len(settings))
    width = 0.17
    for idx, method in enumerate(LEFT_METHODS):
        vals = [value_at(combined, method, n, m, "interaction_f1_mean") for n, m in settings]
        axes[1].bar(
            x + (idx - (len(LEFT_METHODS) - 1) / 2) * width,
            vals,
            width=width,
            label=LEFT_LABELS[method],
            color=LEFT_COLORS[method],
            edgecolor="white",
            linewidth=0.45,
        )
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("Interaction F1")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([rf"$n={n}$" + "\n" + rf"$m={m}$" for n, m in settings])
    axes[1].text(
        0.02,
        0.98,
        r"(b) Support controls",
        transform=axes[1].transAxes,
        ha="left",
        va="top",
        fontsize=7.8,
    )
    axes[1].text(0.02, 0.85, r"$c=0.25,\ d=100$", transform=axes[1].transAxes, ha="left", va="top", fontsize=7.2)
    axes[1].legend(
        frameon=False,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.56, 1.17),
        columnspacing=0.85,
        handlelength=1.05,
    )
    clean_axis(axes[1], grid=True)

    x2 = np.arange(len(PAIR_METHODS))
    width2 = 0.32
    for idx, support_method in enumerate(SUPPORT_METHODS):
        vals = [pair_value_at(pair, support_method, method, "interaction_f1_mean") for method in PAIR_METHODS]
        axes[2].bar(
            x2 + (idx - 0.5) * width2,
            vals,
            width=width2,
            label=SUPPORT_LABELS[support_method],
            color=SUPPORT_COLORS[support_method],
            alpha=0.90 if idx == 0 else 0.72,
            edgecolor="white",
            linewidth=0.45,
        )
    axes[2].set_ylim(0, 1.05)
    axes[2].set_ylabel("Interaction F1")
    axes[2].set_xticks(x2)
    axes[2].set_xticklabels([PAIR_LABELS[m] for m in PAIR_METHODS], rotation=18, ha="right")
    axes[2].text(
        0.02,
        0.98,
        r"(c) Pair scoring",
        transform=axes[2].transAxes,
        ha="left",
        va="top",
        fontsize=7.8,
    )
    axes[2].text(0.02, 0.85, r"$n=512,\ m=4$", transform=axes[2].transAxes, ha="left", va="top", fontsize=7.2)
    clean_axis(axes[2], grid=True)

    fig.tight_layout(w_pad=0.9)
    save_figure(fig, Path(args.out))
    plt.close(fig)


if __name__ == "__main__":
    main()
