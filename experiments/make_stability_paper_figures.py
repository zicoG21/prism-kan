from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paper_plot_style import OKABE_ITO, clean_axis, configure_paper_plots, save_figure


TAG_RE = re.compile(
    r"(?P<function>core_interaction_c[0-9]+)_n(?P<n>[0-9]+)_d(?P<d>[0-9]+)_topm(?P<top_m>[0-9]+)"
)

METHOD_LABELS = {
    "raw": "Raw",
    "rf": "RF",
    "oracle_support": "Oracle",
    "ss_kan_variable": "SS-KAN-V",
    "ss_kan_pair": "SS-KAN-P",
}

METHOD_COLORS = {
    "raw": OKABE_ITO["gray"],
    "rf": OKABE_ITO["orange"],
    "oracle_support": OKABE_ITO["black"],
    "ss_kan_variable": OKABE_ITO["green"],
    "ss_kan_pair": OKABE_ITO["blue"],
}

METRIC_COLORS = {
    "Support": OKABE_ITO["gray"],
    "Endpoints": OKABE_ITO["blue"],
    "Pair F1": OKABE_ITO["vermillion"],
}


def parse_tag(path: Path) -> dict:
    match = TAG_RE.search(path.name)
    if not match:
        raise ValueError(f"Cannot parse tag from {path}")
    out = match.groupdict()
    out["n"] = int(out["n"])
    out["d"] = int(out["d"])
    out["top_m"] = int(out["top_m"])
    return out


def load_long_summaries(summary_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(summary_dir.glob("*_summary.csv")):
        meta = parse_tag(path)
        df = pd.read_csv(path)
        for key, value in meta.items():
            df[key] = value
        frames.append(df)
    if not frames:
        raise RuntimeError(f"No summary files found under {summary_dir}")
    return pd.concat(frames, ignore_index=True)


def method_table(df: pd.DataFrame, d: int, ss_top_m: int) -> pd.DataFrame:
    methods = ["raw", "rf", "oracle_support", "ss_kan_variable", "ss_kan_pair"]
    rows = []
    sub = df[df["d"] == d].copy()
    for _, row in sub.iterrows():
        mode = str(row["screen_mode"])
        if mode not in methods:
            continue
        if mode.startswith("ss_kan_") and int(row["top_m"]) != ss_top_m:
            continue
        if not mode.startswith("ss_kan_") and int(row["top_m"]) != 4:
            continue
        rows.append(row)
    out = pd.DataFrame(rows)
    out["screen_mode"] = pd.Categorical(out["screen_mode"], methods, ordered=True)
    return out.sort_values(["interaction_strength", "n", "screen_mode"])


def plot_main_interaction(table: pd.DataFrame, out_base: Path) -> None:
    methods = ["raw", "rf", "oracle_support", "ss_kan_variable", "ss_kan_pair"]
    strengths = sorted(table["interaction_strength"].dropna().unique())
    ns = sorted(table["n"].unique())

    fig, axes = plt.subplots(
        len(strengths),
        len(ns),
        figsize=(6.7, 4.35),
        sharey=True,
        constrained_layout=True,
    )
    if len(strengths) == 1:
        axes = np.array([axes])
    if len(ns) == 1:
        axes = axes.reshape(len(strengths), 1)

    for r, c in enumerate(strengths):
        for col, n in enumerate(ns):
            ax = axes[r, col]
            sub = table[(table["interaction_strength"] == c) & (table["n"] == n)]
            vals = []
            labels = []
            colors = []
            for method in methods:
                hit = sub[sub["screen_mode"] == method]
                vals.append(float(hit["interaction_f1_mean"].iloc[0]) if not hit.empty else np.nan)
                labels.append(METHOD_LABELS[method])
                colors.append(METHOD_COLORS[method])
            bars = ax.bar(
                np.arange(len(methods)),
                vals,
                color=colors,
                width=0.72,
                edgecolor="white",
                linewidth=0.45,
            )
            ax.set_ylim(0, 1.05)
            ax.set_title(rf"$c={c:g}$, $n={int(n)}$", pad=3)
            clean_axis(ax, grid=True)
            ax.set_xticks([])
            if col == 0:
                ax.set_ylabel("Interaction F1")

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=METHOD_COLORS[m], edgecolor="white", linewidth=0.45)
        for m in methods
    ]
    fig.legend(
        legend_handles,
        [METHOD_LABELS[m] for m in methods],
        loc="upper center",
        ncol=len(methods),
        frameon=False,
        bbox_to_anchor=(0.5, 1.04),
        columnspacing=1.0,
        handlelength=1.2,
    )
    save_figure(fig, out_base)
    plt.close(fig)


def plot_failure_ladder(table: pd.DataFrame, out_base: Path) -> None:
    sub = table[
        (table["interaction_strength"] == 0.25)
        & (table["d"] == 100)
        & (table["n"].isin([512, 1024]))
        & (table["screen_mode"].isin(["ss_kan_variable", "ss_kan_pair"]))
    ].copy()
    sub = sub[sub["top_m"].isin([5, 6])].copy()
    sub = sub.sort_values(["n", "screen_mode", "top_m"])
    sub["label"] = (
        sub["screen_mode"].map(METHOD_LABELS).astype(str).str.replace("SS-KAN-", "", regex=False)
        + ", $m="
        + sub["top_m"].astype(int).astype(str)
        + "$"
    )

    metrics = [
        ("screen_contains_all_true_vars_mean", "Support"),
        ("explain_interaction_endpoint_recall_mean", "Endpoints"),
        ("interaction_f1_mean", "Pair F1"),
    ]
    ns = sorted(sub["n"].unique())
    width = 0.24

    fig, axes = plt.subplots(1, len(ns), figsize=(6.7, 2.55), sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    legend_handles = []
    for ax_i, (ax, n) in enumerate(zip(axes, ns)):
        part = sub[sub["n"] == n].copy()
        x = np.arange(len(part))
        for idx, (metric, label) in enumerate(metrics):
            bars = ax.bar(
                x + (idx - 1) * width,
                part[metric],
                width=width,
                label=label,
                color=METRIC_COLORS[label],
                edgecolor="white",
                linewidth=0.45,
            )
            if ax_i == 0:
                legend_handles.append(bars[0])
        ax.set_ylim(0, 1.05)
        ax.set_title(rf"$n={int(n)}$", pad=3)
        ax.set_xticks(x)
        ax.set_xticklabels(part["label"], rotation=18, ha="right")
        clean_axis(ax, grid=True)
    axes[0].set_ylabel("Score")
    fig.legend(
        legend_handles,
        [label for _, label in metrics],
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.06),
    )
    save_figure(fig, out_base)
    plt.close(fig)


def best_stability_table(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[df["screen_mode"].isin(["ss_kan_variable", "ss_kan_pair"])].copy()
    sub = sub.sort_values(
        ["function", "d", "n", "interaction_f1_mean", "test_mse_mean"],
        ascending=[True, True, True, False, True],
    )
    best = sub.groupby(["function", "interaction_strength", "d", "n"], as_index=False).head(1)
    cols = [
        "function",
        "interaction_strength",
        "d",
        "n",
        "screen_mode",
        "top_m",
        "test_mse_mean",
        "variable_f1_mean",
        "explain_interaction_endpoint_recall_mean",
        "interaction_f1_mean",
        "true_interaction_mean_score_margin_mean",
    ]
    return best[cols].sort_values(["interaction_strength", "d", "n"])


def latex_table(table: pd.DataFrame, out_path: Path) -> None:
    rows = []
    rows.append(r"\begin{tabular}{rrrlrrrr}")
    rows.append(r"\toprule")
    rows.append(r"$c$ & $d$ & $n$ & Method & $m$ & MSE & Endpoint & Int. F1 \\")
    rows.append(r"\midrule")
    for _, row in table.iterrows():
        method = METHOD_LABELS.get(str(row["screen_mode"]), str(row["screen_mode"]))
        rows.append(
            f"{row['interaction_strength']:.2f} & "
            f"{int(row['d'])} & "
            f"{int(row['n'])} & "
            f"{method} & "
            f"{int(row['top_m'])} & "
            f"{row['test_mse_mean']:.3g} & "
            f"{row['explain_interaction_endpoint_recall_mean']:.2f} & "
            f"{row['interaction_f1_mean']:.2f} \\\\"
        )
    rows.append(r"\bottomrule")
    rows.append(r"\end{tabular}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary_dir",
        default="results/stability_kan/long_topm_long_boundary_v1/summaries",
    )
    parser.add_argument("--out_dir", default="results/stability_kan/paper_figures")
    parser.add_argument("--d", type=int, default=100)
    parser.add_argument("--ss_top_m", type=int, default=6)
    args = parser.parse_args()

    configure_paper_plots(usetex=True)

    out_dir = Path(args.out_dir)
    df = load_long_summaries(Path(args.summary_dir))
    main = method_table(df, d=args.d, ss_top_m=args.ss_top_m)
    best = best_stability_table(df)

    out_dir.mkdir(parents=True, exist_ok=True)
    main.to_csv(out_dir / f"sskan_d{args.d}_method_comparison.csv", index=False)
    best.to_csv(out_dir / "sskan_best_stability_settings.csv", index=False)
    latex_table(best, out_dir / "sskan_best_stability_settings.tex")
    plot_main_interaction(main, out_dir / f"fig5_sskan_boundary_d{args.d}")
    plot_failure_ladder(df, out_dir / "fig6_weak_signal_failure_ladder")

    print(main.to_string(index=False))
    print(f"Saved outputs under {out_dir}")


if __name__ == "__main__":
    main()
