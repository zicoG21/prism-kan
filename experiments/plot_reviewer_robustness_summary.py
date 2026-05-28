from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.paper_plot_style import OKABE_ITO, clean_axis, configure_paper_plots, save_figure


METHOD_ORDER = ["rf", "mi", "lasso", "lassonet"]
METHOD_LABELS = {
    "rf": "RF",
    "mi": "MI",
    "lasso": "Lasso",
    "lassonet": "LassoNet",
    "fasskan": "Residual\nFA-SS-KAN",
}
METHOD_COLORS = {
    "rf": OKABE_ITO["gray"],
    "mi": OKABE_ITO["sky"],
    "lasso": OKABE_ITO["orange"],
    "lassonet": OKABE_ITO["blue"],
    "fasskan": OKABE_ITO["green"],
}


def load_fasskan_row(summary_path: Path, cache_path: Path) -> dict:
    summary = pd.read_csv(summary_path)
    hit = summary[
        (summary["method"] == "residual_feature_stability_var")
        & (pd.to_numeric(summary["top_m"], errors="coerce") == 8)
    ]
    if hit.empty:
        hit = summary.head(1)
    row = hit.iloc[0]
    cache = pd.read_csv(cache_path)
    return {
        "method": "fasskan",
        "exact_pair_retained_mean": float(row["screen_contains_true_interactions"]),
        "exact_endpoint_recall_mean": float(row["screen_interaction_endpoint_recall"]),
        "screen_runtime_sec_mean": float(pd.to_numeric(cache["runtime_sec"], errors="coerce").sum()),
        "rho": float(row["nuisance_correlation"]),
        "noise": float(row["noise"]),
        "top_m": int(row["top_m"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--external_summary",
        type=Path,
        default=Path("results/reviewer_robustness_audit/quick_c025_d100_n1024_top20/support_baseline_summary.csv"),
    )
    parser.add_argument(
        "--fasskan_summary",
        type=Path,
        default=Path("results/reviewer_robustness_audit/fasskan_support_c025_d100_n1024_rho09_noise01/residual_support_summary.csv"),
    )
    parser.add_argument(
        "--fasskan_cache",
        type=Path,
        default=Path("results/reviewer_robustness_audit/fasskan_support_c025_d100_n1024_rho09_noise01/residual_probe_cache.csv"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/reviewer_robustness_audit/figures/reviewer_robustness_summary"),
    )
    args = parser.parse_args()

    configure_paper_plots(usetex=False)
    external = pd.read_csv(args.external_summary)
    fasskan = load_fasskan_row(args.fasskan_summary, args.fasskan_cache)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.2, 2.35),
        gridspec_kw={"width_ratios": [1.0, 1.0, 1.05], "wspace": 0.42},
    )

    rhos = sorted(external["rho"].unique())
    for ax, noise in zip(axes[:2], sorted(external["noise"].unique())):
        subset = external[external["noise"] == noise]
        for method in METHOD_ORDER:
            hit = subset[subset["method"] == method].sort_values("rho")
            ax.plot(
                hit["rho"],
                hit["exact_pair_retained_mean"],
                marker="o",
                markersize=3.2,
                linewidth=1.1,
                color=METHOD_COLORS[method],
                label=METHOD_LABELS[method],
            )
        ax.set_title(f"noise={noise:g}", loc="left", pad=3)
        ax.set_xlabel("Proxy correlation $\\rho$")
        ax.set_xticks(rhos)
        ax.set_ylim(-0.05, 1.05)
        ax.set_yticks([0, 0.5, 1.0])
        clean_axis(ax, grid=True)
    axes[0].set_ylabel("Exact pair retained")
    axes[1].legend(frameon=False, loc="lower left", ncol=2, handlelength=1.4)

    hard = external[(external["rho"] == fasskan["rho"]) & (external["noise"] == fasskan["noise"])].copy()
    hard_rows = []
    for method in METHOD_ORDER:
        hit = hard[hard["method"] == method]
        if not hit.empty:
            hard_rows.append(
                {
                    "method": method,
                    "pair": float(hit["exact_pair_retained_mean"].iloc[0]),
                    "runtime": float(hit["screen_runtime_sec_mean"].iloc[0]),
                }
            )
    hard_rows.append({"method": "fasskan", "pair": fasskan["exact_pair_retained_mean"], "runtime": fasskan["screen_runtime_sec_mean"]})
    hard_df = pd.DataFrame(hard_rows)

    ax = axes[2]
    x = np.arange(len(hard_df))
    bars = ax.bar(
        x,
        hard_df["pair"],
        color=[METHOD_COLORS[m] for m in hard_df["method"]],
        edgecolor="#1F2937",
        linewidth=0.35,
        width=0.68,
    )
    for bar, row in zip(bars, hard_df.itertuples()):
        runtime = float(row.runtime)
        runtime_text = f"{runtime:.1f}s" if runtime < 10 else f"{runtime:.0f}s"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(1.04, bar.get_height() + 0.045),
            runtime_text,
            ha="center",
            va="bottom",
            fontsize=6.7,
            color="#374151",
        )
    ax.set_title("$\\rho=0.9$, noise=0.1", loc="left", pad=3)
    ax.set_ylabel("Exact pair retained")
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[m] for m in hard_df["method"]], rotation=35, ha="right")
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.5, 1.0])
    clean_axis(ax, grid=True)

    fig.suptitle("$d=100,n=1024,c=0.25$: correlated/noisy support robustness", y=1.03, fontsize=9.5)
    save_figure(fig, args.out)


if __name__ == "__main__":
    main()
