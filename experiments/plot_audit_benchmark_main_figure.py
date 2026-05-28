from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from experiments.paper_plot_style import OKABE_ITO, clean_axis, configure_paper_plots, save_figure
except Exception:  # pragma: no cover
    from paper_plot_style import OKABE_ITO, clean_axis, configure_paper_plots, save_figure


def add_count_labels(ax, bars, counts: list[str]) -> None:
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(height + 0.035, 1.04),
            count,
            ha="center",
            va="bottom",
            fontsize=7.0,
            color="#374151",
        )


def main() -> None:
    configure_paper_plots(usetex=False)
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.15, 2.45),
        gridspec_kw={"width_ratios": [1.0, 1.15, 1.35]},
    )

    # Panel A: prediction calibration.
    c_labels = ["0.10", "0.25", "0.50", "1.00"]
    x = np.arange(len(c_labels))
    additive = np.array([0.00191, 0.01181, 0.04574, 0.16196])
    raw_kan = np.array([0.00249, 0.00581, 0.01145, 0.06369])
    width = 0.36
    axes[0].bar(
        x - width / 2,
        additive,
        width,
        label="Additive-only",
        color=OKABE_ITO["gray"],
        edgecolor="white",
        linewidth=0.45,
    )
    axes[0].bar(
        x + width / 2,
        raw_kan,
        width,
        label="Raw KAN",
        color=OKABE_ITO["blue"],
        edgecolor="white",
        linewidth=0.45,
    )
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Test MSE")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(c_labels)
    axes[0].set_xlabel("Interaction strength c")
    axes[0].text(0.02, 0.98, "(a) Prediction calibration", transform=axes[0].transAxes, ha="left", va="top", fontsize=7.8)
    axes[0].legend(frameon=False, loc="upper left", bbox_to_anchor=(-0.02, 1.23), ncol=1)
    clean_axis(axes[0], grid=True)

    # Panel B: finite-data KAN-FE sample-size grid at d=100.
    n_grid = np.array([512, 640, 768, 896, 1024, 1280])
    pair_success = np.array([0 / 30, 1 / 30, 6 / 30, 22 / 30, 30 / 30, 30 / 30], dtype=float)
    endpoint_recall = np.array([0.05, 0.10, 0.35, 0.7833, 1.0, 1.0])
    axes[1].plot(
        n_grid,
        pair_success,
        marker="o",
        linewidth=1.6,
        markersize=4.0,
        color=OKABE_ITO["green"],
        label="Top-1 pair",
    )
    axes[1].plot(
        n_grid,
        endpoint_recall,
        marker="s",
        linewidth=1.2,
        markersize=3.8,
        color=OKABE_ITO["purple"],
        label="Endpoint recall",
    )
    for n, y, count in zip(n_grid, pair_success, ["0/30", "1/30", "6/30", "22/30", "30/30", "30/30"]):
        axes[1].text(n, min(y + 0.06, 1.05), count, ha="center", va="bottom", fontsize=6.8, color="#374151")
    axes[1].set_ylim(0, 1.12)
    axes[1].set_ylabel("Recovery")
    axes[1].set_xticks(n_grid)
    axes[1].set_xticklabels([str(n) for n in n_grid], rotation=25, ha="right")
    axes[1].set_xlabel("Training samples n")
    axes[1].text(0.02, 0.98, "(b) Same-data KAN-FE at d=100", transform=axes[1].transAxes, ha="left", va="top", fontsize=7.8)
    axes[1].legend(frameon=False, loc="upper center", bbox_to_anchor=(0.55, 1.23), ncol=2)
    clean_axis(axes[1], grid=True)

    # Panel C: high-dimensional failure and calibration.
    settings = ["c=.25\nn=2048", "c=.50\nn=1024", "c=1.0\nn=1024"]
    methods_hd = ["KAN-FE", "RF", "Resid."]
    vals = np.array(
        [
            [0 / 8, 0 / 8, 6 / 10],
            [0 / 8, 0 / 8, 10 / 10],
            [0 / 8, 3 / 8, 10 / 10],
        ]
    )
    count_labels = [
        ["0/8", "0/8", "6/10"],
        ["0/8", "0/8", "10/10"],
        ["0/8", "3/8", "10/10"],
    ]
    x3 = np.arange(len(settings))
    width3 = 0.19
    offsets = np.linspace(-width3, width3, len(methods_hd))
    colors_hd = [OKABE_ITO["green"], OKABE_ITO["orange"], OKABE_ITO["blue"]]
    for k, method in enumerate(methods_hd):
        bars = axes[2].bar(
            x3 + offsets[k],
            vals[:, k],
            width3,
            label=method,
            color=colors_hd[k],
            edgecolor="white",
            linewidth=0.45,
        )
        add_count_labels(axes[2], bars, [row[k] for row in count_labels])
    axes[2].set_ylim(0, 1.14)
    axes[2].set_ylabel("Top-1 pair acc.")
    axes[2].set_xticks(x3)
    axes[2].set_xticklabels(settings)
    axes[2].set_xlabel("d=1000 stress rows")
    axes[2].text(0.02, 0.98, "(c) Support failure at d=1000", transform=axes[2].transAxes, ha="left", va="top", fontsize=7.8)
    axes[2].legend(frameon=False, loc="upper center", bbox_to_anchor=(0.52, 1.23), ncol=4)
    clean_axis(axes[2], grid=True)

    fig.tight_layout(w_pad=1.2)
    save_figure(fig, Path("results/innovation_loop/final_candidate_figures/audit_benchmark_main"))


if __name__ == "__main__":
    main()
