from __future__ import annotations

from pathlib import Path

import matplotlib as mpl


OKABE_ITO = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "gray": "#6B7280",
}


def configure_paper_plots(*, usetex: bool = True) -> None:
    mpl.rcParams.update(
        {
            "text.usetex": usetex,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "CMU Serif", "Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "axes.titlesize": 9,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "figure.titlesize": 10,
            "axes.linewidth": 0.55,
            "axes.edgecolor": "#9CA3AF",
            "xtick.color": "#374151",
            "ytick.color": "#374151",
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 450,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.015,
        }
    )


def clean_axis(ax, *, grid: bool = False) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ["bottom", "left"]:
        ax.spines[side].set_color("#B6BDC8")
        ax.spines[side].set_linewidth(0.55)
    ax.tick_params(axis="both", colors="#374151", width=0.5, length=2.5)
    ax.set_axisbelow(True)
    if grid:
        ax.grid(True, axis="y", color="#D7DCE3", linestyle=(0, (1.5, 2.5)), linewidth=0.45)


def save_figure(fig, out_base: Path) -> None:
    out_base = Path(out_base)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".pdf"))
    fig.savefig(out_base.with_suffix(".png"), dpi=450)
    print("[saved]", out_base.with_suffix(".pdf"))
    print("[saved]", out_base.with_suffix(".png"))
