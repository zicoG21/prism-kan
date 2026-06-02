#!/usr/bin/env python3
"""Plot threshold-sensitivity curves for the workshop manuscript.

This figure is deliberately small: it shows that the main conclusions are not
only artifacts of top-m=4 endpoint containment or rank-1 pair recovery.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "manuscripts" / "workshop_case_study" / "figures"
SOURCE_DIR = ROOT / "local_notes" / "generated"

READOUT_ROOT = ROOT / "results" / "revision" / "greatlakes_readout_taxonomy"
FULL_ROOT = ROOT / "results" / "revision" / "fullkan_anova_boundary"

SETTINGS = [
    ("Clean n=512", "clean_w16_n512", "clean_w16_n512_60seed", "#0072B2", "o"),
    ("Grid n=512", "gridupdate_w16_n512", "gridupdate_w16_n512_60seed", "#D55E00", "s"),
    ("Grid n=768", "gridupdate_w16_n768", "gridupdate_w16_n768_60seed", "#E69F00", "^"),
    ("Grid n=1024", "gridupdate_w16_n1024", "gridupdate_w16_n1024_60seed", "#009E73", "D"),
]


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#222222",
            "axes.linewidth": 0.75,
            "axes.labelsize": 8.2,
            "axes.titlesize": 9.0,
            "axes.titleweight": "bold",
            "xtick.labelsize": 7.4,
            "ytick.labelsize": 7.4,
            "legend.fontsize": 7.0,
            "grid.color": "#E5E7EB",
            "grid.linewidth": 0.45,
            "axes.axisbelow": True,
        }
    )


def clean_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ["bottom", "left"]:
        ax.spines[side].set_color("#9CA3AF")
        ax.spines[side].set_linewidth(0.7)
    ax.tick_params(axis="both", colors="#374151", width=0.6, length=2.8)
    ax.grid(True, axis="y")


def direct_label(ax, x: float, y: float, text: str, color: str, *, va: str = "center") -> None:
    ax.text(
        x,
        y,
        text,
        color=color,
        fontsize=7.0,
        va=va,
        ha="left",
        path_effects=[pe.withStroke(linewidth=2.8, foreground="white")],
        clip_on=False,
    )


def read_endpoint_curves() -> pd.DataFrame:
    records = []
    for label, read_dir, _, color, marker in SETTINGS:
        path = READOUT_ROOT / read_dir / "support_sensitivity_summary.csv"
        df = pd.read_csv(path)
        df = df[df["method"].eq("feature_edge_hybrid")].copy()
        for _, row in df.sort_values("top_m").iterrows():
            records.append(
                {
                    "setting": label,
                    "budget": int(row["top_m"]),
                    "endpoint_rate": float(row["screen_contains_all_interaction_endpoints_mean"]),
                    "endpoint_rank": float(row["true_endpoint_rank_worst_mean"]),
                    "color": color,
                    "marker": marker,
                }
            )
    return pd.DataFrame(records)


def read_pair_curves() -> pd.DataFrame:
    cutoffs = [1, 5, 10, 50, 100, 500, 1000, 2000]
    records = []
    for label, _, full_dir, color, marker in SETTINGS:
        path = FULL_ROOT / full_dir / "full_kan_pair_anova_detail.csv"
        df = pd.read_csv(path)
        ranks = pd.to_numeric(df["true_pair_rank"], errors="coerce")
        for q in cutoffs:
            records.append(
                {
                    "setting": label,
                    "cutoff": q,
                    "pair_rate": float((ranks <= q).mean()),
                    "mean_rank": float(ranks.mean()),
                    "color": color,
                    "marker": marker,
                }
            )
    return pd.DataFrame(records)


def main() -> None:
    set_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)

    endpoint = read_endpoint_curves()
    pair = read_pair_curves()
    endpoint.to_csv(SOURCE_DIR / "budget_endpoint_curve_source.csv", index=False)
    pair.to_csv(SOURCE_DIR / "budget_pair_curve_source.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(7.35, 2.62), gridspec_kw={"width_ratios": [1.0, 1.22]})
    fig.suptitle("Predicate sensitivity, not evidence-object equivalence", y=0.975, fontsize=10.2, weight="bold")
    fig.text(
        0.5,
        0.895,
        "Sweeping the cutoff changes how strict a stage is; it does not transfer a claim from one stage to another.",
        ha="center",
        va="center",
        fontsize=7.6,
        color="#6B7280",
    )

    ax = axes[0]
    for label, _, _, color, marker in SETTINGS:
        g = endpoint[endpoint["setting"].eq(label)].sort_values("budget")
        ax.plot(
            g["budget"],
            g["endpoint_rate"],
            color=color,
            marker=marker,
            linewidth=1.35,
            markersize=3.4,
            label=label,
        )
    ax.set_title("(a) Exposed readout: endpoint budget", loc="left")
    ax.set_xlabel("support budget $m$")
    ax.set_ylabel("endpoints retained")
    ax.set_xticks([4, 6, 10, 20])
    ax.set_xlim(3.2, 23.2)
    ax.set_ylim(-0.04, 1.04)
    clean_axis(ax)
    ax.text(0.05, 0.09, "KAN-FE readout", transform=ax.transAxes, fontsize=7.2, color="#374151")
    direct_label(ax, 20.65, 1.00, "clean / grid 768 / grid 1024", "#0F766E")
    direct_label(ax, 20.65, 0.035, "grid 512", "#D55E00")
    ax.annotate(
        "endpoint claim\nunchanged by larger $m$",
        xy=(10, 1.0),
        xytext=(7.0, 0.78),
        fontsize=6.8,
        color="#374151",
        arrowprops=dict(arrowstyle="-|>", lw=0.7, color="#9CA3AF"),
    )

    ax = axes[1]
    for label, _, _, color, marker in SETTINGS:
        g = pair[pair["setting"].eq(label)].sort_values("cutoff")
        ax.plot(
            g["cutoff"],
            g["pair_rate"],
            color=color,
            marker=marker,
            linewidth=1.35,
            markersize=3.2,
            label=label,
        )
    ax.set_title("(b) Fitted function: pair-rank cutoff", loc="left")
    ax.set_xlabel("pair-rank cutoff $q$")
    ax.set_ylabel("true pair within top-$q$")
    ax.set_xscale("log")
    ax.set_xticks([1, 5, 10, 50, 100, 500, 1000, 2000])
    ax.set_xticklabels(["1", "5", "10", "50", "100", "500", "1k", "2k"], rotation=20)
    ax.set_xlim(0.65, 3600)
    ax.set_ylim(-0.04, 1.04)
    clean_axis(ax)
    direct_label(ax, 2150, 0.87, "grid 1024", "#009E73")
    direct_label(ax, 2150, 0.72, "clean 512", "#0072B2")
    direct_label(ax, 2150, 0.53, "grid 768", "#E69F00")
    direct_label(ax, 2150, 0.48, "grid 512", "#D55E00", va="top")
    ax.axvline(1, color="#9CA3AF", lw=0.8, ls=(0, (2, 2)))
    ax.text(
        1.08,
        0.96,
        "rank-1",
        fontsize=6.8,
        color="#6B7280",
        va="top",
        path_effects=[pe.withStroke(linewidth=2.8, foreground="white")],
    )
    ax.annotate(
        "pair evidence remains\ngraded even as $q$ grows",
        xy=(500, 0.50),
        xytext=(27, 0.70),
        fontsize=6.8,
        color="#374151",
        arrowprops=dict(arrowstyle="-|>", lw=0.7, color="#9CA3AF"),
    )

    fig.subplots_adjust(left=0.075, right=0.965, bottom=0.19, top=0.76, wspace=0.58)

    for ext in ("pdf", "png"):
        out = OUT_DIR / f"budget_sensitivity.{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=300)
        print(f"wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
