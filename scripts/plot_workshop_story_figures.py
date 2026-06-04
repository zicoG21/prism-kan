#!/usr/bin/env python3
"""Create compact, story-first figures for the workshop manuscript.

The figures are intentionally small and self-contained.  They turn the main
stage-record evidence into visual objects so the workshop paper reads less like
an experiment log.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "manuscripts" / "workshop_foundation" / "figures"
HORIZONTAL = ROOT / "local_notes" / "generated" / "horizontal_evidence_table_20260531.csv"
STAGE_DISCORDANCE = ROOT / "results" / "revision" / "stage_discordance_summary.csv"
MINISUITE = ROOT / "results" / "workshop_review_tables" / "formal_minisuite" / "formal_minisuite_baseline_table.csv"

COLORS = {
    "ink": "#222222",
    "muted": "#6B7280",
    "grid": "#E5E7EB",
    "panel": "#F8FAFC",
    "prediction": "#EEF2F7",
    "full": "#DCEEF3",
    "readout": "#E8E1F3",
    "refit": "#F2E8D5",
    "extract": "#F4DEDD",
    "claim": "#DBEAD8",
    # Okabe-Ito-ish anchors.
    "blue": "#0072B2",
    "sky": "#56B4E9",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#8B70B8",
    "gray": "#6B7280",
}

RATE_CMAP = LinearSegmentedColormap.from_list(
    "claim_rate",
    ["#F7F7F7", "#F6E8A6", "#A8DDB5", "#2B8CBE", "#084081"],
)


def set_style() -> None:
    """Use a compact vector-friendly style for conference PDFs."""

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": COLORS["ink"],
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.labelsize": 8.8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.8,
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.45,
            "grid.color": COLORS["grid"],
            "axes.axisbelow": True,
            "savefig.facecolor": "white",
        }
    )


def savefig(name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        path = OUT_DIR / f"{name}.{ext}"
        plt.savefig(path, bbox_inches="tight", dpi=300)
        print(f"wrote {path}")
    plt.close()


def stage_record_flow() -> None:
    fig, ax = plt.subplots(figsize=(10.0, 2.85))
    ax.set_axis_off()

    stages = [
        ("Prediction", "test MSE\nsignal scale"),
        ("Full KAN", "pair reliance\nANOVA rank"),
        ("Readout", "endpoint rank\nmargin"),
        ("Refit", "pair scorer\non support"),
        ("Prune / symbolic", "retained vars\nformula status"),
        ("Claim", "support + pair\nwith provenance"),
    ]
    xs = np.linspace(0.088, 0.912, len(stages))
    y = 0.61
    box_w, box_h = 0.137, 0.37
    colors = [
        COLORS["prediction"],
        COLORS["full"],
        COLORS["readout"],
        COLORS["refit"],
        COLORS["extract"],
        COLORS["claim"],
    ]

    for idx, ((title, body), x, c) in enumerate(zip(stages, xs, colors), start=1):
        patch = FancyBboxPatch(
            (x - box_w / 2, y - box_h / 2),
            box_w,
            box_h,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            facecolor=c,
            edgecolor="#334155",
            linewidth=0.9,
            transform=ax.transAxes,
        )
        patch.set_path_effects([pe.SimplePatchShadow(offset=(1.1, -1.1), alpha=0.12), pe.Normal()])
        ax.add_patch(patch)
        ax.text(
            x - box_w / 2 + 0.014,
            y + box_h / 2 - 0.036,
            f"{idx}",
            ha="center",
            va="center",
            fontsize=6.3,
            color="white",
            weight="bold",
            bbox=dict(boxstyle="circle,pad=0.14", facecolor="#334155", edgecolor="#334155", linewidth=0),
            transform=ax.transAxes,
        )
        ax.text(x, y + 0.074, title, ha="center", va="center", fontsize=8.7, weight="bold", color=COLORS["ink"], transform=ax.transAxes)
        ax.text(x, y - 0.060, body, ha="center", va="center", fontsize=7.25, color="#374151", transform=ax.transAxes)
        if idx < len(stages):
            arrow = FancyArrowPatch(
                (x + box_w / 2 + 0.008, y),
                (xs[idx] - box_w / 2 - 0.008, y),
                arrowstyle="-|>",
                mutation_scale=10,
                lw=1.05,
                color="#475569",
                transform=ax.transAxes,
            )
            ax.add_patch(arrow)

    callouts = [
        (0.19, 0.18, "C1 prediction is not structure", COLORS["red"]),
        (0.50, 0.075, "C2 evidence stages can disagree", COLORS["blue"]),
        (0.80, 0.18, "C3 extraction changes provenance", "#9a6b20"),
    ]
    for x, yy, txt, col in callouts:
        ax.text(
            x,
            yy,
            txt,
            ha="center",
            va="center",
            fontsize=7.8,
            color=col,
            bbox=dict(boxstyle="round,pad=0.25,rounding_size=0.08", facecolor="white", edgecolor=col, linewidth=0.9),
            transform=ax.transAxes,
        )

    ax.text(
        0.5,
        0.94,
        "Claim provenance through a KAN workflow",
        ha="center",
        va="center",
        fontsize=9.8,
        weight="bold",
        color=COLORS["ink"],
        transform=ax.transAxes,
    )
    ax.text(
        0.5,
        0.885,
        "The same support or pair statement means different things at different stages.",
        ha="center",
        va="center",
        fontsize=7.5,
        color=COLORS["muted"],
        transform=ax.transAxes,
    )
    savefig("stage_record_flow")


def stage_discordance_heatmap() -> None:
    df = pd.read_csv(HORIZONTAL)
    rows = [
        ("clean w16 n512", "Clean w16 n=512"),
        ("clean w16 n1024", "Clean w16 n=1024"),
        ("gridupdate w16 n512", "Grid update n=512"),
        ("gridupdate w16 n1024", "Grid update n=1024"),
        ("noise010 w16 n512", "Noise .10 n=512"),
        ("noise010 w16 n1024", "Noise .10 n=1024"),
        ("clean w32 n768", "Clean w32 n=768"),
    ]
    df = df.set_index("condition").loc[[r[0] for r in rows]].reset_index()

    values = df[["full_rate", "readout_rate"]].copy()
    values["prune_rate"] = df["prune_endpoints"].str.split("/").apply(lambda x: int(x[0]) / int(x[1]))
    mat = values.to_numpy(float)

    counts = np.column_stack(
        [
            df["full_rank1"].to_numpy(str),
            df["readout_endpoints"].to_numpy(str),
            df["prune_endpoints"].to_numpy(str),
        ]
    )

    fig, ax = plt.subplots(figsize=(7.45, 3.05))
    im = ax.imshow(mat, cmap=RATE_CMAP, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(3))
    ax.set_xticklabels(["Full function\npair rank-1", "Exposed readout\nendpoints@4", "Pruned support\nendpoints"], fontsize=8.0)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[1] for r in rows], fontsize=7.8)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            color = "white" if mat[i, j] > 0.72 else COLORS["ink"]
            ax.text(j, i - 0.08, counts[i, j], ha="center", va="center", fontsize=7.6, weight="bold", color=color)
            ax.text(j, i + 0.17, f"{mat[i,j]:.2f}", ha="center", va="center", fontsize=6.3, color=color)

    ax.set_title("pyKAN claim transfer splits by workflow object", fontsize=9.2, weight="bold", pad=11)
    ax.text(
        0.5,
        1.015,
        "Counts are matched settings; columns authorize different typed claims.",
        ha="center",
        va="bottom",
        transform=ax.transAxes,
        fontsize=6.8,
        color=COLORS["muted"],
    )
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.6)
    ax.tick_params(which="minor", bottom=False, left=False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.030, pad=0.028)
    cbar.set_label("success rate", fontsize=7.2)
    cbar.ax.tick_params(labelsize=7)
    ax.text(
        0.5,
        -0.17,
        "The discordance across columns is the audit target, not a metric inconsistency.",
        ha="center",
        va="top",
        transform=ax.transAxes,
        fontsize=6.7,
        color=COLORS["muted"],
    )
    savefig("stage_discordance_heatmap")


def stage_discordance_phase_diagram() -> None:
    """Plot the joined stage record as an evidence-object phase diagram."""

    df = pd.read_csv(STAGE_DISCORDANCE)
    df["full_rank1_rate"] = pd.to_numeric(df["full_rank1_rate"], errors="coerce")
    df["readout_endpoints_at4"] = pd.to_numeric(df["readout_endpoints_at4"], errors="coerce")
    df["full_margin"] = pd.to_numeric(df["full_margin"], errors="coerce")
    df["readout_margin"] = pd.to_numeric(df["readout_margin"], errors="coerce")
    df = df.dropna(subset=["full_rank1_rate", "readout_endpoints_at4"]).copy()

    # Deterministic jitter keeps repeated readout-family rows visible without
    # implying extra uncertainty.
    rng = np.random.default_rng(20260531)
    x = np.clip(df["full_rank1_rate"].to_numpy() + rng.normal(0, 0.012, size=len(df)), -0.03, 1.03)
    y = np.clip(df["readout_endpoints_at4"].to_numpy() + rng.normal(0, 0.012, size=len(df)), -0.03, 1.03)

    colors = {
        "clean": COLORS["blue"],
        "gridupdate": COLORS["orange"],
        "noise005": COLORS["green"],
        "noise010": COLORS["purple"],
    }
    labels = {
        "clean": "clean",
        "gridupdate": "grid update",
        "noise005": "noise .05",
        "noise010": "noise .10",
    }

    fig, ax = plt.subplots(figsize=(6.75, 4.0))

    # Regions follow the classification used in make_stage_discordance_summary.
    regions = [
        ((-0.03, 0.8), 0.53, 0.26, "#FBEAE7"),  # surfacing without reliance
        ((0.8, 1.03), -0.03, 0.53, "#E9EEF8"),  # reliance without surfacing
        ((0.8, 1.03), 0.8, 0.23, "#E7F3E6"),  # aligned high
        ((-0.03, 0.5), -0.03, 0.53, "#F1F2F4"),  # aligned low
    ]
    for (x0, x1), y0, h, color in regions:
        ax.add_patch(Rectangle((x0, y0), x1 - x0, h, facecolor=color, edgecolor="none", alpha=0.88, zorder=0))

    for cond, group in df.groupby("condition"):
        idx = group.index.to_numpy()
        ax.scatter(
            x[df.index.get_indexer(idx)],
            y[df.index.get_indexer(idx)],
            s=24,
            color=colors.get(cond, "#666666"),
            alpha=0.78,
            edgecolor="white",
            linewidth=0.45,
            label=labels.get(cond, cond),
        )

    ax.axvline(0.5, color="#64748B", lw=0.9, ls="--")
    ax.axhline(0.5, color="#64748B", lw=0.9, ls="--")
    ax.axvline(0.8, color="#94A3B8", lw=0.8, ls=":")
    ax.axhline(0.8, color="#94A3B8", lw=0.8, ls=":")

    counts = df["discordance_label"].value_counts()
    ax.text(0.18, 0.94, f"surfacing without reliance\n{counts.get('surfacing without reliance', 0)} records",
            ha="center", va="center", fontsize=7.0, color="#8a1f1f")
    ax.text(0.91, 0.93, f"aligned high\n{counts.get('aligned high', 0)}",
            ha="center", va="center", fontsize=7.0, color="#216b2b")
    ax.text(0.18, 0.15, f"aligned low\n{counts.get('aligned low', 0)}",
            ha="center", va="center", fontsize=7.0, color="#444444")
    ax.text(0.57, 0.55, f"mixed boundary\n{counts.get('mixed boundary', 0)}",
            ha="center", va="center", fontsize=7.6, color="#5a4b8a",
            bbox=dict(facecolor="white", edgecolor="#b8b8b8", boxstyle="round,pad=0.25", alpha=0.85))

    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("Full-KAN pair reliance: true-pair rank-1 rate")
    ax.set_ylabel("Exposed-readout endpoint surfacing: endpoints@4")
    ax.set_title("Full-model reliance vs exposed-readout surfacing", fontsize=9.8, weight="bold", pad=7)
    ax.grid(alpha=0.30)
    ax.legend(frameon=True, loc="lower right", fontsize=7.6, edgecolor="#E5E7EB", facecolor="white", title="condition", title_fontsize=7.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    savefig("stage_discordance_phase_diagram")


def breadth_summary() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.68), gridspec_kw={"width_ratios": [1.00, 1.86, 1.00]})

    # Semi-synthetic real-covariate geometry at noise 0.05.
    ax = axes[0]
    datasets = ["breast\ncancer", "diabetes", "wine"]
    kan = np.array([90 / 90, 83 / 90, 64 / 90])
    residual = np.array([0 / 90, 85 / 90, 61 / 90])
    x = np.arange(len(datasets))
    w = 0.34
    for i in range(len(datasets)):
        ax.plot([kan[i], residual[i]], [i, i], color="#CBD5E1", lw=2.2, zorder=1)
    ax.scatter(kan, x, s=54, color=COLORS["blue"], label="KAN-FE endpoints", zorder=3, edgecolor="white", linewidth=0.6)
    ax.scatter(residual, x, s=54, color=COLORS["orange"], label="Residual top-1", zorder=3, edgecolor="white", linewidth=0.6)
    for i, (kv, rv) in enumerate(zip(kan, residual)):
        ax.text(kv + (0.035 if kv < 0.94 else -0.035), i - 0.16, f"{kv:.2f}", fontsize=6.5, color=COLORS["blue"], ha="left" if kv < 0.94 else "right")
        ax.text(rv + (0.035 if rv < 0.94 else -0.035), i + 0.18, f"{rv:.2f}", fontsize=6.5, color=COLORS["orange"], ha="left" if rv < 0.94 else "right")
    ax.set_title("(a) Real covariates", loc="left", fontsize=9.2, weight="bold")
    ax.set_xlim(-0.04, 1.04)
    ax.set_yticks(x)
    ax.set_yticklabels(datasets, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("recovery rate")
    ax.legend(
        fontsize=7.0,
        frameon=True,
        loc="lower left",
        bbox_to_anchor=(0.02, 0.05),
        ncol=1,
        handletextpad=0.25,
        borderpad=0.25,
        labelspacing=0.25,
        facecolor="white",
        edgecolor="#E5E7EB",
        framealpha=0.92,
    )
    ax.grid(axis="x", alpha=0.35)

    # Mini-suite formula-level summary.
    ax = axes[1]
    mini = pd.read_csv(MINISUITE)
    order = [
        "weak-centered",
        "nested-trig",
        "three-way-product",
        "mixed-sparse",
        "rational-product",
        "division-mixed",
        "bilinear",
        "trig-product",
        "log-product",
        "exp-product",
        "sqrt-energy",
    ]
    mini = mini.set_index("family").loc[order].reset_index()
    yy = np.arange(len(mini))
    true_vals = mini["true_support_kan"].to_numpy(float)
    rf_vals = mini["rf_support_kan"].to_numpy(float)
    for y0, tv, rv in zip(yy, true_vals, rf_vals):
        ax.plot([rv, tv], [y0, y0], color="#CBD5E1", lw=1.4, zorder=1)
    ax.scatter(true_vals, yy, s=30, color=COLORS["green"], label="True support", zorder=3, edgecolor="white", linewidth=0.50)
    ax.scatter(rf_vals, yy, s=30, color=COLORS["purple"], label="RF support", zorder=3, edgecolor="white", linewidth=0.50)
    ax.set_title("(b) Formula suite", loc="left", fontsize=9.2, weight="bold")
    ax.set_xlim(-0.04, 1.04)
    ax.set_yticks(yy)
    ax.set_yticklabels(mini["family"], fontsize=7.2)
    ax.invert_yaxis()
    ax.set_xlabel("pair recovery", labelpad=1.5)
    ax.legend(fontsize=6.9, frameon=False, loc="lower right", handletextpad=0.25)
    ax.grid(axis="x", alpha=0.30)

    # NID-style neural pair-score controls.
    ax = axes[2]
    labels = ["Weak\ncent.", "Strong\ncent.", "Bilin.", "Log\nprod.", "Exp\nprod."]
    vals = [0.00, 0.995, 0.99, 0.97, 1.00]
    colors = [COLORS["red"], COLORS["green"], COLORS["green"], COLORS["green"], COLORS["green"]]
    xx = np.arange(len(labels))
    ax.vlines(xx, 0, vals, color=colors, lw=3.2, alpha=0.85)
    ax.scatter(xx, vals, color=colors, s=54, edgecolor="white", linewidth=0.6, zorder=3)
    for idx, v in enumerate(vals):
        ax.text(idx, min(v + 0.045, 1.04), f"{v:.2f}", ha="center", fontsize=6.6, color=colors[idx])
    ax.set_title("(c) NID-style controls", loc="left", fontsize=9.2, weight="bold")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=7.6)
    ax.set_ylabel("pair F1")
    ax.grid(axis="y", alpha=0.30)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    savefig("breadth_summary")


def main() -> None:
    set_style()
    stage_record_flow()
    stage_discordance_heatmap()
    stage_discordance_phase_diagram()
    breadth_summary()


if __name__ == "__main__":
    main()
