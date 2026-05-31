#!/usr/bin/env python3
"""Create compact, story-first figures for the workshop manuscript.

The figures are intentionally small and self-contained.  They turn the main
stage-record evidence into visual objects so the workshop paper reads less like
an experiment log.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "manuscripts" / "workshop_case_study" / "figures"
HORIZONTAL = ROOT / "local_notes" / "generated" / "horizontal_evidence_table_20260531.csv"
STAGE_DISCORDANCE = ROOT / "results" / "revision" / "stage_discordance_summary.csv"

COLORS = {
    "prediction": "#e8edf3",
    "full": "#d7ecef",
    "readout": "#e8e0f3",
    "refit": "#f2ead8",
    "extract": "#f4dfdd",
    "claim": "#dcebd8",
    "blue": "#4C78A8",
    "orange": "#F58518",
    "green": "#54A24B",
    "red": "#B94A48",
    "purple": "#8B7EC8",
    "gray": "#6B6B6B",
}


def set_style() -> None:
    """Use a compact vector-friendly style for conference PDFs."""

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.45,
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
    fig, ax = plt.subplots(figsize=(10.1, 2.85))
    ax.set_axis_off()

    stages = [
        ("Prediction", "test MSE\nsignal scale"),
        ("Full KAN", "pair reliance\nANOVA rank"),
        ("Readout", "endpoint rank\nmargin"),
        ("Refit", "pair scorer\non support"),
        ("Prune / symbolic", "retained vars\nformula status"),
        ("Claim", "support + pair\nwith provenance"),
    ]
    xs = np.linspace(0.07, 0.93, len(stages))
    y = 0.63
    box_w, box_h = 0.145, 0.37
    colors = [
        COLORS["prediction"],
        COLORS["full"],
        COLORS["readout"],
        COLORS["refit"],
        COLORS["extract"],
        COLORS["claim"],
    ]

    for idx, ((title, body), x, c) in enumerate(zip(stages, xs, colors)):
        ax.add_patch(
            plt.Rectangle(
                (x - box_w / 2, y - box_h / 2),
                box_w,
                box_h,
                facecolor=c,
                edgecolor="#333333",
                linewidth=1.0,
                transform=ax.transAxes,
            )
        )
        ax.text(x, y + 0.075, title, ha="center", va="center", fontsize=9.0, weight="bold", transform=ax.transAxes)
        ax.text(x, y - 0.060, body, ha="center", va="center", fontsize=7.6, transform=ax.transAxes)
        if idx < len(stages) - 1:
            ax.annotate(
                "",
                xy=(xs[idx + 1] - box_w / 2 - 0.008, y),
                xytext=(x + box_w / 2 + 0.008, y),
                arrowprops=dict(arrowstyle="->", lw=1.2, color="#333333"),
                xycoords=ax.transAxes,
                textcoords=ax.transAxes,
            )

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
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=col, linewidth=0.9),
            transform=ax.transAxes,
        )

    ax.text(
        0.5,
        0.94,
        "A structural claim is tied to the workflow object that produced it",
        ha="center",
        va="center",
        fontsize=10.2,
        weight="bold",
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

    fig, ax = plt.subplots(figsize=(8.6, 3.55))
    im = ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(3))
    ax.set_xticklabels(["Full KAN\npair rank-1", "Exposed readout\nendpoints@4", "Prune-input\nendpoints"], fontsize=9)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[1] for r in rows], fontsize=9)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            color = "white" if mat[i, j] < 0.35 else "#202020"
            ax.text(j, i, counts[i, j], ha="center", va="center", fontsize=9, weight="bold", color=color)

    ax.set_title("Same structural claim, different evidence objects", fontsize=10.2, weight="bold", pad=8)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label("success rate", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
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

    fig, ax = plt.subplots(figsize=(6.75, 4.2))

    # Regions follow the classification used in make_stage_discordance_summary.
    ax.axvspan(-0.03, 0.5, ymin=0.8 / 1.06, ymax=1, color="#f8dfdf", alpha=0.55, zorder=0)
    ax.axvspan(0.8, 1.03, ymin=0, ymax=0.5 / 1.06, color="#e3e7fb", alpha=0.55, zorder=0)
    ax.axvspan(0.8, 1.03, ymin=0.8 / 1.06, ymax=1, color="#def2df", alpha=0.65, zorder=0)
    ax.axvspan(-0.03, 0.5, ymin=0, ymax=0.5 / 1.06, color="#eeeeee", alpha=0.85, zorder=0)

    for cond, group in df.groupby("condition"):
        idx = group.index.to_numpy()
        ax.scatter(
            x[df.index.get_indexer(idx)],
            y[df.index.get_indexer(idx)],
            s=34,
            color=colors.get(cond, "#666666"),
            alpha=0.78,
            edgecolor="white",
            linewidth=0.45,
            label=labels.get(cond, cond),
        )

    ax.axvline(0.5, color="#555555", lw=0.9, ls="--")
    ax.axhline(0.5, color="#555555", lw=0.9, ls="--")
    ax.axvline(0.8, color="#777777", lw=0.8, ls=":")
    ax.axhline(0.8, color="#777777", lw=0.8, ls=":")

    counts = df["discordance_label"].value_counts()
    ax.text(0.15, 0.94, f"surfacing without reliance\n{counts.get('surfacing without reliance', 0)} records",
            ha="center", va="center", fontsize=7.6, color="#8a1f1f")
    ax.text(0.90, 0.94, f"aligned high\n{counts.get('aligned high', 0)}",
            ha="center", va="center", fontsize=7.6, color="#216b2b")
    ax.text(0.18, 0.14, f"aligned low\n{counts.get('aligned low', 0)}",
            ha="center", va="center", fontsize=7.6, color="#444444")
    ax.text(0.55, 0.55, f"mixed boundary\n{counts.get('mixed boundary', 0)}",
            ha="center", va="center", fontsize=7.6, color="#5a4b8a",
            bbox=dict(facecolor="white", edgecolor="#b8b8b8", boxstyle="round,pad=0.25", alpha=0.85))

    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("Full-KAN pair reliance: true pair rank-1 rate")
    ax.set_ylabel("Exposed-readout endpoint surfacing: endpoints@4")
    ax.set_title("Stage-discordance phase diagram", fontsize=10.2, weight="bold")
    ax.grid(alpha=0.16)
    ax.legend(frameon=False, loc="lower right", fontsize=8.2)
    fig.tight_layout()
    savefig("stage_discordance_phase_diagram")


def breadth_summary() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(9.8, 2.95), gridspec_kw={"width_ratios": [1.25, 0.9, 1.2]})

    # Semi-synthetic real-covariate geometry at noise 0.05.
    ax = axes[0]
    datasets = ["breast\ncancer", "diabetes", "wine"]
    kan = np.array([90 / 90, 83 / 90, 64 / 90])
    residual = np.array([0 / 90, 85 / 90, 61 / 90])
    x = np.arange(len(datasets))
    w = 0.34
    ax.bar(x - w / 2, kan, width=w, color=COLORS["blue"], label="KAN-FE endpoints")
    ax.bar(x + w / 2, residual, width=w, color=COLORS["orange"], label="Residual top-1")
    ax.set_title("(a) Real covariates", loc="left", fontsize=9.2, weight="bold")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=8)
    ax.set_ylabel("recovery rate")
    ax.legend(fontsize=7.5, frameon=False, loc="lower left")
    ax.grid(axis="y", alpha=0.25)

    # Mini-suite summary.
    ax = axes[1]
    labels = ["True\nsupport", "RF\nsupport"]
    vals = [0.84, 0.67]
    ax.bar(labels, vals, color=[COLORS["green"], "#B279C2"], width=0.55)
    ax.set_title("(b) Formula suite", loc="left", fontsize=9.2, weight="bold")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("mean pair recovery")
    for idx, v in enumerate(vals):
        ax.text(idx, v + 0.035, f"{v:.2f}", ha="center", fontsize=9)
    ax.grid(axis="y", alpha=0.25)

    # NID-style neural pair-score controls.
    ax = axes[2]
    labels = ["Weak\ncentered", "Strong\ncentered", "Bilinear", "Log\nproduct", "Exp\nproduct"]
    vals = [0.00, 0.995, 0.99, 0.97, 1.00]
    colors = [COLORS["red"], COLORS["green"], COLORS["green"], COLORS["green"], COLORS["green"]]
    ax.bar(np.arange(len(labels)), vals, color=colors, width=0.62)
    ax.set_title("(c) NID-style controls", loc="left", fontsize=9.2, weight="bold")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("pair F1")
    ax.grid(axis="y", alpha=0.25)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Breadth checks: covariates, formulas, and pair-score controls", fontsize=10.4, weight="bold", y=1.03)
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
