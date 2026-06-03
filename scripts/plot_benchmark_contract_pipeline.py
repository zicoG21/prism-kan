#!/usr/bin/env python3
"""Draw the ClaimTransfer-Bench contract pipeline figure.

This is a static specification figure: it shows the benchmark input/output
contract rather than an experiment result.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "manuscripts" / "workshop_foundation" / "figures"

COLORS = {
    "ink": "#1F2937",
    "muted": "#64748B",
    "card": "#EAF3F4",
    "adapter": "#EEE9F7",
    "record": "#F7ECD8",
    "report": "#E7F1DF",
    "edge": "#526173",
    "accent": "#0F4C81",
}


def add_box(ax, x, y, w, h, title, fields, color):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.010,rounding_size=0.018",
        facecolor=color,
        edgecolor=COLORS["edge"],
        linewidth=0.75,
        transform=ax.transAxes,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h - 0.041,
        title,
        ha="center",
        va="center",
        fontsize=8.4,
        weight="bold",
        color=COLORS["ink"],
        transform=ax.transAxes,
    )
    ax.text(
        x + 0.022,
        y + h - 0.084,
        fields,
        ha="left",
        va="top",
        fontsize=6.9,
        color="#374151",
        linespacing=1.24,
        transform=ax.transAxes,
    )


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )

    fig, ax = plt.subplots(figsize=(9.2, 2.35))
    ax.set_axis_off()

    boxes = [
        (
            0.035,
            "Task card",
            "formula, covariates\nsupport labels\nlegal claims\nofficial scorers",
            COLORS["card"],
        ),
        (
            0.278,
            "Workflow adapter",
            "prediction output\nselected support\npair scores\nreadout / symbolic fields",
            COLORS["adapter"],
        ),
        (
            0.522,
            "claim_record.csv",
            "task_id, adapter, seed\nevidence_object\nclaim_type, target\nscorer, rank, margin\npredicate, pass",
            COLORS["record"],
        ),
        (
            0.765,
            "Score report",
            "by task card\nx claim type\nx evidence object\nCIs and quantiles\nmissing fields explicit",
            COLORS["report"],
        ),
    ]
    w, h, y = 0.19, 0.50, 0.29
    for x, title, fields, color in boxes:
        add_box(ax, x, y, w, h, title, fields, color)

    for i in range(len(boxes) - 1):
        x0 = boxes[i][0] + w + 0.012
        x1 = boxes[i + 1][0] - 0.012
        arrow = FancyArrowPatch(
            (x0, y + h / 2),
            (x1, y + h / 2),
            arrowstyle="-|>",
            mutation_scale=10,
            lw=1.0,
            color=COLORS["edge"],
            transform=ax.transAxes,
        )
        ax.add_patch(arrow)

    ax.text(
        0.035,
        0.925,
        "ClaimTransfer-Bench contract",
        ha="left",
        va="center",
        fontsize=10.0,
        weight="bold",
        color=COLORS["ink"],
        transform=ax.transAxes,
    )
    ax.text(
        0.035,
        0.845,
        "Evaluation unit: task card x workflow adapter x evidence object x typed structural claim.",
        ha="left",
        va="center",
        fontsize=7.15,
        color=COLORS["muted"],
        transform=ax.transAxes,
    )

    ax.text(
        0.035,
        0.115,
        "The score report is not one leaderboard scalar: it preserves provenance before aggregating success rates.",
        ha="left",
        va="center",
        fontsize=7.0,
        color=COLORS["muted"],
        transform=ax.transAxes,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        out = OUT_DIR / f"benchmark_contract_pipeline.{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=300)
        print(f"Wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
