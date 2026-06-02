#!/usr/bin/env python3
"""Draw the ClaimTransfer-Bench contract pipeline figure.

This is a static specification figure: it shows the benchmark input/output
contract rather than an experiment result.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import matplotlib.patheffects as pe


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "manuscripts" / "workshop_foundation" / "figures"

COLORS = {
    "ink": "#1F2937",
    "muted": "#64748B",
    "card": "#E8F1F2",
    "adapter": "#E9E3F5",
    "record": "#F5E9D6",
    "report": "#E2EEDC",
    "edge": "#334155",
}


def add_box(ax, x, y, w, h, title, fields, color):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.025",
        facecolor=color,
        edgecolor=COLORS["edge"],
        linewidth=0.9,
        transform=ax.transAxes,
    )
    patch.set_path_effects([pe.SimplePatchShadow(offset=(1.0, -1.0), alpha=0.12), pe.Normal()])
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h - 0.052,
        title,
        ha="center",
        va="center",
        fontsize=9.2,
        weight="bold",
        color=COLORS["ink"],
        transform=ax.transAxes,
    )
    ax.text(
        x + 0.022,
        y + h - 0.105,
        fields,
        ha="left",
        va="top",
        fontsize=7.45,
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

    fig, ax = plt.subplots(figsize=(10.4, 3.1))
    ax.set_axis_off()

    boxes = [
        (
            0.035,
            "Task card",
            "formula / covariates\nsupport labels\nlegal claims\nofficial scorers\nseeds",
            COLORS["card"],
        ),
        (
            0.278,
            "Workflow adapter",
            "prediction output\nselected support\npair scores\nreadout fields\nsymbolic output",
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
            "by task card\nx claim type\nx evidence object\nCI / quantiles\nmissing fields explicit",
            COLORS["report"],
        ),
    ]
    w, h, y = 0.19, 0.54, 0.31
    for x, title, fields, color in boxes:
        add_box(ax, x, y, w, h, title, fields, color)

    for i in range(len(boxes) - 1):
        x0 = boxes[i][0] + w + 0.012
        x1 = boxes[i + 1][0] - 0.012
        arrow = FancyArrowPatch(
            (x0, y + h / 2),
            (x1, y + h / 2),
            arrowstyle="-|>",
            mutation_scale=12,
            lw=1.15,
            color=COLORS["edge"],
            transform=ax.transAxes,
        )
        ax.add_patch(arrow)

    ax.text(
        0.5,
        0.94,
        "ClaimTransfer-Bench input-output contract",
        ha="center",
        va="center",
        fontsize=11.2,
        weight="bold",
        color=COLORS["ink"],
        transform=ax.transAxes,
    )
    ax.text(
        0.5,
        0.885,
        "A method is evaluated by the structural claims its native workflow objects can support.",
        ha="center",
        va="center",
        fontsize=8.0,
        color=COLORS["muted"],
        transform=ax.transAxes,
    )

    ax.text(
        0.5,
        0.13,
        "Ordinary metrics report a number; the benchmark also records where the evidence came from and which scorer/predicate made it valid.",
        ha="center",
        va="center",
        fontsize=7.8,
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
