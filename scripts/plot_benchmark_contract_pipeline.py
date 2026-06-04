#!/usr/bin/env python3
"""Draw the ClaimTransfer audit-contract pipeline figure.

This is a static protocol figure: it shows the audit input/output contract
rather than an experiment result.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "manuscripts" / "submission_neurips_workshop_v0" / "figures"

COLORS = {
    "ink": "#1F2937",
    "muted": "#64748B",
    "card": "#EAF3F4",
    "adapter": "#EEE9F7",
    "record": "#F7ECD8",
    "report": "#E7F1DF",
    "edge": "#526173",
    "accent": "#0F4C81",
    "band": "#F8FAFC",
    "line": "#CBD5E1",
}


def add_box(ax, x, y, w, h, title, fields, color, tag):
    tag_patch = FancyBboxPatch(
        (x + 0.015, y + h + 0.042),
        0.066,
        0.040,
        boxstyle="round,pad=0.006,rounding_size=0.020",
        facecolor="white",
        edgecolor=COLORS["line"],
        linewidth=0.55,
        transform=ax.transAxes,
    )
    ax.add_patch(tag_patch)
    ax.text(
        x + 0.048,
        y + h + 0.062,
        tag,
        ha="center",
        va="center",
        fontsize=5.25,
        color=COLORS["muted"],
        transform=ax.transAxes,
    )

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
        x + 0.018,
        y + h - 0.048,
        title,
        ha="left",
        va="center",
        fontsize=7.65,
        weight="bold",
        color=COLORS["ink"],
        transform=ax.transAxes,
    )
    ax.text(
        x + 0.022,
        y + h - 0.095,
        fields,
        ha="left",
        va="top",
        fontsize=5.45,
        color="#374151",
        linespacing=1.16,
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

    fig, ax = plt.subplots(figsize=(9.0, 2.08))
    ax.set_axis_off()

    band = FancyBboxPatch(
        (0.025, 0.090),
        0.95,
        0.710,
        boxstyle="round,pad=0.010,rounding_size=0.030",
        facecolor=COLORS["band"],
        edgecolor="#E2E8F0",
        linewidth=0.65,
        transform=ax.transAxes,
    )
    ax.add_patch(band)

    boxes = [
        (
            0.035,
            "Task card",
            "formula, covariates\nsupport + claims\nofficial scorers",
            COLORS["card"],
            "fixed",
        ),
        (
            0.278,
            "Workflow adapter",
            "prediction output\nselected support\npair/readout fields\nsymbolic fields",
            COLORS["adapter"],
            "method",
        ),
        (
            0.522,
            "claim_record.csv",
            "official rows\ntask_id, adapter, seed\nevidence object\nrank, margin\npredicate, pass",
            COLORS["record"],
            "scored",
        ),
        (
            0.765,
            "Score report",
            "by task card\nclaim type\nevidence object\nCI / quantiles\nmissing explicit",
            COLORS["report"],
            "score",
        ),
    ]
    w, h, y = 0.19, 0.405, 0.310
    for x, title, fields, color, tag in boxes:
        add_box(ax, x, y, w, h, title, fields, color, tag)

    for i in range(len(boxes) - 1):
        x0 = boxes[i][0] + w + 0.012
        x1 = boxes[i + 1][0] - 0.012
        arrow = FancyArrowPatch(
            (x0, y + h / 2 + 0.002),
            (x1, y + h / 2 + 0.002),
            arrowstyle="-|>",
            mutation_scale=9.5,
            lw=0.95,
            color=COLORS["edge"],
            transform=ax.transAxes,
        )
        ax.add_patch(arrow)

    ax.text(
        0.035,
        0.940,
        "ClaimTransfer audit contract",
        ha="left",
        va="center",
        fontsize=9.8,
        weight="bold",
        color=COLORS["ink"],
        transform=ax.transAxes,
    )
    ax.text(
        0.035,
        0.865,
        "Audit unit: task card x workflow adapter x evidence object x typed structural claim.",
        ha="left",
        va="center",
        fontsize=7.15,
        color=COLORS["muted"],
        transform=ax.transAxes,
    )

    ribbon = FancyBboxPatch(
        (0.055, 0.135),
        0.89,
        0.082,
        boxstyle="round,pad=0.008,rounding_size=0.020",
        facecolor="white",
        edgecolor=COLORS["line"],
        linewidth=0.55,
        transform=ax.transAxes,
    )
    ax.add_patch(ribbon)
    ax.text(
        0.5,
        0.176,
        "Provenance is preserved before aggregation: no row becomes an untyped model-level success sentence.",
        ha="center",
        va="center",
        fontsize=6.35,
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
