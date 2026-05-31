#!/usr/bin/env python3
"""Plot claim-decision cases as a compact evidence-object decision strip."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DECISION_COLORS = {
    "accept controlled recovery": "#009E73",
    "revise claim to endpoint surfacing": "#E69F00",
    "partial support with fragile extraction": "#56B4E9",
    "reject structure claim": "#D55E00",
    "boundary case": "#CC79A7",
    "boundary or reject": "#999999",
}


def parse_rate(text: object) -> float:
    s = str(text)
    if "(" in s and ")" in s:
        return float(s.split("(")[-1].split(")")[0])
    return float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="local_notes/generated/claim_decision_cases_20260531.csv",
    )
    parser.add_argument(
        "--out",
        default="manuscripts/workshop_case_study/figures/claim_decision_cases",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    stage_cols = [
        ("full_pair_rank1", "Full KAN\npair"),
        ("readout_endpoints_at4", "Readout\nendpoints"),
        ("prune_endpoints", "Prune\nendpoints"),
    ]
    values = np.array([[parse_rate(row[c]) for c, _ in stage_cols] for _, row in df.iterrows()])

    fig_h = max(3.6, 0.46 * len(df) + 1.2)
    fig, ax = plt.subplots(figsize=(8.8, fig_h))
    im = ax.imshow(values, vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")

    ax.set_xticks(range(len(stage_cols)))
    ax.set_xticklabels([label for _, label in stage_cols], fontsize=9)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["condition"], fontsize=8)
    ax.tick_params(length=0)

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            val = values[i, j]
            color = "white" if val > 0.62 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color=color)

    for i, row in df.iterrows():
        decision = row["stage_record_decision"]
        color = DECISION_COLORS.get(decision, "#666666")
        ax.text(
            len(stage_cols) + 0.15,
            i,
            decision.replace(" claim to ", " to\n").replace(" controlled ", "\n"),
            va="center",
            ha="left",
            fontsize=8,
            color=color,
            fontweight="bold",
        )

    ax.set_xlim(-0.5, len(stage_cols) + 2.8)
    ax.set_title("Stage-record claim decisions from matched evidence objects", fontsize=11, pad=10)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(stage_cols), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(df), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("success rate", fontsize=8)
    cbar.ax.tick_params(labelsize=8)

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=220, bbox_inches="tight")
    print(f"Wrote {out.with_suffix('.pdf')}")
    print(f"Wrote {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
