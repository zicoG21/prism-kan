#!/usr/bin/env python3
"""Plot claim-decision cases as a compact evidence-object decision strip."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch
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

RATE_CMAP = LinearSegmentedColormap.from_list(
    "claim_rate",
    ["#F7F7F7", "#F6E8A6", "#A8DDB5", "#2B8CBE", "#084081"],
)


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#222222",
            "axes.titlesize": 10,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "axes.axisbelow": True,
            "savefig.facecolor": "white",
        }
    )


def parse_rate(text: object) -> float:
    s = str(text)
    if "(" in s and ")" in s:
        return float(s.split("(")[-1].split(")")[0])
    return float("nan")


def display_condition(text: object) -> str:
    s = str(text)
    replacements = {
        "gridupdate": "grid update",
        "noise010": "noise .10",
        " n512": " n=512",
        " n768": " n=768",
        " n1024": " n=1024",
    }
    for old, new in replacements.items():
        s = s.replace(old, new)
    return s


def main() -> None:
    set_style()
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

    fig_h = max(3.55, 0.43 * len(df) + 1.18)
    fig, ax = plt.subplots(figsize=(8.75, fig_h))
    im = ax.imshow(values, vmin=0, vmax=1, cmap=RATE_CMAP, aspect="auto")

    ax.set_xticks(range(len(stage_cols)))
    ax.set_xticklabels([label for _, label in stage_cols], fontsize=8.8)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels([display_condition(v) for v in df["condition"]], fontsize=8.5)
    ax.tick_params(length=0)
    ax.text(
        1.0,
        -0.72,
        "queried evidence object",
        ha="center",
        va="center",
        fontsize=7.4,
        color="#555555",
    )

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            val = values[i, j]
            color = "white" if val > 0.72 else "#222222"
            count = str(df.iloc[i][stage_cols[j][0]]).split(" ")[0]
            ax.text(j, i - 0.08, count, ha="center", va="center", fontsize=8.2, color=color, fontweight="bold")
            ax.text(j, i + 0.16, f"{val:.2f}", ha="center", va="center", fontsize=6.9, color=color)

    for i, row in df.iterrows():
        decision = row["stage_record_decision"]
        color = DECISION_COLORS.get(decision, "#666666")
        label = (
            decision.replace("accept controlled recovery", "accept recovery")
            .replace("revise claim to endpoint surfacing", "revise to endpoint surfacing")
            .replace("reject structure claim", "reject structure claim")
        )
        pill = FancyBboxPatch(
            (len(stage_cols) + 0.08, i - 0.25),
            2.08,
            0.50,
            boxstyle="round,pad=0.03,rounding_size=0.08",
            facecolor="white",
            edgecolor=color,
            linewidth=1.0,
        )
        pill.set_path_effects([pe.SimplePatchShadow(offset=(0.8, -0.8), alpha=0.10), pe.Normal()])
        ax.add_patch(pill)
        ax.text(
            len(stage_cols) + 0.17,
            i,
            label,
            va="center",
            ha="left",
            fontsize=7.7,
            color=color,
            fontweight="bold",
        )

    ax.set_xlim(-0.5, len(stage_cols) + 2.9)
    ax.text(
        0.5,
        1.055,
        "Claim decisions from matched stage-record evidence",
        ha="center",
        va="bottom",
        transform=ax.transAxes,
        fontsize=10.2,
        fontweight="bold",
    )
    ax.text(
        len(stage_cols) + 0.14,
        -0.72,
        "stage-record decision",
        ha="left",
        va="center",
        fontsize=7.4,
        color="#555555",
    )

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(stage_cols), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(df), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.6)
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
