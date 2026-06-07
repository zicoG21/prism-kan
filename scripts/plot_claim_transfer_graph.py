#!/usr/bin/env python3
"""Plot the ClaimTransfer overclaim-risk graph for the paper."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


POSITIONS = {
    "support": (0.05, 0.72),
    "prediction": (0.31, 0.72),
    "candidate": (0.05, 0.28),
    "fitted pair": (0.31, 0.28),
    "symbolic status": (0.05, 0.50),
    "pair": (0.62, 0.72),
    "readout": (0.62, 0.42),
    "pruning": (0.62, 0.18),
    "expression quality": (0.92, 0.50),
}


NODE_LABELS = {
    "support": "Support",
    "prediction": "Prediction",
    "candidate": "Candidate",
    "fitted pair": "Fitted\npair",
    "symbolic status": "Symbolic\nstatus",
    "pair": "Verified\npair",
    "readout": "Readout\nendpoints",
    "pruning": "Pruning\nendpoints",
    "expression quality": "Expression\nquality",
}


def main() -> None:
    path = ROOT / "score_reports" / "claim_transfer_graph_edges.csv"
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run scripts/build_overclaim_signature.py first.")
    df = pd.read_csv(path)

    fig, ax = plt.subplots(figsize=(9.2, 4.2))
    ax.set_xlim(-0.04, 1.02)
    ax.set_ylim(0.04, 0.90)
    ax.axis("off")

    for node, (x, y) in POSITIONS.items():
        face = "#f7fbff" if node not in {"pair", "readout", "pruning", "expression quality"} else "#fff7ed"
        edge = "#334155"
        ax.text(
            x,
            y,
            NODE_LABELS[node],
            ha="center",
            va="center",
            fontsize=9.5,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.35,rounding_size=0.08", fc=face, ec=edge, lw=0.9),
        )

    for _, row in df.iterrows():
        source = str(row["source_node"])
        target = str(row["target_node"])
        risk = float(row["overclaim_risk"])
        x0, y0 = POSITIONS[source]
        x1, y1 = POSITIONS[target]
        width = 1.0 + 4.2 * risk
        color = "#b91c1c" if risk >= 0.5 else "#c2410c" if risk >= 0.25 else "#475569"
        ax.annotate(
            "",
            xy=(x1 - 0.045 if x1 > x0 else x1 + 0.045, y1),
            xytext=(x0 + 0.055 if x1 > x0 else x0 - 0.055, y0),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=width, shrinkA=7, shrinkB=7),
        )
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        dy = 0.035 if abs(y0 - y1) < 0.08 else 0.0
        ax.text(
            mx,
            my + dy,
            f"{risk:.1%}",
            ha="center",
            va="center",
            fontsize=9,
            color=color,
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.88),
        )

    ax.text(
        0.5,
        0.86,
        "Directed edges are unsupported claim transfers; edge labels are conditional overclaim risk.",
        ha="center",
        va="center",
        fontsize=10,
        color="#334155",
    )

    out = ROOT / "manuscripts" / "foundation_benchmark_dev" / "figures" / "claim_transfer_graph.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
