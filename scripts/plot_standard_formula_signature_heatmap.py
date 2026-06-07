#!/usr/bin/env python3
"""Plot standard-formula-only method overclaim signatures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

EDGE_ORDER = [
    "prediction_to_pair",
    "support_to_prediction",
    "symbolic_status_to_expression_quality",
]

EDGE_LABELS = {
    "prediction_to_pair": "Pred. -> pair",
    "support_to_prediction": "Support -> pred.",
    "symbolic_status_to_expression_quality": "Symbolic -> expr.",
}

METHOD_LABELS = {
    "pysr_symbolic_regressor": "PySR",
    "gplearn_symbolic_regressor": "gplearn",
    "mlp_hessian": "MLP-Hessian",
    "poly2_ridge": "Sparse poly",
    "main_effect_corr": "Support screen",
    "pair_corr_screen": "Pair corr.",
    "oracle_symbolic": "Oracle symbolic",
}

METHOD_ORDER = [
    "pysr_symbolic_regressor",
    "gplearn_symbolic_regressor",
    "mlp_hessian",
    "poly2_ridge",
    "main_effect_corr",
    "pair_corr_screen",
    "oracle_symbolic",
]


def main() -> None:
    src = ROOT / "score_reports" / "standard_formula_overclaim_signature_by_method.csv"
    if not src.exists():
        raise SystemExit(f"Missing {src}. Run scripts/build_full_benchmark_analysis_reports.py first.")
    df = pd.read_csv(src)
    if df.empty:
        raise SystemExit(f"{src} is empty")
    df = df.set_index("adapter")
    row_order = [row for row in METHOD_ORDER if row in df.index]
    row_order.extend([row for row in df.index if row not in row_order])
    matrix = df.reindex(row_order)
    source_cols = [f"{col}_source_passes" for col in EDGE_ORDER]
    values = matrix.reindex(columns=EDGE_ORDER).to_numpy(dtype=float)
    sources = matrix.reindex(columns=source_cols).to_numpy(dtype=float)
    row_labels = [METHOD_LABELS.get(row, row) for row in matrix.index]
    col_labels = [EDGE_LABELS[col] for col in EDGE_ORDER]

    masked = np.ma.masked_invalid(values)
    cmap = plt.cm.OrRd.copy()
    cmap.set_bad("#f1f5f9")

    fig, ax = plt.subplots(figsize=(7.4, 3.25))
    im = ax.imshow(masked, vmin=0, vmax=1, cmap=cmap, aspect="auto")
    ax.set_xticks(np.arange(len(col_labels)), col_labels, rotation=24, ha="right")
    ax.set_yticks(np.arange(len(row_labels)), row_labels)
    ax.tick_params(axis="both", labelsize=8.6)

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            val = values[i, j]
            if np.isnan(val):
                ax.text(j, i, "--", ha="center", va="center", fontsize=8, color="#64748b")
                continue
            src_count = sources[i, j]
            color = "white" if val >= 0.55 else "#111827"
            label = f"{100 * val:.0f}%"
            if not np.isnan(src_count):
                label += f"\n(n={int(src_count)})"
            ax.text(j, i, label, ha="center", va="center", fontsize=7.4, color=color)

    ax.set_title("Standard-formula overclaim signatures", fontsize=12, fontweight="bold", pad=9)
    ax.set_xlabel("Claim-transfer edge")
    ax.set_ylabel("Standard-formula adapter")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Overclaim risk")
    fig.tight_layout()

    out_pdf = ROOT / "manuscripts" / "foundation_benchmark_dev" / "figures" / "standard_formula_signature_heatmap.pdf"
    out_png = out_pdf.with_suffix(".png")
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    print(f"Wrote {out_pdf}")
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
