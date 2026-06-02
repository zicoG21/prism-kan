#!/usr/bin/env python3
"""Plot a compact cross-method evidence-object success matrix.

The transfer-link heatmap is useful for diagnostics, but too dense for a main
paper figure.  This plot shows the same benchmark in a simpler form: for each
formula/method row, which evidence objects succeed?
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from experiments.paper_plot_style import OKABE_ITO, configure_paper_plots
except Exception:  # pragma: no cover
    from paper_plot_style import OKABE_ITO, configure_paper_plots


METRICS = [
    ("Prediction", "prediction_success_mean"),
    ("Support", "support_success_all_true_mean"),
    ("Endpoints", "endpoint_success_mean"),
    ("Pair", "pair_success_all_true_at_budget_mean"),
]

FUNCTION_LABELS = {
    "core_interaction_c025": "weak centered",
    "formula_bilinear": "bilinear",
    "formula_division_mixed": "division mixed",
    "formula_mixed_sparse": "mixed sparse",
}

FUNCTION_ORDER = [
    "core_interaction_c025",
    "formula_bilinear",
    "formula_division_mixed",
    "formula_mixed_sparse",
]

METHOD_ORDER = [
    "ga2m_spline",
    "gbm_hstat",
    "sparse_lasso",
    "sparse_spline_lasso",
    "symbolic_lasso",
]

METHOD_LABELS = {
    "ga2m_spline": "GA2M-style",
    "gbm_hstat": "GBM-H",
    "sparse_lasso": "Sparse Lasso",
    "sparse_spline_lasso": "Spline Lasso",
    "symbolic_lasso": "Symbolic library",
}


def load_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Keep the main GL rows and drop d=40 smoke rows.
    if "samples" in df.columns:
        df = df[pd.to_numeric(df["samples"], errors="coerce").eq(1024)].copy()
    if "dimension" in df.columns:
        df = df[pd.to_numeric(df["dimension"], errors="coerce").eq(100)].copy()
    df = df[df["function"].isin(FUNCTION_ORDER) & df["method"].isin(METHOD_ORDER)].copy()
    for _, col in METRICS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["num_runs"] = pd.to_numeric(df["num_runs"], errors="coerce").fillna(0).astype(int)
    df["function_order"] = df["function"].map({f: i for i, f in enumerate(FUNCTION_ORDER)})
    df["method_order"] = df["method"].map({m: i for i, m in enumerate(METHOD_ORDER)})
    df = df.sort_values(["function_order", "method_order"], kind="stable")
    return df


def write_compact(df: pd.DataFrame, path: Path) -> None:
    rows = []
    for _, row in df.iterrows():
        out = {
            "function": row["function"],
            "method": row["method"],
            "num_runs": int(row["num_runs"]),
        }
        for label, col in METRICS:
            out[label.lower()] = float(row[col]) if pd.notna(row[col]) else np.nan
        rows.append(out)
    pd.DataFrame(rows).to_csv(path, index=False)


def plot(df: pd.DataFrame, out_base: Path) -> None:
    configure_paper_plots(usetex=False)
    data = df[[col for _, col in METRICS]].to_numpy(dtype=float)
    row_labels = [
        f"{FUNCTION_LABELS.get(str(f), str(f))} / {METHOD_LABELS.get(str(m), str(m))}"
        for f, m in zip(df["function"], df["method"])
    ]
    function_labels = [FUNCTION_LABELS.get(str(f), str(f)) for f in df["function"]]

    fig_h = max(3.8, 0.255 * len(df) + 0.9)
    fig, ax = plt.subplots(figsize=(6.3, fig_h))
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad("#F3F4F6")
    im = ax.imshow(np.ma.masked_invalid(data), vmin=0.0, vmax=1.0, cmap=cmap, aspect="auto")

    ax.set_xticks(np.arange(len(METRICS)))
    ax.set_xticklabels([label for label, _ in METRICS], rotation=28, ha="right")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title("Structural evidence is object-specific across workflows", loc="left", pad=7)

    # Function group separators.
    for i in range(len(df)):
        if i > 0 and function_labels[i] != function_labels[i - 1]:
            ax.axhline(i - 0.5, color="#111827", linewidth=0.55)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if np.isfinite(val):
                txt = f"{val:.2f}"
                color = "white" if val > 0.62 else "#111827"
                ax.text(j, i, txt, ha="center", va="center", fontsize=6.7, color=color)
            else:
                ax.text(j, i, "NA", ha="center", va="center", fontsize=6.2, color="#6B7280")

    ax.tick_params(axis="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlim(-0.5, len(METRICS) - 0.5)
    ax.set_ylabel("formula / workflow")
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.025)
    cbar.set_label("success rate")
    cbar.outline.set_linewidth(0.4)

    note = "Rows use the same structural labels; columns are different evidence objects."
    fig.text(0.08, 0.015, note, ha="left", va="bottom", fontsize=6.7, color="#4B5563")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".pdf"))
    fig.savefig(out_base.with_suffix(".png"), dpi=450)
    plt.close(fig)
    print(f"[saved] {out_base.with_suffix('.pdf')}")
    print(f"[saved] {out_base.with_suffix('.png')}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("local_notes/generated/cross_method_transfer_matrix_20260601_gl_method_summary.csv"),
    )
    parser.add_argument(
        "--out-base",
        type=Path,
        default=Path("local_notes/generated/cross_method_success_matrix_20260601"),
    )
    parser.add_argument(
        "--manuscript-out-base",
        type=Path,
        default=Path("manuscripts/workshop_case_study/figures/cross_method_success_matrix"),
    )
    args = parser.parse_args()
    df = load_summary(args.input)
    compact_path = args.out_base.with_name(args.out_base.name + ".csv")
    write_compact(df, compact_path)
    plot(df, args.out_base)
    plot(df, args.manuscript_out_base)
    print(f"[saved] {compact_path}")


if __name__ == "__main__":
    main()
