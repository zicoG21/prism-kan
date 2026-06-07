#!/usr/bin/env python3
"""Plot family-level overclaim signatures from official ClaimTransfer rows."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

EDGE_ORDER = [
    "prediction_to_pair",
    "support_to_prediction",
    "candidate_to_pair",
    "symbolic_status_to_expression_quality",
    "fitted_pair_to_readout",
    "fitted_pair_to_pruning",
]

EDGE_LABELS = {
    "prediction_to_pair": "Pred. -> pair",
    "support_to_prediction": "Support -> pred.",
    "candidate_to_pair": "Candidate -> pair",
    "symbolic_status_to_expression_quality": "Symbolic -> expr.",
    "fitted_pair_to_readout": "Fitted pair -> readout",
    "fitted_pair_to_pruning": "Fitted pair -> pruning",
}

FAMILY_ORDER = [
    "pyKAN",
    "neural_blackbox",
    "symbolic_library",
    "tree_gate",
    "sparse_lasso",
    "sparse_spline_lasso",
    "symbolic_lasso",
    "sparse_library",
    "support_screen",
    "ga2m_spline",
    "gbm_hstat",
    "epim_pairverify",
]

FAMILY_LABELS = {
    "pyKAN": "pyKAN",
    "neural_blackbox": "MLP-Hessian",
    "symbolic_library": "Symbolic/PySR",
    "tree_gate": "Tree gates",
    "sparse_lasso": "Sparse Lasso",
    "sparse_spline_lasso": "Spline Lasso",
    "symbolic_lasso": "Symbolic Lasso",
    "sparse_library": "Sparse library",
    "support_screen": "Support screen",
    "ga2m_spline": "GA2M",
    "gbm_hstat": "GBM-H",
    "epim_pairverify": "EPIM verifier",
}


def weighted_family_signature(df: pd.DataFrame) -> pd.DataFrame:
    block = df.copy()
    block["family"] = block["adapter_family"].astype(str)
    block = block[block["transfer_id"].isin(EDGE_ORDER)]
    rows: list[dict[str, object]] = []
    for (family, transfer_id), group in block.groupby(["family", "transfer_id"], dropna=False):
        source = group["source_passes"].sum()
        failures = group["target_failures_given_source_pass"].sum()
        if source <= 0:
            risk = np.nan
        else:
            risk = failures / source
        rows.append(
            {
                "adapter_family_display": family,
                "transfer_id": transfer_id,
                "source_passes": int(source),
                "target_failures_given_source_pass": int(failures),
                "overclaim_risk": risk,
            }
        )
    out = pd.DataFrame(rows)
    out["adapter_family_label"] = out["adapter_family_display"].map(FAMILY_LABELS).fillna(out["adapter_family_display"])
    return out


def main() -> None:
    by_adapter_path = ROOT / "score_reports" / "overclaim_risk_by_adapter.csv"
    if not by_adapter_path.exists():
        raise SystemExit(f"Missing {by_adapter_path}. Run scripts/build_overclaim_risk_report.py first.")
    by_adapter = pd.read_csv(by_adapter_path)
    family = weighted_family_signature(by_adapter)

    out_csv = ROOT / "score_reports" / "overclaim_signature_by_family.csv"
    out_md = out_csv.with_suffix(".md")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    family.to_csv(out_csv, index=False)

    matrix = family.pivot_table(
        index="adapter_family_display",
        columns="transfer_id",
        values="overclaim_risk",
        aggfunc="mean",
    )
    row_order = [row for row in FAMILY_ORDER if row in matrix.index]
    row_order.extend([row for row in matrix.index if row not in row_order])
    matrix = matrix.reindex(row_order)
    matrix = matrix.reindex(columns=EDGE_ORDER)

    source_matrix = family.pivot_table(
        index="adapter_family_display",
        columns="transfer_id",
        values="source_passes",
        aggfunc="sum",
    ).reindex(index=row_order, columns=EDGE_ORDER)

    display = matrix.copy()
    display.index = [FAMILY_LABELS.get(idx, idx) for idx in display.index]
    display.columns = [EDGE_LABELS[col] for col in display.columns]

    md = display.copy()
    for col in md.columns:
        md[col] = md[col].map(lambda x: "" if pd.isna(x) else f"{100*x:.1f}%")
    out_md.write_text(
        "# Family-Level Overclaim Signature\n\n"
        "Cells are weighted overclaim risk within an adapter family. Blank cells mean the family does not expose that transfer edge.\n\n"
        + md.to_markdown()
        + "\n",
        encoding="utf-8",
    )

    values = display.to_numpy(dtype=float)
    masked = np.ma.masked_invalid(values)
    cmap = plt.cm.OrRd.copy()
    cmap.set_bad("#f1f5f9")

    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    im = ax.imshow(masked, vmin=0, vmax=1, cmap=cmap, aspect="auto")
    ax.set_xticks(np.arange(display.shape[1]), display.columns, rotation=28, ha="right")
    ax.set_yticks(np.arange(display.shape[0]), display.index)
    ax.tick_params(axis="both", labelsize=8.8)

    for i in range(display.shape[0]):
        for j in range(display.shape[1]):
            val = values[i, j]
            if np.isnan(val):
                ax.text(j, i, "--", ha="center", va="center", fontsize=8, color="#64748b")
                continue
            source = source_matrix.iloc[i, j]
            text_color = "white" if val >= 0.55 else "#111827"
            label = f"{100*val:.0f}%\n(n={int(source)})" if not pd.isna(source) else f"{100*val:.0f}%"
            ax.text(j, i, label, ha="center", va="center", fontsize=7.6, color=text_color)

    ax.set_title("Method-family overclaim signatures", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Claim-transfer edge")
    ax.set_ylabel("Workflow family")
    cbar = fig.colorbar(im, ax=ax, fraction=0.028, pad=0.018)
    cbar.set_label("Overclaim risk", rotation=90)
    fig.tight_layout()

    out_pdf = ROOT / "manuscripts" / "foundation_benchmark_dev" / "figures" / "overclaim_signature_heatmap.pdf"
    out_png = out_pdf.with_suffix(".png")
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_md}")
    print(f"Wrote {out_pdf}")
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
