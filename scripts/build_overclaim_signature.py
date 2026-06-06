#!/usr/bin/env python3
"""Build method-level overclaim signatures and claim-transfer graph files."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def markdown_table(df: pd.DataFrame, max_rows: int = 80) -> str:
    if df.empty:
        return "No rows."
    table = df.head(max_rows).fillna("").astype(str)
    cols = list(table.columns)
    widths = [max(len(col), *(len(v) for v in table[col].tolist())) for col in cols]

    def row(vals: list[str]) -> str:
        return "| " + " | ".join(vals[i].ljust(widths[i]) for i in range(len(vals))) + " |"

    lines = [row(cols), "| " + " | ".join("-" * w for w in widths) + " |"]
    lines.extend(row([str(v) for v in values]) for values in table[cols].values.tolist())
    if len(df) > max_rows:
        lines.append(f"\nShowing first {max_rows} of {len(df)} rows.")
    return "\n".join(lines)


def build_signature(by_adapter: pd.DataFrame) -> pd.DataFrame:
    block = by_adapter.copy()
    block["method"] = block["adapter_family"].astype(str) + " / " + block["adapter"].astype(str)
    pivot = block.pivot_table(
        index=["adapter_family", "adapter", "method"],
        columns="transfer_id",
        values="overclaim_risk",
        aggfunc="mean",
    ).reset_index()
    count_pivot = block.pivot_table(
        index=["adapter_family", "adapter", "method"],
        columns="transfer_id",
        values="source_passes",
        aggfunc="sum",
    ).reset_index()
    for col in count_pivot.columns:
        if col not in {"adapter_family", "adapter", "method"}:
            count_pivot = count_pivot.rename(columns={col: f"{col}_source_passes"})
    out = pivot.merge(count_pivot, on=["adapter_family", "adapter", "method"], how="left")
    risk_cols = [c for c in out.columns if c not in {"adapter_family", "adapter", "method"} and not str(c).endswith("_source_passes")]
    out["dominant_overclaim_edge"] = out[risk_cols].idxmax(axis=1, skipna=True)
    out["dominant_overclaim_risk"] = out[risk_cols].max(axis=1, skipna=True)
    return out.sort_values(["dominant_overclaim_risk", "adapter_family", "adapter"], ascending=[False, True, True])


def build_graph(summary: pd.DataFrame) -> pd.DataFrame:
    edges = []
    for _, row in summary.iterrows():
        transfer_id = str(row["transfer_id"])
        if "_to_" in transfer_id:
            source, target = transfer_id.split("_to_", 1)
        else:
            source, target = str(row["transfer"]).split(" -> ", 1)
        edges.append(
            {
                "source_node": source.replace("_", " "),
                "target_node": target.replace("_", " "),
                "transfer_id": transfer_id,
                "transfer": row["transfer"],
                "source_passes": int(row["source_passes"]),
                "target_failures_given_source_pass": int(row["target_failures_given_source_pass"]),
                "overclaim_risk": float(row["overclaim_risk"]),
                "wilson_low": float(row["wilson_low"]),
                "wilson_high": float(row["wilson_high"]),
                "edge_label": f"{row['transfer']} ({float(row['overclaim_risk']):.1%})",
            }
        )
    return pd.DataFrame(edges).sort_values("overclaim_risk", ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", default="score_reports/overclaim_risk_report.csv")
    parser.add_argument("--by-adapter", default="score_reports/overclaim_risk_by_adapter.csv")
    parser.add_argument("--signature-out", default="score_reports/overclaim_signature_by_method.csv")
    parser.add_argument("--graph-out", default="score_reports/claim_transfer_graph_edges.csv")
    args = parser.parse_args()

    summary_path = ROOT / args.summary
    by_adapter_path = ROOT / args.by_adapter
    summary = pd.read_csv(summary_path)
    by_adapter = pd.read_csv(by_adapter_path)

    signature = build_signature(by_adapter)
    graph = build_graph(summary)

    signature_out = ROOT / args.signature_out
    graph_out = ROOT / args.graph_out
    signature_out.parent.mkdir(parents=True, exist_ok=True)
    graph_out.parent.mkdir(parents=True, exist_ok=True)
    signature.to_csv(signature_out, index=False)
    graph.to_csv(graph_out, index=False)

    signature_md = signature_out.with_suffix(".md")
    graph_md = graph_out.with_suffix(".md")
    show_sig = signature.copy()
    for col in show_sig.columns:
        if col.endswith("_risk") or col in set(summary["transfer_id"].astype(str)):
            show_sig[col] = pd.to_numeric(show_sig[col], errors="coerce").map(
                lambda x: "" if pd.isna(x) else f"{x:.3f}"
            )
    signature_md.write_text(
        "# Method-Level Overclaim Signature\n\n"
        "Rows are adapter methods; columns are claim-transfer edges.  Cells are "
        "conditional overclaim risk among rows where the source claim passes.\n\n"
        + markdown_table(show_sig)
        + "\n",
        encoding="utf-8",
    )
    show_graph = graph.copy()
    for col in ["overclaim_risk", "wilson_low", "wilson_high"]:
        show_graph[col] = show_graph[col].map(lambda x: f"{x:.3f}")
    graph_md.write_text(
        "# Claim-Transfer Graph Edges\n\n"
        "Each edge is weighted by pooled overclaim risk from the official claim records.\n\n"
        + markdown_table(show_graph)
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {signature_out} ({len(signature)} rows)")
    print(f"Wrote {signature_md}")
    print(f"Wrote {graph_out} ({len(graph)} rows)")
    print(f"Wrote {graph_md}")


if __name__ == "__main__":
    main()
