#!/usr/bin/env python3
"""Summarize full-benchmark coverage gaps into actionable merge/GL targets."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

ACTION_HINTS = {
    "pykan_workflow": "merge pyKAN claimcard/scorergram/readout outputs; rerun quick scorer refresh",
    "epim_treegate": "merge EPIM/TreeGate candidate-verifier and breadth outputs",
    "tree_interaction": "merge TreeGate/GBM-H standard CPU outputs",
    "ga2m_additive": "merge GA2M/EBM-style cross-method standard outputs",
    "sparse_library": "merge sparse-lasso/spline-lasso cross-method rows for the missing task families",
    "symbolic_library": "merge symbolic-library/prune-symbolic expression rows or keep as optional future symbolic track",
}


def compact_list(values: list[str], limit: int = 6) -> str:
    vals = sorted(set(map(str, values)))
    if len(vals) <= limit:
        return ", ".join(vals)
    return ", ".join(vals[:limit]) + f", ... (+{len(vals) - limit})"


def to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "# Coverage gap summary\n\nNo missing cells.\n"
    lines = ["# Coverage gap summary", ""]
    lines.append("| adapter family | claim type | missing cells | task families | suggested action |")
    lines.append("| --- | --- | ---: | --- | --- |")
    for row in df.itertuples(index=False):
        lines.append(
            f"| {row.canonical_adapter_family} | {row.claim_type} | {row.missing_cells} | "
            f"{row.task_families} | {row.suggested_action} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="score_reports/coverage_gap_report.csv")
    parser.add_argument("--out", default="score_reports/coverage_gap_summary.csv")
    args = parser.parse_args()

    src = ROOT / args.input
    if not src.exists():
        raise SystemExit(f"Missing coverage gap report: {src}")
    gap = pd.read_csv(src)
    missing = gap[gap["coverage_status"] != "covered"].copy()
    rows = []
    for (adapter_family, claim_type), block in missing.groupby(["canonical_adapter_family", "claim_type"]):
        rows.append(
            {
                "canonical_adapter_family": adapter_family,
                "claim_type": claim_type,
                "missing_cells": int(len(block)),
                "task_families": compact_list(block["task_family"].tolist()),
                "suggested_action": ACTION_HINTS.get(
                    str(adapter_family), "merge rows for this adapter family, then rerun scripts/run_benchmark.py --quick"
                ),
            }
        )
    if rows:
        out_df = pd.DataFrame(rows).sort_values(
            ["missing_cells", "canonical_adapter_family", "claim_type"], ascending=[False, True, True]
        )
    else:
        out_df = pd.DataFrame(
            columns=[
                "canonical_adapter_family",
                "claim_type",
                "missing_cells",
                "task_families",
                "suggested_action",
            ]
        )

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out, index=False)
    out.with_suffix(".md").write_text(to_markdown(out_df), encoding="utf-8")
    print(f"Wrote {out} ({len(out_df)} grouped gaps)")
    if not out_df.empty:
        print(out_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
