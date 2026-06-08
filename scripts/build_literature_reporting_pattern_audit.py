#!/usr/bin/env python3
"""Summarize the representative literature reporting-pattern audit.

The audit is deliberately modest: it codes which evidence object a workflow
family foregrounds in its evaluation/reporting protocol.  It is not a
corpus-level prevalence estimate and does not judge whether any individual
paper overclaims.
"""

from __future__ import annotations

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


def main() -> None:
    src = ROOT / "score_reports" / "literature_reporting_pattern_audit.csv"
    df = pd.read_csv(src)

    by_group = (
        df.groupby(["workflow_group", "evidence_object_coded"], dropna=False)
        .size()
        .reset_index(name="papers")
        .sort_values(["workflow_group", "papers"], ascending=[True, False])
    )
    by_edge = (
        df.groupby("promoted_edge_motivated", dropna=False)
        .size()
        .reset_index(name="papers")
        .sort_values("papers", ascending=False)
    )
    by_family = (
        df.groupby("workflow_group", dropna=False)
        .agg(
            papers=("paper_key", "size"),
            evidence_objects=("evidence_object_coded", lambda x: ", ".join(sorted(set(map(str, x))))),
            motivated_edges=("promoted_edge_motivated", lambda x: ", ".join(sorted(set(map(str, x))))),
        )
        .reset_index()
        .sort_values("papers", ascending=False)
    )

    out_dir = ROOT / "score_reports"
    by_group.to_csv(out_dir / "literature_reporting_pattern_by_group.csv", index=False)
    by_edge.to_csv(out_dir / "literature_reporting_pattern_by_edge.csv", index=False)
    by_family.to_csv(out_dir / "literature_reporting_pattern_by_family.csv", index=False)

    md = [
        "# Representative Literature Reporting-Pattern Audit",
        "",
        "This audit codes reporting patterns rather than correctness of individual papers.",
        "Each row records the evidence object foregrounded by a real workflow paper or",
        "benchmark and the ClaimTransfer edge that such a reporting pattern motivates.",
        "It is not a corpus-level prevalence estimate.",
        "",
        f"Total coded papers/workflows: {len(df)}",
        "",
        "## By motivated ClaimTransfer edge",
        "",
        markdown_table(by_edge),
        "",
        "## By workflow family",
        "",
        markdown_table(by_family),
        "",
        "## By workflow family and evidence object",
        "",
        markdown_table(by_group),
        "",
        "## Full coded rows",
        "",
        markdown_table(df, max_rows=120),
        "",
    ]
    (out_dir / "literature_reporting_pattern_audit.md").write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote {out_dir / 'literature_reporting_pattern_audit.md'}")
    print(f"coded rows: {len(df)}")


if __name__ == "__main__":
    main()
