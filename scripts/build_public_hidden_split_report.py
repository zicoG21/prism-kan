#!/usr/bin/env python3
"""Report public-vs-hidden split status for ClaimTransfer-Bench.

This script distinguishes three cases:
1. public diagnostic rows;
2. standard-formula public rows;
3. hidden/private scored rows, if present.

If hidden/private scored rows are absent, the report says so explicitly and
points to the GL jobs needed to materialize them.  It does not fabricate hidden
consistency statistics from public rows.
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


def split_bucket(df: pd.DataFrame) -> pd.Series:
    split = df.get("split", pd.Series("", index=df.index)).astype(str).str.lower()
    registry = df.get("registry_version", pd.Series("", index=df.index)).astype(str)
    task_id = df.get("task_id", pd.Series("", index=df.index)).astype(str)
    bucket = pd.Series("public diagnostic", index=df.index)
    standard = registry.eq("claimtransfer_v1_standard_formula_public") | task_id.str.startswith("std_")
    hidden = split.str.contains("hidden|private") | registry.str.contains("hidden|private", case=False, na=False)
    bucket.loc[standard] = "standard formula public"
    bucket.loc[hidden] = "hidden/private scored"
    return bucket


def main() -> None:
    out_dir = ROOT / "score_reports"
    claims = pd.read_csv(ROOT / "claim_records" / "released_claim_records.csv", low_memory=False)
    claims["split_bucket"] = split_bucket(claims)

    summary = (
        claims.groupby("split_bucket", dropna=False)
        .agg(
            claim_rows=("task_id", "size"),
            task_ids=("task_id", "nunique"),
            adapter_families=("adapter_family", "nunique"),
            adapters=("adapter", "nunique"),
            claim_types=("claim_type", "nunique"),
        )
        .reset_index()
        .sort_values("split_bucket")
    )
    hidden_rows = int(summary.loc[summary["split_bucket"].eq("hidden/private scored"), "claim_rows"].sum())

    actions = []
    if hidden_rows == 0:
        actions.append(
            {
                "status": "not_actionable_without_new_gl_rows",
                "reason": "released claim records contain no rows whose split or registry is hidden/private",
                "gl_entry_point": "scripts/submit_claimtransfer_hidden_split_gl.sh",
            }
        )
    else:
        actions.append(
            {
                "status": "ready",
                "reason": "hidden/private scored rows are present in released claim records",
                "gl_entry_point": "",
            }
        )
    actions_df = pd.DataFrame(actions)

    summary_path = out_dir / "public_hidden_split_readiness.csv"
    actions_path = out_dir / "public_hidden_split_action_items.csv"
    summary.to_csv(summary_path, index=False)
    actions_df.to_csv(actions_path, index=False)

    md = [
        "# Public-vs-Hidden Split Readiness",
        "",
        "This report checks whether the released artifact contains scored",
        "hidden/private rows.  It does not infer hidden consistency from public",
        "diagnostic rows.",
        "",
        "## Split summary",
        "",
        markdown_table(summary),
        "",
        "## Action items",
        "",
        markdown_table(actions_df),
        "",
    ]
    summary_path.with_suffix(".md").write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote {summary_path}")
    print(f"hidden/private scored rows: {hidden_rows}")
    if hidden_rows == 0:
        print("Hidden consistency table requires new GL hidden/private scored rows.")


if __name__ == "__main__":
    main()
