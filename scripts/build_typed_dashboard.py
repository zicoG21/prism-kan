#!/usr/bin/env python3
"""Build typed dashboard views from ClaimTransfer score reports."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path.with_suffix(".csv"), index=False)
    table = df.copy()
    for col in table.columns:
        if col.endswith("rate") or col in {"pass_rate", "wilson_low", "wilson_high"}:
            table[col] = table[col].map(lambda v: "" if pd.isna(v) else f"{v:.3f}")
        elif pd.api.types.is_float_dtype(table[col]):
            table[col] = table[col].map(lambda v: "" if pd.isna(v) else f"{v:.3f}")
    path.write_text(markdown_table(table) + "\n", encoding="utf-8")


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "No rows."
    table = df.astype(str)
    cols = list(table.columns)
    widths = [max(len(c), *(len(v) for v in table[c].tolist())) for c in cols]

    def row(vals: list[str]) -> str:
        return "| " + " | ".join(vals[i].ljust(widths[i]) for i in range(len(vals))) + " |"

    lines = [row(cols), "| " + " | ".join("-" * w for w in widths) + " |"]
    lines.extend(row([str(v) for v in values]) for values in table[cols].values.tolist())
    return "\n".join(lines)


def aggregate(df: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    g = df.groupby(groups, dropna=False).agg(
        score_rows=("rows", "sum"),
        trials=("trials", "sum"),
        successes=("successes", "sum"),
        missing_pass_rows=("missing_pass_rows", "sum"),
        median_rank=("median_rank", "median"),
        median_margin=("median_margin", "median"),
    )
    out = g.reset_index()
    out["pass_rate"] = out["successes"] / out["trials"].where(out["trials"] != 0)
    return out.sort_values(groups)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-report", default="score_reports/score_report.csv")
    parser.add_argument("--coverage", default="score_reports/coverage_table.csv")
    parser.add_argument("--out-dir", default="dashboards")
    args = parser.parse_args()

    score = pd.read_csv(args.score_report)
    coverage = pd.read_csv(args.coverage)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    views = {
        "adapter_by_claim.md": aggregate(score, ["adapter_family", "claim_type"]),
        "task_by_claim.md": aggregate(score, ["task_family", "claim_type"]),
        "object_by_claim.md": aggregate(score, ["evidence_object", "claim_type"]),
        "scorer_by_claim.md": aggregate(score, ["scorer", "claim_type"]),
    }
    for name, table in views.items():
        write_table(table, out / name)

    missing = coverage[coverage["missing_pass_rows"] > 0].copy()
    missing = missing.sort_values(["missing_pass_rows", "score_rows"], ascending=False)
    write_table(missing, out / "missingness.md")

    index = [
        "# ClaimTransfer Typed Dashboard",
        "",
        "Generated from official score reports.  These views avoid a single",
        "merged leaderboard and keep structural claims typed by adapter, task,",
        "evidence object, scorer, and claim type.",
        "",
        "## Views",
        "",
        "- `adapter_by_claim.md`",
        "- `task_by_claim.md`",
        "- `object_by_claim.md`",
        "- `scorer_by_claim.md`",
        "- `missingness.md`",
        "",
        "Regenerate with:",
        "",
        "```bash",
        "python scripts/build_typed_dashboard.py",
        "```",
        "",
    ]
    (out / "README.md").write_text("\n".join(index), encoding="utf-8")
    print(f"Wrote typed dashboard views to {out}")


if __name__ == "__main__":
    main()
