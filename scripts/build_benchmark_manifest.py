#!/usr/bin/env python3
"""Build a lightweight manifest for the ClaimTransfer benchmark artifact."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_DOCS = {
    "docs/adapter_fairness_and_budget_policy.md",
    "docs/claimtransfer_full_benchmark_status_20260604.md",
    "docs/claimtransfer_full_benchmark_todo_20260604.md",
    "docs/hidden_evaluation_protocol.md",
    "docs/release_checklist_full_benchmark.md",
    "docs/reproducibility_checklist.md",
    "docs/statistical_reporting_policy.md",
    "docs/submission_format.md",
    "docs/task_card_authoring_protocol.md",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def count_rows(path: Path) -> int | str:
    if path.suffix.lower() != ".csv":
        return ""
    try:
        return int(sum(1 for _ in path.open("r", encoding="utf-8")) - 1)
    except Exception:
        return ""


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "No files."
    table = df.astype(str)
    cols = list(table.columns)
    widths = [max(len(c), *(len(v) for v in table[c].tolist())) for c in cols]

    def row(vals: list[str]) -> str:
        return "| " + " | ".join(vals[i].ljust(widths[i]) for i in range(len(vals))) + " |"

    lines = [row(cols), "| " + " | ".join("-" * w for w in widths) + " |"]
    lines.extend(row([str(v) for v in values]) for values in table[cols].values.tolist())
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="score_reports/benchmark_manifest.csv")
    args = parser.parse_args()

    roots = ["task_cards", "adapters", "scorers", "claim_records", "score_reports", "docs"]
    rows = []
    for root_name in roots:
        root = ROOT / root_name
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            if path.name.startswith("."):
                continue
            rel = path.relative_to(ROOT)
            if root_name == "docs" and str(rel) not in OFFICIAL_DOCS:
                continue
            rows.append(
                {
                    "path": str(rel),
                    "artifact_group": root_name,
                    "bytes": path.stat().st_size,
                    "rows_if_csv": count_rows(path),
                    "sha256": sha256(path),
                }
            )

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    md = df[["path", "artifact_group", "bytes", "rows_if_csv"]].copy()
    out.with_suffix(".md").write_text(
        "# ClaimTransfer benchmark manifest\n\n"
        + markdown_table(md)
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {out} ({len(df)} files)")


if __name__ == "__main__":
    main()
