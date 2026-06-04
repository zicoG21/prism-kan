#!/usr/bin/env python3
"""Validate normalized ClaimTransfer adapter-output rows.

Adapter outputs are the raw evidence submitted to the official scorer.  They
must contain the fields needed to recompute claim records, and they must not
ship trusted final verdict columns such as ``pass`` or aggregate score fields.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN_COLUMNS = {
    "pass",
    "success",
    "successes",
    "trials",
    "pass_rate",
    "wilson_low",
    "wilson_high",
    "rank_num",
    "margin_num",
    "raw_num",
}


def required_columns() -> list[str]:
    schema = json.loads((ROOT / "adapters/adapter_output_schema.json").read_text(encoding="utf-8"))
    return list(schema.get("required", []))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", default="claim_records/released_adapter_outputs.csv")
    parser.add_argument("--out", default="score_reports/adapter_output_validation.csv")
    parser.add_argument("--allow-empty", action="store_true")
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        raise SystemExit(f"Adapter-output CSV does not exist: {path}")
    df = pd.read_csv(path, low_memory=False)
    if df.empty and not args.allow_empty:
        raise SystemExit(f"Adapter-output CSV is empty: {path}")

    required = required_columns()
    missing_columns = [col for col in required if col not in df.columns]
    forbidden_columns = [col for col in sorted(FORBIDDEN_COLUMNS) if col in df.columns]

    rows = []
    for col in required:
        if col in df.columns:
            missing_values = int(df[col].isna().sum() + (df[col].astype(str).str.strip() == "").sum())
            rows.append(
                {
                    "field": col,
                    "status": "present",
                    "missing_or_blank_rows": missing_values,
                    "rows": int(len(df)),
                }
            )
        else:
            rows.append({"field": col, "status": "missing_column", "missing_or_blank_rows": "", "rows": int(len(df))})

    for col in forbidden_columns:
        rows.append({"field": col, "status": "forbidden_column", "missing_or_blank_rows": "", "rows": int(len(df))})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)

    if missing_columns or forbidden_columns:
        problems = []
        if missing_columns:
            problems.append("missing required columns: " + ", ".join(missing_columns))
        if forbidden_columns:
            problems.append("forbidden official-result columns: " + ", ".join(forbidden_columns))
        raise SystemExit("; ".join(problems))

    print(f"Validated adapter outputs: {path}")
    print(f"rows: {len(df)}")
    print(f"required columns: {len(required)}")
    print(f"wrote: {out}")


if __name__ == "__main__":
    main()
