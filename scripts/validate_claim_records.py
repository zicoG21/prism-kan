#!/usr/bin/env python3
"""Validate official ClaimTransfer claim-record rows."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def required_columns() -> list[str]:
    schema = json.loads((ROOT / "claim_records/claim_record_schema.json").read_text(encoding="utf-8"))
    return list(schema.get("required", []))


def valid_pass_value(value: object) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except Exception:
        pass
    text = str(value).strip().lower()
    if text in {"", "nan", "none", "null"}:
        return True
    try:
        numeric = float(text)
    except Exception:
        return False
    if math.isnan(numeric):
        return True
    return numeric in {0.0, 1.0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", default="claim_records/released_claim_records.csv")
    parser.add_argument("--out", default="score_reports/claim_record_validation.csv")
    parser.add_argument("--allow-empty", action="store_true")
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        raise SystemExit(f"Claim-record CSV does not exist: {path}")
    df = pd.read_csv(path, low_memory=False)
    if df.empty and not args.allow_empty:
        raise SystemExit(f"Claim-record CSV is empty: {path}")

    required = required_columns()
    missing_columns = [col for col in required if col not in df.columns]

    rows = []
    for col in required:
        if col in df.columns:
            blank_rows = int(df[col].isna().sum() + (df[col].astype(str).str.strip() == "").sum())
            rows.append(
                {
                    "field": col,
                    "status": "present",
                    "missing_or_blank_rows": blank_rows,
                    "rows": int(len(df)),
                }
            )
        else:
            rows.append({"field": col, "status": "missing_column", "missing_or_blank_rows": "", "rows": int(len(df))})

    invalid_pass_rows = 0
    if "pass" in df.columns:
        invalid_pass_rows = int((~df["pass"].map(valid_pass_value)).sum())
        rows.append(
            {
                "field": "pass",
                "status": "invalid_pass_values",
                "missing_or_blank_rows": invalid_pass_rows,
                "rows": int(len(df)),
            }
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)

    problems = []
    if missing_columns:
        problems.append("missing required columns: " + ", ".join(missing_columns))
    if invalid_pass_rows:
        problems.append(f"invalid pass values: {invalid_pass_rows}")
    if problems:
        raise SystemExit("; ".join(problems))

    print(f"Validated claim records: {path}")
    print(f"rows: {len(df)}")
    print(f"required columns: {len(required)}")
    print(f"wrote: {out}")


if __name__ == "__main__":
    main()
