#!/usr/bin/env python3
"""Validate generated ClaimTransfer score, coverage, gap, and missingness reports."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

REPORTS = {
    "score_report": {
        "path": "score_reports/score_report.csv",
        "required": {
            "registry_version",
            "split",
            "task_family",
            "task_id",
            "adapter_family",
            "adapter",
            "evidence_object",
            "claim_type",
            "scorer",
            "predicate",
            "rows",
            "successes",
            "trials",
            "pass_rate",
            "wilson_low",
            "wilson_high",
        },
    },
    "coverage_table": {
        "path": "score_reports/coverage_table.csv",
        "required": {
            "registry_version",
            "split",
            "adapter_family",
            "task_family",
            "claim_type",
            "score_rows",
            "successes",
            "trials",
            "pass_rate",
            "wilson_low",
            "wilson_high",
        },
    },
    "coverage_gap_report": {
        "path": "score_reports/coverage_gap_report.csv",
        "required": {
            "canonical_adapter_family",
            "task_family",
            "claim_type",
            "coverage_status",
            "score_rows",
            "trials",
        },
    },
    "missingness_report": {
        "path": "score_reports/missingness_report.csv",
        "required": {
            "registry_version",
            "split",
            "adapter_family",
            "adapter",
            "task_family",
            "claim_type",
            "evidence_object",
            "scorer",
            "predicate",
            "rows",
            "missing_pass_rows",
            "missing_pass_rate",
        },
    },
}


def in_unit_interval(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.isna() | ((numeric >= 0) & (numeric <= 1))


def nonnegative(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.isna() | (numeric >= 0)


def validate_report(name: str, path: Path, required: set[str]) -> list[dict[str, object]]:
    if not path.exists():
        return [{"report": name, "check": "file_exists", "passed": False, "details": str(path)}]
    df = pd.read_csv(path, low_memory=False)
    rows: list[dict[str, object]] = [{"report": name, "check": "file_exists", "passed": True, "details": str(path)}]
    missing_columns = sorted(required - set(df.columns))
    rows.append(
        {
            "report": name,
            "check": "required_columns",
            "passed": not missing_columns,
            "details": ",".join(missing_columns),
        }
    )
    rows.append({"report": name, "check": "nonempty", "passed": not df.empty, "details": len(df)})

    for col in ["rows", "score_rows", "successes", "trials", "missing_pass_rows"]:
        if col in df.columns:
            bad = int((~nonnegative(df[col])).sum())
            rows.append({"report": name, "check": f"{col}_nonnegative", "passed": bad == 0, "details": bad})

    for col in ["pass_rate", "wilson_low", "wilson_high", "missing_pass_rate", "explicit_missing_reason_rate"]:
        if col in df.columns:
            bad = int((~in_unit_interval(df[col])).sum())
            rows.append({"report": name, "check": f"{col}_unit_interval", "passed": bad == 0, "details": bad})

    if {"wilson_low", "wilson_high"}.issubset(df.columns):
        lo = pd.to_numeric(df["wilson_low"], errors="coerce")
        hi = pd.to_numeric(df["wilson_high"], errors="coerce")
        bad = int(((lo.notna()) & (hi.notna()) & (lo > hi)).sum())
        rows.append({"report": name, "check": "wilson_interval_order", "passed": bad == 0, "details": bad})

    if name == "coverage_gap_report" and "coverage_status" in df.columns:
        allowed = {"covered", "missing_cell", "insufficient_trials"}
        bad_values = sorted(set(map(str, df["coverage_status"].dropna())) - allowed)
        rows.append(
            {
                "report": name,
                "check": "coverage_status_values",
                "passed": not bad_values,
                "details": ",".join(bad_values),
            }
        )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="score_reports/report_validation.csv")
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for name, spec in REPORTS.items():
        rows.extend(validate_report(name, ROOT / str(spec["path"]), set(spec["required"])))

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)

    failures = [r for r in rows if not bool(r["passed"])]
    if failures:
        preview = "\n".join(f"{r['report']} {r['check']}: {r['details']}" for r in failures[:20])
        raise SystemExit(f"Score-report validation failed:\n{preview}")

    print("Validated score reports.")
    print(f"checks: {len(rows)}")
    print(f"wrote: {out}")


if __name__ == "__main__":
    main()
