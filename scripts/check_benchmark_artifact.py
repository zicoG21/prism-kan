#!/usr/bin/env python3
"""Check that the ClaimTransfer benchmark artifact has the expected contract files."""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = [
    "task_cards/task_card_schema.json",
    "task_cards/claimtransfer_v0_public.json",
    "task_cards/claimtransfer_v0_hidden_template.json",
    "task_cards/claimtransfer_v1_scientific_templates.json",
    "adapters/adapter_output_schema.json",
    "adapters/adapter_family_registry.json",
    "claim_records/claim_record_schema.json",
    "scripts/validate_task_cards.py",
    "scripts/validate_adapter_registry.py",
    "scripts/build_claim_records.py",
    "scripts/build_score_report.py",
    "scripts/build_benchmark_manifest.py",
    "scripts/run_benchmark.py",
    "scripts/score_submission.py",
    "scripts/build_claimtransfer_release_bundle.sh",
    "scripts/build_hidden_private_bundle.py",
    "scripts/check_release_bundle.sh",
    "score_reports/task_card_validation.csv",
    "score_reports/adapter_family_validation.csv",
    "score_reports/score_report.csv",
    "score_reports/coverage_table.csv",
    "score_reports/benchmark_manifest.csv",
]


def csv_rows(path: Path) -> int:
    return int(pd.read_csv(path, low_memory=False).shape[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-claim-rows", type=int, default=100000)
    parser.add_argument("--min-score-rows", type=int, default=600)
    parser.add_argument("--min-coverage-rows", type=int, default=200)
    args = parser.parse_args()

    missing = [p for p in REQUIRED_FILES if not (ROOT / p).exists()]
    if missing:
        raise SystemExit("Missing required artifact files:\n" + "\n".join(missing))

    checks = {
        "task_card_validation": csv_rows(ROOT / "score_reports/task_card_validation.csv"),
        "adapter_family_validation": csv_rows(ROOT / "score_reports/adapter_family_validation.csv"),
        "score_report": csv_rows(ROOT / "score_reports/score_report.csv"),
        "coverage_table": csv_rows(ROOT / "score_reports/coverage_table.csv"),
        "benchmark_manifest": csv_rows(ROOT / "score_reports/benchmark_manifest.csv"),
    }
    if checks["score_report"] < args.min_score_rows:
        raise SystemExit(f"score_report too small: {checks['score_report']}")
    if checks["coverage_table"] < args.min_coverage_rows:
        raise SystemExit(f"coverage_table too small: {checks['coverage_table']}")

    claim_records = ROOT / "claim_records/released_claim_records.csv"
    if claim_records.exists():
        claim_rows = csv_rows(claim_records)
        checks["claim_records"] = claim_rows
        if claim_rows < args.min_claim_rows:
            raise SystemExit(f"claim_records too small: {claim_rows}")

    print("ClaimTransfer benchmark artifact check passed.")
    for key, value in checks.items():
        print(f"{key}: {value} rows")


if __name__ == "__main__":
    main()
