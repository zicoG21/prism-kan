#!/usr/bin/env python3
"""Quick ClaimTransfer-Bench runner.

This script is the lightweight reviewer entry point.  It runs checks that do
not require retraining long pyKAN jobs and writes score-report style summaries
from checked-in or previously generated CSVs.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        default="results/workshop_review_tables/standard_audit_protocol",
        help="Output directory for the standard audit protocol summary.",
    )
    parser.add_argument(
        "--skip-cross-method",
        action="store_true",
        help="Skip the cross-method transfer matrix build.",
    )
    parser.add_argument(
        "--skip-minisuite",
        action="store_true",
        help="Skip the formal mini-suite summary.",
    )
    args = parser.parse_args()

    py = sys.executable
    run([py, "scripts/print_artifact_env.py"])
    run([py, "scripts/validate_task_cards.py"])
    run([py, "scripts/build_claim_records.py"])
    run([py, "scripts/build_score_report.py"])
    run([py, "scripts/run_standard_audit_protocol.py", "--out-dir", args.out_dir])

    if not args.skip_minisuite:
        run([py, "scripts/build_formal_minisuite_baseline_table.py"])

    if not args.skip_cross_method:
        run(
            [
                py,
                "scripts/build_cross_method_transfer_matrix.py",
                "--output-prefix",
                "local_notes/generated/reviewer_cross_method_transfer",
            ]
        )

    print("\nClaimTransfer-Bench quick check complete.")
    print(f"Released adapter outputs: {ROOT / 'claim_records/released_adapter_outputs.csv'}")
    print(f"Official claim records: {ROOT / 'claim_records/released_claim_records.csv'}")
    print(f"Official score report: {ROOT / 'score_reports/score_report.csv'}")
    print(f"Coverage table: {ROOT / 'score_reports/coverage_table.csv'}")
    print(f"Standard score report: {ROOT / args.out_dir}")


if __name__ == "__main__":
    main()
