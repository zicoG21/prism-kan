#!/usr/bin/env python3
"""Score a ClaimTransfer normalized adapter-output submission.

The submission file contains raw or normalized evidence rows.  It must not
contain trusted pass/fail labels.  This harness calls the official scorer and
writes claim records, aggregate score reports, and coverage tables under the
requested output directory.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    env = os.environ.copy()
    existing = env.get("PYTHONWARNINGS", "")
    filters = ["ignore::UserWarning"]
    env["PYTHONWARNINGS"] = ",".join([*filters, existing]) if existing else ",".join(filters)
    subprocess.run(cmd, cwd=ROOT, check=True, env=env)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission_csv", help="Normalized adapter-output CSV to score.")
    parser.add_argument(
        "--out-dir",
        default="score_reports/submission_score",
        help="Directory for official claim records, score report, and coverage table.",
    )
    parser.add_argument(
        "--validate-task-cards",
        action="store_true",
        help="Validate task cards before scoring the submission.",
    )
    args = parser.parse_args()

    submission = Path(args.submission_csv)
    if not submission.exists():
        raise SystemExit(f"Submission file does not exist: {submission}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    py = sys.executable

    if args.validate_task_cards:
        run([py, "scripts/validate_task_cards.py"])

    run(
        [
            py,
            "scripts/build_score_report.py",
            "--input",
            str(submission),
            "--claim-record-out",
            str(out_dir / "claim_records.csv"),
            "--score-report-out",
            str(out_dir / "score_report.csv"),
            "--coverage-out",
            str(out_dir / "coverage_table.csv"),
        ]
    )
    print(f"\nOfficial submission score written to {out_dir}")


if __name__ == "__main__":
    main()
