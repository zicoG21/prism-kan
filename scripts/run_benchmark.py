#!/usr/bin/env python3
"""Quick ClaimTransfer-Bench runner.

This script is the lightweight reviewer entry point.  It runs checks that do
not require retraining long pyKAN jobs and writes score-report style summaries
from checked-in or previously generated CSVs.
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
    filters = ["ignore::UserWarning"]
    existing = env.get("PYTHONWARNINGS", "")
    env["PYTHONWARNINGS"] = ",".join([*filters, existing]) if existing else ",".join(filters)
    subprocess.run(cmd, cwd=ROOT, check=True, env=env)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["quick", "public", "full", "hidden"],
        default="quick",
        help=(
            "quick rebuilds reports from released outputs; public/full also run "
            "paper summary builders; hidden validates hidden-template cards and "
            "scores a supplied hidden adapter-output file if provided."
        ),
    )
    for mode_name in ("quick", "public", "full", "hidden"):
        parser.add_argument(
            f"--{mode_name}",
            dest=f"mode_{mode_name}",
            action="store_true",
            help=f"Shortcut for --mode {mode_name}.",
        )
    parser.add_argument(
        "--out-dir",
        default="results/workshop_review_tables/standard_audit_protocol",
        help="Output directory for the standard audit protocol summary.",
    )
    parser.add_argument(
        "--hidden-input",
        default="",
        help="Optional hidden/private normalized adapter-output CSV to score in hidden mode.",
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
    parser.add_argument(
        "--rebuild-adapter-outputs",
        action="store_true",
        help=(
            "Force rebuilding released adapter outputs from local result CSVs. "
            "By default the runner uses bundled released outputs when results/revision is absent."
        ),
    )
    args = parser.parse_args()
    shortcut_modes = [
        mode_name
        for mode_name in ("quick", "public", "full", "hidden")
        if getattr(args, f"mode_{mode_name}")
    ]
    if len(shortcut_modes) > 1:
        parser.error("Use at most one mode shortcut.")
    if shortcut_modes:
        args.mode = shortcut_modes[0]

    py = sys.executable
    run([py, "scripts/print_artifact_env.py"])
    run([py, "scripts/validate_release_contract.py"])
    run([py, "scripts/validate_task_cards.py"])
    run([py, "scripts/validate_adapter_registry.py"])
    released_outputs = ROOT / "claim_records" / "released_adapter_outputs.csv"
    if args.rebuild_adapter_outputs or not released_outputs.exists() or released_outputs.stat().st_size == 0:
        run([py, "scripts/build_claim_records.py"])
    else:
        print(
            "+ using bundled claim_records/released_adapter_outputs.csv",
            flush=True,
        )
    run([py, "scripts/validate_adapter_outputs.py", str(released_outputs)])
    run([py, "scripts/build_score_report.py"])
    run([py, "scripts/validate_claim_records.py"])
    run([py, "scripts/build_coverage_gap_report.py"])
    run([py, "scripts/summarize_coverage_gaps.py"])
    run([py, "scripts/validate_score_reports.py"])
    run([py, "scripts/build_benchmark_manifest.py"])
    run([py, "scripts/build_full_benchmark_readiness_report.py"])
    run([py, "scripts/build_release_candidate_report.py"])
    run([py, "scripts/build_full_benchmark_readiness_report.py"])
    run([py, "scripts/build_typed_dashboard.py"])
    run(
        [
            py,
            "scripts/check_benchmark_artifact.py",
            "--min-claim-rows",
            "100000",
            "--min-score-rows",
            "600",
            "--min-coverage-rows",
            "200",
        ]
    )

    if args.mode == "hidden":
        if args.hidden_input:
            run(
                [
                    py,
                    "scripts/build_score_report.py",
                    "--input",
                    args.hidden_input,
                    "--claim-record-out",
                    "claim_records/hidden_claim_records.csv",
                    "--score-report-out",
                    "score_reports/hidden_score_report.csv",
                    "--coverage-out",
                    "score_reports/hidden_coverage_table.csv",
                ]
            )
        print("\nHidden mode validated task-card contracts. Provide --hidden-input to score private rows.")
        return

    if args.mode in {"public", "full"}:
        run([py, "scripts/run_standard_audit_protocol.py", "--out-dir", args.out_dir])

    if args.mode == "full" and not args.skip_minisuite:
        run([py, "scripts/build_formal_minisuite_baseline_table.py"])

    if args.mode == "full" and not args.skip_cross_method:
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
    print(f"Typed dashboard: {ROOT / 'dashboards'}")
    print(f"Standard score report: {ROOT / args.out_dir}")


if __name__ == "__main__":
    main()
