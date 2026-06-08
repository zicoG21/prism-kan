#!/usr/bin/env python3
"""Create and score a lightweight offline private-eval demo submission.

The demo samples released adapter-output rows, removes any trusted pass labels,
and routes the file through the official submission scorer.  It demonstrates
the typed score-report path without claiming to be a hosted leaderboard.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="claim_records/released_adapter_outputs.csv")
    parser.add_argument("--out-dir", default="artifacts/private_eval_demo")
    parser.add_argument("--rows-per-adapter", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20260608)
    args = parser.parse_args()

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    src = pd.read_csv(ROOT / args.input, low_memory=False)
    sampled_parts = []
    for i, (_, group) in enumerate(src.groupby("adapter", dropna=False)):
        sampled_parts.append(
            group.sample(min(len(group), args.rows_per_adapter), random_state=args.seed + i)
        )
    sampled = pd.concat(sampled_parts, ignore_index=True)
    if "pass" in sampled.columns:
        sampled = sampled.drop(columns=["pass"])
    submission = out_dir / "demo_submission_adapter_outputs.csv"
    sampled.to_csv(submission, index=False)

    score_dir = out_dir / "official_score"
    run(
        [
            sys.executable,
            "scripts/score_submission.py",
            str(submission),
            "--out-dir",
            str(score_dir),
            "--validate-task-cards",
        ]
    )

    readme = [
        "# Offline Private-Evaluation Demo",
        "",
        "This directory demonstrates the ClaimTransfer submission path.  It is",
        "not a hosted leaderboard.  The demo submission is sampled from released",
        "adapter-output rows and scored by the official scorer.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "python scripts/build_private_eval_demo.py",
        "```",
        "",
        "Outputs:",
        "",
        "- `demo_submission_adapter_outputs.csv`: raw/normalized adapter evidence;",
        "- `official_score/claim_records.csv`: official typed claim records;",
        "- `official_score/score_report.csv`: aggregate typed score report;",
        "- `official_score/dashboard/`: typed dashboard views.",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(readme), encoding="utf-8")
    print(f"Wrote private-eval demo to {out_dir}")


if __name__ == "__main__":
    main()
