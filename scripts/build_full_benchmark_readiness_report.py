#!/usr/bin/env python3
"""Build a P0/P1/P2 readiness report for the full ClaimTransfer benchmark."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Check:
    priority: str
    area: str
    status: str
    evidence: str
    next_action: str


def csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    return int(pd.read_csv(path, low_memory=False).shape[0])


def file_exists(path: str) -> bool:
    return (ROOT / path).exists()


def ok_or_blocked(condition: bool, blocked_status: str = "blocked") -> str:
    return "complete" if condition else blocked_status


def read_json(path: str) -> dict:
    p = ROOT / path
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def validation_status(path: str, min_rows: int = 1) -> tuple[str, str]:
    p = ROOT / path
    rows = csv_rows(p)
    return ok_or_blocked(rows >= min_rows), f"{rows} validation rows in {path}"


def count_task_cards() -> int:
    total = 0
    for path in (ROOT / "task_cards").glob("*.json"):
        if path.name in {"task_card_schema.json"}:
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if "cards" in data:
            total += len(data.get("cards", []))
        elif data.get("task_id"):
            total += 1
    return total


def coverage_gap_summary(path: Path) -> tuple[int, int, str]:
    if not path.exists():
        return 0, 0, "coverage gap report missing"
    df = pd.read_csv(path)
    covered = int((df["coverage_status"] == "covered").sum())
    missing = int((df["coverage_status"] != "covered").sum())
    if missing:
        top = (
            df[df["coverage_status"] != "covered"]
            .groupby(["canonical_adapter_family", "claim_type"], dropna=False)
            .size()
            .sort_values(ascending=False)
            .head(5)
        )
        summary = "; ".join(f"{a}/{c}: {n}" for (a, c), n in top.items())
    else:
        summary = "no missing expected cells"
    return covered, missing, summary


def unique_values(path: Path, column: str) -> set[str]:
    if not path.exists():
        return set()
    df = pd.read_csv(path, usecols=lambda c: c == column)
    if column not in df:
        return set()
    return set(map(str, df[column].dropna().unique()))


def build_checks() -> list[Check]:
    claim_rows = csv_rows(ROOT / "claim_records/released_claim_records.csv")
    adapter_rows = csv_rows(ROOT / "claim_records/released_adapter_outputs.csv")
    score_rows = csv_rows(ROOT / "score_reports/score_report.csv")
    coverage_rows = csv_rows(ROOT / "score_reports/coverage_table.csv")
    missingness_rows = csv_rows(ROOT / "score_reports/missingness_report.csv")
    manifest_rows = csv_rows(ROOT / "score_reports/benchmark_manifest.csv")
    task_cards = count_task_cards()
    scorers = unique_values(ROOT / "score_reports/score_report.csv", "scorer")
    adapters = unique_values(ROOT / "score_reports/score_report.csv", "adapter_family")
    covered_cells, missing_cells, missing_summary = coverage_gap_summary(
        ROOT / "score_reports/coverage_gap_report.csv"
    )

    checks: list[Check] = []

    for area, path in [
        ("release contract", "score_reports/release_contract_validation.csv"),
        ("task-card schema", "score_reports/task_card_validation.csv"),
        ("adapter-family registry", "score_reports/adapter_family_validation.csv"),
        ("raw adapter-output schema", "score_reports/adapter_output_validation.csv"),
        ("official claim-record schema", "score_reports/claim_record_validation.csv"),
        ("score-report schema", "score_reports/report_validation.csv"),
    ]:
        status, evidence = validation_status(path)
        checks.append(Check("P0", area, status, evidence, "rerun validators if this file is absent or empty"))

    checks.extend(
        [
            Check(
                "P0",
                "task-card inventory",
                ok_or_blocked(task_cards >= 21),
                f"{task_cards} public/hidden/template cards loaded",
                "add complete task-card examples before expanding claims",
            ),
            Check(
                "P0",
                "official scorer rebuild",
                ok_or_blocked(claim_rows >= 100000 and claim_rows == adapter_rows),
                f"{adapter_rows} raw adapter rows -> {claim_rows} official claim rows",
                "rerun scripts/build_claim_records.py after merging GL outputs",
            ),
            Check(
                "P0",
                "score and coverage reports",
                ok_or_blocked(score_rows >= 600 and coverage_rows >= 200 and missingness_rows >= 200),
                f"{score_rows} score rows, {coverage_rows} coverage rows, {missingness_rows} missingness rows",
                "rerun scripts/build_score_report.py and validators",
            ),
            Check(
                "P0",
                "reviewer quick path",
                ok_or_blocked(file_exists("scripts/run_benchmark.py") and file_exists("score_reports/benchmark_manifest.csv")),
                f"{manifest_rows} manifest entries generated",
                "run python scripts/run_benchmark.py --quick",
            ),
        ]
    )

    checks.extend(
        [
            Check(
                "P1",
                "adapter-family breadth",
                ok_or_blocked(len(adapters) >= 8, "partial"),
                f"{len(adapters)} adapter families in score_report: {', '.join(sorted(adapters))}",
                "merge remaining GL cross-method and TreeGate rows",
            ),
            Check(
                "P1",
                "scorer sensitivity breadth",
                ok_or_blocked({"functional_anova", "fd", "hessian", "epim"}.issubset(scorers), "partial"),
                f"{len(scorers)} scorer labels include: {', '.join(sorted(scorers)[:12])}",
                "wait for/merge remaining scorergram GL rows, especially Hessian and TreeGate verifier cells",
            ),
            Check(
                "P1",
                "coverage completeness",
                ok_or_blocked(missing_cells == 0, "blocked_on_data"),
                f"{covered_cells} covered expected cells; {missing_cells} missing. Largest gaps: {missing_summary}",
                "use score_reports/coverage_gap_report.csv to prioritize GL jobs; rerun quick path after merge",
            ),
            Check(
                "P1",
                "symbolic expression layer",
                ok_or_blocked(
                    {"symbolic_operator_recall", "symbolic_complexity"}.issubset(
                        unique_values(ROOT / "score_reports/coverage_gap_report.csv", "claim_type")
                    ),
                    "partial",
                ),
                "symbolic claim types are represented in the expected coverage grid",
                "add real symbolic-regression adapters before treating expression claims as primary",
            ),
        ]
    )

    hidden_validation_rows = csv_rows(ROOT / "score_reports/hidden_bundle_validation.csv")
    hidden_participant = file_exists("artifacts/private_hidden/claimtransfer_v0_hidden_participant.json")
    hidden_private = file_exists("artifacts/private_hidden/claimtransfer_v0_hidden_private_scoring.json")
    release_bundle_exists = any((ROOT / "artifacts/release").glob("claimtransfer_release_*.tar.gz"))

    checks.extend(
        [
            Check(
                "P2",
                "offline submission harness",
                ok_or_blocked(file_exists("scripts/score_submission.py") and file_exists("adapters/submission_metadata_schema.json")),
                "score_submission.py and submission metadata schema are present",
                "run example submission scoring after any schema change",
            ),
            Check(
                "P2",
                "hidden/private bundle",
                ok_or_blocked(hidden_validation_rows >= 40 and hidden_participant and hidden_private, "alpha_complete"),
                f"{hidden_validation_rows} hidden-bundle validation rows; participant={hidden_participant}; private={hidden_private}",
                "store real private cards/seeds outside the public repo for live evaluation",
            ),
            Check(
                "P2",
                "release packaging",
                ok_or_blocked(release_bundle_exists and manifest_rows >= 40, "alpha_complete"),
                f"release bundle exists={release_bundle_exists}; {manifest_rows} manifest entries",
                "run release overlay smoke test and tag only after coverage/data blockers are resolved",
            ),
            Check(
                "P2",
                "public benchmark release",
                "future_work",
                "no public tag or hosted submission server is expected yet",
                "freeze registry version, publish release bundle, then tag",
            ),
        ]
    )
    return checks


def to_markdown(checks: Iterable[Check]) -> str:
    rows = list(checks)
    lines = ["# Full benchmark readiness report", ""]
    for priority in ["P0", "P1", "P2"]:
        block = [r for r in rows if r.priority == priority]
        counts = pd.Series([r.status for r in block]).value_counts().to_dict()
        lines.append(f"## {priority}")
        lines.append("")
        lines.append("Status counts: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) + ".")
        lines.append("")
        lines.append("| area | status | evidence | next action |")
        lines.append("| --- | --- | --- | --- |")
        for r in block:
            lines.append(
                f"| {r.area} | {r.status} | {r.evidence.replace('|', '/')} | {r.next_action.replace('|', '/')} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="score_reports/full_benchmark_readiness.csv")
    args = parser.parse_args()

    checks = build_checks()
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([r.__dict__ for r in checks])
    df.to_csv(out, index=False)
    md = out.with_suffix(".md")
    md.write_text(to_markdown(checks), encoding="utf-8")

    print(f"Wrote {out} ({len(df)} checks)")
    print(df.groupby(["priority", "status"]).size().to_string())


if __name__ == "__main__":
    main()
