#!/usr/bin/env python3
"""Build a release-candidate gate report for ClaimTransfer.

This report is deliberately stricter than "the quick path runs" and more
honest than "the public benchmark is released."  It records whether the local
artifact is ready to be frozen as a release candidate, and it keeps data
coverage blockers separate from packaging and public-tag blockers.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Gate:
    gate: str
    status: str
    evidence: str
    next_action: str


def csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    return int(pd.read_csv(path, low_memory=False).shape[0])


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def task_card_count() -> int:
    total = 0
    for path in (ROOT / "task_cards").glob("*.json"):
        if path.name == "task_card_schema.json":
            continue
        data = read_json(path)
        if "cards" in data:
            total += len(data["cards"])
        elif data.get("task_id"):
            total += 1
    return total


def readiness_counts() -> dict[tuple[str, str], int]:
    p = ROOT / "score_reports/full_benchmark_readiness.csv"
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    return {
        (str(priority), str(status)): int(count)
        for (priority, status), count in df.groupby(["priority", "status"]).size().items()
    }


def coverage_gap_counts() -> tuple[int, int]:
    p = ROOT / "score_reports/coverage_gap_report.csv"
    if not p.exists():
        return 0, 0
    df = pd.read_csv(p)
    covered = int((df["coverage_status"] == "covered").sum())
    missing = int((df["coverage_status"] != "covered").sum())
    return covered, missing


def latest_release_artifact() -> str:
    bundles = sorted((ROOT / "artifacts/release").glob("claimtransfer_release_*.tar.gz"), key=lambda p: p.stat().st_mtime)
    if bundles:
        return str(bundles[-1].relative_to(ROOT))
    manifests = sorted(
        (ROOT / "artifacts/release").glob("claimtransfer_release_manifest_*.txt"),
        key=lambda p: p.stat().st_mtime,
    )
    return str(manifests[-1].relative_to(ROOT)) if manifests else ""


def status_if(condition: bool, success: str = "complete", fail: str = "blocked") -> str:
    return success if condition else fail


def build_gates() -> list[Gate]:
    release = read_json(ROOT / "benchmark_release.json")
    counts = readiness_counts()
    covered, missing = coverage_gap_counts()
    release_artifact = latest_release_artifact()

    p0_complete = counts.get(("P0", "complete"), 0)
    p1_blockers = sum(counts.get(("P1", status), 0) for status in ["blocked", "blocked_on_data", "partial"])
    p2_ready = counts.get(("P2", "complete"), 0) + counts.get(("P2", "alpha_complete"), 0)

    adapter_rows = csv_rows(ROOT / "claim_records/released_adapter_outputs.csv")
    claim_rows = csv_rows(ROOT / "claim_records/released_claim_records.csv")
    score_rows = csv_rows(ROOT / "score_reports/score_report.csv")
    manifest_rows = csv_rows(ROOT / "score_reports/benchmark_manifest.csv")

    gates = [
        Gate(
            "release metadata",
            status_if(bool(release.get("release_id")) and bool(release.get("schema_version"))),
            f"release_id={release.get('release_id', '')}; stage={release.get('release_stage', '')}; commit={git_commit()}",
            "fill benchmark_release.json before freezing a candidate",
        ),
        Gate(
            "registry and schema validation",
            status_if(csv_rows(ROOT / "score_reports/task_card_validation.csv") >= 1 and task_card_count() >= 21),
            f"{task_card_count()} task-card/template rows; task-card validation rows={csv_rows(ROOT / 'score_reports/task_card_validation.csv')}",
            "rerun validate_task_cards.py and fix any failing registry row",
        ),
        Gate(
            "official-scored outputs",
            status_if(adapter_rows >= 100000 and adapter_rows == claim_rows and score_rows >= 600),
            f"{adapter_rows} adapter rows -> {claim_rows} claim rows; {score_rows} score rows",
            "rerun run_benchmark.py --quick --rebuild-adapter-outputs after merging GL results",
        ),
        Gate(
            "readiness P0",
            status_if(p0_complete >= 10),
            f"P0 complete checks={p0_complete}",
            "keep P0 green before interpreting P1/P2 gates",
        ),
        Gate(
            "claim-grammar coverage",
            status_if(missing == 0, fail="blocked_on_data"),
            f"{covered} covered expected cells; {missing} missing expected cells",
            "merge GL gap-fill outputs and rerun the quick path",
        ),
        Gate(
            "P1 data blockers",
            status_if(p1_blockers == 0, fail="blocked_on_data"),
            f"P1 blocker/partial checks={p1_blockers}",
            "resolve data-dependent P1 rows before tagging a mature benchmark release",
        ),
        Gate(
            "offline P2 scaffold",
            status_if(p2_ready >= 3, success="alpha_complete", fail="blocked"),
            f"P2 complete-or-alpha checks={p2_ready}",
            "rerun hidden/private bundle and submission scoring checks after schema changes",
        ),
        Gate(
            "release bundle",
            status_if(bool(release_artifact) and manifest_rows >= 40, success="alpha_complete", fail="blocked"),
            f"latest_release_artifact={release_artifact or 'missing'}; manifest rows={manifest_rows}",
            "run build_claimtransfer_release_bundle.sh and overlay smoke tests",
        ),
        Gate(
            "public tag/server",
            "future_work",
            "no public git tag or hosted submission server is required for the local alpha candidate",
            "after P1 coverage is complete, tag the frozen registry and publish the bundle",
        ),
    ]
    return gates


def markdown_table(df: pd.DataFrame) -> str:
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
    parser.add_argument("--out", default="score_reports/release_candidate_report.csv")
    args = parser.parse_args()

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([g.__dict__ for g in build_gates()])
    df.to_csv(out, index=False)
    out.with_suffix(".md").write_text(
        "# ClaimTransfer release-candidate gate report\n\n"
        "A public release tag should wait until data-dependent P1 gates are complete. "
        "Alpha packaging and offline-scoring gates may be complete earlier.\n\n"
        + markdown_table(df)
        + "\n",
        encoding="utf-8",
    )
    out.with_suffix(".json").write_text(
        json.dumps(
            {
                "git_commit": git_commit(),
                "gates": df.to_dict(orient="records"),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {out} ({len(df)} gates)")
    print(df.groupby("status").size().to_string())


if __name__ == "__main__":
    main()
