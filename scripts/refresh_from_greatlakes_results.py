#!/usr/bin/env python3
"""Unpack optional Great Lakes result bundles and refresh official benchmark reports."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Snapshot:
    label: str
    adapter_rows: int
    claim_rows: int
    score_rows: int
    coverage_rows: int
    covered_cells: int
    missing_cells: int
    missingness_rows: int


def csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    return int(pd.read_csv(path, low_memory=False).shape[0])


def coverage_counts(path: Path) -> tuple[int, int]:
    if not path.exists():
        return 0, 0
    df = pd.read_csv(path)
    covered = int((df["coverage_status"] == "covered").sum())
    missing = int((df["coverage_status"] != "covered").sum())
    return covered, missing


def snapshot(label: str) -> Snapshot:
    covered, missing = coverage_counts(ROOT / "score_reports/coverage_gap_report.csv")
    return Snapshot(
        label=label,
        adapter_rows=csv_rows(ROOT / "claim_records/released_adapter_outputs.csv"),
        claim_rows=csv_rows(ROOT / "claim_records/released_claim_records.csv"),
        score_rows=csv_rows(ROOT / "score_reports/score_report.csv"),
        coverage_rows=csv_rows(ROOT / "score_reports/coverage_table.csv"),
        covered_cells=covered,
        missing_cells=missing,
        missingness_rows=csv_rows(ROOT / "score_reports/missingness_report.csv"),
    )


def safe_extract(tar_path: Path, dest: Path) -> int:
    dest_resolved = dest.resolve()
    count = 0
    with tarfile.open(tar_path, "r:*") as tar:
        for member in tar.getmembers():
            target = (dest / member.name).resolve()
            if os.path.commonpath([str(dest_resolved), str(target)]) != str(dest_resolved):
                raise SystemExit(f"Unsafe path in tarball {tar_path}: {member.name}")
        tar.extractall(dest)
        count = len(tar.getmembers())
    return count


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def write_report(before: Snapshot, after: Snapshot, unpacked: list[Path], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for snap in [before, after]:
        rows.append(snap.__dict__)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    delta = {
        "adapter_rows_delta": after.adapter_rows - before.adapter_rows,
        "claim_rows_delta": after.claim_rows - before.claim_rows,
        "score_rows_delta": after.score_rows - before.score_rows,
        "coverage_rows_delta": after.coverage_rows - before.coverage_rows,
        "covered_cells_delta": after.covered_cells - before.covered_cells,
        "missing_cells_delta": after.missing_cells - before.missing_cells,
        "missingness_rows_delta": after.missingness_rows - before.missingness_rows,
    }
    md = out.with_suffix(".md")
    lines = [
        "# Great Lakes refresh report",
        "",
        "## Unpacked bundles",
        "",
    ]
    if unpacked:
        lines.extend(f"- `{p}`" for p in unpacked)
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Before/after",
            "",
            df.to_markdown(index=False),
            "",
            "## Delta",
            "",
        ]
    )
    lines.extend(f"- {k}: {v:+d}" for k, v in delta.items())
    lines.extend(
        [
            "",
            "## Current top gaps",
            "",
        ]
    )
    gap_md = ROOT / "score_reports/coverage_gap_summary.md"
    if gap_md.exists():
        lines.extend(gap_md.read_text(encoding="utf-8").splitlines()[0:35])
    else:
        lines.append("No coverage gap summary found.")
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out}")
    print(f"Wrote {md}")
    print("Delta:")
    for key, value in delta.items():
        print(f"  {key}: {value:+d}")


def latest_tarballs() -> list[Path]:
    candidates = sorted((ROOT / "artifacts/greatlakes").glob("revision_results_*.tar.gz"))
    candidates += sorted((ROOT / "artifacts/greatlakes/old_tarballs").glob("revision_results_*.tar.gz"))
    if not candidates:
        return []
    return [max(candidates, key=lambda p: p.stat().st_mtime)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "tarballs",
        nargs="*",
        help="Optional Great Lakes revision_results_*.tar.gz bundles to unpack before refreshing.",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Unpack the newest artifacts/greatlakes revision_results_*.tar.gz bundle.",
    )
    parser.add_argument(
        "--no-unpack",
        action="store_true",
        help="Skip unpacking and only rebuild official reports from the current results/revision tree.",
    )
    parser.add_argument("--out", default="score_reports/greatlakes_refresh_report.csv")
    args = parser.parse_args()

    before = snapshot("before")
    tarballs = [ROOT / p for p in args.tarballs]
    if args.latest:
        tarballs.extend(latest_tarballs())
    unpacked: list[Path] = []
    if not args.no_unpack:
        for tarball in tarballs:
            if not tarball.exists():
                raise SystemExit(f"Tarball not found: {tarball}")
            count = safe_extract(tarball, ROOT)
            print(f"Unpacked {tarball} ({count} entries)")
            unpacked.append(tarball)

    py = sys.executable
    run([py, "scripts/run_benchmark.py", "--quick", "--rebuild-adapter-outputs"])
    after = snapshot("after")
    write_report(before, after, unpacked, ROOT / args.out)


if __name__ == "__main__":
    main()
