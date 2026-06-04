#!/usr/bin/env python3
"""Validate the ClaimTransfer benchmark release contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Missing release-contract file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", default="benchmark_release.json")
    parser.add_argument("--out", default="score_reports/release_contract_validation.csv")
    args = parser.parse_args()

    contract_path = ROOT / args.contract
    contract = load_json(contract_path)
    rows: list[dict[str, object]] = []

    for key in ["release_id", "release_stage", "schema_version"]:
        rows.append(
            {
                "item": key,
                "path": str(contract_path),
                "check": "present",
                "passed": bool(str(contract.get(key, "")).strip()),
                "details": contract.get(key, ""),
            }
        )

    for group in ["schemas", "official_validators", "official_reports"]:
        for rel in contract.get(group, []):
            path = ROOT / str(rel)
            rows.append(
                {
                    "item": group,
                    "path": str(rel),
                    "check": "file_exists",
                    "passed": path.exists(),
                    "details": path.stat().st_size if path.exists() else "missing",
                }
            )

    for registry in contract.get("task_registries", []):
        rel = str(registry.get("path", ""))
        path = ROOT / rel
        rows.append(
            {
                "item": "task_registry",
                "path": rel,
                "check": "file_exists",
                "passed": path.exists(),
                "details": path.stat().st_size if path.exists() else "missing",
            }
        )
        if not path.exists():
            continue
        data = load_json(path)
        expected_version = str(registry.get("version", ""))
        expected_split = str(registry.get("split", ""))
        rows.append(
            {
                "item": "task_registry",
                "path": rel,
                "check": "top_level_version",
                "passed": data.get("version") == expected_version,
                "details": data.get("version", ""),
            }
        )
        cards = list(data.get("cards", []))
        bad_version = [c.get("task_id", "") for c in cards if c.get("registry_version") != expected_version]
        bad_split = [c.get("task_id", "") for c in cards if c.get("split") != expected_split]
        rows.append(
            {
                "item": "task_registry",
                "path": rel,
                "check": "card_registry_versions",
                "passed": not bad_version,
                "details": ",".join(map(str, bad_version[:10])),
            }
        )
        rows.append(
            {
                "item": "task_registry",
                "path": rel,
                "check": "card_splits",
                "passed": not bad_split,
                "details": ",".join(map(str, bad_split[:10])),
            }
        )

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)

    failures = [r for r in rows if not bool(r["passed"])]
    if failures:
        preview = "\n".join(f"{r['item']} {r['path']} {r['check']}: {r['details']}" for r in failures[:20])
        raise SystemExit(f"Release-contract validation failed:\n{preview}")

    print(f"Validated release contract: {contract_path}")
    print(f"checks: {len(rows)}")
    print(f"wrote: {out}")


if __name__ == "__main__":
    main()
