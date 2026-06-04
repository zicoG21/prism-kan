#!/usr/bin/env python3
"""Validate ClaimTransfer hidden/private offline bundle leakage rules."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Missing hidden-bundle file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def iter_claims(card: dict[str, Any]):
    specs = card.get("claim_specification", {})
    for value in specs.values():
        if isinstance(value, list):
            for claim in value:
                if isinstance(claim, dict):
                    yield claim


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", default="artifacts/private_hidden")
    parser.add_argument("--out", default="score_reports/hidden_bundle_validation.csv")
    args = parser.parse_args()

    bundle_dir = ROOT / args.bundle_dir
    participant = load_json(bundle_dir / "claimtransfer_v0_hidden_participant.json")
    private = load_json(bundle_dir / "claimtransfer_v0_hidden_private_scoring.json")
    seed_manifest = bundle_dir / "private_seed_manifest.csv"
    if not seed_manifest.exists():
        raise SystemExit(f"Missing hidden-bundle file: {seed_manifest}")

    rows: list[dict[str, Any]] = []

    participant_cards = participant.get("cards", [])
    private_cards = private.get("cards", [])
    if len(participant_cards) != len(private_cards):
        raise SystemExit("Participant and private hidden registries have different card counts.")

    for card in participant_cards:
        task_id = card.get("task_id", "")
        checks = {
            "formula_withheld": card.get("formula") == "withheld",
            "support_withheld": card.get("support") == "withheld",
            "label_visibility_withheld": card.get("label_visibility") == "withheld_until_scoring",
            "no_private_seed_start": "private_seed_start" not in card,
            "no_private_seed_count": "private_seed_count" not in card,
            "seed_block_private": card.get("seed_policy", {}).get("train_test_seed_block") == "private",
        }
        for claim in iter_claims(card):
            claim_id = str(claim.get("claim_id", "claim"))
            rows.append(
                {
                    "view": "participant",
                    "task_id": task_id,
                    "check": f"claim_target_withheld:{claim_id}",
                    "passed": claim.get("target") == "withheld",
                }
            )
        for check, passed in checks.items():
            rows.append({"view": "participant", "task_id": task_id, "check": check, "passed": bool(passed)})

    for card in private_cards:
        task_id = card.get("task_id", "")
        checks = {
            "private_seed_start_present": "private_seed_start" in card,
            "private_seed_count_present": "private_seed_count" in card,
            "private_seed_policy": card.get("seed_policy", {}).get("hidden_seed_policy")
            == "private labels and seeds kept by benchmark maintainer",
        }
        for check, passed in checks.items():
            rows.append({"view": "private", "task_id": task_id, "check": check, "passed": bool(passed)})

    with seed_manifest.open(newline="", encoding="utf-8") as f:
        seed_rows = list(csv.DictReader(f))
    rows.append(
        {
            "view": "private",
            "task_id": "seed_manifest",
            "check": "seed_manifest_rows_match_cards",
            "passed": len(seed_rows) == len(private_cards),
        }
    )

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["view", "task_id", "check", "passed"])
        writer.writeheader()
        writer.writerows(rows)

    failures = [r for r in rows if not r["passed"]]
    if failures:
        preview = "\n".join(f"{r['view']} {r['task_id']} {r['check']}" for r in failures[:20])
        raise SystemExit(f"Hidden bundle validation failed:\n{preview}")

    print(f"Validated hidden/private bundle: {bundle_dir}")
    print(f"checks: {len(rows)}")
    print(f"wrote: {out}")


if __name__ == "__main__":
    main()
