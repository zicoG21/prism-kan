#!/usr/bin/env python3
"""Materialize a private/offline hidden-evaluation bundle.

The public repository contains hidden *templates*.  This script creates an
offline bundle for maintainers:

- a participant view with labels, pair targets, and seed blocks withheld;
- a private scoring registry that keeps labels and seed blocks;
- a seed manifest for the maintainer;
- a README describing the offline workflow.

The generated directory is ignored by git and is meant to live outside the
public release when used for leaderboard-style evaluation.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def scrub_claim(claim: dict[str, Any]) -> dict[str, Any]:
    out = dict(claim)
    if "target" in out:
        out["target"] = "withheld"
    if "claim_id" in out:
        out["claim_id"] = str(out["claim_id"]).split(":")[0] + ":withheld"
    return out


def scrub_card(card: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(card)
    out["split"] = "hidden"
    out["registry_version"] = "claimtransfer_v0_hidden_participant"
    out["formula"] = "withheld"
    out["support"] = "withheld"
    out["label_visibility"] = "withheld_until_scoring"
    out["audit_purpose"] = "participant-facing hidden card; labels and targets withheld"
    out["seed_policy"] = {
        "train_test_seed_block": "private",
        "adapter_seed_block": "participant_declared",
        "hidden_seed_policy": "private seeds and labels are withheld until scoring",
    }
    specs = out.get("claim_specification", {})
    for key, value in list(specs.items()):
        if isinstance(value, list):
            specs[key] = [scrub_claim(c) if isinstance(c, dict) else c for c in value]
    return out


def private_card(card: dict[str, Any], seed_start: int, seed_count: int) -> dict[str, Any]:
    out = copy.deepcopy(card)
    out["split"] = "private"
    out["registry_version"] = "claimtransfer_v0_hidden_private"
    out["seed_policy"] = {
        **dict(out.get("seed_policy", {})),
        "train_test_seed_block": f"{seed_start}-{seed_start + seed_count - 1}",
        "adapter_seed_block": "participant_declared",
        "hidden_seed_policy": "private labels and seeds kept by benchmark maintainer",
    }
    out["private_seed_start"] = seed_start
    out["private_seed_count"] = seed_count
    return out


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", default="task_cards/claimtransfer_v0_hidden_template.json")
    parser.add_argument("--out-dir", default="artifacts/private_hidden")
    parser.add_argument("--seed-base", type=int, default=9000)
    parser.add_argument("--seed-count", type=int, default=12)
    args = parser.parse_args()

    template_path = ROOT / args.template
    template = json.loads(template_path.read_text(encoding="utf-8"))
    cards = list(template.get("cards", []))
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    participant_cards = []
    private_cards = []
    seed_rows = []
    for i, card in enumerate(cards):
        seed_start = args.seed_base + i * args.seed_count
        seed_stop = seed_start + args.seed_count - 1
        participant_cards.append(scrub_card(card))
        private_cards.append(private_card(card, seed_start, args.seed_count))
        seed_rows.append(
            {
                "task_id": card.get("task_id", ""),
                "task_family": card.get("task_family", ""),
                "private_seed_start": seed_start,
                "private_seed_stop": seed_stop,
                "seed_count": args.seed_count,
                "label_visibility": "private",
            }
        )

    created = datetime.now(timezone.utc).isoformat()
    participant = {
        "version": "claimtransfer_v0_hidden_participant",
        "schema_version": template.get("schema_version", ""),
        "split": "hidden",
        "created_at": created,
        "cards": participant_cards,
    }
    private = {
        "version": "claimtransfer_v0_hidden_private",
        "schema_version": template.get("schema_version", ""),
        "split": "private",
        "created_at": created,
        "cards": private_cards,
    }

    write_json(out_dir / "claimtransfer_v0_hidden_participant.json", participant)
    write_json(out_dir / "claimtransfer_v0_hidden_private_scoring.json", private)

    with (out_dir / "private_seed_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(seed_rows[0]))
        writer.writeheader()
        writer.writerows(seed_rows)

    (out_dir / "README.md").write_text(
        "# ClaimTransfer hidden/private offline bundle\n\n"
        "This directory is generated for maintainer-run hidden evaluation.  The\n"
        "participant JSON withholds formulas, labels, pair targets, and private\n"
        "seed blocks.  The private scoring JSON and seed manifest must remain\n"
        "outside the public repository until the evaluation closes.\n\n"
        "Scoring entry point:\n\n"
        "```bash\n"
        "python scripts/run_benchmark.py --mode hidden --hidden-input path/to/private_submission.csv\n"
        "```\n",
        encoding="utf-8",
    )

    print(f"Wrote hidden participant view: {out_dir / 'claimtransfer_v0_hidden_participant.json'}")
    print(f"Wrote private scoring registry: {out_dir / 'claimtransfer_v0_hidden_private_scoring.json'}")
    print(f"Wrote private seed manifest: {out_dir / 'private_seed_manifest.csv'}")


if __name__ == "__main__":
    main()
