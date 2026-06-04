#!/usr/bin/env python3
"""Validate ClaimTransfer-Bench task-card files.

The workshop artifact keeps task cards lightweight, but they still need a
machine-checkable contract: formula metadata, support labels, legal claims, and
official scorers are declared before any adapter output is scored.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_CARD_FIELDS = {
    "task_id",
    "task_family",
    "split",
    "registry_version",
    "formula",
    "covariates",
    "dimension",
    "samples",
    "support",
    "claim_specification",
    "seed_policy",
    "stress_tags",
}

ALLOWED_SPLITS = {"public", "hidden", "hidden_template", "private"}
ALLOWED_PREDICATES = {
    "mse_lt",
    "rank1",
    "rank_at_budget",
    "top_m_contains_all",
    "contains_all",
    "binary_true",
    "stress_card",
    "value_le",
    "value_ge",
    "exact_string_match",
}


def to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "No rows."
    table = df.astype(str)
    cols = list(table.columns)
    widths = [
        max(len(c), *(len(str(v)) for v in table[c].tolist()))
        for c in cols
    ]

    def row(vals: list[str]) -> str:
        return "| " + " | ".join(str(v).ljust(widths[i]) for i, v in enumerate(vals)) + " |"

    return "\n".join(
        [
            row(cols),
            "| " + " | ".join("-" * w for w in widths) + " |",
            *(row([str(v) for v in values]) for values in table[cols].values.tolist()),
        ]
    )


def load_cards(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "cards" in data:
        cards = data["cards"]
    else:
        cards = [data]
    if not isinstance(cards, list):
        raise ValueError(f"{path}: expected a card object or a list under 'cards'")
    return cards


def validate_registry(path: Path, cards: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "cards" in data:
        if "version" not in data:
            errors.append(f"{path.name}: registry missing version")
        if "schema_version" not in data:
            errors.append(f"{path.name}: registry missing schema_version")
        split = data.get("split")
        if split is not None and split not in ALLOWED_SPLITS:
            errors.append(f"{path.name}: invalid registry split {split!r}")
    ids = [str(c.get("task_id", "")) for c in cards]
    dupes = sorted({x for x in ids if ids.count(x) > 1})
    if dupes:
        errors.append(f"{path.name}: duplicate task_id values: {dupes}")
    return errors


def validate_card(card: dict[str, Any], source: Path) -> list[str]:
    errors: list[str] = []
    missing = sorted(REQUIRED_CARD_FIELDS - set(card))
    if missing:
        errors.append(f"missing required fields: {missing}")

    task_id = str(card.get("task_id", ""))
    if not task_id:
        errors.append("empty task_id")

    split = card.get("split")
    if split not in ALLOWED_SPLITS:
        errors.append(f"split must be one of {sorted(ALLOWED_SPLITS)}, got {split!r}")

    if not isinstance(card.get("covariates", {}), dict):
        errors.append("covariates must be an object")

    if not isinstance(card.get("seed_policy", {}), dict):
        errors.append("seed_policy must be an object")

    if not isinstance(card.get("stress_tags", []), list):
        errors.append("stress_tags must be a list")

    support = card.get("support", [])
    if not isinstance(support, list) or not all(isinstance(v, int) for v in support):
        errors.append("support must be a list of integer variable indices")

    claim_spec = card.get("claim_specification", {})
    if not isinstance(claim_spec, dict):
        errors.append("claim_specification must be an object")
        return errors

    has_claim = False
    for key, claims in claim_spec.items():
        if not key.endswith("_claims"):
            continue
        if not isinstance(claims, list):
            errors.append(f"{key} must be a list")
            continue
        has_claim = has_claim or bool(claims)
        for i, claim in enumerate(claims):
            if not isinstance(claim, dict):
                errors.append(f"{key}[{i}] must be an object")
                continue
            for field in ("claim_type", "target", "predicate"):
                if field not in claim:
                    errors.append(f"{key}[{i}] missing {field}")
            if claim.get("predicate") not in ALLOWED_PREDICATES:
                errors.append(f"{key}[{i}] has unknown predicate {claim.get('predicate')!r}")
            if "claim_id" not in claim:
                errors.append(f"{key}[{i}] missing claim_id")
            if "official_scorer" not in claim:
                errors.append(f"{key}[{i}] missing official_scorer")
            if claim.get("claim_type") == "pair" and "official_scorer" not in claim:
                errors.append(f"{key}[{i}] pair claim missing official_scorer")
            target = claim.get("target")
            if claim.get("claim_type") in {"support", "endpoints"}:
                if not isinstance(target, list) or not all(isinstance(v, int) for v in target):
                    errors.append(f"{key}[{i}] target must be integer list")
            if claim.get("claim_type") == "pair":
                if not isinstance(target, list) or len(target) != 2:
                    errors.append(f"{key}[{i}] pair target must be length-2 integer list")

    if not has_claim:
        errors.append("no legal claims declared")

    if card.get("dimension") is not None and support:
        try:
            d = int(card["dimension"])
            bad = [v for v in support if v < 0 or v >= d]
            if bad:
                errors.append(f"support indices outside dimension {d}: {bad}")
        except Exception:
            errors.append("dimension must be an integer")

    return [f"{source.name}:{task_id}: {e}" for e in errors]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", default="task_cards")
    parser.add_argument("--out-dir", default="score_reports")
    args = parser.parse_args()

    task_dir = Path(args.task_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    all_errors: list[str] = []
    for path in sorted(task_dir.glob("*.json")):
        if path.name.endswith("_schema.json"):
            continue
        cards = load_cards(path)
        all_errors.extend(validate_registry(path, cards))
        for card in cards:
            errors = validate_card(card, path)
            all_errors.extend(errors)
            claim_spec = card.get("claim_specification", {})
            n_claims = sum(
                len(v) for k, v in claim_spec.items() if k.endswith("_claims") and isinstance(v, list)
            )
            rows.append(
                {
                    "source": str(path),
                    "registry_version": card.get("registry_version", ""),
                    "split": card.get("split", ""),
                    "task_id": card.get("task_id", ""),
                    "task_family": card.get("task_family", ""),
                    "dimension": card.get("dimension", ""),
                    "samples": card.get("samples", ""),
                    "support_size": len(card.get("support", []) or []),
                    "num_declared_claims": n_claims,
                    "valid": not errors,
                    "errors": "; ".join(errors),
                }
            )

    report = pd.DataFrame(rows)
    report.to_csv(out_dir / "task_card_validation.csv", index=False)
    (out_dir / "task_card_validation.md").write_text(
        "# Task-card validation\n\n"
        + to_markdown(report)
        + "\n",
        encoding="utf-8",
    )

    if all_errors:
        print("\n".join(all_errors))
        raise SystemExit(1)
    print(f"Validated {len(rows)} task cards.")
    print(f"Wrote {out_dir / 'task_card_validation.csv'}")


if __name__ == "__main__":
    main()
