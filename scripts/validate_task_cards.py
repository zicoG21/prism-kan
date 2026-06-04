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
    "exact_expression_match",
    "operator_recall_ge",
    "complexity_le",
    "coefficient_error_le",
    "extrapolation_mse_lt",
}

ALLOWED_CLAIM_TYPES = {
    "prediction",
    "support",
    "endpoints",
    "pair",
    "candidate_pair",
    "symbolic_status",
    "symbolic_operator_recall",
    "symbolic_complexity",
}

CLAIM_BUCKET_TYPES = {
    "prediction_claims": {"prediction"},
    "support_claims": {"support"},
    "endpoint_claims": {"endpoints"},
    "pair_claims": {"pair", "candidate_pair"},
    "candidate_pair_claims": {"candidate_pair"},
    "symbolic_claims": {"symbolic_status", "symbolic_operator_recall", "symbolic_complexity"},
}

ALLOWED_OFFICIAL_SCORERS = {
    "mse",
    "ranked_support_or_endpoint_score",
    "functional_anova",
    "full_model_functional_anova",
    "scorer_stress",
    "symbolic_expression_quality",
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
    claim_ids: list[str] = []
    for card in cards:
        for key, claims in card.get("claim_specification", {}).items():
            if key.endswith("_claims") and isinstance(claims, list):
                claim_ids.extend(str(claim.get("claim_id", "")) for claim in claims if isinstance(claim, dict))
    claim_dupes = sorted({x for x in claim_ids if x and claim_ids.count(x) > 1})
    if claim_dupes:
        errors.append(f"{path.name}: duplicate claim_id values: {claim_dupes}")
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
    support_set = set(support) if isinstance(support, list) else set()
    dimension = None
    try:
        dimension = int(card["dimension"])
    except Exception:
        errors.append("dimension must be an integer")

    claim_spec = card.get("claim_specification", {})
    if not isinstance(claim_spec, dict):
        errors.append("claim_specification must be an object")
        return errors

    has_claim = False
    for key, claims in claim_spec.items():
        if not key.endswith("_claims"):
            errors.append(f"{key} must be named '*_claims'")
            continue
        if not isinstance(claims, list):
            errors.append(f"{key} must be a list")
            continue
        allowed_bucket_types = CLAIM_BUCKET_TYPES.get(key)
        if allowed_bucket_types is None:
            errors.append(f"{key} is not a recognized claim bucket")
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
            claim_type = str(claim.get("claim_type", ""))
            if claim_type not in ALLOWED_CLAIM_TYPES:
                errors.append(f"{key}[{i}] has unknown claim_type {claim_type!r}")
            if allowed_bucket_types is not None and claim_type and claim_type not in allowed_bucket_types:
                errors.append(
                    f"{key}[{i}] claim_type {claim_type!r} does not match bucket {sorted(allowed_bucket_types)}"
                )
            if "claim_id" not in claim:
                errors.append(f"{key}[{i}] missing claim_id")
            elif task_id and not str(claim["claim_id"]).startswith(task_id) and not card.get("split") == "hidden_template":
                errors.append(f"{key}[{i}] claim_id should start with task_id for public/private cards")
            if "official_scorer" not in claim:
                errors.append(f"{key}[{i}] missing official_scorer")
            elif claim.get("official_scorer") not in ALLOWED_OFFICIAL_SCORERS:
                errors.append(f"{key}[{i}] has unknown official_scorer {claim.get('official_scorer')!r}")
            if claim.get("claim_type") == "pair" and "official_scorer" not in claim:
                errors.append(f"{key}[{i}] pair claim missing official_scorer")
            target = claim.get("target")
            if claim.get("claim_type") in {"support", "endpoints"}:
                if not isinstance(target, list) or not all(isinstance(v, int) for v in target):
                    errors.append(f"{key}[{i}] target must be integer list")
                elif dimension is not None:
                    bad = [v for v in target if v < 0 or v >= dimension]
                    if bad:
                        errors.append(f"{key}[{i}] target indices outside dimension {dimension}: {bad}")
                if claim.get("claim_type") == "endpoints" and support_set and not set(target or []).issubset(support_set):
                    errors.append(f"{key}[{i}] endpoint target must be a subset of support")
            if claim.get("claim_type") == "pair":
                if not isinstance(target, list) or len(target) != 2 or not all(isinstance(v, int) for v in target):
                    errors.append(f"{key}[{i}] pair target must be length-2 integer list")
                else:
                    if target[0] == target[1]:
                        errors.append(f"{key}[{i}] pair target must contain two distinct variables")
                    if dimension is not None:
                        bad = [v for v in target if v < 0 or v >= dimension]
                        if bad:
                            errors.append(f"{key}[{i}] pair target indices outside dimension {dimension}: {bad}")
                    if support_set and not set(target).issubset(support_set):
                        errors.append(f"{key}[{i}] pair target must be a subset of support")
            if claim.get("predicate") == "top_m_contains_all":
                m = claim.get("m", claim.get("budget"))
                if not isinstance(m, int) or m <= 0:
                    errors.append(f"{key}[{i}] top_m_contains_all requires positive integer m")
            if claim.get("predicate") in {"operator_recall_ge", "complexity_le", "coefficient_error_le"}:
                threshold = claim.get("threshold")
                if not isinstance(threshold, (int, float)):
                    errors.append(f"{key}[{i}] {claim.get('predicate')} requires numeric threshold")

    if not has_claim:
        errors.append("no legal claims declared")

    if dimension is not None and support:
        try:
            d = dimension
            bad = [v for v in support if v < 0 or v >= d]
            if bad:
                errors.append(f"support indices outside dimension {d}: {bad}")
        except Exception:
            pass

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
            legal_claim_types = sorted(
                {
                    str(claim.get("claim_type", ""))
                    for k, v in claim_spec.items()
                    if k.endswith("_claims") and isinstance(v, list)
                    for claim in v
                    if isinstance(claim, dict) and claim.get("claim_type")
                }
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
                    "legal_claim_types": ",".join(legal_claim_types),
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
