#!/usr/bin/env python3
"""Validate the ClaimTransfer adapter-family registry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


REQUIRED = {
    "adapter_family",
    "native_outputs",
    "licensed_claim_types",
    "tuning_policy",
    "budget_policy",
    "missing_field_policy",
    "positive_control_requirement",
    "stress_card_requirement",
}


def to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "No rows."
    table = df.astype(str)
    cols = list(table.columns)
    widths = [max(len(c), *(len(v) for v in table[c].tolist())) for c in cols]

    def row(vals: list[str]) -> str:
        return "| " + " | ".join(vals[i].ljust(widths[i]) for i in range(len(vals))) + " |"

    return "\n".join(
        [
            row(cols),
            "| " + " | ".join("-" * w for w in widths) + " |",
            *(row([str(v) for v in values]) for values in table[cols].values.tolist()),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default="adapters/adapter_family_registry.json")
    parser.add_argument("--out-dir", default="score_reports")
    args = parser.parse_args()

    path = Path(args.registry)
    data = json.loads(path.read_text(encoding="utf-8"))
    families = data.get("families", [])
    if not isinstance(families, list) or not families:
        raise SystemExit(f"{path}: expected a non-empty families list")

    rows = []
    errors = []
    seen = set()
    for i, family in enumerate(families):
        if not isinstance(family, dict):
            errors.append(f"families[{i}] must be an object")
            continue
        name = str(family.get("adapter_family", ""))
        missing = sorted(REQUIRED - set(family))
        if missing:
            errors.append(f"{name or i}: missing fields {missing}")
        if name in seen:
            errors.append(f"duplicate adapter_family {name!r}")
        seen.add(name)
        for list_field in ("native_outputs", "licensed_claim_types"):
            value = family.get(list_field, [])
            if not isinstance(value, list) or not value:
                errors.append(f"{name}: {list_field} must be a non-empty list")
        rows.append(
            {
                "adapter_family": name,
                "native_outputs": len(family.get("native_outputs", []) or []),
                "licensed_claim_types": len(family.get("licensed_claim_types", []) or []),
                "has_tuning_policy": bool(family.get("tuning_policy")),
                "has_budget_policy": bool(family.get("budget_policy")),
                "has_missing_field_policy": bool(family.get("missing_field_policy")),
                "valid": not missing,
            }
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = pd.DataFrame(rows)
    report.to_csv(out_dir / "adapter_family_validation.csv", index=False)
    (out_dir / "adapter_family_validation.md").write_text(
        "# Adapter-family validation\n\n" + to_markdown(report) + "\n",
        encoding="utf-8",
    )

    if errors:
        print("\n".join(errors))
        raise SystemExit(1)
    print(f"Validated {len(rows)} adapter families.")
    print(f"Wrote {out_dir / 'adapter_family_validation.csv'}")


if __name__ == "__main__":
    main()
