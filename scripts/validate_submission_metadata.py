#!/usr/bin/env python3
"""Validate ClaimTransfer submission metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED = {
    "submission_id",
    "method_name",
    "adapter_family",
    "method_description",
    "native_outputs",
    "tuning_policy",
    "compute_budget",
    "missing_field_policy",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metadata_json")
    args = parser.parse_args()

    path = Path(args.metadata_json)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit("submission metadata must be a JSON object")

    missing = sorted(REQUIRED - set(data))
    if missing:
        raise SystemExit(f"{path}: missing required fields: {missing}")

    if not isinstance(data.get("native_outputs"), list) or not data["native_outputs"]:
        raise SystemExit(f"{path}: native_outputs must be a non-empty list")

    if not isinstance(data.get("compute_budget"), dict):
        raise SystemExit(f"{path}: compute_budget must be an object")

    print(f"Validated submission metadata: {path}")
    print(f"submission_id: {data.get('submission_id')}")
    print(f"method_name: {data.get('method_name')}")
    print(f"adapter_family: {data.get('adapter_family')}")


if __name__ == "__main__":
    main()
