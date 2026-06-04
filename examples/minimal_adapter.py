#!/usr/bin/env python3
"""Write a tiny ClaimTransfer adapter-output submission.

This example does not train a model. It shows the shape expected from an
adapter: expose evidence rows, then let the official scorer recompute verdicts.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


COLUMNS = [
    "registry_version",
    "split",
    "task_id",
    "task_family",
    "adapter",
    "adapter_family",
    "source_kind",
    "source_file",
    "seed",
    "evidence_object",
    "claim_type",
    "target",
    "scorer",
    "predicate",
    "threshold",
    "budget",
    "rank",
    "margin",
    "raw_value",
    "selected_set",
    "candidate_set",
    "missing_reason",
    "runtime_seconds",
    "protocol",
]


ROWS = [
    {
        "task_id": "bilinear_d100_n1024",
        "task_family": "bilinear",
        "adapter": "ExampleSparse",
        "adapter_family": "example_sparse",
        "evidence_object": "prediction",
        "claim_type": "prediction",
        "target": "low_mse",
        "scorer": "mse",
        "predicate": "mse_lt",
        "threshold": "0.05",
        "raw_value": "0.012",
    },
    {
        "task_id": "bilinear_d100_n1024",
        "task_family": "bilinear",
        "adapter": "ExampleSparse",
        "adapter_family": "example_sparse",
        "evidence_object": "selected_support",
        "claim_type": "support",
        "target": "[0, 1, 2]",
        "scorer": "sparse_terms",
        "predicate": "contains_all",
        "selected_set": "[0, 1, 2]",
    },
    {
        "task_id": "bilinear_d100_n1024",
        "task_family": "bilinear",
        "adapter": "ExampleSparse",
        "adapter_family": "example_sparse",
        "evidence_object": "pair_terms",
        "claim_type": "pair",
        "target": "[(0, 1)]",
        "scorer": "sparse_terms",
        "predicate": "rank1",
        "rank": "1",
        "margin": "0.73",
    },
    {
        "task_id": "weak_centered_d100_n1024",
        "task_family": "weak_centered",
        "adapter": "ExamplePredictor",
        "adapter_family": "example_predictor",
        "evidence_object": "prediction",
        "claim_type": "prediction",
        "target": "low_mse",
        "scorer": "mse",
        "predicate": "mse_lt",
        "threshold": "0.05",
        "raw_value": "0.018",
    },
    {
        "task_id": "weak_centered_d100_n1024",
        "task_family": "weak_centered",
        "adapter": "ExamplePredictor",
        "adapter_family": "example_predictor",
        "evidence_object": "selected_support",
        "claim_type": "support",
        "target": "[0, 1, 2, 3]",
        "scorer": "feature_importance",
        "predicate": "contains_all",
        "selected_set": "[0, 1]",
    },
    {
        "task_id": "weak_centered_d100_n1024",
        "task_family": "weak_centered",
        "adapter": "ExamplePredictor",
        "adapter_family": "example_predictor",
        "evidence_object": "pair_scores",
        "claim_type": "pair",
        "target": "[(2, 3)]",
        "scorer": "feature_importance",
        "predicate": "rank1",
        "rank": "17",
        "margin": "-0.04",
    },
    {
        "registry_version": "claimtransfer_v1_scientific_templates",
        "split": "hidden_template",
        "task_id": "feynman_style_energy_hidden_template",
        "task_family": "scientific_expression",
        "adapter": "ExampleSymbolic",
        "adapter_family": "example_symbolic",
        "evidence_object": "symbolic_expression",
        "claim_type": "symbolic_operator_recall",
        "target": "plus,multiply,power",
        "scorer": "symbolic_expression_quality",
        "predicate": "operator_recall_ge",
        "threshold": "0.95",
        "selected_set": "['plus', 'multiply', 'power']",
    },
    {
        "registry_version": "claimtransfer_v1_scientific_templates",
        "split": "hidden_template",
        "task_id": "feynman_style_energy_hidden_template",
        "task_family": "scientific_expression",
        "adapter": "ExampleSymbolic",
        "adapter_family": "example_symbolic",
        "evidence_object": "symbolic_expression",
        "claim_type": "symbolic_complexity",
        "target": "max_complexity",
        "scorer": "symbolic_expression_quality",
        "predicate": "complexity_le",
        "threshold": "12",
        "raw_value": "7",
    },
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="examples/minimal_submission_generated.csv")
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for row in ROWS:
            full = {col: "" for col in COLUMNS}
            full.update(
                {
                    "registry_version": row.get("registry_version", "claimtransfer_v0_public"),
                    "split": row.get("split", "public"),
                    "source_kind": "example",
                    "source_file": str(out),
                    "seed": "0",
                    "runtime_seconds": "0.1",
                    "protocol": "minimal_adapter_example",
                }
            )
            full.update(row)
            writer.writerow(full)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
