#!/usr/bin/env python3
"""Emit a reusable structure-fidelity audit summary.

This is a lightweight paper-facing interface rather than an experiment runner.
It standardizes the columns expected from support/interaction recovery audits and
adds Wilson confidence intervals for success counts.  If no input CSV is given,
the script writes the counts used by the current workshop draft.

Example:
    python scripts/run_standard_audit_protocol.py \
        --out-dir results/workshop_review_tables/standard_audit_protocol

Input CSV format, if provided:
    label,protocol,metric,successes,trials,notes
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


DEFAULT_ROWS = [
    {
        "label": "same_data_c025_d100_n512",
        "protocol": "same_training_set_bootstrap",
        "metric": "exact_support",
        "successes": 0,
        "trials": 30,
        "notes": "KAN-FE, R=20, m=4, ANOVA pair scorer",
    },
    {
        "label": "same_data_c025_d100_n512",
        "protocol": "same_training_set_bootstrap",
        "metric": "endpoint_retention",
        "successes": 0,
        "trials": 30,
        "notes": "KAN-FE, R=20, m=4, ANOVA pair scorer",
    },
    {
        "label": "same_data_c025_d100_n512",
        "protocol": "same_training_set_bootstrap",
        "metric": "top1_pair",
        "successes": 0,
        "trials": 30,
        "notes": "KAN-FE, R=20, m=4, ANOVA pair scorer",
    },
    {
        "label": "same_data_c025_d100_n640",
        "protocol": "same_training_set_bootstrap",
        "metric": "top1_pair",
        "successes": 1,
        "trials": 30,
        "notes": "KAN-FE, R=20, m=4, ANOVA pair scorer",
    },
    {
        "label": "same_data_c025_d100_n768",
        "protocol": "same_training_set_bootstrap",
        "metric": "top1_pair",
        "successes": 6,
        "trials": 30,
        "notes": "KAN-FE, R=20, m=4, ANOVA pair scorer",
    },
    {
        "label": "same_data_c025_d100_n896",
        "protocol": "same_training_set_bootstrap",
        "metric": "top1_pair",
        "successes": 22,
        "trials": 30,
        "notes": "KAN-FE, R=20, m=4, ANOVA pair scorer",
    },
    {
        "label": "same_data_c025_d100_n1024",
        "protocol": "same_training_set_bootstrap",
        "metric": "top1_pair",
        "successes": 30,
        "trials": 30,
        "notes": "KAN-FE, R=20, m=4, ANOVA pair scorer",
    },
    {
        "label": "same_data_c025_d100_n1280",
        "protocol": "same_training_set_bootstrap",
        "metric": "top1_pair",
        "successes": 30,
        "trials": 30,
        "notes": "KAN-FE, R=20, m=4, ANOVA pair scorer",
    },
    {
        "label": "prediction_clean_highdim_c025_d1000_n2048",
        "protocol": "pooled_probe_diagnostic",
        "metric": "endpoint_retention",
        "successes": 0,
        "trials": 8,
        "notes": "not a finite-data comparison; fresh synthetic draw per probe",
    },
    {
        "label": "prediction_clean_highdim_c025_d1000_n2048",
        "protocol": "pooled_probe_diagnostic",
        "metric": "residual_screen_top1_pair",
        "successes": 6,
        "trials": 10,
        "notes": "residualized raw-product spline screen calibration",
    },
    {
        "label": "rff_hsic_c025_d100_n512",
        "protocol": "same_training_set_baseline",
        "metric": "top1_pair",
        "successes": 2,
        "trials": 50,
        "notes": "RFF-HSIC calibration baseline",
    },
    {
        "label": "rff_hsic_c025_d100_n896",
        "protocol": "same_training_set_baseline",
        "metric": "top1_pair",
        "successes": 49,
        "trials": 50,
        "notes": "RFF-HSIC calibration baseline",
    },
    {
        "label": "rff_hsic_c025_d100_n1024",
        "protocol": "same_training_set_baseline",
        "metric": "top1_pair",
        "successes": 50,
        "trials": 50,
        "notes": "RFF-HSIC calibration baseline",
    },
]


SCHEMA = {
    "required_input_columns": [
        "label",
        "protocol",
        "metric",
        "successes",
        "trials",
    ],
    "optional_input_columns": ["notes"],
    "output_columns": [
        "label",
        "protocol",
        "metric",
        "successes",
        "trials",
        "rate",
        "wilson95_low",
        "wilson95_high",
        "notes",
    ],
    "protocols": {
        "same_training_set_bootstrap": "finite-data KAN audit with a fixed train/test split per outer seed",
        "same_training_set_baseline": "finite-data non-KAN or external baseline under the same sample budget",
        "pooled_probe_diagnostic": "fresh synthetic data per probe; diagnostic only, not a fair finite-data comparison",
    },
    "metrics": {
        "prediction_mse": "test-set mean squared error on standardized targets",
        "active_variable_f1": "top-k active-variable F1 in a controlled task",
        "endpoint_retention": "success if all variables participating in true interactions are retained",
        "top1_pair": "success if the top-ranked candidate pair is a true pair",
        "exact_support": "success if the selected support equals the true active support",
    },
}


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if trials <= 0:
        raise ValueError("trials must be positive")
    phat = successes / trials
    denom = 1.0 + z * z / trials
    center = (phat + z * z / (2.0 * trials)) / denom
    radius = z * math.sqrt((phat * (1.0 - phat) + z * z / (4.0 * trials)) / trials) / denom
    return max(0.0, center - radius), min(1.0, center + radius)


def load_rows(path: Path | None) -> list[dict[str, str]]:
    if path is None:
        return [dict(row) for row in DEFAULT_ROWS]
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def normalize_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for row in rows:
        successes = int(float(row["successes"]))
        trials = int(float(row["trials"]))
        low, high = wilson_interval(successes, trials)
        normalized.append(
            {
                "label": row["label"],
                "protocol": row["protocol"],
                "metric": row["metric"],
                "successes": str(successes),
                "trials": str(trials),
                "rate": f"{successes / trials:.4f}",
                "wilson95_low": f"{low:.4f}",
                "wilson95_high": f"{high:.4f}",
                "notes": row.get("notes", ""),
            }
        )
    return normalized


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = SCHEMA["output_columns"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w") as handle:
        handle.write("# Structure-Fidelity Audit Summary\n\n")
        handle.write("| label | protocol | metric | count | rate | Wilson 95% CI |\n")
        handle.write("|---|---|---|---:|---:|---:|\n")
        for row in rows:
            handle.write(
                f"| {row['label']} | {row['protocol']} | {row['metric']} | "
                f"{row['successes']}/{row['trials']} | {row['rate']} | "
                f"[{row['wilson95_low']}, {row['wilson95_high']}] |\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts-csv", type=Path, default=None, help="Optional input count CSV.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/workshop_review_tables/standard_audit_protocol"),
        help="Directory for summary CSV, markdown report, and schema JSON.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = normalize_rows(load_rows(args.counts_csv))

    write_csv(args.out_dir / "audit_protocol_counts_with_ci.csv", rows)
    write_markdown(args.out_dir / "audit_protocol_summary.md", rows)
    with (args.out_dir / "audit_protocol_schema.json").open("w") as handle:
        json.dump(SCHEMA, handle, indent=2, sort_keys=True)

    print(f"Wrote {len(rows)} rows to {args.out_dir}")


if __name__ == "__main__":
    main()
