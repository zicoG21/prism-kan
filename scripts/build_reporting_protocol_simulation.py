#!/usr/bin/env python3
"""Build ordinary-reporting protocol simulations from overclaim-risk rows."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

PROTOCOLS = [
    {
        "protocol": "Prediction-only report",
        "transfer_id": "prediction_to_pair",
        "ordinary_claim": "low error implies recovered interaction",
        "claimtransfer_reading": "prediction is an adequacy claim; pair recovery is scored separately",
    },
    {
        "protocol": "Candidate-screen report",
        "transfer_id": "candidate_to_pair",
        "ordinary_claim": "true pair in candidate set implies interaction found",
        "claimtransfer_reading": "candidate generation and verified pair recovery are different claims",
    },
    {
        "protocol": "Symbolic-return report",
        "transfer_id": "symbolic_status_to_expression_quality",
        "ordinary_claim": "returned symbolic expression implies formula quality",
        "claimtransfer_reading": "symbolic status, operator recall, and complexity are separate claims",
    },
    {
        "protocol": "Support-only report",
        "transfer_id": "support_to_prediction",
        "ordinary_claim": "selected variables imply useful recovered model",
        "claimtransfer_reading": "support recovery and predictive adequacy are different axes",
    },
    {
        "protocol": "Fitted-function report",
        "transfer_id": "fitted_pair_to_readout",
        "ordinary_claim": "fitted pair reliance implies inspectable recovered mechanism",
        "claimtransfer_reading": "full-function reliance does not automatically transfer to readout endpoints",
    },
    {
        "protocol": "Pruning/extraction report",
        "transfer_id": "fitted_pair_to_pruning",
        "ordinary_claim": "fitted pair reliance implies extracted sparse support",
        "claimtransfer_reading": "extraction must retain declared endpoints and support",
    },
]


def fmt_pct(x: float) -> str:
    return f"{100*x:.1f}%"


def main() -> None:
    risk_path = ROOT / "score_reports" / "overclaim_risk_report.csv"
    if not risk_path.exists():
        raise SystemExit(f"Missing {risk_path}. Run scripts/build_overclaim_risk_report.py first.")
    risk = pd.read_csv(risk_path).set_index("transfer_id")

    rows: list[dict[str, object]] = []
    for spec in PROTOCOLS:
        transfer_id = spec["transfer_id"]
        if transfer_id not in risk.index:
            continue
        row = risk.loc[transfer_id]
        source_passes = int(row["source_passes"])
        eligible = int(row["eligible_pairs"])
        failures = int(row["target_failures_given_source_pass"])
        apparent_success = source_passes / eligible if eligible else float("nan")
        unsupported_rate = failures / source_passes if source_passes else float("nan")
        rows.append(
            {
                "reporting_protocol": spec["protocol"],
                "ordinary_claim": spec["ordinary_claim"],
                "transfer_id": transfer_id,
                "apparent_successes": source_passes,
                "eligible_rows": eligible,
                "apparent_success_rate": apparent_success,
                "unsupported_structural_conclusions": failures,
                "unsupported_rate_among_apparent_successes": unsupported_rate,
                "claimtransfer_reading": spec["claimtransfer_reading"],
            }
        )

    total_source = sum(int(risk.loc[p["transfer_id"], "source_passes"]) for p in PROTOCOLS if p["transfer_id"] in risk.index)
    total_fail = sum(
        int(risk.loc[p["transfer_id"], "target_failures_given_source_pass"])
        for p in PROTOCOLS
        if p["transfer_id"] in risk.index
    )
    total_eligible = sum(int(risk.loc[p["transfer_id"], "eligible_pairs"]) for p in PROTOCOLS if p["transfer_id"] in risk.index)
    rows.append(
        {
            "reporting_protocol": "Untyped multi-metric report",
            "ordinary_claim": "passed source metrics can be summarized as recovered structure",
            "transfer_id": "pooled_untyped_transfer",
            "apparent_successes": total_source,
            "eligible_rows": total_eligible,
            "apparent_success_rate": total_source / total_eligible if total_eligible else float("nan"),
            "unsupported_structural_conclusions": total_fail,
            "unsupported_rate_among_apparent_successes": total_fail / total_source if total_source else float("nan"),
            "claimtransfer_reading": "ClaimTransfer keeps these as edge-specific risks instead of one recovered/not-recovered sentence",
        }
    )

    out = pd.DataFrame(rows)
    out_csv = ROOT / "score_reports" / "ordinary_reporting_protocol_simulation.csv"
    out_md = out_csv.with_suffix(".md")
    out.to_csv(out_csv, index=False)

    show = out.copy()
    show["apparent_success_rate"] = show["apparent_success_rate"].map(fmt_pct)
    show["unsupported_rate_among_apparent_successes"] = show["unsupported_rate_among_apparent_successes"].map(fmt_pct)
    out_md.write_text(
        "# Ordinary Reporting Protocol Simulation\n\n"
        "Each row simulates a common reporting shortcut. Apparent successes are source-claim passes; unsupported conclusions are target failures among those source passes.\n\n"
        + show.to_markdown(index=False)
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
