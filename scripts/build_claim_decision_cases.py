#!/usr/bin/env python3
"""Build compact claim-decision cases from existing stage-discordance summaries.

This script is intentionally lightweight: it does not retrain models.  It takes
the already-generated horizontal evidence table and turns it into a small
reviewer-facing decision table.  The table is meant to answer a practical
question:

    Given one apparent KAN structural claim, which workflow object actually
    supports it, and how should the claim be worded?

The output is local-note material by default.  It can later be copied into the
paper or appendix if the seed-aligned Great Lakes records are not ready.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


def parse_count(value: object) -> tuple[int, int]:
    text = str(value)
    m = re.match(r"\s*(\d+)\s*/\s*(\d+)\s*$", text)
    if not m:
        return (0, 0)
    return int(m.group(1)), int(m.group(2))


def rate_from_count(value: object) -> float:
    num, den = parse_count(value)
    return float(num / den) if den else float("nan")


def fmt_rate_count(count: str, rate: float) -> str:
    return f"{count} ({rate:.2f})"


def decision_for_row(row: pd.Series) -> tuple[str, str, str, str]:
    full = float(row["full_rate"])
    readout = float(row["readout_rate"])
    prune = rate_from_count(row["prune_endpoints"])
    support_med = float(row["prune_support_med"])
    full_margin = float(row["full_margin"])
    readout_margin = float(row["readout_margin"])

    if full >= 0.80 and readout >= 0.90 and prune >= 0.80 and support_med <= 4.5:
        return (
            "accept controlled recovery",
            "full-model reliance, exposed endpoints, and sparse pruning agree",
            "support+pair claim",
            "all audited objects support the same structural claim with positive margins",
        )

    if full < 0.20 and readout < 0.20 and prune < 0.30:
        return (
            "reject structure claim",
            "model reliance, readout surfacing, and pruning all fail",
            "prediction-only or no structural claim",
            "the true pair/endpoints are not supported by the audited workflow objects",
        )

    if readout >= 0.90 and full < 0.70:
        return (
            "revise claim to endpoint surfacing",
            "readout surfaces endpoints before fitted-function pair reliance stabilizes",
            "endpoint-readout claim",
            "the exposed readout supports endpoints, but full-KAN pair rank is mixed",
        )

    if readout >= 0.90 and prune < 0.80:
        return (
            "revise claim to pre-pruning evidence",
            "readout and/or full model support structure, sparse extraction is weaker",
            "pre-pruning support/pair claim",
            "downstream pruning does not retain endpoints often enough for a sparse workflow claim",
        )

    if full >= 0.60 and readout >= 0.90 and prune >= 0.60:
        return (
            "partial support with fragile extraction",
            "several stages support the claim but at least one margin or pruning row is weak",
            "qualified structure claim",
            "positive evidence exists, but support provenance is threshold- or margin-sensitive",
        )

    if full_margin > 0 and readout_margin > 0:
        return (
            "boundary case",
            "positive margins but low or uneven success counts",
            "margin-qualified claim",
            "rank/margin fields should be reported before asserting recovery",
        )

    return (
        "boundary or reject",
        "evidence objects disagree without a stable positive margin",
        "no unqualified structure claim",
        "the stage record identifies the broken link rather than certifying recovery",
    )


def build_cases(horizontal: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in horizontal.iterrows():
        prune_rate = rate_from_count(row["prune_endpoints"])
        decision, first_link, allowed_claim, rationale = decision_for_row(row)
        rates = [
            float(row["full_rate"]),
            float(row["readout_rate"]),
            prune_rate,
        ]
        rows.append(
            {
                "condition": row["condition"],
                "allowed_claim": allowed_claim,
                "stage_record_decision": decision,
                "first_missing_or_fragile_link": first_link,
                "full_pair_rank1": fmt_rate_count(row["full_rank1"], float(row["full_rate"])),
                "full_pair_margin": float(row["full_margin"]),
                "readout_endpoints_at4": fmt_rate_count(row["readout_endpoints"], float(row["readout_rate"])),
                "readout_endpoint_margin": float(row["readout_margin"]),
                "prune_endpoints": fmt_rate_count(row["prune_endpoints"], prune_rate),
                "prune_median_support": float(row["prune_support_med"]),
                "min_stage_rate": min(rates),
                "rationale": rationale,
            }
        )
    return pd.DataFrame(rows)


def dataframe_to_markdown(df: pd.DataFrame, index: bool = False) -> str:
    """Render markdown tables without pandas' optional tabulate dependency."""

    if df.empty:
        return ""
    table = df.reset_index() if index else df.copy()
    table = table.astype(str)
    headers = list(table.columns)
    rows = table.values.tolist()
    widths = [
        max(len(str(header)), *(len(str(row[i])) for row in rows))
        for i, header in enumerate(headers)
    ]

    def fmt_row(values: list[object]) -> str:
        cells = [str(value).ljust(widths[i]) for i, value in enumerate(values)]
        return "| " + " | ".join(cells) + " |"

    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    return "\n".join([fmt_row(headers), separator, *(fmt_row(row) for row in rows)])


def write_markdown(df: pd.DataFrame, out: Path) -> None:
    cols = [
        "condition",
        "allowed_claim",
        "stage_record_decision",
        "full_pair_rank1",
        "readout_endpoints_at4",
        "prune_endpoints",
        "prune_median_support",
        "rationale",
    ]
    lines = [
        "# Claim-Decision Cases from the Horizontal Stage Record",
        "",
        "This table is built from the existing horizontal evidence-object comparison.",
        "It is not a new experiment; it converts the same evidence into reviewer-facing",
        "claim decisions.",
        "",
        dataframe_to_markdown(df[cols], index=False),
        "",
        "Reading: an apparent formula-recovery claim is accepted only when fitted-function",
        "pair reliance, exposed endpoint surfacing, and sparse downstream extraction agree.",
        "Rows with high readout evidence but mixed full-model or pruning evidence should be",
        "worded as endpoint-surfacing or pre-pruning claims.",
        "",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")


def latex_escape(text: object) -> str:
    s = str(text)
    return (
        s.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("$", "\\$")
        .replace("#", "\\#")
        .replace("_", "\\_")
        .replace("{", "\\{")
        .replace("}", "\\}")
    )


def write_latex(df: pd.DataFrame, out: Path) -> None:
    cols = [
        ("condition", "Setting"),
        ("stage_record_decision", "Decision"),
        ("full_pair_rank1", "Full pair"),
        ("readout_endpoints_at4", "Readout endpoints"),
        ("prune_endpoints", "Prune endpoints"),
    ]
    lines = [
        "% Generated by scripts/build_claim_decision_cases.py",
        "\\begin{tabular}{lllll}",
        "\\toprule",
        " & ".join(label for _, label in cols) + r" \\",
        "\\midrule",
    ]
    for _, row in df.iterrows():
        lines.append(" & ".join(latex_escape(row[c]) for c, _ in cols) + r" \\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    out.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--horizontal",
        default="local_notes/generated/horizontal_evidence_table_20260531.csv",
    )
    parser.add_argument(
        "--out_csv",
        default="local_notes/generated/claim_decision_cases_20260531.csv",
    )
    parser.add_argument(
        "--out_md",
        default="local_notes/claim_decision_cases_20260531.md",
    )
    parser.add_argument(
        "--out_tex",
        default="local_notes/generated/claim_decision_cases_20260531.tex",
    )
    args = parser.parse_args()

    horizontal = pd.read_csv(args.horizontal)
    cases = build_cases(horizontal)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    cases.to_csv(out_csv, index=False)

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    write_markdown(cases, out_md)

    out_tex = Path(args.out_tex)
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    write_latex(cases, out_tex)

    print(f"Wrote {out_csv}")
    print(f"Wrote {out_md}")
    print(f"Wrote {out_tex}")
    print(cases[["condition", "stage_record_decision", "allowed_claim", "min_stage_rate"]].to_string(index=False))


if __name__ == "__main__":
    main()
