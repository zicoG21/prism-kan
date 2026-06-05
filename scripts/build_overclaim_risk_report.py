#!/usr/bin/env python3
"""Build Overclaim Risk summaries from official ClaimTransfer rows.

An overclaim event is an unsupported transfer: within the same task, method,
and seed, a weaker source claim passes but the stronger target claim it is
often used to imply does not pass.  This script turns the paper's "ordinary
reporting would overclaim" examples into an auditable score-report table.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

GROUP_COLS = [
    "registry_version",
    "split",
    "task_family",
    "task_id",
    "adapter_family",
    "adapter",
    "seed",
]


@dataclass(frozen=True)
class RowFilter:
    claim_types: set[str] | None = None
    evidence_objects: set[str] | None = None
    scorers: set[str] | None = None
    predicates: set[str] | None = None

    def mask(self, df: pd.DataFrame) -> pd.Series:
        mask = pd.Series(True, index=df.index)
        if self.claim_types is not None:
            mask &= df["claim_type"].astype(str).isin(self.claim_types)
        if self.evidence_objects is not None:
            mask &= df["evidence_object"].astype(str).isin(self.evidence_objects)
        if self.scorers is not None:
            mask &= df["scorer"].astype(str).isin(self.scorers)
        if self.predicates is not None:
            mask &= df["predicate"].astype(str).isin(self.predicates)
        return mask


@dataclass(frozen=True)
class TransferSpec:
    transfer_id: str
    transfer: str
    source: RowFilter
    target: RowFilter
    target_rule: Literal["any_pass", "all_pass"]
    source_claim: str
    target_claim: str
    overclaim_meaning: str


SPECS = [
    TransferSpec(
        transfer_id="prediction_to_pair",
        transfer="prediction -> pair",
        source=RowFilter(claim_types={"prediction"}),
        target=RowFilter(claim_types={"pair"}),
        target_rule="any_pass",
        source_claim="predictive adequacy",
        target_claim="verified or fitted pair recovery",
        overclaim_meaning="low error is used to imply interaction recovery",
    ),
    TransferSpec(
        transfer_id="candidate_to_pair",
        transfer="candidate -> pair",
        source=RowFilter(claim_types={"candidate_pair"}),
        target=RowFilter(claim_types={"pair"}),
        target_rule="any_pass",
        source_claim="true pair appears in a candidate set",
        target_claim="pair verifier passes",
        overclaim_meaning="screening/proposal is used to imply verified pair recovery",
    ),
    TransferSpec(
        transfer_id="symbolic_status_to_expression_quality",
        transfer="symbolic-status -> expression quality",
        source=RowFilter(claim_types={"symbolic_status"}),
        target=RowFilter(claim_types={"symbolic_operator_recall", "symbolic_complexity"}),
        target_rule="all_pass",
        source_claim="a symbolic expression/status is returned",
        target_claim="operator recall and complexity constraints pass",
        overclaim_meaning="expression existence is used to imply formula quality",
    ),
    TransferSpec(
        transfer_id="fitted_pair_to_readout",
        transfer="fitted pair -> readout endpoints",
        source=RowFilter(claim_types={"pair"}, evidence_objects={"full_function"}),
        target=RowFilter(claim_types={"endpoints"}, evidence_objects={"exposed_readout"}),
        target_rule="any_pass",
        source_claim="full fitted-function pair passes",
        target_claim="inspectable readout endpoints pass",
        overclaim_meaning="model reliance is used to imply inspectable recovery",
    ),
    TransferSpec(
        transfer_id="fitted_pair_to_pruning",
        transfer="fitted pair -> pruning endpoints",
        source=RowFilter(claim_types={"pair"}, evidence_objects={"full_function"}),
        target=RowFilter(claim_types={"endpoints"}, evidence_objects={"pruning"}),
        target_rule="any_pass",
        source_claim="full fitted-function pair passes",
        target_claim="pruned endpoint support passes",
        overclaim_meaning="fitted reliance is used to imply extraction recovery",
    ),
    TransferSpec(
        transfer_id="support_to_prediction",
        transfer="support -> prediction",
        source=RowFilter(claim_types={"support"}),
        target=RowFilter(claim_types={"prediction"}),
        target_rule="any_pass",
        source_claim="declared support is contained in selected variables",
        target_claim="predictive adequacy passes",
        overclaim_meaning="structural support is used to imply a useful fit",
    ),
]


def pass_values(rows: pd.DataFrame) -> pd.Series:
    return pd.to_numeric(rows["pass"], errors="coerce").dropna()


def row_pass(rows: pd.DataFrame, rule: Literal["any_pass", "all_pass"]) -> float:
    vals = pass_values(rows)
    if vals.empty:
        return float("nan")
    if rule == "all_pass":
        return float((vals >= 0.5).all())
    return float((vals >= 0.5).any())


def wilson(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    if trials <= 0:
        return (float("nan"), float("nan"))
    phat = successes / trials
    denom = 1 + z * z / trials
    center = (phat + z * z / (2 * trials)) / denom
    half = z * math.sqrt((phat * (1 - phat) + z * z / (4 * trials)) / trials) / denom
    return max(0.0, center - half), min(1.0, center + half)


def aggregate_claim(rows: pd.DataFrame, rule: Literal["any_pass", "all_pass"], prefix: str) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame(columns=[*GROUP_COLS, f"{prefix}_rows", f"{prefix}_pass"])
    block = rows.copy()
    block["pass_num"] = pd.to_numeric(block["pass"], errors="coerce")
    block = block.dropna(subset=["pass_num"])
    if block.empty:
        return pd.DataFrame(columns=[*GROUP_COLS, f"{prefix}_rows", f"{prefix}_pass"])
    if rule == "all_pass":
        agg_pass = "min"
    else:
        agg_pass = "max"
    out = (
        block.groupby(GROUP_COLS, dropna=False)
        .agg(**{f"{prefix}_rows": ("pass_num", "size"), f"{prefix}_pass": ("pass_num", agg_pass)})
        .reset_index()
    )
    out[f"{prefix}_pass"] = (out[f"{prefix}_pass"] >= 0.5).astype(float)
    return out


def build_detail(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for spec in SPECS:
        source = aggregate_claim(df[spec.source.mask(df)], "any_pass", "source")
        target = aggregate_claim(df[spec.target.mask(df)], spec.target_rule, "target")
        if source.empty or target.empty:
            continue
        merged = source.merge(target, on=GROUP_COLS, how="inner")
        if merged.empty:
            continue
        merged["transfer_id"] = spec.transfer_id
        merged["transfer"] = spec.transfer
        merged["source_claim"] = spec.source_claim
        merged["target_claim"] = spec.target_claim
        merged["overclaim_meaning"] = spec.overclaim_meaning
        merged["unsupported_transfer"] = (
            (merged["source_pass"] >= 0.5) & (merged["target_pass"] < 0.5)
        ).astype(float)
        rows.extend(merged.to_dict("records"))
    return pd.DataFrame(rows)


def summarize(detail: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame()
    block = detail.copy()
    grouped = []
    for key, group in block.groupby(group_cols, dropna=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        source_passes = int(group["source_pass"].sum())
        unsupported = int(group["unsupported_transfer"].sum())
        eligible = int(len(group))
        risk = unsupported / source_passes if source_passes else float("nan")
        low, high = wilson(unsupported, source_passes)
        grouped.append(
            {
                **dict(zip(group_cols, key_tuple)),
                "eligible_pairs": eligible,
                "source_passes": source_passes,
                "target_failures_given_source_pass": unsupported,
                "overclaim_risk": risk,
                "wilson_low": low,
                "wilson_high": high,
            }
        )
    return pd.DataFrame(grouped).sort_values(group_cols)


def markdown_table(df: pd.DataFrame, max_rows: int = 40) -> str:
    if df.empty:
        return "No rows."
    table = df.head(max_rows).copy()
    for col in ["overclaim_risk", "wilson_low", "wilson_high"]:
        if col in table:
            table[col] = table[col].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
    table = table.fillna("").astype(str)
    cols = list(table.columns)
    widths = [max(len(col), *(len(v) for v in table[col].tolist())) for col in cols]

    def row(vals: list[str]) -> str:
        return "| " + " | ".join(vals[i].ljust(widths[i]) for i in range(len(vals))) + " |"

    lines = [row(cols), "| " + " | ".join("-" * w for w in widths) + " |"]
    lines.extend(row([str(v) for v in values]) for values in table[cols].values.tolist())
    if len(df) > max_rows:
        lines.append(f"\nShowing first {max_rows} of {len(df)} rows.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="claim_records/released_claim_records.csv")
    parser.add_argument("--out", default="score_reports/overclaim_risk_report.csv")
    parser.add_argument("--detail-out", default="score_reports/overclaim_risk_detail.csv")
    parser.add_argument("--by-adapter-out", default="score_reports/overclaim_risk_by_adapter.csv")
    args = parser.parse_args()

    src = ROOT / args.input
    if not src.exists():
        raise SystemExit(f"Missing claim records: {src}")
    df = pd.read_csv(src, low_memory=False)
    required = set(GROUP_COLS + ["claim_type", "evidence_object", "scorer", "predicate", "pass"])
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"Missing required columns in {src}: {missing}")

    detail = build_detail(df)
    summary = summarize(detail, ["transfer_id", "transfer", "source_claim", "target_claim", "overclaim_meaning"])
    by_adapter = summarize(
        detail,
        ["transfer_id", "transfer", "adapter_family", "adapter", "source_claim", "target_claim"],
    )

    out = ROOT / args.out
    detail_out = ROOT / args.detail_out
    by_adapter_out = ROOT / args.by_adapter_out
    out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out, index=False)
    detail.to_csv(detail_out, index=False)
    by_adapter.to_csv(by_adapter_out, index=False)
    out.with_suffix(".md").write_text(
        "# Overclaim Risk Report\n\n"
        "Overclaim risk is the conditional rate at which a source claim passes "
        "but the stronger target claim it is often used to imply fails.\n\n"
        + markdown_table(summary)
        + "\n",
        encoding="utf-8",
    )
    by_adapter_out.with_suffix(".md").write_text(
        "# Overclaim Risk by Adapter\n\n" + markdown_table(by_adapter, max_rows=80) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {out} ({len(summary)} rows)")
    print(f"Wrote {detail_out} ({len(detail)} rows)")
    print(f"Wrote {by_adapter_out} ({len(by_adapter)} rows)")


if __name__ == "__main__":
    main()
