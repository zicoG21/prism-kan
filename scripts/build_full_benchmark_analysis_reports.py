#!/usr/bin/env python3
"""Build higher-level ClaimTransfer-Bench analysis reports.

These reports are derived from official claim records.  They are intended for
the full benchmark paper: standard-formula-only overclaim signatures,
expression-quality breakdowns, public/split consistency summaries, and
ordinary-reporting interpretation flips.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

TRANSFER_ORDER = [
    "prediction_to_pair",
    "support_to_prediction",
    "candidate_to_pair",
    "symbolic_status_to_expression_quality",
    "fitted_pair_to_readout",
    "fitted_pair_to_pruning",
]

TRANSFER_LABELS = {
    "prediction_to_pair": "Prediction -> pair",
    "support_to_prediction": "Support -> prediction",
    "candidate_to_pair": "Candidate -> pair",
    "symbolic_status_to_expression_quality": "Symbolic status -> expression quality",
    "fitted_pair_to_readout": "Fitted pair -> readout",
    "fitted_pair_to_pruning": "Fitted pair -> pruning",
}


def markdown_table(df: pd.DataFrame, max_rows: int = 80) -> str:
    if df.empty:
        return "No rows."
    table = df.head(max_rows).fillna("").astype(str)
    cols = list(table.columns)
    widths = [max(len(col), *(len(v) for v in table[col].tolist())) for col in cols]

    def row(vals: list[str]) -> str:
        return "| " + " | ".join(vals[i].ljust(widths[i]) for i in range(len(vals))) + " |"

    lines = [row(cols), "| " + " | ".join("-" * w for w in widths) + " |"]
    lines.extend(row([str(v) for v in values]) for values in table[cols].values.tolist())
    if len(df) > max_rows:
        lines.append(f"\nShowing first {max_rows} of {len(df)} rows.")
    return "\n".join(lines)


def fmt_rate(value: object) -> str:
    try:
        val = float(value)
    except Exception:
        return ""
    if math.isnan(val):
        return ""
    return f"{100 * val:.1f}%"


def is_standard_formula(df: pd.DataFrame) -> pd.Series:
    registry = df.get("registry_version", pd.Series("", index=df.index)).astype(str)
    task_family = df.get("task_family", pd.Series("", index=df.index)).astype(str)
    task_id = df.get("task_id", pd.Series("", index=df.index)).astype(str)
    return (
        registry.eq("claimtransfer_v1_standard_formula_public")
        | task_family.str.startswith("standard_")
        | task_id.str.startswith("std_")
    )


def rate_summary(df: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    block = df.copy()
    block["pass_num"] = pd.to_numeric(block["pass"], errors="coerce")
    block = block.dropna(subset=["pass_num"])
    if block.empty:
        return pd.DataFrame(columns=[*groups, "rows", "trials", "successes", "pass_rate"])
    out = (
        block.groupby(groups, dropna=False)
        .agg(rows=("pass_num", "size"), trials=("pass_num", "size"), successes=("pass_num", "sum"))
        .reset_index()
    )
    out["successes"] = out["successes"].astype(int)
    out["pass_rate"] = out["successes"] / out["trials"].where(out["trials"] > 0)
    return out


def build_standard_signature(detail: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    std = detail[is_standard_formula(detail)].copy()
    if std.empty:
        signature = pd.DataFrame()
    else:
        rows = []
        for keys, group in std.groupby(["adapter_family", "adapter", "transfer_id"], dropna=False):
            adapter_family, adapter, transfer_id = keys
            source_passes = int(pd.to_numeric(group["source_pass"], errors="coerce").fillna(0).sum())
            failures = int(pd.to_numeric(group["unsupported_transfer"], errors="coerce").fillna(0).sum())
            risk = failures / source_passes if source_passes else float("nan")
            rows.append(
                {
                    "adapter_family": adapter_family,
                    "adapter": adapter,
                    "method": f"{adapter_family} / {adapter}",
                    "transfer_id": transfer_id,
                    "transfer": TRANSFER_LABELS.get(str(transfer_id), str(transfer_id)),
                    "eligible_rows": int(len(group)),
                    "source_passes": source_passes,
                    "target_failures_given_source_pass": failures,
                    "overclaim_risk": risk,
                }
            )
        long = pd.DataFrame(rows)
        risk = long.pivot_table(
            index=["adapter_family", "adapter", "method"],
            columns="transfer_id",
            values="overclaim_risk",
            aggfunc="mean",
        ).reset_index()
        counts = long.pivot_table(
            index=["adapter_family", "adapter", "method"],
            columns="transfer_id",
            values="source_passes",
            aggfunc="sum",
        ).reset_index()
        counts = counts.rename(
            columns={col: f"{col}_source_passes" for col in counts.columns if col not in {"adapter_family", "adapter", "method"}}
        )
        signature = risk.merge(counts, on=["adapter_family", "adapter", "method"], how="left")
        risk_cols = [col for col in TRANSFER_ORDER if col in signature.columns]
        signature["dominant_overclaim_edge"] = signature[risk_cols].idxmax(axis=1, skipna=True)
        signature["dominant_overclaim_risk"] = signature[risk_cols].max(axis=1, skipna=True)
        signature = signature.sort_values(["dominant_overclaim_risk", "adapter_family", "adapter"], ascending=[False, True, True])
        long.to_csv(out_dir / "standard_formula_overclaim_signature_long.csv", index=False)

    out = out_dir / "standard_formula_overclaim_signature_by_method.csv"
    signature.to_csv(out, index=False)
    show = signature.copy()
    for col in show.columns:
        if col in TRANSFER_ORDER or col == "dominant_overclaim_risk":
            show[col] = show[col].map(fmt_rate)
    out.with_suffix(".md").write_text(
        "# Standard-Formula-Only Overclaim Signature\n\n"
        "Rows are methods scored only on the 90 standard-formula settings.  "
        "Cells are conditional overclaim risks among rows where the source "
        "claim passes.\n\n"
        + markdown_table(show)
        + "\n",
        encoding="utf-8",
    )
    return signature


def build_expression_breakdown(claims: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    std = claims[is_standard_formula(claims)].copy()
    wanted = {
        "prediction": "prediction_pass",
        "support": "variable_support_recall",
        "endpoints": "endpoint_recall",
        "pair": "pair_term_recall",
        "symbolic_status": "symbolic_status",
        "symbolic_operator_recall": "operator_recall",
        "symbolic_complexity": "complexity_budget",
    }
    block = std[std["claim_type"].astype(str).isin(wanted)].copy()
    grouped = rate_summary(block, ["adapter_family", "adapter", "claim_type"])
    if grouped.empty:
        out = pd.DataFrame()
    else:
        grouped["metric"] = grouped["claim_type"].map(wanted)
        rates = grouped.pivot_table(
            index=["adapter_family", "adapter"],
            columns="metric",
            values="pass_rate",
            aggfunc="mean",
        ).reset_index()
        trials = grouped.pivot_table(
            index=["adapter_family", "adapter"],
            columns="metric",
            values="trials",
            aggfunc="sum",
        ).reset_index()
        trials = trials.rename(
            columns={col: f"{col}_trials" for col in trials.columns if col not in {"adapter_family", "adapter"}}
        )
        out = rates.merge(trials, on=["adapter_family", "adapter"], how="left")
        out["method"] = out["adapter_family"].astype(str) + " / " + out["adapter"].astype(str)
        expr_cols = [col for col in ["symbolic_status", "operator_recall", "complexity_budget"] if col in out.columns]
        if expr_cols:
            out["expression_quality_min"] = out[expr_cols].min(axis=1, skipna=True)
        out = out.sort_values(["adapter_family", "adapter"])

    path = out_dir / "standard_formula_expression_quality_breakdown.csv"
    out.to_csv(path, index=False)
    show = out.copy()
    for col in show.columns:
        if not col.endswith("_trials") and col not in {"adapter_family", "adapter", "method"}:
            show[col] = show[col].map(fmt_rate)
    path.with_suffix(".md").write_text(
        "# Standard-Formula Expression-Quality Breakdown\n\n"
        "Expression quality is decomposed rather than treated as a single "
        "symbolic-success flag.  Variable recall is the support claim; "
        "pair-term recall is the pair claim; operator and complexity are "
        "official expression-track claims.\n\n"
        + markdown_table(show)
        + "\n",
        encoding="utf-8",
    )
    return out


def build_split_consistency(detail: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    rows = []
    block = detail.copy()
    block["suite_split"] = "public diagnostic"
    block.loc[is_standard_formula(block), "suite_split"] = "standard formula"
    hidden = block["split"].astype(str).str.contains("hidden|private", case=False, na=False)
    block.loc[hidden, "suite_split"] = "hidden/private"
    for keys, group in block.groupby(["suite_split", "transfer_id"], dropna=False):
        suite_split, transfer_id = keys
        source_passes = int(pd.to_numeric(group["source_pass"], errors="coerce").fillna(0).sum())
        failures = int(pd.to_numeric(group["unsupported_transfer"], errors="coerce").fillna(0).sum())
        rows.append(
            {
                "suite_split": suite_split,
                "transfer_id": transfer_id,
                "transfer": TRANSFER_LABELS.get(str(transfer_id), str(transfer_id)),
                "eligible_rows": int(len(group)),
                "source_passes": source_passes,
                "target_failures_given_source_pass": failures,
                "overclaim_risk": failures / source_passes if source_passes else float("nan"),
            }
        )
    out = pd.DataFrame(rows).sort_values(["suite_split", "transfer_id"])
    path = out_dir / "split_overclaim_consistency.csv"
    out.to_csv(path, index=False)
    show = out.copy()
    show["overclaim_risk"] = show["overclaim_risk"].map(fmt_rate)
    path.with_suffix(".md").write_text(
        "# Split Overclaim Consistency\n\n"
        "This report separates public diagnostic rows from standard-formula "
        "rows and hidden/private rows when present.  It is a stability check "
        "for whether overclaim risks are tied only to custom diagnostic cards.\n\n"
        + markdown_table(show)
        + "\n",
        encoding="utf-8",
    )
    return out


def build_interpretation_flip(signature: pd.DataFrame, expr: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    rows = []
    sig = signature.set_index("adapter") if not signature.empty else pd.DataFrame()
    expr_idx = expr.set_index("adapter") if not expr.empty else pd.DataFrame()

    def risk(adapter: str, col: str) -> float:
        if sig.empty or adapter not in sig.index or col not in sig.columns:
            return float("nan")
        value = sig.loc[adapter, col]
        if isinstance(value, pd.Series):
            value = value.iloc[0]
        return float(value)

    def metric(adapter: str, col: str) -> float:
        if expr_idx.empty or adapter not in expr_idx.index or col not in expr_idx.columns:
            return float("nan")
        value = expr_idx.loc[adapter, col]
        if isinstance(value, pd.Series):
            value = value.iloc[0]
        return float(value)

    rows.extend(
        [
            {
                "ordinary_report_would_say": "PySR recovered formulas because prediction, support, and symbolic status are high.",
                "claimtransfer_correction": "PySR has strong predictive/support evidence but a high symbolic-status -> expression-quality overclaim edge.",
                "evidence": f"symbolic status {fmt_rate(metric('pysr_symbolic_regressor', 'symbolic_status'))}; expression-quality min {fmt_rate(metric('pysr_symbolic_regressor', 'expression_quality_min'))}; symbolic overclaim risk {fmt_rate(risk('pysr_symbolic_regressor', 'symbolic_status_to_expression_quality'))}",
            },
            {
                "ordinary_report_would_say": "gplearn returned symbolic expressions, so formula quality is recovered.",
                "claimtransfer_correction": "The current gplearn slice exposes a symbolic-status edge that fails operator/complexity checks much more often than status itself.",
                "evidence": f"symbolic status {fmt_rate(metric('gplearn_symbolic_regressor', 'symbolic_status'))}; expression-quality min {fmt_rate(metric('gplearn_symbolic_regressor', 'expression_quality_min'))}; symbolic overclaim risk {fmt_rate(risk('gplearn_symbolic_regressor', 'symbolic_status_to_expression_quality'))}",
            },
            {
                "ordinary_report_would_say": "MLP-Hessian recovered structure because prediction, support, and endpoints are high.",
                "claimtransfer_correction": "Prediction/support/endpoints are separate from verified pair recovery; the pair edge remains measurable.",
                "evidence": f"prediction -> pair risk {fmt_rate(risk('mlp_hessian', 'prediction_to_pair'))}; support -> prediction risk {fmt_rate(risk('mlp_hessian', 'support_to_prediction'))}",
            },
            {
                "ordinary_report_would_say": "A sparse polynomial screen recovered the relevant variables, so it is a useful formula workflow.",
                "claimtransfer_correction": "Support recovery does not imply predictive adequacy or pair recovery.",
                "evidence": f"support -> prediction risk {fmt_rate(risk('poly2_ridge', 'support_to_prediction'))}; prediction -> pair risk {fmt_rate(risk('poly2_ridge', 'prediction_to_pair'))}",
            },
        ]
    )
    out = pd.DataFrame(rows)
    path = out_dir / "ordinary_reporting_interpretation_flip.csv"
    out.to_csv(path, index=False)
    path.with_suffix(".md").write_text(
        "# Ordinary Reporting Interpretation Flip Table\n\n"
        "Each row shows how a conventional multi-metric interpretation changes "
        "when the same evidence is scored as typed ClaimTransfer rows.\n\n"
        + markdown_table(out)
        + "\n",
        encoding="utf-8",
    )
    return out


def write_benchmark_card(claims: pd.DataFrame, out_dir: Path) -> None:
    std = claims[is_standard_formula(claims)]
    card = f"""# ClaimTransfer-Bench 1.0 Benchmark Card

## Purpose

ClaimTransfer-Bench measures structural overclaim risk in formula-recovery
workflows.  The benchmark unit is a task card x adapter x evidence object x
typed structural claim, not a single model-level score.

## Released Scope

- Official claim rows: {len(claims):,}
- Public diagnostic task ids: {claims.loc[~is_standard_formula(claims), 'task_id'].nunique():,}
- Standard-formula settings: {std['task_id'].nunique():,}
- Adapter families: {claims['adapter_family'].nunique():,}
- Methods/adapters: {claims['adapter'].nunique():,}

## Official Claims

Primary structural-transfer claims cover prediction, support, endpoints, pair,
candidate pair, pruning/extraction, and symbolic status.  The expression-quality
track decomposes formula quality into variable/support recall, pair-term recall,
operator recall, and complexity-budget claims.

## Public and Hidden Use

The public suite supports reproducible diagnostic analysis.  The same adapter
format and scorer support offline hidden/private cards or private seeds for
maintainer-run evaluation.

## Missingness Policy

Missing evidence is explicit.  A method is not penalized for unsupported native
fields, but it is not authorized to make claims whose evidence object is absent.

## Intended Use

Use ClaimTransfer-Bench to identify which claim-transfer edge fails for a method:
prediction -> pair, support -> prediction, candidate -> verified pair, fitted
pair -> readout/pruning, or symbolic status -> expression quality.

## Non-Goals

The benchmark does not collapse all formula-recovery behavior into one scalar
leaderboard.  Exact algebraic equivalence, coefficient error, dimensional
consistency, and extrapolation are task-card-specific fields rather than a
universal primary score.

## Minimal Commands

```bash
python scripts/run_benchmark.py --quick
python scripts/check_benchmark_artifact.py
python scripts/build_full_benchmark_analysis_reports.py
```
"""
    path = ROOT / "BENCHMARK_CARD.md"
    path.write_text(card, encoding="utf-8")
    (out_dir / "benchmark_card.md").write_text(card, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claims", default="claim_records/released_claim_records.csv")
    parser.add_argument("--overclaim-detail", default="score_reports/overclaim_risk_detail.csv")
    parser.add_argument("--out-dir", default="score_reports")
    args = parser.parse_args()

    claims_path = ROOT / args.claims
    detail_path = ROOT / args.overclaim_detail
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    claims = pd.read_csv(claims_path, low_memory=False)
    detail = pd.read_csv(detail_path, low_memory=False)

    signature = build_standard_signature(detail, out_dir)
    expr = build_expression_breakdown(claims, out_dir)
    split = build_split_consistency(detail, out_dir)
    flips = build_interpretation_flip(signature, expr, out_dir)
    write_benchmark_card(claims, out_dir)

    print(f"Wrote standard signature rows: {len(signature)}")
    print(f"Wrote expression breakdown rows: {len(expr)}")
    print(f"Wrote split consistency rows: {len(split)}")
    print(f"Wrote interpretation flip rows: {len(flips)}")
    print("Wrote BENCHMARK_CARD.md")


if __name__ == "__main__":
    main()
