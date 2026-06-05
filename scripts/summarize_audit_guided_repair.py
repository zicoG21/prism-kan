#!/usr/bin/env python3
"""Summarize audit-guided repair stage records."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    text = df.copy().astype(str)
    headers = list(text.columns)
    rows = text.values.tolist()
    widths = [max(len(h), *(len(row[i]) for row in rows)) for i, h in enumerate(headers)]

    def fmt(vals: list[object]) -> str:
        return "| " + " | ".join(str(v).ljust(widths[i]) for i, v in enumerate(vals)) + " |"

    return "\n".join(
        [
            fmt(headers),
            "| " + " | ".join("-" * w for w in widths) + " |",
            *(fmt(row) for row in rows),
        ]
    )


def numeric(df: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def rate(mask: pd.Series) -> float:
    if len(mask) == 0:
        return np.nan
    return float(pd.to_numeric(mask, errors="coerce").fillna(0).mean())


def risk(source: pd.Series, target: pd.Series) -> tuple[int, int, float]:
    src = source.fillna(False).astype(bool)
    tgt = target.fillna(False).astype(bool)
    denom = int(src.sum())
    failures = int((src & ~tgt).sum())
    return denom, failures, failures / denom if denom else np.nan


def parse_setting(setting: object) -> tuple[str, str]:
    text = str(setting)
    match = re.match(r"repair_(?P<repair>.*?)__(?P<card>.*?)__", text)
    if match:
        return match.group("repair"), match.group("card")
    return "unknown", text


def add_pass_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    top_m = numeric(out, "top_m", 4).fillna(4)
    mse_threshold = numeric(out, "mse_threshold", 0.05).fillna(0.05)
    out["prediction_pass"] = numeric(out, "test_mse") <= mse_threshold
    out["full_pair_pass"] = (numeric(out, "full_pair_rank") == 1) & (numeric(out, "full_pair_margin") > 0)
    out["readout_pass"] = (numeric(out, "readout_worst_endpoint_rank") <= top_m) & (
        numeric(out, "readout_endpoint_margin") > 0
    )
    out["support_pass"] = numeric(out, "selected_contains_all_true_vars").fillna(0).astype(int).eq(1)
    out["refit_pair_pass"] = (numeric(out, "refit_pair_rank") == 1) & (numeric(out, "refit_pair_margin") > 0)
    out["pruning_pass"] = numeric(out, "prune_endpoint_contains").fillna(0).astype(int).eq(1)
    out["symbolic_status_pass"] = numeric(out, "symbolic_formula_ok").fillna(0).astype(int).eq(1)
    out["top_m"] = top_m
    parsed = out["setting"].map(parse_setting)
    out["repair_protocol"] = parsed.map(lambda x: x[0])
    out["repair_card"] = parsed.map(lambda x: x[1])
    return out


def summarize(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = add_pass_columns(df)
    rows: list[dict[str, object]] = []
    for (card, repair), part in df.groupby(["repair_card", "repair_protocol"], dropna=False, sort=True):
        pred_denom, pred_fail, pred_risk = risk(part["prediction_pass"], part["full_pair_pass"])
        fit_read_denom, fit_read_fail, fit_read_risk = risk(part["full_pair_pass"], part["readout_pass"])
        fit_refit_denom, fit_refit_fail, fit_refit_risk = risk(part["full_pair_pass"], part["refit_pair_pass"])
        fit_prune_denom, fit_prune_fail, fit_prune_risk = risk(part["full_pair_pass"], part["pruning_pass"])
        rows.append(
            {
                "repair_card": card,
                "repair_protocol": repair,
                "rows": int(len(part)),
                "top_m": int(numeric(part, "top_m", 4).median()),
                "prune_threshold": float(numeric(part, "prune_threshold").median()),
                "median_prune_support_size": float(numeric(part, "prune_support_size").median()),
                "prediction_pass_rate": rate(part["prediction_pass"]),
                "full_pair_pass_rate": rate(part["full_pair_pass"]),
                "readout_pass_rate": rate(part["readout_pass"]),
                "support_pass_rate": rate(part["support_pass"]),
                "refit_pair_pass_rate": rate(part["refit_pair_pass"]),
                "pruning_pass_rate": rate(part["pruning_pass"]),
                "symbolic_status_rate": rate(part["symbolic_status_pass"]),
                "prediction_to_pair_source_passes": pred_denom,
                "prediction_to_pair_failures": pred_fail,
                "prediction_to_pair_risk": pred_risk,
                "fitted_pair_to_readout_source_passes": fit_read_denom,
                "fitted_pair_to_readout_failures": fit_read_fail,
                "fitted_pair_to_readout_risk": fit_read_risk,
                "fitted_pair_to_refit_source_passes": fit_refit_denom,
                "fitted_pair_to_refit_failures": fit_refit_fail,
                "fitted_pair_to_refit_risk": fit_refit_risk,
                "fitted_pair_to_pruning_source_passes": fit_prune_denom,
                "fitted_pair_to_pruning_failures": fit_prune_fail,
                "fitted_pair_to_pruning_risk": fit_prune_risk,
                "median_test_mse": float(numeric(part, "test_mse").median()),
                "median_runtime_sec": float(numeric(part, "runtime_sec").median()),
            }
        )
    summary = pd.DataFrame(rows)

    comparisons: list[dict[str, object]] = []
    if not summary.empty:
        for card, part in summary.groupby("repair_card", sort=True):
            base = part[part["repair_protocol"].eq("baseline")]
            if base.empty:
                continue
            base_row = base.iloc[0]
            for _, row in part[~part["repair_protocol"].eq("baseline")].iterrows():
                comparisons.append(
                    {
                        "repair_card": card,
                        "repair_protocol": row["repair_protocol"],
                        "delta_prediction_to_pair_risk": row["prediction_to_pair_risk"] - base_row["prediction_to_pair_risk"],
                        "delta_fitted_pair_to_readout_risk": row["fitted_pair_to_readout_risk"]
                        - base_row["fitted_pair_to_readout_risk"],
                        "delta_fitted_pair_to_refit_risk": row["fitted_pair_to_refit_risk"]
                        - base_row["fitted_pair_to_refit_risk"],
                        "delta_fitted_pair_to_pruning_risk": row["fitted_pair_to_pruning_risk"]
                        - base_row["fitted_pair_to_pruning_risk"],
                        "delta_readout_pass_rate": row["readout_pass_rate"] - base_row["readout_pass_rate"],
                        "delta_refit_pair_pass_rate": row["refit_pair_pass_rate"] - base_row["refit_pair_pass_rate"],
                        "delta_pruning_pass_rate": row["pruning_pass_rate"] - base_row["pruning_pass_rate"],
                        "delta_median_prune_support_size": row["median_prune_support_size"]
                        - base_row["median_prune_support_size"],
                    }
                )
    return summary, pd.DataFrame(comparisons)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="results/revision/audit_guided_repair_claimcards")
    parser.add_argument("--out-prefix", default="results/revision/audit_guided_repair_claimcards/audit_guided_repair_summary")
    args = parser.parse_args()

    root = Path(args.root)
    paths = sorted(root.glob("**/seed_aligned_stage_records_detail.csv"))
    if not paths:
        raise SystemExit(f"No seed_aligned_stage_records_detail.csv files found under {root}")
    df = pd.concat((pd.read_csv(path).assign(source_file=str(path)) for path in paths), ignore_index=True)
    summary, comparison = summarize(df)

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    summary_csv = out_prefix.with_suffix(".csv")
    comparison_csv = out_prefix.with_name(out_prefix.name + "_vs_baseline").with_suffix(".csv")
    md_path = out_prefix.with_suffix(".md")
    summary.to_csv(summary_csv, index=False)
    comparison.to_csv(comparison_csv, index=False)

    show_cols = [
        "repair_card",
        "repair_protocol",
        "rows",
        "top_m",
        "prune_threshold",
        "prediction_pass_rate",
        "full_pair_pass_rate",
        "readout_pass_rate",
        "refit_pair_pass_rate",
        "pruning_pass_rate",
        "fitted_pair_to_readout_risk",
        "fitted_pair_to_refit_risk",
        "fitted_pair_to_pruning_risk",
    ]
    show = summary[show_cols].copy()
    for col in show.columns:
        if col.endswith("_rate") or col.endswith("_risk"):
            show[col] = pd.to_numeric(show[col], errors="coerce").map(lambda x: "" if pd.isna(x) else f"{x:.2f}")
    md = [
        "# Audit-Guided Repair Summary",
        "",
        "Rows compare the baseline protocol against simple repairs targeted at readout/refit/pruning handoffs.",
        "",
        markdown_table(show),
    ]
    if not comparison.empty:
        comp = comparison.copy()
        for col in comp.columns:
            if col.startswith("delta_"):
                comp[col] = pd.to_numeric(comp[col], errors="coerce").map(lambda x: "" if pd.isna(x) else f"{x:+.2f}")
        md.extend(["", "## Delta Versus Baseline", "", markdown_table(comp)])
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Wrote {summary_csv} ({len(summary)} rows)")
    print(f"Wrote {comparison_csv} ({len(comparison)} rows)")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
