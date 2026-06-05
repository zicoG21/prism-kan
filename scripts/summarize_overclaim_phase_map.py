#!/usr/bin/env python3
"""Summarize pyKAN overclaim-risk phase-map stage records."""

from __future__ import annotations

import argparse
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
    return out


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    df = add_pass_columns(df)
    group_cols = [
        "function",
        "samples",
        "noise",
        "update_grid",
        "nuisance_correlation",
        "n_correlated_proxies",
    ]
    for col in group_cols:
        if col not in df.columns:
            df[col] = ""

    rows: list[dict[str, object]] = []
    for key, part in df.groupby(group_cols, dropna=False, sort=True):
        pred_denom, pred_fail, pred_risk = risk(part["prediction_pass"], part["full_pair_pass"])
        fit_read_denom, fit_read_fail, fit_read_risk = risk(part["full_pair_pass"], part["readout_pass"])
        fit_prune_denom, fit_prune_fail, fit_prune_risk = risk(part["full_pair_pass"], part["pruning_pass"])
        support_pred_denom, support_pred_fail, support_pred_risk = risk(part["support_pass"], part["prediction_pass"])
        row = dict(zip(group_cols, key, strict=False))
        row.update(
            {
                "rows": int(len(part)),
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
                "fitted_pair_to_pruning_source_passes": fit_prune_denom,
                "fitted_pair_to_pruning_failures": fit_prune_fail,
                "fitted_pair_to_pruning_risk": fit_prune_risk,
                "support_to_prediction_source_passes": support_pred_denom,
                "support_to_prediction_failures": support_pred_fail,
                "support_to_prediction_risk": support_pred_risk,
                "median_test_mse": float(numeric(part, "test_mse").median()),
                "median_runtime_sec": float(numeric(part, "runtime_sec").median()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="results/revision/overclaim_phase_map")
    parser.add_argument("--out-prefix", default="results/revision/overclaim_phase_map/overclaim_phase_map_summary")
    args = parser.parse_args()

    root = Path(args.root)
    paths = sorted(root.glob("**/seed_aligned_stage_records_detail.csv"))
    if not paths:
        raise SystemExit(f"No seed_aligned_stage_records_detail.csv files found under {root}")
    df = pd.concat((pd.read_csv(path).assign(source_file=str(path)) for path in paths), ignore_index=True)
    summary = summarize(df)

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = out_prefix.with_suffix(".csv")
    md_path = out_prefix.with_suffix(".md")
    summary.to_csv(csv_path, index=False)

    show_cols = [
        "function",
        "samples",
        "noise",
        "update_grid",
        "nuisance_correlation",
        "n_correlated_proxies",
        "rows",
        "prediction_pass_rate",
        "full_pair_pass_rate",
        "readout_pass_rate",
        "prediction_to_pair_risk",
        "fitted_pair_to_readout_risk",
        "fitted_pair_to_pruning_risk",
    ]
    show = summary[show_cols].copy()
    for col in show.columns:
        if col.endswith("_rate") or col.endswith("_risk"):
            show[col] = pd.to_numeric(show[col], errors="coerce").map(lambda x: "" if pd.isna(x) else f"{x:.2f}")
    md_path.write_text(
        "# Overclaim Phase Map Summary\n\n"
        "Rows are grouped by signal strength, sample size, noise/proxy condition, and grid-update protocol.\n\n"
        + markdown_table(show)
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {csv_path} ({len(summary)} rows)")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
