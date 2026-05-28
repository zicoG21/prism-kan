from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


METRICS = [
    "test_mse_mean",
    "screen_contains_all_true_vars_mean",
    "screen_true_var_recall_mean",
    "screen_contains_all_interaction_endpoints_mean",
    "screen_interaction_endpoint_recall_mean",
    "screen_contains_true_interactions_mean",
    "variable_f1_mean",
    "interaction_f1_mean",
    "interaction_scoring_computed_mean",
    "num_rows",
    "num_failed",
]


def fmt_float(x) -> str:
    if pd.isna(x):
        return ""
    try:
        value = float(x)
    except (TypeError, ValueError):
        return str(x)
    if abs(value) >= 100 or (0 < abs(value) < 0.001):
        return f"{value:.2e}"
    return f"{value:.3f}"


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        vals = []
        for col in cols:
            val = row[col]
            vals.append(fmt_float(val) if pd.api.types.is_number(val) else str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def has_interaction_scoring(summary: pd.DataFrame) -> bool:
    if "interaction_scoring_computed_mean" not in summary.columns:
        return "interaction_f1_mean" in summary.columns and pd.to_numeric(
            summary["interaction_f1_mean"], errors="coerce"
        ).notna().any()
    computed = pd.to_numeric(summary["interaction_scoring_computed_mean"], errors="coerce")
    return bool((computed > 0).any())


def display_summary(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    if not has_interaction_scoring(out):
        out = out.drop(columns=[c for c in out.columns if c.startswith("interaction_")], errors="ignore")
    return out


def compact_summary(summary: pd.DataFrame) -> pd.DataFrame:
    keep = [c for c in ["function", "screen_mode", *METRICS] if c in summary.columns]
    out = summary[keep].copy()
    if "function" in out.columns and "screen_mode" in out.columns:
        out = out.sort_values(["function", "screen_mode"]).reset_index(drop=True)
    return out


def mode_average(summary: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [c for c in METRICS if c in summary.columns]
    if not metric_cols or "screen_mode" not in summary.columns:
        return pd.DataFrame()
    out = summary.groupby("screen_mode", dropna=False)[metric_cols].mean(numeric_only=True).reset_index()
    return out.sort_values("screen_mode").reset_index(drop=True)


def find_prediction_formula_gaps(summary: pd.DataFrame) -> pd.DataFrame:
    if not has_interaction_scoring(summary):
        return pd.DataFrame()

    required = {
        "function",
        "screen_mode",
        "test_mse_mean",
        "screen_interaction_endpoint_recall_mean",
        "interaction_f1_mean",
    }
    if not required.issubset(summary.columns):
        return pd.DataFrame()

    raw = summary[summary["screen_mode"].astype(str).eq("raw")].copy()
    raw["test_mse_mean"] = pd.to_numeric(raw["test_mse_mean"], errors="coerce")
    raw["screen_interaction_endpoint_recall_mean"] = pd.to_numeric(
        raw["screen_interaction_endpoint_recall_mean"], errors="coerce"
    )
    raw["interaction_f1_mean"] = pd.to_numeric(raw["interaction_f1_mean"], errors="coerce")
    raw = raw.sort_values(["test_mse_mean", "screen_interaction_endpoint_recall_mean"], ascending=[True, True])
    keep = [
        "function",
        "test_mse_mean",
        "screen_interaction_endpoint_recall_mean",
        "interaction_f1_mean",
        "variable_f1_mean",
        "num_rows",
    ]
    return raw[[c for c in keep if c in raw.columns]].head(12).reset_index(drop=True)


def write_report(root: Path, out_path: Path) -> None:
    summaries = sorted(root.glob("**/minisuite_summary.csv"))
    lines = [
        "# Formula-Fidelity Mini-Suite Overnight Report",
        "",
        f"Root: `{root}`",
        "",
    ]

    if not summaries:
        lines.append("_No `minisuite_summary.csv` files found yet._")
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    for summary_path in summaries:
        phase = summary_path.parent.relative_to(root)
        summary = display_summary(pd.read_csv(summary_path))
        lines.extend([
            f"## Phase: `{phase}`",
            "",
            "### Screen-Mode Averages",
            "",
            markdown_table(mode_average(summary)),
            "",
            "### Raw Prediction/Formula Gaps",
            "",
            markdown_table(find_prediction_formula_gaps(summary)),
            "",
            "### Compact Function Summary",
            "",
            markdown_table(compact_summary(summary).head(80)),
            "",
        ])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    root = Path(args.root)
    out = Path(args.out) if args.out else root / "overnight_report.md"
    write_report(root, out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
