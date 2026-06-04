#!/usr/bin/env python3
"""Official-score released ClaimTransfer adapter outputs.

Adapters expose raw evidence fields.  This script recomputes derived
predicates, writes row-level claim records, and produces score/coverage reports.
It is intentionally lightweight for the workshop artifact: no model retraining,
but all pass/fail columns are rebuilt from the submitted evidence fields.
"""

from __future__ import annotations

import argparse
import ast
import math
from pathlib import Path
from typing import Any

import pandas as pd


def parse_obj(value: object) -> Any:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (list, tuple, set, dict)):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        return ast.literal_eval(text)
    except Exception:
        return text


def flatten_ints(value: Any) -> set[int]:
    parsed = parse_obj(value)
    out: set[int] = set()
    if parsed is None:
        return out
    if isinstance(parsed, dict):
        parsed = parsed.values()
    if isinstance(parsed, (list, tuple, set)):
        for item in parsed:
            if isinstance(item, (list, tuple, set)):
                out.update(flatten_ints(list(item)))
            else:
                try:
                    out.add(int(item))
                except Exception:
                    pass
    else:
        try:
            out.add(int(parsed))
        except Exception:
            pass
    return out


def as_float(value: object) -> float:
    try:
        if value == "":
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def as_bool_float(value: object) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    text = str(value).strip().lower()
    if text in {"true", "t", "yes", "y"}:
        return 1.0
    if text in {"false", "f", "no", "n"}:
        return 0.0
    return as_float(value)


def official_pass(row: pd.Series) -> float:
    predicate = str(row.get("predicate", "")).strip()
    rank = as_float(row.get("rank", ""))
    raw = as_bool_float(row.get("raw_value", ""))
    threshold = as_float(row.get("threshold", ""))
    budget = as_float(row.get("budget", ""))

    if predicate == "mse_lt":
        if math.isnan(raw):
            return float("nan")
        if math.isnan(threshold):
            threshold = 0.05
        return float(raw < threshold)

    if predicate == "rank1":
        return float(not math.isnan(rank) and rank <= 1)

    if predicate == "rank_at_budget":
        if math.isnan(budget):
            budget = 1
        return float(not math.isnan(rank) and rank <= budget)

    if predicate == "top_m_contains_all":
        if not math.isnan(rank):
            m = budget if not math.isnan(budget) else 4
            return float(rank <= m)
        target = flatten_ints(row.get("target", ""))
        selected = flatten_ints(row.get("selected_set", ""))
        if not target or not selected:
            return float("nan")
        return float(target.issubset(selected))

    if predicate == "contains_all":
        target = flatten_ints(row.get("target", ""))
        selected = flatten_ints(row.get("selected_set", ""))
        if not target or not selected:
            return float("nan")
        return float(target.issubset(selected))

    if predicate == "binary_true":
        if math.isnan(raw):
            return float("nan")
        return float(raw >= 0.5)

    if predicate == "value_le":
        if math.isnan(raw) or math.isnan(threshold):
            return float("nan")
        return float(raw <= threshold)

    if predicate == "value_ge":
        if math.isnan(raw) or math.isnan(threshold):
            return float("nan")
        return float(raw >= threshold)

    if predicate == "exact_string_match":
        expected = str(row.get("target", "")).strip()
        observed = str(row.get("raw_value", "")).strip()
        if not expected or not observed:
            return float("nan")
        return float(expected == observed)

    if predicate == "stress_card":
        return float("nan")

    return float("nan")


def wilson(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    if trials <= 0:
        return (float("nan"), float("nan"))
    phat = successes / trials
    denom = 1 + z * z / trials
    center = (phat + z * z / (2 * trials)) / denom
    half = z * math.sqrt((phat * (1 - phat) + z * z / (4 * trials)) / trials) / denom
    return max(0.0, center - half), min(1.0, center + half)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    groups = [
        "registry_version",
        "split",
        "task_family",
        "task_id",
        "adapter_family",
        "adapter",
        "evidence_object",
        "claim_type",
        "scorer",
        "predicate",
    ]
    rows = []
    for keys, g in df.groupby(groups, dropna=False):
        valid = g["pass"].dropna()
        successes = int(valid.sum()) if len(valid) else 0
        trials = int(len(valid))
        lo, hi = wilson(successes, trials)
        row = dict(zip(groups, keys))
        row.update(
            {
                "rows": int(len(g)),
                "seeds": int(g["seed"].nunique()),
                "missing_pass_rows": int(g["pass"].isna().sum()),
                "successes": successes,
                "trials": trials,
                "pass_rate": successes / trials if trials else float("nan"),
                "wilson_low": lo,
                "wilson_high": hi,
                "median_rank": g["rank_num"].median(skipna=True),
                "median_margin": g["margin_num"].median(skipna=True),
                "median_raw_value": g["raw_num"].median(skipna=True),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(groups)


def coverage(summary: pd.DataFrame) -> pd.DataFrame:
    groups = ["registry_version", "split", "adapter_family", "task_family", "claim_type"]
    rows = []
    for keys, g in summary.groupby(groups, dropna=False):
        row = dict(zip(groups, keys))
        total_trials = int(g["trials"].sum())
        total_successes = int(g["successes"].sum())
        lo, hi = wilson(total_successes, total_trials)
        row.update(
            {
                "score_rows": int(g["rows"].sum()),
                "seed_rows": int(g["seeds"].sum()),
                "report_rows": int(len(g)),
                "missing_pass_rows": int(g["missing_pass_rows"].sum()),
                "successes": total_successes,
                "trials": total_trials,
                "pass_rate": total_successes / total_trials if total_trials else float("nan"),
                "wilson_low": lo,
                "wilson_high": hi,
                "median_rank": g["median_rank"].median(skipna=True),
                "median_margin": g["median_margin"].median(skipna=True),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(groups)


def to_markdown(df: pd.DataFrame, max_rows: int = 80) -> str:
    if df.empty:
        return "No rows."
    table = df.head(max_rows).copy()
    for col in table.columns:
        if pd.api.types.is_float_dtype(table[col]):
            table[col] = table[col].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
    table = table.astype(str)
    cols = list(table.columns)
    widths = [max(len(c), *(len(v) for v in table[c].tolist())) for c in cols]

    def row(vals: list[str]) -> str:
        return "| " + " | ".join(vals[i].ljust(widths[i]) for i in range(len(vals))) + " |"

    lines = [row(cols), "| " + " | ".join("-" * w for w in widths) + " |"]
    lines.extend(row([str(v) for v in values]) for values in table[cols].values.tolist())
    if len(df) > max_rows:
        lines.append(f"\nShowing first {max_rows} of {len(df)} rows.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="claim_records/released_adapter_outputs.csv")
    parser.add_argument("--claim-record-out", default="claim_records/released_claim_records.csv")
    parser.add_argument("--score-report-out", default="score_reports/score_report.csv")
    parser.add_argument("--coverage-out", default="score_reports/coverage_table.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.input, low_memory=False)
    if "split" not in df.columns:
        df["split"] = "public"
    if "registry_version" not in df.columns:
        df["registry_version"] = "claimtransfer_v0_public"
    if "missing_reason" not in df.columns:
        df["missing_reason"] = ""
    if "runtime_seconds" not in df.columns:
        df["runtime_seconds"] = ""
    df["pass"] = df.apply(official_pass, axis=1)
    df["rank_num"] = df["rank"].map(as_float)
    df["margin_num"] = df["margin"].map(as_float)
    df["raw_num"] = df["raw_value"].map(as_float)

    claim_out = Path(args.claim_record_out)
    score_out = Path(args.score_report_out)
    coverage_out = Path(args.coverage_out)
    claim_out.parent.mkdir(parents=True, exist_ok=True)
    score_out.parent.mkdir(parents=True, exist_ok=True)
    coverage_out.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(claim_out, index=False)
    summary = summarize(df)
    cov = coverage(summary)
    summary.to_csv(score_out, index=False)
    cov.to_csv(coverage_out, index=False)

    score_out.with_suffix(".md").write_text("# Official score report\n\n" + to_markdown(summary) + "\n", encoding="utf-8")
    coverage_out.with_suffix(".md").write_text("# Coverage table\n\n" + to_markdown(cov) + "\n", encoding="utf-8")

    print(f"Wrote {claim_out} ({len(df)} claim rows)")
    print(f"Wrote {score_out} ({len(summary)} aggregate rows)")
    print(f"Wrote {coverage_out} ({len(cov)} coverage rows)")


if __name__ == "__main__":
    main()
