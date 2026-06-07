#!/usr/bin/env python3
"""Evaluate optional expression-equivalence fields for symbolic standard rows.

This is a deliberately narrow optional track: it parses expression strings
recorded by PySR/gplearn adapter protocols, evaluates them on the standard
formula held-out grid, and reports numerical expression equivalence.  It does
not replace the official expression-quality track; it demonstrates that the
ClaimTransfer schema can host exact/near-equivalence fields when an adapter
exposes a final expression.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from generate_standard_formula_task_cards import expanded_formulas, make_card
from run_standard_formula_adapter_sweep import ROOT, make_data


def extract_expr(protocol: object) -> str | None:
    text = str(protocol)
    marker = "expr="
    if marker not in text:
        return None
    return text.split(marker, 1)[1].strip()


def protected_div(a: Any, b: Any) -> Any:
    return np.asarray(a) / (np.asarray(b) + 1e-6)


def protected_sqrt(a: Any) -> Any:
    return np.sqrt(np.abs(a))


def protected_log(a: Any) -> Any:
    return np.log(np.abs(a) + 1e-6)


def eval_expr(expr: str, x: np.ndarray, adapter: str) -> np.ndarray:
    env: dict[str, Any] = {
        "np": np,
        "sin": np.sin,
        "cos": np.cos,
        "exp": lambda z: np.exp(np.clip(z, -20, 20)),
        "sqrt": protected_sqrt if "gplearn" in adapter else np.sqrt,
        "log": protected_log if "gplearn" in adapter else np.log,
        "pi": np.pi,
        "add": lambda a, b: np.asarray(a) + np.asarray(b),
        "sub": lambda a, b: np.asarray(a) - np.asarray(b),
        "mul": lambda a, b: np.asarray(a) * np.asarray(b),
        "div": protected_div,
        "neg": lambda a: -np.asarray(a),
        "inv": lambda a: protected_div(1.0, a),
        "abs": np.abs,
    }
    for j in range(x.shape[1]):
        env[f"x{j}"] = x[:, j]
        env[f"X{j}"] = x[:, j]
    text = expr.replace("^", "**")
    out = eval(text, {"__builtins__": {}}, env)
    arr = np.asarray(out, dtype=float)
    if arr.shape == ():
        arr = np.full(x.shape[0], float(arr))
    return arr


def task_lookup() -> dict[str, dict]:
    cards = [
        make_card(spec, split="public", registry_version="claimtransfer_v1_standard_formula_public")
        for spec in expanded_formulas("full")
    ]
    return {str(card["task_id"]): card for card in cards}


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows = []
    for keys, group in df.groupby(["adapter_family", "adapter"], dropna=False):
        adapter_family, adapter = keys
        valid = group.dropna(subset=["expression_mse"])
        near = int((valid["expression_mse"] < 0.05).sum())
        strict = int((valid["expression_mse"] < 1e-8).sum())
        rows.append(
            {
                "adapter_family": adapter_family,
                "adapter": adapter,
                "rows": int(len(group)),
                "evaluated_rows": int(len(valid)),
                "near_equivalence_successes": near,
                "near_equivalence_rate_mse_lt_005": near / len(valid) if len(valid) else math.nan,
                "exact_like_successes_mse_lt_1e8": strict,
                "exact_like_rate_mse_lt_1e8": strict / len(valid) if len(valid) else math.nan,
                "median_expression_mse": float(valid["expression_mse"].median()) if len(valid) else math.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["adapter_family", "adapter"])


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "No rows."
    table = df.fillna("").copy()
    for col in ["near_equivalence_rate_mse_lt_005", "exact_like_rate_mse_lt_1e8"]:
        if col in table:
            table[col] = table[col].map(lambda x: "" if x == "" else f"{100 * float(x):.1f}%")
    if "median_expression_mse" in table:
        table["median_expression_mse"] = table["median_expression_mse"].map(
            lambda x: "" if x == "" else f"{float(x):.4g}"
        )
    table = table.astype(str)
    cols = list(table.columns)
    widths = [max(len(col), *(len(v) for v in table[col].tolist())) for col in cols]

    def row(vals: list[str]) -> str:
        return "| " + " | ".join(vals[i].ljust(widths[i]) for i in range(len(vals))) + " |"

    lines = [row(cols), "| " + " | ".join("-" * w for w in widths) + " |"]
    lines.extend(row([str(v) for v in values]) for values in table[cols].values.tolist())
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-outputs", default="claim_records/released_adapter_outputs.csv")
    parser.add_argument("--out", default="score_reports/standard_formula_expression_equivalence_subset.csv")
    args = parser.parse_args()

    src = ROOT / args.adapter_outputs
    rows = pd.read_csv(src, low_memory=False)
    block = rows[
        (rows["registry_version"].astype(str) == "claimtransfer_v1_standard_formula_public")
        & (rows["claim_type"].astype(str) == "prediction")
        & (rows["adapter"].astype(str).isin(["pysr_symbolic_regressor", "gplearn_symbolic_regressor"]))
    ].copy()
    cards = task_lookup()
    out_rows: list[dict[str, object]] = []
    for _, row in block.iterrows():
        expr = extract_expr(row.get("protocol", ""))
        task_id = str(row["task_id"])
        card = cards.get(task_id)
        if expr is None or card is None:
            mse = math.nan
            error = "missing_expression_or_card"
        else:
            try:
                _, _, x_test, y_test = make_data(card, int(row["seed"]))
                pred = eval_expr(expr, x_test, str(row["adapter"]))
                mse = float(np.nanmean((pred - y_test) ** 2))
                if not np.isfinite(mse):
                    error = "nonfinite_expression"
                    mse = math.nan
                else:
                    error = ""
            except Exception as exc:
                mse = math.nan
                error = f"{type(exc).__name__}: {str(exc)[:80]}"
        out_rows.append(
            {
                "registry_version": row["registry_version"],
                "split": row["split"],
                "task_id": task_id,
                "task_family": row["task_family"],
                "adapter_family": row["adapter_family"],
                "adapter": row["adapter"],
                "seed": int(row["seed"]),
                "expression_mse": mse,
                "near_equivalence_pass_mse_lt_005": float(mse < 0.05) if not math.isnan(mse) else math.nan,
                "exact_like_pass_mse_lt_1e8": float(mse < 1e-8) if not math.isnan(mse) else math.nan,
                "expression": expr or "",
                "eval_error": error,
            }
        )

    detail = pd.DataFrame(out_rows)
    summary = summarize(detail)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    detail.to_csv(out, index=False)
    summary_out = out.with_name("standard_formula_expression_equivalence_summary.csv")
    summary.to_csv(summary_out, index=False)
    summary_out.with_suffix(".md").write_text(
        "# Standard-Formula Expression Equivalence Subset\n\n"
        "Optional numerical expression-equivalence check for symbolic adapters "
        "that expose final expressions.  Near equivalence uses held-out MSE < "
        "0.05; exact-like equivalence uses held-out MSE < 1e-8.\n\n"
        + markdown_table(summary)
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {out} ({len(detail)} rows)")
    print(f"Wrote {summary_out} ({len(summary)} rows)")


if __name__ == "__main__":
    main()
