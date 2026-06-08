#!/usr/bin/env python3
"""Build a secondary expression-track report.

The primary expression track is the declared structural contract:
symbolic status, variable/support recall, pair-term recall, operator recall,
and complexity budget.  Optional numerical equivalence is only computed for
adapters that expose parseable final expressions.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def fmt_rate(value: object) -> str:
    try:
        val = float(value)
    except Exception:
        return ""
    if math.isnan(val):
        return ""
    return f"{100 * val:.1f}%"


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


def main() -> None:
    out_dir = ROOT / "score_reports"
    quality = pd.read_csv(out_dir / "standard_formula_expression_quality_breakdown.csv")
    equivalence = pd.read_csv(out_dir / "standard_formula_expression_equivalence_summary.csv")

    rows: list[dict[str, object]] = []
    eq_by_adapter = equivalence.set_index("adapter") if not equivalence.empty else pd.DataFrame()
    for _, row in quality.iterrows():
        adapter = str(row["adapter"])
        eq = eq_by_adapter.loc[adapter] if adapter in eq_by_adapter.index else None
        exposes_expression = bool(pd.notna(row.get("symbolic_status")) and float(row.get("symbolic_status")) >= 0.0)
        if eq is not None:
            equivalence_scope = "evaluated: parseable final expression"
            near = float(eq["near_equivalence_rate_mse_lt_005"])
            exact = float(eq["exact_like_rate_mse_lt_1e8"])
            evaluated = int(eq["evaluated_rows"])
        elif exposes_expression:
            equivalence_scope = "not evaluated: no parseable final-expression protocol in current artifact"
            near = math.nan
            exact = math.nan
            evaluated = 0
        else:
            equivalence_scope = "not applicable: adapter does not expose a symbolic expression"
            near = math.nan
            exact = math.nan
            evaluated = 0
        rows.append(
            {
                "adapter_family": row["adapter_family"],
                "adapter": adapter,
                "primary_symbolic_status": row.get("symbolic_status"),
                "primary_operator_recall": row.get("operator_recall"),
                "primary_complexity_budget": row.get("complexity_budget"),
                "primary_expression_contract_min": row.get("expression_quality_min"),
                "optional_equivalence_scope": equivalence_scope,
                "optional_equivalence_evaluated_rows": evaluated,
                "optional_near_equivalence_mse_lt_005": near,
                "optional_exact_like_mse_lt_1e8": exact,
            }
        )

    out = pd.DataFrame(rows).sort_values(["adapter_family", "adapter"])
    path = out_dir / "expression_track_secondary_report.csv"
    out.to_csv(path, index=False)

    show = out.copy()
    for col in [
        "primary_symbolic_status",
        "primary_operator_recall",
        "primary_complexity_budget",
        "primary_expression_contract_min",
        "optional_near_equivalence_mse_lt_005",
        "optional_exact_like_mse_lt_1e8",
    ]:
        show[col] = show[col].map(fmt_rate)

    md = [
        "# Secondary Expression-Track Report",
        "",
        "Primary expression claims use the declared structural contract: symbolic",
        "status, operator recall, and complexity budget.  Optional numerical",
        "equivalence is evaluated only for adapters that expose parseable final",
        "expressions.  This report prevents the optional equivalence track from",
        "being mistaken for a universal primary score.",
        "",
        markdown_table(show),
        "",
    ]
    path.with_suffix(".md").write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote {path}")
    print(f"rows: {len(out)}")


if __name__ == "__main__":
    main()
