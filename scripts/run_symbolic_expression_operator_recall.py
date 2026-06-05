#!/usr/bin/env python3
"""Generate symbolic-library operator-recall diagnostic rows.

This is a lightweight expression-level adapter/control for ClaimTransfer-Bench.
It is not a full symbolic-regression search.  Its purpose is to exercise the
official expression-quality scorer contract on scientific-expression cards:
symbolic status, operator recall, and expression complexity are emitted as raw
adapter outputs and later rescored by scripts/build_score_report.py.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class ExpressionCase:
    task_id: str
    function: str
    expression: str
    support: tuple[int, ...]
    pairs: tuple[tuple[int, int], ...]
    target_operators: tuple[str, ...]
    variants: dict[str, tuple[str, tuple[str, ...], int]]


CASES: dict[str, ExpressionCase] = {
    "energy": ExpressionCase(
        task_id="feynman_style_energy_hidden_template",
        function="feynman_energy",
        expression="0.5*m*v^2 + m*g*h",
        support=(0, 1, 2, 3),
        pairs=((0, 1), (0, 2)),
        target_operators=("plus", "multiply", "power"),
        variants={
            "oracle": ("0.5*m*v^2 + m*g*h", ("plus", "multiply", "power"), 7),
            "missing_power": ("0.5*m*v + m*g*h", ("plus", "multiply"), 6),
            "support_only": ("[m,v,g,h]", tuple(), 4),
            "overcomplex": ("0.5*m*v^2 + m*g*h + 0*x0*x1*x2", ("plus", "multiply", "power"), 14),
        },
    ),
    "gravity": ExpressionCase(
        task_id="feynman_gravity_hidden_template",
        function="feynman_gravity",
        expression="G*m1*m2/r^2",
        support=(0, 1, 2),
        pairs=((0, 1), (0, 2), (1, 2)),
        target_operators=("multiply", "divide", "power"),
        variants={
            "oracle": ("G*m1*m2/r^2", ("multiply", "divide", "power"), 6),
            "missing_power": ("G*m1*m2/r", ("multiply", "divide"), 5),
            "multiplicative_only": ("m1*m2*r", ("multiply",), 4),
            "overcomplex": ("G*m1*m2/r^2 + 0*m1*r + 0*m2*r", ("plus", "multiply", "divide", "power"), 14),
        },
    ),
    "coulomb": ExpressionCase(
        task_id="feynman_coulomb_hidden_template",
        function="feynman_coulomb",
        expression="k*q1*q2/r^2",
        support=(0, 1, 2),
        pairs=((0, 1), (0, 2), (1, 2)),
        target_operators=("multiply", "divide", "power"),
        variants={
            "oracle": ("k*q1*q2/r^2", ("multiply", "divide", "power"), 6),
            "missing_divide": ("k*q1*q2*r^2", ("multiply", "power"), 5),
            "variables_only": ("[q1,q2,r]", tuple(), 3),
            "overcomplex": ("k*q1*q2/r^2 + 0*q1*q2*r", ("plus", "multiply", "divide", "power"), 13),
        },
    ),
    "damped_wave": ExpressionCase(
        task_id="feynman_damped_wave_hidden_template",
        function="feynman_damped_wave",
        expression="A*exp(-b*t)*sin(w*t)",
        support=(0, 1, 2),
        pairs=((0, 1), (0, 2)),
        target_operators=("multiply", "exp", "sin"),
        variants={
            "oracle": ("A*exp(-b*t)*sin(w*t)", ("multiply", "exp", "sin"), 7),
            "missing_trig": ("A*exp(-b*t)*w*t", ("multiply", "exp"), 6),
            "support_only": ("[A,b,t]", tuple(), 3),
            "overcomplex": ("A*exp(-b*t)*sin(w*t) + 0*A*b*t", ("plus", "multiply", "exp", "sin"), 13),
        },
    ),
}


def operator_recall(target: tuple[str, ...], observed: tuple[str, ...]) -> float:
    target_set = set(target)
    observed_set = set(observed)
    if not target_set:
        return math.nan
    return len(target_set & observed_set) / len(target_set)


def parse_seed_range(text: str) -> list[int]:
    out: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", default=["energy"], choices=sorted(CASES))
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["oracle", "missing_power", "support_only", "overcomplex"],
        help="Variant names to emit when present for a case.",
    )
    parser.add_argument("--seeds", default="0-29")
    parser.add_argument("--out-dir", default="results/revision/symbolic_expression_operator_recall")
    parser.add_argument("--operator-threshold", type=float, default=0.95)
    parser.add_argument("--complexity-threshold", type=float, default=12.0)
    parser.add_argument("--label", default="")
    args = parser.parse_args()

    seeds = parse_seed_range(args.seeds)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for case_name in args.cases:
        case = CASES[case_name]
        for variant_name in args.variants:
            if variant_name not in case.variants:
                continue
            expr, observed_ops, complexity = case.variants[variant_name]
            recall = operator_recall(case.target_operators, observed_ops)
            for seed in seeds:
                rows.append(
                    {
                        "task_id": case.task_id,
                        "task_family": "scientific_expression",
                        "function": case.function,
                        "adapter": f"symbolic_library_{variant_name}",
                        "adapter_family": "symbolic_library",
                        "seed": seed,
                        "evidence_object": "symbolic_expression",
                        "scorer": "symbolic_expression_quality",
                        "protocol": f"fixed_symbolic_library:{variant_name}:{args.label}".strip(":"),
                        "expression": expr,
                        "target_expression": case.expression,
                        "selected_support": repr(list(case.support)),
                        "selected_pairs": repr([tuple(pair) for pair in case.pairs]),
                        "target_operators": ",".join(case.target_operators),
                        "observed_operators": ",".join(observed_ops),
                        "operator_recall": recall,
                        "operator_recall_threshold": args.operator_threshold,
                        "expression_complexity": complexity,
                        "complexity_threshold": args.complexity_threshold,
                        "symbolic_status": int(bool(expr)),
                        "variant": variant_name,
                    }
                )

    detail = pd.DataFrame(rows)
    detail.to_csv(out_dir / "symbolic_expression_detail.csv", index=False)

    summary = (
        detail.groupby(["function", "adapter", "variant"], dropna=False)
        .agg(
            n=("seed", "count"),
            operator_recall_mean=("operator_recall", "mean"),
            operator_recall_min=("operator_recall", "min"),
            complexity_median=("expression_complexity", "median"),
            symbolic_status_rate=("symbolic_status", "mean"),
        )
        .reset_index()
    )
    summary["operator_recall_pass_rate"] = (
        detail.assign(pass_op=detail["operator_recall"] >= args.operator_threshold)
        .groupby(["function", "adapter", "variant"], dropna=False)["pass_op"]
        .mean()
        .to_numpy()
    )
    summary["complexity_pass_rate"] = (
        detail.assign(pass_cx=detail["expression_complexity"] <= args.complexity_threshold)
        .groupby(["function", "adapter", "variant"], dropna=False)["pass_cx"]
        .mean()
        .to_numpy()
    )
    summary.to_csv(out_dir / "symbolic_expression_summary.csv", index=False)

    print(f"Wrote {out_dir / 'symbolic_expression_detail.csv'} ({len(detail)} rows)")
    print(f"Wrote {out_dir / 'symbolic_expression_summary.csv'} ({len(summary)} rows)")
    if not summary.empty:
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
