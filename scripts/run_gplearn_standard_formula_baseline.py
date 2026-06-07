#!/usr/bin/env python3
"""Run a gplearn symbolic-regression adapter on standard-formula cards.

This is a lightweight mainstream symbolic-regression baseline for the
ClaimTransfer v1 standard-formula wrapper.  It emits normalized adapter-output
rows, then the official scorer recomputes support, endpoint, pair, symbolic
status, operator-recall, and complexity predicates.
"""

from __future__ import annotations

import argparse
import re
from itertools import combinations
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

try:
    from gplearn.genetic import SymbolicRegressor
except Exception as exc:  # pragma: no cover - optional baseline dependency
    raise SystemExit(
        "run_gplearn_standard_formula_baseline.py requires gplearn. "
        "Install it in the active environment with `pip install gplearn` "
        f"or `pip install -r requirements.txt`. Import failed with: "
        f"{type(exc).__name__}: {exc}"
    )

from run_standard_formula_adapter_sweep import (
    ROOT,
    emit_common,
    load_cards,
    make_data,
    operator_target,
    pair_claims,
    parse_seed_range,
)


GP_TO_CLAIM_OP = {
    "add": "plus",
    "sub": "plus",
    "mul": "multiply",
    "div": "divide",
    "sqrt": "sqrt",
    "log": "log",
    "sin": "sin",
    "cos": "cos",
}


def expression_text(model: SymbolicRegressor) -> str:
    return str(model._program)


def expression_variables(expr: str) -> list[int]:
    return sorted({int(m.group(1)) for m in re.finditer(r"\bX(\d+)\b", expr)})


def expression_operators(expr: str) -> str:
    ops = set()
    for token, claim_op in GP_TO_CLAIM_OP.items():
        if re.search(rf"\b{re.escape(token)}\(", expr):
            ops.add(claim_op)
    return ",".join(sorted(ops))


def expression_complexity(model: SymbolicRegressor) -> float:
    try:
        return float(model._program.length_)
    except Exception:
        return float(len(str(model._program)))


def pair_scores_from_expression(expr: str, d: int) -> dict[tuple[int, int], float]:
    variables = expression_variables(expr)
    variable_set = set(variables)
    scores = {pair: 0.0 for pair in combinations(range(d), 2)}

    # Direct products receive stronger credit.  This is intentionally simple:
    # gplearn expressions are prefix strings, so we score obvious local
    # multiplicative co-occurrence and otherwise leave only weak co-selection.
    for i, j in combinations(variables, 2):
        xi = f"X{i}"
        xj = f"X{j}"
        direct = (
            re.search(rf"mul\([^)]*{xi}[^)]*{xj}[^)]*\)", expr)
            or re.search(rf"mul\([^)]*{xj}[^)]*{xi}[^)]*\)", expr)
        )
        scores[tuple(sorted((i, j)))] = 1.0 if direct else 0.25
    for i, j in combinations(range(d), 2):
        if i not in variable_set or j not in variable_set:
            scores[(i, j)] = 0.0
    return scores


def run_card(card: dict, seed: int, args: argparse.Namespace, rows: list[dict]) -> None:
    x_train, y_train, x_test, y_test = make_data(card, seed)
    t0 = perf_counter()
    model = SymbolicRegressor(
        population_size=args.population_size,
        generations=args.generations,
        tournament_size=args.tournament_size,
        stopping_criteria=args.stopping_criteria,
        const_range=(-2.0, 2.0),
        init_depth=(2, 5),
        init_method="half and half",
        function_set=tuple(args.function_set.split(",")),
        metric="mean absolute error",
        parsimony_coefficient=args.parsimony_coefficient,
        p_crossover=0.7,
        p_subtree_mutation=0.1,
        p_hoist_mutation=0.05,
        p_point_mutation=0.1,
        max_samples=0.9,
        verbose=0,
        random_state=seed,
        n_jobs=args.n_jobs,
    )
    model.fit(x_train, y_train)
    runtime = perf_counter() - t0
    pred = model.predict(x_test)
    mse = float(np.mean((pred - y_test) ** 2))
    expr = expression_text(model)
    variables = expression_variables(expr)
    operators = expression_operators(expr)
    complexity = expression_complexity(model)
    d = int(card["dimension"])
    pair_scores = pair_scores_from_expression(expr, d) if pair_claims(card) else None

    before = len(rows)
    emit_common(
        rows,
        card,
        adapter="gplearn_symbolic_regressor",
        adapter_family="symbolic_library",
        seed=seed,
        evidence_object="gplearn_expression",
        mse=mse,
        support_selected=variables,
        pair_scores=pair_scores,
        operators=operators or operator_target(card),
        complexity=complexity,
    )
    for row in rows[before:]:
        row["source_kind"] = "gplearn_standard_formula_baseline"
        row["source_file"] = "gplearn_standard_formula_adapter_outputs.csv"
        row["runtime_seconds"] = f"{runtime:.3f}"
        row["protocol"] = (
            f"gplearn pop={args.population_size} gen={args.generations} "
            f"parsimony={args.parsimony_coefficient}; expr={expr}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-cards", default="task_cards/claimtransfer_v1_standard_formula_public.json")
    parser.add_argument("--seeds", default="0-2")
    parser.add_argument("--out-dir", default="results/revision/gplearn_standard_formula_baseline")
    parser.add_argument("--population-size", type=int, default=300)
    parser.add_argument("--generations", type=int, default=12)
    parser.add_argument("--tournament-size", type=int, default=20)
    parser.add_argument("--stopping-criteria", type=float, default=0.01)
    parser.add_argument("--parsimony-coefficient", type=float, default=0.003)
    parser.add_argument("--function-set", default="add,sub,mul,div,sqrt,log,sin,cos")
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args()

    cards = load_cards(ROOT / args.task_cards)
    seeds = parse_seed_range(args.seeds)
    rows: list[dict] = []
    for card in cards:
        for seed in seeds:
            run_card(card, seed, args, rows)

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    detail = pd.DataFrame(rows)
    detail.to_csv(out_dir / "gplearn_standard_formula_adapter_outputs.csv", index=False)
    summary = (
        detail.groupby(["task_family", "claim_type"], dropna=False)
        .size()
        .reset_index(name="rows")
    )
    summary.to_csv(out_dir / "gplearn_standard_formula_summary.csv", index=False)
    print(f"Wrote {out_dir / 'gplearn_standard_formula_adapter_outputs.csv'} ({len(detail)} rows)")
    print(f"Wrote {out_dir / 'gplearn_standard_formula_summary.csv'}")


if __name__ == "__main__":
    main()
