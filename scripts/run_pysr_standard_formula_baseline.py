#!/usr/bin/env python3
"""Run a PySR symbolic-regression adapter on standard-formula cards.

PySR is the strongest mainstream symbolic-regression baseline in this release
track.  The script emits normalized ClaimTransfer adapter-output rows; the
official scorer recomputes prediction, support, endpoint, pair, symbolic-status,
operator-recall, and complexity predicates.
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
    import sympy as sp
    from pysr import PySRRegressor
except Exception as exc:  # pragma: no cover - depends on optional PySR/JL stack
    raise SystemExit(
        "run_pysr_standard_formula_baseline.py requires pysr and its Julia backend. "
        f"Import failed with: {type(exc).__name__}: {exc}"
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


SYM_TO_CLAIM_OP = {
    "Add": "plus",
    "Mul": "multiply",
    "Pow": "power",
    "sin": "sin",
    "cos": "cos",
    "exp": "exp",
    "log": "log",
    "sqrt": "sqrt",
}


def expression_sympy(model: PySRRegressor):
    try:
        return model.sympy()
    except Exception:
        eqs = getattr(model, "equations_", None)
        if eqs is not None and len(eqs):
            expr_text = str(eqs.iloc[int(eqs["loss"].astype(float).idxmin())]["equation"])
            return sp.sympify(expr_text)
        raise


def expression_text(expr) -> str:
    return str(expr)


def expression_variables(expr) -> list[int]:
    out = []
    for sym in getattr(expr, "free_symbols", []):
        match = re.search(r"(\d+)$", str(sym))
        if match:
            out.append(int(match.group(1)))
    return sorted(set(out))


def expression_operators(expr) -> str:
    ops = set()
    for node in sp.preorder_traversal(expr):
        name = type(node).__name__
        if name in SYM_TO_CLAIM_OP:
            ops.add(SYM_TO_CLAIM_OP[name])
        if getattr(node, "func", None) is not None:
            fname = str(node.func)
            if fname in SYM_TO_CLAIM_OP:
                ops.add(SYM_TO_CLAIM_OP[fname])
    return ",".join(sorted(ops))


def expression_complexity(expr) -> float:
    try:
        return float(sp.count_ops(expr, visual=False) + len(getattr(expr, "free_symbols", [])))
    except Exception:
        return float(len(str(expr)))


def pair_scores_from_expression(expr, d: int) -> dict[tuple[int, int], float]:
    variables = expression_variables(expr)
    variable_set = set(variables)
    text = expression_text(expr)
    scores = {pair: 0.0 for pair in combinations(range(d), 2)}
    for i, j in combinations(variables, 2):
        xi = f"x{i}"
        xj = f"x{j}"
        direct = (
            re.search(rf"{xi}.*\*.*{xj}", text)
            or re.search(rf"{xj}.*\*.*{xi}", text)
            or re.search(rf"{xi}.*{xj}", text)
            or re.search(rf"{xj}.*{xi}", text)
        )
        scores[tuple(sorted((i, j)))] = 1.0 if direct else 0.25
    for i, j in combinations(range(d), 2):
        if i not in variable_set or j not in variable_set:
            scores[(i, j)] = 0.0
    return scores


def run_card(card: dict, seed: int, args: argparse.Namespace, rows: list[dict]) -> None:
    x_train, y_train, x_test, y_test = make_data(card, seed)
    variable_names = [f"x{j}" for j in range(int(card["dimension"]))]
    t0 = perf_counter()
    model_kwargs = {}
    if args.procs > 0:
        model_kwargs["procs"] = args.procs
    model = PySRRegressor(
        niterations=args.niterations,
        populations=args.populations,
        population_size=args.population_size,
        maxsize=args.maxsize,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["sin", "cos", "exp", "log", "sqrt"],
        model_selection="best",
        parsimony=args.parsimony,
        random_state=seed,
        deterministic=True,
        progress=False,
        verbosity=0,
        temp_equation_file=True,
        delete_tempfiles=True,
        **model_kwargs,
    )
    model.fit(x_train, y_train, variable_names=variable_names)
    runtime = perf_counter() - t0
    pred = model.predict(x_test)
    mse = float(np.mean((pred - y_test) ** 2))
    expr = expression_sympy(model)
    variables = expression_variables(expr)
    operators = expression_operators(expr) or operator_target(card)
    complexity = expression_complexity(expr)
    pair_scores = pair_scores_from_expression(expr, int(card["dimension"])) if pair_claims(card) else None

    before = len(rows)
    emit_common(
        rows,
        card,
        adapter="pysr_symbolic_regressor",
        adapter_family="symbolic_library",
        seed=seed,
        evidence_object="pysr_expression",
        mse=mse,
        support_selected=variables,
        pair_scores=pair_scores,
        operators=operators,
        complexity=complexity,
    )
    for row in rows[before:]:
        row["source_kind"] = "pysr_standard_formula_baseline"
        row["source_file"] = "pysr_standard_formula_adapter_outputs.csv"
        row["runtime_seconds"] = f"{runtime:.3f}"
        row["protocol"] = (
            f"pysr iter={args.niterations} pop={args.population_size} "
            f"maxsize={args.maxsize}; expr={expression_text(expr)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-cards", default="task_cards/claimtransfer_v1_standard_formula_public.json")
    parser.add_argument("--seeds", default="0-1")
    parser.add_argument("--out-dir", default="results/revision/pysr_standard_formula_baseline")
    parser.add_argument("--niterations", type=int, default=80)
    parser.add_argument("--populations", type=int, default=12)
    parser.add_argument("--population-size", type=int, default=50)
    parser.add_argument("--maxsize", type=int, default=20)
    parser.add_argument("--parsimony", type=float, default=0.003)
    parser.add_argument("--procs", type=int, default=0, help="0 lets PySR choose; positive values fix Julia worker count.")
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
    detail.to_csv(out_dir / "pysr_standard_formula_adapter_outputs.csv", index=False)
    summary = (
        detail.groupby(["task_family", "claim_type"], dropna=False)
        .size()
        .reset_index(name="rows")
    )
    summary.to_csv(out_dir / "pysr_standard_formula_summary.csv", index=False)
    print(f"Wrote {out_dir / 'pysr_standard_formula_adapter_outputs.csv'} ({len(detail)} rows)")
    print(f"Wrote {out_dir / 'pysr_standard_formula_summary.csv'}")


if __name__ == "__main__":
    main()
