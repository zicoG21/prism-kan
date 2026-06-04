#!/usr/bin/env python3
"""Build released adapter-output rows from existing experiment CSVs.

This is the workshop-size bridge from the historical experiment outputs to the
ClaimTransfer-Bench contract.  It does not trust adapter-provided pass/fail
columns.  Instead it writes standardized raw evidence rows; the official scorer
script recomputes predicates and aggregate score reports from these rows.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


FORMULA_META: dict[str, dict[str, Any]] = {
    "core_interaction_c025": {"family": "weak_centered", "support": [0, 1, 2, 3], "pairs": [(2, 3)], "endpoints": [2, 3]},
    "formula_weak_centered": {"family": "weak_centered", "support": [0, 1, 2, 3], "pairs": [(2, 3)], "endpoints": [2, 3]},
    "formula_bilinear": {"family": "bilinear", "support": [0, 1, 2], "pairs": [(0, 1)], "endpoints": [0, 1]},
    "formula_trig_product": {"family": "trig_product", "support": [0, 1, 2], "pairs": [(0, 1)], "endpoints": [0, 1]},
    "formula_rational_product": {"family": "rational_product", "support": [0, 1, 2], "pairs": [(0, 1)], "endpoints": [0, 1, 2]},
    "formula_division_mixed": {"family": "division_mixed", "support": [0, 1, 2, 3], "pairs": [(0, 1), (2, 3)], "endpoints": [0, 1, 2, 3]},
    "formula_mixed_sparse": {"family": "mixed_sparse", "support": [0, 1, 2, 3], "pairs": [(1, 2)], "endpoints": [1, 2, 3]},
    "formula_sqrt_energy": {"family": "sqrt_energy", "support": [0, 1, 2], "pairs": [(0, 1)], "endpoints": [0, 1]},
    "formula_nested_trig": {"family": "nested_trig", "support": [0, 1, 2], "pairs": [(1, 2)], "endpoints": [1, 2]},
    "formula_three_way_product": {"family": "three_way_product", "support": [0, 1, 2, 3], "pairs": [(0, 1)], "endpoints": [0, 1, 2]},
    "formula_exp_product": {"family": "exp_product", "support": [0, 1, 2], "pairs": [(0, 1)], "endpoints": [0, 1]},
    "formula_log_product": {"family": "log_product", "support": [0, 1, 2], "pairs": [(0, 1)], "endpoints": [0, 1]},
}


def parse_obj(value: object, default: Any = None) -> Any:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default
    if isinstance(value, (list, tuple, dict)):
        return value
    text = str(value)
    try:
        return ast.literal_eval(text)
    except Exception:
        return default if default is not None else text


def norm_pair(pair: Any) -> tuple[int, int] | None:
    if pair is None:
        return None
    if isinstance(pair, str):
        pair = parse_obj(pair)
    try:
        a, b = pair
        return tuple(sorted((int(a), int(b))))  # type: ignore[return-value]
    except Exception:
        return None


def norm_pairs(value: object, fallback: list[tuple[int, int]]) -> list[tuple[int, int]]:
    parsed = parse_obj(value, default=None)
    if parsed is None:
        return fallback
    pairs: list[tuple[int, int]] = []
    for item in parsed:
        p = norm_pair(item)
        if p is not None:
            pairs.append(p)
    return pairs or fallback


def task_id(function: str, row: pd.Series) -> str:
    family = FORMULA_META.get(function, {"family": function})["family"]
    d = int(row.get("dimension", 0) or 0)
    n = int(row.get("samples", 0) or 0)
    noise = float(row.get("noise", 0.0) or 0.0)
    suffix = f"d{d}_n{n}"
    if noise:
        suffix += f"_noise{noise:g}".replace(".", "p")
    return f"{family}_{suffix}"


def add(rows: list[dict[str, Any]], **kwargs: Any) -> None:
    base = {
        "task_id": "",
        "task_family": "",
        "adapter": "",
        "adapter_family": "",
        "source_kind": "",
        "source_file": "",
        "seed": "",
        "evidence_object": "",
        "claim_type": "",
        "target": "",
        "scorer": "",
        "predicate": "",
        "threshold": "",
        "budget": "",
        "rank": "",
        "margin": "",
        "raw_value": "",
        "selected_set": "",
        "candidate_set": "",
        "protocol": "",
    }
    base.update(kwargs)
    rows.append(base)


def target_str(value: Any) -> str:
    return repr(value)


def convert_stage_detail(path: Path, rows: list[dict[str, Any]]) -> None:
    df = pd.read_csv(path)
    for _, r in df.iterrows():
        function = str(r["function"])
        meta = FORMULA_META.get(function, {})
        tid = task_id(function, r)
        common = {
            "task_id": tid,
            "task_family": meta.get("family", function),
            "adapter": "pyKAN",
            "adapter_family": "pyKAN",
            "source_kind": "seed_aligned_stage_record",
            "source_file": str(path),
            "seed": int(r["seed"]),
            "protocol": str(r.get("setting", "")),
        }
        pairs = norm_pairs(r.get("true_pair"), meta.get("pairs", []))
        support = parse_obj(r.get("true_variables"), meta.get("support", []))
        endpoints = sorted({v for p in pairs for v in p}) or meta.get("endpoints", [])

        add(rows, **common, evidence_object="prediction", claim_type="prediction", target="low_mse", scorer="mse", predicate="mse_lt", threshold=0.05, raw_value=r.get("test_mse", ""))
        add(rows, **common, evidence_object="full_function", claim_type="pair", target=target_str(pairs), scorer="functional_anova", predicate="rank1", rank=r.get("full_pair_rank", ""), margin=r.get("full_pair_margin", ""))
        add(rows, **common, evidence_object="exposed_readout", claim_type="endpoints", target=target_str(endpoints), scorer="KAN-FE", predicate="top_m_contains_all", budget=4, rank=r.get("readout_worst_endpoint_rank", ""), margin=r.get("readout_endpoint_margin", ""))
        add(rows, **common, evidence_object="selected_support", claim_type="support", target=target_str(support), scorer="KAN-FE", predicate="contains_all", selected_set=r.get("selected_support", ""))
        add(rows, **common, evidence_object="support_refit", claim_type="pair", target=target_str(pairs), scorer="functional_anova", predicate="rank1", rank=r.get("refit_pair_rank", ""), margin=r.get("refit_pair_margin", ""))
        if "prune_endpoint_contains" in r:
            add(rows, **common, evidence_object="pruning", claim_type="endpoints", target=target_str(endpoints), scorer="prune_input", predicate="binary_true", raw_value=r.get("prune_endpoint_contains", ""), selected_set=r.get("prune_selected_inputs", ""))
        if "symbolic_formula_ok" in r:
            add(rows, **common, evidence_object="symbolic", claim_type="symbolic_status", target="syntactic_expression", scorer="pyKAN_symbolic", predicate="binary_true", raw_value=r.get("symbolic_formula_ok", ""))


def convert_cross_method(path: Path, rows: list[dict[str, Any]]) -> None:
    df = pd.read_csv(path)
    for _, r in df.iterrows():
        function = str(r["function"])
        meta = FORMULA_META.get(function, {})
        tid = task_id(function, r)
        method = str(r.get("method", "external"))
        common = {
            "task_id": tid,
            "task_family": meta.get("family", function),
            "adapter": method,
            "adapter_family": method,
            "source_kind": "cross_method_transfer",
            "source_file": str(path),
            "seed": int(r["seed"]),
            "protocol": method,
        }
        add(rows, **common, evidence_object="prediction", claim_type="prediction", target="low_mse", scorer="mse", predicate="mse_lt", threshold=0.05, raw_value=r.get("test_mse", ""))
        add(rows, **common, evidence_object=str(r.get("support_object", "support")), claim_type="support", target=target_str(meta.get("support", [])), scorer=method, predicate="contains_all", selected_set=r.get("selected_support", ""))
        add(rows, **common, evidence_object="endpoint_selection", claim_type="endpoints", target=target_str(meta.get("endpoints", [])), scorer=method, predicate="binary_true", raw_value=r.get("endpoint_success", ""))
        budget = r.get("pair_budget", len(meta.get("pairs", [])) or 1)
        add(rows, **common, evidence_object=str(r.get("evidence_object", "pair_scores")), claim_type="pair", target=target_str(meta.get("pairs", [])), scorer=method, predicate="rank_at_budget", budget=budget, rank=r.get("true_pair_rank_worst", r.get("true_pair_rank_best", "")), margin=r.get("true_pair_margin_min", ""))


def convert_treegate(path: Path, rows: list[dict[str, Any]]) -> None:
    df = pd.read_csv(path)
    for _, r in df.iterrows():
        if str(r.get("status", "ok")) != "ok":
            continue
        function = str(r["function"])
        meta = FORMULA_META.get(function, {})
        tid = task_id(function, r)
        forest = str(r.get("forest_type", "tree"))
        gate = str(r.get("gate_score", "gate"))
        adapter = f"TreeGate-{forest}-{gate}"
        common = {
            "task_id": tid,
            "task_family": meta.get("family", function),
            "adapter": adapter,
            "adapter_family": "tree_gate",
            "source_kind": "treegate",
            "source_file": str(path),
            "seed": int(r["seed"]),
            "protocol": f"gate{r.get('gate_size', '')}_budget{r.get('direct_pair_budget', '')}",
        }
        pairs = norm_pairs(r.get("true_pairs"), meta.get("pairs", []))
        endpoints = parse_obj(r.get("true_endpoints"), meta.get("endpoints", []))
        add(rows, **common, evidence_object="prediction", claim_type="prediction", target="low_mse", scorer="mse", predicate="mse_lt", threshold=0.05, raw_value=r.get("test_mse", ""))
        add(rows, **common, evidence_object="candidate_gate", claim_type="endpoints", target=target_str(endpoints), scorer=gate, predicate="binary_true", raw_value=r.get("all_pair_endpoints_in_gate", ""))
        add(rows, **common, evidence_object="candidate_pairs", claim_type="candidate_pair", target=target_str(pairs), scorer=gate, predicate="binary_true", raw_value=r.get("true_pair_in_candidates", ""), candidate_set=r.get("candidate_pair_top20", ""))
        add(rows, **common, evidence_object="verified_candidate_pairs", claim_type="pair", target=target_str(pairs), scorer="candidate_functional_anova", predicate="rank_at_budget", budget=len(pairs) or 1, rank=r.get("true_pair_rank_best", ""), margin="")


def convert_scorergram(path: Path, rows: list[dict[str, Any]]) -> None:
    df = pd.read_csv(path)
    for _, r in df.iterrows():
        if str(r.get("status", "ok")) != "ok":
            continue
        function = str(r["function"])
        meta = FORMULA_META.get(function, {})
        tid = task_id(function, r)
        scorer = str(r.get("pair_scorer", "pair_scorer"))
        common = {
            "task_id": tid,
            "task_family": meta.get("family", function),
            "adapter": "pyKAN-scorergram",
            "adapter_family": "pyKAN",
            "source_kind": "pair_scorer_claim_grammar",
            "source_file": str(path),
            "seed": int(r["seed"]),
            "protocol": f"scorer={scorer}",
        }
        pairs = norm_pairs(r.get("true_pairs"), meta.get("pairs", []))
        add(rows, **common, evidence_object="prediction", claim_type="prediction", target="low_mse", scorer="mse", predicate="mse_lt", threshold=0.05, raw_value=r.get("test_mse", ""))
        add(rows, **common, evidence_object="pair_scorer", claim_type="pair", target=target_str(pairs), scorer=scorer, predicate="rank_at_budget", budget=len(pairs) or 1, rank=r.get("true_interaction_worst_rank", r.get("true_pair_worst_rank", "")), margin=r.get("true_interaction_mean_score_margin", r.get("min_true_minus_max_false", "")))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default="results/revision")
    parser.add_argument("--out", default="claim_records/released_adapter_outputs.csv")
    args = parser.parse_args()

    root = Path(args.results_root)
    rows: list[dict[str, Any]] = []

    patterns = [
        ("**/seed_aligned_stage_records_detail.csv", convert_stage_detail),
        ("**/cross_method_transfer_detail.csv", convert_cross_method),
        ("**/treegate_pair_screen_detail.csv", convert_treegate),
        ("**/pair_scorer_claim_grammar_detail.csv", convert_scorergram),
    ]
    for pattern, converter in patterns:
        for path in sorted(root.glob(pattern)):
            converter(path, rows)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print(f"Wrote {out} ({len(df)} raw evidence rows)")
    if not df.empty:
        print(df.groupby(["source_kind", "claim_type"]).size().to_string())


if __name__ == "__main__":
    main()
