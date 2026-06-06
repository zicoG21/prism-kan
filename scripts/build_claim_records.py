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
    "feynman_energy": {"family": "scientific_expression", "support": [0, 1], "pairs": [(0, 1)], "endpoints": [0, 1]},
    "feynman_gravity": {"family": "scientific_expression", "support": [0, 1, 2], "pairs": [(0, 1), (0, 2), (1, 2)], "endpoints": [0, 1, 2]},
    "feynman_coulomb": {"family": "scientific_expression", "support": [0, 1, 2], "pairs": [(0, 1), (0, 2), (1, 2)], "endpoints": [0, 1, 2]},
    "feynman_damped_wave": {"family": "scientific_expression", "support": [0, 1, 2], "pairs": [(0, 1), (0, 2)], "endpoints": [0, 1, 2]},
}

SEMISYNTH_META: dict[str, dict[str, Any]] = {
    "breast_cancer": {
        "family": "semi_breast_cancer",
        "task_id": "semi_breast_cancer_core",
        "support": [0, 1, 2, 3],
        "pairs": [(2, 3)],
        "endpoints": [2, 3],
    },
    "diabetes": {
        "family": "semi_diabetes",
        "task_id": "semi_diabetes_core",
        "support": [0, 1, 2, 3],
        "pairs": [(2, 3)],
        "endpoints": [2, 3],
    },
    "wine": {
        "family": "semi_wine",
        "task_id": "semi_wine_core",
        "support": [0, 1, 2, 3],
        "pairs": [(2, 3)],
        "endpoints": [2, 3],
    },
}


def meta_for_row(function: str, row: pd.Series) -> tuple[str, str, dict[str, Any]]:
    """Return task id/family metadata, with overrides for template-style rows."""
    if function.startswith("semisynthetic_"):
        dataset = function.removeprefix("semisynthetic_")
        meta = SEMISYNTH_META.get(dataset)
        if meta is not None:
            return str(meta["task_id"]), str(meta["family"]), dict(meta)
    meta = dict(FORMULA_META.get(function, {"family": function, "support": [], "pairs": [], "endpoints": []}))
    setting_text = " ".join(
        str(row.get(col, ""))
        for col in ["setting", "protocol", "task_id", "task_family", "label"]
        if col in row
    ).lower()
    rho = as_float(row.get("nuisance_correlation", ""))
    proxies = as_float(row.get("n_correlated_proxies", ""))
    if (
        "correlated_covariates" in setting_text
        or "correlated-covariate" in setting_text
        or (not pd.isna(rho) and rho > 0 and not pd.isna(proxies) and proxies > 0)
    ):
        meta["family"] = "correlated_covariates"
        return "correlated_covariate_pair_hidden_template", "correlated_covariates", meta
    return task_id(function, row), str(meta.get("family", function)), meta


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


def as_float(value: object) -> float:
    try:
        if value == "":
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def flatten_ints(value: object) -> set[int]:
    parsed = parse_obj(value, default=None)
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
        "registry_version": "claimtransfer_v0_public",
        "split": "public",
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
        "missing_reason": "",
        "runtime_seconds": "",
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
        tid, family, meta = meta_for_row(function, r)
        common = {
            "task_id": tid,
            "task_family": family,
            "adapter": "pyKAN",
            "adapter_family": "pyKAN",
            "source_kind": "seed_aligned_stage_record",
            "source_file": str(path),
            "seed": int(r["seed"]),
            "protocol": (
                f"{r.get('setting', '')}:top_m={r.get('top_m', 4)}:"
                f"prune={r.get('prune_threshold', '')}:refit_steps={r.get('refit_steps', '')}"
            ),
        }
        top_m = r.get("top_m", 4)
        pairs = norm_pairs(r.get("true_pair"), meta.get("pairs", []))
        support = parse_obj(r.get("true_variables"), meta.get("support", []))
        endpoints = sorted({v for p in pairs for v in p}) or meta.get("endpoints", [])

        add(rows, **common, evidence_object="prediction", claim_type="prediction", target="low_mse", scorer="mse", predicate="mse_lt", threshold=0.05, raw_value=r.get("test_mse", ""))
        add(rows, **common, evidence_object="full_function", claim_type="pair", target=target_str(pairs), scorer="functional_anova", predicate="rank1", rank=r.get("full_pair_rank", ""), margin=r.get("full_pair_margin", ""))
        add(rows, **common, evidence_object="exposed_readout", claim_type="endpoints", target=target_str(endpoints), scorer="KAN-FE", predicate="top_m_contains_all", budget=top_m, rank=r.get("readout_worst_endpoint_rank", ""), margin=r.get("readout_endpoint_margin", ""))
        add(rows, **common, evidence_object="selected_support", claim_type="support", target=target_str(support), scorer="KAN-FE", predicate="contains_all", budget=top_m, selected_set=r.get("selected_support", ""))
        add(rows, **common, evidence_object="support_refit", claim_type="pair", target=target_str(pairs), scorer="functional_anova", predicate="rank1", rank=r.get("refit_pair_rank", ""), margin=r.get("refit_pair_margin", ""))
        if "prune_endpoint_contains" in r:
            add(rows, **common, evidence_object="pruning", claim_type="endpoints", target=target_str(endpoints), scorer="prune_input", predicate="binary_true", raw_value=r.get("prune_endpoint_contains", ""), selected_set=r.get("prune_selected_inputs", ""))
        if "symbolic_formula_ok" in r:
            add(rows, **common, evidence_object="symbolic", claim_type="symbolic_status", target="syntactic_expression", scorer="pyKAN_symbolic", predicate="binary_true", raw_value=r.get("symbolic_formula_ok", ""))


def convert_cross_method(path: Path, rows: list[dict[str, Any]]) -> None:
    df = pd.read_csv(path)
    for _, r in df.iterrows():
        function = str(r["function"])
        tid, family, meta = meta_for_row(function, r)
        method = str(r.get("method", "external"))
        common = {
            "task_id": tid,
            "task_family": family,
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
        if method == "symbolic_lasso":
            has_symbolic_terms = bool(flatten_ints(r.get("selected_support", ""))) or bool(parse_obj(r.get("selected_pairs", ""), []))
            complexity = as_float(r.get("num_symbolic_main_features", "")) + as_float(
                r.get("num_symbolic_pair_features", "")
            )
            add(
                rows,
                **common,
                evidence_object=str(r.get("evidence_object", "fixed_symbolic_library")),
                claim_type="symbolic_status",
                target="library_expression",
                scorer=method,
                predicate="binary_true",
                raw_value=float(has_symbolic_terms),
                selected_set=r.get("selected_support", ""),
                candidate_set=r.get("selected_pairs", ""),
            )
            add(
                rows,
                **common,
                evidence_object=str(r.get("evidence_object", "fixed_symbolic_library")),
                claim_type="symbolic_complexity",
                target="max_complexity",
                scorer=method,
                predicate="complexity_le",
                threshold=12,
                raw_value=complexity,
                selected_set=r.get("selected_support", ""),
                candidate_set=r.get("selected_pairs", ""),
            )


def convert_symbolic_expression(path: Path, rows: list[dict[str, Any]]) -> None:
    df = pd.read_csv(path)
    for _, r in df.iterrows():
        task_id_value = str(r.get("task_id", "feynman_style_energy_hidden_template") or "feynman_style_energy_hidden_template")
        function = str(r.get("function", "feynman_energy") or "feynman_energy")
        tid, family, _meta = meta_for_row(function, r)
        if task_id_value:
            tid = task_id_value
        family = str(r.get("task_family", family) or family)
        adapter = str(r.get("adapter", "symbolic_library_diagnostic") or "symbolic_library_diagnostic")
        common = {
            "task_id": tid,
            "task_family": family,
            "adapter": adapter,
            "adapter_family": str(r.get("adapter_family", "symbolic_library") or "symbolic_library"),
            "source_kind": "symbolic_expression_operator_recall",
            "source_file": str(path),
            "seed": int(r.get("seed", 0) or 0),
            "protocol": str(r.get("protocol", "fixed_symbolic_expression_library") or "fixed_symbolic_expression_library"),
        }
        target_ops = str(r.get("target_operators", "plus,multiply,power") or "plus,multiply,power")
        observed_ops = str(r.get("observed_operators", "") or "")
        expression = str(r.get("expression", "") or "")
        complexity = r.get("expression_complexity", "")
        support = r.get("selected_support", "")
        pair_terms = r.get("selected_pairs", "")
        status_raw = r.get("symbolic_status", "")
        if status_raw == "":
            status_raw = float(bool(expression))

        add(
            rows,
            **common,
            evidence_object=str(r.get("evidence_object", "symbolic_expression") or "symbolic_expression"),
            claim_type="symbolic_operator_recall",
            target=target_ops,
            scorer=str(r.get("scorer", "symbolic_expression_quality") or "symbolic_expression_quality"),
            predicate="operator_recall_ge",
            threshold=r.get("operator_recall_threshold", 0.95),
            raw_value=r.get("operator_recall", ""),
            selected_set=observed_ops,
            candidate_set=pair_terms,
        )
        add(
            rows,
            **common,
            evidence_object=str(r.get("evidence_object", "symbolic_expression") or "symbolic_expression"),
            claim_type="symbolic_complexity",
            target="max_complexity",
            scorer=str(r.get("scorer", "symbolic_expression_quality") or "symbolic_expression_quality"),
            predicate="complexity_le",
            threshold=r.get("complexity_threshold", 12),
            raw_value=complexity,
            selected_set=support,
            candidate_set=pair_terms,
        )
        add(
            rows,
            **common,
            evidence_object=str(r.get("evidence_object", "symbolic_expression") or "symbolic_expression"),
            claim_type="symbolic_status",
            target="expression_returned",
            scorer=str(r.get("scorer", "symbolic_expression_quality") or "symbolic_expression_quality"),
            predicate="binary_true",
            raw_value=status_raw,
            selected_set=support,
            candidate_set=pair_terms,
        )


def convert_treegate(path: Path, rows: list[dict[str, Any]]) -> None:
    df = pd.read_csv(path)
    for _, r in df.iterrows():
        if str(r.get("status", "ok")) != "ok":
            continue
        function = str(r["function"])
        tid, family, meta = meta_for_row(function, r)
        forest = str(r.get("forest_type", "tree"))
        gate = str(r.get("gate_score", "gate"))
        adapter = f"TreeGate-{forest}-{gate}"
        common = {
            "task_id": tid,
            "task_family": family,
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
        tid, family, meta = meta_for_row(function, r)
        scorer = str(r.get("pair_scorer", "pair_scorer"))
        common = {
            "task_id": tid,
            "task_family": family,
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


def convert_epim_pairverify(path: Path, rows: list[dict[str, Any]]) -> None:
    df = pd.read_csv(path)
    for _, r in df.iterrows():
        if str(r.get("status", "ok")) != "ok":
            continue
        function = str(r["function"])
        tid, family, meta = meta_for_row(function, r)
        q = r.get("proposal_q", "")
        common = {
            "task_id": tid,
            "task_family": family,
            "adapter": "EPIM-PairVerify",
            "adapter_family": "epim_pairverify",
            "source_kind": "epim_pairverify",
            "source_file": str(path),
            "seed": int(r["seed"]),
            "protocol": f"proposal_q={q}",
        }
        pairs = norm_pairs(r.get("true_pair"), meta.get("pairs", []))
        endpoints = sorted({v for p in pairs for v in p}) or meta.get("endpoints", [])

        add(rows, **common, evidence_object="prediction", claim_type="prediction", target="low_mse", scorer="mse", predicate="mse_lt", threshold=0.05, raw_value=r.get("test_mse", ""))
        add(rows, **common, evidence_object="epim_endpoint_proposal", claim_type="endpoints", target=target_str(endpoints), scorer="EPIM", predicate="binary_true", budget=q, raw_value=r.get("epim_endpoint_contains_true_pair", ""), rank=r.get("epim_true_pair_rank", ""))
        add(rows, **common, evidence_object="epim_pair_proposal", claim_type="candidate_pair", target=target_str(pairs), scorer="EPIM", predicate="binary_true", budget=q, raw_value=r.get("epim_proposal_contains_true_pair", ""), rank=r.get("epim_true_pair_rank", ""), candidate_set=r.get("epim_top_pairs", ""))
        add(rows, **common, evidence_object="pairverify_practical", claim_type="pair", target=target_str(pairs), scorer="candidate_functional_anova", predicate="binary_true", budget=q, raw_value=r.get("practical_verified_top_is_true_pair", ""), rank=r.get("verified_true_pair_rank", ""), margin=r.get("verified_true_minus_max_candidate_false", ""), candidate_set=r.get("epim_top_pairs", ""))
        add(rows, **common, evidence_object="pairverify_probe", claim_type="pair", target=target_str(pairs), scorer="candidate_functional_anova_oracle", predicate="binary_true", budget=q, raw_value=r.get("verified_top_is_true_pair", ""), rank=r.get("verified_true_pair_rank", ""), margin=r.get("verified_true_minus_max_candidate_false", ""), candidate_set=r.get("epim_top_pairs", ""))


def convert_semisynthetic(path: Path, rows: list[dict[str, Any]]) -> None:
    df = pd.read_csv(path)
    for _, r in df.iterrows():
        dataset = str(r.get("dataset", "")).strip()
        meta = SEMISYNTH_META.get(dataset)
        if meta is None:
            continue
        method = str(r.get("method", "semisynthetic_screen"))
        common = {
            "task_id": meta["task_id"],
            "task_family": meta["family"],
            "adapter": f"pyKAN-semisynthetic-{method}",
            "adapter_family": "pyKAN",
            "source_kind": "semisynthetic_covariate_audit",
            "source_file": str(path),
            "seed": int(r["outer_seed"]),
            "protocol": f"{dataset}:{method}:top{r.get('top_m', '')}",
            "runtime_seconds": r.get("runtime_sec", ""),
        }
        pairs = meta["pairs"]
        endpoints = meta["endpoints"]
        support = meta["support"]
        add(
            rows,
            **common,
            evidence_object="prediction",
            claim_type="prediction",
            target="low_mse",
            scorer="mse",
            predicate="mse_lt",
            threshold=0.05,
            raw_value=r.get("probe_test_mse_min", r.get("probe_test_mse_mean", "")),
        )
        add(
            rows,
            **common,
            evidence_object="semisynthetic_screen",
            claim_type="support",
            target=target_str(support),
            scorer=method,
            predicate="contains_all",
            selected_set=r.get("selected_screen_features", r.get("top_selection_variables", "")),
        )
        add(
            rows,
            **common,
            evidence_object="semisynthetic_screen",
            claim_type="endpoints",
            target=target_str(endpoints),
            scorer=method,
            predicate="binary_true",
            raw_value=r.get("screen_contains_all_interaction_endpoints", ""),
            selected_set=r.get("top_selection_variables", ""),
        )
        add(
            rows,
            **common,
            evidence_object="residual_pair_screen",
            claim_type="pair",
            target=target_str(pairs),
            scorer="residual_functional_anova",
            predicate="rank_at_budget",
            budget=len(pairs) or 1,
            rank=r.get("residual_true_pair_rank_worst", ""),
            margin=(
                as_float(r.get("residual_true_pair_score_mean", ""))
                - as_float(r.get("residual_max_false_pair_score", ""))
                if "residual_true_pair_score_mean" in r and "residual_max_false_pair_score" in r
                else ""
            ),
        )


def convert_prune_symbolic(path: Path, rows: list[dict[str, Any]]) -> None:
    df = pd.read_csv(path)
    for _, r in df.iterrows():
        function = str(r["function"])
        meta = FORMULA_META.get(function, {})
        if not meta:
            continue
        tid = task_id(function, r)
        workflow = str(r.get("workflow", "prune"))
        threshold = r.get("threshold", "")
        common = {
            "task_id": tid,
            "task_family": meta.get("family", function),
            "adapter": "pyKAN-prune-symbolic",
            "adapter_family": "pyKAN",
            "source_kind": "pykan_prune_symbolic",
            "source_file": str(path),
            "seed": int(r["seed"]),
            "protocol": f"{workflow}:threshold={threshold}",
        }
        pairs = meta.get("pairs", [])
        endpoints = sorted({v for p in pairs for v in p}) or meta.get("endpoints", [])
        support = meta.get("support", [])
        add(
            rows,
            **common,
            evidence_object="prediction",
            claim_type="prediction",
            target="low_mse",
            scorer="mse",
            predicate="mse_lt",
            threshold=0.05,
            raw_value=r.get("full_mse", ""),
        )
        add(
            rows,
            **common,
            evidence_object=workflow,
            claim_type="support",
            target=target_str(support),
            scorer="prune_input",
            predicate="contains_all",
            selected_set=r.get("selected_inputs", ""),
        )
        add(
            rows,
            **common,
            evidence_object=workflow,
            claim_type="endpoints",
            target=target_str(endpoints),
            scorer="prune_input",
            predicate="binary_true",
            raw_value=r.get("endpoint_contains", ""),
            rank=r.get("endpoint_rank_feature", ""),
            selected_set=r.get("selected_inputs", ""),
        )
        add(
            rows,
            **common,
            evidence_object="symbolic",
            claim_type="symbolic_status",
            target="syntactic_expression",
            scorer="pyKAN_symbolic",
            predicate="binary_true",
            raw_value=r.get("symbolic_formula_ok", ""),
            selected_set=r.get("selected_inputs", ""),
        )


def convert_pair_feature_lasso(path: Path, rows: list[dict[str, Any]]) -> None:
    df = pd.read_csv(path)
    for _, r in df.iterrows():
        function = str(r["function"])
        tid, family, meta = meta_for_row(function, r)
        common = {
            "task_id": tid,
            "task_family": family,
            "adapter": "pair_feature_lasso",
            "adapter_family": "sparse_lasso",
            "source_kind": "pair_feature_lasso",
            "source_file": str(path),
            "seed": int(r["seed"]),
            "protocol": f"alpha={r.get('alpha', '')}:top_m={r.get('top_m', '')}",
        }
        pairs = meta.get("pairs", [])
        endpoints = sorted({v for p in pairs for v in p}) or meta.get("endpoints", [])
        add(
            rows,
            **common,
            evidence_object="selected_variables",
            claim_type="support",
            target=target_str(meta.get("support", [])),
            scorer="pair_feature_lasso",
            predicate="contains_all",
            selected_set=r.get("selected_support", r.get("selected_variables", "")),
        )
        add(
            rows,
            **common,
            evidence_object="selected_variables",
            claim_type="endpoints",
            target=target_str(endpoints),
            scorer="pair_feature_lasso",
            predicate="binary_true",
            raw_value=float(as_float(r.get("endpoint_recall_at_m", "")) >= 0.999),
            selected_set=r.get("selected_variables", ""),
        )
        add(
            rows,
            **common,
            evidence_object="selected_interactions",
            claim_type="pair",
            target=target_str(pairs),
            scorer="pair_feature_lasso",
            predicate="binary_true",
            raw_value=r.get("pair_retained_at_m", r.get("top1_pair_accuracy", "")),
            margin=as_float(r.get("true_pair_score_mean", "")) - as_float(r.get("top_pair_score", "")),
            candidate_set=r.get("selected_interactions", ""),
        )


def convert_residual_hsic(path: Path, rows: list[dict[str, Any]]) -> None:
    df = pd.read_csv(path)
    for _, r in df.iterrows():
        function = str(r["function"])
        tid, family, meta = meta_for_row(function, r)
        common = {
            "task_id": tid,
            "task_family": family,
            "adapter": "residual_rff_hsic_pair_screen",
            "adapter_family": "epim_pairverify",
            "source_kind": "residual_rff_hsic_pair_screen",
            "source_file": str(path),
            "seed": int(r["seed"]),
            "protocol": f"rff={r.get('rff_dim', '')}:top_pairs={r.get('top_pairs_for_support', '')}",
            "runtime_seconds": r.get("runtime_sec", ""),
        }
        pairs = meta.get("pairs", [])
        endpoints = sorted({v for p in pairs for v in p}) or meta.get("endpoints", [])
        q = r.get("top_pairs_for_support", "")
        add(
            rows,
            **common,
            evidence_object="hsic_endpoint_proposal",
            claim_type="endpoints",
            target=target_str(endpoints),
            scorer="residual_rff_hsic",
            predicate="binary_true",
            budget=q,
            raw_value=float(as_float(r.get("endpoint_recall_at_top_pairs", "")) >= 0.999),
            candidate_set=r.get("selected_interactions", ""),
        )
        add(
            rows,
            **common,
            evidence_object="hsic_pair_proposal",
            claim_type="candidate_pair",
            target=target_str(pairs),
            scorer="residual_rff_hsic",
            predicate="binary_true",
            budget=q,
            raw_value=r.get("pair_retained_at_top_pairs", ""),
            rank=r.get("true_interaction_rank_worst", ""),
            candidate_set=r.get("selected_interactions", ""),
        )
        add(
            rows,
            **common,
            evidence_object="hsic_pair_screen",
            claim_type="pair",
            target=target_str(pairs),
            scorer="residual_rff_hsic",
            predicate="rank_at_budget",
            budget=len(pairs) or 1,
            rank=r.get("true_interaction_rank_worst", ""),
            margin=as_float(r.get("true_pair_score_mean", "")) - as_float(r.get("max_false_pair_score", "")),
            candidate_set=r.get("selected_interactions", ""),
        )


def convert_normalized_adapter_outputs(path: Path, rows: list[dict[str, Any]]) -> None:
    """Pass through already-normalized ClaimTransfer adapter-output rows."""
    df = pd.read_csv(path)
    for _, r in df.iterrows():
        add(rows, **{col: r.get(col, "") for col in df.columns})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default="results/revision")
    parser.add_argument("--out", default="claim_records/released_adapter_outputs.csv")
    parser.add_argument(
        "--split",
        default="public",
        help="Split label to attach to released evidence rows.",
    )
    parser.add_argument(
        "--registry-version",
        default="claimtransfer_v0_public",
        help="Task-card registry version to attach to released evidence rows.",
    )
    args = parser.parse_args()

    root = Path(args.results_root)
    rows: list[dict[str, Any]] = []

    patterns = [
        ("**/seed_aligned_stage_records_detail.csv", convert_stage_detail),
        ("**/symbolic_expression_detail.csv", convert_symbolic_expression),
        ("**/cross_method_transfer_detail.csv", convert_cross_method),
        ("**/treegate_pair_screen_detail.csv", convert_treegate),
        ("**/pair_scorer_claim_grammar_detail.csv", convert_scorergram),
        ("**/epim_pairverify_detail.csv", convert_epim_pairverify),
        ("**/semisynthetic_covariate_audit_detail.csv", convert_semisynthetic),
        ("**/pykan_prune_symbolic_detail.csv", convert_prune_symbolic),
        ("**/pair_feature_lasso_detail.csv", convert_pair_feature_lasso),
        ("**/residual_rff_hsic_pair_screen_detail.csv", convert_residual_hsic),
        ("**/standard_formula_adapter_outputs.csv", convert_normalized_adapter_outputs),
        ("**/gplearn_standard_formula_adapter_outputs.csv", convert_normalized_adapter_outputs),
        ("**/pysr_standard_formula_adapter_outputs.csv", convert_normalized_adapter_outputs),
        ("**/mlp_hessian_standard_formula_adapter_outputs.csv", convert_normalized_adapter_outputs),
    ]
    for pattern, converter in patterns:
        for path in sorted(root.glob(pattern)):
            converter(path, rows)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if not df.empty:
        if "split" not in df.columns:
            df["split"] = args.split
        else:
            split_text = df["split"].astype(str)
            df.loc[df["split"].isna() | split_text.str.strip().eq(""), "split"] = args.split
        if "registry_version" not in df.columns:
            df["registry_version"] = args.registry_version
        else:
            registry_text = df["registry_version"].astype(str)
            df.loc[
                df["registry_version"].isna() | registry_text.str.strip().eq(""),
                "registry_version",
            ] = args.registry_version
    df.to_csv(out, index=False)
    print(f"Wrote {out} ({len(df)} raw evidence rows)")
    if not df.empty:
        print(df.groupby(["source_kind", "claim_type"]).size().to_string())


if __name__ == "__main__":
    main()
