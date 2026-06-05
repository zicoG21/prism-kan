#!/usr/bin/env python3
"""Run a lightweight standard-formula adapter sweep.

The sweep is deliberately dependency-light: it uses NumPy only and emits
normalized ClaimTransfer adapter-output rows.  It is a bridge toward a larger
v1 method sweep with PySR/SINDy/EQL/KAN variants, not a replacement for those
heavier adapters.
"""

from __future__ import annotations

import argparse
import json
import math
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def safe_eval_formula(formula: str, x: np.ndarray) -> np.ndarray:
    expr = formula.replace("^", "**")
    env: dict[str, Any] = {
        "np": np,
        "sin": np.sin,
        "cos": np.cos,
        "exp": np.exp,
        "log": np.log,
        "sqrt": np.sqrt,
        "pi": np.pi,
    }
    for j in range(x.shape[1]):
        env[f"x{j}"] = x[:, j]
    y = eval(expr, {"__builtins__": {}}, env)
    return np.asarray(y, dtype=float)


def load_cards(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(data["cards"] if isinstance(data, dict) and "cards" in data else [data])


def parse_seed_range(text: str) -> list[int]:
    seeds: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            seeds.extend(range(int(lo), int(hi) + 1))
        else:
            seeds.append(int(part))
    return sorted(set(seeds))


def standardize(train_y: np.ndarray, test_y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = float(train_y.mean())
    scale = float(train_y.std())
    if scale <= 1e-12:
        scale = 1.0
    return (train_y - mean) / scale, (test_y - mean) / scale


def make_data(card: dict[str, Any], seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    d = int(card["dimension"])
    n = int(card["samples"])
    cov = card.get("covariates", {})
    low = float(cov.get("low", -1.0))
    high = float(cov.get("high", 1.0))
    x_train = rng.uniform(low, high, size=(n, d))
    x_test = rng.uniform(low, high, size=(max(1024, n // 2), d))
    y_train = safe_eval_formula(str(card["formula"]), x_train)
    y_test = safe_eval_formula(str(card["formula"]), x_test)
    noise = float(card.get("noise", 0.0) or 0.0)
    if noise:
        y_train = y_train + rng.normal(0.0, noise, size=y_train.shape)
        y_test = y_test + rng.normal(0.0, noise, size=y_test.shape)
    y_train, y_test = standardize(y_train, y_test)
    return x_train, y_train, x_test, y_test


def support_claim(card: dict[str, Any]) -> list[int]:
    return list(map(int, card.get("support", [])))


def pair_claims(card: dict[str, Any]) -> list[tuple[int, int]]:
    claims = card.get("claim_specification", {}).get("pair_claims", [])
    out = []
    for claim in claims:
        target = claim.get("target", [])
        if isinstance(target, list) and len(target) == 2:
            out.append(tuple(sorted(map(int, target))))
    return out


def operator_target(card: dict[str, Any]) -> str:
    for claim in card.get("claim_specification", {}).get("symbolic_claims", []):
        if claim.get("claim_type") == "symbolic_operator_recall":
            return str(claim.get("target", ""))
    return ""


def complexity_threshold(card: dict[str, Any]) -> float:
    for claim in card.get("claim_specification", {}).get("symbolic_claims", []):
        if claim.get("claim_type") == "symbolic_complexity":
            return float(claim.get("threshold", 12.0))
    return 12.0


def pair_rank(scores: dict[tuple[int, int], float], targets: list[tuple[int, int]]) -> tuple[float, float]:
    if not targets or not scores:
        return (math.nan, math.nan)
    sorted_scores = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    ranks = {pair: i + 1 for i, (pair, _) in enumerate(sorted_scores)}
    target_scores = [scores.get(pair, -math.inf) for pair in targets]
    false_scores = [score for pair, score in scores.items() if pair not in set(targets)]
    worst_rank = max(ranks.get(pair, math.inf) for pair in targets)
    margin = min(target_scores) - (max(false_scores) if false_scores else -math.inf)
    return float(worst_rank), float(margin)


def ridge_fit(features: np.ndarray, y: np.ndarray, alpha: float = 1e-6) -> np.ndarray:
    xtx = features.T @ features
    reg = alpha * np.eye(xtx.shape[0])
    reg[0, 0] = 0.0
    return np.linalg.pinv(xtx + reg) @ features.T @ y


def main_and_pair_features(x: np.ndarray) -> tuple[np.ndarray, list[str], list[tuple[int, int] | None]]:
    cols = [np.ones(x.shape[0])]
    names = ["intercept"]
    pairs: list[tuple[int, int] | None] = [None]
    for j in range(x.shape[1]):
        cols.append(x[:, j])
        names.append(f"x{j}")
        pairs.append(None)
    for i, j in combinations(range(x.shape[1]), 2):
        cols.append(x[:, i] * x[:, j])
        names.append(f"x{i}:x{j}")
        pairs.append((i, j))
    return np.column_stack(cols), names, pairs


def corr_scores(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    scores = []
    y0 = y - y.mean()
    y_norm = np.linalg.norm(y0) + 1e-12
    for j in range(x.shape[1]):
        z = x[:, j] - x[:, j].mean()
        scores.append(abs(float(z @ y0 / ((np.linalg.norm(z) + 1e-12) * y_norm))))
    return np.asarray(scores)


def pair_corr_scores(x: np.ndarray, y: np.ndarray) -> dict[tuple[int, int], float]:
    y0 = y - y.mean()
    y_norm = np.linalg.norm(y0) + 1e-12
    out: dict[tuple[int, int], float] = {}
    for i, j in combinations(range(x.shape[1]), 2):
        z = x[:, i] * x[:, j]
        z = z - z.mean()
        out[(i, j)] = abs(float(z @ y0 / ((np.linalg.norm(z) + 1e-12) * y_norm)))
    return out


def operator_recall(target: str, observed: str) -> float:
    target_set = {s.strip().lower() for s in target.split(",") if s.strip()}
    observed_set = {s.strip().lower() for s in observed.split(",") if s.strip()}
    if not target_set:
        return math.nan
    return len(target_set & observed_set) / len(target_set)


def add(rows: list[dict[str, Any]], **kwargs: Any) -> None:
    base = {
        "registry_version": "claimtransfer_v1_standard_formula_public",
        "split": "public",
        "task_id": "",
        "task_family": "",
        "adapter": "",
        "adapter_family": "",
        "source_kind": "standard_formula_adapter_sweep",
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


def emit_common(
    rows: list[dict[str, Any]],
    card: dict[str, Any],
    adapter: str,
    adapter_family: str,
    seed: int,
    evidence_object: str,
    mse: float | None,
    support_selected: list[int],
    pair_scores: dict[tuple[int, int], float] | None,
    operators: str = "",
    complexity: float | None = None,
) -> None:
    common = {
        "task_id": card["task_id"],
        "task_family": card["task_family"],
        "adapter": adapter,
        "adapter_family": adapter_family,
        "seed": seed,
        "protocol": evidence_object,
    }
    support = support_claim(card)
    pairs = pair_claims(card)
    endpoints = sorted({v for pair in pairs for v in pair})
    if mse is not None:
        add(rows, **common, evidence_object="prediction", claim_type="prediction", target="low_mse", scorer="mse", predicate="mse_lt", threshold=0.05, raw_value=mse)
    add(rows, **common, evidence_object=evidence_object, claim_type="support", target=repr(support), scorer=adapter, predicate="contains_all", selected_set=repr(support_selected))
    if endpoints:
        add(rows, **common, evidence_object=evidence_object, claim_type="endpoints", target=repr(endpoints), scorer=adapter, predicate="binary_true", raw_value=float(set(endpoints).issubset(support_selected)), selected_set=repr(support_selected))
    if pairs and pair_scores is not None:
        rank, margin = pair_rank(pair_scores, pairs)
        add(rows, **common, evidence_object=evidence_object, claim_type="pair", target=repr(pairs), scorer=adapter, predicate="rank_at_budget", budget=len(pairs), rank=rank, margin=margin)
    if operators:
        target_ops = operator_target(card)
        recall = operator_recall(target_ops, operators)
        add(rows, **common, evidence_object="symbolic_expression", claim_type="symbolic_status", target="expression_returned", scorer="symbolic_expression_quality", predicate="binary_true", raw_value=1.0, selected_set=repr(support_selected))
        add(rows, **common, evidence_object="symbolic_expression", claim_type="symbolic_operator_recall", target=target_ops, scorer="symbolic_expression_quality", predicate="operator_recall_ge", threshold=0.95, raw_value=recall, selected_set=operators)
        if complexity is not None:
            add(rows, **common, evidence_object="symbolic_expression", claim_type="symbolic_complexity", target="max_complexity", scorer="symbolic_expression_quality", predicate="complexity_le", threshold=complexity_threshold(card), raw_value=complexity)


def run_card(card: dict[str, Any], seed: int, rows: list[dict[str, Any]]) -> None:
    x_train, y_train, x_test, y_test = make_data(card, seed)
    support = support_claim(card)
    pairs = pair_claims(card)
    all_pair_scores = pair_corr_scores(x_train, y_train)

    # Oracle symbolic adapter: exact task-card fields, for positive control.
    emit_common(
        rows,
        card,
        adapter="oracle_symbolic",
        adapter_family="symbolic_library",
        seed=seed,
        evidence_object="oracle_expression",
        mse=0.0,
        support_selected=support,
        pair_scores={pair: (1.0 if pair in set(pairs) else 0.0) for pair in combinations(range(int(card["dimension"])), 2)},
        operators=operator_target(card),
        complexity=max(1.0, complexity_threshold(card) - 2.0),
    )

    # Main-effect correlation: cheap support-only probe.
    scores = corr_scores(x_train, y_train)
    m = max(1, len(support))
    selected = list(map(int, np.argsort(-scores)[:m]))
    xtr = np.column_stack([np.ones(len(x_train)), x_train[:, selected]])
    xte = np.column_stack([np.ones(len(x_test)), x_test[:, selected]])
    coef = ridge_fit(xtr, y_train, alpha=1e-3)
    mse = float(np.mean((xte @ coef - y_test) ** 2))
    emit_common(rows, card, "main_effect_corr", "support_screen", seed, "main_effect_correlation", mse, selected, None)

    # Pair-correlation screen: candidate pair evidence without fitted expression.
    top_pairs = sorted(all_pair_scores, key=all_pair_scores.get, reverse=True)
    selected_pair_vars = sorted({v for pair in top_pairs[: max(1, len(pairs) or 1)] for v in pair})
    emit_common(rows, card, "pair_corr_screen", "pair_screen", seed, "pair_product_correlation", None, selected_pair_vars, all_pair_scores)

    # Degree-2 polynomial ridge: a lightweight sparse-library stand-in.
    train_feat, _names, pair_map = main_and_pair_features(x_train)
    test_feat, _, _ = main_and_pair_features(x_test)
    coef = ridge_fit(train_feat, y_train, alpha=1e-3)
    pred = test_feat @ coef
    mse = float(np.mean((pred - y_test) ** 2))
    main_abs = np.abs(coef[1 : 1 + x_train.shape[1]])
    pair_scores = {
        pair: abs(float(coef[k]))
        for k, pair in enumerate(pair_map)
        if pair is not None
    }
    var_score = main_abs.copy()
    for (i, j), val in pair_scores.items():
        var_score[i] = max(var_score[i], val)
        var_score[j] = max(var_score[j], val)
    selected = list(map(int, np.argsort(-var_score)[: max(1, len(support))]))
    emit_common(rows, card, "poly2_ridge", "sparse_library", seed, "degree2_library_coefficients", mse, selected, pair_scores)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-cards", default="task_cards/claimtransfer_v1_standard_formula_public.json")
    parser.add_argument("--seeds", default="0-4")
    parser.add_argument("--out-dir", default="results/revision/standard_formula_adapter_sweep")
    args = parser.parse_args()

    cards = load_cards(ROOT / args.task_cards)
    seeds = parse_seed_range(args.seeds)
    rows: list[dict[str, Any]] = []
    for card in cards:
        for seed in seeds:
            run_card(card, seed, rows)

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    detail = pd.DataFrame(rows)
    detail.to_csv(out_dir / "standard_formula_adapter_outputs.csv", index=False)
    summary = (
        detail.groupby(["task_family", "adapter", "claim_type"], dropna=False)
        .size()
        .reset_index(name="rows")
    )
    summary.to_csv(out_dir / "standard_formula_adapter_summary.csv", index=False)
    print(f"Wrote {out_dir / 'standard_formula_adapter_outputs.csv'} ({len(detail)} rows)")
    print(f"Wrote {out_dir / 'standard_formula_adapter_summary.csv'}")


if __name__ == "__main__":
    main()
