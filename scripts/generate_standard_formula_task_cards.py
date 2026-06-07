#!/usr/bin/env python3
"""Generate ClaimTransfer task cards for a small standard-formula subset.

This is the first v1 bridge from custom diagnostic cards to standard
formula-recovery families.  The formulas are intentionally small and
machine-readable: each card declares support, operator claims, and optional
pair claims before any adapter is run.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class FormulaSpec:
    task_id: str
    family: str
    formula: str
    dimension: int
    support: tuple[int, ...]
    pairs: tuple[tuple[int, int], ...]
    operators: tuple[str, ...]
    complexity: int
    stress_tags: tuple[str, ...]
    audit_purpose: str
    samples: int = 2048
    noise: float = 0.0
    low: float = -1.0
    high: float = 1.0


FORMULAS: tuple[FormulaSpec, ...] = (
    FormulaSpec(
        task_id="std_nguyen1_poly_d6_n2048",
        family="standard_srbench_polynomial",
        formula="x0^3 + x0^2 + x0",
        dimension=6,
        support=(0,),
        pairs=(),
        operators=("plus", "power"),
        complexity=5,
        stress_tags=("srbench_style", "univariate_expression"),
        audit_purpose="standard SR-style univariate polynomial; expression/operator recovery without pair claim",
    ),
    FormulaSpec(
        task_id="std_nguyen5_trig_d6_n2048",
        family="standard_srbench_trig",
        formula="sin(x0^2)*cos(x0) - 1",
        dimension=6,
        support=(0,),
        pairs=(),
        operators=("sin", "cos", "multiply", "power", "plus"),
        complexity=8,
        stress_tags=("srbench_style", "univariate_composition"),
        audit_purpose="standard SR-style trigonometric composition; symbolic status must not imply operator recovery",
    ),
    FormulaSpec(
        task_id="std_keijzer_log_d6_n2048",
        family="standard_srbench_log",
        formula="log(x0^2 + 1) + x1",
        dimension=6,
        support=(0, 1),
        pairs=(),
        operators=("log", "plus", "power"),
        complexity=7,
        stress_tags=("srbench_style", "additive_composition"),
        audit_purpose="standard SR-style log/additive card; support and operator claims are separate",
    ),
    FormulaSpec(
        task_id="std_bilinear_product_d8_n2048",
        family="standard_pair_product",
        formula="x0*x1 + 0.5*x2",
        dimension=8,
        support=(0, 1, 2),
        pairs=((0, 1),),
        operators=("plus", "multiply"),
        complexity=5,
        stress_tags=("positive_product_control", "pair_claim"),
        audit_purpose="aligned product positive control for pair and support claims",
    ),
    FormulaSpec(
        task_id="std_feynman_energy_d8_n2048",
        family="standard_feynman_style",
        formula="0.5*x0*x1^2 + x0*x2*x3",
        dimension=8,
        support=(0, 1, 2, 3),
        pairs=((0, 1),),
        operators=("plus", "multiply", "power"),
        complexity=8,
        stress_tags=("feynman_style", "three_way_stress"),
        audit_purpose="kinetic-plus-potential style card; pair claims are declared only for explicit bivariate term",
        low=0.2,
        high=2.0,
    ),
    FormulaSpec(
        task_id="std_feynman_gravity_d8_n2048",
        family="standard_feynman_style",
        formula="x0*x1/(x2^2 + 0.1)",
        dimension=8,
        support=(0, 1, 2),
        pairs=((0, 1), (0, 2), (1, 2)),
        operators=("multiply", "divide", "power", "plus"),
        complexity=7,
        stress_tags=("feynman_style", "rational_pair_claims"),
        audit_purpose="gravity/Coulomb-style rational product; all declared pair claims are scorer-indexed",
        low=0.2,
        high=2.0,
    ),
    FormulaSpec(
        task_id="std_ideal_gas_d8_n2048",
        family="standard_feynman_style",
        formula="x0*x1/(x2 + 1.5)",
        dimension=8,
        support=(0, 1, 2),
        pairs=((0, 1), (0, 2), (1, 2)),
        operators=("multiply", "divide", "plus"),
        complexity=6,
        stress_tags=("feynman_style", "division_mixed"),
        audit_purpose="ideal-gas-style ratio; support/pair/operator claims can split",
        low=0.2,
        high=2.0,
    ),
    FormulaSpec(
        task_id="std_damped_wave_d8_n2048",
        family="standard_feynman_style",
        formula="x0*exp(-x1*x2)*sin(x3*x2)",
        dimension=8,
        support=(0, 1, 2, 3),
        pairs=((1, 2), (2, 3)),
        operators=("multiply", "exp", "sin"),
        complexity=9,
        stress_tags=("feynman_style", "composition_pair_stress"),
        audit_purpose="damped-wave-style composition; pair evidence is scorer-sensitive",
    ),
    FormulaSpec(
        task_id="std_harmonic_period_d6_n2048",
        family="standard_feynman_style",
        formula="sqrt((x0 + 1.5)/(x1 + 1.5))",
        dimension=6,
        support=(0, 1),
        pairs=((0, 1),),
        operators=("sqrt", "divide", "plus"),
        complexity=7,
        stress_tags=("feynman_style", "division_sqrt"),
        audit_purpose="period-style sqrt ratio; operator and pair claims are distinct",
        low=0.2,
        high=2.0,
    ),
    FormulaSpec(
        task_id="std_rosenbrock_d6_n2048",
        family="standard_srbench_polynomial",
        formula="(x1 - x0^2)^2 + (1 - x0)^2",
        dimension=6,
        support=(0, 1),
        pairs=((0, 1),),
        operators=("plus", "multiply", "power"),
        complexity=10,
        stress_tags=("srbench_style", "nested_polynomial"),
        audit_purpose="Rosenbrock-style nested polynomial; pair claim tests nonlinear coupling",
    ),
    FormulaSpec(
        task_id="std_additive_no_pair_d8_n2048",
        family="standard_negative_control",
        formula="x0 + x1^2 + sin(x2)",
        dimension=8,
        support=(0, 1, 2),
        pairs=(),
        operators=("plus", "power", "sin"),
        complexity=7,
        stress_tags=("additive_only", "negative_pair_control"),
        audit_purpose="additive-only negative control; prediction/support claims must not imply pair recovery",
    ),
    FormulaSpec(
        task_id="std_additive_exp_log_d8_n2048",
        family="standard_negative_control",
        formula="exp(0.3*x0) + log(x1^2 + 1) + x2",
        dimension=8,
        support=(0, 1, 2),
        pairs=(),
        operators=("plus", "exp", "log", "power", "multiply"),
        complexity=10,
        stress_tags=("additive_only", "operator_mix"),
        audit_purpose="operator-rich additive card; symbolic/operator claims without pair claims",
    ),
    FormulaSpec(
        task_id="std_two_products_d8_n2048",
        family="standard_pair_product",
        formula="x0*x1 + x2*x3 + 0.2*x4",
        dimension=8,
        support=(0, 1, 2, 3, 4),
        pairs=((0, 1), (2, 3)),
        operators=("plus", "multiply"),
        complexity=8,
        stress_tags=("multi_pair_product", "positive_product_control"),
        audit_purpose="two independent products; pair claims require all declared pair rows",
    ),
    FormulaSpec(
        task_id="std_trig_product_d8_n2048",
        family="standard_pair_product",
        formula="sin(x0)*x1 + cos(x2)",
        dimension=8,
        support=(0, 1, 2),
        pairs=((0, 1),),
        operators=("plus", "multiply", "sin", "cos"),
        complexity=7,
        stress_tags=("trig_product", "pair_claim"),
        audit_purpose="trigonometric product card; prediction and pair evidence can split",
    ),
    FormulaSpec(
        task_id="std_rational_plus_product_d8_n2048",
        family="standard_rational_pair",
        formula="x0/(1 + x1^2) + x2*x3",
        dimension=8,
        support=(0, 1, 2, 3),
        pairs=((0, 1), (2, 3)),
        operators=("plus", "divide", "power", "multiply"),
        complexity=9,
        stress_tags=("rational_pair_claims", "multi_pair"),
        audit_purpose="rational plus product; support, partial-pair, and all-pair claims are separated",
    ),
    FormulaSpec(
        task_id="std_exp_product_d8_n2048",
        family="standard_compositional_pair",
        formula="exp(0.2*x0*x1) + x2",
        dimension=8,
        support=(0, 1, 2),
        pairs=((0, 1),),
        operators=("plus", "exp", "multiply"),
        complexity=7,
        stress_tags=("composition_pair_stress", "exp_product"),
        audit_purpose="exponential product; expression status must not imply pair/operator recovery",
    ),
    FormulaSpec(
        task_id="std_log_product_d8_n2048",
        family="standard_compositional_pair",
        formula="log(2.5 + x0*x1) + x2",
        dimension=8,
        support=(0, 1, 2),
        pairs=((0, 1),),
        operators=("plus", "log", "multiply"),
        complexity=7,
        stress_tags=("composition_pair_stress", "log_product"),
        audit_purpose="log product with positive offset; pair and operator claims are distinct",
    ),
    FormulaSpec(
        task_id="std_sqrt_product_d8_n2048",
        family="standard_compositional_pair",
        formula="sqrt(2.5 + x0*x1) + x2",
        dimension=8,
        support=(0, 1, 2),
        pairs=((0, 1),),
        operators=("plus", "sqrt", "multiply"),
        complexity=7,
        stress_tags=("composition_pair_stress", "sqrt_product"),
        audit_purpose="sqrt product with positive offset; pair evidence and symbolic operator evidence split",
    ),
    FormulaSpec(
        task_id="std_nested_pair_d8_n2048",
        family="standard_compositional_pair",
        formula="sin(x0 + x1*x2) + x3",
        dimension=8,
        support=(0, 1, 2, 3),
        pairs=((1, 2),),
        operators=("plus", "sin", "multiply"),
        complexity=8,
        stress_tags=("nested_pair", "grammar_stress"),
        audit_purpose="nested pair inside a sinusoid; pair scorer is a declared evidence object, not formula truth",
    ),
    FormulaSpec(
        task_id="std_three_way_product_d8_n2048",
        family="standard_higher_order",
        formula="x0*x1*x2 + 0.5*x3",
        dimension=8,
        support=(0, 1, 2, 3),
        pairs=((0, 1),),
        operators=("plus", "multiply"),
        complexity=7,
        stress_tags=("three_way_stress", "pairwise_boundary"),
        audit_purpose="three-way product stress card; pair claim is intentionally limited and scorer-indexed",
    ),
    FormulaSpec(
        task_id="std_highdim_sparse_product_d12_n2048",
        family="standard_pair_product",
        formula="x0*x1 + x2^2 + sin(x3)",
        dimension=12,
        support=(0, 1, 2, 3),
        pairs=((0, 1),),
        operators=("plus", "multiply", "power", "sin"),
        complexity=8,
        stress_tags=("sparse_highdim", "pair_claim"),
        audit_purpose="higher-dimensional sparse product; tests support budget and pair authorization",
    ),
    FormulaSpec(
        task_id="std_feynman_coulomb_d8_n2048",
        family="standard_feynman_style",
        formula="x0*x1/(x2^2 + 0.2)",
        dimension=8,
        support=(0, 1, 2),
        pairs=((0, 1), (0, 2), (1, 2)),
        operators=("multiply", "divide", "power", "plus"),
        complexity=7,
        stress_tags=("feynman_style", "rational_pair_claims"),
        audit_purpose="Coulomb-style rational product; multiple pair claims are scorer-indexed",
        low=0.2,
        high=2.0,
    ),
    FormulaSpec(
        task_id="std_feynman_ohm_d6_n2048",
        family="standard_feynman_style",
        formula="x0*x1",
        dimension=6,
        support=(0, 1),
        pairs=((0, 1),),
        operators=("multiply",),
        complexity=3,
        stress_tags=("feynman_style", "simple_product_control"),
        audit_purpose="Ohm/power-style product; simple positive-control pair card",
        low=0.2,
        high=2.0,
    ),
    FormulaSpec(
        task_id="std_feynman_lens_d6_n2048",
        family="standard_feynman_style",
        formula="1/(1/(x0 + 2) + 1/(x1 + 2))",
        dimension=6,
        support=(0, 1),
        pairs=((0, 1),),
        operators=("divide", "plus"),
        complexity=9,
        stress_tags=("feynman_style", "harmonic_ratio"),
        audit_purpose="thin-lens-style harmonic ratio; support/pair/operator claims can split",
        low=0.2,
        high=2.0,
    ),
    FormulaSpec(
        task_id="std_feynman_frequency_d6_n2048",
        family="standard_feynman_style",
        formula="sqrt((x0 + 1)*(x1 + 1))/(x2 + 2)",
        dimension=6,
        support=(0, 1, 2),
        pairs=((0, 1), (0, 2), (1, 2)),
        operators=("sqrt", "multiply", "divide", "plus"),
        complexity=10,
        stress_tags=("feynman_style", "composition_pair_stress"),
        audit_purpose="frequency-style sqrt ratio; expression quality and pair claims are distinct",
        low=0.2,
        high=2.0,
    ),
    FormulaSpec(
        task_id="std_nested_polynomial_pair_d8_n2048",
        family="standard_srbench_polynomial",
        formula="(x0*x1 + x2)^2 + x3",
        dimension=8,
        support=(0, 1, 2, 3),
        pairs=((0, 1),),
        operators=("plus", "multiply", "power"),
        complexity=9,
        stress_tags=("nested_polynomial", "pair_claim"),
        audit_purpose="nested polynomial pair; tests expression complexity versus pair recovery",
    ),
    FormulaSpec(
        task_id="std_trig_mixed_two_pair_d8_n2048",
        family="standard_compositional_pair",
        formula="sin(x0*x1) + cos(x2*x3)",
        dimension=8,
        support=(0, 1, 2, 3),
        pairs=((0, 1), (2, 3)),
        operators=("plus", "sin", "cos", "multiply"),
        complexity=10,
        stress_tags=("multi_pair", "trig_composition"),
        audit_purpose="two trigonometric products; all-pair claim is stricter than symbolic status",
    ),
    FormulaSpec(
        task_id="std_division_pair_d8_n2048",
        family="standard_rational_pair",
        formula="x0/(x1 + 2) + x2",
        dimension=8,
        support=(0, 1, 2),
        pairs=((0, 1),),
        operators=("plus", "divide"),
        complexity=6,
        stress_tags=("division_pair", "support_pair_split"),
        audit_purpose="division pair with additive nuisance; pair evidence and support evidence can split",
    ),
    FormulaSpec(
        task_id="std_feynman_kepler_d6_n2048",
        family="standard_feynman_style",
        formula="sqrt((x0 + 0.2)^3/(x1 + 0.2))",
        dimension=6,
        support=(0, 1),
        pairs=((0, 1),),
        operators=("sqrt", "divide", "power", "plus"),
        complexity=9,
        stress_tags=("feynman_style", "kepler_style"),
        audit_purpose="Kepler-style period relation; symbolic operator recovery is separate from pair recovery",
        low=0.2,
        high=2.0,
    ),
    FormulaSpec(
        task_id="std_affine_no_pair_d12_n2048",
        family="standard_negative_control",
        formula="0.5*x0 - 1.2*x4 + 0.3*x7",
        dimension=12,
        support=(0, 4, 7),
        pairs=(),
        operators=("plus", "multiply"),
        complexity=6,
        stress_tags=("additive_only", "negative_pair_control", "sparse_highdim"),
        audit_purpose="sparse affine negative control; pair claims are illegal despite easy prediction",
    ),
)


def with_suffix(task_id: str, suffix: str) -> str:
    return f"{task_id}_{suffix}"


def expanded_formulas(mode: str) -> list[FormulaSpec]:
    """Return the base cards plus deterministic v1 expansion variants."""
    specs = list(FORMULAS)
    if mode not in {"none", "noise", "full"}:
        raise ValueError(f"Unknown augment mode {mode!r}; expected none, noise, or full")

    if mode in {"noise", "full"}:
        specs.extend(
            replace(
                spec,
                task_id=with_suffix(spec.task_id, "noise005"),
                family=f"{spec.family}_noise005",
                noise=0.05,
                stress_tags=tuple((*spec.stress_tags, "noise005")),
                audit_purpose=f"{spec.audit_purpose}; noise variant for robustness of claim-transfer edges",
            )
            for spec in FORMULAS
        )

    if mode == "full":
        specs.extend(
            replace(
                spec,
                task_id=with_suffix(spec.task_id, f"d{max(16, spec.dimension + 8)}_n4096"),
                family=f"{spec.family}_highdim",
                dimension=max(16, spec.dimension + 8),
                samples=4096,
                stress_tags=tuple((*spec.stress_tags, "highdim_sparse")),
                audit_purpose=f"{spec.audit_purpose}; high-dimensional sparse variant for support/pair overclaim stress",
            )
            for spec in FORMULAS
        )

    return specs


def claim_id(task_id: str, claim_type: str, predicate: str, target: Any) -> str:
    return f"{task_id}:{claim_type}:{predicate}:{target}"


def make_card(spec: FormulaSpec, split: str, registry_version: str) -> dict[str, Any]:
    claims: dict[str, list[dict[str, Any]]] = {
        "support_claims": [
            {
                "claim_id": claim_id(spec.task_id, "support", "top_m_contains_all", list(spec.support)),
                "claim_type": "support",
                "target": list(spec.support),
                "official_scorer": "ranked_support_or_endpoint_score",
                "predicate": "top_m_contains_all",
                "m": len(spec.support),
            }
        ],
        "symbolic_claims": [
            {
                "claim_id": claim_id(spec.task_id, "symbolic_operator_recall", "operator_recall_ge", ",".join(spec.operators)),
                "claim_type": "symbolic_operator_recall",
                "target": ",".join(spec.operators),
                "official_scorer": "symbolic_expression_quality",
                "predicate": "operator_recall_ge",
                "threshold": 0.95,
            },
            {
                "claim_id": claim_id(spec.task_id, "symbolic_complexity", "complexity_le", spec.complexity + 2),
                "claim_type": "symbolic_complexity",
                "target": "max_complexity",
                "official_scorer": "symbolic_expression_quality",
                "predicate": "complexity_le",
                "threshold": float(spec.complexity + 2),
            },
        ],
    }
    if spec.pairs:
        endpoints = sorted({v for pair in spec.pairs for v in pair})
        claims["endpoint_claims"] = [
            {
                "claim_id": claim_id(spec.task_id, "endpoints", "top_m_contains_all", endpoints),
                "claim_type": "endpoints",
                "target": endpoints,
                "official_scorer": "ranked_support_or_endpoint_score",
                "predicate": "top_m_contains_all",
                "m": len(spec.support),
            }
        ]
        claims["pair_claims"] = [
            {
                "claim_id": claim_id(spec.task_id, "pair", "rank1", list(pair)),
                "claim_type": "pair",
                "target": list(pair),
                "official_scorer": "functional_anova",
                "predicate": "rank1",
            }
            for pair in spec.pairs
        ]

    return {
        "task_id": spec.task_id,
        "task_family": spec.family,
        "split": split,
        "registry_version": registry_version,
        "formula": spec.formula,
        "covariates": {"type": "independent_uniform", "low": spec.low, "high": spec.high},
        "dimension": spec.dimension,
        "samples": spec.samples,
        "noise": spec.noise,
        "target_standardization": "train_mean_train_scale",
        "support": list(spec.support),
        "claim_specification": claims,
        "seed_policy": {
            "train_test_seed_block": "declared_by_score_report",
            "adapter_seed_block": "declared_by_adapter",
            "hidden_seed_policy": "private_when_split_is_hidden",
        },
        "stress_tags": list(spec.stress_tags),
        "audit_purpose": spec.audit_purpose,
        "standard_source": "ClaimTransfer v1 curated SRBench/Feynman-style subset",
        "official_outputs": ["claim_records", "score_report", "coverage_table"],
    }


def to_markdown(cards: list[dict[str, Any]]) -> str:
    lines = [
        "# ClaimTransfer v1 standard-formula task cards",
        "",
        "This generated registry maps a compact SRBench/Feynman-style subset into",
        "ClaimTransfer task cards.  It is a public diagnostic wrapper, not a",
        "hosted private leaderboard.",
        "",
        "| task_id | family | support | pair claims | operators | purpose |",
        "| --- | --- | ---: | ---: | --- | --- |",
    ]
    for card in cards:
        spec = card["claim_specification"]
        n_pairs = len(spec.get("pair_claims", []))
        ops = [
            claim["target"]
            for claim in spec.get("symbolic_claims", [])
            if claim.get("claim_type") == "symbolic_operator_recall"
        ]
        lines.append(
            f"| `{card['task_id']}` | {card['task_family']} | {len(card['support'])} | {n_pairs} | {ops[0] if ops else ''} | {card.get('audit_purpose','')} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="task_cards/claimtransfer_v1_standard_formula_public.json")
    parser.add_argument("--doc", default="task_cards/standard_formula_task_card_map.md")
    parser.add_argument("--split", default="public")
    parser.add_argument("--registry-version", default="claimtransfer_v1_standard_formula_public")
    parser.add_argument(
        "--augment",
        choices=["none", "noise", "full"],
        default="none",
        help="Add deterministic noise/high-dimensional variants for larger standard-card sweeps.",
    )
    args = parser.parse_args()

    formulas = expanded_formulas(args.augment)
    cards = [make_card(spec, args.split, args.registry_version) for spec in formulas]
    registry = {
        "version": args.registry_version,
        "schema_version": "claimtransfer_task_card_schema_v1",
        "split": args.split,
        "cards": cards,
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
    doc = ROOT / args.doc
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text(to_markdown(cards), encoding="utf-8")
    print(f"Wrote {out} ({len(cards)} cards)")
    print(f"Wrote {doc}")


if __name__ == "__main__":
    main()
