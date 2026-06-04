#!/usr/bin/env python3
"""Build a ClaimTransfer coverage-gap report.

The ordinary coverage table reports observed rows.  This script adds the
benchmark-planning view: for each canonical adapter family, task family, and
legal claim type, it records whether the released outputs currently cover that
cell.

Expected cells are claim-grammar aware by default.  A task family is expected
to cover only the structural claim types declared by its task card, plus the
prediction adequacy gate.  Candidate-pair rows are expected only for adapters
that expose candidate pairs on task families with legal pair claims.  This keeps
the benchmark contract from silently turning optional expression-level claims
into requirements for every support/pair diagnostic card.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

ADAPTER_FAMILY_ALIASES = {
    "pyKAN": "pykan_workflow",
    "pykan": "pykan_workflow",
    "ga2m_spline": "ga2m_additive",
    "sparse_lasso": "sparse_library",
    "sparse_spline_lasso": "sparse_library",
    "symbolic_lasso": "symbolic_library",
    "gbm_hstat": "tree_interaction",
    "tree_gate": "tree_interaction",
    "epim_pairverify": "epim_treegate",
}


def claim_types_from_card(card: dict[str, Any]) -> set[str]:
    claim_types: set[str] = {"prediction"}
    spec = card.get("claim_specification", {})
    if not isinstance(spec, dict):
        return claim_types
    for value in spec.values():
        claims: list[Any]
        if isinstance(value, list):
            claims = value
        elif isinstance(value, dict):
            claims = [value]
        else:
            continue
        for claim in claims:
            if not isinstance(claim, dict):
                continue
            claim_type = str(claim.get("claim_type", "")).strip()
            if claim_type:
                claim_types.add(claim_type)
    return claim_types


def load_public_task_family_claim_types() -> dict[str, set[str]]:
    families: dict[str, set[str]] = {}
    for name in ["claimtransfer_v0_public.json", "claimtransfer_v1_scientific_templates.json"]:
        path = ROOT / "task_cards" / name
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for card in data.get("cards", []):
            family = str(card.get("task_family", "")).strip()
            if family:
                families.setdefault(family, set()).update(claim_types_from_card(card))
    return families


def load_adapter_contracts() -> list[dict[str, Any]]:
    path = ROOT / "adapters/adapter_family_registry.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(data.get("families", []))


def to_markdown(df: pd.DataFrame, max_rows: int = 100) -> str:
    if df.empty:
        return "No rows."
    table = df.head(max_rows).copy()
    for col in table.columns:
        if pd.api.types.is_float_dtype(table[col]):
            table[col] = table[col].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
    table = table.fillna("").astype(str)
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
    parser.add_argument("--coverage", default="score_reports/coverage_table.csv")
    parser.add_argument("--out", default="score_reports/coverage_gap_report.csv")
    parser.add_argument("--min-trials", type=int, default=1)
    parser.add_argument(
        "--expectation-mode",
        choices=["claim_grammar", "registry_product"],
        default="claim_grammar",
        help=(
            "claim_grammar expects only task-card legal claim types; "
            "registry_product uses the legacy adapter x task x licensed-claim grid."
        ),
    )
    args = parser.parse_args()

    coverage_path = ROOT / args.coverage
    if not coverage_path.exists():
        raise SystemExit(f"Coverage table does not exist: {coverage_path}")
    coverage = pd.read_csv(coverage_path)
    coverage["canonical_adapter_family"] = coverage["adapter_family"].map(
        lambda x: ADAPTER_FAMILY_ALIASES.get(str(x), str(x))
    )

    grouped = (
        coverage.groupby(["canonical_adapter_family", "task_family", "claim_type"], dropna=False)
        .agg(
            actual_adapter_families=("adapter_family", lambda s: ",".join(sorted(set(map(str, s))))),
            score_rows=("score_rows", "sum"),
            report_rows=("report_rows", "sum"),
            trials=("trials", "sum"),
            missing_pass_rows=("missing_pass_rows", "sum"),
            median_rank=("median_rank", "median"),
            median_margin=("median_margin", "median"),
        )
        .reset_index()
    )
    observed = {
        (str(r.canonical_adapter_family), str(r.task_family), str(r.claim_type)): r
        for r in grouped.itertuples(index=False)
    }

    rows = []
    task_family_claim_types = load_public_task_family_claim_types()
    for contract in load_adapter_contracts():
        adapter_family = str(contract.get("adapter_family", ""))
        licensed = {str(c) for c in contract.get("licensed_claim_types", [])}
        for task_family, legal_claims in sorted(task_family_claim_types.items()):
            if args.expectation_mode == "registry_product":
                expected_claims = set(licensed)
            else:
                expected_claims = set(legal_claims) & licensed
                if "pair" in legal_claims and "candidate_pair" in licensed:
                    expected_claims.add("candidate_pair")
            for claim_type in sorted(expected_claims):
                key = (adapter_family, task_family, str(claim_type))
                got = observed.get(key)
                if got is None:
                    rows.append(
                        {
                            "canonical_adapter_family": adapter_family,
                            "task_family": task_family,
                            "claim_type": claim_type,
                            "coverage_status": "missing_cell",
                            "actual_adapter_families": "",
                            "score_rows": 0,
                            "report_rows": 0,
                            "trials": 0,
                            "missing_pass_rows": 0,
                            "median_rank": "",
                            "median_margin": "",
                        }
                    )
                    continue
                trials = int(got.trials)
                status = "covered" if trials >= args.min_trials else "insufficient_trials"
                rows.append(
                    {
                        "canonical_adapter_family": adapter_family,
                        "task_family": task_family,
                        "claim_type": claim_type,
                        "coverage_status": status,
                        "actual_adapter_families": got.actual_adapter_families,
                        "score_rows": int(got.score_rows),
                        "report_rows": int(got.report_rows),
                        "trials": trials,
                        "missing_pass_rows": int(got.missing_pass_rows),
                        "median_rank": got.median_rank,
                        "median_margin": got.median_margin,
                    }
                )

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows).sort_values(["coverage_status", "canonical_adapter_family", "task_family", "claim_type"])
    df.to_csv(out, index=False)
    out.with_suffix(".md").write_text("# Coverage gap report\n\n" + to_markdown(df) + "\n", encoding="utf-8")

    counts = df["coverage_status"].value_counts().to_dict()
    print(f"Wrote {out} ({len(df)} expected cells)")
    print("coverage status:", counts)


if __name__ == "__main__":
    main()
