#!/usr/bin/env python3
"""Map ClaimTransfer coverage gaps to concrete GL/merge actions.

The gap report is the source of truth for what is missing.  This script adds a
thin operational layer: which existing Great Lakes job family, merge step, or
future symbolic track is expected to close each missing cell.  It does not mark
coverage as complete; it makes the remaining P1 data blocker auditable.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


CROSS_METHOD_TASK_FAMILIES = {
    "trig_product",
    "exp_product",
    "log_product",
    "sqrt_energy",
    "three_way_product",
    "scientific_expression",
    "correlated_covariates",
    "semi_breast_cancer",
    "semi_diabetes",
    "semi_wine",
}

TREEGATE_TASK_FAMILIES = {
    "trig_product",
    "exp_product",
    "log_product",
    "sqrt_energy",
    "three_way_product",
    "scientific_expression",
    "correlated_covariates",
    "semi_breast_cancer",
    "semi_diabetes",
    "semi_wine",
}


@dataclass(frozen=True)
class ActionRule:
    action_id: str
    action_type: str
    priority: str
    adapter_families: set[str]
    task_families: set[str] | None
    claim_types: set[str] | None
    command: str
    expected_output_roots: str
    note: str

    def matches(self, row: pd.Series) -> bool:
        if str(row["canonical_adapter_family"]) not in self.adapter_families:
            return False
        if self.task_families is not None and str(row["task_family"]) not in self.task_families:
            return False
        if self.claim_types is not None and str(row["claim_type"]) not in self.claim_types:
            return False
        return True


RULES = [
    ActionRule(
        action_id="cross_method_gapfill_standard",
        action_type="gl_standard_cpu",
        priority="P1",
        adapter_families={"ga2m_additive", "sparse_library", "symbolic_library", "tree_interaction"},
        task_families=CROSS_METHOD_TASK_FAMILIES,
        claim_types={"prediction", "support", "endpoints", "pair", "candidate_pair"},
        command="bash scripts/submit_claimtransfer_gapfill_gl.sh",
        expected_output_roots=(
            "results/revision/cross_method_transfer_gapfill; "
            "dependent score refresh via scripts/greatlakes_claimtransfer_score_refresh_standard.sbatch"
        ),
        note=(
            "Targets public task families absent from the original cross-method sweep. "
            "Verify actual closure by rerunning run_benchmark.py --quick --rebuild-adapter-outputs after merge."
        ),
    ),
    ActionRule(
        action_id="treegate_gapfill_standard",
        action_type="gl_standard_cpu",
        priority="P1",
        adapter_families={"epim_treegate"},
        task_families=TREEGATE_TASK_FAMILIES,
        claim_types={"prediction", "candidate_pair", "pair", "support"},
        command="SUBMIT_XFER=0 SUBMIT_TREEGATE=1 bash scripts/submit_claimtransfer_gapfill_gl.sh",
        expected_output_roots=(
            "results/revision/treegate_pair_screen_gapfill; "
            "dependent score refresh via scripts/greatlakes_claimtransfer_score_refresh_standard.sbatch"
        ),
        note=(
            "Targets candidate/verifier and TreeGate-style rows for gap families. "
            "Includes correlated-covariate, formula, scientific-expression, and semi-synthetic gap rows."
        ),
    ),
    ActionRule(
        action_id="pykan_merge_or_gapfill",
        action_type="merge_or_gl_gpu",
        priority="P1",
        adapter_families={"pykan_workflow"},
        task_families=None,
        claim_types=None,
        command=(
            "python scripts/refresh_from_greatlakes_results.py --latest "
            "or submit targeted pyKAN claimcard/scorergram rows if no result pack exists"
        ),
        expected_output_roots="results/revision/claimtransfer_*; results/revision/pair_scorer_claim_grammar_a40",
        note="These gaps usually close by merging already-finished claimcard/scorergram/readout outputs.",
    ),
    ActionRule(
        action_id="symbolic_expression_track",
        action_type="optional_track",
        priority="P2",
        adapter_families={"symbolic_library"},
        task_families={"scientific_expression"},
        claim_types={"symbolic_operator_recall", "symbolic_complexity"},
        command=(
            "sbatch --account=$ACCOUNT --array=0-3 "
            "--export=ALL,PYTHON_BIN=$PWD/.venv/bin/python "
            "scripts/greatlakes_symbolic_expression_operator_recall_standard.sbatch"
        ),
        expected_output_roots="results/revision/symbolic_expression_operator_recall",
        note=(
            "Expression-level symbolic claims are optional unless promoted to a primary benchmark track; "
            "this diagnostic sweep exercises the official operator-recall scorer."
        ),
    ),
]


def compact(values: list[str], limit: int = 8) -> str:
    vals = sorted(set(map(str, values)))
    if len(vals) <= limit:
        return ", ".join(vals)
    return ", ".join(vals[:limit]) + f", ... (+{len(vals) - limit})"


def to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "# Coverage gap action plan\n\nNo missing cells.\n"
    lines = ["# Coverage gap action plan", ""]
    lines.append(
        "| action | type | priority | targeted missing cells | adapter families | task families | claim types | command | note |"
    )
    lines.append("| --- | --- | --- | ---: | --- | --- | --- | --- | --- |")
    for row in df.itertuples(index=False):
        lines.append(
            f"| {row.action_id} | {row.action_type} | {row.priority} | {row.targeted_missing_cells} | "
            f"{row.adapter_families} | {row.task_families} | {row.claim_types} | `{row.command}` | {row.note} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="score_reports/coverage_gap_report.csv")
    parser.add_argument("--out", default="score_reports/coverage_gap_action_plan.csv")
    args = parser.parse_args()

    src = ROOT / args.input
    if not src.exists():
        raise SystemExit(f"Missing coverage gap report: {src}")
    gap = pd.read_csv(src)
    missing = gap[gap["coverage_status"] != "covered"].copy()

    rows = []
    assigned = pd.Series(False, index=missing.index)
    for rule in RULES:
        mask = missing.apply(rule.matches, axis=1) & ~assigned
        block = missing[mask]
        if block.empty:
            continue
        assigned.loc[block.index] = True
        rows.append(
            {
                "action_id": rule.action_id,
                "action_type": rule.action_type,
                "priority": rule.priority,
                "targeted_missing_cells": int(len(block)),
                "adapter_families": compact(block["canonical_adapter_family"].tolist()),
                "task_families": compact(block["task_family"].tolist()),
                "claim_types": compact(block["claim_type"].tolist()),
                "command": rule.command,
                "expected_output_roots": rule.expected_output_roots,
                "note": rule.note,
            }
        )

    residual = missing[~assigned]
    if not residual.empty:
        rows.append(
            {
                "action_id": "unassigned_gap",
                "action_type": "manual_triage",
                "priority": "P1",
                "targeted_missing_cells": int(len(residual)),
                "adapter_families": compact(residual["canonical_adapter_family"].tolist()),
                "task_families": compact(residual["task_family"].tolist()),
                "claim_types": compact(residual["claim_type"].tolist()),
                "command": "inspect score_reports/coverage_gap_report.csv and add a targeted action rule",
                "expected_output_roots": "",
                "note": "No current GL/merge action rule claims these gaps.",
            }
        )

    out_df = pd.DataFrame(rows).sort_values(
        ["priority", "targeted_missing_cells", "action_id"], ascending=[True, False, True]
    )
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out, index=False)
    out.with_suffix(".md").write_text(to_markdown(out_df), encoding="utf-8")
    print(f"Wrote {out} ({len(out_df)} actions)")
    if not out_df.empty:
        print(out_df[["action_id", "priority", "targeted_missing_cells", "command"]].to_string(index=False))


if __name__ == "__main__":
    main()
