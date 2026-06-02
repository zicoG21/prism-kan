#!/usr/bin/env python3
"""Build a compact seed-level evidence-transfer trace table.

The table is intentionally small and deterministic. It is not a population
estimate; it selects representative same-seed workflow records that illustrate
different transfer outcomes already present in the stage-record CSVs.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd


DEFAULT_PATTERNS = [
    "results/revision/local_gpu_anchor_trace_followup/**/seed_aligned_stage_records_detail.csv",
    "results/revision/local_gpu_highvalue_10h_queue/**/seed_aligned_stage_records_detail.csv",
    "results/revision/seed_aligned_stage_records*/**/seed_aligned_stage_records_detail.csv",
]


CASE_SPECS = [
    (
        "aligned positive",
        {"function": "formula_bilinear", "first_broken_stage": "none"},
        ["formula_bilinear_s372_387", "formula_bilinear_s316_331", "formula_bilinear_s300_307"],
    ),
    (
        "KAN aligned positive",
        {"setting": "core_clean_w32_n768_s372_387", "first_broken_stage": "none"},
        ["core_clean_w32_n768_s372_387", "core_clean_w32_n768_s316_331"],
    ),
    (
        "full-function bottleneck",
        {"setting": "formula_division_mixed_s372_387", "first_broken_stage": "full-model reliance"},
        ["formula_division_mixed_s372_387"],
    ),
    (
        "readout bottleneck",
        {"setting": "formula_mixed_sparse_s372_387", "first_broken_stage": "readout endpoints"},
        ["formula_mixed_sparse_s372_387"],
    ),
    (
        "pruning bottleneck",
        {"setting": "formula_mixed_sparse_s372_387", "first_broken_stage": "pruning"},
        ["formula_mixed_sparse_s372_387"],
    ),
    (
        "prediction boundary",
        {"setting": "formula_rational_product_s372_387", "first_broken_stage": "prediction"},
        ["formula_rational_product_s372_387"],
    ),
    (
        "grid-update bottleneck",
        {"setting": "core_grid_w16_n1024_s316_331", "first_broken_stage": "full-model reliance"},
        ["core_grid_w16_n1024_s316_331", "gridupdate_w16_n1024"],
    ),
]


def find_inputs(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(Path(".").glob(pattern))
    return sorted(set(paths))


def source_priority(path: str) -> int:
    if "local_gpu_anchor_trace_followup" in path:
        return 0
    if "local_gpu_highvalue_10h_queue" in path:
        return 1
    if "seed_aligned_stage_records" in path:
        return 2
    return 3


def read_records(paths: list[Path]) -> pd.DataFrame:
    rows = []
    for path in paths:
        df = pd.read_csv(path)
        if df.empty:
            continue
        df["source_path"] = str(path)
        df["_source_priority"] = source_priority(str(path))
        rows.append(df)
    if not rows:
        raise SystemExit("No seed-aligned stage-record detail CSVs found.")
    detail = pd.concat(rows, ignore_index=True, sort=False)
    key = [
        "setting",
        "function",
        "seed",
        "samples",
        "dimension",
        "noise",
        "update_grid",
        "width_hidden",
        "prune_threshold",
    ]
    key = [c for c in key if c in detail.columns]
    detail = detail.sort_values(["_source_priority", "source_path", "seed"])
    detail = detail.drop_duplicates(subset=key, keep="first")
    return detail.drop(columns=["_source_priority"], errors="ignore")


def finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def fmt_float(value: object, digits: int = 3) -> str:
    if not finite(value):
        return "--"
    value = float(value)
    if abs(value) < 1e-3 and value != 0:
        return f"{value:.1e}"
    return f"{value:.{digits}f}"


def fmt_rank_margin(rank: object, margin: object) -> str:
    if not finite(rank) or int(rank) < 0:
        return "--"
    return f"{int(rank)} / {fmt_float(margin, 3)}"


def fmt_support(value: object, max_chars: int = 18) -> str:
    s = str(value)
    if s == "nan":
        return "--"
    return s if len(s) <= max_chars else s[: max_chars - 1] + "..."


def select_cases(detail: pd.DataFrame) -> pd.DataFrame:
    selected = []
    used = set()
    for case, filters, preferred_settings in CASE_SPECS:
        g = detail.copy()
        for col, val in filters.items():
            if col not in g.columns:
                g = g.iloc[0:0]
                break
            g = g[g[col].astype(str) == str(val)]
        if g.empty:
            continue
        g["_pref"] = 999
        for idx, setting in enumerate(preferred_settings):
            g.loc[g["setting"].astype(str) == setting, "_pref"] = idx
        # Pick the smallest seed in the most preferred setting, avoiding repeats.
        g = g.sort_values(["_pref", "seed", "source_path"])
        row = None
        for _, candidate in g.iterrows():
            identity = (candidate.get("setting"), int(candidate.get("seed", -1)), case)
            if identity not in used:
                row = candidate.copy()
                used.add(identity)
                break
        if row is None:
            continue
        row["case"] = case
        selected.append(row)
    if not selected:
        raise SystemExit("No representative trace cases matched.")
    return pd.DataFrame(selected).drop(columns=["_pref"], errors="ignore")


def compact_table(selected: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in selected.iterrows():
        prune_ep = int(r.get("prune_endpoint_contains", 0)) if finite(r.get("prune_endpoint_contains", 0)) else 0
        symbolic = int(r.get("symbolic_formula_ok", 0)) if finite(r.get("symbolic_formula_ok", 0)) else 0
        rows.append(
            {
                "case": r["case"],
                "formula/setting": str(r.get("function", "")) if "formula_" in str(r.get("function", "")) else str(r.get("setting", "")),
                "seed": int(r.get("seed", -1)),
                "MSE": fmt_float(r.get("test_mse"), 3),
                "full rank/margin": fmt_rank_margin(r.get("full_pair_rank"), r.get("full_pair_margin")),
                "readout rank/margin": f"{int(r.get('readout_worst_endpoint_rank')) if finite(r.get('readout_worst_endpoint_rank')) else '--'} / {fmt_float(r.get('readout_endpoint_margin'), 3)}",
                "selected support": fmt_support(r.get("selected_support")),
                "refit rank/margin": fmt_rank_margin(r.get("refit_pair_rank"), r.get("refit_pair_margin")),
                "prune": f"{int(r.get('prune_support_size')) if finite(r.get('prune_support_size')) else '--'} vars, EP {prune_ep}, sym {symbolic}",
                "decision": str(r.get("first_broken_stage", "")),
            }
        )
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |"]
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in df.iterrows():
        vals = [str(row[c]).replace("|", "/") for c in cols]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def latex_escape(s: object) -> str:
    out = str(s)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        out = out.replace(old, new)
    return out


def write_latex(df: pd.DataFrame, path: Path) -> None:
    cols = [
        ("case", "Trace"),
        ("seed", "Seed"),
        ("MSE", "MSE"),
        ("full rank/margin", "Full pair"),
        ("readout rank/margin", "Readout EP"),
        ("selected support", "Support"),
        ("refit rank/margin", "Refit pair"),
        ("prune", "Prune/sym."),
        ("decision", "Decision"),
    ]
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        r"\caption{\textbf{Seed-level evidence-transfer traces.} Each row is one trained workflow, not a population estimate. Full/refit columns show rank / margin; readout shows worst endpoint rank / endpoint margin; prune reports retained support size, endpoint retention (EP), and symbolic status.}",
        r"\label{tab:seed-traces}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{llrllllll}",
        r"\toprule",
        " & ".join(label for _, label in cols) + r" \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        lines.append(" & ".join(latex_escape(row[c]) for c, _ in cols) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-prefix", default="local_notes/generated/seed_trace_table_20260601")
    parser.add_argument("--patterns", nargs="*", default=DEFAULT_PATTERNS)
    args = parser.parse_args()

    paths = find_inputs(args.patterns)
    detail = read_records(paths)
    selected = select_cases(detail)
    table = compact_table(selected)

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(prefix.with_suffix(".selected_detail.csv"), index=False)
    table.to_csv(prefix.with_suffix(".csv"), index=False)
    prefix.with_suffix(".md").write_text(
        "# Seed-Level Evidence-Transfer Traces\n\n"
        "These rows are case traces, not population estimates.\n\n"
        + markdown_table(table)
        + "\n",
        encoding="utf-8",
    )
    write_latex(table, prefix.with_suffix(".tex"))
    print(f"Wrote {prefix.with_suffix('.csv')}")
    print(f"Wrote {prefix.with_suffix('.tex')}")


if __name__ == "__main__":
    main()
