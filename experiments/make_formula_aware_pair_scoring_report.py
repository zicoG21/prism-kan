from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


METHOD_ORDER = ["fd", "anova_abs", "anova_var", "fd_anova_hybrid"]
METHOD_LABELS = {
    "fd": "FD",
    "anova_abs": "ANOVA-abs",
    "anova_var": "ANOVA-var",
    "fd_anova_hybrid": "Hybrid",
}
SUPPORT_LABELS = {
    "feature_stability_var": "KAN-F",
    "feature_edge_hybrid": "KAN-FE",
}
FUNCTION_LABELS = {
    "core_interaction_c025": "Core ($c=0.25$)",
    "feynman_energy": "Energy",
    "feynman_gravity": "Gravity",
    "feynman_coulomb": "Coulomb",
}


def fmt_float(value: float, digits: int = 2) -> str:
    if pd.isna(value):
        return "--"
    return f"{float(value):.{digits}f}"


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    display = df.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda value: fmt_float(value))
    values = [[str(col) for col in display.columns]]
    values.extend([[str(value) for value in row] for row in display.to_numpy()])
    widths = [max(len(row[idx]) for row in values) for idx in range(len(values[0]))]

    def render_row(row: list[str]) -> str:
        return "| " + " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(row)) + " |"

    header = render_row(values[0])
    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    body = [render_row(row) for row in values[1:]]
    return "\n".join([header, separator, *body])


def method_pivot(df: pd.DataFrame, index_cols: list[str], value_col: str) -> pd.DataFrame:
    work = df.copy()
    work["pair_score_method"] = pd.Categorical(work["pair_score_method"], METHOD_ORDER, ordered=True)
    pivot = work.pivot_table(
        index=index_cols,
        columns="pair_score_method",
        values=value_col,
        aggfunc="first",
        observed=False,
    ).reset_index()
    for method in METHOD_ORDER:
        if method not in pivot.columns:
            pivot[method] = pd.NA
    return pivot[index_cols + METHOD_ORDER]


def build_core_table(core_summary: Path) -> pd.DataFrame:
    df = pd.read_csv(core_summary)
    keep = df[
        df["pair_score_method"].isin(METHOD_ORDER)
        & df["source_method"].isin(SUPPORT_LABELS)
        & df["function"].eq("core_interaction_c025")
        & df["dimension"].eq(100)
        & df["samples"].isin([512, 1024])
        & df["top_m"].eq(4)
    ].copy()
    if keep.empty:
        raise ValueError(f"No core rows found in {core_summary}")
    pivot = method_pivot(
        keep,
        ["source_method", "samples", "dimension", "top_m"],
        "interaction_f1_mean",
    )
    pivot["support"] = pivot["source_method"].map(SUPPORT_LABELS).fillna(pivot["source_method"])
    pivot["runs"] = (
        keep.groupby(["source_method", "samples", "dimension", "top_m"], dropna=False)["num_runs"]
        .max()
        .reset_index(drop=True)
    )
    out = pivot[["support", "samples", "dimension", "top_m", *METHOD_ORDER, "runs"]].copy()
    out = out.rename(
        columns={
            "samples": "n",
            "dimension": "d",
            "top_m": "m",
            **METHOD_LABELS,
        }
    )
    return out.sort_values(["n", "support"]).reset_index(drop=True)


def build_feynman_table(feynman_summary: Path) -> pd.DataFrame:
    df = pd.read_csv(feynman_summary)
    keep = df[df["pair_score_method"].isin(METHOD_ORDER)].copy()
    if keep.empty:
        raise ValueError(f"No Feynman rows found in {feynman_summary}")
    pivot = method_pivot(
        keep,
        ["function", "samples", "dimension", "top_m"],
        "interaction_f1_mean",
    )
    runs = (
        keep.groupby(["function", "samples", "dimension", "top_m"], dropna=False)["num_runs"]
        .max()
        .reset_index(drop=True)
    )
    pivot["runs"] = runs
    pivot["formula"] = pivot["function"].map(FUNCTION_LABELS).fillna(pivot["function"])
    out = pivot[["formula", "samples", "dimension", "top_m", *METHOD_ORDER, "runs"]].copy()
    out = out.rename(
        columns={
            "samples": "n",
            "dimension": "d",
            "top_m": "m",
            **METHOD_LABELS,
        }
    )
    return out.sort_values(["formula"]).reset_index(drop=True)


def dataframe_to_latex_rows(df: pd.DataFrame, label_col: str) -> list[str]:
    rows = []
    for _, row in df.iterrows():
        rows.append(
            " & ".join(
                [
                    str(row[label_col]),
                    str(int(row["n"])),
                    str(int(row["d"])),
                    str(int(row["m"])),
                    fmt_float(row["FD"]),
                    fmt_float(row["ANOVA-abs"]),
                    fmt_float(row["ANOVA-var"]),
                    fmt_float(row["Hybrid"]),
                    str(int(row["runs"])),
                ]
            )
            + r" \\"
        )
    return rows


def write_latex(core: pd.DataFrame, feynman: pd.DataFrame, out_path: Path) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{\textbf{Formula-aware pair scoring on weak centered interactions.} Interaction F1 after KAN-native stable support recovery at $c=0.25,d=100,m=4$. Functional-ANOVA scores recover the true pair in every seed where local finite-difference scoring under-ranks it.}",
        r"\label{tab:formula-aware-core}",
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        r"Support & $n$ & $d$ & $m$ & FD & ANOVA-abs & ANOVA-var & Hybrid & Runs \\",
        r"\midrule",
        *dataframe_to_latex_rows(core, "support"),
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{\textbf{Feynman-style oracle-support sanity check.} Interaction F1 after refitting KAN on oracle support embedded in $d=100$ nuisance dimensions. This is a formula-ground-truth sanity check for pair scoring, not a full external benchmark.}",
        r"\label{tab:formula-aware-feynman}",
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        r"Formula & $n$ & $d$ & $m$ & FD & ANOVA-abs & ANOVA-var & Hybrid & Runs \\",
        r"\midrule",
        *dataframe_to_latex_rows(feynman, "formula"),
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_markdown(core: pd.DataFrame, feynman: pd.DataFrame, out_path: Path) -> None:
    lines = [
        "# Formula-Aware Pair Scoring Report",
        "",
        "## Core Weak-Interaction Table",
        "",
        dataframe_to_markdown(core),
        "",
        "## Feynman-Style Oracle-Support Sanity Check",
        "",
        dataframe_to_markdown(feynman),
        "",
        "Interpretation: the core table is the main claim-supporting result. The Feynman table is a limited sanity check on known-formula functions embedded in nuisance dimensions.",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create compact report tables for formula-aware pair scoring.")
    parser.add_argument(
        "--core_summary",
        default="results/innovation_loop/anova_pair_rescore_validation_20260526_113620_retry/pair_rescore_summary.csv",
    )
    parser.add_argument(
        "--feynman_summary",
        default="results/formula_aware_pair_scoring/feynman_oracle_sanity_20260526/pair_scoring_summary.csv",
    )
    parser.add_argument("--out_dir", default="results/formula_aware_pair_scoring/report_20260526")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    core = build_core_table(Path(args.core_summary))
    feynman = build_feynman_table(Path(args.feynman_summary))
    core.to_csv(out_dir / "core_pair_scoring_compact.csv", index=False)
    feynman.to_csv(out_dir / "feynman_pair_scoring_compact.csv", index=False)
    write_latex(core, feynman, out_dir / "formula_aware_pair_scoring_tables.tex")
    write_markdown(core, feynman, out_dir / "formula_aware_pair_scoring_report.md")
    print(core.to_string(index=False))
    print()
    print(feynman.to_string(index=False))
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
