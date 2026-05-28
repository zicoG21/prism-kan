from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd


def read_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(Path(path))


def fmt_float(x: float, digits: int = 3) -> str:
    if pd.isna(x):
        return "--"
    return f"{float(x):.{digits}f}"


def fmt_success(mean: float, n: int) -> str:
    if pd.isna(mean) or not n:
        return "--"
    return f"{int(round(float(mean) * int(n)))}/{int(n)}"


def conditional_pair(df: pd.DataFrame) -> tuple[float, int]:
    ok = df[pd.to_numeric(df["screen_contains_true_interactions"], errors="coerce").eq(1)]
    if ok.empty:
        return np.nan, 0
    return float(pd.to_numeric(ok["interaction_f1"], errors="coerce").mean()), int(len(ok))


def collect_ladder_rows(args: argparse.Namespace) -> pd.DataFrame:
    native = read_csv(args.native_detail)
    screen_paths = [
        args.screen_c025_n512,
        args.screen_c025_n1024,
        args.screen_c025_n2048,
    ]
    screened = pd.concat([read_csv(p) for p in screen_paths], ignore_index=True)

    rows = []

    specs = [
        ("KAN-FE", native, "method", "feature_edge_hybrid", 512, 4),
        ("RF-screened", screened, "screen_mode", "rf", 512, 4),
        ("Given-support", screened, "screen_mode", "oracle_support", 512, 4),
        ("Random", screened, "screen_mode", "random", 512, 4),
        ("KAN-FE", native, "method", "feature_edge_hybrid", 1024, 4),
        ("RF-screened", screened, "screen_mode", "rf", 1024, 4),
        ("Given-support", screened, "screen_mode", "oracle_support", 1024, 4),
        ("KAN-FE", native, "method", "feature_edge_hybrid", 2048, 6),
        ("RF-screened", screened, "screen_mode", "rf", 2048, 6),
        ("Given-support", screened, "screen_mode", "oracle_support", 2048, 6),
    ]

    for label, df, method_col, method, n, m in specs:
        sub = df[
            df[method_col].astype(str).eq(method)
            & df["function"].astype(str).eq("core_interaction_c025")
            & pd.to_numeric(df["dimension"], errors="coerce").eq(100)
            & pd.to_numeric(df["samples"], errors="coerce").eq(n)
            & pd.to_numeric(df["top_m"], errors="coerce").eq(m)
        ].copy()
        if sub.empty:
            continue
        for col in ["test_mse", "variable_f1", "screen_interaction_endpoint_recall", "interaction_f1"]:
            sub[col] = pd.to_numeric(sub[col], errors="coerce")
        cond, cond_n = conditional_pair(sub)
        rows.append(
            {
                "method": label,
                "n": n,
                "m": m,
                "runs": int(len(sub)),
                "test_mse": float(sub["test_mse"].mean()),
                "variable_f1": float(sub["variable_f1"].mean()),
                "endpoint_recall": float(sub["screen_interaction_endpoint_recall"].mean()),
                "pair_acc": float(sub["interaction_f1"].mean()),
                "pair_given_endpoint": cond,
                "pair_given_endpoint_n": cond_n,
            }
        )
    return pd.DataFrame(rows)


def ladder_latex(rows: pd.DataFrame) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{\textbf{Structure-fidelity ladder on the weak centered interaction.} Entries are seed averages except the conditional column, which reports top-1 pair accuracy among runs where both endpoints are retained.}",
        r"\label{tab:ladder}",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Method & Setting & MSE & Var. F1 & Endpt. & Pair & Pair$\mid$endpt. \\",
        r"\midrule",
    ]
    for r in rows.itertuples(index=False):
        setting = f"$n={int(r.n)},m={int(r.m)}$"
        lines.append(
            f"{r.method} & {setting} & "
            f"{fmt_float(r.test_mse, 4)} & {fmt_float(r.variable_f1, 2)} & "
            f"{fmt_float(r.endpoint_recall, 2)} & {fmt_success(r.pair_acc, r.runs)} & "
            f"{fmt_success(r.pair_given_endpoint, r.pair_given_endpoint_n)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def collect_minisuite(args: argparse.Namespace) -> pd.DataFrame:
    summary = read_csv(args.minisuite_summary)
    detail = read_csv(args.minisuite_detail)
    formulas = (
        detail[["function", "formula", "true_interactions", "num_true_variables", "num_true_interactions"]]
        .drop_duplicates("function")
        .copy()
    )
    keep_modes = ["oracle_support", "rf"]
    piv = summary[summary["screen_mode"].isin(keep_modes)].pivot_table(
        index="function",
        columns="screen_mode",
        values="interaction_f1_mean",
        aggfunc="first",
    )
    out = formulas.merge(piv.reset_index(), on="function", how="left")
    out = out.sort_values("function")
    return out


def shorten_formula(s: str, max_len: int = 42) -> str:
    s = str(s).replace("_", r"\_")
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


def minisuite_latex(rows: pd.DataFrame) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{\textbf{Eleven-formula mini-suite.} Pair recovery uses the same KAN refit and Hessian-style interaction scoring under RF support and given-support refit conditions.}",
        r"\label{tab:minisuite}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Family & \#vars & Given-support & RF \\",
        r"\midrule",
    ]
    for r in rows.itertuples(index=False):
        name = str(r.function).replace("formula_", "").replace("_", "-")
        lines.append(
            f"{name} & {int(r.num_true_variables)} & "
            f"{fmt_float(getattr(r, 'oracle_support'), 2)} & {fmt_float(getattr(r, 'rf'), 2)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--native_detail",
        default="results/innovation_loop/strict_validation_20260526_011917/innovation_detail.csv",
    )
    parser.add_argument(
        "--screen_c025_n512",
        default="results/innovation_loop/strict_screened_baseline_controls_20260526_104243/c025_n512_d100_top4_screen_eval.csv",
    )
    parser.add_argument(
        "--screen_c025_n1024",
        default="results/innovation_loop/strict_screened_baseline_controls_20260526_104243/c025_n1024_d100_top4_screen_eval.csv",
    )
    parser.add_argument(
        "--screen_c025_n2048",
        default="results/innovation_loop/strict_screened_baseline_controls_20260526_104243/c025_n2048_d100_top6_screen_eval.csv",
    )
    parser.add_argument(
        "--minisuite_summary",
        default="results/formula_fidelity_minisuite/overnight_tmlr_20260527_tmlr_overnight/d100_interaction_scoring/minisuite_summary.csv",
    )
    parser.add_argument(
        "--minisuite_detail",
        default="results/formula_fidelity_minisuite/overnight_tmlr_20260527_tmlr_overnight/d100_interaction_scoring/minisuite_detail.csv",
    )
    parser.add_argument("--out_dir", default="results/workshop_review_tables")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ladder = collect_ladder_rows(args)
    ladder.to_csv(out_dir / "structure_ladder_table.csv", index=False)
    (out_dir / "structure_ladder_table.tex").write_text(ladder_latex(ladder), encoding="utf-8")

    minisuite = collect_minisuite(args)
    minisuite.to_csv(out_dir / "minisuite_table.csv", index=False)
    (out_dir / "minisuite_table.tex").write_text(minisuite_latex(minisuite), encoding="utf-8")

    print(ladder.to_string(index=False))
    print()
    print(minisuite[["function", "num_true_variables", "num_true_interactions", "oracle_support", "rf"]].to_string(index=False))
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
