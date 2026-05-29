from __future__ import annotations

import argparse
import ast
from pathlib import Path

import pandas as pd


FAMILY = {
    "formula_bilinear": "bilinear",
    "formula_weak_centered": "weak-centered",
    "formula_trig_product": "trig-product",
    "formula_nested_trig": "nested-trig",
    "formula_rational_product": "rational-product",
    "formula_division_mixed": "division-mixed",
    "formula_exp_product": "exp-product",
    "formula_log_product": "log-product",
    "formula_three_way_product": "three-way-product",
    "formula_mixed_sparse": "mixed-sparse",
    "formula_sqrt_energy": "sqrt-energy",
}

FORMULA = {
    "formula_bilinear": r"$x_0x_1+0.5x_2$",
    "formula_weak_centered": r"$\sin(2\pi x_0)+x_1^2+0.25x_2x_3$",
    "formula_trig_product": r"$\sin(\pi x_0x_1)+0.5x_2$",
    "formula_nested_trig": r"$\sin(2\pi(x_0+0.5x_1x_2))$",
    "formula_rational_product": r"$x_0x_1/(1+x_2^2)$",
    "formula_division_mixed": r"$(x_0+1.2)/(1.5+x_1^2)+0.3x_2x_3$",
    "formula_exp_product": r"$\exp(0.5x_0x_1)+0.2x_2$",
    "formula_log_product": r"$\log(2+x_0x_1)+0.25x_2^2$",
    "formula_three_way_product": r"$x_0x_1x_2+0.5x_3$",
    "formula_mixed_sparse": r"$\sin(2\pi x_0)+x_1x_2/(1+x_3^2)$",
    "formula_sqrt_energy": r"$\sqrt{x_0+1.2}(x_1+1.5)^2+0.25x_2$",
}


def parse_pairs(value: object) -> int:
    if pd.isna(value):
        return 0
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            return len(parsed)
        except Exception:
            return value.count("(")
    try:
        return len(value)
    except Exception:
        return 0


def load_base(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["function"].isin(FAMILY)].copy()
    df["family"] = df["function"].map(FAMILY)
    df["formula_tex"] = df["function"].map(FORMULA)
    df["num_pairs"] = df["true_interactions"].map(parse_pairs)
    out = df[
        [
            "function",
            "family",
            "formula_tex",
            "num_pairs",
            "oracle_support",
            "rf",
        ]
    ].rename(
        columns={
            "oracle_support": "true_support_kan",
            "rf": "rf_support_kan",
        }
    )
    return out


def load_screen(path: Path, col_name: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["function", col_name, f"{col_name}_runs"])
    df = pd.read_csv(path)
    if "function" not in df.columns:
        return pd.DataFrame(columns=["function", col_name, f"{col_name}_runs"])
    df = df[df["function"].isin(FAMILY)].copy()
    metric_col = "top1_pair_accuracy_mean"
    if metric_col not in df.columns:
        metric_col = "interaction_f1_mean"
    runs_col = "num_runs" if "num_runs" in df.columns else None
    out = df[["function", metric_col] + ([runs_col] if runs_col else [])].copy()
    out = out.rename(columns={metric_col: col_name})
    if runs_col:
        out = out.rename(columns={runs_col: f"{col_name}_runs"})
    else:
        out[f"{col_name}_runs"] = pd.NA
    return out


def failure_tag(row: pd.Series) -> str:
    ts = float(row.get("true_support_kan", 0.0))
    rf = float(row.get("rf_support_kan", 0.0))
    raw = row.get("raw_resid", pd.NA)
    tensor = row.get("tensor_resid", pd.NA)
    classical_best = max(
        [float(v) for v in [raw, tensor] if not pd.isna(v)] or [0.0]
    )
    if ts < 0.5:
        return "refit/scoring"
    if rf + 1e-9 < ts:
        return "support"
    if classical_best + 1e-9 < ts:
        return "screening"
    return "recovered"


def fmt(x: object) -> str:
    if pd.isna(x):
        return "--"
    return f"{float(x):.2f}"


def write_tex(df: pd.DataFrame, out_path: Path) -> None:
    rows = []
    for _, row in df.iterrows():
        rows.append(
            " & ".join(
                [
                    str(row["family"]),
                    str(int(row["num_pairs"])),
                    fmt(row["true_support_kan"]),
                    fmt(row["rf_support_kan"]),
                    fmt(row.get("raw_resid")),
                    fmt(row.get("tensor_resid")),
                    fmt(row.get("gbm_h_stat")),
                    str(row["failure_tag"]),
                ]
            )
            + r" \\"
        )
    tex = r"""\begin{table}[t]
\centering
\small
\caption{\textbf{Formalized 11-formula mini-suite.} Values are mean top-$q$ pair F1 or top-1 accuracy, depending on the number of labeled pairs. Residual tensor-spline is a GA2M-style pairwise spline screen; GBM-H is a gradient-boosted H-statistic screen when available.}
\label{tab:minisuite-formal}
\begin{tabular}{lrrrrrrl}
\toprule
Family & \#pairs & True-KAN & RF-KAN & Raw resid. & Tensor resid. & GBM-H & Main bottleneck \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    out_path.write_text(tex)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="results/workshop_review_tables/minisuite_table.csv")
    parser.add_argument(
        "--raw_resid",
        default="results/interaction_baselines/residual_pair_screen_minisuite_trainresid_alpha1_d100_n1024_10seed/residual_pair_screen_summary.csv",
    )
    parser.add_argument(
        "--tensor_resid",
        default="results/interaction_baselines/residual_tensor_spline_minisuite_trainresid_alpha1_d100_n1024_10seed/residual_tensor_spline_pair_screen_summary.csv",
    )
    parser.add_argument(
        "--gbm_h",
        default="results/interaction_baselines/gbm_h_statistic_minisuite_d100_n1024_10seed/gbm_h_statistic_summary.csv",
    )
    parser.add_argument("--out_dir", default="results/workshop_review_tables/formal_minisuite")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_base(Path(args.base))
    for path, name in [
        (Path(args.raw_resid), "raw_resid"),
        (Path(args.tensor_resid), "tensor_resid"),
        (Path(args.gbm_h), "gbm_h_stat"),
    ]:
        df = df.merge(load_screen(path, name), on="function", how="left")

    df["failure_tag"] = df.apply(failure_tag, axis=1)
    df = df.sort_values("family")
    csv_path = out_dir / "formal_minisuite_baseline_table.csv"
    tex_path = out_dir / "formal_minisuite_baseline_table.tex"
    df.to_csv(csv_path, index=False)
    write_tex(df, tex_path)
    print(df.to_string(index=False))
    print(f"Wrote {csv_path}")
    print(f"Wrote {tex_path}")


if __name__ == "__main__":
    main()
