from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def fmt_count(v: float) -> str:
    return f"{int(round(float(v)))}/10"


def build_table(summary_csv: Path) -> tuple[pd.DataFrame, str]:
    df = pd.read_csv(summary_csv)
    edge = df[df["method"] == "feature_edge_hybrid"].copy()
    rows = []
    for (dataset, c), group in edge.groupby(["dataset", "c"], sort=True):
        group = group.sort_values("samples")
        kan_counts = " / ".join(fmt_count(v) for v in group["support_pair_successes"])
        resid_counts = " / ".join(fmt_count(v) for v in group["residual_top1_successes"])
        mse_384 = group.loc[group["samples"] == 384, "probe_test_mse_mean_mean"]
        rows.append(
            {
                "dataset": dataset.replace("_", "\\_"),
                "c": f"{float(c):.2f}",
                "KAN-FE pair @ 128/256/384": kan_counts,
                "Residual pair @ 128/256/384": resid_counts,
                "MSE @ 384": f"{float(mse_384.iloc[0]):.2e}" if len(mse_384) else "--",
            }
        )
    table = pd.DataFrame(rows)
    latex = table.to_latex(index=False, escape=False)
    return table, latex


def to_markdown_simple(table: pd.DataFrame) -> str:
    headers = [str(c) for c in table.columns]
    rows = [[str(v) for v in row] for row in table.to_numpy()]
    out = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    out.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("results/revision/semisynthetic_covariates_3h/semisynthetic_covariate_audit_summary.csv"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/revision/semisynthetic_covariates_3h/summary"),
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    table, latex = build_table(args.summary_csv)
    table.to_csv(args.out_dir / "semisynthetic_covariate_compact_table.csv", index=False)
    (args.out_dir / "semisynthetic_covariate_compact_table.tex").write_text(latex)
    (args.out_dir / "summary.md").write_text(
        "# Semi-synthetic covariate compact table\n\n"
        + to_markdown_simple(table)
        + "\n"
    )
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
