from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def pick_metric(row: pd.Series, name: str) -> float:
    for col in (f"{name}_mean", name):
        if col in row and pd.notna(row[col]):
            return float(row[col])
    return float("nan")


def add_rows_from_summary(rows, path: Path, model_label: str, function: str, modes: list[str]):
    if not path.exists():
        return
    df = pd.read_csv(path)
    if "function" in df.columns:
        df = df[df["function"].astype(str) == function].copy()
    for mode in modes:
        hit = df[df["screen_mode"].astype(str) == mode]
        if hit.empty:
            continue
        row = hit.iloc[0]
        rows.append({
            "model_family": model_label,
            "function": function,
            "screen_mode": mode,
            "dimension": int(row["dimension"]) if "dimension" in row and pd.notna(row["dimension"]) else 100,
            "samples": int(row["samples"]) if "samples" in row and pd.notna(row["samples"]) else 1024,
            "test_mse": pick_metric(row, "test_mse"),
            "variable_f1": pick_metric(row, "variable_f1"),
            "screen_endpoint_recall": pick_metric(row, "screen_interaction_endpoint_recall"),
            "explain_endpoint_recall": pick_metric(row, "explain_interaction_endpoint_recall"),
            "interaction_f1": pick_metric(row, "interaction_f1"),
            "source_file": str(path),
        })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/paper_figures/final_baseline_table.csv")
    args = parser.parse_args()

    rows = []

    add_rows_from_summary(
        rows,
        Path("results/hard_regime/summaries/core_interaction_c1_n1024_d100_summary.csv"),
        "KAN tuned fixed",
        "core_interaction_c1",
        ["raw", "rf", "oracle_support", "random", "exclude_interaction"],
    )

    add_rows_from_summary(
        rows,
        Path("results/mlp_tuning/mlp_screened_summary.csv"),
        "MLP early stopped",
        "core_interaction",
        ["raw", "rf", "oracle_support", "random", "exclude_interaction"],
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(out, index=False)
    print(f"[saved] {out}")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
