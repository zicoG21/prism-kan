from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_csv_safe(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        print(f"[WARN] Missing file: {path}")
        return None
    return pd.read_csv(path)


def summarize_oracle(path: Path) -> Dict:
    df = read_csv_safe(path)
    if df is None or df.empty:
        return {
            "oracle_var_f1": np.nan,
            "oracle_var_auroc": np.nan,
            "oracle_var_auprc": np.nan,
            "oracle_int_f1": np.nan,
        }

    return {
        "oracle_var_f1": float(df["variable_f1"].mean()) if "variable_f1" in df.columns else np.nan,
        "oracle_var_auroc": float(df["variable_auroc"].mean()) if "variable_auroc" in df.columns else np.nan,
        "oracle_var_auprc": float(df["variable_auprc"].mean()) if "variable_auprc" in df.columns else np.nan,
        "oracle_int_f1": float(df["interaction_f1"].mean()) if "interaction_f1" in df.columns else np.nan,
    }


def summarize_kan(path: Path) -> Dict:
    df = read_csv_safe(path)
    if df is None or df.empty:
        return {
            "kan_test_mse": np.nan,
            "kan_var_f1": np.nan,
            "kan_var_auroc": np.nan,
            "kan_var_auprc": np.nan,
            "kan_int_f1": np.nan,
        }

    if "model" in df.columns:
        df = df[df["model"].astype(str).str.upper() == "KAN"].copy()

    return {
        "kan_test_mse": float(df["test_mse"].mean()) if "test_mse" in df.columns else np.nan,
        "kan_var_f1": float(df["variable_f1"].mean()) if "variable_f1" in df.columns else np.nan,
        "kan_var_auroc": float(df["variable_auroc"].mean()) if "variable_auroc" in df.columns else np.nan,
        "kan_var_auprc": float(df["variable_auprc"].mean()) if "variable_auprc" in df.columns else np.nan,
        "kan_int_f1": float(df["interaction_f1"].mean()) if "interaction_f1" in df.columns else np.nan,
    }


def extract_delta(summary_df: pd.DataFrame, variable: int, method: str = "permute") -> float:
    sub = summary_df[
        (summary_df["row_type"] == "single_variable")
        & (summary_df["intervention_method"] == method)
    ].copy()

    if sub.empty:
        return np.nan

    sub["variable_num"] = pd.to_numeric(sub["variable"], errors="coerce")
    hit = sub[sub["variable_num"] == variable]

    if hit.empty:
        return np.nan

    return float(hit["delta_mse_mean"].iloc[0])


def extract_pair_delta(summary_df: pd.DataFrame, pair: str, method: str = "permute") -> float:
    sub = summary_df[
        (summary_df["row_type"] == "pair")
        & (summary_df["intervention_method"] == method)
    ].copy()

    if sub.empty:
        return np.nan

    hit = sub[sub["pair"].astype(str) == pair]

    if hit.empty:
        return np.nan

    return float(hit["delta_mse_mean"].iloc[0])


def extract_pair_synergy(summary_df: pd.DataFrame, pair: str, method: str = "permute") -> float:
    sub = summary_df[
        (summary_df["row_type"] == "pair")
        & (summary_df["intervention_method"] == method)
    ].copy()

    if sub.empty:
        return np.nan

    hit = sub[sub["pair"].astype(str) == pair]

    if hit.empty:
        return np.nan

    return float(hit["synergy_mean"].iloc[0])


def summarize_intervention(path: Path) -> Dict:
    df = read_csv_safe(path)
    if df is None or df.empty:
        return {}

    out = {}

    for v in [0, 1, 2, 3, 4, 5]:
        out[f"delta_x{v}"] = extract_delta(df, v)

    out["delta_pair_23"] = extract_pair_delta(df, "(2, 3)")
    out["synergy_pair_23"] = extract_pair_synergy(df, "(2, 3)")
    out["delta_pair_04"] = extract_pair_delta(df, "(0, 4)")
    out["synergy_pair_04"] = extract_pair_synergy(df, "(0, 4)")
    out["delta_pair_01"] = extract_pair_delta(df, "(0, 1)")
    out["synergy_pair_01"] = extract_pair_synergy(df, "(0, 1)")

    return out


def infer_failure_source(row: Dict) -> str:
    oracle_good = (
        row.get("oracle_var_f1", np.nan) >= 0.9
        and row.get("oracle_int_f1", np.nan) >= 0.9
    )

    kan_var_bad = row.get("kan_var_f1", np.nan) < 0.8
    kan_int_bad = row.get("kan_int_f1", np.nan) < 0.8

    x23_low = (
        np.nan_to_num(row.get("delta_x2", np.nan), nan=0.0) < 0.05
        and np.nan_to_num(row.get("delta_x3", np.nan), nan=0.0) < 0.05
        and np.nan_to_num(row.get("delta_pair_23", np.nan), nan=0.0) < 0.05
    )

    proxy_used = np.nan_to_num(row.get("delta_x4", np.nan), nan=0.0) > 0.02

    if row["setting"] == "clean_core_d20":
        return "No failure: oracle, KAN explanation, and intervention agree."

    if oracle_good and (kan_var_bad or kan_int_bad) and x23_low:
        return "Model-learning/representation failure: KAN does not functionally rely on true interaction variables."

    if oracle_good and proxy_used and row["setting"] == "correlated_proxy_d100":
        return "Proxy reliance / formula-level fidelity failure: KAN relies on correlated proxy features."

    if oracle_good and (kan_var_bad or kan_int_bad):
        return "KAN explanation/model failure; intervention needed for finer attribution."

    if not oracle_good:
        return "Metric or identifiability failure: oracle explanation does not recover ground truth."

    return "No clear failure attribution."


def make_attribution_table(args) -> pd.DataFrame:
    root = Path(args.root)

    settings = [
        {
            "setting": "clean_core_d20",
            "description": "Clean low-dimensional core interaction",
            "oracle": root / "oracle" / "oracle_core_d20.csv",
            "kan": root / "stage1" / "core_interaction_with_interactions.csv",
            "intervention": root / "intervention" / "clean_core_d20_summary.csv",
        },
        {
            "setting": "highdim_core_d100",
            "description": "High-dimensional sparse core interaction",
            "oracle": root / "oracle" / "oracle_core_d100.csv",
            "kan": root / "stage1" / "core_d100_noise00_interactions.csv",
            "intervention": root / "intervention" / "core_d100_summary.csv",
        },
        {
            "setting": "correlated_proxy_d100",
            "description": "High-dimensional correlated proxy",
            "oracle": root / "oracle" / "oracle_correlated_proxy_d100.csv",
            "kan": root / "stage1" / "correlated_proxy_d100_n512_noise005.csv",
            "intervention": root / "intervention" / "correlated_proxy_d100_summary.csv",
        },
        {
            "setting": "strong_interaction_c5_d100",
            "description": "High-dimensional strengthened interaction coefficient",
            "oracle": root / "oracle" / "oracle_core_c5_d100.csv",
            "kan": root / "stage2_interaction_strength" / "core_d100_c5.csv",
            "intervention": None,
        },
    ]

    rows = []

    for item in settings:
        row = {
            "setting": item["setting"],
            "description": item["description"],
        }
        row.update(summarize_oracle(item["oracle"]))
        row.update(summarize_kan(item["kan"]))

        if item["intervention"] is not None:
            row.update(summarize_intervention(item["intervention"]))

        row["failure_source"] = infer_failure_source(row)
        rows.append(row)

    return pd.DataFrame(rows)


def plot_oracle_vs_kan(df: pd.DataFrame, out_dir: Path):
    x = np.arange(len(df))
    width = 0.2

    plt.figure(figsize=(10, 5))

    plt.bar(x - 1.5 * width, df["oracle_var_f1"], width=width, label="Oracle Var F1")
    plt.bar(x - 0.5 * width, df["kan_var_f1"], width=width, label="KAN Var F1")
    plt.bar(x + 0.5 * width, df["oracle_int_f1"], width=width, label="Oracle Int F1")
    plt.bar(x + 1.5 * width, df["kan_int_f1"], width=width, label="KAN Int F1")

    plt.ylim(0, 1.1)
    plt.ylabel("Mean F1")
    plt.xticks(x, df["setting"], rotation=25, ha="right")
    plt.title("Oracle explanations recover structure; trained KAN explanations can fail")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "fig_oracle_vs_kan_f1.png", dpi=250)
    plt.close()


def plot_intervention_reliance(df: pd.DataFrame, out_dir: Path):
    sub = df[df["setting"].isin(["clean_core_d20", "highdim_core_d100", "correlated_proxy_d100"])].copy()

    vars_to_plot = ["delta_x0", "delta_x1", "delta_x2", "delta_x3", "delta_x4"]
    labels = ["x0", "x1", "x2", "x3", "x4"]

    x = np.arange(len(labels))
    width = 0.25

    plt.figure(figsize=(10, 5))

    for idx, (_, row) in enumerate(sub.iterrows()):
        vals = [row.get(v, np.nan) for v in vars_to_plot]
        offset = x + (idx - 1) * width
        plt.bar(offset, vals, width=width, label=row["setting"])

    plt.axhline(0, linewidth=1)
    plt.ylabel("Mean increase in test MSE after permutation")
    plt.xlabel("Intervened variable")
    plt.xticks(x, labels)
    plt.title("Functional reliance differs across settings")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "fig_intervention_reliance.png", dpi=250)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="results")
    parser.add_argument("--out", type=str, default="results/failure_attribution")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    table = make_attribution_table(args)

    table.to_csv(out_dir / "failure_attribution_table.csv", index=False)

    with pd.ExcelWriter(out_dir / "failure_attribution_summary.xlsx", engine="openpyxl") as writer:
        table.to_excel(writer, sheet_name="Failure Attribution", index=False)

    plot_oracle_vs_kan(table, out_dir)
    plot_intervention_reliance(table, out_dir)

    print(table.to_string(index=False))
    print(f"\nSaved outputs to {out_dir}")


if __name__ == "__main__":
    main()