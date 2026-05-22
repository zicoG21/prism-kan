#!/usr/bin/env python3
"""
Make paper-ready figures for the KAN formula-level fidelity project.

Run from the project root:

    cd ~/kan_xai_benchmark
    PYTHONPATH=. python scripts/plot_paper_figures.py

Expected inputs, if available:
    results/next_round/dim_transition/dim_transition_all_summary.csv
    results/next_round/dim_transition/dim_*_summary.csv
    results/next_round/tuned_screened/tuned_screened_summary.csv
    results/next_round/feynman_interaction/feynman_interaction_summary.csv

Outputs:
    results/paper_figures/fig1_dimension_transition_core_interaction.pdf/png
    results/paper_figures/fig2_tuned_support_interaction_f1.pdf/png
    results/paper_figures/fig2_tuned_support_test_mse.pdf/png
    results/paper_figures/fig2_tuned_support_variable_f1.pdf/png
    results/paper_figures/fig3_feynman_summary.pdf/png
    results/paper_figures/table_master_summary.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


SCREEN_LABELS = {
    "raw": "Raw KAN",
    "rf": "RF-screened",
    "oracle_support": "Oracle-support",
    "random": "Random-screened",
    "exclude_interaction": "Exclude-interaction",
    "rf_exclude_interaction": "RF excl. interaction",
}

SCREEN_ORDER = [
    "raw",
    "rf",
    "oracle_support",
    "random",
    "exclude_interaction",
    "rf_exclude_interaction",
]

TUNED_LABELS = {
    "raw": "Raw tuned KAN",
    "rf": "RF-screened tuned KAN",
    "oracle_support": "Oracle-support tuned KAN",
    "random": "Random-screened tuned KAN",
    "exclude_interaction": "Exclude-interaction tuned KAN",
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize common column variants across our experiment scripts."""
    df = df.copy()

    rename = {
        "function_name": "function",
        "screen": "screen_mode",
        "dim": "dimension",
        "d": "dimension",
    }
    for old, new in rename.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})

    # If this is already a summary CSV, copy *_mean into canonical metric names.
    for metric in ["test_mse", "variable_f1", "interaction_f1"]:
        mean_col = f"{metric}_mean"
        if mean_col in df.columns:
            df[metric] = df[mean_col]

    if "dimension" in df.columns:
        df["dimension"] = pd.to_numeric(df["dimension"], errors="coerce")

    return df


def aggregate_if_needed(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """If the input is raw per-seed data, aggregate to means and stds."""
    df = normalize_columns(df)
    metric_cols = [c for c in ["test_mse", "variable_f1", "interaction_f1"] if c in df.columns]
    if not metric_cols:
        raise ValueError(f"No metric columns found. Columns are: {list(df.columns)}")

    missing = [c for c in group_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing group columns {missing}. Columns are: {list(df.columns)}")

    # Already summary-like if one row per group.
    if df.groupby(group_cols, dropna=False).size().max() == 1:
        keep = group_cols + metric_cols
        extra = [f"{m}_std" for m in metric_cols if f"{m}_std" in df.columns]
        return df[keep + extra].copy()

    agg = df.groupby(group_cols, dropna=False)[metric_cols].agg(["mean", "std"]).reset_index()
    agg.columns = [
        "_".join([x for x in col if x]).rstrip("_") if isinstance(col, tuple) else col
        for col in agg.columns
    ]
    for metric in metric_cols:
        agg[metric] = agg[f"{metric}_mean"]
    return agg


def load_dimension_transition(results_root: Path) -> pd.DataFrame:
    combined = results_root / "next_round" / "dim_transition" / "dim_transition_all_summary.csv"

    if combined.exists():
        df = pd.read_csv(combined)
    else:
        pieces = []
        dim_dir = results_root / "next_round" / "dim_transition"
        for p in sorted(dim_dir.glob("dim_*_summary.csv")):
            part = normalize_columns(pd.read_csv(p))
            if "dimension" not in part.columns or part["dimension"].isna().all():
                try:
                    part["dimension"] = int(p.name.split("_")[1])
                except Exception:
                    pass
            pieces.append(part)
        if not pieces:
            raise FileNotFoundError(f"No dimension transition summaries found in {dim_dir}")
        df = pd.concat(pieces, ignore_index=True)

    return aggregate_if_needed(df, ["function", "screen_mode", "dimension"])


def load_summary(path: Path, group_cols: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required summary file: {path}")
    return aggregate_if_needed(pd.read_csv(path), group_cols)


def ordered_modes(present_modes: Iterable[str]) -> list[str]:
    present = list(dict.fromkeys(str(x) for x in present_modes))
    return [m for m in SCREEN_ORDER if m in present] + [m for m in present if m not in SCREEN_ORDER]


def savefig(out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ["pdf", "png"]:
        path = out_dir / f"{name}.{ext}"
        plt.savefig(path, bbox_inches="tight", dpi=300)
        print(f"Wrote {path}")
    plt.close()


def plot_dimension_transition(dim_df: pd.DataFrame, out_dir: Path, function: str) -> None:
    df = dim_df[dim_df["function"] == function].copy()
    if df.empty:
        raise ValueError(f"No dimension-transition rows found for function={function!r}")

    modes = ordered_modes(df["screen_mode"].dropna().unique())
    metrics = ["test_mse", "variable_f1", "interaction_f1"]
    ylabels = ["Test MSE", "Variable F1", "Interaction F1"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    for ax, metric, ylabel in zip(axes, metrics, ylabels):
        for mode in modes:
            sub = df[df["screen_mode"].astype(str) == mode].sort_values("dimension")
            if sub.empty or metric not in sub.columns:
                continue
            ax.plot(
                sub["dimension"],
                sub[metric],
                marker="o",
                linewidth=2,
                label=SCREEN_LABELS.get(mode, mode),
            )

        ax.set_xlabel("Input dimension d")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.grid(True, alpha=0.3)

        if metric == "test_mse":
            vals = df[metric].replace([np.inf, -np.inf], np.nan).dropna()
            if len(vals) and vals.min() > 0:
                ax.set_yscale("log")
        else:
            ax.set_ylim(-0.05, 1.05)

    axes[0].legend(loc="best", fontsize=9)
    fig.suptitle(f"Formula fidelity degrades with nuisance dimensions: {function}", y=1.04)
    savefig(out_dir, f"fig1_dimension_transition_{function}")


def plot_tuned_support_comparison(
    tuned_df: pd.DataFrame,
    out_dir: Path,
    functions: Iterable[str],
) -> None:
    functions = list(functions)
    df = tuned_df[tuned_df["function"].isin(functions)].copy()
    if df.empty:
        raise ValueError(f"No tuned rows found for functions={functions}")

    keep = ["raw", "rf", "oracle_support", "random", "exclude_interaction"]
    keep = [m for m in keep if m in set(df["screen_mode"].astype(str))]
    df = df[df["screen_mode"].astype(str).isin(keep)]

    metrics = ["interaction_f1", "test_mse", "variable_f1"]
    ylabels = ["Interaction F1", "Test MSE", "Variable F1"]

    for metric, ylabel in zip(metrics, ylabels):
        pivot = df.pivot_table(
            index="function",
            columns="screen_mode",
            values=metric,
            aggfunc="mean",
            observed=True,
        ).reindex(functions)

        modes = [m for m in keep if m in pivot.columns]
        x = np.arange(len(pivot.index))
        width = 0.8 / max(len(modes), 1)

        plt.figure(figsize=(8.2, 4.8))
        for idx, mode in enumerate(modes):
            offset = (idx - (len(modes) - 1) / 2) * width
            plt.bar(
                x + offset,
                pivot[mode].values,
                width=width,
                label=TUNED_LABELS.get(mode, SCREEN_LABELS.get(mode, mode)),
            )

        plt.xticks(x, pivot.index, rotation=35, ha="right")
        plt.ylabel(ylabel)
        plt.title(f"Tuned KAN: {ylabel}")
        if metric == "test_mse":
            vals = pivot[modes].to_numpy().ravel()
            vals = vals[np.isfinite(vals) & (vals > 0)]
            if len(vals):
                plt.yscale("log")
        else:
            plt.ylim(0, 1.08)
        plt.legend(fontsize=9)
        plt.tight_layout()
        savefig(out_dir, f"fig2_tuned_support_{metric}")


def plot_feynman_summary(
    feynman_df: pd.DataFrame,
    out_dir: Path,
    functions: Iterable[str],
) -> None:
    functions = list(functions)
    df = feynman_df[feynman_df["function"].isin(functions)].copy()
    if df.empty:
        raise ValueError(f"No Feynman rows found for functions={functions}")

    keep = ["raw", "rf", "oracle_support", "random", "exclude_interaction"]
    keep = [m for m in keep if m in set(df["screen_mode"].astype(str))]
    df = df[df["screen_mode"].astype(str).isin(keep)]

    metrics = ["interaction_f1", "test_mse", "variable_f1"]
    ylabels = ["Interaction F1", "Test MSE", "Variable F1"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    for ax, metric, ylabel in zip(axes, metrics, ylabels):
        pivot = df.pivot_table(
            index="function",
            columns="screen_mode",
            values=metric,
            aggfunc="mean",
            observed=True,
        ).reindex(functions)

        modes = [m for m in keep if m in pivot.columns]
        x = np.arange(len(pivot.index))
        width = 0.8 / max(len(modes), 1)

        for idx, mode in enumerate(modes):
            offset = (idx - (len(modes) - 1) / 2) * width
            ax.bar(x + offset, pivot[mode].values, width=width, label=SCREEN_LABELS.get(mode, mode))

        ax.set_xticks(x)
        ax.set_xticklabels(pivot.index, rotation=35, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)

        if metric == "test_mse":
            vals = pivot[modes].to_numpy().ravel()
            vals = vals[np.isfinite(vals) & (vals > 0)]
            if len(vals):
                ax.set_yscale("log")
        else:
            ax.set_ylim(0, 1.08)

    axes[0].legend(loc="best", fontsize=9)
    fig.suptitle("Feynman-style physics formulas embedded with nuisance features", y=1.04)
    savefig(out_dir, "fig3_feynman_summary")


def write_master_table(
    out_dir: Path,
    dim_df: pd.DataFrame | None,
    tuned_df: pd.DataFrame | None,
    feynman_df: pd.DataFrame | None,
) -> None:
    rows = []
    if dim_df is not None:
        tmp = dim_df.copy()
        tmp["experiment"] = "dimension_transition"
        rows.append(tmp)
    if tuned_df is not None:
        tmp = tuned_df.copy()
        tmp["experiment"] = "tuned_support"
        rows.append(tmp)
    if feynman_df is not None:
        tmp = feynman_df.copy()
        tmp["experiment"] = "feynman_interaction"
        rows.append(tmp)

    if not rows:
        return

    table = pd.concat(rows, ignore_index=True)
    cols = [
        "experiment",
        "function",
        "dimension",
        "screen_mode",
        "test_mse",
        "variable_f1",
        "interaction_f1",
    ]
    cols = [c for c in cols if c in table.columns]
    table = table[cols]
    sort_cols = [c for c in ["experiment", "function", "dimension", "screen_mode"] if c in table.columns]
    table = table.sort_values(sort_cols)

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "table_master_summary.csv"
    table.to_csv(out, index=False)
    print(f"Wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root", type=str, default="results")
    parser.add_argument("--out_dir", type=str, default="results/paper_figures")
    parser.add_argument("--dim_function", type=str, default="core_interaction")
    parser.add_argument("--tuned_functions", nargs="+", default=["core_interaction", "feynman_coulomb"])
    parser.add_argument(
        "--feynman_functions",
        nargs="+",
        default=["feynman_energy", "feynman_gravity", "feynman_coulomb"],
    )
    args = parser.parse_args()

    root = Path(args.results_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dim_df = tuned_df = feynman_df = None

    try:
        dim_df = load_dimension_transition(root)
        plot_dimension_transition(dim_df, out_dir, function=args.dim_function)
    except Exception as e:
        print(f"[skip] dimension transition figure: {e}")

    try:
        tuned_path = root / "next_round" / "tuned_screened" / "tuned_screened_summary.csv"
        tuned_df = load_summary(tuned_path, ["function", "screen_mode"])
        plot_tuned_support_comparison(tuned_df, out_dir, functions=args.tuned_functions)
    except Exception as e:
        print(f"[skip] tuned support figures: {e}")

    try:
        feynman_path = root / "next_round" / "feynman_interaction" / "feynman_interaction_summary.csv"
        feynman_df = load_summary(feynman_path, ["function", "screen_mode"])
        plot_feynman_summary(feynman_df, out_dir, functions=args.feynman_functions)
    except Exception as e:
        print(f"[skip] Feynman summary figure: {e}")

    write_master_table(out_dir, dim_df, tuned_df, feynman_df)
    print("\nDone. Check:", out_dir)


if __name__ == "__main__":
    main()
