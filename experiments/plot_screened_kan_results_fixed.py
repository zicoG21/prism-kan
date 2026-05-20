from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def normalize_screened_summary(df: pd.DataFrame, label: str) -> pd.DataFrame:
    out = df.copy()
    out["setting_label"] = label
    needed = [
        "screen_method",
        "top_m",
        "variable_f1_mean",
        "interaction_f1_mean",
        "test_mse_mean",
        "screen_contains_all_true_vars_mean",
        "screen_contains_true_interactions_mean",
    ]
    for col in needed:
        if col not in out.columns:
            out[col] = np.nan
    return out


def load_screened_summaries(paths: List[str], labels: List[str]) -> pd.DataFrame:
    frames = []
    for i, path in enumerate(paths):
        label = labels[i] if i < len(labels) else Path(path).stem
        frames.append(normalize_screened_summary(read_csv(path), label))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def scalar_mean(df: pd.DataFrame, col: str) -> float:
    if df.empty or col not in df.columns:
        return np.nan
    vals = pd.to_numeric(df[col], errors="coerce").dropna()
    if vals.empty:
        return np.nan
    return float(vals.mean())


def plot_screened_recovery(df: pd.DataFrame, out_path: Path, title: str):
    if df.empty:
        return

    plot_df = df.copy()
    if "explain_method" in plot_df.columns and "grad" in set(plot_df["explain_method"].astype(str)):
        plot_df = plot_df[plot_df["explain_method"].astype(str) == "grad"].copy()

    plot_df = plot_df.sort_values(["setting_label", "screen_method", "top_m"])
    plot_df["label"] = plot_df.apply(
        lambda r: f"{r['setting_label']}\n{r['screen_method']} M={int(r['top_m'])}",
        axis=1,
    )

    x = np.arange(len(plot_df))
    width = 0.35

    plt.figure(figsize=(max(11, len(plot_df) * 0.72), 5.5))
    plt.bar(x - width / 2, plot_df["variable_f1_mean"], width=width, label="Variable F1")
    plt.bar(x + width / 2, plot_df["interaction_f1_mean"], width=width, label="Interaction F1")
    plt.ylim(0, 1.08)
    plt.ylabel("Mean F1")
    plt.xticks(x, plot_df["label"], rotation=55, ha="right")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=250)
    plt.close()


def plot_screen_success(df: pd.DataFrame, out_path: Path):
    if df.empty:
        return

    plot_df = df.copy()
    if "explain_method" in plot_df.columns and "grad" in set(plot_df["explain_method"].astype(str)):
        plot_df = plot_df[plot_df["explain_method"].astype(str) == "grad"].copy()

    plot_df = plot_df.sort_values(["setting_label", "screen_method", "top_m"])
    plot_df["label"] = plot_df.apply(
        lambda r: f"{r['setting_label']}\n{r['screen_method']} M={int(r['top_m'])}",
        axis=1,
    )

    x = np.arange(len(plot_df))
    width = 0.35

    plt.figure(figsize=(max(11, len(plot_df) * 0.72), 5.5))
    plt.bar(x - width / 2, plot_df["screen_contains_all_true_vars_mean"], width=width, label="All true variables retained")
    plt.bar(x + width / 2, plot_df["screen_contains_true_interactions_mean"], width=width, label="True interaction variables retained")
    plt.ylim(0, 1.08)
    plt.ylabel("Fraction across seeds")
    plt.xticks(x, plot_df["label"], rotation=55, ha="right")
    plt.title("Screening success determines whether KAN can recover structure")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=250)
    plt.close()


def normalize_path_summary(df: pd.DataFrame, label: str) -> pd.DataFrame:
    out = df.copy()
    out["setting_label"] = label
    for col in ["screen_method", "top_m", "row_type", "target", "pair", "delta_mse_mean", "synergy_mean"]:
        if col not in out.columns:
            out[col] = np.nan
    return out


def load_path_summaries(paths: List[str], labels: List[str]) -> pd.DataFrame:
    frames = []
    for i, path in enumerate(paths):
        label = labels[i] if i < len(labels) else Path(path).stem
        frames.append(normalize_path_summary(read_csv(path), label))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def aggregate_path_rows(sub: pd.DataFrame, key_col: str) -> pd.DataFrame:
    """Remove duplicate target/pair rows caused by screened-out variables or groupby artifacts.

    If both NaN and non-NaN rows exist, we use the mean of non-NaN deltas.
    """
    group_cols = ["setting_label", "screen_method", "top_m", key_col]
    keep_cols = group_cols + ["delta_mse_mean", "synergy_mean"]
    tmp = sub[keep_cols].copy()

    out = (
        tmp.groupby(group_cols, dropna=False)
        .agg(
            delta_mse_mean=("delta_mse_mean", lambda s: pd.to_numeric(s, errors="coerce").dropna().mean()),
            synergy_mean=("synergy_mean", lambda s: pd.to_numeric(s, errors="coerce").dropna().mean()),
        )
        .reset_index()
    )
    return out


def plot_feature_path(df: pd.DataFrame, out_path: Path, title: str):
    if df.empty:
        return

    sub = df[df["row_type"] == "feature_path"].copy()
    if sub.empty:
        return

    wanted = ["x0", "x1", "x2", "x3", "x4", "x5"]
    sub = sub[sub["target"].astype(str).isin(wanted)].copy()
    if sub.empty:
        return

    sub = aggregate_path_rows(sub, "target")
    sub = sub.sort_values(["setting_label", "screen_method", "top_m", "target"])

    groups = list(sub[["setting_label", "screen_method", "top_m"]].drop_duplicates().itertuples(index=False, name=None))
    x = np.arange(len(wanted))
    width = 0.8 / max(len(groups), 1)

    plt.figure(figsize=(max(10, len(groups) * 1.4), 5.5))
    for idx, key in enumerate(groups):
        setting, method, top_m = key
        g = sub[(sub["setting_label"] == setting) & (sub["screen_method"] == method) & (sub["top_m"] == top_m)]
        vals = []
        for t in wanted:
            vals.append(scalar_mean(g[g["target"].astype(str) == t], "delta_mse_mean"))
        label = f"{setting}, {method} M={int(top_m)}"
        plt.bar(x + (idx - (len(groups) - 1) / 2) * width, vals, width=width, label=label)

    plt.axhline(0, linewidth=1)
    plt.xticks(x, wanted)
    plt.ylabel("Mean test MSE increase")
    plt.xlabel("Deleted screened-KAN feature path")
    plt.title(title)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=250)
    plt.close()


def plot_pair_path(df: pd.DataFrame, out_path: Path, title: str):
    if df.empty:
        return

    sub = df[df["row_type"] == "feature_pair_path"].copy()
    if sub.empty:
        return

    wanted = ["(2, 3)", "(0, 1)", "(0, 4)", "(1, 5)"]
    sub = sub[sub["pair"].astype(str).isin(wanted)].copy()
    if sub.empty:
        return

    sub = aggregate_path_rows(sub, "pair")
    sub = sub.sort_values(["setting_label", "screen_method", "top_m", "pair"])

    groups = list(sub[["setting_label", "screen_method", "top_m"]].drop_duplicates().itertuples(index=False, name=None))
    x = np.arange(len(wanted))
    width = 0.8 / max(len(groups), 1)

    plt.figure(figsize=(max(10, len(groups) * 1.4), 5.5))
    for idx, key in enumerate(groups):
        setting, method, top_m = key
        g = sub[(sub["setting_label"] == setting) & (sub["screen_method"] == method) & (sub["top_m"] == top_m)]
        vals = []
        for t in wanted:
            vals.append(scalar_mean(g[g["pair"].astype(str) == t], "delta_mse_mean"))
        label = f"{setting}, {method} M={int(top_m)}"
        plt.bar(x + (idx - (len(groups) - 1) / 2) * width, vals, width=width, label=label)

    plt.axhline(0, linewidth=1)
    plt.xticks(x, wanted)
    plt.ylabel("Mean test MSE increase")
    plt.xlabel("Deleted screened-KAN pair path")
    plt.title(title)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=250)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--screened_summary", nargs="*", default=[])
    parser.add_argument("--screened_labels", nargs="*", default=[])
    parser.add_argument("--path_summary", nargs="*", default=[])
    parser.add_argument("--path_labels", nargs="*", default=[])
    parser.add_argument("--out_dir", default="results/screened_kan/figures")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    screened_df = load_screened_summaries(args.screened_summary, args.screened_labels)
    if not screened_df.empty:
        screened_df.to_csv(out_dir / "combined_screened_summary.csv", index=False)
        plot_screened_recovery(
            screened_df,
            out_dir / "screened_kan_recovery.png",
            "Feature screening can rescue high-dimensional KAN recovery",
        )
        plot_screen_success(screened_df, out_dir / "screening_success.png")

    path_df = load_path_summaries(args.path_summary, args.path_labels)
    if not path_df.empty:
        path_df.to_csv(out_dir / "combined_screened_path_summary.csv", index=False)
        plot_feature_path(
            path_df,
            out_dir / "screened_feature_path_reliance.png",
            "Screened KAN feature-path reliance",
        )
        plot_pair_path(
            path_df,
            out_dir / "screened_pair_path_reliance.png",
            "Screened KAN pair-path reliance",
        )

    print(f"Wrote figures to {out_dir}")


if __name__ == "__main__":
    main()
