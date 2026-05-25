import argparse
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


C_MAP = {
    "core_interaction_c01": 0.1,
    "core_interaction_c025": 0.25,
    "core_interaction_c05": 0.5,
    "core_interaction_c1": 1.0,
}

SCREEN_ORDER = ["raw", "rf", "oracle_support", "random", "exclude_interaction"]
SCREEN_LABEL = {
    "raw": "Raw KAN",
    "rf": "RF-screened",
    "oracle_support": "Oracle-support",
    "random": "Random",
    "exclude_interaction": "Exclude",
}


def parse_tag(path):
    name = Path(path).name
    m = re.search(r"(core_interaction_c(?:01|025|05|1))_n(\d+)_d(\d+)", name)
    if not m:
        return None
    return m.group(1), int(m.group(2)), int(m.group(3))


def normalize_summary(df, fn, n, d):
    df = df.copy()
    if "function" not in df.columns:
        df["function"] = fn
    df["samples"] = n
    df["dimension"] = d
    df["interaction_strength"] = df["function"].map(C_MAP)
    if df["interaction_strength"].isna().any() and fn in C_MAP:
        df["interaction_strength"] = C_MAP[fn]

    for base in ["test_mse", "variable_f1", "interaction_f1"]:
        mean_col = base + "_mean"
        if mean_col in df.columns:
            df[base] = df[mean_col]

    return df


def load_summaries(pattern):
    rows = []
    for p in sorted(glob.glob(pattern)):
        parsed = parse_tag(p)
        if parsed is None:
            print("[skip bad filename]", p)
            continue
        fn, n, d = parsed
        df = pd.read_csv(p)
        rows.append(normalize_summary(df, fn, n, d))

    if not rows:
        raise FileNotFoundError(f"No summary files matched: {pattern}")

    out = pd.concat(rows, ignore_index=True)
    return out


def plot_heatmap(df, metric, mode, dim, out_dir):
    sub = df[(df["screen_mode"] == mode) & (df["dimension"] == dim)].copy()
    if sub.empty:
        print(f"[skip] no data for {metric}, {mode}, d={dim}")
        return

    cs = sorted(sub["interaction_strength"].dropna().unique(), reverse=True)
    ns = sorted(sub["samples"].dropna().unique())

    mat = np.full((len(cs), len(ns)), np.nan)
    for i, c in enumerate(cs):
        for j, n in enumerate(ns):
            g = sub[(sub["interaction_strength"] == c) & (sub["samples"] == n)]
            if not g.empty:
                mat[i, j] = float(g[metric].mean())

    plot_mat = mat.copy()
    color_label = metric
    if metric == "test_mse":
        plot_mat = np.log10(np.maximum(plot_mat, 1e-12))
        color_label = "log10(test_mse)"

    plt.figure(figsize=(7.2, 4.8))
    im = plt.imshow(plot_mat, aspect="auto")
    plt.colorbar(im, label=color_label)

    plt.xticks(np.arange(len(ns)), ns)
    plt.yticks(np.arange(len(cs)), cs)
    plt.xlabel("sample size n")
    plt.ylabel("interaction strength c")
    plt.title(f"{SCREEN_LABEL.get(mode, mode)} | {metric} | d={dim}")

    for i in range(len(cs)):
        for j in range(len(ns)):
            val = mat[i, j]
            if np.isnan(val):
                txt = "NA"
            elif metric == "test_mse":
                txt = f"{val:.2g}"
            else:
                txt = f"{val:.2f}"
            plt.text(j, i, txt, ha="center", va="center", fontsize=9)

    plt.tight_layout()
    out_path = out_dir / f"heatmap_{metric}_{mode}_d{dim}.pdf"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print("[saved]", out_path)


def plot_line_hard(df, metric, dim, out_dir):
    plt.figure(figsize=(8, 5))

    for mode in ["raw", "rf", "oracle_support"]:
        for c in [0.1, 0.25, 0.5, 1.0]:
            sub = df[
                (df["screen_mode"] == mode)
                & (df["dimension"] == dim)
                & (df["interaction_strength"] == c)
            ].sort_values("samples")
            if sub.empty:
                continue
            label = f"{SCREEN_LABEL.get(mode, mode)}, c={c}"
            plt.plot(sub["samples"], sub[metric], marker="o", linewidth=2, label=label)

    plt.xlabel("sample size n")
    plt.ylabel(metric)
    plt.title(f"Hard-regime transition: {metric}, d={dim}")
    if metric == "test_mse":
        plt.yscale("log")
    else:
        plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()

    out_path = out_dir / f"lines_{metric}_d{dim}.pdf"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print("[saved]", out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_glob", default="results/hard_regime/summaries/*_summary.csv")
    parser.add_argument("--detail_glob", default=None)  # reserved, not needed here
    parser.add_argument("--out_dir", default="results/hard_regime/phase_figures")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_summaries(args.summary_glob)
    master_path = out_dir / "hard_regime_master_summary.csv"
    df.to_csv(master_path, index=False)
    print("[saved]", master_path)

    print("\nLoaded rows:", len(df))
    print(df[["function", "samples", "dimension", "screen_mode", "test_mse", "variable_f1", "interaction_f1"]].head().to_string(index=False))

    for dim in sorted(df["dimension"].dropna().unique()):
        for mode in SCREEN_ORDER:
            for metric in ["interaction_f1", "variable_f1", "test_mse"]:
                if metric in df.columns:
                    plot_heatmap(df, metric, mode, int(dim), out_dir)

        for metric in ["interaction_f1", "variable_f1", "test_mse"]:
            if metric in df.columns:
                plot_line_hard(df, metric, int(dim), out_dir)

    print("\nDone. Figures written to:", out_dir)


if __name__ == "__main__":
    main()
