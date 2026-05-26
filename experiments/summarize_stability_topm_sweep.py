from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


TAG_RE = re.compile(r"(?P<function>core_interaction_c[0-9]+)_n(?P<n>[0-9]+)_d(?P<d>[0-9]+)_topm(?P<top_m>[0-9]+)")


def parse_tag(path: Path) -> dict:
    match = TAG_RE.search(path.name)
    if not match:
        raise ValueError(f"Cannot parse sweep tag from {path}")
    out = match.groupdict()
    out["n"] = int(out["n"])
    out["d"] = int(out["d"])
    out["top_m"] = int(out["top_m"])
    return out


def function_to_c(function: str) -> float:
    mapping = {
        "core_interaction_c01": 0.10,
        "core_interaction_c025": 0.25,
        "core_interaction_c05": 0.50,
        "core_interaction_c1": 1.00,
    }
    return mapping.get(function, float("nan"))


def literal(value, default):
    if not isinstance(value, str):
        return default
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return default


def load_summary(summary_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(summary_dir.glob("*_summary.csv")):
        meta = parse_tag(path)
        df = pd.read_csv(path)
        df = df[df["screen_mode"].astype(str).isin(["ss_kan_variable", "ss_kan_pair"])].copy()
        if df.empty:
            continue
        for key, value in meta.items():
            df[key] = value
        df["interaction_strength"] = df["function"].map(function_to_c)
        frames.append(df)
    if not frames:
        raise RuntimeError(f"No stability-selected KAN summaries found in {summary_dir}")
    return pd.concat(frames, ignore_index=True)


def load_detail_support_rates(detail_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(detail_dir.glob("*_detail.csv")):
        meta = parse_tag(path)
        df = pd.read_csv(path)
        df = df[df["screen_mode"].astype(str).isin(["ss_kan_variable", "ss_kan_pair"])].copy()
        if df.empty:
            continue
        selected = df["selected_support"].apply(lambda x: literal(x, []))
        df["support_contains_all_true"] = selected.apply(lambda xs: int({0, 1, 2, 3}.issubset(set(xs))))
        df["support_contains_interaction_endpoints"] = selected.apply(lambda xs: int({2, 3}.issubset(set(xs))))
        for key, value in meta.items():
            df[key] = value
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    detail = pd.concat(frames, ignore_index=True)
    return (
        detail.groupby(["function", "screen_mode", "n", "d", "top_m"], dropna=False)
        .agg(
            support_contains_all_true_mean=("support_contains_all_true", "mean"),
            support_contains_interaction_endpoints_mean=("support_contains_interaction_endpoints", "mean"),
        )
        .reset_index()
    )


def plot_metric(df: pd.DataFrame, metric: str, out_path: Path, method: str) -> None:
    df = df[df["screen_mode"].astype(str) == method].copy()
    if df.empty:
        return
    strengths = sorted(df["interaction_strength"].dropna().unique())
    fig, axes = plt.subplots(1, len(strengths), figsize=(12, 3.2), sharey=True)
    if len(strengths) == 1:
        axes = [axes]

    for ax, c in zip(axes, strengths):
        sub = df[df["interaction_strength"] == c]
        for (d, top_m), g in sorted(sub.groupby(["d", "top_m"]), key=lambda kv: (kv[0][0], kv[0][1])):
            g = g.sort_values("n")
            ax.plot(g["n"], g[metric], marker="o", linewidth=2, label=f"d={int(d)}, m={int(top_m)}")
        ax.set_title(f"c={c:g}")
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted(df["n"].unique()))
        ax.set_xticklabels([str(int(n)) for n in sorted(df["n"].unique())])
        ax.set_xlabel("n")
        ax.grid(True, axis="y", alpha=0.25)

    axes[0].set_ylabel(metric.replace("_mean", "").replace("_", " "))
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(5, max(1, len(labels))), frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_dir", default="results/stability_kan/afternoon_topm/summaries")
    parser.add_argument("--detail_dir", default="results/stability_kan/afternoon_topm/details")
    parser.add_argument("--out_dir", default="results/stability_kan/afternoon_topm")
    args = parser.parse_args()

    summary_dir = Path(args.summary_dir)
    detail_dir = Path(args.detail_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = load_summary(summary_dir)
    support_rates = load_detail_support_rates(detail_dir)
    if not support_rates.empty:
        summary = summary.merge(support_rates, on=["function", "screen_mode", "n", "d", "top_m"], how="left")

    keep = [
        "function",
        "interaction_strength",
        "screen_mode",
        "n",
        "d",
        "top_m",
        "test_mse_mean",
        "variable_f1_mean",
        "explain_interaction_endpoint_recall_mean",
        "interaction_f1_mean",
        "true_interaction_mean_score_margin_mean",
        "support_contains_all_true_mean",
        "support_contains_interaction_endpoints_mean",
        "num_runs",
    ]
    keep = [col for col in keep if col in summary.columns]
    summary = summary[keep].sort_values(["interaction_strength", "screen_mode", "d", "n", "top_m"])

    csv_path = out_dir / "topm_sweep_summary.csv"
    summary.to_csv(csv_path, index=False)

    fig_dir = out_dir / "figures"
    for method in sorted(summary["screen_mode"].dropna().astype(str).unique()):
        label = method.replace("ss_kan_", "")
        plot_metric(summary, "interaction_f1_mean", fig_dir / f"topm_sweep_{label}_interaction_f1.png", method)
        plot_metric(summary, "explain_interaction_endpoint_recall_mean", fig_dir / f"topm_sweep_{label}_endpoint_recall.png", method)

    print(summary.to_string(index=False))
    print(f"Saved summary: {csv_path}")
    print(f"Saved figures: {fig_dir}")


if __name__ == "__main__":
    main()
