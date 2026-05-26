from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from experiments.paper_plot_style import configure_paper_plots, clean_axis, save_figure
except Exception:  # pragma: no cover - fallback for standalone use
    configure_paper_plots = None
    clean_axis = None
    save_figure = None


def parse_literal(value: Any, default: Any):
    if isinstance(value, (list, tuple, dict)):
        return value
    if value is None:
        return default
    try:
        if isinstance(value, float) and np.isnan(value):
            return default
    except TypeError:
        pass
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return default
    try:
        return ast.literal_eval(text)
    except Exception:
        return default


def canonical_pair(pair: Iterable[int]) -> tuple[int, int]:
    a, b = pair
    return tuple(sorted((int(a), int(b))))


def pair_text(value: Any) -> str:
    pairs = parse_literal(value, [])
    if not pairs:
        return "none"
    try:
        return ";".join(f"({int(i)},{int(j)})" for i, j in pairs)
    except Exception:
        return str(value)


def support_text(value: Any) -> str:
    vals = parse_literal(value, [])
    if not vals:
        return "[]"
    return "[" + ",".join(str(int(v)) for v in vals) + "]"


def add_parsed_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["support_text"] = out["selected_screen_features"].map(support_text)
    out["selected_pair_text"] = out["selected_interactions"].map(pair_text)
    out["contains_pair_23"] = out["support_text"].str.contains(r"\b2\b") & out["support_text"].str.contains(r"\b3\b")
    for col in [
        "interaction_f1",
        "test_mse",
        "screen_contains_true_interactions",
        "screen_interaction_endpoint_recall",
        "true_interaction_rank_mean",
        "true_interaction_mean_score_margin",
        "true_interaction_beats_all_false",
    ]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def aggregate_detail(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["wave", "method", "function", "samples", "dimension", "top_m"]
    metrics = [
        "interaction_f1",
        "test_mse",
        "screen_contains_true_interactions",
        "screen_interaction_endpoint_recall",
        "true_interaction_rank_mean",
        "true_interaction_mean_score_margin",
        "true_interaction_beats_all_false",
    ]
    agg = {}
    for col in metrics:
        if col in df.columns:
            agg[col] = ["mean", "std"]
    out = df.groupby(group_cols, dropna=False).agg(agg).reset_index()
    out.columns = ["_".join(str(x) for x in col if x != "").rstrip("_") for col in out.columns]
    counts = df.groupby(group_cols, dropna=False).size().reset_index(name="num_runs")
    return out.merge(counts, on=group_cols, how="left")


def make_key_regime_table(summary: pd.DataFrame) -> pd.DataFrame:
    key = summary[
        summary["function"].astype(str).isin([
            "core_interaction_c025",
            "core_interaction_c05",
            "core_interaction_c1",
            "feynman_energy",
            "feynman_coulomb",
            "feynman_gravity",
        ])
    ].copy()
    return key.sort_values(
        ["function", "dimension", "samples", "top_m", "interaction_f1_mean", "method"],
        ascending=[True, True, True, True, False, True],
    )


def make_failure_table(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, sub in detail.groupby(["wave", "method", "function", "samples", "dimension", "top_m"], dropna=False):
        failed = sub[sub["interaction_f1"].fillna(0.0) < 1.0]
        if failed.empty:
            continue
        pair_counts = failed["selected_pair_text"].value_counts(dropna=False)
        support_counts = failed["support_text"].value_counts(dropna=False)
        rows.append({
            "wave": keys[0],
            "method": keys[1],
            "function": keys[2],
            "samples": keys[3],
            "dimension": keys[4],
            "top_m": keys[5],
            "num_failures": int(len(failed)),
            "num_runs": int(len(sub)),
            "failure_rate": float(len(failed) / len(sub)),
            "mean_true_pair_rank_on_fail": float(failed["true_interaction_rank_mean"].mean()),
            "mean_margin_on_fail": float(failed["true_interaction_mean_score_margin"].mean()),
            "top_wrong_pairs": " | ".join(f"{idx}:{cnt}" for idx, cnt in pair_counts.head(5).items()),
            "top_failed_supports": " | ".join(f"{idx}:{cnt}" for idx, cnt in support_counts.head(5).items()),
        })
    return pd.DataFrame(rows).sort_values(
        ["failure_rate", "function", "dimension", "samples", "method"],
        ascending=[False, True, True, True, True],
    )


def make_support_table(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, sub in detail.groupby(["wave", "method", "function", "samples", "dimension", "top_m"], dropna=False):
        counts = sub["support_text"].value_counts(dropna=False)
        rows.append({
            "wave": keys[0],
            "method": keys[1],
            "function": keys[2],
            "samples": keys[3],
            "dimension": keys[4],
            "top_m": keys[5],
            "num_unique_supports": int(counts.size),
            "top_support": str(counts.index[0]),
            "top_support_count": int(counts.iloc[0]),
            "support_contains_true_pair_rate": float(sub["screen_contains_true_interactions"].mean()),
            "endpoint_recall": float(sub["screen_interaction_endpoint_recall"].mean()),
        })
    return pd.DataFrame(rows).sort_values(
        ["function", "dimension", "samples", "top_m", "method"]
    )


def plot_c025_boundary(summary: pd.DataFrame, out_dir: Path) -> None:
    sub = summary[summary["function"].astype(str).eq("core_interaction_c025")].copy()
    if sub.empty:
        return
    keep_methods = ["grad_stability_var", "feature_stability_var", "edge_stability_var", "feature_edge_hybrid"]
    sub = sub[sub["method"].isin(keep_methods)]
    labels = {
        "grad_stability_var": "Gradient",
        "feature_stability_var": "KAN feature",
        "edge_stability_var": "KAN edge",
        "feature_edge_hybrid": "Feature+edge",
    }
    settings = []
    for _, row in sub[["dimension", "samples", "top_m"]].drop_duplicates().sort_values(["dimension", "samples", "top_m"]).iterrows():
        settings.append((int(row["dimension"]), int(row["samples"]), int(row["top_m"])))
    x = np.arange(len(settings))
    width = 0.18
    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    for idx, method in enumerate(keep_methods):
        vals = []
        for d, n, m in settings:
            hit = sub[(sub["method"] == method) & (sub["dimension"] == d) & (sub["samples"] == n) & (sub["top_m"] == m)]
            vals.append(float(hit["interaction_f1_mean"].iloc[0]) if not hit.empty else np.nan)
        ax.bar(x + (idx - 1.5) * width, vals, width=width, label=labels[method])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Interaction F1")
    ax.set_xticks(x)
    ax.set_xticklabels([f"d={d}\nn={n}\nm={m}" for d, n, m in settings], fontsize=7)
    ax.legend(ncol=2, frameon=False, fontsize=8)
    if clean_axis is not None:
        clean_axis(ax)
    fig.tight_layout()
    if save_figure is not None:
        save_figure(fig, out_dir / "c025_boundary_validation")
    else:
        fig.savefig(out_dir / "c025_boundary_validation.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_stress(summary: pd.DataFrame, out_dir: Path) -> None:
    sub = summary[
        summary["function"].astype(str).isin(["core_interaction_c025", "core_interaction_c05", "core_interaction_c1"])
        & summary["wave"].astype(str).eq("stress")
    ].copy()
    if sub.empty:
        return
    sub = sub[sub["method"].isin(["feature_stability_var", "feature_edge_hybrid"])]
    fig, ax = plt.subplots(figsize=(6.8, 2.7))
    labels = {
        "core_interaction_c025": "c=0.25",
        "core_interaction_c05": "c=0.5",
        "core_interaction_c1": "c=1.0",
    }
    colors = {"feature_stability_var": "#0072B2", "feature_edge_hybrid": "#D55E00"}
    xlabels = []
    x = []
    vals = {"feature_stability_var": [], "feature_edge_hybrid": []}
    pos = 0
    for fn in ["core_interaction_c025", "core_interaction_c05", "core_interaction_c1"]:
        for dim in sorted(sub[sub["function"] == fn]["dimension"].unique()):
            x.append(pos)
            xlabels.append(f"{labels[fn]}\nd={int(dim)}")
            for method in vals:
                hit = sub[(sub["function"] == fn) & (sub["dimension"] == dim) & (sub["method"] == method)]
                vals[method].append(float(hit["interaction_f1_mean"].iloc[0]) if not hit.empty else np.nan)
            pos += 1
    width = 0.32
    ax.bar(np.array(x) - width / 2, vals["feature_stability_var"], width=width, color=colors["feature_stability_var"], label="KAN feature")
    ax.bar(np.array(x) + width / 2, vals["feature_edge_hybrid"], width=width, color=colors["feature_edge_hybrid"], label="Feature+edge")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Interaction F1")
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=7)
    ax.legend(frameon=False, fontsize=8)
    if clean_axis is not None:
        clean_axis(ax)
    fig.tight_layout()
    if save_figure is not None:
        save_figure(fig, out_dir / "high_dim_stress_validation")
    else:
        fig.savefig(out_dir / "high_dim_stress_validation.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    if configure_paper_plots is not None:
        configure_paper_plots(usetex=False)

    detail = pd.read_csv(run_dir / "innovation_detail.csv")
    detail = add_parsed_columns(detail)
    summary = aggregate_detail(detail)
    key = make_key_regime_table(summary)
    failures = make_failure_table(detail)
    supports = make_support_table(detail)

    summary.to_csv(out_dir / "strict_validation_summary_recomputed.csv", index=False)
    key.to_csv(out_dir / "strict_validation_key_regimes.csv", index=False)
    failures.to_csv(out_dir / "strict_validation_failure_modes.csv", index=False)
    supports.to_csv(out_dir / "strict_validation_supports.csv", index=False)

    plot_c025_boundary(summary, out_dir)
    plot_stress(summary, out_dir)

    print(f"Wrote analysis to {out_dir}")
    print("\nTop key regimes:")
    cols = [
        "wave", "method", "function", "samples", "dimension", "top_m",
        "interaction_f1_mean", "screen_contains_true_interactions_mean",
        "test_mse_mean", "num_runs",
    ]
    cols = [c for c in cols if c in key.columns]
    print(key.sort_values("interaction_f1_mean", ascending=False)[cols].head(20).to_string(index=False))
    print("\nLargest failure modes:")
    if not failures.empty:
        print(failures.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
