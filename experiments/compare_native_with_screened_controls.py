from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from experiments.paper_plot_style import OKABE_ITO, clean_axis, configure_paper_plots, save_figure
except Exception:  # pragma: no cover
    OKABE_ITO = {
        "blue": "#0072B2",
        "orange": "#E69F00",
        "green": "#009E73",
        "vermillion": "#D55E00",
        "purple": "#CC79A7",
        "gray": "#6B7280",
    }
    clean_axis = None
    configure_paper_plots = None
    save_figure = None


METRICS = [
    "interaction_f1",
    "test_mse",
    "variable_f1",
    "screen_contains_true_interactions",
    "screen_interaction_endpoint_recall",
    "true_interaction_rank_mean",
    "true_interaction_mean_score_margin",
    "true_interaction_beats_all_false",
]

DISPLAY_METHODS = {
    "feature_stability_var": "KAN feature-stability",
    "feature_edge_hybrid": "KAN feature+edge",
    "grad_stability_var": "KAN grad-stability",
    "edge_stability_var": "KAN edge-stability",
    "single_grad_var": "Single KAN gradient",
    "single_feature_var": "Single KAN feature",
    "single_edge_var": "Single KAN edge",
    "single_feature_edge_hybrid": "Single KAN feature+edge",
    "single_edge_pair_hybrid": "Single KAN edge-pair",
    "rf": "RF-screened KAN",
    "oracle_support": "Oracle-support KAN",
    "random": "Random-support KAN",
    "exclude_interaction": "Exclude-endpoints KAN",
}

PLOT_METHOD_LABELS = {
    "feature_stability_var": "KAN-F",
    "feature_edge_hybrid": "KAN-FE",
    "rf": "RF",
    "oracle_support": "Oracle",
    "random": "Random",
    "exclude_interaction": "Exclude",
}

METHOD_ORDER = [
    "feature_stability_var",
    "feature_edge_hybrid",
    "grad_stability_var",
    "edge_stability_var",
    "single_feature_var",
    "single_grad_var",
    "single_edge_var",
    "single_feature_edge_hybrid",
    "single_edge_pair_hybrid",
    "rf",
    "oracle_support",
    "random",
    "exclude_interaction",
]


def numericize(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def aggregate(df: pd.DataFrame, method_col: str) -> pd.DataFrame:
    df = df[df["status"].astype(str).eq("ok")].copy()
    df = numericize(df, METRICS + ["samples", "dimension", "top_m"])
    group_cols = [method_col, "function", "samples", "dimension", "top_m"]
    agg = {}
    for col in METRICS:
        if col in df.columns:
            agg[col] = ["mean", "std"]
    if not agg:
        return pd.DataFrame()
    out = df.groupby(group_cols, dropna=False).agg(agg).reset_index()
    out.columns = ["_".join(str(x) for x in col if x != "").rstrip("_") for col in out.columns]
    counts = df.groupby(group_cols, dropna=False).size().reset_index(name="num_runs")
    out = out.merge(counts, on=group_cols, how="left")
    out = out.rename(columns={method_col: "method"})
    return out


def load_native(native_run_dir: Path) -> pd.DataFrame:
    path = native_run_dir / "innovation_detail.csv"
    if not path.exists():
        return pd.DataFrame()
    detail = pd.read_csv(path)
    summary = aggregate(detail, "method")
    summary["family"] = "kan_native"
    return summary


def load_screened(screened_run_dir: Path) -> pd.DataFrame:
    pieces = []
    for path in sorted(screened_run_dir.glob("*_screen_eval.csv")):
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            print(f"[WARN] could not read {path}: {exc}")
            continue
        df["source_file"] = path.name
        pieces.append(df)
    if not pieces:
        return pd.DataFrame()
    detail = pd.concat(pieces, ignore_index=True, sort=False)
    summary = aggregate(detail, "screen_mode")
    summary["family"] = "screened_control"
    return summary


def load_one_shot(one_shot_dir: Path) -> pd.DataFrame:
    pieces = []
    for path in sorted(one_shot_dir.glob("*_detail.csv")):
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            print(f"[WARN] could not read {path}: {exc}")
            continue
        df["source_file"] = path.name
        pieces.append(df)
    if not pieces:
        direct = one_shot_dir / "detail.csv"
        if direct.exists():
            pieces.append(pd.read_csv(direct))
    if not pieces:
        return pd.DataFrame()
    detail = pd.concat(pieces, ignore_index=True, sort=False)
    summary = aggregate(detail, "method")
    summary["family"] = "one_shot_kan"
    return summary


def add_readable_columns(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    out["method_label"] = out["method"].map(DISPLAY_METHODS).fillna(out["method"])
    out["setting"] = (
        out["function"].astype(str)
        + " | d="
        + out["dimension"].astype(int).astype(str)
        + " | n="
        + out["samples"].astype(int).astype(str)
        + " | m="
        + out["top_m"].astype(int).astype(str)
    )
    order = {m: i for i, m in enumerate(METHOD_ORDER)}
    out["method_order"] = out["method"].map(order).fillna(len(order)).astype(int)
    return out.sort_values(["function", "dimension", "samples", "top_m", "method_order"])


def make_key_table(summary: pd.DataFrame) -> pd.DataFrame:
    keep_methods = set(METHOD_ORDER)
    key = summary[summary["method"].isin(keep_methods)].copy()
    key = key[
        key["function"].astype(str).isin(
            ["core_interaction_c025", "core_interaction_c05", "core_interaction_c1"]
        )
    ]
    cols = [
        "family",
        "method",
        "method_label",
        "function",
        "dimension",
        "samples",
        "top_m",
        "interaction_f1_mean",
        "interaction_f1_std",
        "test_mse_mean",
        "screen_contains_true_interactions_mean",
        "screen_interaction_endpoint_recall_mean",
        "num_runs",
    ]
    cols = [c for c in cols if c in key.columns]
    return key[cols].sort_values(["function", "dimension", "samples", "top_m", "family", "method_label"])


def make_gap_table(summary: pd.DataFrame) -> pd.DataFrame:
    metric = "interaction_f1_mean"
    if metric not in summary.columns:
        return pd.DataFrame()
    base_cols = ["function", "dimension", "samples", "top_m"]
    pivot = summary.pivot_table(index=base_cols, columns="method", values=metric, aggfunc="mean").reset_index()
    rows = []
    for _, row in pivot.iterrows():
        native = row.get("feature_stability_var", np.nan)
        hybrid = row.get("feature_edge_hybrid", np.nan)
        rf = row.get("rf", np.nan)
        oracle = row.get("oracle_support", np.nan)
        random = row.get("random", np.nan)
        rows.append(
            {
                "function": row["function"],
                "dimension": int(row["dimension"]),
                "samples": int(row["samples"]),
                "top_m": int(row["top_m"]),
                "feature_stability_f1": native,
                "feature_edge_hybrid_f1": hybrid,
                "single_feature_f1": row.get("single_feature_var", np.nan),
                "single_grad_f1": row.get("single_grad_var", np.nan),
                "rf_f1": rf,
                "oracle_f1": oracle,
                "random_f1": random,
                "feature_minus_rf": native - rf if pd.notna(native) and pd.notna(rf) else np.nan,
                "hybrid_minus_rf": hybrid - rf if pd.notna(hybrid) and pd.notna(rf) else np.nan,
                "feature_minus_single_feature": native - row.get("single_feature_var", np.nan)
                if pd.notna(native) and pd.notna(row.get("single_feature_var", np.nan))
                else np.nan,
                "feature_minus_single_grad": native - row.get("single_grad_var", np.nan)
                if pd.notna(native) and pd.notna(row.get("single_grad_var", np.nan))
                else np.nan,
                "feature_minus_random": native - random if pd.notna(native) and pd.notna(random) else np.nan,
                "feature_oracle_gap": oracle - native if pd.notna(oracle) and pd.notna(native) else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["function", "dimension", "samples", "top_m"])


def plot_selected(summary: pd.DataFrame, out_dir: Path) -> None:
    if summary.empty or "interaction_f1_mean" not in summary.columns:
        return
    plot_rows = [
        ("core_interaction_c025", 100, 512, 4),
        ("core_interaction_c025", 100, 1024, 4),
        ("core_interaction_c025", 100, 1024, 6),
        ("core_interaction_c025", 100, 2048, 6),
    ]
    methods = [
        "feature_stability_var",
        "feature_edge_hybrid",
        "rf",
        "oracle_support",
        "random",
        "exclude_interaction",
    ]
    colors = {
        "feature_stability_var": OKABE_ITO["blue"],
        "feature_edge_hybrid": OKABE_ITO["green"],
        "rf": OKABE_ITO["orange"],
        "oracle_support": OKABE_ITO["purple"],
        "random": OKABE_ITO["gray"],
        "exclude_interaction": OKABE_ITO["vermillion"],
    }
    x = np.arange(len(plot_rows))
    width = 0.12
    fig, ax = plt.subplots(figsize=(6.85, 2.55))
    for idx, method in enumerate(methods):
        vals = []
        for fn, dim, n, top_m in plot_rows:
            hit = summary[
                summary["method"].eq(method)
                & summary["function"].eq(fn)
                & summary["dimension"].eq(dim)
                & summary["samples"].eq(n)
                & summary["top_m"].eq(top_m)
            ]
            vals.append(float(hit["interaction_f1_mean"].iloc[0]) if not hit.empty else np.nan)
        ax.bar(
            x + (idx - (len(methods) - 1) / 2) * width,
            vals,
            width=width,
            label=DISPLAY_METHODS[method],
            color=colors[method],
            edgecolor="white",
            linewidth=0.45,
        )
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Interaction F1")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [rf"$n={n}$" + "\n" + rf"$m={top_m}$" for _, _, n, top_m in plot_rows],
        fontsize=7.5,
    )
    ax.text(
        0.01,
        0.98,
        r"$c=0.25,\ d=100$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.0,
    )
    ax.legend(
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.24),
        columnspacing=0.9,
        handlelength=1.15,
        labels=[PLOT_METHOD_LABELS[m] for m in methods],
    )
    if clean_axis is not None:
        clean_axis(ax, grid=True)
    fig.tight_layout()
    if save_figure is not None:
        save_figure(fig, out_dir / "native_vs_screened_controls")
    else:
        fig.savefig(out_dir / "native_vs_screened_controls.pdf", bbox_inches="tight")
        fig.savefig(out_dir / "native_vs_screened_controls.png", dpi=450, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare KAN-native stability selection with screened controls.")
    parser.add_argument("--native_run_dir", default="results/innovation_loop/strict_validation_20260526_011917")
    parser.add_argument("--screened_run_dir", required=True)
    parser.add_argument("--one_shot_dir", default=None)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument(
        "--min_runs",
        type=int,
        default=1,
        help="Only include method/setting cells with at least this many runs in key tables and plots.",
    )
    args = parser.parse_args()

    native_run_dir = Path(args.native_run_dir)
    screened_run_dir = Path(args.screened_run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else screened_run_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    if configure_paper_plots is not None:
        configure_paper_plots(usetex=False)

    native = load_native(native_run_dir)
    screened = load_screened(screened_run_dir)
    pieces = [native, screened]
    if args.one_shot_dir:
        pieces.append(load_one_shot(Path(args.one_shot_dir)))
    combined = pd.concat(pieces, ignore_index=True, sort=False)
    if combined.empty:
        raise SystemExit("No native or screened rows found.")
    combined = add_readable_columns(combined)
    analysis_source = combined.copy()
    if args.min_runs > 1 and "num_runs" in analysis_source.columns:
        analysis_source = analysis_source[pd.to_numeric(analysis_source["num_runs"], errors="coerce") >= args.min_runs].copy()

    key = make_key_table(analysis_source)
    gaps = make_gap_table(analysis_source)
    combined.to_csv(out_dir / "native_screened_combined_summary.csv", index=False)
    key.to_csv(out_dir / "native_screened_key_table.csv", index=False)
    gaps.to_csv(out_dir / "native_screened_gap_table.csv", index=False)
    plot_selected(analysis_source, out_dir)

    cols = [
        "family",
        "method_label",
        "function",
        "dimension",
        "samples",
        "top_m",
        "interaction_f1_mean",
        "screen_contains_true_interactions_mean",
        "test_mse_mean",
        "num_runs",
    ]
    cols = [c for c in cols if c in key.columns]
    print(f"Wrote comparison to {out_dir}")
    if args.min_runs > 1:
        print(f"Filtered key tables/plots to cells with num_runs >= {args.min_runs}.")
    print(key[cols].to_string(index=False))
    if not gaps.empty:
        print("\nGaps vs controls:")
        print(gaps.to_string(index=False))


if __name__ == "__main__":
    main()
