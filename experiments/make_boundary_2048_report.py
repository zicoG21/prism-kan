from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


METHOD_LABELS = {
    "raw": "Raw",
    "rf": "RF",
    "oracle_support": "Oracle",
    "random": "Random",
    "exclude_interaction": "Exclude",
    "ss_kan_variable": "SS-KAN-V",
    "ss_kan_pair": "SS-KAN-P",
}

METHOD_COLORS = {
    "raw": "#4b5563",
    "rf": "#b45309",
    "oracle_support": "#111827",
    "random": "#9ca3af",
    "exclude_interaction": "#dc2626",
    "ss_kan_variable": "#0f766e",
    "ss_kan_pair": "#2563eb",
}


def load_baseline(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    keep = [
        "function",
        "interaction_strength",
        "screen_mode",
        "dimension",
        "samples",
        "test_mse_mean",
        "variable_f1_mean",
        "explain_interaction_endpoint_recall_mean",
        "interaction_f1_mean",
        "true_interaction_mean_score_margin_mean",
        "num_runs",
    ]
    out = df[[col for col in keep if col in df.columns]].copy()
    out["top_m"] = 4
    out["source"] = "baseline"
    return out


def load_stability(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    rename = {"d": "dimension", "n": "samples"}
    df = df.rename(columns=rename)
    df["source"] = "stability"
    return df


def best_stability(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[df["screen_mode"].isin(["ss_kan_variable", "ss_kan_pair"])].copy()
    if sub.empty:
        return sub
    sub = sub.sort_values(
        ["interaction_f1_mean", "test_mse_mean"],
        ascending=[False, True],
    )
    return sub.head(1)


def make_report(baseline: pd.DataFrame, stability: pd.DataFrame) -> pd.DataFrame:
    methods = ["raw", "rf", "oracle_support", "random", "exclude_interaction"]
    rows = [baseline[baseline["screen_mode"] == method] for method in methods]
    rows.append(stability[stability["screen_mode"].isin(["ss_kan_variable", "ss_kan_pair"])])
    report = pd.concat([x for x in rows if not x.empty], ignore_index=True, sort=False)
    order = methods + ["ss_kan_variable", "ss_kan_pair"]
    report["screen_mode"] = pd.Categorical(report["screen_mode"], order, ordered=True)
    return report.sort_values(["screen_mode", "top_m"])


def plot_report(report: pd.DataFrame, out_path: Path) -> None:
    plot = report[
        report["screen_mode"].isin(["raw", "rf", "oracle_support", "ss_kan_variable", "ss_kan_pair"])
    ].copy()
    plot["label"] = plot["screen_mode"].map(METHOD_LABELS).astype(str)
    is_stability = plot["screen_mode"].astype(str).str.startswith("ss_kan_")
    plot.loc[is_stability, "label"] = plot.loc[is_stability, "label"] + ", m=" + plot.loc[
        is_stability, "top_m"
    ].astype(int).astype(str)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    metrics = [
        ("interaction_f1_mean", "Interaction F1"),
        ("explain_interaction_endpoint_recall_mean", "Endpoint recall"),
        ("test_mse_mean", "Test MSE"),
    ]
    colors = [METHOD_COLORS.get(str(m), "#6b7280") for m in plot["screen_mode"]]
    for ax, (metric, title) in zip(axes, metrics):
        ax.bar(range(len(plot)), plot[metric], color=colors, width=0.72)
        ax.set_title(title)
        ax.set_xticks(range(len(plot)))
        ax.set_xticklabels(plot["label"], rotation=45, ha="right", fontsize=8)
        ax.grid(True, axis="y", alpha=0.25)
        if metric != "test_mse_mean":
            ax.set_ylim(0, 1.05)
    fig.suptitle("Boundary follow-up: c=0.25, d=100, n=2048", y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_summary", required=True)
    parser.add_argument("--stability_summary", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    baseline = load_baseline(Path(args.baseline_summary))
    stability = load_stability(Path(args.stability_summary))
    report = make_report(baseline, stability)
    best = best_stability(stability)

    report.to_csv(out_dir / "boundary_2048_method_report.csv", index=False)
    best.to_csv(out_dir / "boundary_2048_best_stability.csv", index=False)
    plot_report(report, out_dir / "figures" / "boundary_2048_method_report.png")
    plot_report(report, out_dir / "figures" / "boundary_2048_method_report.pdf")

    cols = [
        "screen_mode",
        "top_m",
        "test_mse_mean",
        "variable_f1_mean",
        "explain_interaction_endpoint_recall_mean",
        "interaction_f1_mean",
        "true_interaction_mean_score_margin_mean",
        "num_runs",
    ]
    print(report[[c for c in cols if c in report.columns]].to_string(index=False))
    if not best.empty:
        print("\nBest stability setting:")
        print(best[[c for c in cols if c in best.columns]].to_string(index=False))
    print(f"Saved report under {out_dir}")


if __name__ == "__main__":
    main()
