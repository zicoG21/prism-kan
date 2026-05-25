from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


METHOD_LABELS = {
    "raw": "Raw",
    "ss_kan_variable": "SS-KAN-V",
    "rf": "RF",
    "oracle_support": "Oracle",
}

METHOD_COLORS = {
    "raw": "#4b5563",
    "ss_kan_variable": "#0f766e",
    "rf": "#b45309",
    "oracle_support": "#111827",
}


def parse_summary_specs(specs: list[str]) -> list[tuple[int, Path]]:
    parsed = []
    for spec in specs:
        if ":" not in spec:
            raise ValueError(f"Expected N:path summary spec, got {spec!r}")
        n_text, path_text = spec.split(":", 1)
        parsed.append((int(n_text), Path(path_text)))
    return parsed


def load_boundary(summary_specs: list[tuple[int, Path]], methods: list[str]) -> pd.DataFrame:
    frames = []
    for n, path in summary_specs:
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        df["n"] = int(n)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    keep_cols = [
        "n",
        "function",
        "interaction_strength",
        "screen_mode",
        "test_mse_mean",
        "variable_f1_mean",
        "explain_interaction_endpoint_recall_mean",
        "interaction_f1_mean",
        "true_interaction_mean_score_margin_mean",
        "num_runs",
    ]
    out = combined[combined["screen_mode"].isin(methods)][keep_cols].copy()
    out["screen_mode"] = pd.Categorical(out["screen_mode"], methods, ordered=True)
    return out.sort_values(["interaction_strength", "n", "screen_mode"])


def plot_boundary(df: pd.DataFrame, methods: list[str], out_path: Path) -> None:
    strengths = sorted(df["interaction_strength"].dropna().unique())
    fig, axes = plt.subplots(1, len(strengths), figsize=(12, 3.2), sharey=True)
    if len(strengths) == 1:
        axes = [axes]

    for ax, strength in zip(axes, strengths):
        sub = df[df["interaction_strength"] == strength]
        for method in methods:
            method_df = sub[sub["screen_mode"] == method].sort_values("n")
            if method_df.empty:
                continue
            ax.plot(
                method_df["n"],
                method_df["interaction_f1_mean"],
                marker="o",
                linewidth=2,
                label=METHOD_LABELS.get(method, method),
                color=METHOD_COLORS.get(method),
            )
        ax.set_title(f"c={strength:g}")
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted(df["n"].unique()))
        ax.set_xticklabels([str(int(n)) for n in sorted(df["n"].unique())])
        ax.set_xlabel("n")
        ax.set_ylim(-0.03, 1.05)
        ax.grid(True, axis="y", alpha=0.25)

    axes[0].set_ylabel("Interaction F1")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(methods), frameon=False)
    fig.suptitle("Quick SS-KAN-V recovery boundary (d=100)", y=1.08)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summaries",
        nargs="+",
        default=[
            "256:results/stability_kan/quick_d100_n256_variable_summary.csv",
            "512:results/stability_kan/quick_d100_n512_summary.csv",
            "1024:results/stability_kan/quick_d100_n1024_summary.csv",
        ],
        help="Summary inputs as N:path.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["raw", "ss_kan_variable", "rf", "oracle_support"],
    )
    parser.add_argument("--out_csv", default="results/stability_kan/quick_variable_boundary_summary.csv")
    parser.add_argument(
        "--out_fig",
        default="results/stability_kan/figures/quick_variable_boundary_interaction_f1.png",
    )
    args = parser.parse_args()

    boundary = load_boundary(parse_summary_specs(args.summaries), args.methods)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    boundary.to_csv(out_csv, index=False)
    plot_boundary(boundary, args.methods, Path(args.out_fig))

    print(boundary.to_string(index=False))
    print(f"Saved CSV: {out_csv}")
    print(f"Saved figure: {args.out_fig}")


if __name__ == "__main__":
    main()
