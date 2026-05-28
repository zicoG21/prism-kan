from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from experiments.paper_plot_style import OKABE_ITO, clean_axis, configure_paper_plots, save_figure
except ModuleNotFoundError:
    from paper_plot_style import OKABE_ITO, clean_axis, configure_paper_plots, save_figure


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results" / "formula_aware_pair_scoring"

SETTINGS = [
    {
        "n": 1024,
        "support": RESULT_ROOT / "residual_native_c01_d1000_n1024_4probe_20260526" / "residual_support_detail.csv",
        "refit": RESULT_ROOT
        / "residual_native_c01_d1000_n1024_refit_top12_lamb0_width8_steps150_10seeds_20260526"
        / "pair_rescore_summary.csv",
        "adaptive_m": 12,
    },
    {
        "n": 2048,
        "support": RESULT_ROOT / "residual_native_c01_d1000_n2048_4probe_20260526" / "residual_support_detail.csv",
        "refit": RESULT_ROOT
        / "residual_native_c01_d1000_n2048_refit_top4_lamb0_width8_steps150_20260526"
        / "pair_rescore_summary.csv",
        "adaptive_m": 4,
    },
    {
        "n": 4096,
        "support": RESULT_ROOT / "residual_native_c01_d1000_n4096_4probe_20260526" / "residual_support_detail.csv",
        "refit": RESULT_ROOT
        / "residual_native_c01_d1000_n4096_refit_top4_lamb0_width8_steps150_20260526"
        / "pair_rescore_summary.csv",
        "adaptive_m": 4,
    },
]


def _require_row(df: pd.DataFrame, mask: pd.Series, description: str) -> pd.Series:
    rows = df.loc[mask]
    if rows.empty:
        raise ValueError(f"Missing row for {description}")
    return rows.iloc[0]


def build_summary() -> pd.DataFrame:
    rows = []
    for setting in SETTINGS:
        support_df = pd.read_csv(setting["support"])
        refit_df = pd.read_csv(setting["refit"])
        n = setting["n"]

        for support_name, top_m in [("fixed top-4", 4), ("adaptive", setting["adaptive_m"])]:
            support_row = _require_row(
                support_df,
                (support_df["method"] == "residual_feature_stability_var")
                & (support_df["top_m"] == top_m),
                f"support n={n}, top_m={top_m}",
            )
            rows.append(
                {
                    "samples": n,
                    "curve": support_name,
                    "top_m": top_m,
                    "endpoint_recall": float(support_row["screen_interaction_endpoint_recall"]),
                    "true_pair_retained": float(support_row["screen_contains_true_interactions"]),
                    "interaction_f1_mean": np.nan,
                    "interaction_f1_std": np.nan,
                    "test_mse_mean": np.nan,
                    "num_runs": np.nan,
                }
            )

        pair_row = _require_row(
            refit_df,
            (refit_df["source_method"] == "residual_feature_stability_var")
            & (refit_df["pair_score_method"] == "anova_abs"),
            f"refit n={n}, ANOVA-abs",
        )
        rows.append(
            {
                "samples": n,
                "curve": "adaptive refit + ANOVA",
                "top_m": int(pair_row["top_m"]),
                "endpoint_recall": float(pair_row["screen_interaction_endpoint_recall_mean"]),
                "true_pair_retained": float(pair_row["screen_contains_true_interactions_mean"]),
                "interaction_f1_mean": float(pair_row["interaction_f1_mean"]),
                "interaction_f1_std": float(pair_row["interaction_f1_std"]),
                "test_mse_mean": float(pair_row["test_mse_mean"]),
                "num_runs": int(pair_row["num_runs"]),
            }
        )
    return pd.DataFrame(rows)


def plot(summary: pd.DataFrame, out_base: Path) -> None:
    configure_paper_plots(usetex=False)
    ns = np.array(sorted(summary["samples"].unique()))
    x = np.arange(len(ns))

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(6.7, 2.25),
        gridspec_kw={"width_ratios": [1.05, 1.0], "wspace": 0.34},
    )

    ax = axes[0]
    for curve, color, marker, label in [
        ("fixed top-4", OKABE_ITO["gray"], "o", "fixed top-4"),
        ("adaptive", OKABE_ITO["blue"], "s", "adaptive top-$m$"),
    ]:
        subset = summary[summary["curve"] == curve].sort_values("samples")
        ax.plot(
            x,
            subset["endpoint_recall"].to_numpy(),
            marker=marker,
            markersize=4.0,
            linewidth=1.3,
            color=color,
            label=label,
        )
    for xpos, row in enumerate(summary[summary["curve"] == "adaptive"].sort_values("samples").itertuples()):
        ax.text(
            xpos,
            min(1.04, row.endpoint_recall + 0.055),
            f"$m={int(row.top_m)}$",
            ha="center",
            va="bottom",
            fontsize=7,
            color="#374151",
        )
    ax.set_title("(a) Residual support retention", loc="left", pad=3)
    ax.set_ylabel("Endpoint recall")
    ax.set_xlabel("Training samples $n$")
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_ylim(-0.04, 1.12)
    ax.set_yticks([0, 0.5, 1.0])
    ax.legend(frameon=False, loc="lower right", handlelength=1.6)
    clean_axis(ax, grid=True)

    ax = axes[1]
    refit = summary[summary["curve"] == "adaptive refit + ANOVA"].sort_values("samples")
    values = refit["interaction_f1_mean"].to_numpy()
    errors = refit["interaction_f1_std"].fillna(0.0).to_numpy()
    bars = ax.bar(
        x,
        values,
        yerr=errors,
        width=0.58,
        color=OKABE_ITO["green"],
        edgecolor="#1F2937",
        linewidth=0.35,
        error_kw={"linewidth": 0.7, "capsize": 2.0, "capthick": 0.7, "ecolor": "#374151"},
    )
    for bar, row in zip(bars, refit.itertuples()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            0.06,
            f"{int(row.num_runs)} runs",
            ha="center",
            va="bottom",
            rotation=90,
            fontsize=6.8,
            color="white",
        )
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(1.05, bar.get_height() + 0.045),
            f"$m={int(row.top_m)}$",
            ha="center",
            va="bottom",
            fontsize=7,
            color="#374151",
        )
    ax.set_title("(b) Final pair recovery", loc="left", pad=3)
    ax.set_ylabel("Interaction F1")
    ax.set_xlabel("Training samples $n$")
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.5, 1.0])
    clean_axis(ax, grid=True)

    fig.suptitle("$d=1000$, $c=0.10$: adaptive residual FA-SS-KAN", y=1.02, fontsize=9.5)
    save_figure(fig, out_base)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=RESULT_ROOT / "figures" / "residual_adaptive_boundary_c01_d1000",
    )
    args = parser.parse_args()
    summary = build_summary()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out.with_name(args.out.name + "_summary.csv"), index=False)
    print("[saved]", args.out.with_name(args.out.name + "_summary.csv"))
    plot(summary, args.out)


if __name__ == "__main__":
    main()
