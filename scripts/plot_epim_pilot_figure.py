#!/usr/bin/env python3
"""Plot pilot evidence for the EPIM PairVerify novelty route."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from experiments.paper_plot_style import OKABE_ITO, clean_axis, configure_paper_plots, save_figure
except Exception:  # pragma: no cover
    from paper_plot_style import OKABE_ITO, clean_axis, configure_paper_plots, save_figure


OUT_BASE = ROOT / "local_notes" / "generated" / "epim_pilot_comparison_20260601"
MANUSCRIPT_OUT_BASE = ROOT / "manuscripts" / "workshop_case_study" / "figures" / "epim_pilot_comparison"

SOURCE_LABELS = {
    "clean_width16_family_search": "clean",
    "gridupdate_width16_family_search": "grid update",
    "noise010_width16_family_search": "noise 0.10",
}

METHOD_LABELS = {
    "feature_stability_var": "Feature",
    "feature_edge_hybrid": "KAN-FE",
    "edge_endpoint_mass": "EPIM endpoint",
    "edge_pair_hybrid": "EPIM pair",
}

METHOD_ORDER = ["feature_stability_var", "feature_edge_hybrid", "edge_endpoint_mass", "edge_pair_hybrid"]
METHOD_COLORS = {
    "feature_stability_var": OKABE_ITO["gray"],
    "feature_edge_hybrid": OKABE_ITO["blue"],
    "edge_endpoint_mass": OKABE_ITO["green"],
    "edge_pair_hybrid": OKABE_ITO["purple"],
}


def load_rows() -> pd.DataFrame:
    rows = []
    for path in sorted((ROOT / "results" / "revision" / "greatlakes_innovation_probe").glob("*/innovation_summary.csv")):
        source = path.parent.name
        if source not in SOURCE_LABELS:
            continue
        df = pd.read_csv(path)
        df["source"] = source
        rows.append(df)
    if not rows:
        raise SystemExit("No Great Lakes innovation summaries found.")
    out = pd.concat(rows, ignore_index=True, sort=False)
    out = out[out["method"].isin(METHOD_ORDER)].copy()
    for col in [
        "num_runs",
        "screen_contains_all_interaction_endpoints_mean",
        "interaction_f1_mean",
        "true_interaction_best_rank_mean",
        "true_interaction_mean_score_margin_mean",
        "test_mse_mean",
    ]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def weighted_mean(group: pd.DataFrame, value_col: str) -> float:
    values = pd.to_numeric(group[value_col], errors="coerce")
    weights = pd.to_numeric(group["num_runs"], errors="coerce").fillna(1.0)
    mask = values.notna() & weights.notna() & (weights > 0)
    if not mask.any():
        return np.nan
    return float(np.average(values[mask], weights=weights[mask]))


def aggregate_by_regime_method(df: pd.DataFrame) -> pd.DataFrame:
    # Focus on the core c=.25 pressure test; it is the row family where the
    # reviewer objections and PairVerify motivation are sharpest.
    core = df[df["function"].eq("core_interaction_c025")].copy()
    rows = []
    for (source, method), group in core.groupby(["source", "method"], dropna=False):
        rows.append(
            {
                "source": source,
                "regime": SOURCE_LABELS[source],
                "method": method,
                "method_label": METHOD_LABELS[method],
                "runs": int(pd.to_numeric(group["num_runs"], errors="coerce").fillna(0).sum()),
                "endpoint_rate": weighted_mean(group, "screen_contains_all_interaction_endpoints_mean"),
                "pair_f1": weighted_mean(group, "interaction_f1_mean"),
                "rank": weighted_mean(group, "true_interaction_best_rank_mean"),
                "margin": weighted_mean(group, "true_interaction_mean_score_margin_mean"),
                "mse": weighted_mean(group, "test_mse_mean"),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    configure_paper_plots(usetex=False)
    raw = load_rows()
    summary = aggregate_by_regime_method(raw)
    epim_summary = summary[summary["method"].eq("edge_pair_hybrid")].copy()
    OUT_BASE.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT_BASE.with_name(OUT_BASE.name + "_summary.csv"), index=False)
    epim_summary.to_csv(OUT_BASE.with_name(OUT_BASE.name + "_epim_rows.csv"), index=False)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.15, 2.35),
        gridspec_kw={"width_ratios": [1.25, 1.25, 1.0]},
    )

    regimes = ["clean", "grid update", "noise 0.10"]
    x = np.arange(len(regimes))
    width = 0.18
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(METHOD_ORDER))

    # Panel A: endpoint proposal.
    ax = axes[0]
    for offset, method in zip(offsets, METHOD_ORDER):
        vals = []
        for regime in regimes:
            row = summary[(summary["regime"].eq(regime)) & (summary["method"].eq(method))]
            vals.append(float(row["endpoint_rate"].iloc[0]) if len(row) else np.nan)
        ax.bar(
            x + offset,
            vals,
            width,
            label=METHOD_LABELS[method],
            color=METHOD_COLORS[method],
            edgecolor="white",
            linewidth=0.45,
        )
    ax.set_title("(a) Proposal: endpoints", loc="left")
    ax.set_ylabel("endpoint success")
    ax.set_xticks(x)
    ax.set_xticklabels(regimes, rotation=18, ha="right")
    ax.set_ylim(0, 1.10)
    clean_axis(ax, grid=True)

    # Panel B: pair-level evidence after selected-support refit.
    ax = axes[1]
    for offset, method in zip(offsets, METHOD_ORDER):
        vals = []
        for regime in regimes:
            row = summary[(summary["regime"].eq(regime)) & (summary["method"].eq(method))]
            vals.append(float(row["pair_f1"].iloc[0]) if len(row) else np.nan)
        ax.bar(
            x + offset,
            vals,
            width,
            label=METHOD_LABELS[method],
            color=METHOD_COLORS[method],
            edgecolor="white",
            linewidth=0.45,
        )
    ax.set_title("(b) Refit pair evidence", loc="left")
    ax.set_ylabel("pair F1")
    ax.set_xticks(x)
    ax.set_xticklabels(regimes, rotation=18, ha="right")
    ax.set_ylim(0, 1.10)
    clean_axis(ax, grid=True)

    # Panel C: EPIM pair proposal is informative but not certification.
    ax = axes[2]
    regime_colors = {"clean": OKABE_ITO["green"], "grid update": OKABE_ITO["orange"], "noise 0.10": OKABE_ITO["vermillion"]}
    for _, row in epim_summary.iterrows():
        regime = str(row["regime"])
        ax.scatter(
            float(row["endpoint_rate"]),
            float(row["pair_f1"]),
            s=58,
            c=regime_colors[regime],
            edgecolors="white",
            linewidths=0.6,
            zorder=3,
        )
        ax.text(
            min(float(row["endpoint_rate"]) + 0.025, 0.98),
            float(row["pair_f1"]),
            regime,
            fontsize=6.8,
            color="#374151",
            va="center",
        )
    ax.plot([0, 1], [0, 1], color="#B6BDC8", lw=0.8, ls=(0, (2, 2)), zorder=1)
    ax.set_title("(c) EPIM proposal gap", loc="left")
    ax.set_xlabel("endpoint proposal")
    ax.set_ylabel("pair F1")
    ax.set_xlim(-0.05, 1.08)
    ax.set_ylim(-0.05, 1.08)
    clean_axis(ax, grid=True)
    ax.text(
        0.04,
        0.10,
        "proposal success\nis not certification",
        transform=ax.transAxes,
        fontsize=6.6,
        color="#374151",
    )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=4,
        handlelength=1.6,
        columnspacing=1.2,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93), w_pad=1.05)

    save_figure(fig, OUT_BASE)
    save_figure(fig, MANUSCRIPT_OUT_BASE)
    plt.close(fig)


if __name__ == "__main__":
    main()
