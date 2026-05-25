import argparse
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


C_MAP = {
    "core_interaction_c01": 0.10,
    "core_interaction_c025": 0.25,
    "core_interaction_c05": 0.50,
    "core_interaction_c1": 1.00,
}

C_LABEL = {
    0.10: "0.10",
    0.25: "0.25",
    0.50: "0.50",
    1.00: "1.00",
}

SCREEN_LABEL = {
    "raw": "Raw KAN",
    "rf": "RF-screened KAN",
    "oracle_support": "Oracle-support KAN",
    "random": "Random support",
    "exclude_interaction": "Exclude interaction",
}

METHOD_ORDER_MAIN = ["raw", "rf", "oracle_support"]


def set_paper_style():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.titlesize": 14,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 400,
        "axes.linewidth": 0.8,
    })


def parse_tag(path):
    name = Path(path).name
    m = re.search(r"(core_interaction_c(?:01|025|05|1))_n(\d+)_d(\d+)", name)
    if not m:
        return None
    return m.group(1), int(m.group(2)), int(m.group(3))


def load_summaries(pattern):
    rows = []
    for p in sorted(glob.glob(pattern)):
        parsed = parse_tag(p)
        if parsed is None:
            print(f"[skip] cannot parse filename: {p}")
            continue

        fn, n, d = parsed
        df = pd.read_csv(p)
        df = df.copy()

        if "function" not in df.columns:
            df["function"] = fn

        df["samples"] = n
        df["dimension"] = d
        df["interaction_strength"] = C_MAP.get(fn, np.nan)

        for base in [
            "test_mse",
            "variable_f1",
            "explain_interaction_endpoint_recall",
            "interaction_f1",
            "true_interaction_mean_score_margin",
            "true_interaction_beats_all_false",
        ]:
            mean_col = f"{base}_mean"
            if mean_col in df.columns:
                df[base] = df[mean_col]

        rows.append(df)

    if not rows:
        raise FileNotFoundError(f"No files matched pattern: {pattern}")

    out = pd.concat(rows, ignore_index=True)
    needed = ["screen_mode", "samples", "dimension", "interaction_strength",
              "test_mse", "variable_f1", "interaction_f1"]
    missing = [c for c in needed if c not in out.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    return out


def make_matrix(df, metric, mode, dim):
    sub = df[(df["screen_mode"] == mode) & (df["dimension"] == dim)].copy()

    cs = [1.00, 0.50, 0.25, 0.10]
    ns = [128, 256, 512, 1024]

    mat = np.full((len(cs), len(ns)), np.nan)
    for i, c in enumerate(cs):
        for j, n in enumerate(ns):
            g = sub[(np.isclose(sub["interaction_strength"], c)) & (sub["samples"] == n)]
            if not g.empty:
                mat[i, j] = float(g[metric].mean())

    return mat, cs, ns


def annotate_heatmap(ax, mat, metric, text_color_threshold=None):
    finite = mat[np.isfinite(mat)]
    if finite.size == 0:
        return

    if text_color_threshold is None:
        text_color_threshold = np.nanmedian(finite)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            if not np.isfinite(val):
                txt = "NA"
                color = "black"
            else:
                if metric == "test_mse":
                    txt = f"{val:.1e}" if val < 1e-2 else f"{val:.2g}"
                else:
                    txt = f"{val:.2f}"
                color = "white" if val < text_color_threshold else "black"

            ax.text(j, i, txt, ha="center", va="center", fontsize=8.5, color=color)


def draw_heatmap_panel(
    ax,
    df,
    metric,
    mode,
    dim,
    *,
    vmin,
    vmax,
    cmap,
    log_mse=False,
    title=None,
    show_ylabel=False,
):
    mat, cs, ns = make_matrix(df, metric, mode, dim)

    plot_mat = mat.copy()
    if log_mse:
        plot_mat = np.log10(np.maximum(plot_mat, 1e-8))

    im = ax.imshow(plot_mat, aspect="auto", vmin=vmin, vmax=vmax, cmap=cmap)

    ax.set_xticks(np.arange(len(ns)))
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_yticks(np.arange(len(cs)))
    ax.set_yticklabels([C_LABEL[c] for c in cs])

    ax.set_xlabel("Sample size $n$")
    if show_ylabel:
        ax.set_ylabel("Interaction strength $c$")
    else:
        ax.set_ylabel("")

    if title is None:
        title = SCREEN_LABEL.get(mode, mode)
    ax.set_title(title, pad=8)

    # thin white grid between cells
    ax.set_xticks(np.arange(-.5, len(ns), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(cs), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    # annotation uses original metric values, not log values
    if metric == "test_mse":
        # lower MSE cells are darker in reversed colormap, so threshold by log value
        log_mat = np.log10(np.maximum(mat, 1e-8))
        threshold = np.nanmedian(log_mat)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat[i, j]
                if not np.isfinite(val):
                    txt = "NA"
                    color = "black"
                else:
                    txt = f"{val:.1e}" if val < 1e-2 else f"{val:.2g}"
                    color = "white" if np.log10(max(val, 1e-8)) < threshold else "black"
                ax.text(j, i, txt, ha="center", va="center", fontsize=8.3, color=color)
    else:
        # high F1 is dark in viridis if using vmin=0, vmax=1? actually high is yellow.
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat[i, j]
                if not np.isfinite(val):
                    txt = "NA"
                    color = "black"
                else:
                    txt = f"{val:.2f}"
                    color = "black" if val > 0.55 else "white"
                ax.text(j, i, txt, ha="center", va="center", fontsize=8.5, color=color)

    return im


def save_both(fig, out_base):
    out_base = Path(out_base)
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight", dpi=400)
    print("[saved]", out_base.with_suffix(".pdf"))
    print("[saved]", out_base.with_suffix(".png"))


def figure1_formula_phase(df, dim, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.35), constrained_layout=True)

    im = None
    for idx, mode in enumerate(METHOD_ORDER_MAIN):
        im = draw_heatmap_panel(
            axes[idx],
            df,
            metric="interaction_f1",
            mode=mode,
            dim=dim,
            vmin=0.0,
            vmax=1.0,
            cmap="viridis",
            title=SCREEN_LABEL[mode],
            show_ylabel=(idx == 0),
        )

    cbar = fig.colorbar(im, ax=axes, shrink=0.88, pad=0.018)
    cbar.set_label("Interaction F1")

    fig.suptitle(
        f"Formula-fidelity recovery boundary under nuisance dimensions ($d={dim}$)",
        y=1.04,
    )

    save_both(fig, out_dir / f"fig1_formula_phase_d{dim}")
    plt.close(fig)


def figure2_prediction_vs_formula_raw(df, dim, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.35), constrained_layout=True)

    im0 = draw_heatmap_panel(
        axes[0],
        df,
        metric="test_mse",
        mode="raw",
        dim=dim,
        vmin=-4.0,
        vmax=0.4,
        cmap="magma_r",
        log_mse=True,
        title="Prediction error",
        show_ylabel=True,
    )

    im1 = draw_heatmap_panel(
        axes[1],
        df,
        metric="interaction_f1",
        mode="raw",
        dim=dim,
        vmin=0.0,
        vmax=1.0,
        cmap="viridis",
        title="Formula interaction recovery",
        show_ylabel=False,
    )

    cbar0 = fig.colorbar(im0, ax=axes[0], shrink=0.88, pad=0.02)
    cbar0.set_label("$\\log_{10}$(test MSE)")

    cbar1 = fig.colorbar(im1, ax=axes[1], shrink=0.88, pad=0.02)
    cbar1.set_label("Interaction F1")

    fig.suptitle(
        f"Prediction improves before formula recovery: Raw KAN ($d={dim}$)",
        y=1.04,
    )

    save_both(fig, out_dir / f"fig2_prediction_vs_formula_raw_d{dim}")
    plt.close(fig)


def figure3_variable_vs_interaction_raw(df, dim, out_dir):
    metrics = [("variable_f1", "Variable recovery")]
    if "explain_interaction_endpoint_recall" in df.columns:
        metrics.append(("explain_interaction_endpoint_recall", "Endpoint retention"))
    metrics.append(("interaction_f1", "Interaction recovery"))

    fig_width = 10.8 if len(metrics) == 3 else 7.8
    fig, axes = plt.subplots(1, len(metrics), figsize=(fig_width, 3.35), constrained_layout=True)
    axes = np.atleast_1d(axes)

    im = None
    for idx, (metric, title) in enumerate(metrics):
        im = draw_heatmap_panel(
            axes[idx],
            df,
            metric=metric,
            mode="raw",
            dim=dim,
            vmin=0.0,
            vmax=1.0,
            cmap="viridis",
            title=title,
            show_ylabel=(idx == 0),
        )

    cbar = fig.colorbar(im, ax=axes, shrink=0.88, pad=0.018)
    cbar.set_label("Recovery score")

    fig.suptitle(
        f"Formula fidelity requires variables, endpoints, and interactions: Raw KAN ($d={dim}$)",
        y=1.04,
    )

    save_both(fig, out_dir / f"fig3_variable_vs_interaction_raw_d{dim}")
    plt.close(fig)


def figure4_controls(df, dim, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.35), constrained_layout=True)

    im0 = draw_heatmap_panel(
        axes[0],
        df,
        metric="interaction_f1",
        mode="random",
        dim=dim,
        vmin=0.0,
        vmax=1.0,
        cmap="viridis",
        title="Random support",
        show_ylabel=True,
    )

    im1 = draw_heatmap_panel(
        axes[1],
        df,
        metric="interaction_f1",
        mode="exclude_interaction",
        dim=dim,
        vmin=0.0,
        vmax=1.0,
        cmap="viridis",
        title="Exclude interaction",
        show_ylabel=False,
    )

    cbar = fig.colorbar(im1, ax=axes, shrink=0.88, pad=0.018)
    cbar.set_label("Interaction F1")

    fig.suptitle(
        f"Negative controls do not recover the interaction ($d={dim}$)",
        y=1.04,
    )

    save_both(fig, out_dir / f"fig4_negative_controls_d{dim}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary_glob",
        default="results/hard_regime/summaries/*_summary.csv",
    )
    parser.add_argument(
        "--out_dir",
        default="results/hard_regime/paper_figures",
    )
    parser.add_argument(
        "--dims",
        nargs="+",
        type=int,
        default=[100],
    )
    args = parser.parse_args()

    set_paper_style()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_summaries(args.summary_glob)
    df.to_csv(out_dir / "paper_figure_source_data.csv", index=False)

    print("[loaded]", len(df), "rows")
    print("[dims]", sorted(df["dimension"].unique()))

    for dim in args.dims:
        print(f"\nDrawing figures for d={dim}")
        figure1_formula_phase(df, dim, out_dir)
        figure2_prediction_vs_formula_raw(df, dim, out_dir)
        figure3_variable_vs_interaction_raw(df, dim, out_dir)
        figure4_controls(df, dim, out_dir)

    print("\nDone. Paper-ready figures are in:", out_dir)


if __name__ == "__main__":
    main()
