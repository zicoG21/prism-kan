#!/usr/bin/env python3
"""Summarize EPIM PairVerify runs into paper-ready tables and a heatmap."""

from __future__ import annotations

import argparse
import glob
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_PATTERNS = ["results/revision/epim_pairverify/**/epim_pairverify_detail.csv"]

COUNT_METRICS = [
    ("epim_proposal", "epim_proposal_contains_true_pair"),
    ("epim_endpoint", "epim_endpoint_contains_true_pair"),
    ("verified_top", "practical_verified_top_is_true_pair"),
    ("verified_margin", "practical_verified_true_beats_candidate_false"),
]

MEAN_METRICS = [
    "test_mse",
    "epim_true_pair_rank",
    "verified_true_pair_rank",
    "verified_true_minus_max_candidate_false",
    "verified_true_minus_max_random_control",
]


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return math.nan, math.nan
    phat = k / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n) / denom
    return max(0.0, center - half), min(1.0, center + half)


def find_inputs(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        if Path(pattern).is_absolute():
            paths.extend(Path(p) for p in glob.glob(pattern, recursive=True))
        else:
            paths.extend(Path(".").glob(pattern))
    return sorted(set(paths))


def read_details(paths: list[Path], dedupe: bool) -> pd.DataFrame:
    rows = []
    for path in paths:
        df = pd.read_csv(path)
        if df.empty:
            continue
        df["source_path"] = str(path)
        rows.append(df)
    if not rows:
        raise SystemExit("No EPIM PairVerify detail CSVs found.")
    detail = pd.concat(rows, ignore_index=True, sort=False)
    if dedupe:
        key = [
            c
            for c in [
                "function",
                "seed",
                "samples",
                "test_samples",
                "dimension",
                "noise",
                "nuisance_correlation",
                "n_correlated_proxies",
                "update_grid",
                "width_hidden",
                "proposal_q",
                "random_controls",
            ]
            if c in detail.columns
        ]
        detail = detail.sort_values("source_path").drop_duplicates(subset=key, keep="last")
    return detail


def build_summary(detail: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        c
        for c in [
            "function",
            "samples",
            "dimension",
            "noise",
            "update_grid",
            "width_hidden",
            "proposal_q",
            "random_controls",
        ]
        if c in detail.columns
    ]
    rows: list[dict[str, object]] = []
    for group_key, group in detail.groupby(group_cols, dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        row: dict[str, object] = dict(zip(group_cols, group_key))
        if "status" in group.columns:
            ok = group[group["status"].astype(str).eq("ok")].copy()
        else:
            ok = group.copy()
        row["num_runs"] = int(len(ok))
        for label, col in COUNT_METRICS:
            vals = pd.to_numeric(ok[col], errors="coerce") if col in ok.columns else pd.Series(dtype=float)
            valid = vals.dropna().astype(int)
            n = int(len(valid))
            k = int(valid.sum()) if n else 0
            lo, hi = wilson_interval(k, n)
            row[f"{label}_count"] = k
            row[f"{label}_n"] = n
            row[f"{label}_rate"] = k / n if n else math.nan
            row[f"{label}_ci_low"] = lo
            row[f"{label}_ci_high"] = hi
            row[f"{label}_display"] = f"{k}/{n}" if n else ""
        for col in MEAN_METRICS:
            if col in ok.columns:
                vals = pd.to_numeric(ok[col], errors="coerce")
                row[f"{col}_mean"] = float(vals.mean()) if vals.notna().any() else math.nan
                row[f"{col}_std"] = float(vals.std()) if vals.notna().sum() > 1 else math.nan
        rows.append(row)
    return pd.DataFrame(rows)


def row_label(row: pd.Series) -> str:
    parts = [str(row.get("function", ""))]
    if pd.notna(row.get("dimension", np.nan)):
        parts.append(f"d={int(row['dimension'])}")
    if pd.notna(row.get("samples", np.nan)):
        parts.append(f"n={int(row['samples'])}")
    if pd.notna(row.get("noise", np.nan)):
        noise = float(row["noise"])
        if abs(noise) > 1e-12:
            parts.append(f"noise={noise:g}")
    if int(row.get("update_grid", 0)) == 1:
        parts.append("grid")
    if pd.notna(row.get("proposal_q", np.nan)):
        parts.append(f"q={int(row['proposal_q'])}")
    return " | ".join(parts)


def markdown_table(df: pd.DataFrame) -> list[str]:
    show = df.copy()
    for col in show.columns:
        if pd.api.types.is_float_dtype(show[col]):
            show[col] = show[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.3f}")
    widths = {col: max(len(str(col)), *(len(str(v)) for v in show[col])) for col in show.columns}
    lines = []
    lines.append("| " + " | ".join(str(col).ljust(widths[col]) for col in show.columns) + " |")
    lines.append("| " + " | ".join("-" * widths[col] for col in show.columns) + " |")
    for _, row in show.iterrows():
        lines.append("| " + " | ".join(str(row[col]).ljust(widths[col]) for col in show.columns) + " |")
    return lines


def write_markdown(summary: pd.DataFrame, path: Path, max_rows: int) -> None:
    lines = [
        "# EPIM PairVerify Summary",
        "",
        "Rates are Wilson-interval-ready descriptive summaries. Practical verification",
        "counts require the true pair to be proposed by EPIM before the ANOVA verifier",
        "can count it as a success.",
        "",
    ]
    show = summary.copy()
    show["setting"] = show.apply(row_label, axis=1)
    keep = [
        "setting",
        "num_runs",
        "epim_proposal_display",
        "epim_proposal_rate",
        "epim_endpoint_display",
        "epim_endpoint_rate",
        "verified_top_display",
        "verified_top_rate",
        "verified_margin_display",
        "verified_margin_rate",
        "verified_true_minus_max_candidate_false_mean",
        "verified_true_minus_max_random_control_mean",
    ]
    show = show[[c for c in keep if c in show.columns]].head(max_rows)
    lines.extend(markdown_table(show))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_heatmap(summary: pd.DataFrame, path: Path) -> None:
    if summary.empty:
        return
    metrics = [
        ("EPIM proposes pair", "epim_proposal_rate"),
        ("Endpoint mass contains endpoints", "epim_endpoint_rate"),
        ("Verified top pair", "verified_top_rate"),
        ("Verified beats candidate false", "verified_margin_rate"),
    ]
    df = summary.copy()
    df["setting"] = df.apply(row_label, axis=1)
    df = df.sort_values(["function", "dimension", "samples", "noise", "update_grid"], kind="stable")
    data = df[[col for _, col in metrics]].to_numpy(dtype=float)
    labels = [label for label, _ in metrics]

    height = max(2.5, 0.34 * len(df) + 1.4)
    fig, ax = plt.subplots(figsize=(7.2, height))
    im = ax.imshow(data, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=28, ha="right")
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["setting"].tolist())
    ax.set_title("EPIM PairVerify: proposal versus function-level verification")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color="white" if val < 0.55 else "black")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("success rate")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="*", default=DEFAULT_PATTERNS)
    parser.add_argument("--out-prefix", type=Path, default=Path("local_notes/generated/epim_pairverify_summary_20260601"))
    parser.add_argument("--no-dedupe", action="store_true")
    parser.add_argument("--max-markdown-rows", type=int, default=80)
    args = parser.parse_args()

    paths = find_inputs(args.inputs)
    detail = read_details(paths, dedupe=not args.no_dedupe)
    summary = build_summary(detail)
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)

    detail_path = args.out_prefix.with_name(args.out_prefix.name + "_detail.csv")
    summary_path = args.out_prefix.with_name(args.out_prefix.name + ".csv")
    md_path = args.out_prefix.with_name(args.out_prefix.name + ".md")
    fig_path = args.out_prefix.with_name(args.out_prefix.name + ".pdf")
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    write_markdown(summary, md_path, max_rows=args.max_markdown_rows)
    plot_heatmap(summary, fig_path)

    print(f"Wrote detail:  {detail_path}")
    print(f"Wrote summary: {summary_path}")
    print(f"Wrote notes:   {md_path}")
    print(f"Wrote figure:  {fig_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
