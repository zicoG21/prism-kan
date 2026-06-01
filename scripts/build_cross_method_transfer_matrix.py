#!/usr/bin/env python3
"""Build cross-method evidence-transfer matrices from baseline detail CSVs."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_PATTERNS = [
    "results/revision/cross_method_transfer_baselines/**/cross_method_transfer_detail.csv",
]

TRANSFER_LINKS = [
    ("prediction->support", "prediction_success", "support_success_all_true"),
    ("prediction->endpoint", "prediction_success", "endpoint_success"),
    ("prediction->pair", "prediction_success", "pair_success_all_true_at_budget"),
    ("support->pair", "support_success_all_true", "pair_success_all_true_at_budget"),
    ("endpoint->pair", "endpoint_success", "pair_success_all_true_at_budget"),
    ("pair->support", "pair_success_all_true_at_budget", "support_success_all_true"),
]


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (math.nan, math.nan)
    phat = k / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n) / denom
    return max(0.0, center - half), min(1.0, center + half)


def find_inputs(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(Path(".").glob(pattern))
    return sorted(set(paths))


def source_priority(path: str) -> int:
    # Larger GL runs should supersede local pilots when the same method/function
    # seed is present in both places.
    if "/gl_" in path or "greatlakes" in path or "engin1" in path:
        return 0
    if "pilot" in path:
        return 2
    return 1


def read_details(paths: list[Path], dedupe: bool) -> pd.DataFrame:
    rows = []
    for path in paths:
        df = pd.read_csv(path)
        if df.empty:
            continue
        df["source_path"] = str(path)
        df["_source_priority"] = source_priority(str(path))
        rows.append(df)
    if not rows:
        raise SystemExit("No cross-method transfer detail CSVs found.")
    detail = pd.concat(rows, ignore_index=True, sort=False)
    if dedupe:
        key = [
            c
            for c in [
                "function",
                "method",
                "seed",
                "samples",
                "test_samples",
                "dimension",
                "noise",
                "nuisance_correlation",
                "n_correlated_proxies",
                "top_m",
                "pair_budget",
            ]
            if c in detail.columns
        ]
        detail = detail.sort_values(["_source_priority", "source_path"])
        detail = detail.drop_duplicates(subset=key, keep="first")
    return detail.drop(columns=["_source_priority"], errors="ignore")


def bool_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series([np.nan] * len(df), index=df.index)
    return pd.to_numeric(df[col], errors="coerce")


def build_transfer_long(detail: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        c
        for c in [
            "function",
            "method",
            "evidence_object",
            "samples",
            "dimension",
            "noise",
            "nuisance_correlation",
            "n_correlated_proxies",
            "top_m",
            "pair_budget",
        ]
        if c in detail.columns
    ]
    rows: list[dict[str, object]] = []
    for group_key, group in detail.groupby(group_cols, dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        base = dict(zip(group_cols, group_key))
        for link, source_col, target_col in TRANSFER_LINKS:
            source = bool_series(group, source_col)
            target = bool_series(group, target_col)
            valid = source.notna() & target.notna()
            n = int(valid.sum())
            if n == 0:
                continue
            s = source[valid].astype(int)
            t = target[valid].astype(int)
            fail = (s.eq(1) & t.eq(0)).astype(int)
            source_count = int(s.sum())
            target_count = int(t.sum())
            fail_count = int(fail.sum())
            uncond = fail_count / n
            cond = fail_count / source_count if source_count > 0 else math.nan
            uncond_lo, uncond_hi = wilson_interval(fail_count, n)
            cond_lo, cond_hi = wilson_interval(fail_count, source_count) if source_count > 0 else (math.nan, math.nan)
            row = {
                **base,
                "transfer_link": link,
                "source_event": source_col,
                "target_event": target_col,
                "num_runs": n,
                "source_success_count": source_count,
                "target_success_count": target_count,
                "transfer_failure_count": fail_count,
                "transfer_failure_rate": uncond,
                "transfer_failure_ci_low": uncond_lo,
                "transfer_failure_ci_high": uncond_hi,
                "conditional_failure_rate": cond,
                "conditional_failure_ci_low": cond_lo,
                "conditional_failure_ci_high": cond_hi,
            }
            rows.append(row)
    return pd.DataFrame(rows)


def build_method_summary(detail: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        c
        for c in [
            "function",
            "method",
            "evidence_object",
            "samples",
            "dimension",
            "noise",
            "nuisance_correlation",
            "n_correlated_proxies",
            "top_m",
            "pair_budget",
        ]
        if c in detail.columns
    ]
    metric_cols = [
        "test_mse",
        "prediction_success",
        "support_success_all_true",
        "endpoint_success",
        "pair_success_all_true_at_budget",
        "pair_success_any_true_at_budget",
        "true_pair_rank_worst",
        "true_pair_margin_min",
        "candidate_contains_all_true_pairs",
    ]
    metric_cols = [c for c in metric_cols if c in detail.columns]
    out = detail.groupby(group_cols, dropna=False)[metric_cols].agg(["mean", "std"]).reset_index()
    out.columns = ["_".join(str(v) for v in col if v != "").rstrip("_") for col in out.columns]
    counts = detail.groupby(group_cols, dropna=False).size().reset_index(name="num_runs")
    return out.merge(counts, on=group_cols, how="left")


def write_markdown(transfer: pd.DataFrame, method_summary: pd.DataFrame, path: Path, max_rows: int) -> None:
    lines = ["# Cross-Method Evidence-Transfer Matrix", ""]
    lines.append("## Method Success Summary")
    lines.append("")
    keep = [
        "function",
        "method",
        "num_runs",
        "test_mse_mean",
        "prediction_success_mean",
        "support_success_all_true_mean",
        "endpoint_success_mean",
        "pair_success_all_true_at_budget_mean",
        "true_pair_rank_worst_mean",
        "true_pair_margin_min_mean",
    ]
    show = method_summary[[c for c in keep if c in method_summary.columns]].head(max_rows).copy()
    lines.extend(markdown_table(show))
    lines.append("")
    lines.append("## Transfer Links")
    lines.append("")
    keep = [
        "function",
        "method",
        "transfer_link",
        "num_runs",
        "source_success_count",
        "target_success_count",
        "transfer_failure_count",
        "conditional_failure_rate",
        "transfer_failure_rate",
    ]
    show = transfer[[c for c in keep if c in transfer.columns]].head(max_rows).copy()
    lines.extend(markdown_table(show))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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


def label_row(row: pd.Series) -> str:
    parts = [str(row["function"])]
    if "dimension" in row and pd.notna(row["dimension"]):
        parts.append(f"d={int(row['dimension'])}")
    if "samples" in row and pd.notna(row["samples"]):
        parts.append(f"n={int(row['samples'])}")
    parts.append(str(row["method"]))
    return " | ".join(parts)


def plot_heatmap(transfer: pd.DataFrame, path: Path, value_col: str) -> None:
    if transfer.empty:
        return
    df = transfer.copy()
    df["row_label"] = df.apply(label_row, axis=1)
    link_order = [link for link, _, _ in TRANSFER_LINKS]
    rows = sorted(df["row_label"].unique())
    pivot = df.pivot_table(index="row_label", columns="transfer_link", values=value_col, aggfunc="mean")
    pivot = pivot.reindex(index=rows, columns=link_order)

    data = pivot.to_numpy(dtype=float)
    masked = np.ma.masked_invalid(data)
    fig_h = max(3.8, 0.36 * len(rows) + 1.6)
    fig_w = max(8.0, 1.35 * len(link_order) + 2.8)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    cmap = plt.get_cmap("magma_r").copy()
    cmap.set_bad(color="#f1f1f1")
    im = ax.imshow(masked, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")

    ax.set_xticks(np.arange(len(link_order)))
    ax.set_xticklabels(link_order, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(rows, fontsize=8)
    ax.set_title("Cross-method evidence-transfer failure", fontsize=12, pad=12)
    ax.set_xlabel("Transfer link: source evidence -> target claim")
    ax.set_ylabel("Formula | method")

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if np.isfinite(data[i, j]):
                ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=7, color="#202020")
            else:
                ax.text(j, i, "NA", ha="center", va="center", fontsize=6, color="#777777")

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Conditional failure rate" if value_col == "conditional_failure_rate" else "Failure rate")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-prefix", default="local_notes/generated/cross_method_transfer_matrix_20260601")
    parser.add_argument("--max-md-rows", type=int, default=80)
    parser.add_argument("--no-dedupe", action="store_true")
    parser.add_argument("patterns", nargs="*", default=DEFAULT_PATTERNS)
    args = parser.parse_args()

    paths = find_inputs(args.patterns)
    detail = read_details(paths, dedupe=not args.no_dedupe)
    transfer = build_transfer_long(detail)
    method_summary = build_method_summary(detail)

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    detail_path = prefix.with_name(prefix.name + "_detail.csv")
    transfer_path = prefix.with_name(prefix.name + "_transfer_long.csv")
    method_path = prefix.with_name(prefix.name + "_method_summary.csv")
    md_path = prefix.with_suffix(".md")
    fig_path = prefix.with_suffix(".pdf")

    detail.to_csv(detail_path, index=False)
    transfer.to_csv(transfer_path, index=False)
    method_summary.to_csv(method_path, index=False)
    write_markdown(transfer, method_summary, md_path, args.max_md_rows)
    plot_heatmap(transfer, fig_path, value_col="conditional_failure_rate")

    print(f"Inputs: {len(paths)} files")
    print(f"Wrote {detail_path}")
    print(f"Wrote {method_path}")
    print(f"Wrote {transfer_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {fig_path}")


if __name__ == "__main__":
    main()
