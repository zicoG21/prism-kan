#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def read_csvs(root: Path, pattern: str) -> list[tuple[Path, pd.DataFrame]]:
    out: list[tuple[Path, pd.DataFrame]] = []
    for path in sorted(root.glob(pattern)):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if not df.empty:
            out.append((path, df))
    return out


def rel_label(root: Path, path: Path) -> str:
    try:
        return str(path.parent.relative_to(root))
    except Exception:
        return str(path.parent)


def collect_kan(root: Path) -> pd.DataFrame:
    rows = []
    for path, df in read_csvs(root, "kan_sensitivity/**/support_sensitivity_summary.csv"):
        label = rel_label(root, path)
        df = df.copy()
        df.insert(0, "run_label", label)
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    focus = out[
        (out["method"].astype(str) == "feature_edge_hybrid")
        & (pd.to_numeric(out["top_m"], errors="coerce").isin([4, 6, 20]))
    ].copy()
    keep = [
        "run_label",
        "function",
        "samples",
        "dimension",
        "noise",
        "nuisance_correlation",
        "n_correlated_proxies",
        "width_hidden",
        "grid",
        "lamb",
        "probe_steps",
        "top_m",
        "screen_contains_all_interaction_endpoints_mean",
        "screen_contains_true_interactions_mean",
        "true_endpoint_rank_worst_mean",
        "endpoint_minus_max_nuisance_mean",
        "num_support_evals",
    ]
    return focus[[c for c in keep if c in focus.columns]]


def collect_semisynth(root: Path) -> pd.DataFrame:
    rows = []
    for path, df in read_csvs(root, "semisynthetic/**/semisynthetic_covariate_audit_summary.csv"):
        label = rel_label(root, path)
        df = df[df["method"].astype(str).eq("feature_edge_hybrid")].copy()
        if df.empty:
            continue
        df.insert(0, "run_label", label)
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    keep = [
        "run_label",
        "dataset",
        "c",
        "samples",
        "dimension",
        "noise",
        "endpoint_successes",
        "support_pair_successes",
        "residual_top1_successes",
        "probe_test_mse_mean_mean",
        "num_outer_seeds",
    ]
    return out[[c for c in keep if c in out.columns]]


def collect_lasso(root: Path) -> pd.DataFrame:
    rows = []
    for path, df in read_csvs(root, "baselines/lasso_*/pair_feature_lasso_summary.csv"):
        label = rel_label(root, path)
        df = df.copy()
        df.insert(0, "run_label", label)
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    keep = [
        "run_label",
        "function",
        "samples",
        "dimension",
        "noise",
        "nuisance_correlation",
        "n_correlated_proxies",
        "top1_pair_accuracy_mean",
        "pair_retained_at_m_mean",
        "endpoint_recall_at_m_mean",
        "num_runs",
    ]
    return out[[c for c in keep if c in out.columns]]


def collect_hsic(root: Path) -> pd.DataFrame:
    rows = []
    for path, df in read_csvs(root, "baselines/hsic_*/residual_rff_hsic_pair_screen_summary.csv"):
        label = rel_label(root, path)
        df = df.copy()
        df.insert(0, "run_label", label)
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    keep = [
        "run_label",
        "function",
        "samples",
        "dimension",
        "noise",
        "nuisance_correlation",
        "n_correlated_proxies",
        "top1_pair_accuracy_mean",
        "pair_retained_at_top_pairs_mean",
        "true_interaction_rank_worst_mean",
        "num_runs",
        "top1_successes",
    ]
    return out[[c for c in keep if c in out.columns]]


def md_table(df: pd.DataFrame, max_rows: int = 30) -> str:
    if df.empty:
        return "_No rows found._"
    view = df.head(max_rows).copy()
    headers = [str(c) for c in view.columns]
    rows = [[str(v) for v in row] for row in view.to_numpy()]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    if len(df) > max_rows:
        lines.append(f"\n_Showing first {max_rows} of {len(df)} rows._")
    return "\n".join(lines)


def write_outputs(root: Path) -> None:
    out_dir = root / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    kan = collect_kan(root)
    semisynth = collect_semisynth(root)
    lasso = collect_lasso(root)
    hsic = collect_hsic(root)

    if not kan.empty:
        kan.to_csv(out_dir / "kan_boundary_focus.csv", index=False)
    if not semisynth.empty:
        semisynth.to_csv(out_dir / "semisynthetic_focus.csv", index=False)
    if not lasso.empty:
        lasso.to_csv(out_dir / "lasso_focus.csv", index=False)
    if not hsic.empty:
        hsic.to_csv(out_dir / "hsic_focus.csv", index=False)

    lines = [
        "# Boundary Overnight Summary",
        "",
        "## pyKAN readout boundary focus",
        md_table(kan.sort_values(["samples", "run_label", "top_m"]) if not kan.empty else kan, 40),
        "",
        "## Semi-synthetic covariate focus",
        md_table(semisynth.sort_values(["run_label", "dataset", "c", "samples"]) if not semisynth.empty else semisynth, 40),
        "",
        "## Pair-feature Lasso focus",
        md_table(lasso.sort_values(["run_label", "samples"]) if not lasso.empty else lasso, 40),
        "",
        "## Residual RFF-HSIC focus",
        md_table(hsic.sort_values(["run_label", "samples"]) if not hsic.empty else hsic, 40),
        "",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print((out_dir / "summary.md").read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results/revision/boundary_overnight_12h"))
    args = parser.parse_args()
    write_outputs(args.root)


if __name__ == "__main__":
    main()
