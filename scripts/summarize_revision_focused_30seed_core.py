from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def markdown_table(df: pd.DataFrame) -> str:
    """Small dependency-free markdown table writer.

    Pandas' ``to_markdown`` requires the optional ``tabulate`` package, which is
    not installed in the paper environment.  Keep this summary script
    self-contained so reviewer quick checks do not fail on optional deps.
    """

    if df.empty:
        return "_No rows._"
    text_df = df.copy()
    for col in text_df.columns:
        text_df[col] = text_df[col].map(lambda x: "" if pd.isna(x) else str(x))
    cols = list(text_df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in text_df.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in cols) + " |")
    return "\n".join(lines)


def setting_from_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).parts[0]
    except Exception:
        return path.parent.name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="results/revision/focused_30seed_core")
    args = parser.parse_args()
    root = Path(args.root)
    out_dir = root / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for path in sorted(root.glob("*/support_sensitivity_summary.csv")):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if df.empty:
            continue
        df.insert(0, "setting", setting_from_path(path, root))
        rows.append(df)

    if not rows:
        (out_dir / "summary.md").write_text("No completed focused 30-seed rows found.\n")
        return

    all_df = pd.concat(rows, ignore_index=True)
    all_df.to_csv(out_dir / "focused_30seed_all.csv", index=False)

    focus = all_df[
        (all_df["method"].astype(str) == "feature_edge_hybrid")
        & (pd.to_numeric(all_df["top_m"], errors="coerce") == 4)
    ].copy()
    keep = [
        "setting",
        "function",
        "samples",
        "dimension",
        "noise",
        "width_hidden",
        "grid",
        "probe_steps",
        "screen_contains_all_interaction_endpoints_mean",
        "screen_contains_true_interactions_mean",
        "true_endpoint_rank_worst_mean",
        "endpoint_minus_max_nuisance_mean",
        "num_support_evals",
    ]
    keep = [c for c in keep if c in focus.columns]
    focus = focus[keep].sort_values(["function", "dimension", "setting", "samples"])
    focus.to_csv(out_dir / "focused_30seed_main_rows.csv", index=False)

    md = ["# Focused 30-seed core summary", ""]
    md.append("Feature-edge hybrid, top-m=4:")
    md.append("")
    md.append(markdown_table(focus))
    md.append("")
    md.append("Interpretation: use these rows to replace or validate the 8/10/12-seed paper-facing boundary and margin diagnostics.")
    (out_dir / "summary.md").write_text("\n".join(md))


if __name__ == "__main__":
    main()
