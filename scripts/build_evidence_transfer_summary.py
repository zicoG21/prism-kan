#!/usr/bin/env python3
"""Summarize seed-aligned stage-record CSVs as evidence-transfer failures."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd


DEFAULT_PATTERNS = [
    "results/revision/seed_aligned_stage_records*/**/seed_aligned_stage_records_compact.csv",
    "results/revision/local_gpu_highvalue_stage_queue/**/seed_aligned_stage_records_compact.csv",
    "results/revision/local_gpu_stage_record_followup_lowimpact/**/seed_aligned_stage_records_compact.csv",
    "results/revision/engin1_formula_seed_aligned_stage_records*/**/seed_aligned_stage_records_compact.csv",
    "results/revision/engin1_formula_stage_marathon/**/seed_aligned_stage_records_compact.csv",
]


def rate(mask: pd.Series) -> float:
    return float(mask.mean()) if len(mask) else float("nan")


def count_success(mask: pd.Series) -> str:
    return f"{int(mask.sum())}/{len(mask)}"


def summarize_file(path: Path) -> dict[str, object]:
    df = pd.read_csv(path)
    if df.empty:
        return {}

    pred = df["test_mse"] <= 0.05
    full = df["full_pair_rank"] == 1
    readout = df["selected_contains_endpoints"] == 1
    refit = df["refit_pair_rank"] == 1
    prune = df["prune_endpoint_contains"] == 1

    first_broken = Counter(str(v) for v in df["first_broken_stage"].fillna("none"))
    dominant = "; ".join(f"{k}:{v}" for k, v in first_broken.most_common(3))

    row = {
        "setting": str(df["setting"].iloc[0]),
        "function": str(df["function"].iloc[0]),
        "source_path": str(path),
        "runs": len(df),
        "mean_test_mse": df["test_mse"].mean(),
        "prediction_success": count_success(pred),
        "full_pair_success": count_success(full),
        "readout_endpoint_success": count_success(readout),
        "refit_pair_success": count_success(refit),
        "prune_endpoint_success": count_success(prune),
        "pred_to_full_fail_rate": rate(pred & ~full),
        "full_to_readout_fail_rate": rate(full & ~readout),
        "readout_without_full_rate": rate(readout & ~full),
        "readout_to_refit_fail_rate": rate(readout & ~refit),
        "readout_to_prune_fail_rate": rate(readout & ~prune),
        "mean_full_margin": df["full_pair_margin"].mean(),
        "mean_readout_margin": df["readout_endpoint_margin"].mean(),
        "mean_refit_margin": df["refit_pair_margin"].mean(),
        "median_prune_support_size": df["prune_support_size"].median(),
        "dominant_first_broken": dominant,
    }
    return row


def find_inputs(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(Path(".").glob(pattern))
    return sorted(set(paths))


def write_markdown(df: pd.DataFrame, path: Path, max_rows: int) -> None:
    keep = [
        "setting",
        "runs",
        "mean_test_mse",
        "full_pair_success",
        "readout_endpoint_success",
        "prune_endpoint_success",
        "pred_to_full_fail_rate",
        "full_to_readout_fail_rate",
        "readout_without_full_rate",
        "readout_to_prune_fail_rate",
        "dominant_first_broken",
    ]
    show = df[keep].head(max_rows).copy()
    for col in [
        "mean_test_mse",
        "pred_to_full_fail_rate",
        "full_to_readout_fail_rate",
        "readout_without_full_rate",
        "readout_to_prune_fail_rate",
    ]:
        show[col] = show[col].map(lambda x: f"{x:.3f}")

    widths = {col: max(len(col), *(len(str(v)) for v in show[col])) for col in show.columns}
    lines = ["# Evidence-Transfer Seed-Aligned Summary", ""]
    lines.append("| " + " | ".join(col.ljust(widths[col]) for col in show.columns) + " |")
    lines.append("| " + " | ".join("-" * widths[col] for col in show.columns) + " |")
    for _, row in show.iterrows():
        lines.append("| " + " | ".join(str(row[col]).ljust(widths[col]) for col in show.columns) + " |")
    lines.append("")
    lines.append(f"Rows shown: {len(show)} of {len(df)}.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-csv", default="local_notes/generated/evidence_transfer_seed_aligned_summary_20260601.csv")
    parser.add_argument("--output-md", default="local_notes/generated/evidence_transfer_seed_aligned_summary_20260601.md")
    parser.add_argument("--max-md-rows", type=int, default=40)
    parser.add_argument("patterns", nargs="*", default=DEFAULT_PATTERNS)
    args = parser.parse_args()

    paths = find_inputs(args.patterns)
    rows = [summarize_file(path) for path in paths]
    rows = [row for row in rows if row]
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No seed-aligned stage-record CSVs found.")

    # Put paper-facing rows first, then formula breadth rows.
    df["_sort_core"] = (~df["function"].eq("core_interaction_c025")).astype(int)
    df = df.sort_values(["_sort_core", "function", "setting", "source_path"]).drop(columns=["_sort_core"])

    out_csv = Path(args.output_csv)
    out_md = Path(args.output_md)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    write_markdown(df, out_md, args.max_md_rows)
    print(f"Wrote {out_csv} ({len(df)} rows)")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()

