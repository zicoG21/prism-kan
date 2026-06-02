#!/usr/bin/env python3
"""Merge Great Lakes seed-aligned stage-record outputs into paper-ready tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PRIORITY = {
    "none": 0,
    "full-model reliance": 1,
    "readout endpoints": 2,
    "support refit": 3,
    "pruning": 4,
    "prediction": 5,
}


def latex_escape(text: object) -> str:
    s = str(text)
    return (
        s.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("$", "\\$")
        .replace("#", "\\#")
        .replace("_", "\\_")
        .replace("{", "\\{")
        .replace("}", "\\}")
    )


def safe_list(text: object) -> list[int]:
    try:
        return [int(v) for v in json.loads(str(text))]
    except Exception:
        return []


def dataframe_to_markdown(df: pd.DataFrame, index: bool = False) -> str:
    """Render markdown tables without pandas' optional tabulate dependency."""

    if df.empty:
        return ""
    table = df.reset_index() if index else df.copy()
    table = table.astype(str)
    headers = list(table.columns)
    rows = table.values.tolist()
    widths = [
        max(len(str(header)), *(len(str(row[i])) for row in rows))
        for i, header in enumerate(headers)
    ]

    def fmt_row(values: list[object]) -> str:
        cells = [str(value).ljust(widths[i]) for i, value in enumerate(values)]
        return "| " + " | ".join(cells) + " |"

    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    return "\n".join([fmt_row(headers), separator, *(fmt_row(row) for row in rows)])


def fmt_float(value: object, digits: int = 3) -> str:
    try:
        x = float(value)
    except Exception:
        return "NA"
    if not np.isfinite(x):
        return "NA"
    if abs(x) < 0.001 and x != 0:
        return f"{x:.1e}"
    return f"{x:.{digits}f}"


def read_all(roots: list[Path]) -> pd.DataFrame:
    frames = []
    for root in roots:
        for path in sorted(root.glob("*/seed_aligned_stage_records_detail.csv")):
            try:
                df = pd.read_csv(path)
            except Exception:
                continue
            df["source_root"] = str(root)
            df["source_file"] = str(path)
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def choose_showcase_rows(df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    if df.empty:
        return df
    work = df.copy()
    if "status" in work.columns:
        work = work[work["status"].fillna("ok") != "failed"].copy()
    if work.empty:
        return work
    work["_priority"] = work["first_broken_stage"].map(PRIORITY).fillna(9)
    rows = []
    seen = set()
    # Choose one row per setting and first-broken stage when possible.
    for _, row in work.sort_values(["_priority", "setting", "seed"]).iterrows():
        key = (str(row["setting"]), str(row["first_broken_stage"]))
        if key in seen:
            continue
        rows.append(row)
        seen.add(key)
        if len(rows) >= max_rows:
            break
    if len(rows) < max_rows:
        chosen = {(str(r["setting"]), int(r["seed"])) for r in rows}
        for _, row in work.sort_values(["setting", "seed"]).iterrows():
            key = (str(row["setting"]), int(row["seed"]))
            if key in chosen:
                continue
            rows.append(row)
            chosen.add(key)
            if len(rows) >= max_rows:
                break
    return pd.DataFrame(rows).drop(columns=["_priority"], errors="ignore")


def compact_for_paper(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        endpoint_ranks = safe_list(row.get("readout_endpoint_ranks", "[]"))
        selected = safe_list(row.get("selected_support", "[]"))
        prune_support = int(row["prune_support_size"]) if pd.notna(row.get("prune_support_size")) else -1
        rows.append(
            {
                "setting": row.get("setting", ""),
                "seed": int(row.get("seed", -1)),
                "mse": fmt_float(row.get("test_mse"), 4),
                "full_pair": f"{int(row.get('full_pair_rank', -1))} / {fmt_float(row.get('full_pair_margin'), 3)}",
                "readout": f"{endpoint_ranks} / {fmt_float(row.get('readout_endpoint_margin'), 3)}",
                "support": selected,
                "refit_pair": (
                    f"{int(row.get('refit_pair_rank', -1))} / {fmt_float(row.get('refit_pair_margin'), 3)}"
                    if pd.notna(row.get("refit_pair_rank"))
                    else "NA"
                ),
                "prune": f"{prune_support}, ep={int(row.get('prune_endpoint_contains', 0))}",
                "first_broken_stage": row.get("first_broken_stage", ""),
            }
        )
    return pd.DataFrame(rows)


def write_markdown(df: pd.DataFrame, out: Path) -> None:
    lines = [
        "# Merged Seed-Aligned Stage Records",
        "",
        "Columns `full_pair`, `readout`, and `refit_pair` are formatted as",
        "`rank / margin`. `prune` reports support size and endpoint containment.",
        "",
        dataframe_to_markdown(df, index=False) if not df.empty else "No rows found.",
        "",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")


def write_latex(df: pd.DataFrame, out: Path) -> None:
    cols = [
        ("setting", "Setting"),
        ("seed", "Seed"),
        ("mse", "MSE"),
        ("full_pair", "Full pair"),
        ("readout", "Readout endpoints"),
        ("prune", "Prune"),
        ("first_broken_stage", "First broken"),
    ]
    lines = [
        "% Generated by scripts/merge_seed_aligned_stage_records.py",
        "\\begin{tabular}{lllllll}",
        "\\toprule",
        " & ".join(label for _, label in cols) + r" \\",
        "\\midrule",
    ]
    for _, row in df.iterrows():
        lines.append(" & ".join(latex_escape(row[c]) for c, _ in cols) + r" \\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    out.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        nargs="+",
        default=[
            "results/revision/seed_aligned_stage_records",
            "results/revision/seed_aligned_stage_records_more",
        ],
        help="One or more seed-aligned result roots to merge.",
    )
    parser.add_argument(
        "--out_dir",
        default="local_notes/generated/seed_aligned_stage_records",
    )
    parser.add_argument("--max_rows", type=int, default=8)
    args = parser.parse_args()

    roots = [Path(p) for p in args.root]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    detail = read_all(roots)
    detail_path = out_dir / "merged_seed_aligned_stage_records_detail.csv"
    compact_path = out_dir / "merged_seed_aligned_stage_records_compact.csv"
    md_path = out_dir / "merged_seed_aligned_stage_records.md"
    tex_path = out_dir / "merged_seed_aligned_stage_records.tex"

    detail.to_csv(detail_path, index=False)
    showcase = choose_showcase_rows(detail, args.max_rows)
    compact = compact_for_paper(showcase) if not showcase.empty else pd.DataFrame()
    compact.to_csv(compact_path, index=False)
    write_markdown(compact, md_path)
    write_latex(compact, tex_path)

    print(f"Read {len(detail)} rows from {', '.join(str(p) for p in roots)}")
    print(f"Wrote {detail_path}")
    print(f"Wrote {compact_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {tex_path}")
    if not compact.empty:
        print(compact.to_string(index=False))


if __name__ == "__main__":
    main()
