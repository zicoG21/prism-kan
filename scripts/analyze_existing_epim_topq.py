#!/usr/bin/env python3
"""Mine existing edge-path top-pair caches for EPIM proposal evidence.

This is not a replacement for EPIM PairVerify.  It is a low-cost pre-analysis
of already-generated readout/taxonomy rows: do KAN edge-path pair scores place
the declared true pair into a small candidate set, even when they do not rank it
first?  That distinction is exactly the motivation for proposal-plus-verifier
rather than a raw readout leaderboard.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_INPUTS = [
    "results/revision/greatlakes_innovation_probe/*/innovation_detail.csv",
    "results/revision/local_gpu_innovation_12h/*/innovation_detail.csv",
    "results/revision/greatlakes_readout_taxonomy/*/support_sensitivity_detail.csv",
    "results/revision/local_gpu_readout_taxonomy_*/*/support_sensitivity_detail.csv",
]


def parse_literal(value, default):
    if value is None:
        return default
    try:
        if isinstance(value, float) and np.isnan(value):
            return default
    except TypeError:
        pass
    if isinstance(value, (list, tuple, dict)):
        return value
    try:
        return ast.literal_eval(str(value))
    except Exception:
        return default


def parse_pairs(value) -> list[tuple[int, int, float]]:
    out: list[tuple[int, int, float]] = []
    for item in parse_literal(value, []):
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            i, j = sorted((int(item[0]), int(item[1])))
            score = float(item[2]) if len(item) >= 3 else np.nan
            out.append((i, j, score))
    return out


def parse_true_pairs(row: pd.Series) -> list[tuple[int, int]]:
    if "true_interactions" in row and pd.notna(row.get("true_interactions")):
        out = []
        for item in parse_literal(row.get("true_interactions"), []):
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                out.append(tuple(sorted((int(item[0]), int(item[1])))))
        if out:
            return sorted(set(out))
    function = str(row.get("function", ""))
    if function == "core_interaction_c025":
        return [(2, 3)]
    return []


def find_inputs(patterns: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(Path(".").glob(pattern))
    return sorted(set(paths))


def source_label(path: Path) -> str:
    parts = path.parts
    for marker in [
        "greatlakes_innovation_probe",
        "local_gpu_innovation_12h",
        "greatlakes_readout_taxonomy",
        "local_gpu_readout_taxonomy_extra_12h",
        "local_gpu_readout_taxonomy_gaps",
    ]:
        if marker in parts:
            idx = parts.index(marker)
            if idx + 1 < len(parts):
                return f"{marker}/{parts[idx + 1]}"
            return marker
    return str(path.parent)


def load_rows(paths: list[Path]) -> pd.DataFrame:
    rows = []
    for path in paths:
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if "top_edge_pairs" not in df.columns or "function" not in df.columns:
            continue
        df = df.copy()
        df["source_path"] = str(path)
        df["source_label"] = source_label(path)
        rows.append(df)
    if not rows:
        raise SystemExit("No existing EPIM top-pair rows found.")
    return pd.concat(rows, ignore_index=True, sort=False)


def build_detail(raw: pd.DataFrame, budgets: list[int]) -> pd.DataFrame:
    detail_rows = []
    for _, row in raw.iterrows():
        true_pairs = parse_true_pairs(row)
        edge_pairs = parse_pairs(row.get("top_edge_pairs"))
        if not true_pairs or not edge_pairs:
            continue
        ranked_pairs = [(i, j) for i, j, _ in edge_pairs]
        edge_scores = {(i, j): float(score) for i, j, score in edge_pairs}
        # Current planted formulas use one target pair in the rows we can parse.
        true_pair = true_pairs[0]
        if true_pair in ranked_pairs:
            top10_rank = ranked_pairs.index(true_pair) + 1
            true_score = edge_scores.get(true_pair, np.nan)
        else:
            top10_rank = 11
            true_score = np.nan
        max_false = max(
            [score for pair, score in edge_scores.items() if pair != true_pair and np.isfinite(score)],
            default=np.nan,
        )
        for q in budgets:
            proposal = set(ranked_pairs[:q])
            endpoints = {v for pair in ranked_pairs[:q] for v in pair}
            detail_rows.append(
                {
                    "source_label": row.get("source_label"),
                    "source_path": row.get("source_path"),
                    "function": row.get("function"),
                    "samples": row.get("samples", np.nan),
                    "dimension": row.get("dimension", np.nan),
                    "noise": row.get("noise", np.nan),
                    "width_hidden": row.get("width_hidden", np.nan),
                    "update_grid": int("gridupdate" in str(row.get("source_label", ""))),
                    "method": row.get("method", "edge_path_cache"),
                    "seed": row.get("seed", row.get("heldout_seed", "aggregate")),
                    "top_m": row.get("top_m", np.nan),
                    "proposal_q": int(q),
                    "true_pair": str(true_pair),
                    "epim_top10_rank": int(top10_rank),
                    "epim_true_pair_in_topq": int(true_pair in proposal),
                    "epim_true_endpoints_in_topq_pairs": int(set(true_pair).issubset(endpoints)),
                    "epim_true_score_if_top10": true_score,
                    "epim_true_minus_top10_max_false": (
                        true_score - max_false if np.isfinite(true_score) and np.isfinite(max_false) else np.nan
                    ),
                }
            )
    return pd.DataFrame(detail_rows)


def build_summary(detail: pd.DataFrame) -> pd.DataFrame:
    if detail.empty:
        return detail
    group_cols = [
        "source_label",
        "function",
        "samples",
        "dimension",
        "noise",
        "width_hidden",
        "update_grid",
        "proposal_q",
    ]
    rows = []
    for key, group in detail.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, key))
        # Deduplicate repeated method/top_m rows when they share the same seed
        # and same top-edge cache; readout taxonomy stores several top_m rows
        # for one cache.
        dedup_cols = [
            "source_label",
            "function",
            "samples",
            "dimension",
            "noise",
            "width_hidden",
            "seed",
            "proposal_q",
            "epim_top10_rank",
        ]
        g = group.drop_duplicates(subset=[c for c in dedup_cols if c in group.columns])
        n = len(g)
        row["num_records"] = int(n)
        for col in ["epim_true_pair_in_topq", "epim_true_endpoints_in_topq_pairs"]:
            vals = pd.to_numeric(g[col], errors="coerce").dropna()
            row[f"{col}_count"] = int(vals.sum()) if len(vals) else 0
            row[f"{col}_rate"] = float(vals.mean()) if len(vals) else np.nan
        row["mean_top10_rank_capped11"] = float(pd.to_numeric(g["epim_top10_rank"], errors="coerce").mean())
        row["median_top10_rank_capped11"] = float(pd.to_numeric(g["epim_top10_rank"], errors="coerce").median())
        row["mean_true_minus_top10_max_false"] = float(
            pd.to_numeric(g["epim_true_minus_top10_max_false"], errors="coerce").mean()
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols, kind="stable")


def write_markdown(summary: pd.DataFrame, path: Path, max_rows: int) -> None:
    show = summary.copy().head(max_rows)
    keep = [
        "source_label",
        "function",
        "samples",
        "noise",
        "update_grid",
        "proposal_q",
        "num_records",
        "epim_true_pair_in_topq_rate",
        "epim_true_endpoints_in_topq_pairs_rate",
        "median_top10_rank_capped11",
        "mean_true_minus_top10_max_false",
    ]
    show = show[[c for c in keep if c in show.columns]].copy()
    for col in show.columns:
        if pd.api.types.is_float_dtype(show[col]):
            show[col] = show[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.3f}")
    lines = [
        "# Existing EPIM Top-q Evidence",
        "",
        "This mines already-generated edge-path top-pair caches.  Rank 11 means",
        "the true pair was not present in the stored top-10 edge-path list.",
        "",
        show.to_markdown(index=False),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="*", default=DEFAULT_INPUTS)
    parser.add_argument("--budgets", default="1,3,5,10")
    parser.add_argument("--out-prefix", type=Path, default=Path("local_notes/generated/existing_epim_topq_20260601"))
    parser.add_argument("--max-markdown-rows", type=int, default=80)
    args = parser.parse_args()

    budgets = sorted({int(v.strip()) for v in args.budgets.split(",") if v.strip()})
    paths = find_inputs(args.inputs)
    raw = load_rows(paths)
    detail = build_detail(raw, budgets)
    summary = build_summary(detail)

    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    detail_path = args.out_prefix.with_name(args.out_prefix.name + "_detail.csv")
    summary_path = args.out_prefix.with_name(args.out_prefix.name + "_summary.csv")
    md_path = args.out_prefix.with_name(args.out_prefix.name + ".md")
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    write_markdown(summary, md_path, max_rows=args.max_markdown_rows)
    print(f"Wrote detail:  {detail_path}")
    print(f"Wrote summary: {summary_path}")
    print(f"Wrote notes:   {md_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
