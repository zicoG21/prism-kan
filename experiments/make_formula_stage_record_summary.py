#!/usr/bin/env python3
"""Summarize formula-family stage-discordance expansion rows.

The Great Lakes array writes one directory per formula family with two evidence
objects:

  - full/full_kan_pair_anova_summary.csv
  - readout/support_sensitivity_summary.csv

This script joins those rows into the same stage-record schema used in the main
paper, selecting top-m equal to the labeled active support size for each
formula.  It intentionally keeps ontology labels so nested or higher-order
targets are not averaged into simple pair claims without context.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_ROOT = Path("results/revision/formula_stage_record")
DEFAULT_OUT = DEFAULT_ROOT / "formula_stage_record_summary.csv"
DEFAULT_MD = Path("local_notes/formula_stage_record_summary_20260531.md")

ONTOLOGY_DEFAULTS = {
    "formula_bilinear": "simple_pair",
    "formula_weak_centered": "weak_centered_pair",
    "formula_trig_product": "nonlinear_pair",
    "formula_exp_product": "nonlinear_pair",
    "formula_log_product": "nonlinear_pair",
    "formula_sqrt_energy": "nonlinear_pair",
    "formula_nested_trig": "nested_pair_label",
}

TRUE_STRUCTURE = {
    "formula_bilinear": (3, 1),
    "formula_weak_centered": (4, 1),
    "formula_trig_product": (3, 1),
    "formula_exp_product": (3, 1),
    "formula_log_product": (3, 1),
    "formula_sqrt_energy": (3, 1),
    "formula_nested_trig": (3, 1),
}


def load_manifest(path: Path) -> dict[str, str]:
    manifest = path / "stage_record_manifest.txt"
    out: dict[str, str] = {}
    if not manifest.exists():
        return out
    for line in manifest.read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def true_support_size(function: str, samples: int, dimension: int, noise: float) -> tuple[int, int]:
    if function in TRUE_STRUCTURE:
        return TRUE_STRUCTURE[function]
    raise ValueError(
        f"No true-structure metadata for {function!r}. "
        "Add it to TRUE_STRUCTURE before summarizing this formula."
    )


def classify(full_rate: float, endpoint_rate: float) -> tuple[str, str]:
    if full_rate >= 0.8 and endpoint_rate >= 0.8:
        return "aligned high", "none/late"
    if full_rate < 0.5 and endpoint_rate < 0.5:
        return "aligned low", "model reliance or earlier"
    if full_rate >= 0.8 and endpoint_rate < 0.5:
        return "reliance without surfacing", "endpoint surfacing"
    if full_rate < 0.5 and endpoint_rate >= 0.8:
        return "surfacing without reliance", "model reliance"
    return "mixed boundary", "margin-sensitive"


def first_float(row: pd.Series, key: str, default: float = np.nan) -> float:
    try:
        return float(row.get(key, default))
    except Exception:
        return default


def summarize_one(path: Path, method: str) -> dict[str, object] | None:
    manifest = load_manifest(path)
    full_path = path / "full" / "full_kan_pair_anova_summary.csv"
    readout_path = path / "readout" / "support_sensitivity_summary.csv"
    if not full_path.exists() or not readout_path.exists():
        return None

    full = pd.read_csv(full_path)
    readout = pd.read_csv(readout_path)
    if full.empty or readout.empty:
        return None

    frow = full.iloc[0]
    function = str(frow["function"])
    samples = int(float(frow["samples"]))
    dimension = int(float(frow["dimension"]))
    noise = float(frow.get("noise", manifest.get("noise", 0.0)))
    active_size, pair_count = true_support_size(function, samples, dimension, noise)

    candidate = readout[
        readout["method"].astype(str).eq(method)
        & pd.to_numeric(readout["top_m"], errors="coerce").eq(active_size)
    ].copy()
    if candidate.empty:
        candidate = readout[readout["method"].astype(str).eq(method)].copy()
    if candidate.empty:
        return None
    rrow = candidate.iloc[0]

    full_rate = first_float(frow, "true_pair_beats_candidates_mean")
    endpoint_rate = first_float(rrow, "screen_contains_all_interaction_endpoints_mean")
    label, first_broken = classify(full_rate, endpoint_rate)
    ontology = manifest.get("ontology", ONTOLOGY_DEFAULTS.get(function, "formula_pair_target"))

    return {
        "label": path.name,
        "function": function,
        "ontology": ontology,
        "samples": samples,
        "dimension": dimension,
        "noise": noise,
        "width_hidden": int(float(frow["width_hidden"])),
        "active_size": active_size,
        "pair_count": pair_count,
        "readout_method": method,
        "readout_top_m": int(float(rrow["top_m"])),
        "test_mse": first_float(frow, "test_mse_mean"),
        "full_rank1_rate": full_rate,
        "full_mean_rank": first_float(frow, "true_pair_rank_mean"),
        "full_margin": first_float(frow, "true_minus_max_false_mean"),
        "readout_endpoints_at_m": endpoint_rate,
        "readout_worst_endpoint_rank": first_float(rrow, "true_endpoint_rank_worst_mean"),
        "readout_margin": first_float(rrow, "endpoint_minus_max_nuisance_mean"),
        "discordance_gap_endpoint_minus_full": endpoint_rate - full_rate,
        "discordance_label": label,
        "first_broken_stage": first_broken,
        "full_runs": int(float(frow.get("num_runs", 0))),
        "readout_runs": int(float(rrow.get("num_support_evals", 0))),
    }


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


def write_markdown(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Formula-family stage-record summary",
        "",
        "This table joins full-KAN pair reliance with exposed-readout endpoint surfacing for formula-suite rows.",
        "",
    ]
    if df.empty:
        lines.append("No completed rows found.")
    else:
        show_cols = [
            "function",
            "ontology",
            "full_rank1_rate",
            "readout_endpoints_at_m",
            "full_margin",
            "readout_margin",
            "discordance_label",
            "first_broken_stage",
        ]
        show = df[show_cols].copy()
        for col in ["full_rank1_rate", "readout_endpoints_at_m", "full_margin", "readout_margin"]:
            show[col] = pd.to_numeric(show[col], errors="coerce").map(lambda x: f"{x:.3f}")
        lines.append(dataframe_to_markdown(show, index=False))
        lines.extend(
            [
                "",
                "Label counts:",
                "",
                dataframe_to_markdown(
                    df["discordance_label"].value_counts().rename_axis("label").reset_index(name="count"),
                    index=False,
                ),
            ]
        )
    path.write_text("\n".join(lines) + "\n")
    print(f"wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--md", type=Path, default=DEFAULT_MD)
    parser.add_argument("--method", default="feature_edge_hybrid")
    args = parser.parse_args()

    rows = []
    for path in sorted(args.root.glob("*")):
        if not path.is_dir():
            continue
        row = summarize_one(path, args.method)
        if row is not None:
            rows.append(row)

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"wrote {args.out} ({len(df)} rows)")
    write_markdown(df, args.md)


if __name__ == "__main__":
    main()
