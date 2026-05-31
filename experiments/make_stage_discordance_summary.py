#!/usr/bin/env python3
"""Build a compact stage-discordance summary from revision CSVs.

The manuscript's new contribution is a stage record rather than another
interaction detector.  This helper makes that object explicit by joining
full-KAN functional-reliance rows with exposed-readout endpoint-surfacing rows
whenever both exist for the same controlled setting.
"""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "revision"
OUT_CSV = RESULTS / "stage_discordance_summary.csv"
OUT_MD = ROOT / "local_notes" / "stage_discordance_summary_20260531.md"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def infer_condition(path: Path) -> str:
    name = path.parent.name
    if name.startswith("gridupdate"):
        return "gridupdate"
    if name.startswith("noise005"):
        return "noise005"
    if name.startswith("noise010"):
        return "noise010"
    if name.startswith("clean") or name.startswith("c01"):
        return "clean"
    return "other"


def make_key(row: dict[str, str], condition: str) -> tuple[str, int, int, int, str]:
    return (
        row["function"],
        int(float(row["samples"])),
        int(float(row["dimension"])),
        int(float(row["width_hidden"])),
        condition,
    )


def collect_fullkan() -> dict[tuple[str, int, int, int, str], dict[str, str]]:
    full: dict[tuple[str, int, int, int, str], dict[str, str]] = {}
    roots = [
        RESULTS / "fullkan_anova_boundary",
        RESULTS / "fullkan_anova_followup",
    ]
    for root in roots:
        for path in sorted(root.glob("*/full_kan_pair_anova_summary.csv")):
            if "_60seed" not in path.parent.name:
                continue
            for row in read_rows(path):
                condition = infer_condition(path)
                key = make_key(row, condition)
                row = dict(row)
                row["setting"] = path.parent.name
                row["condition"] = condition
                full[key] = row
    return full


def collect_readouts() -> dict[tuple[str, int, int, int, str, str], dict[str, str]]:
    readouts: dict[tuple[str, int, int, int, str, str], dict[str, str]] = {}
    roots = [
        RESULTS / "greatlakes_readout_taxonomy",
        RESULTS / "local_gpu_readout_taxonomy_gaps",
        RESULTS / "local_gpu_readout_taxonomy_extra_12h",
    ]
    for root in roots:
        for path in sorted(root.glob("*/support_sensitivity_summary.csv")):
            condition = infer_condition(path)
            for row in read_rows(path):
                if int(float(row.get("top_m", 0))) != 4:
                    continue
                method = row["method"]
                key = make_key(row, condition) + (method,)
                row = dict(row)
                row["setting"] = path.parent.name
                row["condition"] = condition
                readouts[key] = row
    # Include the merged Great Lakes summary in case a future sync only carries
    # that consolidated file.
    merged = RESULTS / "greatlakes_readout_taxonomy" / "readout_taxonomy_merged_summary.csv"
    if merged.exists():
        for row in read_rows(merged):
            if int(float(row.get("top_m", 0))) != 4:
                continue
            condition = infer_condition(Path(row.get("setting", "clean")) / "x")
            method = row["method"]
            key = make_key(row, condition) + (method,)
            row = dict(row)
            row["condition"] = condition
            readouts.setdefault(key, row)
    return readouts


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


def build_rows() -> list[dict[str, str]]:
    full = collect_fullkan()
    readouts = collect_readouts()
    methods = [
        "feature_stability_var",
        "edge_stability_var",
        "edge_endpoint_mass",
        "feature_edge_hybrid",
        "edge_pair_hybrid",
    ]
    rows: list[dict[str, str]] = []
    for key, frow in sorted(full.items(), key=lambda item: item[0]):
        function, samples, dimension, width, condition = key
        for method in methods:
            rrow = readouts.get(key + (method,))
            if not rrow:
                continue
            full_rate = as_float(frow, "true_pair_beats_candidates_mean")
            endpoint_rate = as_float(rrow, "screen_contains_all_interaction_endpoints_mean")
            label, first_broken = classify(full_rate, endpoint_rate)
            rows.append(
                {
                    "function": function,
                    "n": str(samples),
                    "d": str(dimension),
                    "width": str(width),
                    "condition": condition,
                    "readout": method,
                    "full_rank1_rate": f"{full_rate:.3f}",
                    "readout_endpoints_at4": f"{endpoint_rate:.3f}",
                    "discordance_gap_endpoint_minus_full": f"{endpoint_rate - full_rate:.3f}",
                    "full_mean_rank": f"{as_float(frow, 'true_pair_rank_mean'):.1f}",
                    "readout_worst_endpoint_rank": f"{as_float(rrow, 'true_endpoint_rank_worst_mean'):.1f}",
                    "full_margin": f"{as_float(frow, 'true_minus_max_false_mean'):.5f}",
                    "readout_margin": f"{as_float(rrow, 'endpoint_minus_max_nuisance_mean'):.5f}",
                    "discordance_label": label,
                    "first_broken_stage": first_broken,
                    "full_runs": str(int(float(frow.get("num_runs", 0)))),
                    "readout_runs": str(int(float(rrow.get("num_support_evals", 0)))),
                }
            )
    return rows


def write_csv(rows: list[dict[str, str]]) -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def compact_md_table(rows: list[dict[str, str]]) -> str:
    preferred = {
        ("core_interaction_c025", "512", "100", "16", "clean"),
        ("core_interaction_c025", "768", "100", "16", "clean"),
        ("core_interaction_c025", "1024", "100", "16", "clean"),
        ("core_interaction_c025", "512", "100", "16", "gridupdate"),
        ("core_interaction_c025", "1024", "100", "16", "gridupdate"),
        ("core_interaction_c025", "512", "100", "16", "noise005"),
        ("core_interaction_c025", "1024", "100", "16", "noise005"),
        ("core_interaction_c025", "512", "100", "16", "noise010"),
        ("core_interaction_c025", "1024", "100", "16", "noise010"),
        ("core_interaction_c025", "512", "100", "32", "clean"),
        ("core_interaction_c025", "768", "100", "32", "clean"),
        ("core_interaction_c025", "512", "100", "32", "gridupdate"),
        ("core_interaction_c025", "1024", "100", "32", "gridupdate"),
    }
    selected = [
        row
        for row in rows
        if row["readout"] == "feature_edge_hybrid"
        and (row["function"], row["n"], row["d"], row["width"], row["condition"]) in preferred
    ]
    selected.sort(key=lambda r: (r["condition"], int(r["width"]), int(r["n"])))
    lines = [
        "| setting | full rank-1 | endpoints@4 | gap | full rank | endpoint rank | label | first broken |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in selected:
        setting = f"{row['condition']} w{row['width']} n{row['n']}"
        lines.append(
            "| "
            + " | ".join(
                [
                    setting,
                    row["full_rank1_rate"],
                    row["readout_endpoints_at4"],
                    row["discordance_gap_endpoint_minus_full"],
                    row["full_mean_rank"],
                    row["readout_worst_endpoint_rank"],
                    row["discordance_label"],
                    row["first_broken_stage"],
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def write_md(rows: list[dict[str, str]]) -> None:
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    total = len(rows)
    labels: dict[str, int] = {}
    for row in rows:
        labels[row["discordance_label"]] = labels.get(row["discordance_label"], 0) + 1
    label_text = ", ".join(f"{k}: {v}" for k, v in sorted(labels.items()))
    body = f"""# Stage Discordance Summary - 2026-05-31

This note turns the current revision results into the paper's main innovation
object: a stage-discordance record.  Each joined row compares two evidence
objects for the same controlled setting:

- **Full-KAN ANOVA**: whether the trained full-dimensional KAN function ranks
  the true pair above all candidate false pairs.
- **Exposed readout endpoint surfacing**: whether a pyKAN readout retains both
  interaction endpoints in the top-4 variables.

Output CSV: `results/revision/stage_discordance_summary.csv`.

Total joined rows: {total}.  Label counts: {label_text}.

## Compact Hybrid-Readout View

The table below uses `feature_edge_hybrid` as the compact exposed-readout row.
The full CSV also includes feature-only, edge-only, endpoint-mass, and
edge-pair hybrid readouts.

{compact_md_table(rows)}

## Main Reading

1. **The innovation point is the discordance record, not a new readout.**  The
   joined rows make the paper's contribution operational: the same known pair
   can be supported by one evidence object and missed by another.
2. **Clean width-16 rows show readout surfacing can be easier than full pair
   reliance.**  At `n=512`, endpoints surface while full-KAN all-pairs ANOVA
   rank-1 is only partial.  This motivates separating endpoint evidence from
   fitted-function pair reliance.
3. **Grid-update rows are a training-protocol boundary, not a simple KAN-wide
   verdict.**  At smaller `n`, both model reliance and endpoint surfacing are
   poor; by `n=1024`, endpoint surfacing can recover while full-model pair
   reliance remains mixed.
4. **Noise rows expose margin sensitivity.**  Endpoint and full-model margins
   move with the same qualitative pressure but not always at the same rate,
   which is exactly why the stage record reports margins instead of only
   success counts.

## Writing Use

This table can be used to sharpen the novelty paragraph:

> The audit output is not a single score. It is a discordance map across
> evidence objects. In the same controlled setting, full-model reliance,
> exposed endpoint surfacing, and pair evidence can be aligned, jointly absent,
> or split. The first-broken-stage label makes that split reportable.

"""
    OUT_MD.write_text(body)


def main() -> None:
    rows = build_rows()
    if not rows:
        raise SystemExit("No joined stage-discordance rows found.")
    write_csv(rows)
    write_md(rows)
    print(f"Wrote {OUT_CSV.relative_to(ROOT)} ({len(rows)} rows)")
    print(f"Wrote {OUT_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
