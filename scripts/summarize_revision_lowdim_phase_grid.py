#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


ROOT = Path("results/revision/lowdim_phase_grid")
OUT = ROOT / "summary"

FUNCTION_LABEL = {
    "core_interaction_c01": "c=0.10",
    "core_interaction_c025": "c=0.25",
    "core_interaction_c05": "c=0.50",
}


def _read_summary(path: Path) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if df.empty:
        return None
    return df


def load_phase() -> pd.DataFrame:
    rows = []
    for path in sorted((ROOT / "phase").glob("core_interaction_c*/d*/n*/support_sensitivity_summary.csv")):
        df = _read_summary(path)
        if df is None:
            continue
        fn = path.parents[2].name
        d_match = re.search(r"d(\d+)", path.parents[1].name)
        n_match = re.search(r"n(\d+)", path.parent.name)
        df.insert(0, "source", "phase")
        df.insert(1, "function_dir", fn)
        df.insert(2, "dimension_dir", int(d_match.group(1)) if d_match else pd.NA)
        df.insert(3, "samples_dir", int(n_match.group(1)) if n_match else pd.NA)
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def load_width() -> pd.DataFrame:
    rows = []
    for path in sorted((ROOT / "width_check").glob("core_interaction_c025/d*/n512/w*/support_sensitivity_summary.csv")):
        df = _read_summary(path)
        if df is None:
            continue
        d_match = re.search(r"d(\d+)", path.parents[1].name)
        w_match = re.search(r"w(\d+)", path.parent.name)
        df.insert(0, "source", "width_check")
        df.insert(1, "dimension_dir", int(d_match.group(1)) if d_match else pd.NA)
        df.insert(2, "width_dir", int(w_match.group(1)) if w_match else pd.NA)
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def focus(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df[
        (df["method"].astype(str) == "feature_edge_hybrid")
        & (pd.to_numeric(df["top_m"], errors="coerce").isin([4, 6, 10, 20]))
    ].copy()
    keep = [
        "source",
        "function",
        "function_dir",
        "samples",
        "samples_dir",
        "dimension",
        "dimension_dir",
        "width_hidden",
        "width_dir",
        "top_m",
        "screen_contains_all_true_vars_mean",
        "screen_true_var_recall_mean",
        "screen_contains_all_interaction_endpoints_mean",
        "screen_interaction_endpoint_recall_mean",
        "screen_contains_true_interactions_mean",
        "true_endpoint_rank_worst_mean",
        "endpoint_minus_max_nuisance_mean",
        "num_support_evals",
    ]
    return out[[c for c in keep if c in out.columns]].copy()


def write_markdown(phase_focus: pd.DataFrame, width_focus: pd.DataFrame) -> str:
    lines = ["# Low-Dimensional pyKAN Readout Phase Grid", ""]
    if not phase_focus.empty:
        phase4 = phase_focus[pd.to_numeric(phase_focus["top_m"], errors="coerce").eq(4)].copy()
        lines.extend(["## Fixed-protocol endpoint retention at top-m=4", ""])
        for fn in ["core_interaction_c01", "core_interaction_c025", "core_interaction_c05"]:
            part = phase4[phase4["function"].astype(str).eq(fn)].copy()
            if part.empty:
                continue
            lines.extend([f"### {FUNCTION_LABEL.get(fn, fn)}", ""])
            for _, row in part.sort_values(["dimension", "samples"]).iterrows():
                lines.append(
                    "- d={d}, n={n}: endpoints@4={ep:.2f}, pair@4={pair:.2f}, "
                    "worst-rank={rank:.1f}, margin={margin:.3f}, evals={evals}".format(
                        d=int(row["dimension"]),
                        n=int(row["samples"]),
                        ep=float(row["screen_contains_all_interaction_endpoints_mean"]),
                        pair=float(row["screen_contains_true_interactions_mean"]),
                        rank=float(row["true_endpoint_rank_worst_mean"]),
                        margin=float(row["endpoint_minus_max_nuisance_mean"]),
                        evals=int(row["num_support_evals"]),
                    )
                )
            lines.append("")

    if not width_focus.empty:
        width4 = width_focus[pd.to_numeric(width_focus["top_m"], errors="coerce").eq(4)].copy()
        lines.extend(["## Width check at c=0.25, n=512", ""])
        for _, row in width4.sort_values(["dimension", "width_hidden"]).iterrows():
            lines.append(
                "- d={d}, width={w}: endpoints@4={ep:.2f}, pair@4={pair:.2f}, "
                "worst-rank={rank:.1f}, margin={margin:.3f}, evals={evals}".format(
                    d=int(row["dimension"]),
                    w=int(row["width_hidden"]),
                    ep=float(row["screen_contains_all_interaction_endpoints_mean"]),
                    pair=float(row["screen_contains_true_interactions_mean"]),
                    rank=float(row["true_endpoint_rank_worst_mean"]),
                    margin=float(row["endpoint_minus_max_nuisance_mean"]),
                    evals=int(row["num_support_evals"]),
                )
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    phase = load_phase()
    width = load_width()
    if not phase.empty:
        phase.to_csv(OUT / "lowdim_phase_grid_all.csv", index=False)
    if not width.empty:
        width.to_csv(OUT / "lowdim_width_check_all.csv", index=False)

    phase_focus = focus(phase)
    width_focus = focus(width)
    if not phase_focus.empty:
        phase_focus.to_csv(OUT / "lowdim_phase_grid_focus.csv", index=False)
    if not width_focus.empty:
        width_focus.to_csv(OUT / "lowdim_width_check_focus.csv", index=False)

    summary = write_markdown(phase_focus, width_focus)
    (OUT / "summary.md").write_text(summary, encoding="utf-8")
    print(summary if summary.strip() else "No low-dimensional results found.")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
