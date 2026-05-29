#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path("results/revision/d100_c025_hparam_sensitivity")
OUT = ROOT / "summary"


def load_all() -> pd.DataFrame:
    rows = []
    for path in sorted(ROOT.glob("n*/**/support_sensitivity_summary.csv")):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if df.empty:
            continue
        df.insert(0, "config", path.parent.name)
        df.insert(0, "n_label", path.parents[1].name)
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_all()
    if df.empty:
        print("No sensitivity summaries found.")
        return

    df.to_csv(OUT / "d100_c025_hparam_sensitivity_all.csv", index=False)

    focus = df[
        (df["method"].astype(str) == "feature_edge_hybrid")
        & (pd.to_numeric(df["top_m"], errors="coerce").isin([4, 6, 20]))
    ].copy()
    keep = [
        "n_label",
        "config",
        "samples",
        "width_hidden",
        "grid",
        "lamb",
        "probe_steps",
        "top_m",
        "screen_contains_all_interaction_endpoints_mean",
        "screen_interaction_endpoint_recall_mean",
        "screen_contains_true_interactions_mean",
        "true_endpoint_rank_worst_mean",
        "endpoint_minus_max_nuisance_mean",
        "num_support_evals",
    ]
    focus = focus[[c for c in keep if c in focus.columns]].copy()
    focus.to_csv(OUT / "d100_c025_hparam_sensitivity_focus.csv", index=False)

    lines = ["# d=100, c=0.25 pyKAN Readout Hyperparameter Sensitivity", ""]
    for n_label, part in focus.groupby("n_label", sort=True):
        lines.extend([f"## {n_label}", ""])
        for _, row in part[part["top_m"].eq(4)].sort_values("config").iterrows():
            lines.append(
                "- {config}: endpoints@4={ep:.2f}, pair-retained@4={pair:.2f}, "
                "worst-rank={rank:.1f}, margin={margin:.3f}, evals={evals}".format(
                    config=row["config"],
                    ep=float(row["screen_contains_all_interaction_endpoints_mean"]),
                    pair=float(row["screen_contains_true_interactions_mean"]),
                    rank=float(row["true_endpoint_rank_worst_mean"]),
                    margin=float(row["endpoint_minus_max_nuisance_mean"]),
                    evals=int(row["num_support_evals"]),
                )
            )
        lines.append("")

    # Compact table for paper text: default, best, and worst by endpoints@4.
    compact_rows = []
    for n_label, part in focus[focus["top_m"].eq(4)].groupby("n_label", sort=True):
        part = part.copy()
        part["score"] = part["screen_contains_all_interaction_endpoints_mean"].astype(float)
        default = part[part["config"].eq("default_w8_g5_l1e-3_s35")]
        if not default.empty:
            compact_rows.append(default.iloc[0].to_dict() | {"selection": "default"})
        compact_rows.append(part.sort_values("score", ascending=False).iloc[0].to_dict() | {"selection": "best"})
        compact_rows.append(part.sort_values("score", ascending=True).iloc[0].to_dict() | {"selection": "worst"})
    compact = pd.DataFrame(compact_rows)
    compact.to_csv(OUT / "d100_c025_hparam_sensitivity_compact.csv", index=False)

    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print((OUT / "summary.md").read_text(encoding="utf-8"))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
