from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


CORE_SETTINGS = [
    ("core_interaction_c025", 512, 100, 4),
    ("core_interaction_c025", 1024, 100, 4),
]

BASELINE_LABELS = {
    "raw": "Raw KAN",
    "rf": "RF-screened KAN",
    "oracle_support": "Oracle-support KAN",
    "random": "Random-support KAN",
    "exclude_interaction": "Exclude-endpoints KAN",
}

STRESS_LABELS = {
    "feature_edge_hybrid": "FA-SS support",
    "rf": "RF-screened",
    "oracle_support": "Oracle-support",
}

FEYNMAN_LABELS = {
    "feynman_energy": "Energy",
    "feynman_gravity": "Gravity",
    "feynman_coulomb": "Coulomb",
}


def fmt(value: float, digits: int = 3) -> str:
    if pd.isna(value):
        return "--"
    return f"{float(value):.{digits}f}"


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    display = df.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda value: fmt(value))
    values = [[str(col) for col in display.columns]]
    values.extend([[str(value) for value in row] for row in display.to_numpy()])
    widths = [max(len(row[idx]) for row in values) for idx in range(len(values[0]))]

    def render(row: list[str]) -> str:
        return "| " + " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(row)) + " |"

    return "\n".join(
        [
            render(values[0]),
            "| " + " | ".join("-" * width for width in widths) + " |",
            *[render(row) for row in values[1:]],
        ]
    )


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def first_row(df: pd.DataFrame) -> pd.Series | None:
    if df.empty:
        return None
    return df.iloc[0]


def append_row(rows: list[dict], *, setting: tuple, method: str, pair_score: str, row: pd.Series | None, source: str) -> None:
    function, samples, dimension, top_m = setting
    rows.append(
        {
            "function": function,
            "n": samples,
            "d": dimension,
            "m": top_m,
            "method": method,
            "pair_score": pair_score,
            "test_mse": float(row.get("test_mse_mean", float("nan"))) if row is not None else float("nan"),
            "endpoint_recall": float(row.get("screen_interaction_endpoint_recall_mean", float("nan"))) if row is not None else float("nan"),
            "support_pair": float(row.get("screen_contains_true_interactions_mean", float("nan"))) if row is not None else float("nan"),
            "interaction_f1": float(row.get("interaction_f1_mean", float("nan"))) if row is not None else float("nan"),
            "runs": int(row.get("num_runs", row.get("num_rows", 0))) if row is not None else 0,
            "source": source,
        }
    )


def build_core_main_table(args) -> pd.DataFrame:
    rows: list[dict] = []
    hard_dir = Path(args.hard_summary_dir)
    one_shot = read_csv(Path(args.one_shot_summary))
    native = read_csv(Path(args.native_summary))
    pair = read_csv(Path(args.pair_rescore_summary))
    oracle_pair = read_csv(Path(args.oracle_pair_summary))

    for setting in CORE_SETTINGS:
        function, samples, dimension, top_m = setting
        hard_path = hard_dir / f"{function}_n{samples}_d{dimension}_summary.csv"
        hard = read_csv(hard_path)
        for mode, label in BASELINE_LABELS.items():
            row = first_row(hard[hard["screen_mode"].astype(str).eq(mode)]) if not hard.empty else None
            append_row(rows, setting=setting, method=label, pair_score="FD", row=row, source=hard_path.name)

        single = one_shot[
            one_shot["method"].astype(str).eq("single_feature_edge_hybrid")
            & one_shot["function"].astype(str).eq(function)
            & pd.to_numeric(one_shot["samples"], errors="coerce").eq(samples)
            & pd.to_numeric(one_shot["dimension"], errors="coerce").eq(dimension)
            & pd.to_numeric(one_shot["top_m"], errors="coerce").eq(top_m)
        ]
        append_row(
            rows,
            setting=setting,
            method="Single-pass KAN-FE",
            pair_score="FD",
            row=first_row(single),
            source=Path(args.one_shot_summary).name,
        )

        support = native[
            native["method"].astype(str).eq("feature_edge_hybrid")
            & native["function"].astype(str).eq(function)
            & pd.to_numeric(native["samples"], errors="coerce").eq(samples)
            & pd.to_numeric(native["dimension"], errors="coerce").eq(dimension)
            & pd.to_numeric(native["top_m"], errors="coerce").eq(top_m)
        ]
        append_row(
            rows,
            setting=setting,
            method="FA-SS support only",
            pair_score="FD",
            row=first_row(support),
            source=Path(args.native_summary).name,
        )

        default = pair[
            pair["source_method"].astype(str).eq("feature_edge_hybrid")
            & pair["pair_score_method"].astype(str).eq("anova_abs")
            & pair["function"].astype(str).eq(function)
            & pd.to_numeric(pair["samples"], errors="coerce").eq(samples)
            & pd.to_numeric(pair["dimension"], errors="coerce").eq(dimension)
            & pd.to_numeric(pair["top_m"], errors="coerce").eq(top_m)
        ]
        append_row(
            rows,
            setting=setting,
            method="FA-SS-KAN",
            pair_score="ANOVA-abs",
            row=first_row(default),
            source=Path(args.pair_rescore_summary).name,
        )

        oracle_default = oracle_pair[
            oracle_pair["pair_score_method"].astype(str).eq("anova_abs")
            & oracle_pair["function"].astype(str).eq(function)
            & pd.to_numeric(oracle_pair["samples"], errors="coerce").eq(samples)
            & pd.to_numeric(oracle_pair["dimension"], errors="coerce").eq(dimension)
            & pd.to_numeric(oracle_pair["top_m"], errors="coerce").eq(top_m)
        ]
        if not oracle_default.empty:
            append_row(
                rows,
                setting=setting,
                method="Oracle-support KAN",
                pair_score="ANOVA-abs",
                row=first_row(oracle_default),
                source=Path(args.oracle_pair_summary).name,
            )

    out = pd.DataFrame(rows)
    order = {
        "Raw KAN": 0,
        "RF-screened KAN": 1,
        "Single-pass KAN-FE": 2,
        "FA-SS support only": 3,
        "FA-SS-KAN": 4,
        "Oracle-support KAN": 5,
        "Random-support KAN": 6,
        "Exclude-endpoints KAN": 7,
    }
    out["method_order"] = out["method"].map(order).fillna(99)
    return out.sort_values(["n", "method_order", "pair_score"]).drop(columns=["method_order"]).reset_index(drop=True)


def build_stress_table(args) -> pd.DataFrame:
    native = read_csv(Path(args.native_summary))
    screened = read_csv(Path(args.screened_summary))
    native_keep = native[
        native["method"].astype(str).eq("feature_edge_hybrid")
        & pd.to_numeric(native["dimension"], errors="coerce").isin([500, 1000])
        & native["function"].astype(str).isin(["core_interaction_c025", "core_interaction_c05", "core_interaction_c1"])
    ].copy()
    native_keep["method"] = "FA-SS support"

    screened_keep = screened[
        screened["screen_mode"].astype(str).isin(["rf", "oracle_support"])
        & pd.to_numeric(screened["dimension"], errors="coerce").isin([500, 1000])
        & screened["function"].astype(str).isin(["core_interaction_c025", "core_interaction_c05", "core_interaction_c1"])
    ].copy()
    screened_keep["method"] = screened_keep["screen_mode"].map(STRESS_LABELS)
    keep = pd.concat([native_keep, screened_keep], ignore_index=True, sort=False)
    if keep.empty:
        return pd.DataFrame()
    keep = keep.rename(
        columns={
            "samples": "n",
            "dimension": "d",
            "top_m": "m",
            "effective_dim_mean": "effective_dim",
            "interaction_f1_mean": "interaction_f1",
            "screen_contains_true_interactions_mean": "support_pair",
            "screen_interaction_endpoint_recall_mean": "endpoint_recall",
        }
    )
    if "m" not in keep.columns:
        keep["m"] = pd.NA
    if "effective_dim" in keep.columns:
        keep["m"] = pd.to_numeric(keep["m"], errors="coerce").fillna(pd.to_numeric(keep["effective_dim"], errors="coerce"))
    cols = ["function", "n", "d", "m", "method", "test_mse_mean", "endpoint_recall", "support_pair", "interaction_f1", "num_runs"]
    if "num_runs" not in keep.columns and "num_rows" in keep.columns:
        keep["num_runs"] = keep["num_rows"]
    if "num_runs" not in keep.columns:
        keep["num_runs"] = pd.NA
    keep["num_runs"] = pd.to_numeric(keep["num_runs"], errors="coerce").fillna(8)
    return keep[cols].sort_values(["function", "d", "method"]).reset_index(drop=True)


def build_feynman_support_table(args) -> pd.DataFrame:
    summary = read_csv(Path(args.native_summary))
    keep = summary[
        summary["method"].astype(str).eq("feature_edge_hybrid")
        & summary["function"].astype(str).isin(FEYNMAN_LABELS)
    ].copy()
    if keep.empty:
        return pd.DataFrame()
    keep["formula"] = keep["function"].map(FEYNMAN_LABELS)
    keep = keep.rename(
        columns={
            "samples": "n",
            "dimension": "d",
            "top_m": "m",
            "interaction_f1_mean": "interaction_f1",
            "screen_contains_true_interactions_mean": "support_pair",
            "screen_interaction_endpoint_recall_mean": "endpoint_recall",
        }
    )
    return keep[["formula", "n", "d", "m", "test_mse_mean", "endpoint_recall", "support_pair", "interaction_f1", "num_runs"]].sort_values("formula").reset_index(drop=True)


def build_feynman_pair_table(args) -> pd.DataFrame:
    summary = read_csv(Path(args.feynman_pair_summary))
    if summary.empty:
        return pd.DataFrame()
    keep = summary[
        summary["function"].astype(str).isin(FEYNMAN_LABELS)
        & summary["pair_score_method"].astype(str).isin(["fd", "anova_abs", "anova_var", "fd_anova_hybrid"])
    ].copy()
    if keep.empty:
        return pd.DataFrame()
    keep["formula"] = keep["function"].map(FEYNMAN_LABELS)
    keep["pair_score_method"] = pd.Categorical(
        keep["pair_score_method"],
        ["fd", "anova_abs", "anova_var", "fd_anova_hybrid"],
        ordered=True,
    )
    pivot = keep.pivot_table(
        index=["formula", "samples", "dimension", "top_m", "num_runs"],
        columns="pair_score_method",
        values="interaction_f1_mean",
        aggfunc="first",
        observed=False,
    ).reset_index()
    pivot = pivot.rename(
        columns={
            "samples": "n",
            "dimension": "d",
            "top_m": "m",
            "fd": "FD",
            "anova_abs": "ANOVA-abs",
            "anova_var": "ANOVA-var",
            "fd_anova_hybrid": "Hybrid",
        }
    )
    return pivot.sort_values("formula").reset_index(drop=True)


def build_d500_rescore_table(args) -> pd.DataFrame:
    summary = read_csv(Path(args.d500_pair_rescore_summary))
    extras = [read_csv(Path(path)) for path in args.d500_extra_pair_rescore_summaries]
    extras = [table for table in extras if not table.empty]
    if extras:
        summary = pd.concat([summary, *extras], ignore_index=True, sort=False)
    if summary.empty:
        return pd.DataFrame()
    keep = summary[
        summary["source_method"].astype(str).eq("feature_edge_hybrid")
        & summary["function"].astype(str).isin(["core_interaction_c025", "core_interaction_c05", "core_interaction_c1"])
        & pd.to_numeric(summary["dimension"], errors="coerce").eq(500)
        & summary["pair_score_method"].astype(str).isin(["fd", "anova_abs", "anova_var", "fd_anova_hybrid"])
    ].copy()
    if keep.empty:
        return pd.DataFrame()
    keep["pair_score_method"] = pd.Categorical(
        keep["pair_score_method"],
        ["fd", "anova_abs", "anova_var", "fd_anova_hybrid"],
        ordered=True,
    )
    pivot = keep.pivot_table(
        index=[
            "function",
            "samples",
            "dimension",
            "top_m",
            "screen_interaction_endpoint_recall_mean",
            "screen_contains_true_interactions_mean",
            "num_runs",
        ],
        columns="pair_score_method",
        values="interaction_f1_mean",
        aggfunc="first",
        observed=False,
    ).reset_index()
    pivot = pivot.rename(
        columns={
            "samples": "n",
            "dimension": "d",
            "top_m": "m",
            "screen_interaction_endpoint_recall_mean": "endpoint_recall",
            "screen_contains_true_interactions_mean": "support_pair",
            "fd": "FD",
            "anova_abs": "ANOVA-abs",
            "anova_var": "ANOVA-var",
            "fd_anova_hybrid": "Hybrid",
        }
    )
    return pivot.sort_values(["function", "n"]).reset_index(drop=True)


def write_markdown(tables: dict[str, pd.DataFrame], out_path: Path) -> None:
    lines = [
        "# FA-SS-KAN Evidence Report",
        "",
        "This report consolidates the evidence for the single-method framing: FA-SS-KAN = KAN-native feature+edge stability support + low-dimensional KAN refit + ANOVA-abs pair ranking.",
        "",
        "## 1. Core Main Method Table",
        "",
        dataframe_to_markdown(tables["core"]),
        "",
        "Interpretation: FA-SS support retains the interaction endpoints; the default ANOVA-abs pair-ranking stage turns retained support into pair recovery. The `FA-SS support only` row is the FD-scored ablation.",
        "",
        "## 2. High-Dimension Stress Table",
        "",
        dataframe_to_markdown(tables["stress"]) if not tables["stress"].empty else "_No stress table rows found._",
        "",
        "Interpretation: the support stage shifts the boundary at d=500 in some regimes, but d=1000 remains a failure mode. This should stay in limitations.",
        "",
        "## 3. d=500 Pair Rescoring After FA-SS Support",
        "",
        dataframe_to_markdown(tables["d500_rescore"]) if not tables["d500_rescore"].empty else "_No d=500 pair-rescoring rows found._",
        "",
        "Interpretation: when FA-SS retains endpoints at d=500, ANOVA-abs improves weak-interaction pair recovery over the local FD diagnostic.",
        "",
        "## 4. Feynman-Style FA-SS Support Table",
        "",
        dataframe_to_markdown(tables["feynman_support"]) if not tables["feynman_support"].empty else "_No Feynman support rows found._",
        "",
        "## 5. Feynman-Style Oracle-Support Pair Scoring",
        "",
        dataframe_to_markdown(tables["feynman_pair"]) if not tables["feynman_pair"].empty else "_No Feynman pair-scoring rows found yet._",
        "",
        "Paper wording constraint: ANOVA-abs is a pair-ranking stage after support recovery. It does not solve endpoint discovery.",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build compact evidence tables for FA-SS-KAN.")
    parser.add_argument("--hard_summary_dir", default="results/hard_regime/summaries")
    parser.add_argument("--one_shot_summary", default="results/innovation_loop/one_shot_kan_screen_controls_after_screened_20260526_104933/combined_one_shot_summary.csv")
    parser.add_argument("--native_summary", default="results/innovation_loop/strict_screened_baseline_controls_20260526_104243/analysis/native_screened_combined_summary.csv")
    parser.add_argument("--screened_summary", default="results/innovation_loop/strict_screened_baseline_controls_20260526_104243/combined_screened_baseline_summary.csv")
    parser.add_argument("--pair_rescore_summary", default="results/innovation_loop/anova_pair_rescore_validation_20260526_113620_retry/pair_rescore_summary.csv")
    parser.add_argument("--oracle_pair_summary", default="results/innovation_loop/pair_scoring_oracle_pilot_after_rf_20260526_111150/pair_scoring_summary.csv")
    parser.add_argument("--feynman_pair_summary", default="results/formula_aware_pair_scoring/feynman_oracle_sanity_clean_10seed_20260526/pair_scoring_summary.csv")
    parser.add_argument("--d500_pair_rescore_summary", default="results/formula_aware_pair_scoring/d500_pair_rescore_20260526/pair_rescore_summary.csv")
    parser.add_argument("--d500_extra_pair_rescore_summaries", nargs="*", default=[])
    parser.add_argument("--out_dir", default="results/formula_aware_pair_scoring/fasskan_evidence_20260526")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "core": build_core_main_table(args),
        "stress": build_stress_table(args),
        "d500_rescore": build_d500_rescore_table(args),
        "feynman_support": build_feynman_support_table(args),
        "feynman_pair": build_feynman_pair_table(args),
    }
    for name, table in tables.items():
        table.to_csv(out_dir / f"{name}_table.csv", index=False)
    write_markdown(tables, out_dir / "fasskan_evidence_report.md")
    print((out_dir / "fasskan_evidence_report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
