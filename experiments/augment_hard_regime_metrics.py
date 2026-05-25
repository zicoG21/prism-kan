from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import Iterable, Sequence, Tuple

import numpy as np
import pandas as pd


Pair = Tuple[int, int]


def parse_literal(value, default):
    if isinstance(value, (list, tuple)):
        return value
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default
    text = str(value).strip()
    if text == "" or text.lower() == "nan":
        return default
    try:
        return ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return default


def canonical_pairs(pairs: Sequence[Tuple[int, int]]) -> Tuple[Pair, ...]:
    out = []
    for pair in pairs:
        if len(pair) != 2:
            continue
        i, j = pair
        out.append(tuple(sorted((int(i), int(j)))))
    return tuple(out)


def interaction_endpoints(pairs: Iterable[Pair]) -> Tuple[int, ...]:
    endpoints = set()
    for i, j in pairs:
        endpoints.add(int(i))
        endpoints.add(int(j))
    return tuple(sorted(endpoints))


def recall_and_hit(selected: Iterable[int], true_items: Iterable[int]) -> tuple[float, float]:
    selected_set = set(int(v) for v in selected)
    true_set = set(int(v) for v in true_items)
    if not true_set:
        return np.nan, np.nan
    return (
        len(selected_set & true_set) / len(true_set),
        float(true_set.issubset(selected_set)),
    )


def derive_row(row: pd.Series) -> dict:
    true_interactions = canonical_pairs(parse_literal(row.get("true_interactions"), []))
    true_endpoints = interaction_endpoints(true_interactions)

    screen_features = parse_literal(row.get("selected_screen_features"), [])
    selected_variables = parse_literal(row.get("selected_variables"), [])
    selected_interactions = canonical_pairs(parse_literal(row.get("selected_interactions"), []))
    selected_interaction_endpoints = interaction_endpoints(selected_interactions)

    screen_endpoint_recall, screen_endpoint_hit = recall_and_hit(screen_features, true_endpoints)
    explain_endpoint_recall, explain_endpoint_hit = recall_and_hit(selected_variables, true_endpoints)
    selected_pair_endpoint_recall, selected_pair_endpoint_hit = recall_and_hit(
        selected_interaction_endpoints,
        true_endpoints,
    )

    true_score = pd.to_numeric(row.get("true_interaction_score_mean"), errors="coerce")
    max_false = pd.to_numeric(row.get("max_nontrue_interaction_score"), errors="coerce")
    margin = true_score - max_false if np.isfinite(true_score) and np.isfinite(max_false) else np.nan
    beats_all_false = float(margin > 0.0) if np.isfinite(margin) else np.nan

    interaction_f1 = pd.to_numeric(row.get("interaction_f1"), errors="coerce")
    if not true_endpoints:
        failure_stage = "no_true_interaction"
    elif screen_endpoint_hit != 1.0:
        failure_stage = "screen_misses_endpoint"
    elif explain_endpoint_hit != 1.0:
        failure_stage = "variable_ranking_misses_endpoint"
    elif not np.isfinite(interaction_f1) or interaction_f1 < 1.0:
        failure_stage = "interaction_ranking_misses_pair"
    else:
        failure_stage = "recovered"

    # Existing detail files did not store the full pair-score ranking. For the
    # single-pair core benchmark, a positive score margin is exactly rank 1;
    # otherwise the exact rank is only known to be greater than 1.
    if len(true_interactions) == 1 and np.isfinite(margin):
        rank_bucket = "1" if margin > 0.0 else ">1"
        rank_lower_bound = 1.0 if margin > 0.0 else 2.0
    else:
        rank_bucket = ""
        rank_lower_bound = np.nan

    return {
        "screen_endpoint_recall": screen_endpoint_recall,
        "screen_contains_all_endpoints": screen_endpoint_hit,
        "explain_interaction_endpoint_recall": explain_endpoint_recall,
        "explain_contains_all_interaction_endpoints": explain_endpoint_hit,
        "selected_interaction_endpoint_recall": selected_pair_endpoint_recall,
        "selected_interaction_contains_all_endpoints": selected_pair_endpoint_hit,
        "true_interaction_mean_score_margin": margin,
        "true_interaction_beats_all_false": beats_all_false,
        "true_interaction_rank_bucket": rank_bucket,
        "true_interaction_rank_lower_bound": rank_lower_bound,
        "failure_stage": failure_stage,
    }


def augment_detail(df: pd.DataFrame) -> pd.DataFrame:
    derived = pd.DataFrame([derive_row(row) for _, row in df.iterrows()])
    out = df.copy()
    for col in derived.columns:
        out[col] = derived[col].values
    return out


def summarize_detail(df: pd.DataFrame) -> pd.DataFrame:
    status = df["status"] if "status" in df.columns else pd.Series("ok", index=df.index)
    ok = df[status.astype(str) == "ok"].copy()
    if ok.empty:
        return pd.DataFrame()

    group_cols = [
        "model", "function", "screen_mode", "dimension", "samples", "noise",
        "grid", "k", "width_hidden", "lamb", "steps", "opt", "update_grid",
        "interaction_method",
    ]
    group_cols = [c for c in group_cols if c in ok.columns]

    numeric_cols = [
        "train_mse", "test_mse", "effective_dim",
        "screen_contains_all_true_vars", "screen_true_var_recall",
        "screen_contains_all_interaction_endpoints", "screen_interaction_endpoint_recall",
        "screen_contains_true_interactions", "screen_endpoint_recall",
        "screen_contains_all_endpoints",
        "explain_contains_all_interaction_endpoints", "explain_interaction_endpoint_recall",
        "selected_interaction_endpoint_recall", "selected_interaction_contains_all_endpoints",
        "variable_f1", "variable_auroc", "variable_auprc",
        "interaction_f1", "true_interaction_score_mean", "max_nontrue_interaction_score",
        "true_interaction_mean_score_margin", "true_interaction_beats_all_false",
        "true_interaction_rank_lower_bound",
    ]
    numeric_cols = [c for c in numeric_cols if c in ok.columns]
    for col in numeric_cols:
        ok[col] = pd.to_numeric(ok[col], errors="coerce")

    agg = {}
    mean_std_cols = {
        "train_mse", "test_mse", "variable_f1", "variable_auroc", "variable_auprc",
        "interaction_f1", "true_interaction_score_mean", "max_nontrue_interaction_score",
        "true_interaction_mean_score_margin", "true_interaction_rank_lower_bound",
    }
    for col in numeric_cols:
        agg[col] = ["mean", "std"] if col in mean_std_cols else ["mean"]

    summary = ok.groupby(group_cols, dropna=False).agg(agg).reset_index()
    summary.columns = ["_".join(str(x) for x in col if x != "").rstrip("_") for col in summary.columns]

    counts = ok.groupby(group_cols, dropna=False).size().reset_index(name="num_rows")
    summary = summary.merge(counts, on=group_cols, how="left")

    if "failure_stage" in ok.columns:
        stage_counts = (
            ok.groupby(group_cols + ["failure_stage"], dropna=False)
            .size()
            .unstack(fill_value=0)
            .reset_index()
        )
        stage_cols = [c for c in stage_counts.columns if c not in group_cols]
        stage_counts = stage_counts.rename(columns={c: f"failure_stage_{c}_count" for c in stage_cols})
        summary = summary.merge(stage_counts, on=group_cols, how="left")

    status = df["status"] if "status" in df.columns else pd.Series("ok", index=df.index)
    failed = df[status.astype(str) != "ok"]
    if failed.empty:
        summary["num_failed"] = 0
    else:
        failed_counts = failed.groupby(group_cols, dropna=False).size().reset_index(name="num_failed")
        summary = summary.merge(failed_counts, on=group_cols, how="left")
        summary["num_failed"] = summary["num_failed"].fillna(0).astype(int)

    return summary


def summary_path_for_detail(detail_path: Path, summary_dir: Path) -> Path:
    return summary_dir / detail_path.name.replace("_detail.csv", "_summary.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detail_dir", default="results/hard_regime/details")
    parser.add_argument("--summary_dir", default="results/hard_regime/summaries")
    parser.add_argument("--combined_out", default="results/hard_regime/paper_figures/hard_regime_augmented_summary.csv")
    parser.add_argument("--write_details", action="store_true")
    args = parser.parse_args()

    detail_dir = Path(args.detail_dir)
    summary_dir = Path(args.summary_dir)
    summary_dir.mkdir(parents=True, exist_ok=True)

    combined = []
    for detail_path in sorted(detail_dir.glob("*_detail.csv")):
        df = pd.read_csv(detail_path)
        augmented = augment_detail(df)
        if args.write_details:
            augmented.to_csv(detail_path, index=False)

        summary = summarize_detail(augmented)
        summary_path = summary_path_for_detail(detail_path, summary_dir)
        summary.to_csv(summary_path, index=False)
        combined.append(summary)
        print(f"[updated] {summary_path}")

    if combined:
        combined_df = pd.concat(combined, ignore_index=True)
        out = Path(args.combined_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        combined_df.to_csv(out, index=False)
        print(f"[saved] {out}")


if __name__ == "__main__":
    main()
