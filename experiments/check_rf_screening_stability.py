from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import make_synthetic
from experiments.run_tuned_kan_recovery import canonical_pairs, interaction_endpoints, support_stats


def rf_scores(X: np.ndarray, y: np.ndarray, seed: int, trees: int, n_jobs: int) -> np.ndarray:
    rf = RandomForestRegressor(
        n_estimators=trees,
        random_state=seed,
        n_jobs=n_jobs,
        min_samples_leaf=2,
    )
    rf.fit(X, y.reshape(-1))
    return np.asarray(rf.feature_importances_, dtype=float)


def endpoint_ranks(scores: np.ndarray, endpoints: set[int]) -> dict[str, float]:
    order = list(np.argsort(-scores))
    ranks = [order.index(int(v)) + 1 for v in endpoints if int(v) in order]
    return {
        "endpoint_best_rank": float(np.min(ranks)) if ranks else np.nan,
        "endpoint_worst_rank": float(np.max(ranks)) if ranks else np.nan,
        "endpoint_mean_rank": float(np.mean(ranks)) if ranks else np.nan,
    }


def run_one(function_name: str, samples: int, dimension: int, seed: int, trees: int, top_m: int, n_jobs: int) -> dict:
    data = make_synthetic(
        function_name=function_name,
        n_train=samples,
        n_test=512,
        d=dimension,
        noise=0.0,
        seed=seed,
        standardize_target=True,
    )
    gt = data["ground_truth"]
    true_vars = tuple(int(v) for v in gt.active_variables)
    true_interactions = canonical_pairs(gt.interactions)
    endpoints = set(interaction_endpoints(true_interactions))
    scores = rf_scores(data["X_train"], data["y_train"], seed=seed, trees=trees, n_jobs=n_jobs)
    selected = np.array(sorted(np.argsort(-scores)[:top_m]), dtype=int)
    row = {
        "function": function_name,
        "samples": samples,
        "dimension": dimension,
        "seed": seed,
        "trees": trees,
        "top_m": top_m,
        "selected_features": selected.tolist(),
        "true_variables": list(true_vars),
        "true_interactions": list(true_interactions),
        "top12": np.argsort(-scores)[: min(12, dimension)].astype(int).tolist(),
        "endpoint_scores": {int(v): float(scores[int(v)]) for v in endpoints},
        "max_noise_score": float(np.max([scores[j] for j in range(dimension) if j not in set(true_vars)])),
    }
    row.update(support_stats(selected, true_vars, true_interactions))
    row.update(endpoint_ranks(scores, endpoints))
    return row


def summarize(detail: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        "screen_contains_all_true_vars",
        "screen_true_var_recall",
        "screen_contains_all_interaction_endpoints",
        "screen_interaction_endpoint_recall",
        "screen_contains_true_interactions",
        "endpoint_best_rank",
        "endpoint_worst_rank",
        "endpoint_mean_rank",
        "max_noise_score",
    ]
    for col in numeric_cols:
        if col in detail.columns:
            detail[col] = pd.to_numeric(detail[col], errors="coerce")
    group_cols = ["function", "samples", "dimension", "trees", "top_m"]
    agg = {col: ["mean", "std"] for col in numeric_cols if col in detail.columns}
    out = detail.groupby(group_cols, dropna=False).agg(agg).reset_index()
    flat_cols = []
    for col in out.columns:
        if isinstance(col, tuple):
            flat_cols.append("_".join(str(x) for x in col if x != "").rstrip("_"))
        else:
            flat_cols.append(str(col))
    out.columns = flat_cols
    counts = detail.groupby(group_cols, dropna=False).size().reset_index(name="num_runs")
    return out.merge(counts, on=group_cols, how="left")


def main() -> None:
    parser = argparse.ArgumentParser(description="RF screening-only diagnostic for interaction endpoint retention.")
    parser.add_argument("--functions", nargs="+", default=["core_interaction_c025"])
    parser.add_argument("--samples", nargs="+", type=int, default=[512, 1024])
    parser.add_argument("--dimension", type=int, default=100)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(100, 120)))
    parser.add_argument("--trees", nargs="+", type=int, default=[500, 2000])
    parser.add_argument("--top_ms", nargs="+", type=int, default=[4, 5, 6])
    parser.add_argument("--n_jobs", type=int, default=2)
    parser.add_argument("--out_dir", default="results/innovation_loop/rf_screening_diagnostic")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for fn in args.functions:
        for n in args.samples:
            for trees in args.trees:
                for seed in args.seeds:
                    print(f"[RF] function={fn} n={n} d={args.dimension} trees={trees} seed={seed}", flush=True)
                    per_top = {}
                    # Fit once per seed/tree and reuse by computing all top_m through run_one's selected output.
                    data = make_synthetic(
                        function_name=fn,
                        n_train=n,
                        n_test=512,
                        d=args.dimension,
                        noise=0.0,
                        seed=seed,
                        standardize_target=True,
                    )
                    gt = data["ground_truth"]
                    true_vars = tuple(int(v) for v in gt.active_variables)
                    true_interactions = canonical_pairs(gt.interactions)
                    endpoints = set(interaction_endpoints(true_interactions))
                    scores = rf_scores(data["X_train"], data["y_train"], seed=seed, trees=trees, n_jobs=args.n_jobs)
                    for top_m in args.top_ms:
                        selected = np.array(sorted(np.argsort(-scores)[:top_m]), dtype=int)
                        row = {
                            "function": fn,
                            "samples": n,
                            "dimension": args.dimension,
                            "seed": seed,
                            "trees": trees,
                            "top_m": top_m,
                            "selected_features": selected.tolist(),
                            "true_variables": list(true_vars),
                            "true_interactions": list(true_interactions),
                            "top12": np.argsort(-scores)[: min(12, args.dimension)].astype(int).tolist(),
                            "endpoint_scores": {int(v): float(scores[int(v)]) for v in endpoints},
                            "max_noise_score": float(np.max([scores[j] for j in range(args.dimension) if j not in set(true_vars)])),
                        }
                        row.update(support_stats(selected, true_vars, true_interactions))
                        row.update(endpoint_ranks(scores, endpoints))
                        rows.append(row)
                    pd.DataFrame(rows).to_csv(out_dir / "rf_screening_detail.csv", index=False)

    detail = pd.DataFrame(rows)
    summary = summarize(detail)
    detail.to_csv(out_dir / "rf_screening_detail.csv", index=False)
    summary.to_csv(out_dir / "rf_screening_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"Wrote {out_dir / 'rf_screening_detail.csv'}")
    print(f"Wrote {out_dir / 'rf_screening_summary.csv'}")


if __name__ == "__main__":
    main()
