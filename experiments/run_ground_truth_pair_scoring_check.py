from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Callable, Dict, Sequence, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import evaluate_synthetic_function, make_synthetic
from experiments.run_tuned_kan_recovery import canonical_pairs, evaluate_interaction_recovery


Pair = Tuple[int, int]


def fill_support(true_vars: Sequence[int], dimension: int, top_m: int) -> np.ndarray:
    selected: list[int] = []
    seen: set[int] = set()
    for value in true_vars:
        value = int(value)
        if value not in seen and len(selected) < top_m:
            selected.append(value)
            seen.add(value)
    for value in range(dimension):
        if len(selected) >= top_m:
            break
        if value not in seen:
            selected.append(value)
            seen.add(value)
    return np.asarray(sorted(selected), dtype=int)


def make_local_formula_predictor(
    function_name: str,
    support: np.ndarray,
    dimension: int,
    target_mean: float,
    target_std: float,
) -> Callable[[np.ndarray], np.ndarray]:
    def predict(X_local: np.ndarray) -> np.ndarray:
        X_full = np.zeros((X_local.shape[0], dimension), dtype=np.float32)
        X_full[:, support] = X_local.astype(np.float32)
        y, _ = evaluate_synthetic_function(function_name, X_full)
        return ((y.reshape(-1) - target_mean) / target_std).astype(float)

    return predict


def finite_difference_scores(
    predict: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    *,
    points: int,
    h: float,
) -> Dict[Pair, float]:
    d = X.shape[1]
    base = X[: min(points, len(X))].copy()
    f0 = predict(base)
    scores: Dict[Pair, float] = {}
    for i, j in itertools.combinations(range(d), 2):
        Xi = base.copy()
        Xj = base.copy()
        Xij = base.copy()
        Xi[:, i] += h
        Xj[:, j] += h
        Xij[:, i] += h
        Xij[:, j] += h
        mixed = (predict(Xij) - predict(Xi) - predict(Xj) + f0) / (h**2)
        scores[(i, j)] = float(np.mean(np.abs(mixed)))
    return scores


def anova_scores(
    predict: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    *,
    points: int,
    background: int,
    score: str,
) -> Dict[Pair, float]:
    d = X.shape[1]
    base = X[: min(points, len(X))].copy()
    bg = X[: min(background, len(X))].copy()
    f_mean = float(np.mean(predict(bg)))
    scores: Dict[Pair, float] = {}
    for i, j in itertools.combinations(range(d), 2):
        comps = []
        for row in base:
            Xij = bg.copy()
            Xi = bg.copy()
            Xj = bg.copy()
            Xij[:, i] = row[i]
            Xij[:, j] = row[j]
            Xi[:, i] = row[i]
            Xj[:, j] = row[j]
            comps.append(float(np.mean(predict(Xij)) - np.mean(predict(Xi)) - np.mean(predict(Xj)) + f_mean))
        comps_arr = np.asarray(comps, dtype=float)
        if score == "abs":
            scores[(i, j)] = float(np.mean(np.abs(comps_arr)))
        elif score == "var":
            scores[(i, j)] = float(np.var(comps_arr))
        else:
            raise ValueError(f"Unknown score={score!r}")
    return scores


def normalize(scores: Dict[Pair, float]) -> Dict[Pair, float]:
    if not scores:
        return {}
    max_value = max(max(float(value), 0.0) for value in scores.values())
    if max_value <= 0:
        return {pair: 0.0 for pair in scores}
    return {pair: max(float(value), 0.0) / max_value for pair, value in scores.items()}


def hybrid_scores(fd: Dict[Pair, float], anova_abs: Dict[Pair, float]) -> Dict[Pair, float]:
    keys = sorted(set(fd) | set(anova_abs))
    fd_norm = normalize(fd)
    anova_norm = normalize(anova_abs)
    return {pair: 0.5 * (fd_norm.get(pair, 0.0) + anova_norm.get(pair, 0.0)) for pair in keys}


def local_to_full(scores: Dict[Pair, float], support: np.ndarray, dimension: int) -> Dict[Pair, float]:
    out: Dict[Pair, float] = {}
    for (i, j), value in scores.items():
        out[tuple(sorted((int(support[i]), int(support[j]))))] = float(value)
    for i, j in itertools.combinations(range(dimension), 2):
        out.setdefault((i, j), 0.0)
    return out


def serialize_top_scores(scores: Dict[Pair, float], limit: int = 10) -> str:
    top = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:limit]
    return json.dumps([{"pair": [int(i), int(j)], "score": float(value)} for (i, j), value in top])


def run_one(args, function_name: str, seed: int) -> list[dict]:
    data = make_synthetic(
        function_name=function_name,
        n_train=args.samples,
        n_test=args.test_samples,
        d=args.dimension,
        seed=seed,
        noise=0.0,
        standardize_target=True,
    )
    gt = data["ground_truth"]
    true_vars = tuple(int(v) for v in gt.active_variables)
    true_interactions = canonical_pairs(gt.interactions)
    support = fill_support(true_vars, args.dimension, args.top_m)
    X_local = data["X_test"][:, support]
    predict = make_local_formula_predictor(
        function_name,
        support,
        args.dimension,
        float(data["target_mean"]),
        float(data["target_std"]),
    )
    local_fd = finite_difference_scores(predict, X_local, points=args.fd_points, h=args.fd_h)
    local_anova_abs = anova_scores(
        predict,
        X_local,
        points=args.anova_points,
        background=args.anova_background,
        score="abs",
    )
    local_anova_var = anova_scores(
        predict,
        X_local,
        points=args.anova_points,
        background=args.anova_background,
        score="var",
    )
    methods = {
        "fd": local_fd,
        "anova_abs": local_anova_abs,
        "anova_var": local_anova_var,
        "fd_anova_hybrid": hybrid_scores(local_fd, local_anova_abs),
    }
    rows = []
    for method, local_scores in methods.items():
        full_scores = local_to_full(local_scores, support, args.dimension)
        row = {
            "function": function_name,
            "seed": seed,
            "samples": args.samples,
            "test_samples": args.test_samples,
            "dimension": args.dimension,
            "top_m": args.top_m,
            "support": support.tolist(),
            "true_variables": list(true_vars),
            "true_interactions": list(true_interactions),
            "formula": gt.formula,
            "pair_score_method": method,
            "status": "ok",
            "pair_scores_top10": serialize_top_scores(full_scores, limit=10),
        }
        row.update(evaluate_interaction_recovery(full_scores, true_interactions))
        rows.append(row)
    return rows


def summarize(detail: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["function", "pair_score_method", "samples", "dimension", "top_m"]
    numeric_cols = [
        "interaction_f1",
        "true_interaction_rank_mean",
        "true_interaction_mean_score_margin",
        "true_interaction_beats_all_false",
    ]
    ok = detail[detail["status"].eq("ok")].copy()
    for col in numeric_cols:
        ok[col] = pd.to_numeric(ok[col], errors="coerce")
    agg = ok.groupby(group_cols, dropna=False)[numeric_cols].agg(["mean", "std"]).reset_index()
    agg.columns = ["_".join(str(v) for v in col if v != "").rstrip("_") for col in agg.columns]
    counts = ok.groupby(group_cols, dropna=False).size().reset_index(name="num_runs")
    return agg.merge(counts, on=group_cols, how="left")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pair scorers directly on ground-truth formulas.")
    parser.add_argument(
        "--functions",
        nargs="+",
        default=[
            "core_interaction_c01",
            "core_interaction_c025",
            "core_interaction_c05",
            "core_interaction_c1",
            "feynman_energy",
            "feynman_gravity",
            "feynman_coulomb",
            "feynman_damped_wave",
        ],
    )
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--test_samples", type=int, default=4096)
    parser.add_argument("--dimension", type=int, default=100)
    parser.add_argument("--top_m", type=int, default=4)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--fd_points", type=int, default=256)
    parser.add_argument("--fd_h", type=float, default=1e-2)
    parser.add_argument("--anova_points", type=int, default=96)
    parser.add_argument("--anova_background", type=int, default=96)
    parser.add_argument("--out_dir", default="results/formula_aware_pair_scoring/ground_truth_check_20260526")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for function_name in args.functions:
        for seed in args.seeds:
            print(f"[GROUND-TRUTH] function={function_name} seed={seed}", flush=True)
            rows.extend(run_one(args, function_name, int(seed)))
    detail = pd.DataFrame(rows)
    summary = summarize(detail)
    detail.to_csv(out_dir / "ground_truth_pair_scoring_detail.csv", index=False)
    summary.to_csv(out_dir / "ground_truth_pair_scoring_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
