from __future__ import annotations

import argparse
import itertools
import sys
import traceback
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.data import make_synthetic


Pair = Tuple[int, int]


def canonical_pairs(pairs: Sequence[Tuple[int, int]]) -> Tuple[Pair, ...]:
    return tuple(tuple(sorted((int(i), int(j)))) for i, j in pairs)


def interaction_endpoints(pairs: Sequence[Pair]) -> Tuple[int, ...]:
    endpoints = set()
    for i, j in pairs:
        endpoints.add(int(i))
        endpoints.add(int(j))
    return tuple(sorted(endpoints))


def parse_seeds(spec: str) -> List[int]:
    seeds: List[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            seeds.extend(range(int(a), int(b) + 1))
        else:
            seeds.append(int(part))
    return sorted(set(seeds))


def mse(pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((pred.reshape(-1, 1) - y.reshape(-1, 1)) ** 2))


def batch_predict(predict_fn: Callable[[np.ndarray], np.ndarray], X: np.ndarray, batch_size: int) -> np.ndarray:
    outs = []
    for start in range(0, len(X), batch_size):
        outs.append(predict_fn(X[start:start + batch_size]).reshape(-1))
    return np.concatenate(outs, axis=0)


def candidate_pair_scores(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    X_np: np.ndarray,
    candidates: Sequence[Pair],
    points: int,
    seed: int,
    batch_size: int,
    pair_chunk_size: int,
) -> Dict[Pair, float]:
    """Pair-permutation synergy only on gated candidate pairs.

    This is the verifier part of TreeGate. It intentionally does not enumerate
    all d(d-1)/2 pairs. The gate decides which endpoints are worth verifying;
    this scorer verifies only pairs induced by those endpoints.
    """
    if not candidates:
        return {}

    n = min(points, X_np.shape[0])
    X = X_np[:n].copy()
    rng = np.random.default_rng(seed + 31717)
    base = batch_predict(predict_fn, X, batch_size=batch_size)

    involved = sorted({v for pair in candidates for v in pair})
    perms = {j: rng.permutation(n) for j in involved}

    single_delta: Dict[int, float] = {}
    single_blocks = []
    single_features = []
    for j in involved:
        X_j = X.copy()
        X_j[:, j] = X_j[perms[j], j]
        single_blocks.append(X_j)
        single_features.append(j)

    X_single = np.vstack(single_blocks)
    pred_single = batch_predict(predict_fn, X_single, batch_size=batch_size)
    pred_single = pred_single.reshape(len(single_blocks), n)
    for block_idx, j in enumerate(single_features):
        single_delta[j] = float(np.mean((pred_single[block_idx] - base) ** 2))

    scores: Dict[Pair, float] = {}
    for start in range(0, len(candidates), pair_chunk_size):
        chunk = list(candidates[start:start + pair_chunk_size])
        joint_blocks = []
        for i, j in chunk:
            X_ij = X.copy()
            X_ij[:, i] = X_ij[perms[i], i]
            X_ij[:, j] = X_ij[perms[j], j]
            joint_blocks.append(X_ij)

        X_joint = np.vstack(joint_blocks)
        pred_joint = batch_predict(predict_fn, X_joint, batch_size=batch_size)
        pred_joint = pred_joint.reshape(len(chunk), n)
        for block_idx, (i, j) in enumerate(chunk):
            joint = float(np.mean((pred_joint[block_idx] - base) ** 2))
            scores[(i, j)] = abs(joint - single_delta[i] - single_delta[j])
    return scores


def summarize_pair_scores(pair_scores: Dict[Pair, float], true_pairs: Sequence[Pair]) -> Dict:
    true_set = set(canonical_pairs(true_pairs))
    ranked = sorted(pair_scores.items(), key=lambda kv: kv[1], reverse=True)
    selected_topq = {pair for pair, _ in ranked[: max(1, len(true_set))]}

    if not true_set:
        return {
            "verified_pair_topq_success": np.nan,
            "verified_pair_recall": np.nan,
            "true_pair_rank_best": np.nan,
            "true_pair_score_max": np.nan,
            "max_false_pair_score": np.nan,
            "candidate_pair_top20": [],
        }

    true_ranks = []
    true_scores = []
    false_scores = []
    for idx, (pair, score) in enumerate(ranked, start=1):
        if pair in true_set:
            true_ranks.append(idx)
            true_scores.append(float(score))
        else:
            false_scores.append(float(score))

    return {
        "verified_pair_topq_success": int(bool(true_set & selected_topq)),
        "verified_pair_recall": len(true_set & {pair for pair, _ in ranked}) / len(true_set),
        "true_pair_rank_best": min(true_ranks) if true_ranks else np.nan,
        "true_pair_score_max": max(true_scores) if true_scores else np.nan,
        "max_false_pair_score": max(false_scores) if false_scores else np.nan,
        "candidate_pair_top20": [(int(i), int(j), float(s)) for (i, j), s in ranked[:20]],
    }


def run_one(args: argparse.Namespace, function_name: str, seed: int) -> Dict:
    data = make_synthetic(
        function_name=function_name,
        n_train=args.samples,
        n_test=args.test_samples,
        d=args.dimension,
        noise=args.noise,
        seed=seed,
        standardize_target=True,
    )
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]
    gt = data["ground_truth"]

    true_vars = tuple(int(v) for v in gt.active_variables)
    true_pairs = canonical_pairs(gt.interactions)
    true_endpoints = set(interaction_endpoints(true_pairs))
    true_pair_set = set(true_pairs)

    rf = RandomForestRegressor(
        n_estimators=args.rf_trees,
        random_state=seed,
        n_jobs=args.n_jobs,
        max_depth=args.rf_max_depth,
        min_samples_leaf=args.rf_min_samples_leaf,
    )
    rf.fit(X_train, y_train.reshape(-1))
    test_mse = mse(rf.predict(X_test), y_test)

    endpoint_scores = np.asarray(rf.feature_importances_, dtype=float)
    gate_size = min(args.gate_size, args.dimension)
    gated_endpoints = tuple(int(i) for i in np.argsort(-endpoint_scores)[:gate_size])
    gated_endpoint_set = set(gated_endpoints)
    candidate_pairs = [tuple(sorted(p)) for p in itertools.combinations(gated_endpoints, 2)]

    pair_scores = candidate_pair_scores(
        predict_fn=rf.predict,
        X_np=X_test,
        candidates=candidate_pairs,
        points=args.verify_points,
        seed=seed,
        batch_size=args.batch_size,
        pair_chunk_size=args.pair_chunk_size,
    )

    full_pair_count = args.dimension * (args.dimension - 1) // 2
    pair_summary = summarize_pair_scores(pair_scores, true_pairs)

    return {
        "status": "ok",
        "error": "",
        "function": function_name,
        "seed": seed,
        "samples": args.samples,
        "test_samples": args.test_samples,
        "dimension": args.dimension,
        "noise": args.noise,
        "rf_trees": args.rf_trees,
        "gate_size": gate_size,
        "verify_points": args.verify_points,
        "test_mse": test_mse,
        "true_variables": list(true_vars),
        "true_pairs": list(true_pairs),
        "true_endpoints": sorted(true_endpoints),
        "gated_endpoints": list(gated_endpoints),
        "any_pair_endpoint_recall": len(true_endpoints & gated_endpoint_set) / len(true_endpoints) if true_endpoints else np.nan,
        "all_pair_endpoints_in_gate": int(true_endpoints.issubset(gated_endpoint_set)) if true_endpoints else np.nan,
        "candidate_pair_count": len(candidate_pairs),
        "full_pair_count": full_pair_count,
        "candidate_fraction": len(candidate_pairs) / full_pair_count if full_pair_count else np.nan,
        "nominal_pair_speedup": full_pair_count / len(candidate_pairs) if candidate_pairs else np.inf,
        "true_pair_in_candidates": int(any(pair in set(candidate_pairs) for pair in true_pair_set)) if true_pair_set else np.nan,
        **pair_summary,
    }


def append_rows(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if path.exists():
        old_cols = list(pd.read_csv(path, nrows=0).columns)
        for col in old_cols:
            if col not in df.columns:
                df[col] = np.nan
        df = df[old_cols]
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        df.to_csv(path, index=False)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["function", "samples", "dimension", "noise", "gate_size", "verify_points", "rf_trees"]
    rows = []
    for key, g in df.groupby(keys, dropna=False):
        row = dict(zip(keys, key))
        row.update({
            "runs": int(len(g)),
            "mean_test_mse": float(g["test_mse"].mean()),
            "endpoint_anypair_recall_mean": float(g["any_pair_endpoint_recall"].mean()),
            "all_pair_endpoints_in_gate_rate": float(g["all_pair_endpoints_in_gate"].mean()),
            "true_pair_in_candidates_rate": float(g["true_pair_in_candidates"].mean()),
            "verified_pair_topq_success_rate": float(g["verified_pair_topq_success"].mean()),
            "median_candidate_pair_count": float(g["candidate_pair_count"].median()),
            "median_nominal_pair_speedup": float(g["nominal_pair_speedup"].replace([np.inf, -np.inf], np.nan).median()),
            "median_true_pair_rank_best": float(g["true_pair_rank_best"].median(skipna=True)),
        })
        rows.append(row)
    return pd.DataFrame(rows).sort_values(keys)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tree-gated any-pair endpoint screen plus candidate pair verifier.")
    parser.add_argument("--functions", nargs="+", required=True)
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--test-samples", type=int, default=2048)
    parser.add_argument("--dimension", type=int, default=100)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--seeds", type=str, default="0-9")
    parser.add_argument("--gate-size", type=int, default=20)
    parser.add_argument("--verify-points", type=int, default=256)
    parser.add_argument("--rf-trees", type=int, default=500)
    parser.add_argument("--rf-max-depth", type=int, default=None)
    parser.add_argument("--rf-min-samples-leaf", type=int, default=2)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--pair-chunk-size", type=int, default=128)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    detail_path = out_dir / "treegate_pair_screen_detail.csv"
    summary_path = out_dir / "treegate_pair_screen_summary.csv"

    completed = set()
    if args.resume and detail_path.exists():
        existing = pd.read_csv(detail_path)
        completed = set(zip(existing["function"].astype(str), existing["seed"].astype(int)))
        print(f"[resume] completed rows={len(completed)}", flush=True)

    rows = []
    for function_name in args.functions:
        for seed in parse_seeds(args.seeds):
            key = (function_name, int(seed))
            if key in completed:
                print(f"[resume] skip {key}", flush=True)
                continue
            try:
                print(f"Running TreeGate label={args.label} function={function_name} seed={seed}", flush=True)
                rows.append(run_one(args, function_name, seed))
                append_rows(detail_path, [rows[-1]])
            except Exception as exc:
                failed = {
                    "status": "failed",
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                    "function": function_name,
                    "seed": seed,
                    "samples": args.samples,
                    "test_samples": args.test_samples,
                    "dimension": args.dimension,
                    "noise": args.noise,
                    "rf_trees": args.rf_trees,
                    "gate_size": args.gate_size,
                    "verify_points": args.verify_points,
                }
                append_rows(detail_path, [failed])
                print(f"[WARN] failed function={function_name} seed={seed}: {exc}", flush=True)

    df = pd.read_csv(detail_path)
    ok = df[df["status"].eq("ok")].copy()
    if len(ok):
        summary = summarize(ok)
        summary.to_csv(summary_path, index=False)
        print(summary.to_string(index=False), flush=True)
    print(f"Wrote {detail_path}", flush=True)
    print(f"Wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
