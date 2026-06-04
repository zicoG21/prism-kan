from __future__ import annotations

import argparse
import itertools
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes, load_wine
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from src.data import make_synthetic


Pair = Tuple[int, int]


def load_real_covariates(name: str) -> np.ndarray:
    if name == "diabetes":
        X = load_diabetes().data.astype(np.float32)
    elif name == "breast_cancer":
        X = load_breast_cancer().data.astype(np.float32)
    elif name == "wine":
        X = load_wine().data.astype(np.float32)
    else:
        raise ValueError(f"Unknown semisynthetic covariate dataset {name!r}")
    return StandardScaler().fit_transform(X).astype(np.float32)


def make_semisynthetic_treegate_data(
    dataset: str,
    n_train: int,
    n_test: int,
    c: float,
    noise: float,
    seed: int,
) -> dict[str, object]:
    X_pool = load_real_covariates(dataset)
    rng = np.random.default_rng(int(seed))
    n_total = int(n_train) + int(n_test)
    idx = rng.choice(len(X_pool), size=n_total, replace=n_total > len(X_pool))
    Z = np.tanh(X_pool[idx]).astype(np.float32)
    y_clean = (np.sin(np.pi * Z[:, 0]) + Z[:, 1] ** 2 + float(c) * Z[:, 2] * Z[:, 3]).astype(np.float32)
    if noise > 0:
        y_clean_std = float(np.std(y_clean)) or 1.0
        y = y_clean + rng.normal(0.0, float(noise) * y_clean_std, size=n_total).astype(np.float32)
    else:
        y = y_clean
    X_train = Z[:n_train]
    X_test = Z[n_train:]
    y_train = y[:n_train].reshape(-1, 1)
    y_test = y[n_train:].reshape(-1, 1)
    mean = float(y_train.mean())
    std = float(y_train.std()) or 1.0
    gt = SimpleNamespace(active_variables=(0, 1, 2, 3), interactions=((2, 3),))
    return {
        "X_train": X_train.astype(np.float32),
        "y_train": ((y_train - mean) / std).astype(np.float32),
        "X_test": X_test.astype(np.float32),
        "y_test": ((y_test - mean) / std).astype(np.float32),
        "ground_truth": gt,
    }


def make_treegate_data(args: argparse.Namespace, function_name: str, seed: int) -> dict[str, object]:
    if function_name.startswith("semisynthetic_"):
        dataset = function_name.removeprefix("semisynthetic_")
        return make_semisynthetic_treegate_data(
            dataset=dataset,
            n_train=args.samples,
            n_test=args.test_samples,
            c=args.semisynthetic_c,
            noise=args.noise,
            seed=seed,
        )
    return make_synthetic(
        function_name=function_name,
        n_train=args.samples,
        n_test=args.test_samples,
        d=args.dimension,
        noise=args.noise,
        seed=seed,
        standardize_target=True,
        nuisance_correlation=args.nuisance_correlation,
        n_correlated_proxies=args.n_correlated_proxies,
    )


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


def _normalize(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    finite = np.isfinite(scores)
    if not finite.any():
        return np.zeros_like(scores, dtype=float)
    lo = float(np.nanmin(scores[finite]))
    hi = float(np.nanmax(scores[finite]))
    if hi <= lo:
        return np.zeros_like(scores, dtype=float)
    return (scores - lo) / (hi - lo)


def tree_path_cooccurrence(model, dimension: int) -> Tuple[np.ndarray, Dict[Pair, float]]:
    """Extract path co-occurrence scores from a fitted tree ensemble.

    A path contributes to every unordered pair of variables that appears along
    that root-to-leaf path.  The score is weighted by the fraction of training
    samples reaching the leaf.  This is a tree-native candidate-generator score,
    not a verifier: it says that a pair is repeatedly used in the same rule
    context, not that the fitted function has a declared interaction component.
    """
    endpoint_sum = np.zeros(dimension, dtype=float)
    pair_scores: Dict[Pair, float] = {}

    for estimator in model.estimators_:
        tree = estimator.tree_
        root_count = max(float(tree.n_node_samples[0]), 1.0)

        def visit(node: int, path_features: Tuple[int, ...]) -> None:
            feature = int(tree.feature[node])
            left = int(tree.children_left[node])
            right = int(tree.children_right[node])

            if left == right:
                uniq = tuple(sorted(set(f for f in path_features if 0 <= f < dimension)))
                if not uniq:
                    return
                weight = float(tree.n_node_samples[node]) / root_count
                for f in uniq:
                    endpoint_sum[f] += weight
                for i, j in itertools.combinations(uniq, 2):
                    pair = (int(i), int(j))
                    pair_scores[pair] = pair_scores.get(pair, 0.0) + weight
                return

            next_path = path_features
            if feature >= 0:
                next_path = path_features + (feature,)
            visit(left, next_path)
            visit(right, next_path)

        visit(0, tuple())

    scale = max(len(model.estimators_), 1)
    endpoint_sum /= scale
    pair_scores = {pair: score / scale for pair, score in pair_scores.items()}
    return endpoint_sum, pair_scores


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
    data = make_treegate_data(args, function_name, seed)
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]
    gt = data["ground_truth"]

    true_vars = tuple(int(v) for v in gt.active_variables)
    true_pairs = canonical_pairs(gt.interactions)
    true_endpoints = set(interaction_endpoints(true_pairs))
    true_pair_set = set(true_pairs)

    forest_cls = ExtraTreesRegressor if args.forest_type == "extra" else RandomForestRegressor
    rf = forest_cls(
        n_estimators=args.rf_trees,
        random_state=seed,
        n_jobs=args.n_jobs,
        max_depth=args.rf_max_depth,
        min_samples_leaf=args.rf_min_samples_leaf,
    )
    rf.fit(X_train, y_train.reshape(-1))
    test_mse = mse(rf.predict(X_test), y_test)

    feature_scores = np.asarray(rf.feature_importances_, dtype=float)
    path_endpoint_scores, path_pair_scores = tree_path_cooccurrence(rf, args.dimension)

    path_endpoint_from_pairs = np.zeros(args.dimension, dtype=float)
    for (i, j), score in path_pair_scores.items():
        path_endpoint_from_pairs[i] = max(path_endpoint_from_pairs[i], score)
        path_endpoint_from_pairs[j] = max(path_endpoint_from_pairs[j], score)

    if args.gate_score == "feature_importance":
        endpoint_scores = feature_scores
    elif args.gate_score == "path_endpoint":
        endpoint_scores = path_endpoint_scores
    elif args.gate_score == "path_pair_endpoint":
        endpoint_scores = path_endpoint_from_pairs
    elif args.gate_score == "hybrid":
        endpoint_scores = _normalize(feature_scores) + _normalize(path_endpoint_from_pairs)
    else:
        raise ValueError(f"Unknown gate_score={args.gate_score}")

    gate_size = min(args.gate_size, args.dimension)
    gated_endpoints = tuple(int(i) for i in np.argsort(-endpoint_scores)[:gate_size])
    gated_endpoint_set = set(gated_endpoints)
    candidate_pairs = {tuple(sorted(p)) for p in itertools.combinations(gated_endpoints, 2)}

    direct_pairs_added = []
    if args.direct_pair_budget > 0 and path_pair_scores:
        ranked_path_pairs = sorted(path_pair_scores.items(), key=lambda kv: kv[1], reverse=True)
        direct_pairs_added = [pair for pair, _ in ranked_path_pairs[: args.direct_pair_budget]]
        candidate_pairs.update(direct_pairs_added)
    candidate_pairs = sorted(candidate_pairs)

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
        "nuisance_correlation": float(args.nuisance_correlation),
        "n_correlated_proxies": int(args.n_correlated_proxies),
        "forest_type": args.forest_type,
        "gate_score": args.gate_score,
        "rf_trees": args.rf_trees,
        "gate_size": gate_size,
        "direct_pair_budget": args.direct_pair_budget,
        "verify_points": args.verify_points,
        "test_mse": test_mse,
        "true_variables": list(true_vars),
        "true_pairs": list(true_pairs),
        "true_endpoints": sorted(true_endpoints),
        "gated_endpoints": list(gated_endpoints),
        "top_path_pairs": [(int(i), int(j), float(s)) for (i, j), s in sorted(path_pair_scores.items(), key=lambda kv: kv[1], reverse=True)[:20]],
        "direct_pairs_added": [(int(i), int(j)) for i, j in direct_pairs_added[:50]],
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
    keys = [
        "function",
        "samples",
        "dimension",
        "noise",
        "nuisance_correlation",
        "n_correlated_proxies",
        "forest_type",
        "gate_score",
        "gate_size",
        "direct_pair_budget",
        "verify_points",
        "rf_trees",
    ]
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
    parser.add_argument("--nuisance-correlation", type=float, default=0.0)
    parser.add_argument("--n-correlated-proxies", type=int, default=0)
    parser.add_argument("--semisynthetic-c", type=float, default=0.25)
    parser.add_argument("--seeds", type=str, default="0-9")
    parser.add_argument("--gate-size", type=int, default=20)
    parser.add_argument("--verify-points", type=int, default=256)
    parser.add_argument("--rf-trees", type=int, default=500)
    parser.add_argument("--forest-type", choices=["rf", "extra"], default="rf")
    parser.add_argument(
        "--gate-score",
        choices=["feature_importance", "path_endpoint", "path_pair_endpoint", "hybrid"],
        default="feature_importance",
    )
    parser.add_argument("--direct-pair-budget", type=int, default=0)
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
                    "forest_type": args.forest_type,
                    "gate_score": args.gate_score,
                    "gate_size": args.gate_size,
                    "direct_pair_budget": args.direct_pair_budget,
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
