from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes, load_wine
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.preprocessing import SplineTransformer, StandardScaler

from src.data import make_synthetic


Pair = tuple[int, int]


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


def make_semisynthetic_transfer_data(
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


def make_transfer_data(args: argparse.Namespace, function_name: str, seed: int) -> dict[str, object]:
    if function_name.startswith("semisynthetic_"):
        dataset = function_name.removeprefix("semisynthetic_")
        return make_semisynthetic_transfer_data(
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


def canonical_pairs(pairs: Iterable[tuple[int, int]]) -> set[Pair]:
    return {tuple(sorted((int(i), int(j)))) for i, j in pairs}


def f1_from_sets(pred: set, true: set) -> tuple[float, float, float]:
    if not true:
        return np.nan, np.nan, np.nan
    if not pred:
        return 0.0, 0.0, 0.0
    tp = len(pred & true)
    precision = tp / len(pred)
    recall = tp / len(true)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def pair_design(X: np.ndarray) -> tuple[np.ndarray, list[Pair]]:
    pairs = list(itertools.combinations(range(X.shape[1]), 2))
    Z = np.empty((X.shape[0], len(pairs)), dtype=np.float32)
    for k, (i, j) in enumerate(pairs):
        Z[:, k] = X[:, i] * X[:, j]
    return Z, [(int(i), int(j)) for i, j in pairs]


def rank_pair_scores(
    pairs: list[Pair],
    scores: np.ndarray,
    true_pairs: set[Pair],
    pair_budget: int,
) -> dict[str, object]:
    if not true_pairs:
        return {
            "selected_pairs": [],
            "selected_pair_endpoints": [],
            "pair_precision": np.nan,
            "pair_recall": np.nan,
            "pair_f1": np.nan,
            "pair_success_all_true_at_budget": np.nan,
            "pair_success_any_true_at_budget": np.nan,
            "endpoint_success_from_pairs": np.nan,
            "true_pair_rank_best": np.nan,
            "true_pair_rank_worst": np.nan,
            "true_pair_score_min": np.nan,
            "max_false_pair_score": np.nan,
            "true_pair_margin_min": np.nan,
            "top_pair": None,
        }

    order = np.argsort(-np.asarray(scores, dtype=float))
    rank_by_pair = {tuple(sorted(pairs[int(idx)])): rank + 1 for rank, idx in enumerate(order)}
    score_by_pair = {tuple(sorted(pair)): float(scores[k]) for k, pair in enumerate(pairs)}
    budget = max(1, int(pair_budget))
    top_pairs = [tuple(sorted(pairs[int(idx)])) for idx in order[:budget]]
    selected_pairs = set(top_pairs[: max(1, len(true_pairs))])
    precision, recall, f1 = f1_from_sets(selected_pairs, true_pairs)

    true_ranks = [rank_by_pair.get(pair, np.inf) for pair in true_pairs]
    true_scores = [score_by_pair.get(pair, np.nan) for pair in true_pairs]
    false_scores = [score for pair, score in score_by_pair.items() if pair not in true_pairs]
    top_endpoint_support = {v for pair in top_pairs for v in pair}
    true_endpoints = {v for pair in true_pairs for v in pair}

    true_score_min = float(np.nanmin(true_scores)) if true_scores else np.nan
    max_false = float(np.nanmax(false_scores)) if false_scores else np.nan
    margin = true_score_min - max_false if np.isfinite(true_score_min) and np.isfinite(max_false) else np.nan

    return {
        "selected_pairs": sorted(selected_pairs),
        "selected_pair_endpoints": sorted(top_endpoint_support),
        "pair_precision": precision,
        "pair_recall": recall,
        "pair_f1": f1,
        "pair_success_all_true_at_budget": int(all(rank <= budget for rank in true_ranks)),
        "pair_success_any_true_at_budget": int(any(rank <= budget for rank in true_ranks)),
        "endpoint_success_from_pairs": int(true_endpoints.issubset(top_endpoint_support)),
        "true_pair_rank_best": float(np.min(true_ranks)) if true_ranks else np.nan,
        "true_pair_rank_worst": float(np.max(true_ranks)) if true_ranks else np.nan,
        "true_pair_score_min": true_score_min,
        "max_false_pair_score": max_false,
        "true_pair_margin_min": float(margin),
        "top_pair": top_pairs[0] if top_pairs else None,
    }


def common_transfer_fields(row: dict[str, object], pred_threshold: float) -> dict[str, object]:
    pred = int(float(row["test_mse"]) <= float(pred_threshold))
    support = int(row.get("support_success_all_true", 0)) if not pd.isna(row.get("support_success_all_true", np.nan)) else np.nan
    endpoint = int(row.get("endpoint_success", 0)) if not pd.isna(row.get("endpoint_success", np.nan)) else np.nan
    pair = int(row.get("pair_success_all_true_at_budget", 0)) if not pd.isna(row.get("pair_success_all_true_at_budget", np.nan)) else np.nan
    return {
        "prediction_success": pred,
        "pred_to_support_failure": int(pred == 1 and support == 0) if not pd.isna(support) else np.nan,
        "pred_to_endpoint_failure": int(pred == 1 and endpoint == 0) if not pd.isna(endpoint) else np.nan,
        "pred_to_pair_failure": int(pred == 1 and pair == 0) if not pd.isna(pair) else np.nan,
        "support_to_pair_failure": int(support == 1 and pair == 0) if not pd.isna(support) and not pd.isna(pair) else np.nan,
        "pair_without_support": int(pair == 1 and support == 0) if not pd.isna(support) and not pd.isna(pair) else np.nan,
        "endpoint_without_pair": int(endpoint == 1 and pair == 0) if not pd.isna(endpoint) and not pd.isna(pair) else np.nan,
    }


def make_sparse_design(
    X_train: np.ndarray,
    X_test: np.ndarray,
    main_basis: str,
    poly_degrees: list[int],
    spline_n_knots: int,
    spline_degree: int,
) -> tuple[np.ndarray, np.ndarray, list[Pair], list[slice], StandardScaler, StandardScaler]:
    raw_scaler = StandardScaler()
    Xz_train = raw_scaler.fit_transform(X_train).astype(np.float32)
    Xz_test = raw_scaler.transform(X_test).astype(np.float32)

    main_slices: list[slice] = []
    if main_basis == "linear":
        main_train = Xz_train
        main_test = Xz_test
        for j in range(X_train.shape[1]):
            main_slices.append(slice(j, j + 1))
    elif main_basis == "polynomial":
        blocks_train = []
        blocks_test = []
        cursor = 0
        for j in range(X_train.shape[1]):
            cols_train = [(Xz_train[:, j] ** int(deg)).reshape(-1, 1) for deg in poly_degrees]
            cols_test = [(Xz_test[:, j] ** int(deg)).reshape(-1, 1) for deg in poly_degrees]
            block_train = np.concatenate(cols_train, axis=1)
            block_test = np.concatenate(cols_test, axis=1)
            blocks_train.append(block_train)
            blocks_test.append(block_test)
            main_slices.append(slice(cursor, cursor + block_train.shape[1]))
            cursor += block_train.shape[1]
        main_train = np.concatenate(blocks_train, axis=1).astype(np.float32)
        main_test = np.concatenate(blocks_test, axis=1).astype(np.float32)
    elif main_basis == "spline":
        spline = SplineTransformer(
            n_knots=int(spline_n_knots),
            degree=int(spline_degree),
            include_bias=False,
            extrapolation="continue",
        )
        main_train = spline.fit_transform(Xz_train).astype(np.float32)
        main_test = spline.transform(Xz_test).astype(np.float32)
        if main_train.shape[1] % X_train.shape[1] != 0:
            raise RuntimeError("Unexpected spline feature layout; cannot map coefficients to variables.")
        block_width = main_train.shape[1] // X_train.shape[1]
        for j in range(X_train.shape[1]):
            main_slices.append(slice(j * block_width, (j + 1) * block_width))
    else:
        raise ValueError(f"Unknown main_basis={main_basis!r}")

    Z_train, pairs = pair_design(Xz_train)
    Z_test, _ = pair_design(Xz_test)
    D_train = np.concatenate([main_train, Z_train], axis=1)
    D_test = np.concatenate([main_test, Z_test], axis=1)
    design_scaler = StandardScaler()
    D_train = design_scaler.fit_transform(D_train).astype(np.float32)
    D_test = design_scaler.transform(D_test).astype(np.float32)
    return D_train, D_test, pairs, main_slices, raw_scaler, design_scaler


def run_sparse_library(
    args: argparse.Namespace,
    function: str,
    method: str,
    seed: int,
) -> dict[str, object]:
    t0 = time.time()
    data = make_transfer_data(args, function, seed)
    X_train = data["X_train"].astype(np.float32)
    y_train = data["y_train"].reshape(-1).astype(np.float32)
    X_test = data["X_test"].astype(np.float32)
    y_test = data["y_test"].reshape(-1).astype(np.float32)
    gt = data["ground_truth"]
    true_vars = {int(v) for v in gt.active_variables}
    true_pairs = canonical_pairs(gt.interactions)
    true_endpoints = {v for pair in true_pairs for v in pair}
    top_m = min(int(args.top_m), args.dimension)
    pair_budget = max(int(args.pair_budget), len(true_pairs), 1)

    if method in {"sparse_lasso", "sparse_elasticnet"}:
        main_basis = "linear"
    elif method == "sparse_poly_lasso":
        main_basis = "polynomial"
    elif method == "sparse_spline_lasso":
        main_basis = "spline"
    else:
        raise ValueError(f"Unknown sparse method={method!r}")

    D_train, D_test, pairs, main_slices, _, _ = make_sparse_design(
        X_train,
        X_test,
        main_basis=main_basis,
        poly_degrees=args.poly_degrees,
        spline_n_knots=args.spline_n_knots,
        spline_degree=args.spline_degree,
    )
    if method in {"sparse_lasso", "sparse_poly_lasso", "sparse_spline_lasso"}:
        model = LassoCV(cv=args.cv, random_state=seed, max_iter=args.max_iter, n_jobs=1)
    elif method == "sparse_elasticnet":
        model = ElasticNetCV(
            cv=args.cv,
            random_state=seed,
            max_iter=args.max_iter,
            n_jobs=1,
            l1_ratio=args.elasticnet_l1_ratio,
        )
    else:
        raise ValueError(f"Unknown sparse method={method!r}")

    model.fit(D_train, y_train)
    pred = model.predict(D_test)
    test_mse = float(np.mean((pred - y_test) ** 2))
    coef = np.asarray(model.coef_, dtype=float)
    main_width = main_slices[-1].stop if main_slices else args.dimension
    main_scores = np.asarray([float(np.max(np.abs(coef[sl]))) for sl in main_slices], dtype=float)
    pair_scores = np.abs(coef[main_width:])

    main_order = np.argsort(-main_scores)
    selected_support = {int(i) for i in main_order[:top_m]}
    _, _, support_f1 = f1_from_sets(selected_support, true_vars)
    endpoint_success = int(true_endpoints.issubset(selected_support)) if true_endpoints else np.nan

    pair_fields = rank_pair_scores(pairs, pair_scores, true_pairs, pair_budget)
    row: dict[str, object] = {
        "function": function,
        "method": method,
        "evidence_object": "sparse_library_coefficients",
        "seed": int(seed),
        "samples": int(args.samples),
        "test_samples": int(args.test_samples),
        "dimension": int(args.dimension),
        "noise": float(args.noise),
        "nuisance_correlation": float(args.nuisance_correlation),
        "n_correlated_proxies": int(args.n_correlated_proxies),
        "top_m": int(top_m),
        "pair_budget": int(pair_budget),
        "test_mse": test_mse,
        "alpha": float(getattr(model, "alpha_", np.nan)),
        "l1_ratio": float(getattr(model, "l1_ratio_", np.nan)) if hasattr(model, "l1_ratio_") else np.nan,
        "main_basis": main_basis,
        "num_main_features": int(main_width),
        "num_pair_features": len(pairs),
        "support_object": "main_effect_coefficients",
        "selected_support": sorted(selected_support),
        "support_success_all_true": int(true_vars.issubset(selected_support)),
        "support_f1": support_f1,
        "endpoint_success": endpoint_success,
        "endpoint_recall": len(selected_support & true_endpoints) / len(true_endpoints) if true_endpoints else np.nan,
        "candidate_contains_all_true_pairs": 1,
        "runtime_sec": float(time.time() - t0),
    }
    row.update(pair_fields)
    row.update(common_transfer_fields(row, args.pred_mse_threshold))
    return row


def feature_grid(x: np.ndarray, grid_size: int) -> np.ndarray:
    qs = np.linspace(0.08, 0.92, int(grid_size))
    vals = np.unique(np.quantile(x, qs))
    if len(vals) < 2:
        vals = np.linspace(float(np.min(x)), float(np.max(x)), int(grid_size))
    return vals.astype(np.float32)


def raw_product_corr_candidates(
    X: np.ndarray,
    y: np.ndarray,
    pairs: list[Pair],
    top_k: int,
) -> list[Pair]:
    Xz = StandardScaler().fit_transform(X).astype(np.float32)
    yz = y.reshape(-1).astype(np.float64)
    yz = yz - yz.mean()
    yz_norm = float(np.linalg.norm(yz)) + 1e-12
    scores = np.empty(len(pairs), dtype=np.float64)
    for k, (i, j) in enumerate(pairs):
        z = (Xz[:, i] * Xz[:, j]).astype(np.float64)
        z = z - z.mean()
        scores[k] = abs(float(z @ yz)) / ((float(np.linalg.norm(z)) + 1e-12) * yz_norm)
    if top_k <= 0 or top_k >= len(pairs):
        return pairs
    order = np.argsort(-scores)[: int(top_k)]
    return [pairs[int(idx)] for idx in order]


def h_pair_score(
    model: HistGradientBoostingRegressor,
    background: np.ndarray,
    i: int,
    j: int,
    grid_i: np.ndarray,
    grid_j: np.ndarray,
    main_cache: dict[tuple[int, float], float],
    f0: float,
) -> float:
    fij = np.empty((len(grid_i), len(grid_j)), dtype=np.float64)
    for a_idx, a in enumerate(grid_i):
        for b_idx, b in enumerate(grid_j):
            Xp = background.copy()
            Xp[:, i] = float(a)
            Xp[:, j] = float(b)
            fij[a_idx, b_idx] = float(np.mean(model.predict(Xp)))
    fi = np.asarray([main_cache[(i, float(a))] for a in grid_i], dtype=np.float64)
    fj = np.asarray([main_cache[(j, float(b))] for b in grid_j], dtype=np.float64)
    h = fij - fi[:, None] - fj[None, :] + f0
    return float(np.mean(h * h))


def run_gbm_hstat(args: argparse.Namespace, function: str, seed: int) -> dict[str, object]:
    t0 = time.time()
    data = make_transfer_data(args, function, seed)
    X_train = data["X_train"].astype(np.float32)
    y_train = data["y_train"].reshape(-1).astype(np.float32)
    X_test = data["X_test"].astype(np.float32)
    y_test = data["y_test"].reshape(-1).astype(np.float32)
    gt = data["ground_truth"]
    true_pairs = canonical_pairs(gt.interactions)
    true_endpoints = {v for pair in true_pairs for v in pair}
    pair_budget = max(int(args.pair_budget), len(true_pairs), 1)

    model = HistGradientBoostingRegressor(
        max_iter=args.gbm_max_iter,
        learning_rate=args.gbm_learning_rate,
        max_leaf_nodes=args.gbm_max_leaf_nodes,
        l2_regularization=args.gbm_l2_regularization,
        random_state=seed,
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    test_mse = float(np.mean((pred - y_test) ** 2))

    rng = np.random.default_rng(seed + 991)
    bg_idx = rng.choice(len(X_test), size=min(args.h_background, len(X_test)), replace=False)
    background = X_test[bg_idx].copy()
    f0 = float(np.mean(model.predict(background)))
    grids = [feature_grid(X_test[:, j], args.h_grid_size) for j in range(args.dimension)]

    main_cache: dict[tuple[int, float], float] = {}
    for j, grid in enumerate(grids):
        for val in grid:
            Xp = background.copy()
            Xp[:, j] = float(val)
            main_cache[(j, float(val))] = float(np.mean(model.predict(Xp)))

    all_pairs = [(int(i), int(j)) for i, j in itertools.combinations(range(args.dimension), 2)]
    pairs = raw_product_corr_candidates(X_train, y_train, all_pairs, args.candidate_pairs)
    scores = np.empty(len(pairs), dtype=np.float64)
    for k, (i, j) in enumerate(pairs):
        scores[k] = h_pair_score(model, background, i, j, grids[i], grids[j], main_cache, f0)

    pair_fields = rank_pair_scores(pairs, scores, true_pairs, pair_budget)
    endpoint_success = pair_fields["endpoint_success_from_pairs"]
    row: dict[str, object] = {
        "function": function,
        "method": "gbm_hstat",
        "evidence_object": "gbm_functional_h_statistic",
        "seed": int(seed),
        "samples": int(args.samples),
        "test_samples": int(args.test_samples),
        "dimension": int(args.dimension),
        "noise": float(args.noise),
        "nuisance_correlation": float(args.nuisance_correlation),
        "n_correlated_proxies": int(args.n_correlated_proxies),
        "top_m": int(args.top_m),
        "pair_budget": int(pair_budget),
        "test_mse": test_mse,
        "gbm_max_iter": int(args.gbm_max_iter),
        "gbm_max_leaf_nodes": int(args.gbm_max_leaf_nodes),
        "h_background": int(len(background)),
        "h_grid_size": int(args.h_grid_size),
        "num_candidate_pairs": int(len(pairs)),
        "num_all_pairs": int(len(all_pairs)),
        "candidate_contains_all_true_pairs": int(true_pairs.issubset(set(pairs))) if true_pairs else np.nan,
        "support_object": "top_hstat_pair_endpoints",
        "selected_support": pair_fields["selected_pair_endpoints"],
        "support_success_all_true": endpoint_success,
        "support_f1": np.nan,
        "endpoint_success": endpoint_success,
        "endpoint_recall": (
            len(set(pair_fields["selected_pair_endpoints"]) & true_endpoints) / len(true_endpoints)
            if true_endpoints
            else np.nan
        ),
        "runtime_sec": float(time.time() - t0),
    }
    row.update(pair_fields)
    row.update(common_transfer_fields(row, args.pred_mse_threshold))
    return row


def make_grouped_spline_basis(
    X_train: np.ndarray,
    X_test: np.ndarray,
    n_knots: int,
    degree: int,
) -> tuple[np.ndarray, np.ndarray, list[slice]]:
    raw_scaler = StandardScaler()
    Xz_train = raw_scaler.fit_transform(X_train).astype(np.float32)
    Xz_test = raw_scaler.transform(X_test).astype(np.float32)
    spline = SplineTransformer(
        n_knots=int(n_knots),
        degree=int(degree),
        include_bias=False,
        extrapolation="continue",
    )
    B_train = spline.fit_transform(Xz_train).astype(np.float32)
    B_test = spline.transform(Xz_test).astype(np.float32)
    if B_train.shape[1] % X_train.shape[1] != 0:
        raise RuntimeError("Unexpected spline feature layout; cannot map coefficients to variables.")
    block_width = B_train.shape[1] // X_train.shape[1]
    slices = [slice(j * block_width, (j + 1) * block_width) for j in range(X_train.shape[1])]
    scaler = StandardScaler()
    B_train = scaler.fit_transform(B_train).astype(np.float32)
    B_test = scaler.transform(B_test).astype(np.float32)
    return B_train, B_test, slices


def symbolic_main_terms(x: np.ndarray) -> list[np.ndarray]:
    return [
        x,
        x ** 2,
        x ** 3,
        np.sin(np.pi * x),
        np.cos(np.pi * x),
        np.sin(2.0 * np.pi * x),
        np.cos(2.0 * np.pi * x),
    ]


def symbolic_pair_terms(z: np.ndarray) -> list[np.ndarray]:
    safe_log = np.log(np.clip(2.0 + z, 1e-6, None))
    return [
        z,
        z ** 2,
        z ** 3,
        np.sin(np.pi * z),
        np.cos(np.pi * z),
        np.exp(0.5 * np.clip(z, -3.0, 3.0)),
        safe_log,
    ]


def make_symbolic_library_design(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    candidate_pairs: int,
) -> tuple[np.ndarray, np.ndarray, list[slice], list[Pair], list[slice]]:
    main_blocks_train: list[np.ndarray] = []
    main_blocks_test: list[np.ndarray] = []
    main_slices: list[slice] = []
    cursor = 0
    for j in range(X_train.shape[1]):
        block_train = np.stack(symbolic_main_terms(X_train[:, j].astype(np.float64)), axis=1)
        block_test = np.stack(symbolic_main_terms(X_test[:, j].astype(np.float64)), axis=1)
        main_blocks_train.append(block_train)
        main_blocks_test.append(block_test)
        main_slices.append(slice(cursor, cursor + block_train.shape[1]))
        cursor += block_train.shape[1]

    all_pairs = [(int(i), int(j)) for i, j in itertools.combinations(range(X_train.shape[1]), 2)]
    pairs = raw_product_corr_candidates(X_train, y_train, all_pairs, candidate_pairs)
    pair_blocks_train: list[np.ndarray] = []
    pair_blocks_test: list[np.ndarray] = []
    pair_slices: list[slice] = []
    for i, j in pairs:
        z_train = (X_train[:, i] * X_train[:, j]).astype(np.float64)
        z_test = (X_test[:, i] * X_test[:, j]).astype(np.float64)
        block_train = np.stack(symbolic_pair_terms(z_train), axis=1)
        block_test = np.stack(symbolic_pair_terms(z_test), axis=1)
        pair_blocks_train.append(block_train)
        pair_blocks_test.append(block_test)
        pair_slices.append(slice(cursor, cursor + block_train.shape[1]))
        cursor += block_train.shape[1]

    D_train = np.concatenate([*main_blocks_train, *pair_blocks_train], axis=1)
    D_test = np.concatenate([*main_blocks_test, *pair_blocks_test], axis=1)
    scaler = StandardScaler()
    D_train = scaler.fit_transform(D_train).astype(np.float32)
    D_test = scaler.transform(D_test).astype(np.float32)
    return D_train, D_test, main_slices, pairs, pair_slices


def run_symbolic_lasso(args: argparse.Namespace, function: str, seed: int) -> dict[str, object]:
    t0 = time.time()
    data = make_transfer_data(args, function, seed)
    X_train = data["X_train"].astype(np.float32)
    y_train = data["y_train"].reshape(-1).astype(np.float32)
    X_test = data["X_test"].astype(np.float32)
    y_test = data["y_test"].reshape(-1).astype(np.float32)
    gt = data["ground_truth"]
    true_vars = {int(v) for v in gt.active_variables}
    true_pairs = canonical_pairs(gt.interactions)
    true_endpoints = {v for pair in true_pairs for v in pair}
    top_m = min(int(args.top_m), args.dimension)
    pair_budget = max(int(args.pair_budget), len(true_pairs), 1)

    D_train, D_test, main_slices, pairs, pair_slices = make_symbolic_library_design(
        X_train,
        X_test,
        y_train,
        candidate_pairs=args.symbolic_candidate_pairs,
    )
    model = LassoCV(cv=args.cv, random_state=seed, max_iter=args.max_iter, n_jobs=1)
    model.fit(D_train, y_train)
    pred = model.predict(D_test)
    test_mse = float(np.mean((pred - y_test) ** 2))
    coef = np.asarray(model.coef_, dtype=float)
    main_scores = np.asarray([float(np.max(np.abs(coef[sl]))) for sl in main_slices], dtype=float)
    pair_scores = np.asarray([float(np.max(np.abs(coef[sl]))) for sl in pair_slices], dtype=float)

    selected_support = {int(i) for i in np.argsort(-main_scores)[:top_m]}
    _, _, support_f1 = f1_from_sets(selected_support, true_vars)
    main_endpoint_success = int(true_endpoints.issubset(selected_support)) if true_endpoints else np.nan
    pair_fields = rank_pair_scores(pairs, pair_scores, true_pairs, pair_budget)
    endpoint_success = pair_fields["endpoint_success_from_pairs"]
    row: dict[str, object] = {
        "function": function,
        "method": "symbolic_lasso",
        "evidence_object": "fixed_symbolic_library_coefficients",
        "seed": int(seed),
        "samples": int(args.samples),
        "test_samples": int(args.test_samples),
        "dimension": int(args.dimension),
        "noise": float(args.noise),
        "nuisance_correlation": float(args.nuisance_correlation),
        "n_correlated_proxies": int(args.n_correlated_proxies),
        "top_m": int(top_m),
        "pair_budget": int(pair_budget),
        "test_mse": test_mse,
        "alpha": float(getattr(model, "alpha_", np.nan)),
        "num_symbolic_main_features": int(sum(sl.stop - sl.start for sl in main_slices)),
        "num_symbolic_pair_features": int(sum(sl.stop - sl.start for sl in pair_slices)),
        "num_candidate_pairs": int(len(pairs)),
        "num_all_pairs": int(args.dimension * (args.dimension - 1) / 2),
        "candidate_contains_all_true_pairs": int(true_pairs.issubset(set(pairs))) if true_pairs else np.nan,
        "support_object": "symbolic_main_terms",
        "selected_support": sorted(selected_support),
        "support_success_all_true": int(true_vars.issubset(selected_support)),
        "support_f1": support_f1,
        "main_endpoint_success": main_endpoint_success,
        "endpoint_success": endpoint_success,
        "endpoint_recall": (
            len(set(pair_fields["selected_pair_endpoints"]) & true_endpoints) / len(true_endpoints)
            if true_endpoints
            else np.nan
        ),
        "runtime_sec": float(time.time() - t0),
    }
    row.update(pair_fields)
    row.update(common_transfer_fields(row, args.pred_mse_threshold))
    return row


def pair_tensor_features(
    x_i_train: np.ndarray,
    x_j_train: np.ndarray,
    x_i_eval: np.ndarray,
    x_j_eval: np.ndarray,
    n_knots: int,
    degree: int,
) -> tuple[np.ndarray, np.ndarray]:
    spline_i = SplineTransformer(
        n_knots=int(n_knots),
        degree=int(degree),
        include_bias=False,
        extrapolation="continue",
    )
    spline_j = SplineTransformer(
        n_knots=int(n_knots),
        degree=int(degree),
        include_bias=False,
        extrapolation="continue",
    )
    Bi_train = spline_i.fit_transform(x_i_train.reshape(-1, 1)).astype(np.float32)
    Bj_train = spline_j.fit_transform(x_j_train.reshape(-1, 1)).astype(np.float32)
    Bi_eval = spline_i.transform(x_i_eval.reshape(-1, 1)).astype(np.float32)
    Bj_eval = spline_j.transform(x_j_eval.reshape(-1, 1)).astype(np.float32)
    T_train = np.einsum("na,nb->nab", Bi_train, Bj_train).reshape(len(Bi_train), -1)
    T_eval = np.einsum("na,nb->nab", Bi_eval, Bj_eval).reshape(len(Bi_eval), -1)
    scaler = StandardScaler()
    T_train = scaler.fit_transform(T_train).astype(np.float32)
    T_eval = scaler.transform(T_eval).astype(np.float32)
    return T_train, T_eval


def score_tensor_pair_ridge(
    X_train: np.ndarray,
    residual: np.ndarray,
    pair: Pair,
    rng: np.random.Generator,
    args: argparse.Namespace,
) -> float:
    n = len(X_train)
    eval_size = max(32, int(round(n * float(args.ga2m_pair_eval_fraction))))
    eval_size = min(eval_size, max(1, n - 32))
    eval_idx = rng.choice(n, size=eval_size, replace=False)
    mask = np.ones(n, dtype=bool)
    mask[eval_idx] = False
    fit_idx = np.flatnonzero(mask)
    i, j = pair
    T_fit, T_eval = pair_tensor_features(
        X_train[fit_idx, i],
        X_train[fit_idx, j],
        X_train[eval_idx, i],
        X_train[eval_idx, j],
        n_knots=args.ga2m_pair_spline_n_knots,
        degree=args.ga2m_pair_spline_degree,
    )
    y_fit = residual[fit_idx]
    y_eval = residual[eval_idx]
    var_eval = float(np.var(y_eval)) + 1e-12
    model = RidgeCV(alphas=np.asarray(args.ga2m_alphas, dtype=float))
    model.fit(T_fit, y_fit)
    pred = model.predict(T_eval)
    mse = float(np.mean((pred - y_eval) ** 2))
    return float(max(0.0, 1.0 - mse / var_eval))


def run_ga2m_spline(args: argparse.Namespace, function: str, seed: int) -> dict[str, object]:
    t0 = time.time()
    data = make_transfer_data(args, function, seed)
    X_train = data["X_train"].astype(np.float32)
    y_train = data["y_train"].reshape(-1).astype(np.float32)
    X_test = data["X_test"].astype(np.float32)
    y_test = data["y_test"].reshape(-1).astype(np.float32)
    gt = data["ground_truth"]
    true_vars = {int(v) for v in gt.active_variables}
    true_pairs = canonical_pairs(gt.interactions)
    true_endpoints = {v for pair in true_pairs for v in pair}
    top_m = min(int(args.top_m), args.dimension)
    pair_budget = max(int(args.pair_budget), len(true_pairs), 1)

    B_train, B_test, main_slices = make_grouped_spline_basis(
        X_train,
        X_test,
        n_knots=args.ga2m_main_spline_n_knots,
        degree=args.ga2m_main_spline_degree,
    )
    main_model = RidgeCV(alphas=np.asarray(args.ga2m_alphas, dtype=float))
    main_model.fit(B_train, y_train)
    main_pred_train = main_model.predict(B_train)
    residual = y_train - main_pred_train

    coef = np.asarray(main_model.coef_, dtype=float)
    main_scores = np.asarray([float(np.linalg.norm(coef[sl], ord=2)) for sl in main_slices], dtype=float)
    selected_support = {int(i) for i in np.argsort(-main_scores)[:top_m]}
    _, _, support_f1 = f1_from_sets(selected_support, true_vars)
    main_endpoint_success = int(true_endpoints.issubset(selected_support)) if true_endpoints else np.nan

    all_pairs = [(int(i), int(j)) for i, j in itertools.combinations(range(args.dimension), 2)]
    candidates = raw_product_corr_candidates(X_train, residual, all_pairs, args.candidate_pairs)
    rng = np.random.default_rng(seed + 7771)
    scores = np.empty(len(candidates), dtype=np.float64)
    for k, pair in enumerate(candidates):
        scores[k] = score_tensor_pair_ridge(X_train, residual, pair, rng, args)

    pair_fields = rank_pair_scores(candidates, scores, true_pairs, pair_budget)
    top_for_prediction = [
        tuple(sorted(candidates[int(idx)]))
        for idx in np.argsort(-scores)[: max(pair_budget, int(args.ga2m_prediction_pairs))]
    ]
    pair_blocks_train: list[np.ndarray] = []
    pair_blocks_test: list[np.ndarray] = []
    for pair in top_for_prediction:
        i, j = pair
        T_train, T_test = pair_tensor_features(
            X_train[:, i],
            X_train[:, j],
            X_test[:, i],
            X_test[:, j],
            n_knots=args.ga2m_pair_spline_n_knots,
            degree=args.ga2m_pair_spline_degree,
        )
        pair_blocks_train.append(T_train)
        pair_blocks_test.append(T_test)
    if pair_blocks_train:
        D_train = np.concatenate([B_train, *pair_blocks_train], axis=1)
        D_test = np.concatenate([B_test, *pair_blocks_test], axis=1)
    else:
        D_train = B_train
        D_test = B_test
    final_model = RidgeCV(alphas=np.asarray(args.ga2m_alphas, dtype=float))
    final_model.fit(D_train, y_train)
    pred = final_model.predict(D_test)
    test_mse = float(np.mean((pred - y_test) ** 2))

    endpoint_success = pair_fields["endpoint_success_from_pairs"]
    row: dict[str, object] = {
        "function": function,
        "method": "ga2m_spline",
        "evidence_object": "spline_additive_plus_tensor_pair_screen",
        "seed": int(seed),
        "samples": int(args.samples),
        "test_samples": int(args.test_samples),
        "dimension": int(args.dimension),
        "noise": float(args.noise),
        "nuisance_correlation": float(args.nuisance_correlation),
        "n_correlated_proxies": int(args.n_correlated_proxies),
        "top_m": int(top_m),
        "pair_budget": int(pair_budget),
        "test_mse": test_mse,
        "main_basis": "spline",
        "num_main_features": int(B_train.shape[1]),
        "num_candidate_pairs": int(len(candidates)),
        "num_all_pairs": int(len(all_pairs)),
        "ga2m_prediction_pairs": int(len(top_for_prediction)),
        "candidate_contains_all_true_pairs": int(true_pairs.issubset(set(candidates))) if true_pairs else np.nan,
        "support_object": "additive_spline_group_norm",
        "selected_support": sorted(selected_support),
        "support_success_all_true": int(true_vars.issubset(selected_support)),
        "support_f1": support_f1,
        "main_endpoint_success": main_endpoint_success,
        "endpoint_success": endpoint_success,
        "endpoint_recall": (
            len(set(pair_fields["selected_pair_endpoints"]) & true_endpoints) / len(true_endpoints)
            if true_endpoints
            else np.nan
        ),
        "runtime_sec": float(time.time() - t0),
    }
    row.update(pair_fields)
    row.update(common_transfer_fields(row, args.pred_mse_threshold))
    return row


def summarize(detail: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "function",
        "method",
        "evidence_object",
        "samples",
        "test_samples",
        "dimension",
        "noise",
        "nuisance_correlation",
        "n_correlated_proxies",
        "top_m",
        "pair_budget",
    ]
    numeric_cols = [
        "test_mse",
        "prediction_success",
        "support_success_all_true",
        "endpoint_success",
        "pair_success_all_true_at_budget",
        "pair_success_any_true_at_budget",
        "pred_to_support_failure",
        "pred_to_endpoint_failure",
        "pred_to_pair_failure",
        "support_to_pair_failure",
        "pair_without_support",
        "endpoint_without_pair",
        "true_pair_rank_best",
        "true_pair_rank_worst",
        "true_pair_margin_min",
        "candidate_contains_all_true_pairs",
        "runtime_sec",
    ]
    numeric_cols = [c for c in numeric_cols if c in detail.columns]
    out = detail.groupby(group_cols, dropna=False)[numeric_cols].agg(["mean", "std"]).reset_index()
    out.columns = ["_".join(str(v) for v in col if v != "").rstrip("_") for col in out.columns]
    counts = detail.groupby(group_cols, dropna=False).size().reset_index(name="num_runs")
    for col in [
        "prediction_success",
        "support_success_all_true",
        "endpoint_success",
        "pair_success_all_true_at_budget",
    ]:
        if col in detail.columns:
            successes = detail.groupby(group_cols, dropna=False)[col].sum(min_count=1).reset_index(name=f"{col}_count")
            out = out.merge(successes, on=group_cols, how="left")
    return out.merge(counts, on=group_cols, how="left")


def write_markdown(summary: pd.DataFrame, path: Path) -> None:
    keep = [
        "function",
        "method",
        "num_runs",
        "test_mse_mean",
        "prediction_success_mean",
        "support_success_all_true_mean",
        "endpoint_success_mean",
        "pair_success_all_true_at_budget_mean",
        "pred_to_pair_failure_mean",
        "support_to_pair_failure_mean",
        "endpoint_without_pair_mean",
        "true_pair_rank_worst_mean",
        "true_pair_margin_min_mean",
    ]
    show = summary[[c for c in keep if c in summary.columns]].copy()
    for col in show.columns:
        if col.endswith("_mean") or col.endswith("_std"):
            show[col] = show[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.3f}")
    widths = {col: max(len(col), *(len(str(v)) for v in show[col])) for col in show.columns}
    lines = ["# Cross-Method Evidence-Transfer Pilot", ""]
    lines.append("| " + " | ".join(col.ljust(widths[col]) for col in show.columns) + " |")
    lines.append("| " + " | ".join("-" * widths[col] for col in show.columns) + " |")
    for _, row in show.iterrows():
        lines.append("| " + " | ".join(str(row[col]).ljust(widths[col]) for col in show.columns) + " |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def run_one(args: argparse.Namespace, function: str, method: str, seed: int) -> dict[str, object]:
    if method in {"sparse_lasso", "sparse_elasticnet", "sparse_poly_lasso", "sparse_spline_lasso"}:
        return run_sparse_library(args, function, method, seed)
    if method == "gbm_hstat":
        return run_gbm_hstat(args, function, seed)
    if method == "ga2m_spline":
        return run_ga2m_spline(args, function, seed)
    if method == "symbolic_lasso":
        return run_symbolic_lasso(args, function, seed)
    raise ValueError(f"Unknown method={method!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--functions", nargs="+", default=["core_interaction_c025", "formula_bilinear", "formula_mixed_sparse"])
    parser.add_argument("--methods", nargs="+", default=["sparse_lasso", "gbm_hstat"])
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--test_samples", type=int, default=2048)
    parser.add_argument("--dimension", type=int, default=100)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--nuisance_correlation", type=float, default=0.0)
    parser.add_argument("--n_correlated_proxies", type=int, default=0)
    parser.add_argument("--semisynthetic_c", type=float, default=0.25)
    parser.add_argument("--top_m", type=int, default=4)
    parser.add_argument("--pair_budget", type=int, default=1)
    parser.add_argument("--pred_mse_threshold", type=float, default=0.05)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(4)))
    parser.add_argument("--cv", type=int, default=3)
    parser.add_argument("--max_iter", type=int, default=5000)
    parser.add_argument("--elasticnet_l1_ratio", type=float, nargs="+", default=[0.5, 0.8, 0.95])
    parser.add_argument("--poly_degrees", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--spline_n_knots", type=int, default=6)
    parser.add_argument("--spline_degree", type=int, default=3)
    parser.add_argument("--ga2m_alphas", type=float, nargs="+", default=[0.01, 0.1, 1.0, 10.0])
    parser.add_argument("--ga2m_main_spline_n_knots", type=int, default=6)
    parser.add_argument("--ga2m_main_spline_degree", type=int, default=3)
    parser.add_argument("--ga2m_pair_spline_n_knots", type=int, default=4)
    parser.add_argument("--ga2m_pair_spline_degree", type=int, default=3)
    parser.add_argument("--ga2m_pair_eval_fraction", type=float, default=0.25)
    parser.add_argument("--ga2m_prediction_pairs", type=int, default=8)
    parser.add_argument("--symbolic_candidate_pairs", type=int, default=800)
    parser.add_argument("--gbm_max_iter", type=int, default=160)
    parser.add_argument("--gbm_learning_rate", type=float, default=0.05)
    parser.add_argument("--gbm_max_leaf_nodes", type=int, default=31)
    parser.add_argument("--gbm_l2_regularization", type=float, default=0.0)
    parser.add_argument("--h_background", type=int, default=64)
    parser.add_argument("--h_grid_size", type=int, default=5)
    parser.add_argument("--candidate_pairs", type=int, default=160)
    parser.add_argument("--out_dir", default="results/revision/cross_method_transfer_baselines/pilot")
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for function in args.functions:
        for method in args.methods:
            for seed in args.seeds:
                print(f"Running function={function} method={method} seed={seed}", flush=True)
                rows.append(run_one(args, function, method, int(seed)))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    detail = pd.DataFrame(rows)
    summary = summarize(detail)
    detail_path = out_dir / "cross_method_transfer_detail.csv"
    summary_path = out_dir / "cross_method_transfer_summary.csv"
    md_path = out_dir / "cross_method_transfer_summary.md"
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    write_markdown(summary, md_path)
    print(f"Wrote {detail_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {md_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
