from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes, load_wine
from sklearn.preprocessing import SplineTransformer, StandardScaler
from sklearn.linear_model import Ridge

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.run_same_data_kan_stability_probe import run_probe_on_fixed_data
from experiments.run_kan_native_innovation_loop import aggregate_probe_scores, select_support


@dataclass(frozen=True)
class SemiSyntheticData:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    active_variables: tuple[int, ...]
    interactions: tuple[tuple[int, int], ...]
    feature_names: list[str]


def canonical_pairs(pairs) -> set[tuple[int, int]]:
    return {tuple(sorted((int(i), int(j)))) for i, j in pairs}


def load_real_covariates(name: str) -> tuple[np.ndarray, list[str]]:
    if name == "diabetes":
        ds = load_diabetes()
        X = ds.data.astype(np.float32)
        names = [str(v) for v in ds.feature_names]
    elif name == "breast_cancer":
        ds = load_breast_cancer()
        X = ds.data.astype(np.float32)
        names = [str(v) for v in ds.feature_names]
    elif name == "wine":
        ds = load_wine()
        X = ds.data.astype(np.float32)
        names = [str(v) for v in ds.feature_names]
    else:
        raise ValueError(f"Unknown covariate dataset {name!r}")
    X = StandardScaler().fit_transform(X).astype(np.float32)
    return X, names


def make_semisynthetic(
    dataset: str,
    n_train: int,
    n_test: int,
    c: float,
    noise: float,
    seed: int,
) -> SemiSyntheticData:
    X_pool, feature_names = load_real_covariates(dataset)
    rng = np.random.default_rng(int(seed))
    n_total = int(n_train) + int(n_test)
    replace = n_total > len(X_pool)
    idx = rng.choice(len(X_pool), size=n_total, replace=replace)
    X = X_pool[idx].astype(np.float32)

    # Compress heavy-tailed standardized covariates while preserving real
    # correlation/proxy structure. This keeps the injected formula numerically
    # close to the synthetic [-1, 1] tasks without making covariates independent.
    Z = np.tanh(X).astype(np.float32)
    y_clean = (
        np.sin(np.pi * Z[:, 0])
        + Z[:, 1] ** 2
        + float(c) * Z[:, 2] * Z[:, 3]
    ).astype(np.float32)
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
    y_train = ((y_train - mean) / std).astype(np.float32)
    y_test = ((y_test - mean) / std).astype(np.float32)

    return SemiSyntheticData(
        X_train=X_train.astype(np.float32),
        y_train=y_train.astype(np.float32),
        X_test=X_test.astype(np.float32),
        y_test=y_test.astype(np.float32),
        active_variables=(0, 1, 2, 3),
        interactions=((2, 3),),
        feature_names=feature_names,
    )


def support_stats(support: np.ndarray, true_vars: tuple[int, ...], true_pairs: set[tuple[int, int]]) -> dict:
    support_set = {int(v) for v in support}
    true_set = {int(v) for v in true_vars}
    endpoints = {v for pair in true_pairs for v in pair}
    tp = len(support_set & true_set)
    precision = tp / len(support_set) if len(support_set) else 0.0
    recall = tp / len(true_set) if len(true_set) else 0.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    selected_pairs = canonical_pairs(itertools.combinations(sorted(support_set), 2))
    return {
        "screen_contains_all_true_vars": int(true_set.issubset(support_set)),
        "screen_true_var_recall": float(recall),
        "screen_true_var_f1": float(f1),
        "screen_contains_all_interaction_endpoints": int(endpoints.issubset(support_set)) if endpoints else np.nan,
        "screen_interaction_endpoint_recall": float(len(support_set & endpoints) / len(endpoints)) if endpoints else np.nan,
        "screen_contains_true_interactions": int(bool(selected_pairs & true_pairs)) if true_pairs else np.nan,
    }


def additive_spline_design(X: np.ndarray, n_knots: int = 6, degree: int = 3) -> np.ndarray:
    blocks = []
    for j in range(X.shape[1]):
        spl = SplineTransformer(n_knots=n_knots, degree=degree, include_bias=False)
        blocks.append(spl.fit_transform(X[:, [j]]))
    return np.concatenate(blocks, axis=1).astype(np.float32)


def residual_product_screen(X: np.ndarray, y: np.ndarray, true_pairs: set[tuple[int, int]]) -> dict:
    Xz = StandardScaler().fit_transform(X).astype(np.float32)
    y1 = y.reshape(-1).astype(np.float32)
    design = additive_spline_design(Xz)
    model = Ridge(alpha=1.0)
    model.fit(design, y1)
    resid = y1 - model.predict(design).reshape(-1)
    resid = resid - resid.mean()
    resid_norm = float(np.linalg.norm(resid)) + 1e-12
    pairs = list(itertools.combinations(range(X.shape[1]), 2))
    scores = []
    for i, j in pairs:
        z = (Xz[:, i] * Xz[:, j]).astype(np.float32)
        z = z - z.mean()
        scores.append(float(abs(z @ resid) / ((np.linalg.norm(z) + 1e-12) * resid_norm)))
    scores_arr = np.asarray(scores)
    order = np.argsort(-scores_arr)
    top_pair = tuple(sorted(pairs[int(order[0])]))
    true_ranks = []
    true_scores = []
    for pair in true_pairs:
        idx = pairs.index(pair)
        true_ranks.append(int(np.where(order == idx)[0][0]) + 1)
        true_scores.append(float(scores_arr[idx]))
    false_scores = [scores_arr[k] for k, p in enumerate(pairs) if tuple(sorted(p)) not in true_pairs]
    return {
        "residual_top1_pair": int(top_pair in true_pairs),
        "residual_top_pair": json.dumps(top_pair),
        "residual_true_pair_rank_worst": float(max(true_ranks)),
        "residual_true_pair_score_mean": float(np.mean(true_scores)),
        "residual_max_false_pair_score": float(np.max(false_scores)),
    }


def run_outer(args: argparse.Namespace, dataset: str, c: float, n_train: int, outer_seed: int, device: str) -> list[dict]:
    t0 = time.time()
    data = make_semisynthetic(
        dataset=dataset,
        n_train=int(n_train),
        n_test=int(args.test_samples),
        c=float(c),
        noise=float(args.noise),
        seed=int(outer_seed),
    )
    true_pairs = canonical_pairs(data.interactions)
    probe_rows = []
    for r in range(int(args.R)):
        probe_seed = int(args.probe_seed_offset + outer_seed * 1000 + r)
        try:
            probe_rows.append(
                run_probe_on_fixed_data(
                    X_train=data.X_train,
                    y_train=data.y_train,
                    X_test=data.X_test,
                    y_test=data.y_test,
                    probe_seed=probe_seed,
                    outer_seed=int(outer_seed),
                    args=args,
                    device=device,
                )
            )
        except Exception as exc:
            probe_rows.append({"seed": probe_seed, "status": "failed", "error": repr(exc)})
    probes = pd.DataFrame([row for row in probe_rows if row.get("status") == "ok"])
    agg = aggregate_probe_scores(probes, data.X_train.shape[1])
    probe_mse = pd.to_numeric(probes.get("test_mse", pd.Series(dtype=float)), errors="coerce")
    residual = residual_product_screen(data.X_train, data.y_train, true_pairs)

    rows = []
    for method in args.methods:
        support, meta = select_support(method, agg, int(args.top_m), data.X_train.shape[1])
        support = np.asarray(sorted(support), dtype=int)
        row = {
            "dataset": dataset,
            "function": "semisynthetic_real_covariates",
            "formula": "sin(pi*x0)+x1^2+c*x2*x3 on tanh-standardized real covariates",
            "c": float(c),
            "noise": float(args.noise),
            "samples": int(n_train),
            "test_samples": int(args.test_samples),
            "dimension": int(data.X_train.shape[1]),
            "outer_seed": int(outer_seed),
            "method": method,
            "R": int(args.R),
            "top_m": int(args.top_m),
            "width_hidden": int(args.width_hidden),
            "grid": int(args.grid),
            "lamb": float(args.lamb),
            "probe_steps": int(args.probe_steps),
            "num_ok_probes": int(len(probes)),
            "probe_test_mse_mean": float(probe_mse.mean()) if len(probe_mse) else np.nan,
            "probe_test_mse_min": float(probe_mse.min()) if len(probe_mse) else np.nan,
            "selected_screen_features": json.dumps([int(v) for v in support.tolist()]),
            "top_selection_variables": json.dumps(meta.get("top_selection_variables", [])),
            "runtime_sec": float(time.time() - t0),
        }
        row.update(support_stats(support, data.active_variables, true_pairs))
        row.update(residual)
        rows.append(row)
    return rows


def summarize(detail: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "dataset",
        "c",
        "samples",
        "dimension",
        "noise",
        "method",
        "R",
        "top_m",
        "width_hidden",
        "grid",
        "lamb",
        "probe_steps",
    ]
    numeric = [
        "screen_contains_all_true_vars",
        "screen_true_var_recall",
        "screen_true_var_f1",
        "screen_contains_all_interaction_endpoints",
        "screen_interaction_endpoint_recall",
        "screen_contains_true_interactions",
        "probe_test_mse_mean",
        "probe_test_mse_min",
        "residual_top1_pair",
        "residual_true_pair_rank_worst",
        "runtime_sec",
    ]
    out = detail.groupby(group_cols, dropna=False)[numeric].agg(["mean", "std"]).reset_index()
    out.columns = ["_".join(str(v) for v in col if v != "").rstrip("_") for col in out.columns]
    counts = detail.groupby(group_cols, dropna=False).size().reset_index(name="num_outer_seeds")
    endpoint_success = detail.groupby(group_cols, dropna=False)["screen_contains_all_interaction_endpoints"].sum().reset_index(name="endpoint_successes")
    pair_success = detail.groupby(group_cols, dropna=False)["screen_contains_true_interactions"].sum().reset_index(name="support_pair_successes")
    residual_success = detail.groupby(group_cols, dropna=False)["residual_top1_pair"].sum().reset_index(name="residual_top1_successes")
    return out.merge(counts, on=group_cols).merge(endpoint_success, on=group_cols).merge(pair_success, on=group_cols).merge(residual_success, on=group_cols)


def parse_float_list(values: list[str]) -> list[float]:
    return [float(v) for v in values]


def main() -> None:
    parser = argparse.ArgumentParser(description="Semi-synthetic real-covariate pyKAN readout audit.")
    parser.add_argument("--datasets", nargs="+", default=["diabetes", "breast_cancer"])
    parser.add_argument("--coefficients", nargs="+", default=["0.10", "0.25", "0.50"])
    parser.add_argument("--samples", type=int, nargs="+", default=[128, 256, 384])
    parser.add_argument("--test-samples", type=int, default=128)
    parser.add_argument("--outer-seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--methods", nargs="+", default=["feature_stability_var", "feature_edge_hybrid"])
    parser.add_argument("--top-m", type=int, default=4)
    parser.add_argument("--R", type=int, default=12)
    parser.add_argument("--resample", choices=["none", "bootstrap", "subsample"], default="bootstrap")
    parser.add_argument("--subsample-frac", type=float, default=0.8)
    parser.add_argument("--probe-seed-offset", type=int, default=71000)
    parser.add_argument("--width-hidden", type=int, default=8)
    parser.add_argument("--grid", type=int, default=5)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--lamb", type=float, default=1e-3)
    parser.add_argument("--probe-steps", type=int, default=35)
    parser.add_argument("--opt", default="LBFGS")
    parser.add_argument("--update-grid", action="store_true")
    parser.add_argument("--grid-update-num", type=int, default=5)
    parser.add_argument("--batch", type=int, default=-1)
    parser.add_argument("--pred-batch-size", type=int, default=4096)
    parser.add_argument("--keep-top-pairs", type=int, default=300)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--out-dir", default="results/revision/semisynthetic_covariates_3h")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    if args.device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
    else:
        device = args.device

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    detail_path = out_dir / "semisynthetic_covariate_audit_detail.csv"
    summary_path = out_dir / "semisynthetic_covariate_audit_summary.csv"
    config_path = out_dir / "config.json"
    config_path.write_text(json.dumps(vars(args), indent=2, sort_keys=True), encoding="utf-8")

    rows = []
    completed = set()
    if detail_path.exists() and not args.no_resume:
        existing = pd.read_csv(detail_path)
        rows = existing.to_dict("records")
        for _, row in existing.iterrows():
            completed.add((str(row["dataset"]), float(row["c"]), int(row["samples"]), int(row["outer_seed"])))

    coefficients = parse_float_list(args.coefficients)
    for dataset in args.datasets:
        for c in coefficients:
            for n_train in args.samples:
                for seed in args.outer_seeds:
                    key = (dataset, float(c), int(n_train), int(seed))
                    if key in completed:
                        print(f"[SKIP] dataset={dataset} c={c:g} n={n_train} seed={seed}", flush=True)
                        continue
                    print(f"[RUN] dataset={dataset} c={c:g} n={n_train} seed={seed} R={args.R}", flush=True)
                    try:
                        new_rows = run_outer(args, dataset, float(c), int(n_train), int(seed), device)
                        rows.extend(new_rows)
                    except Exception as exc:
                        rows.append({
                            "dataset": dataset,
                            "c": float(c),
                            "samples": int(n_train),
                            "outer_seed": int(seed),
                            "method": "failed",
                            "error": repr(exc),
                        })
                    completed.add(key)
                    detail = pd.DataFrame(rows)
                    detail.to_csv(detail_path, index=False)
                    if len(detail) and "screen_contains_all_interaction_endpoints" in detail.columns:
                        summarize(detail[detail["method"].ne("failed")]).to_csv(summary_path, index=False)

    detail = pd.DataFrame(rows)
    detail.to_csv(detail_path, index=False)
    ok = detail[detail["method"].ne("failed")] if "method" in detail.columns else detail
    summary = summarize(ok)
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
