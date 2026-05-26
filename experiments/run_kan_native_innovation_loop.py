from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import make_synthetic
from experiments.run_stability_selected_kan_quick import function_to_c
from experiments.run_tuned_kan_recovery import (
    Pair,
    batch_predict,
    canonical_pairs,
    endpoint_recovery,
    evaluate_interaction_recovery,
    evaluate_variable_recovery,
    anova_pair_scores,
    finite_difference_pair_scores,
    gradient_importance,
    hybrid_pair_scores,
    local_to_full_pair_scores,
    local_to_full_scores,
    mse_np,
    support_stats,
    train_kan,
)


METHODS = [
    "grad_stability_var",
    "feature_stability_var",
    "edge_stability_var",
    "edge_pair_endpoint",
    "edge_endpoint_mass",
    "feature_edge_hybrid",
    "edge_pair_hybrid",
]


@dataclass(frozen=True)
class Setting:
    function: str
    samples: int
    dimension: int
    top_m: int


@dataclass(frozen=True)
class Job:
    wave: str
    setting: Setting
    method: str
    eval_seeds: Tuple[int, ...]
    probe_seeds: Tuple[int, ...]
    probe_steps: int
    refit_steps: int
    width_hidden: int
    grid: int
    k: int
    lamb: float
    fd_points: int


def json_dumps(obj) -> str:
    return json.dumps(obj, sort_keys=True)


def parse_literal(value, default):
    if isinstance(value, (list, tuple, dict)):
        return value
    if value is None:
        return default
    try:
        if isinstance(value, float) and math.isnan(value):
            return default
    except TypeError:
        pass
    try:
        import ast

        return ast.literal_eval(str(value))
    except Exception:
        return default


def normalize_score(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float).reshape(-1)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.maximum(arr, 0.0)
    mx = float(np.max(arr)) if arr.size else 0.0
    if mx <= 0:
        return np.zeros_like(arr, dtype=float)
    return arr / mx


def normalize_pair_scores(pair_scores: Dict[Pair, float]) -> Dict[Pair, float]:
    vals = np.array([float(v) for v in pair_scores.values()], dtype=float)
    if vals.size == 0:
        return {}
    vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
    mx = float(np.max(np.maximum(vals, 0.0)))
    if mx <= 0:
        return {pair: 0.0 for pair in pair_scores}
    return {pair: max(float(score), 0.0) / mx for pair, score in pair_scores.items()}


def train_args(
    *,
    width_hidden: int,
    grid: int,
    k: int,
    steps: int,
    lamb: float,
    opt: str,
    update_grid: bool,
    grid_update_num: int,
    batch: int,
) -> SimpleNamespace:
    return SimpleNamespace(
        width_hidden=width_hidden,
        grid=grid,
        k=k,
        steps=steps,
        lamb=lamb,
        opt=opt,
        update_grid=update_grid,
        grid_update_num=grid_update_num,
        batch=batch,
    )


def safe_feature_score(model, d: int) -> np.ndarray:
    try:
        if hasattr(model, "attribute"):
            model.attribute(plot=False)
    except Exception:
        pass
    try:
        score = getattr(model, "feature_score")
        if score is None:
            return np.zeros(d, dtype=float)
        arr = score.detach().cpu().numpy().astype(float).reshape(-1)
        if len(arr) >= d:
            return normalize_score(arr[:d])
    except Exception:
        pass
    return np.zeros(d, dtype=float)


def safe_edge_path_scores(model, d: int) -> Tuple[np.ndarray, Dict[Pair, float], np.ndarray]:
    try:
        if hasattr(model, "attribute"):
            model.attribute(plot=False)
    except Exception:
        pass

    scales = getattr(model, "edge_actscale", None)
    if scales is None:
        scales = getattr(model, "acts_scale", None)
    if scales is None or len(scales) < 2:
        return np.zeros(d, dtype=float), {}, np.zeros(d, dtype=float)

    try:
        first = scales[0].detach().abs().cpu().numpy().astype(float)
        second = scales[1].detach().abs().cpu().numpy().astype(float)
    except Exception:
        return np.zeros(d, dtype=float), {}, np.zeros(d, dtype=float)

    if first.ndim != 2:
        return np.zeros(d, dtype=float), {}, np.zeros(d, dtype=float)
    if first.shape[1] == d:
        hidden_by_input = first
    elif first.shape[0] == d:
        hidden_by_input = first.T
    else:
        return np.zeros(d, dtype=float), {}, np.zeros(d, dtype=float)

    h = hidden_by_input.shape[0]
    out_scale = second.reshape(-1)
    if out_scale.size < h:
        out_scale = np.pad(out_scale, (0, h - out_scale.size), constant_values=1.0)
    out_scale = out_scale[:h]
    if float(np.max(out_scale)) <= 0:
        out_scale = np.ones(h, dtype=float)

    weighted = hidden_by_input * out_scale[:, None]
    var_score = normalize_score(weighted.sum(axis=0))

    pair_scores: Dict[Pair, float] = {}
    for i, j in itertools.combinations(range(d), 2):
        pair_scores[(i, j)] = float(np.sum(weighted[:, i] * weighted[:, j]))
    pair_scores = normalize_pair_scores(pair_scores)

    endpoint_mass = np.zeros(d, dtype=float)
    for (i, j), score in pair_scores.items():
        endpoint_mass[i] += float(score)
        endpoint_mass[j] += float(score)
    endpoint_mass = normalize_score(endpoint_mass)
    return var_score, pair_scores, endpoint_mass


def top_vars(score: np.ndarray, top_m: int) -> List[int]:
    score = np.asarray(score, dtype=float).reshape(-1)
    order = sorted(range(len(score)), key=lambda j: (-float(score[j]), int(j)))
    return [int(j) for j in order[:top_m]]


def support_from_pairs(pair_scores: Dict[Pair, float], fill_score: np.ndarray, top_m: int) -> List[int]:
    selected: List[int] = []
    seen = set()
    ranked_pairs = sorted(pair_scores.items(), key=lambda kv: (-float(kv[1]), int(kv[0][0]), int(kv[0][1])))
    for (i, j), _ in ranked_pairs:
        for v in (i, j):
            if v not in seen and len(selected) < top_m:
                selected.append(int(v))
                seen.add(int(v))
        if len(selected) >= min(top_m, 2):
            break
    for v in top_vars(fill_score, len(fill_score)):
        if len(selected) >= top_m:
            break
        if v not in seen:
            selected.append(int(v))
            seen.add(int(v))
    return sorted(selected[:top_m])


def pair_endpoint_mass(pair_scores: Dict[Pair, float], d: int) -> np.ndarray:
    out = np.zeros(d, dtype=float)
    for (i, j), score in pair_scores.items():
        out[int(i)] += float(score)
        out[int(j)] += float(score)
    return normalize_score(out)


def combine_scores(*scores: np.ndarray) -> np.ndarray:
    valid = [normalize_score(s) for s in scores if len(s) > 0]
    if not valid:
        return np.array([], dtype=float)
    return normalize_score(np.mean(np.vstack(valid), axis=0))


def aggregate_probe_scores(probes: pd.DataFrame, d: int) -> Dict[str, object]:
    if probes.empty:
        return {
            "grad": np.zeros(d, dtype=float),
            "feature": np.zeros(d, dtype=float),
            "edge": np.zeros(d, dtype=float),
            "edge_pairs": {},
            "edge_endpoint_mass": np.zeros(d, dtype=float),
        }

    grad_scores = []
    feature_scores = []
    edge_scores = []
    pair_accum: Dict[Pair, List[float]] = {}

    for _, row in probes.iterrows():
        grad_scores.append(normalize_score(np.asarray(parse_literal(row.get("grad_scores"), []), dtype=float)[:d]))
        feature_scores.append(normalize_score(np.asarray(parse_literal(row.get("feature_scores"), []), dtype=float)[:d]))
        edge_scores.append(normalize_score(np.asarray(parse_literal(row.get("edge_var_scores"), []), dtype=float)[:d]))

        raw_pairs = parse_literal(row.get("edge_pair_scores_top"), [])
        dense = {(i, j): 0.0 for i, j in itertools.combinations(range(d), 2)}
        for item in raw_pairs:
            if isinstance(item, (list, tuple)) and len(item) == 3:
                pair = tuple(sorted((int(item[0]), int(item[1]))))
                dense[pair] = float(item[2])
        dense = normalize_pair_scores(dense)
        for pair, score in dense.items():
            pair_accum.setdefault(pair, []).append(float(score))

    def mean_or_zero(rows: List[np.ndarray]) -> np.ndarray:
        rows = [r for r in rows if len(r) == d]
        if not rows:
            return np.zeros(d, dtype=float)
        return normalize_score(np.mean(np.vstack(rows), axis=0))

    pair_mean = {pair: float(np.mean(vals)) for pair, vals in pair_accum.items()}
    pair_mean = normalize_pair_scores(pair_mean)
    return {
        "grad": mean_or_zero(grad_scores),
        "feature": mean_or_zero(feature_scores),
        "edge": mean_or_zero(edge_scores),
        "edge_pairs": pair_mean,
        "edge_endpoint_mass": pair_endpoint_mass(pair_mean, d),
    }


def select_support(method: str, agg: Dict[str, object], top_m: int, d: int) -> Tuple[List[int], Dict[str, object]]:
    grad = np.asarray(agg["grad"], dtype=float)
    feature = np.asarray(agg["feature"], dtype=float)
    edge = np.asarray(agg["edge"], dtype=float)
    endpoint_mass = np.asarray(agg["edge_endpoint_mass"], dtype=float)
    pairs = dict(agg["edge_pairs"])

    if method == "grad_stability_var":
        score = grad
        selected = top_vars(score, top_m)
    elif method == "feature_stability_var":
        score = feature
        selected = top_vars(score, top_m)
    elif method == "edge_stability_var":
        score = edge
        selected = top_vars(score, top_m)
    elif method == "edge_endpoint_mass":
        score = combine_scores(endpoint_mass, edge)
        selected = top_vars(score, top_m)
    elif method == "feature_edge_hybrid":
        score = combine_scores(feature, edge, endpoint_mass)
        selected = top_vars(score, top_m)
    elif method == "edge_pair_endpoint":
        score = combine_scores(edge, endpoint_mass)
        selected = support_from_pairs(pairs, score, top_m)
    elif method == "edge_pair_hybrid":
        score = combine_scores(feature, edge, endpoint_mass)
        selected = support_from_pairs(pairs, score, top_m)
    else:
        raise ValueError(f"Unknown method={method}")

    ranked_pairs = sorted(pairs.items(), key=lambda kv: -float(kv[1]))[:10]
    return selected, {
        "selection_score": np.asarray(score, dtype=float).tolist(),
        "top_selection_variables": top_vars(score, min(12, d)),
        "top_edge_pairs": [(int(i), int(j), float(v)) for (i, j), v in ranked_pairs],
    }


def probe_key(function_name: str, samples: int, dimension: int, seed: int) -> str:
    return f"{function_name}__n{samples}__d{dimension}__seed{seed}"


def load_existing_probes(path: Path) -> pd.DataFrame:
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def train_probe(
    *,
    function_name: str,
    samples: int,
    test_samples: int,
    dimension: int,
    noise: float,
    seed: int,
    args,
    device: str,
) -> Dict:
    data = make_synthetic(
        function_name=function_name,
        n_train=samples,
        n_test=test_samples,
        d=dimension,
        noise=noise,
        seed=seed,
        standardize_target=True,
    )
    t0 = time.time()
    row = {
        "probe_key": probe_key(function_name, samples, dimension, seed),
        "function": function_name,
        "samples": samples,
        "dimension": dimension,
        "noise": noise,
        "seed": seed,
        "probe_steps": args.probe_steps,
        "status": "ok",
        "error": "",
        "runtime_sec": np.nan,
    }
    try:
        p_args = train_args(
            width_hidden=args.width_hidden,
            grid=args.grid,
            k=args.k,
            steps=args.probe_steps,
            lamb=args.lamb,
            opt=args.opt,
            update_grid=args.update_grid,
            grid_update_num=args.grid_update_num,
            batch=args.batch,
        )
        model = train_kan(
            data["X_train"],
            data["y_train"],
            data["X_test"],
            data["y_test"],
            p_args,
            seed=seed,
            device=device,
        )
        pred = batch_predict(model, data["X_test"], device=device, batch_size=args.pred_batch_size)
        grad_scores = normalize_score(
            gradient_importance(model, data["X_test"], device=device, points=args.probe_variable_points)
        )
        feature_scores = safe_feature_score(model, dimension)
        edge_scores, pair_scores, endpoint_mass = safe_edge_path_scores(model, dimension)
        top_pairs = sorted(pair_scores.items(), key=lambda kv: -float(kv[1]))[: args.keep_top_pairs]

        row.update(
            {
                "test_mse": mse_np(pred, data["y_test"]),
                "grad_scores": json_dumps(grad_scores.tolist()),
                "feature_scores": json_dumps(feature_scores.tolist()),
                "edge_var_scores": json_dumps(edge_scores.tolist()),
                "edge_endpoint_mass": json_dumps(endpoint_mass.tolist()),
                "edge_pair_scores_top": json_dumps(
                    [(int(i), int(j), float(v)) for (i, j), v in top_pairs]
                ),
                "top_grad_variables": json_dumps(top_vars(grad_scores, min(12, dimension))),
                "top_feature_variables": json_dumps(top_vars(feature_scores, min(12, dimension))),
                "top_edge_variables": json_dumps(top_vars(edge_scores, min(12, dimension))),
            }
        )
    except Exception as exc:
        row.update(
            {
                "status": "failed",
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
        )
    row["runtime_sec"] = float(time.time() - t0)
    return row


def ensure_probes(
    *,
    setting: Setting,
    probe_seeds: Sequence[int],
    args,
    device: str,
    probe_path: Path,
) -> pd.DataFrame:
    existing = load_existing_probes(probe_path)
    rows = existing.to_dict("records") if not existing.empty else []
    existing_keys = {str(r.get("probe_key")) for r in rows}

    for seed in probe_seeds:
        key = probe_key(setting.function, setting.samples, setting.dimension, int(seed))
        if key in existing_keys and not args.force_probe:
            continue
        print(f"[PROBE] {key}", flush=True)
        row = train_probe(
            function_name=setting.function,
            samples=setting.samples,
            test_samples=args.test_samples,
            dimension=setting.dimension,
            noise=args.noise,
            seed=int(seed),
            args=args,
            device=device,
        )
        rows.append(row)
        pd.DataFrame(rows).to_csv(probe_path, index=False)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    mask = (
        (df["function"].astype(str) == setting.function)
        & (pd.to_numeric(df["samples"], errors="coerce") == setting.samples)
        & (pd.to_numeric(df["dimension"], errors="coerce") == setting.dimension)
        & (df["status"].astype(str) == "ok")
        & (pd.to_numeric(df["seed"], errors="coerce").isin([int(s) for s in probe_seeds]))
    )
    return df[mask].copy()


def run_refit(
    *,
    job: Job,
    seed: int,
    support: Sequence[int],
    support_meta: Dict[str, object],
    args,
    device: str,
) -> Dict:
    setting = job.setting
    support = np.array(sorted(int(v) for v in support), dtype=int)
    data = make_synthetic(
        function_name=setting.function,
        n_train=setting.samples,
        n_test=args.test_samples,
        d=setting.dimension,
        noise=args.noise,
        seed=seed,
        standardize_target=True,
    )
    gt = data["ground_truth"]
    true_vars = tuple(int(v) for v in gt.active_variables)
    true_interactions = canonical_pairs(gt.interactions)
    Xtr = data["X_train"][:, support]
    Xte = data["X_test"][:, support]
    t0 = time.time()

    row = {
        "wave": job.wave,
        "method": job.method,
        "function": setting.function,
        "interaction_strength": function_to_c(setting.function),
        "samples": setting.samples,
        "dimension": setting.dimension,
        "noise": args.noise,
        "seed": seed,
        "top_m": setting.top_m,
        "selected_screen_features": support.tolist(),
        "true_variables": list(true_vars),
        "true_interactions": list(true_interactions),
        "num_true_variables": len(true_vars),
        "num_true_interactions": len(true_interactions),
        "formula": gt.formula,
        "probe_steps": job.probe_steps,
        "refit_steps": job.refit_steps,
        "width_hidden": job.width_hidden,
        "grid": job.grid,
        "k": job.k,
        "lamb": job.lamb,
        "fd_points": job.fd_points,
        "status": "ok",
        "error": "",
        "traceback": "",
        "runtime_sec": np.nan,
        **support_meta,
    }
    row.update(support_stats(support, true_vars, true_interactions))

    try:
        r_args = train_args(
            width_hidden=job.width_hidden,
            grid=job.grid,
            k=job.k,
            steps=job.refit_steps,
            lamb=job.lamb,
            opt=args.opt,
            update_grid=args.update_grid,
            grid_update_num=args.grid_update_num,
            batch=args.batch,
        )
        model = train_kan(Xtr, data["y_train"], Xte, data["y_test"], r_args, seed=seed, device=device)
        train_pred = batch_predict(model, Xtr, device=device, batch_size=args.pred_batch_size)
        test_pred = batch_predict(model, Xte, device=device, batch_size=args.pred_batch_size)

        local_var_scores = gradient_importance(model, Xte, device=device, points=args.refit_variable_points)
        full_var_scores = local_to_full_scores(local_var_scores, support, setting.dimension)
        if len(true_interactions) > 0:
            local_pair_scores = finite_difference_pair_scores(
                model,
                Xte,
                device=device,
                points=job.fd_points,
                h=args.fd_h,
                batch_size=args.pred_batch_size,
            ) if args.interaction_method == "fd" else None
            if args.interaction_method == "anova_abs":
                local_pair_scores = anova_pair_scores(
                    model,
                    Xte,
                    device=device,
                    points=args.anova_points,
                    background=args.anova_background,
                    batch_size=args.pred_batch_size,
                    score="abs",
                )
            elif args.interaction_method == "anova_var":
                local_pair_scores = anova_pair_scores(
                    model,
                    Xte,
                    device=device,
                    points=args.anova_points,
                    background=args.anova_background,
                    batch_size=args.pred_batch_size,
                    score="var",
                )
            elif args.interaction_method == "fd_anova_hybrid":
                fd_scores = finite_difference_pair_scores(
                    model,
                    Xte,
                    device=device,
                    points=job.fd_points,
                    h=args.fd_h,
                    batch_size=args.pred_batch_size,
                )
                anova_scores = anova_pair_scores(
                    model,
                    Xte,
                    device=device,
                    points=args.anova_points,
                    background=args.anova_background,
                    batch_size=args.pred_batch_size,
                    score="abs",
                )
                local_pair_scores = hybrid_pair_scores(fd_scores, anova_scores)
            elif args.interaction_method != "fd":
                raise ValueError(f"Unknown interaction_method={args.interaction_method}")
            full_pair_scores = local_to_full_pair_scores(local_pair_scores, support, setting.dimension)
        else:
            full_pair_scores = {}

        row.update(
            {
                "train_mse": mse_np(train_pred, data["y_train"]),
                "test_mse": mse_np(test_pred, data["y_test"]),
                "importance_scores": full_var_scores.tolist(),
                "interaction_method": args.interaction_method,
            }
        )
        var_eval = evaluate_variable_recovery(full_var_scores, true_vars)
        row.update(var_eval)
        row.update(endpoint_recovery(var_eval["selected_variables"], true_interactions, "explain"))
        row.update(evaluate_interaction_recovery(full_pair_scores, true_interactions))
    except Exception as exc:
        row.update(
            {
                "status": "failed",
                "error": repr(exc),
                "traceback": traceback.format_exc(),
                "train_mse": np.nan,
                "test_mse": np.nan,
                "variable_f1": np.nan,
                "interaction_f1": np.nan,
                "selected_variables": [],
                "selected_interactions": [],
            }
        )
    row["runtime_sec"] = float(time.time() - t0)
    return row


def summarize_results(detail: pd.DataFrame) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame()
    ok = detail[detail["status"].astype(str) == "ok"].copy()
    if ok.empty:
        return pd.DataFrame()
    group_cols = ["wave", "method", "function", "samples", "dimension", "top_m"]
    numeric_cols = [
        "train_mse",
        "test_mse",
        "screen_contains_all_true_vars",
        "screen_true_var_recall",
        "screen_contains_all_interaction_endpoints",
        "screen_interaction_endpoint_recall",
        "screen_contains_true_interactions",
        "explain_contains_all_interaction_endpoints",
        "explain_interaction_endpoint_recall",
        "variable_f1",
        "variable_auroc",
        "variable_auprc",
        "interaction_f1",
        "true_interaction_best_rank",
        "true_interaction_rank_mean",
        "true_interaction_mean_score_margin",
        "true_interaction_beats_all_false",
        "runtime_sec",
    ]
    for col in numeric_cols:
        if col in ok.columns:
            ok[col] = pd.to_numeric(ok[col], errors="coerce")
    agg = {}
    for col in numeric_cols:
        if col in ok.columns:
            if col in {
                "train_mse",
                "test_mse",
                "variable_f1",
                "interaction_f1",
                "true_interaction_rank_mean",
                "true_interaction_mean_score_margin",
                "runtime_sec",
            }:
                agg[col] = ["mean", "std"]
            else:
                agg[col] = ["mean"]
    out = ok.groupby(group_cols, dropna=False).agg(agg).reset_index()
    out.columns = ["_".join(str(x) for x in col if x != "").rstrip("_") for col in out.columns]
    counts = ok.groupby(group_cols, dropna=False).size().reset_index(name="num_runs")
    return out.merge(counts, on=group_cols, how="left")


def load_detail(path: Path) -> pd.DataFrame:
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def save_all(detail: pd.DataFrame, detail_path: Path, summary_path: Path, leaderboard_path: Path) -> None:
    detail_path.parent.mkdir(parents=True, exist_ok=True)
    detail.to_csv(detail_path, index=False)
    summary = summarize_results(detail)
    summary.to_csv(summary_path, index=False)
    if not summary.empty and "interaction_f1_mean" in summary.columns:
        leaderboard = summary.sort_values(
            ["interaction_f1_mean", "screen_contains_true_interactions_mean", "test_mse_mean"],
            ascending=[False, False, True],
        )
        leaderboard.to_csv(leaderboard_path, index=False)


def best_methods(detail: pd.DataFrame, default: Sequence[str], max_methods: int) -> Tuple[str, ...]:
    summary = summarize_results(detail)
    if summary.empty or "interaction_f1_mean" not in summary.columns:
        return tuple(default[:max_methods])
    recent = summary[summary["wave"].astype(str).isin(["broad", "exploit"])]
    if recent.empty:
        recent = summary
    by_method = (
        recent.groupby("method", dropna=False)["interaction_f1_mean"]
        .mean()
        .sort_values(ascending=False)
    )
    methods = [str(m) for m in by_method.index.tolist() if str(m) in METHODS]
    for method in default:
        if method not in methods:
            methods.append(method)
    return tuple(methods[:max_methods])


def build_jobs(args, completed: pd.DataFrame) -> List[Job]:
    broad_settings = [
        Setting("core_interaction_c025", 512, 100, 4),
        Setting("core_interaction_c025", 1024, 100, 4),
        Setting("core_interaction_c05", 512, 100, 4),
        Setting("core_interaction_c1", 512, 100, 4),
        Setting("core_interaction_c025", 1024, 100, 6),
        Setting("core_interaction_c05", 512, 100, 6),
    ]
    broad_methods = tuple(args.methods or METHODS)
    broad_jobs = [
        Job(
            wave="broad",
            setting=s,
            method=m,
            eval_seeds=tuple(args.broad_eval_seeds),
            probe_seeds=tuple(args.broad_probe_seeds),
            probe_steps=args.probe_steps,
            refit_steps=args.refit_steps,
            width_hidden=args.width_hidden,
            grid=args.grid,
            k=args.k,
            lamb=args.lamb,
            fd_points=args.fd_points,
        )
        for s in broad_settings
        for m in broad_methods
    ]

    chosen = best_methods(completed, default=("edge_pair_hybrid", "feature_edge_hybrid", "edge_stability_var"), max_methods=3)
    exploit_settings = [
        Setting("core_interaction_c025", 1024, 100, 5),
        Setting("core_interaction_c025", 1024, 100, 6),
        Setting("core_interaction_c025", 2048, 100, 6),
        Setting("core_interaction_c05", 512, 100, 5),
        Setting("core_interaction_c05", 1024, 100, 5),
        Setting("core_interaction_c1", 512, 100, 5),
    ]
    exploit_jobs = [
        Job(
            wave="exploit",
            setting=s,
            method=m,
            eval_seeds=tuple(args.exploit_eval_seeds),
            probe_seeds=tuple(args.exploit_probe_seeds),
            probe_steps=max(args.probe_steps, args.exploit_probe_steps),
            refit_steps=max(args.refit_steps, args.exploit_refit_steps),
            width_hidden=args.width_hidden,
            grid=args.grid,
            k=args.k,
            lamb=args.lamb,
            fd_points=args.fd_points,
        )
        for s in exploit_settings
        for m in chosen
    ]

    stress_methods = best_methods(completed, default=chosen, max_methods=2)
    stress_settings = []
    for dim in args.stress_dimensions:
        stress_settings.extend([
            Setting("core_interaction_c025", 2048, int(dim), 6),
            Setting("core_interaction_c05", 1024, int(dim), 6),
            Setting("core_interaction_c1", 1024, int(dim), 6),
        ])
    stress_settings.extend([
        Setting("feynman_energy", 1024, 100, 4),
        Setting("feynman_coulomb", 1024, 100, 5),
        Setting("feynman_gravity", 1024, 100, 5),
    ])
    stress_jobs = [
        Job(
            wave="stress",
            setting=s,
            method=m,
            eval_seeds=tuple(args.stress_eval_seeds),
            probe_seeds=tuple(args.stress_probe_seeds),
            probe_steps=max(args.probe_steps, args.stress_probe_steps),
            refit_steps=max(args.refit_steps, args.exploit_refit_steps),
            width_hidden=args.width_hidden,
            grid=args.grid,
            k=args.k,
            lamb=args.lamb,
            fd_points=max(16, args.fd_points // 2),
        )
        for s in stress_settings
        for m in stress_methods
    ]
    return broad_jobs + exploit_jobs + stress_jobs


def is_job_done(detail: pd.DataFrame, job: Job) -> bool:
    if detail.empty:
        return False
    s = job.setting
    mask = (
        (detail["wave"].astype(str) == job.wave)
        & (detail["method"].astype(str) == job.method)
        & (detail["function"].astype(str) == s.function)
        & (pd.to_numeric(detail["samples"], errors="coerce") == s.samples)
        & (pd.to_numeric(detail["dimension"], errors="coerce") == s.dimension)
        & (pd.to_numeric(detail["top_m"], errors="coerce") == s.top_m)
        & (pd.to_numeric(detail["seed"], errors="coerce").isin([int(x) for x in job.eval_seeds]))
    )
    return int(mask.sum()) >= len(job.eval_seeds)


def run_job(job: Job, args, device: str, paths: Dict[str, Path], detail: pd.DataFrame) -> pd.DataFrame:
    setting = job.setting
    print(
        f"[JOB] wave={job.wave} method={job.method} function={setting.function} "
        f"n={setting.samples} d={setting.dimension} top_m={setting.top_m}",
        flush=True,
    )
    probe_args = argparse.Namespace(**vars(args))
    probe_args.probe_steps = job.probe_steps
    probe_args.width_hidden = job.width_hidden
    probe_args.grid = job.grid
    probe_args.k = job.k
    probe_args.lamb = job.lamb
    probes = ensure_probes(
        setting=setting,
        probe_seeds=job.probe_seeds,
        args=probe_args,
        device=device,
        probe_path=paths["probes"],
    )
    rows = detail.to_dict("records") if not detail.empty else []
    existing = detail.copy()

    for seed in job.eval_seeds:
        if not existing.empty:
            duplicate = existing[
                (existing["wave"].astype(str) == job.wave)
                & (existing["method"].astype(str) == job.method)
                & (existing["function"].astype(str) == setting.function)
                & (pd.to_numeric(existing["samples"], errors="coerce") == setting.samples)
                & (pd.to_numeric(existing["dimension"], errors="coerce") == setting.dimension)
                & (pd.to_numeric(existing["top_m"], errors="coerce") == setting.top_m)
                & (pd.to_numeric(existing["seed"], errors="coerce") == int(seed))
            ]
            if not duplicate.empty and not args.force_refit:
                continue

        stable = probes[pd.to_numeric(probes["seed"], errors="coerce") != int(seed)].copy()
        if stable.empty:
            stable = probes.copy()
        agg = aggregate_probe_scores(stable, setting.dimension)
        support, meta = select_support(job.method, agg, setting.top_m, setting.dimension)
        meta.update(
            {
                "num_probe_rows": int(len(stable)),
                "probe_seed_pool": [int(s) for s in stable["seed"].tolist()],
                "selection_meta": json_dumps(meta),
            }
        )
        print(f"[REFIT] seed={seed} support={support}", flush=True)
        row = run_refit(job=job, seed=int(seed), support=support, support_meta=meta, args=args, device=device)
        rows.append(row)
        detail = pd.DataFrame(rows)
        save_all(detail, paths["detail"], paths["summary"], paths["leaderboard"])
    return pd.DataFrame(rows)


def write_status(paths: Dict[str, Path], message: str) -> None:
    paths["status"].parent.mkdir(parents=True, exist_ok=True)
    paths["status"].write_text(message + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="results/innovation_loop/run")
    parser.add_argument("--time_budget_hours", type=float, default=10.0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--test_samples", type=int, default=4096)
    parser.add_argument("--methods", nargs="+", default=None)

    parser.add_argument("--broad_eval_seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--broad_probe_seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5])
    parser.add_argument("--exploit_eval_seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--exploit_probe_seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    parser.add_argument("--stress_eval_seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--stress_probe_seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5])
    parser.add_argument("--stress_dimensions", nargs="+", type=int, default=[500])

    parser.add_argument("--grid", type=int, default=5)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--width_hidden", type=int, default=8)
    parser.add_argument("--lamb", type=float, default=0.001)
    parser.add_argument("--probe_steps", type=int, default=35)
    parser.add_argument("--refit_steps", type=int, default=50)
    parser.add_argument("--exploit_probe_steps", type=int, default=50)
    parser.add_argument("--exploit_refit_steps", type=int, default=80)
    parser.add_argument("--stress_probe_steps", type=int, default=35)
    parser.add_argument("--opt", default="LBFGS")
    parser.add_argument("--update_grid", action="store_true")
    parser.add_argument("--no_update_grid", dest="update_grid", action="store_false")
    parser.set_defaults(update_grid=False)
    parser.add_argument("--grid_update_num", type=int, default=10)
    parser.add_argument("--batch", type=int, default=-1)
    parser.add_argument("--probe_variable_points", type=int, default=512)
    parser.add_argument("--refit_variable_points", type=int, default=512)
    parser.add_argument(
        "--interaction_method",
        choices=["fd", "anova_abs", "anova_var", "fd_anova_hybrid"],
        default="fd",
    )
    parser.add_argument("--fd_points", type=int, default=128)
    parser.add_argument("--fd_h", type=float, default=1e-2)
    parser.add_argument("--anova_points", type=int, default=64)
    parser.add_argument("--anova_background", type=int, default=64)
    parser.add_argument("--pred_batch_size", type=int, default=4096)
    parser.add_argument("--keep_top_pairs", type=int, default=120)
    parser.add_argument("--force_probe", action="store_true")
    parser.add_argument("--force_refit", action="store_true")

    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    paths = {
        "out_dir": out_dir,
        "detail": out_dir / "innovation_detail.csv",
        "summary": out_dir / "innovation_summary.csv",
        "leaderboard": out_dir / "innovation_leaderboard.csv",
        "probes": out_dir / "probe_cache.csv",
        "status": out_dir / "STATUS.txt",
    }
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    start = time.time()
    deadline = start + max(args.time_budget_hours, 0.01) * 3600.0
    write_status(paths, f"START device={device} deadline_epoch={deadline:.0f}")
    print(f"Using device={device}", flush=True)
    print(f"Writing results under {out_dir}", flush=True)

    detail = load_detail(paths["detail"])
    job_cursor = 0
    while time.time() < deadline:
        jobs = build_jobs(args, detail)
        pending = [job for job in jobs if not is_job_done(detail, job)]
        if not pending:
            break
        job = pending[0]
        remaining = deadline - time.time()
        if remaining < 120:
            print("[STOP] Less than two minutes left in budget.", flush=True)
            break
        write_status(
            paths,
            f"RUNNING cursor={job_cursor} wave={job.wave} method={job.method} "
            f"function={job.setting.function} n={job.setting.samples} d={job.setting.dimension}",
        )
        detail = run_job(job, args, device, paths, detail)
        job_cursor += 1

    save_all(detail, paths["detail"], paths["summary"], paths["leaderboard"])
    elapsed = time.time() - start
    write_status(paths, f"DONE elapsed_sec={elapsed:.1f} detail={paths['detail']}")
    print(f"DONE elapsed_sec={elapsed:.1f}", flush=True)
    print(f"Detail: {paths['detail']}", flush=True)
    print(f"Summary: {paths['summary']}", flush=True)
    print(f"Leaderboard: {paths['leaderboard']}", flush=True)


if __name__ == "__main__":
    main()
