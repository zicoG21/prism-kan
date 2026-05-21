from __future__ import annotations

import argparse
import itertools
import sys
import traceback
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import average_precision_score, roc_auc_score

from src.data import make_synthetic


Pair = Tuple[int, int]


# ============================================================
# Basic utilities
# ============================================================

def mse_np(pred: np.ndarray, target: np.ndarray) -> float:
    pred = pred.reshape(-1, 1)
    target = target.reshape(-1, 1)
    return float(np.mean((pred - target) ** 2))


def f1_from_sets(pred: set, true: set) -> Tuple[float, float, float]:
    if len(true) == 0:
        return np.nan, np.nan, np.nan
    if len(pred) == 0:
        return 0.0, 0.0, 0.0
    tp = len(pred & true)
    precision = tp / len(pred)
    recall = tp / len(true)
    f1 = 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)
    return precision, recall, f1


def safe_auroc_auprc(y_true: np.ndarray, scores: np.ndarray) -> Tuple[float, float]:
    try:
        auroc = float(roc_auc_score(y_true, scores))
    except ValueError:
        auroc = np.nan
    try:
        auprc = float(average_precision_score(y_true, scores))
    except ValueError:
        auprc = np.nan
    return auroc, auprc


def normalize_function_alias(name: str) -> str:
    aliases = {
        "core_c5": "core_interaction_c5",
        "proxy": "correlated_proxy",
    }
    return aliases.get(name, name)


def all_pairs(d: int) -> List[Pair]:
    return [(i, j) for i, j in itertools.combinations(range(d), 2)]


def canonical_pairs(pairs: Sequence[Tuple[int, int]]) -> Tuple[Pair, ...]:
    return tuple(tuple(sorted((int(i), int(j)))) for i, j in pairs)


def interaction_endpoints(true_interactions: Sequence[Pair]) -> Tuple[int, ...]:
    s = set()
    for i, j in true_interactions:
        s.add(int(i))
        s.add(int(j))
    return tuple(sorted(s))


# ============================================================
# KAN training / prediction
# ============================================================

def predict_model(model, X_np: np.ndarray, batch_size: int | None = None) -> np.ndarray:
    if batch_size is None or len(X_np) <= batch_size:
        X = torch.tensor(X_np, dtype=torch.float32)
        with torch.no_grad():
            y = model(X)
        return y.detach().cpu().numpy().reshape(-1, 1)

    outs = []
    for start in range(0, len(X_np), batch_size):
        X = torch.tensor(X_np[start:start + batch_size], dtype=torch.float32)
        with torch.no_grad():
            y = model(X)
        outs.append(y.detach().cpu().numpy().reshape(-1, 1))
    return np.vstack(outs)


def fit_kan(model, dataset, opt: str, steps: int, lamb: float, lr: float | None, update_grid_mode: str):
    kwargs = {
        "opt": opt,
        "steps": steps,
        "lamb": lamb,
    }
    if lr is not None:
        kwargs["lr"] = lr

    if update_grid_mode == "false":
        kwargs["update_grid"] = False
    elif update_grid_mode == "true":
        kwargs["update_grid"] = True
    elif update_grid_mode == "default":
        pass
    else:
        raise ValueError(f"Unknown update_grid_mode={update_grid_mode}")

    # pykan versions differ in accepted kwargs.
    try:
        return model.fit(dataset, **kwargs)
    except TypeError:
        kwargs.pop("lr", None)
        try:
            return model.fit(dataset, **kwargs)
        except TypeError:
            kwargs.pop("update_grid", None)
            return model.fit(dataset, **kwargs)


def train_pykan(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    width_hidden: int,
    grid: int,
    k: int,
    opt: str,
    steps: int,
    lamb: float,
    lr: float | None,
    update_grid_mode: str,
    seed: int,
):
    try:
        from kan import KAN
    except Exception as exc:
        raise ImportError("Could not import pykan. Run this in the same env as your previous KAN experiments.") from exc

    torch.manual_seed(seed)
    np.random.seed(seed)

    d = X_train.shape[1]
    model = KAN(width=[d, width_hidden, 1], grid=grid, k=k, seed=seed)

    dataset = {
        "train_input": torch.tensor(X_train, dtype=torch.float32),
        "train_label": torch.tensor(y_train, dtype=torch.float32),
        "test_input": torch.tensor(X_test, dtype=torch.float32),
        "test_label": torch.tensor(y_test, dtype=torch.float32),
    }

    fit_kan(
        model=model,
        dataset=dataset,
        opt=opt,
        steps=steps,
        lamb=lamb,
        lr=lr,
        update_grid_mode=update_grid_mode,
    )
    return model


# ============================================================
# Screening
# ============================================================

def rf_scores(X: np.ndarray, y: np.ndarray, seed: int, n_estimators: int) -> np.ndarray:
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=seed,
        n_jobs=-1,
        min_samples_leaf=2,
    )
    rf.fit(X, y.reshape(-1))
    return np.asarray(rf.feature_importances_, dtype=float)


def fill_with_support(
    must_include: Sequence[int],
    candidate_pool: Sequence[int],
    top_m: int,
    rng: np.random.Generator,
    scores: np.ndarray | None = None,
) -> np.ndarray:
    selected = []
    seen = set()

    for v in must_include:
        v = int(v)
        if v not in seen:
            selected.append(v)
            seen.add(v)

    remaining = [int(v) for v in candidate_pool if int(v) not in seen]
    need = max(0, top_m - len(selected))

    if need > 0:
        if scores is None:
            rng.shuffle(remaining)
            selected.extend(remaining[:need])
        else:
            remaining = sorted(remaining, key=lambda v: float(scores[v]), reverse=True)
            selected.extend(remaining[:need])

    return np.array(sorted(selected), dtype=int)


def select_features(
    mode: str,
    X: np.ndarray,
    y: np.ndarray,
    top_m: int,
    true_vars: Sequence[int],
    true_interactions: Sequence[Pair],
    seed: int,
    rf_trees: int,
) -> Tuple[np.ndarray, np.ndarray, str]:
    d = X.shape[1]
    all_vars = list(range(d))
    rng = np.random.default_rng(seed * 1009 + 17)

    true_vars = tuple(int(v) for v in true_vars)
    endpoints = set(interaction_endpoints(true_interactions))

    scores = np.zeros(d, dtype=float)

    if mode == "raw":
        return np.arange(d, dtype=int), scores, "none"

    if mode == "random":
        selected = np.array(sorted(rng.choice(d, size=min(top_m, d), replace=False).astype(int)), dtype=int)
        return selected, scores, "random"

    if mode == "oracle_support":
        selected = fill_with_support(true_vars, all_vars, top_m, rng, scores=None)
        return selected, scores, "forced_true_support_random_fill"

    if mode == "oracle_interaction":
        selected = fill_with_support(sorted(endpoints), all_vars, top_m, rng, scores=None)
        return selected, scores, "forced_interaction_endpoints_random_fill"

    if mode == "exclude_interaction":
        pool = [v for v in all_vars if v not in endpoints]
        selected = np.array(sorted(rng.choice(pool, size=min(top_m, len(pool)), replace=False).astype(int)), dtype=int)
        return selected, scores, "random_excluding_interaction_endpoints"

    if mode == "rf":
        scores = rf_scores(X, y, seed=seed, n_estimators=rf_trees)
        selected = np.array(sorted(np.argsort(-scores)[:min(top_m, d)].astype(int)), dtype=int)
        return selected, scores, "rf"

    if mode == "rf_exclude_interaction":
        scores = rf_scores(X, y, seed=seed, n_estimators=rf_trees)
        pool = [v for v in all_vars if v not in endpoints]
        selected = np.array(sorted(sorted(pool, key=lambda v: float(scores[v]), reverse=True)[:top_m]), dtype=int)
        return selected, scores, "rf_excluding_interaction_endpoints"

    raise ValueError(f"Unknown screen_mode={mode}")


def support_stats(selected_features: np.ndarray, true_vars: Sequence[int], true_interactions: Sequence[Pair]) -> Dict:
    selected = set(int(v) for v in selected_features)
    true_var_set = set(int(v) for v in true_vars)
    endpoints = set(interaction_endpoints(true_interactions))

    return {
        "effective_dim": len(selected_features),
        "screen_contains_all_true_vars": int(true_var_set.issubset(selected)) if true_var_set else np.nan,
        "screen_true_var_recall": len(true_var_set & selected) / len(true_var_set) if true_var_set else np.nan,
        "screen_contains_all_interaction_endpoints": int(endpoints.issubset(selected)) if endpoints else np.nan,
        "screen_interaction_endpoint_recall": len(endpoints & selected) / len(endpoints) if endpoints else np.nan,
        "screen_contains_true_interactions": int(
            all(int(i) in selected and int(j) in selected for i, j in true_interactions)
        ) if true_interactions else np.nan,
    }


# ============================================================
# Variable explanations
# ============================================================

def gradient_importance(model, X_np: np.ndarray) -> np.ndarray:
    X = torch.tensor(X_np, dtype=torch.float32, requires_grad=True)
    y = model(X).sum()
    grad = torch.autograd.grad(y, X, create_graph=False)[0]
    return grad.abs().mean(dim=0).detach().cpu().numpy()


def local_to_full_scores(local_scores: np.ndarray, selected_features: np.ndarray, d_full: int) -> np.ndarray:
    full_scores = np.zeros(d_full, dtype=float)
    for local_idx, original_idx in enumerate(selected_features):
        full_scores[int(original_idx)] = float(local_scores[local_idx])
    return full_scores


def evaluate_variable_recovery(full_scores: np.ndarray, true_vars: Sequence[int]) -> Dict:
    true_set = set(int(v) for v in true_vars)
    if len(true_set) == 0:
        return {
            "selected_variables": [],
            "variable_precision": np.nan,
            "variable_recall": np.nan,
            "variable_f1": np.nan,
            "variable_auroc": np.nan,
            "variable_auprc": np.nan,
        }

    k = len(true_set)
    selected = set(int(i) for i in np.argsort(-full_scores)[:k])
    precision, recall, f1 = f1_from_sets(selected, true_set)

    y_true = np.array([1 if i in true_set else 0 for i in range(len(full_scores))], dtype=int)
    auroc, auprc = safe_auroc_auprc(y_true, full_scores)

    return {
        "selected_variables": sorted(selected),
        "variable_precision": precision,
        "variable_recall": recall,
        "variable_f1": f1,
        "variable_auroc": auroc,
        "variable_auprc": auprc,
    }


# ============================================================
# Interaction scores
# ============================================================

def local_to_full_pair_scores(
    local_pair_scores: Dict[Pair, float],
    selected_features: np.ndarray,
    d_full: int,
) -> Dict[Pair, float]:
    full_pair_scores: Dict[Pair, float] = {}
    selected_features = np.asarray(selected_features, dtype=int)

    for (i_local, j_local), score in local_pair_scores.items():
        if i_local >= len(selected_features) or j_local >= len(selected_features):
            continue
        i = int(selected_features[i_local])
        j = int(selected_features[j_local])
        full_pair_scores[tuple(sorted((i, j)))] = float(score)

    # Pairs absent from a screened model are assigned zero.
    for i, j in itertools.combinations(range(d_full), 2):
        full_pair_scores.setdefault((i, j), 0.0)

    return full_pair_scores


def evaluate_interaction_recovery(pair_scores: Dict[Pair, float], true_interactions: Sequence[Pair]) -> Dict:
    true_set = set(canonical_pairs(true_interactions))
    if len(true_set) == 0:
        return {
            "selected_interactions": [],
            "interaction_precision": np.nan,
            "interaction_recall": np.nan,
            "interaction_f1": np.nan,
            "true_interaction_score_mean": np.nan,
            "max_nontrue_interaction_score": np.nan,
        }

    k = len(true_set)
    ranked = sorted(pair_scores.items(), key=lambda kv: kv[1], reverse=True)
    selected = {pair for pair, _ in ranked[:k]}
    precision, recall, f1 = f1_from_sets(selected, true_set)

    true_scores = [float(pair_scores.get(pair, 0.0)) for pair in true_set]
    nontrue_scores = [float(v) for pair, v in pair_scores.items() if pair not in true_set]

    return {
        "selected_interactions": sorted(selected),
        "interaction_precision": precision,
        "interaction_recall": recall,
        "interaction_f1": f1,
        "true_interaction_score_mean": float(np.mean(true_scores)) if true_scores else np.nan,
        "max_nontrue_interaction_score": float(np.max(nontrue_scores)) if nontrue_scores else np.nan,
    }


def hessian_interaction_scores(model, X_np: np.ndarray, points: int) -> Dict[Pair, float]:
    d = X_np.shape[1]
    n = min(points, X_np.shape[0])
    X_sub = torch.tensor(X_np[:n], dtype=torch.float32)

    score_mat = torch.zeros((d, d), dtype=torch.float32)

    def scalar_func(z: torch.Tensor) -> torch.Tensor:
        return model(z.unsqueeze(0)).sum()

    for idx in range(n):
        z = X_sub[idx].clone().detach().requires_grad_(True)
        H = torch.autograd.functional.hessian(scalar_func, z)
        score_mat += H.abs().detach()

    score_mat = score_mat / max(n, 1)
    return {(i, j): float(score_mat[i, j].item()) for i, j in itertools.combinations(range(d), 2)}


def finite_difference_interaction_scores(
    model,
    X_np: np.ndarray,
    points: int,
    h: float,
    batch_size: int,
) -> Dict[Pair, float]:
    """Mixed finite difference interaction score.

    I_ij = E | f(x+h e_i+h e_j)-f(x+h e_i)-f(x+h e_j)+f(x) | / h^2

    This is a non-autograd check for Hessian-based ranking.
    """
    d = X_np.shape[1]
    n = min(points, X_np.shape[0])
    X = X_np[:n].copy()

    f0 = predict_model(model, X, batch_size=batch_size).reshape(-1)
    scores: Dict[Pair, float] = {}

    for i, j in itertools.combinations(range(d), 2):
        X_i = X.copy()
        X_j = X.copy()
        X_ij = X.copy()

        X_i[:, i] += h
        X_j[:, j] += h
        X_ij[:, i] += h
        X_ij[:, j] += h

        fi = predict_model(model, X_i, batch_size=batch_size).reshape(-1)
        fj = predict_model(model, X_j, batch_size=batch_size).reshape(-1)
        fij = predict_model(model, X_ij, batch_size=batch_size).reshape(-1)

        mixed = (fij - fi - fj + f0) / (h ** 2)
        scores[(i, j)] = float(np.mean(np.abs(mixed)))

    return scores


def pair_permutation_scores(
    model,
    X_np: np.ndarray,
    points: int,
    seed: int,
    batch_size: int,
) -> Tuple[Dict[Pair, float], Dict[Pair, float]]:
    """Pair permutation interaction checks.

    joint score:
        E[(f(X_{i,j permuted}) - f(X))^2]
    synergy score:
        | joint - single_i - single_j |

    The synergy version is closer to a pure interaction score, while the joint version
    measures total pair reliance.
    """
    d = X_np.shape[1]
    n = min(points, X_np.shape[0])
    X = X_np[:n].copy()
    rng = np.random.default_rng(seed + 9091)

    base = predict_model(model, X, batch_size=batch_size).reshape(-1)

    single_delta = np.zeros(d, dtype=float)
    perms = {}
    for i in range(d):
        p = rng.permutation(n)
        perms[i] = p
        X_i = X.copy()
        X_i[:, i] = X_i[p, i]
        pred_i = predict_model(model, X_i, batch_size=batch_size).reshape(-1)
        single_delta[i] = float(np.mean((pred_i - base) ** 2))

    joint_scores: Dict[Pair, float] = {}
    synergy_scores: Dict[Pair, float] = {}

    for i, j in itertools.combinations(range(d), 2):
        X_ij = X.copy()
        X_ij[:, i] = X_ij[perms[i], i]
        X_ij[:, j] = X_ij[perms[j], j]
        pred_ij = predict_model(model, X_ij, batch_size=batch_size).reshape(-1)

        joint = float(np.mean((pred_ij - base) ** 2))
        synergy = abs(joint - single_delta[i] - single_delta[j])

        joint_scores[(i, j)] = joint
        synergy_scores[(i, j)] = float(synergy)

    return joint_scores, synergy_scores


# ============================================================
# Optional KAN path-pair reliance
# ============================================================

def _get_first_layer_mask(model):
    if not hasattr(model, "act_fun"):
        return None
    try:
        layer0 = model.act_fun[0]
        if hasattr(layer0, "mask"):
            return layer0.mask
    except Exception:
        return None
    return None


def path_pair_deletion_scores(
    model,
    X_np: np.ndarray,
    y_np: np.ndarray,
    candidate_pairs: Sequence[Pair],
    batch_size: int,
) -> Tuple[Dict[Pair, float], Dict[Pair, float]]:
    """KAN-specific pair path deletion.

    joint score:
        MSE(delete i and j feature paths) - base MSE.
    synergy score:
        | joint_delta - delta_i - delta_j |

    This is architecture-specific and should be used as a functional-reliance check,
    not as the only interaction metric.
    """
    mask = _get_first_layer_mask(model)
    if mask is None:
        return {}, {}

    base_pred = predict_model(model, X_np, batch_size=batch_size)
    base_mse = mse_np(base_pred, y_np)

    original = mask.data.clone()

    def delete_features(local_features: Sequence[int]) -> float:
        mask.data.copy_(original)
        for idx in local_features:
            if 0 <= int(idx) < mask.shape[0]:
                mask.data[int(idx), :] = 0
        pred = predict_model(model, X_np, batch_size=batch_size)
        return mse_np(pred, y_np) - base_mse

    # Cache single feature deletion.
    features = sorted(set(i for p in candidate_pairs for i in p) | set(j for p in candidate_pairs for j in p))
    single_delta = {i: delete_features([i]) for i in features}

    joint_scores: Dict[Pair, float] = {}
    synergy_scores: Dict[Pair, float] = {}

    for i, j in candidate_pairs:
        joint = delete_features([i, j])
        synergy = abs(joint - single_delta.get(i, 0.0) - single_delta.get(j, 0.0))
        joint_scores[(i, j)] = float(max(joint, 0.0))
        synergy_scores[(i, j)] = float(synergy)

    mask.data.copy_(original)
    return joint_scores, synergy_scores


def build_path_candidate_pairs(
    selected_features: np.ndarray,
    true_interactions: Sequence[Pair],
    metric_pair_scores: Dict[str, Dict[Pair, float]],
    max_pairs: int,
) -> List[Pair]:
    """Build local candidate pairs for optional path deletion.

    Includes true interaction pairs when present in the screened local coordinates,
    plus top pairs from each already-computed interaction metric.
    """
    full_to_local = {int(v): idx for idx, v in enumerate(selected_features)}

    candidates: set[Pair] = set()

    for i, j in canonical_pairs(true_interactions):
        if i in full_to_local and j in full_to_local:
            candidates.add(tuple(sorted((full_to_local[i], full_to_local[j]))))

    # metric_pair_scores are local pair scores at this stage.
    for scores in metric_pair_scores.values():
        for pair, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:max_pairs]:
            candidates.add(tuple(sorted(pair)))

    return sorted(candidates)[:max_pairs]


# ============================================================
# Run one config
# ============================================================

def run_one(args, function_name: str, seed: int, screen_mode: str) -> List[Dict]:
    function_name = normalize_function_alias(function_name)

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
    true_interactions = canonical_pairs(gt.interactions)

    selected_features, screen_scores, screen_desc = select_features(
        mode=screen_mode,
        X=X_train,
        y=y_train,
        top_m=args.top_m,
        true_vars=true_vars,
        true_interactions=true_interactions,
        seed=seed,
        rf_trees=args.rf_trees,
    )

    X_train_s = X_train[:, selected_features]
    X_test_s = X_test[:, selected_features]

    base = {
        "model": "KAN_INTERACTION_METRIC_VALIDATION",
        "function": function_name,
        "seed": seed,
        "samples": args.samples,
        "test_samples": args.test_samples,
        "dimension": args.dimension,
        "noise": args.noise,
        "screen_mode": screen_mode,
        "screen_score_type": screen_desc,
        "top_m_requested": args.top_m,
        "selected_screen_features": selected_features.tolist(),
        "screen_scores": screen_scores.tolist(),
        "kan_width": args.kan_width,
        "kan_grid": args.kan_grid,
        "kan_k": args.kan_k,
        "kan_steps": args.kan_steps,
        "kan_lamb": args.kan_lamb,
        "opt": args.opt,
        "lr": args.lr,
        "update_grid_mode": args.update_grid_mode,
        "true_variables": list(true_vars),
        "true_interactions": list(true_interactions),
        "num_true_variables": len(true_vars),
        "num_true_interactions": len(true_interactions),
        "formula": gt.formula,
    }
    base.update(support_stats(selected_features, true_vars, true_interactions))

    try:
        model = train_pykan(
            X_train=X_train_s,
            y_train=y_train,
            X_test=X_test_s,
            y_test=y_test,
            width_hidden=args.kan_width,
            grid=args.kan_grid,
            k=args.kan_k,
            opt=args.opt,
            steps=args.kan_steps,
            lamb=args.kan_lamb,
            lr=args.lr,
            update_grid_mode=args.update_grid_mode,
            seed=seed,
        )

        train_mse = mse_np(predict_model(model, X_train_s, batch_size=args.batch_size), y_train)
        test_mse = mse_np(predict_model(model, X_test_s, batch_size=args.batch_size), y_test)

        # Variable recovery for context.
        local_var_scores = gradient_importance(model, X_test_s[:args.variable_points])
        full_var_scores = local_to_full_scores(local_var_scores, selected_features, args.dimension)
        var_eval = evaluate_variable_recovery(full_var_scores, true_vars)

        # Interaction metrics.
        rows: List[Dict] = []

        local_metric_scores: Dict[str, Dict[Pair, float]] = {}

        if "hessian" in args.metrics and len(true_interactions) > 0:
            local_scores = hessian_interaction_scores(model, X_test_s, args.hessian_points)
            local_metric_scores["hessian"] = local_scores
            full_scores = local_to_full_pair_scores(local_scores, selected_features, args.dimension)
            rows.append(make_metric_row(base, var_eval, train_mse, test_mse, "hessian", full_scores, true_interactions))

        if "fd" in args.metrics and len(true_interactions) > 0:
            local_scores = finite_difference_interaction_scores(
                model=model,
                X_np=X_test_s,
                points=args.fd_points,
                h=args.fd_h,
                batch_size=args.batch_size,
            )
            local_metric_scores["finite_difference"] = local_scores
            full_scores = local_to_full_pair_scores(local_scores, selected_features, args.dimension)
            rows.append(make_metric_row(base, var_eval, train_mse, test_mse, "finite_difference", full_scores, true_interactions))

        if "pairperm" in args.metrics and len(true_interactions) > 0:
            local_joint, local_synergy = pair_permutation_scores(
                model=model,
                X_np=X_test_s,
                points=args.perm_points,
                seed=seed,
                batch_size=args.batch_size,
            )
            local_metric_scores["pair_permutation_synergy"] = local_synergy

            full_joint = local_to_full_pair_scores(local_joint, selected_features, args.dimension)
            rows.append(make_metric_row(base, var_eval, train_mse, test_mse, "pair_permutation_joint", full_joint, true_interactions))

            full_synergy = local_to_full_pair_scores(local_synergy, selected_features, args.dimension)
            rows.append(make_metric_row(base, var_eval, train_mse, test_mse, "pair_permutation_synergy", full_synergy, true_interactions))

        if "path" in args.metrics and len(true_interactions) > 0:
            path_pairs = build_path_candidate_pairs(
                selected_features=selected_features,
                true_interactions=true_interactions,
                metric_pair_scores=local_metric_scores,
                max_pairs=args.path_max_pairs,
            )
            local_joint, local_synergy = path_pair_deletion_scores(
                model=model,
                X_np=X_test_s[:args.path_points],
                y_np=y_test[:args.path_points],
                candidate_pairs=path_pairs,
                batch_size=args.batch_size,
            )
            # Note: path scores are only computed for a candidate subset. Missing pairs are zero.
            full_joint = local_to_full_pair_scores(local_joint, selected_features, args.dimension)
            rows.append(make_metric_row(base, var_eval, train_mse, test_mse, "kan_path_pair_joint", full_joint, true_interactions))

            full_synergy = local_to_full_pair_scores(local_synergy, selected_features, args.dimension)
            rows.append(make_metric_row(base, var_eval, train_mse, test_mse, "kan_path_pair_synergy", full_synergy, true_interactions))

        if not rows:
            row = dict(base)
            row.update({
                "status": "ok",
                "error": "",
                "traceback": "",
                "metric": "no_interactions_or_no_metric",
                "train_mse": train_mse,
                "test_mse": test_mse,
            })
            row.update(var_eval)
            row.update(evaluate_interaction_recovery({}, true_interactions))
            rows.append(row)

        return rows

    except Exception as exc:
        row = dict(base)
        row.update({
            "status": "failed",
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "metric": "failed",
            "train_mse": np.nan,
            "test_mse": np.nan,
            "selected_variables": [],
            "variable_precision": np.nan,
            "variable_recall": np.nan,
            "variable_f1": np.nan,
            "variable_auroc": np.nan,
            "variable_auprc": np.nan,
            "selected_interactions": [],
            "interaction_precision": np.nan,
            "interaction_recall": np.nan,
            "interaction_f1": np.nan,
            "true_interaction_score_mean": np.nan,
            "max_nontrue_interaction_score": np.nan,
        })
        print(f"[WARN] failed function={function_name}, seed={seed}, mode={screen_mode}: {exc}")
        return [row]


def make_metric_row(
    base: Dict,
    var_eval: Dict,
    train_mse: float,
    test_mse: float,
    metric_name: str,
    full_pair_scores: Dict[Pair, float],
    true_interactions: Sequence[Pair],
) -> Dict:
    row = dict(base)
    row.update({
        "status": "ok",
        "error": "",
        "traceback": "",
        "metric": metric_name,
        "train_mse": train_mse,
        "test_mse": test_mse,
        # Large column, but useful for debugging. Remove later if file gets too large.
        "pair_scores": sorted([(int(i), int(j), float(v)) for (i, j), v in full_pair_scores.items()], key=lambda x: x[2], reverse=True)[:50],
    })
    row.update(var_eval)
    row.update(evaluate_interaction_recovery(full_pair_scores, true_interactions))
    return row


# ============================================================
# Save / summarize / plot
# ============================================================

def append_rows(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    header = not path.exists()
    df.to_csv(path, mode="a", header=header, index=False)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    ok = df[df["status"].astype(str) == "ok"].copy()
    if ok.empty:
        return pd.DataFrame()

    group_cols = [
        "function",
        "screen_mode",
        "metric",
        "dimension",
        "samples",
        "test_samples",
        "noise",
    ]

    numeric_cols = [
        "effective_dim",
        "screen_contains_all_true_vars",
        "screen_true_var_recall",
        "screen_contains_all_interaction_endpoints",
        "screen_interaction_endpoint_recall",
        "screen_contains_true_interactions",
        "train_mse",
        "test_mse",
        "variable_f1",
        "variable_auroc",
        "variable_auprc",
        "interaction_f1",
        "true_interaction_score_mean",
        "max_nontrue_interaction_score",
        "num_true_variables",
        "num_true_interactions",
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
                "variable_auroc",
                "variable_auprc",
                "interaction_f1",
                "true_interaction_score_mean",
                "max_nontrue_interaction_score",
            }:
                agg[col] = ["mean", "std"]
            else:
                agg[col] = ["mean"]

    summary = ok.groupby(group_cols, dropna=False).agg(agg).reset_index()
    summary.columns = ["_".join(str(x) for x in col if x != "").rstrip("_") for col in summary.columns]

    counts = df.groupby(["function", "screen_mode", "metric"], dropna=False).agg(
        num_rows=("status", "size"),
        num_failed=("status", lambda s: int((s.astype(str) != "ok").sum())),
    ).reset_index()

    summary = summary.merge(counts, on=["function", "screen_mode", "metric"], how="left")
    return summary


def plot_summary(summary: pd.DataFrame, out_dir: Path):
    if summary.empty:
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    metric_order = [
        "hessian",
        "finite_difference",
        "pair_permutation_synergy",
        "pair_permutation_joint",
        "kan_path_pair_synergy",
        "kan_path_pair_joint",
    ]

    screen_order = ["raw", "rf", "oracle_support"]
    functions = summary["function"].drop_duplicates().tolist()

    for screen_mode in screen_order:
        sub_s = summary[summary["screen_mode"] == screen_mode].copy()
        if sub_s.empty:
            continue

        x = np.arange(len(functions))
        width = 0.12

        plt.figure(figsize=(max(10, len(functions) * 1.15), 5.5))
        for idx, metric in enumerate(metric_order):
            vals = []
            for fn in functions:
                hit = sub_s[(sub_s["function"] == fn) & (sub_s["metric"] == metric)]
                if hit.empty:
                    vals.append(np.nan)
                else:
                    vals.append(float(hit["interaction_f1_mean"].iloc[0]))
            if all(np.isnan(v) for v in vals):
                continue
            plt.bar(x + (idx - (len(metric_order)-1)/2) * width, vals, width=width, label=metric)

        plt.ylim(0, 1.08)
        plt.ylabel("Interaction F1")
        plt.xticks(x, functions, rotation=35, ha="right")
        plt.title(f"Interaction-metric validation: {screen_mode}")
        plt.legend(ncol=2, fontsize=8)
        plt.tight_layout()
        plt.savefig(out_dir / f"interaction_metric_validation_{screen_mode}.png", dpi=250)
        plt.close()

    # Agreement summary: metric x screen, averaged over functions.
    avg = summary.groupby(["screen_mode", "metric"], dropna=False)["interaction_f1_mean"].mean().reset_index()
    pivot = avg.pivot(index="screen_mode", columns="metric", values="interaction_f1_mean")
    pivot.to_csv(out_dir / "interaction_metric_validation_average_pivot.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--functions",
        type=str,
        nargs="+",
        default=["core_interaction", "core_interaction_c5", "pairwise_interaction", "dense_quadratic"],
    )
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--test_samples", type=int, default=2048)
    parser.add_argument("--dimension", type=int, default=100)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])

    parser.add_argument("--screen_modes", type=str, nargs="+", default=["raw", "rf", "oracle_support"])
    parser.add_argument("--top_m", type=int, default=20)
    parser.add_argument("--rf_trees", type=int, default=300)

    parser.add_argument("--kan_width", type=int, default=8)
    parser.add_argument("--kan_grid", type=int, default=5)
    parser.add_argument("--kan_k", type=int, default=3)
    parser.add_argument("--kan_steps", type=int, default=50)
    parser.add_argument("--kan_lamb", type=float, default=0.0)
    parser.add_argument("--opt", type=str, default="LBFGS")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--update_grid_mode", type=str, default="default", choices=["default", "true", "false"])

    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=["hessian", "fd", "pairperm"],
        choices=["hessian", "fd", "pairperm", "path"],
    )
    parser.add_argument("--variable_points", type=int, default=2048)
    parser.add_argument("--hessian_points", type=int, default=16)
    parser.add_argument("--fd_points", type=int, default=32)
    parser.add_argument("--fd_h", type=float, default=1e-2)
    parser.add_argument("--perm_points", type=int, default=256)
    parser.add_argument("--path_points", type=int, default=512)
    parser.add_argument("--path_max_pairs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=4096)

    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--summary_out", type=str, default=None)
    parser.add_argument("--fig_dir", type=str, default=None)
    parser.add_argument("--resume", action="store_true")

    args = parser.parse_args()

    out_path = Path(args.out)
    existing_keys = set()

    if args.resume and out_path.exists():
        try:
            old = pd.read_csv(out_path)
            for r in old[["function", "seed", "screen_mode"]].drop_duplicates().itertuples(index=False):
                existing_keys.add((str(r.function), int(r.seed), str(r.screen_mode)))
            print(f"Resume mode: found {len(existing_keys)} completed configs.")
        except Exception as exc:
            print(f"[WARN] Could not read existing output for resume: {exc}")

    for function_name in args.functions:
        function_name = normalize_function_alias(function_name)
        for seed in args.seeds:
            for screen_mode in args.screen_modes:
                key = (function_name, int(seed), screen_mode)
                if key in existing_keys:
                    print(f"Skipping completed {key}")
                    continue

                print(f"Running function={function_name}, seed={seed}, screen_mode={screen_mode}")
                rows = run_one(args, function_name=function_name, seed=seed, screen_mode=screen_mode)
                append_rows(out_path, rows)

    df = pd.read_csv(out_path)
    summary = summarize(df)

    if args.summary_out is not None:
        summary_path = Path(args.summary_out)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(summary_path, index=False)
        print(f"Wrote summary to {summary_path}")

    if args.fig_dir is not None:
        fig_dir = Path(args.fig_dir)
        plot_summary(summary, fig_dir)
        print(f"Wrote figures to {fig_dir}")

    print(f"Wrote rows to {out_path}")


if __name__ == "__main__":
    main()
