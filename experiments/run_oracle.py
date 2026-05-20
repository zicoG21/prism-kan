from __future__ import annotations

import argparse
import itertools
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from src.data import evaluate_synthetic_function


def generate_X(function_name: str, samples: int, dimension: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, size=(samples, dimension)).astype(np.float32)

    if function_name == "correlated_proxy":
        X[:, 4] = X[:, 0] + rng.normal(0.0, 0.05, size=samples).astype(np.float32)
        if dimension > 5:
            X[:, 5] = X[:, 1] + rng.normal(0.0, 0.05, size=samples).astype(np.float32)

    return X


def true_function_torch(function_name: str, X: torch.Tensor) -> torch.Tensor:
    pi = torch.tensor(np.pi, dtype=X.dtype, device=X.device)

    if function_name in {"core_interaction", "highdim_sparse", "correlated_proxy"}:
        return torch.sin(2 * pi * X[:, 0]) + X[:, 1] ** 2 + X[:, 2] * X[:, 3]

    if function_name == "core_interaction_c05":
        return torch.sin(2 * pi * X[:, 0]) + X[:, 1] ** 2 + 0.5 * X[:, 2] * X[:, 3]

    if function_name == "core_interaction_c1":
        return torch.sin(2 * pi * X[:, 0]) + X[:, 1] ** 2 + X[:, 2] * X[:, 3]

    if function_name == "core_interaction_c2":
        return torch.sin(2 * pi * X[:, 0]) + X[:, 1] ** 2 + 2.0 * X[:, 2] * X[:, 3]

    if function_name == "core_interaction_c5":
        return torch.sin(2 * pi * X[:, 0]) + X[:, 1] ** 2 + 5.0 * X[:, 2] * X[:, 3]

    if function_name == "additive_sparse":
        return torch.sin(2 * pi * X[:, 0]) + X[:, 1] ** 2 + torch.exp(X[:, 2])

    if function_name == "pairwise_interaction":
        return X[:, 0] * X[:, 1] + torch.sin(2 * pi * X[:, 2])

    if function_name == "compositional":
        return torch.sin(X[:, 0] * X[:, 1] + X[:, 2] ** 2)

    if function_name == "rational":
        return (X[:, 0] * X[:, 1]) / (1.0 + X[:, 2] ** 2)

    if function_name == "discontinuous":
        return (X[:, 0] > 0).float() + 0.5 * X[:, 1]

    if function_name == "dense_quadratic":
        y = torch.zeros(X.shape[0], dtype=X.dtype, device=X.device)
        for i in range(5):
            for j in range(i + 1, 5):
                coef = ((i + 1) * (j + 2)) / 20.0
                y = y + coef * X[:, i] * X[:, j]
        return y

    raise ValueError(f"Unsupported function_name for torch oracle: {function_name}")


def f1_from_sets(pred: set, true: set) -> Tuple[float, float, float]:
    if len(pred) == 0 and len(true) == 0:
        return 1.0, 1.0, 1.0
    if len(pred) == 0:
        return 0.0, 0.0, 0.0

    tp = len(pred & true)
    precision = tp / len(pred) if len(pred) > 0 else 0.0
    recall = tp / len(true) if len(true) > 0 else 0.0

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1


def variable_metrics(scores: np.ndarray, active_variables: Tuple[int, ...]) -> Dict:
    d = len(scores)
    true_set = set(active_variables)
    k = len(true_set)

    selected = set(np.argsort(-scores)[:k].tolist())

    precision, recall, f1 = f1_from_sets(selected, true_set)

    y_true = np.array([1 if i in true_set else 0 for i in range(d)])

    try:
        auroc = roc_auc_score(y_true, scores)
    except ValueError:
        auroc = np.nan

    try:
        auprc = average_precision_score(y_true, scores)
    except ValueError:
        auprc = np.nan

    active_scores = scores[y_true == 1]
    inactive_scores = scores[y_true == 0]

    return {
        "selected_variables": sorted(selected),
        "variable_precision": precision,
        "variable_recall": recall,
        "variable_f1": f1,
        "variable_auroc": auroc,
        "variable_auprc": auprc,
        "active_score_mean": float(active_scores.mean()) if len(active_scores) else np.nan,
        "inactive_score_mean": float(inactive_scores.mean()) if len(inactive_scores) else np.nan,
        "active_score_min": float(active_scores.min()) if len(active_scores) else np.nan,
        "inactive_score_max": float(inactive_scores.max()) if len(inactive_scores) else np.nan,
    }


def interaction_metrics(
    pair_scores: Dict[Tuple[int, int], float],
    true_interactions: Tuple[Tuple[int, int], ...],
) -> Dict:
    true_set = {tuple(sorted(p)) for p in true_interactions}
    k = max(len(true_set), 1)

    ranked = sorted(pair_scores.items(), key=lambda kv: kv[1], reverse=True)
    selected = {pair for pair, _ in ranked[:k]}

    precision, recall, f1 = f1_from_sets(selected, true_set)

    return {
        "selected_interactions": sorted(selected),
        "interaction_precision": precision,
        "interaction_recall": recall,
        "interaction_f1": f1,
    }


def oracle_gradient_importance(function_name: str, X_np: np.ndarray) -> np.ndarray:
    X = torch.tensor(X_np, dtype=torch.float32, requires_grad=True)
    y = true_function_torch(function_name, X).sum()
    grad = torch.autograd.grad(y, X, create_graph=False)[0]
    scores = grad.abs().mean(dim=0).detach().cpu().numpy()
    return scores


def oracle_permutation_importance(function_name: str, X_np: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 100000)
    X = X_np.copy()

    with torch.no_grad():
        baseline = true_function_torch(function_name, torch.tensor(X, dtype=torch.float32))
        baseline_np = baseline.cpu().numpy()

    scores = []
    for j in range(X.shape[1]):
        X_perm = X.copy()
        perm = rng.permutation(X.shape[0])
        X_perm[:, j] = X_perm[perm, j]

        with torch.no_grad():
            y_perm = true_function_torch(function_name, torch.tensor(X_perm, dtype=torch.float32))
            y_perm_np = y_perm.cpu().numpy()

        score = float(np.mean((y_perm_np - baseline_np) ** 2))
        scores.append(score)

    return np.array(scores, dtype=np.float64)


def oracle_hessian_interactions(
    function_name: str,
    X_np: np.ndarray,
    hessian_points: int,
) -> Dict[Tuple[int, int], float]:
    d = X_np.shape[1]
    n = min(hessian_points, X_np.shape[0])
    X_sub = torch.tensor(X_np[:n], dtype=torch.float32)

    score_mat = torch.zeros((d, d), dtype=torch.float32)

    def scalar_func(z: torch.Tensor) -> torch.Tensor:
        return true_function_torch(function_name, z.unsqueeze(0)).sum()

    for idx in range(n):
        z = X_sub[idx].clone().detach().requires_grad_(True)
        H = torch.autograd.functional.hessian(scalar_func, z)
        score_mat += H.abs().detach()

    score_mat = score_mat / max(n, 1)

    pair_scores = {}
    for i, j in itertools.combinations(range(d), 2):
        pair_scores[(i, j)] = float(score_mat[i, j].item())

    return pair_scores


def run_one(args, seed: int):
    X = generate_X(args.function, args.samples, args.dimension, seed)

    _, gt = evaluate_synthetic_function(args.function, X)

    rows = []
    score_rows = []

    methods = []

    if "grad" in args.methods:
        grad_scores = oracle_gradient_importance(args.function, X)
        methods.append(("grad", grad_scores))

    if "perm" in args.methods:
        perm_scores = oracle_permutation_importance(args.function, X, seed)
        methods.append(("perm", perm_scores))

    interaction_result = {}
    if args.compute_interactions:
        pair_scores = oracle_hessian_interactions(
            args.function,
            X,
            hessian_points=args.hessian_points,
        )
        interaction_result = interaction_metrics(pair_scores, gt.interactions)

    for method, scores in methods:
        var_result = variable_metrics(scores, gt.active_variables)

        row = {
            "model": "ORACLE",
            "function": args.function,
            "seed": seed,
            "samples": args.samples,
            "dimension": args.dimension,
            "noise": 0.0,
            "explain_method": method,
            "formula": gt.formula,
            "true_variables": list(gt.active_variables),
            "true_interactions": list(gt.interactions),
            "importance_scores": scores.tolist(),
        }
        row.update(var_result)

        if args.compute_interactions:
            row.update(interaction_result)

        rows.append(row)

        if args.scores_out is not None:
            active_set = set(gt.active_variables)
            for j, score in enumerate(scores):
                score_rows.append({
                    "model": "ORACLE",
                    "function": args.function,
                    "seed": seed,
                    "samples": args.samples,
                    "dimension": args.dimension,
                    "explain_method": method,
                    "variable_index": j,
                    "score": float(score),
                    "is_active": int(j in active_set),
                })

    return rows, score_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--function", type=str, required=True)
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--dimension", type=int, default=20)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--methods", type=str, nargs="+", default=["grad", "perm"])
    parser.add_argument("--compute_interactions", action="store_true")
    parser.add_argument("--hessian_points", type=int, default=64)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--scores_out", type=str, default=None)
    args = parser.parse_args()

    all_rows = []
    all_score_rows = []

    for seed in args.seeds:
        print(f"Running oracle seed={seed}")
        rows, score_rows = run_one(args, seed)
        all_rows.extend(rows)
        all_score_rows.extend(score_rows)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(out_path, index=False)
    print(f"Wrote {out_path}")

    if args.scores_out is not None:
        scores_path = Path(args.scores_out)
        scores_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(all_score_rows).to_csv(scores_path, index=False)
        print(f"Wrote {scores_path}")


if __name__ == "__main__":
    main()