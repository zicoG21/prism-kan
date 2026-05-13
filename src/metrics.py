from __future__ import annotations

from itertools import combinations
from typing import Iterable, List, Sequence, Set, Tuple

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


def topk_set(scores: np.ndarray, k: int) -> Set[int]:
    scores = np.asarray(scores).reshape(-1)
    if k <= 0:
        return set()
    k = min(k, len(scores))
    idx = np.argsort(-scores)[:k]
    return set(map(int, idx))


def precision_recall_f1(true_set: Iterable, pred_set: Iterable) -> Tuple[float, float, float]:
    true_set = set(true_set)
    pred_set = set(pred_set)
    if len(pred_set) == 0:
        precision = 1.0 if len(true_set) == 0 else 0.0
    else:
        precision = len(true_set & pred_set) / len(pred_set)
    if len(true_set) == 0:
        recall = 1.0 if len(pred_set) == 0 else 0.0
    else:
        recall = len(true_set & pred_set) / len(true_set)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1




def variable_labels(active_variables: Iterable[int], dimension: int) -> np.ndarray:
    """Return a binary vector with 1 for active variables and 0 otherwise."""
    labels = np.zeros(int(dimension), dtype=np.int64)
    for idx in active_variables:
        idx = int(idx)
        if 0 <= idx < dimension:
            labels[idx] = 1
    return labels


def ranking_metrics(scores: np.ndarray, active_variables: Iterable[int], dimension: int) -> Tuple[float, float]:
    """
    Non-oracle ranking metrics for variable importance scores.

    AUROC and AUPRC evaluate whether the full importance ranking separates
    true active variables from irrelevant variables, without assuming the
    number of active variables is known.
    """
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)[:dimension]
    y_true = variable_labels(active_variables, dimension)

    # roc_auc_score is undefined if all labels are one class. This should not
    # happen for our sparse synthetic functions, but return NaN defensively.
    if len(np.unique(y_true)) < 2:
        return float("nan"), float("nan")

    auroc = float(roc_auc_score(y_true, scores))
    auprc = float(average_precision_score(y_true, scores))
    return auroc, auprc


def score_distribution_summary(scores: np.ndarray, active_variables: Iterable[int], dimension: int) -> dict:
    """Summarize active-vs-inactive score separation for quick diagnostics."""
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)[:dimension]
    labels = variable_labels(active_variables, dimension)
    active_scores = scores[labels == 1]
    inactive_scores = scores[labels == 0]
    return {
        "active_score_mean": float(np.mean(active_scores)) if len(active_scores) else float("nan"),
        "active_score_median": float(np.median(active_scores)) if len(active_scores) else float("nan"),
        "inactive_score_mean": float(np.mean(inactive_scores)) if len(inactive_scores) else float("nan"),
        "inactive_score_median": float(np.median(inactive_scores)) if len(inactive_scores) else float("nan"),
        "active_score_min": float(np.min(active_scores)) if len(active_scores) else float("nan"),
        "inactive_score_max": float(np.max(inactive_scores)) if len(inactive_scores) else float("nan"),
    }

def jaccard(a: Iterable, b: Iterable) -> float:
    a = set(a)
    b = set(b)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def mean_pairwise_jaccard(sets: Sequence[Iterable]) -> float:
    sets = [set(s) for s in sets]
    if len(sets) <= 1:
        return 1.0
    vals = []
    for i, j in combinations(range(len(sets)), 2):
        vals.append(jaccard(sets[i], sets[j]))
    return float(np.mean(vals))


def mse_np(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return float(np.mean((y_pred.reshape(-1, 1) - y_true.reshape(-1, 1)) ** 2))


def gradient_importance(model, X: np.ndarray, device: str = "cpu", max_points: int = 512) -> np.ndarray:
    """
    Input-gradient attribution:
      score_j = E_x |d f(x) / d x_j|

    This is not a pure structural KAN score, but it is robust and works for KAN/MLP.
    It is useful as a sanity-check explanation metric.
    """
    model_device = torch.device(device)
    X_sub = X[:max_points]
    xt = torch.tensor(X_sub, dtype=torch.float32, device=model_device, requires_grad=True)
    y = model(xt).sum()
    grad = torch.autograd.grad(y, xt, create_graph=False)[0]
    scores = grad.detach().abs().mean(dim=0).cpu().numpy()
    return scores


def permutation_importance(
    model,
    X: np.ndarray,
    y: np.ndarray,
    device: str = "cpu",
    repeats: int = 3,
    seed: int = 0,
    max_points: int = 1024,
) -> np.ndarray:
    """
    score_j = increase in MSE after permuting feature j.
    """
    rng = np.random.default_rng(seed)
    X0 = X[:max_points].copy()
    y0 = y[:max_points].reshape(-1, 1).copy()

    def pred(arr):
        xt = torch.tensor(arr, dtype=torch.float32, device=device)
        with torch.no_grad():
            return model(xt).detach().cpu().numpy()

    baseline = np.mean((pred(X0) - y0) ** 2)
    d = X0.shape[1]
    scores = np.zeros(d, dtype=np.float64)

    for j in range(d):
        vals = []
        for _ in range(repeats):
            Xp = X0.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            vals.append(np.mean((pred(Xp) - y0) ** 2) - baseline)
        scores[j] = np.mean(vals)

    scores = np.maximum(scores, 0.0)
    return scores


def hessian_interaction_scores(
    model,
    X: np.ndarray,
    device: str = "cpu",
    max_points: int = 128,
    candidate_variables: Sequence[int] | None = None,
) -> np.ndarray:
    """
    Pairwise interaction proxy:
      H_ij = E_x |d^2 f(x) / d x_i d x_j|

    For d=20 and max_points<=128 this is usually okay. For larger d, pass candidate_variables.
    """
    model_device = torch.device(device)
    X_sub = X[:max_points]
    xt = torch.tensor(X_sub, dtype=torch.float32, device=model_device, requires_grad=True)
    y = model(xt).sum()
    grad = torch.autograd.grad(y, xt, create_graph=True)[0]
    d = xt.shape[1]

    vars_to_check = list(range(d)) if candidate_variables is None else list(candidate_variables)
    scores = np.zeros((d, d), dtype=np.float64)

    for i in vars_to_check:
        gi = grad[:, i].sum()
        grad2 = torch.autograd.grad(gi, xt, retain_graph=True, create_graph=False)[0]
        vals = grad2.detach().abs().mean(dim=0).cpu().numpy()
        for j in vars_to_check:
            if i != j:
                scores[i, j] = vals[j]

    scores = 0.5 * (scores + scores.T)
    return scores


def topk_interactions(score_matrix: np.ndarray, k: int) -> Set[Tuple[int, int]]:
    d = score_matrix.shape[0]
    pairs = []
    for i in range(d):
        for j in range(i + 1, d):
            pairs.append(((i, j), score_matrix[i, j]))
    pairs.sort(key=lambda x: -x[1])
    return set(pair for pair, _ in pairs[:k])
