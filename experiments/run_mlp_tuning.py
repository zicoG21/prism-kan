from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import average_precision_score, roc_auc_score

from src.data import make_synthetic


Pair = Tuple[int, int]


# ============================================================
# Utility metrics
# ============================================================

def canonical_pairs(pairs: Sequence[Tuple[int, int]]) -> Tuple[Pair, ...]:
    return tuple(tuple(sorted((int(i), int(j)))) for i, j in pairs)


def interaction_endpoints(pairs: Sequence[Pair]) -> Tuple[int, ...]:
    s = set()
    for i, j in pairs:
        s.add(int(i))
        s.add(int(j))
    return tuple(sorted(s))


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


def mse_np(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((pred.reshape(-1, 1) - target.reshape(-1, 1)) ** 2))


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


# ============================================================
# Model
# ============================================================

class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden: int, depth: int, activation: str = "silu", dropout: float = 0.0):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")

        act_layer = {
            "silu": nn.SiLU,
            "tanh": nn.Tanh,
            "gelu": nn.GELU,
            "softplus": nn.Softplus,
        }.get(activation.lower())
        if act_layer is None:
            raise ValueError(f"Unsupported activation={activation}. Use silu/tanh/gelu/softplus.")

        layers: List[nn.Module] = []
        in_dim = input_dim
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(act_layer())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_tensor(x: np.ndarray, device: str) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.float32, device=device)


@torch.no_grad()
def predict_model(model: nn.Module, X_np: np.ndarray, device: str, batch_size: int = 8192) -> np.ndarray:
    model.eval()
    outs = []
    for start in range(0, len(X_np), batch_size):
        X = to_tensor(X_np[start:start + batch_size], device)
        outs.append(model(X).detach().cpu().numpy().reshape(-1, 1))
    return np.vstack(outs)


def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: Dict,
    seed: int,
    device: str,
    max_epochs: int,
    patience: int,
    batch_size: int,
    verbose: bool = False,
) -> Tuple[nn.Module, Dict]:
    set_seed(seed)

    model = MLP(
        input_dim=X_train.shape[1],
        hidden=int(params["hidden"]),
        depth=int(params["depth"]),
        activation=str(params["activation"]),
        dropout=float(params.get("dropout", 0.0)),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(params["lr"]),
        weight_decay=float(params.get("weight_decay", 0.0)),
    )
    loss_fn = nn.MSELoss()

    Xtr = to_tensor(X_train, device)
    ytr = to_tensor(y_train, device)
    Xv = to_tensor(X_val, device)
    yv = to_tensor(y_val, device)

    if batch_size <= 0 or batch_size >= len(X_train):
        loader = [(Xtr, ytr)]
    else:
        ds = TensorDataset(Xtr, ytr)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    best_state = None
    best_val = float("inf")
    best_epoch = -1
    bad = 0

    history = []
    for epoch in range(max_epochs):
        model.train()
        epoch_losses = []

        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu().item()))

        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(Xv), yv).detach().cpu().item())
            train_loss = float(np.mean(epoch_losses))

        history.append((epoch, train_loss, val_loss))

        if val_loss < best_val - 1e-10:
            best_val = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1

        if verbose and (epoch % 200 == 0 or epoch == max_epochs - 1):
            print(f"epoch={epoch} train={train_loss:.4e} val={val_loss:.4e} best={best_val:.4e}")

        if bad >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    info = {
        "best_val_mse": best_val,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "final_train_mse": history[-1][1] if history else np.nan,
        "final_val_mse": history[-1][2] if history else np.nan,
    }
    return model, info


def train_mlp_full_fixed_epochs(
    X_train: np.ndarray,
    y_train: np.ndarray,
    params: Dict,
    seed: int,
    device: str,
    epochs: int,
    batch_size: int,
    lbfgs_finetune_steps: int = 0,
    verbose: bool = False,
) -> Tuple[nn.Module, Dict]:
    """Train final MLP on the full training set.

    This is useful after hyperparameter tuning: tune on a train/validation split,
    then refit the selected architecture on all available training samples.
    """
    set_seed(seed)

    model = MLP(
        input_dim=X_train.shape[1],
        hidden=int(params["hidden"]),
        depth=int(params["depth"]),
        activation=str(params["activation"]),
        dropout=float(params.get("dropout", 0.0)),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(params["lr"]),
        weight_decay=float(params.get("weight_decay", 0.0)),
    )
    loss_fn = nn.MSELoss()

    Xtr = to_tensor(X_train, device)
    ytr = to_tensor(y_train, device)

    if batch_size <= 0 or batch_size >= len(X_train):
        loader = [(Xtr, ytr)]
    else:
        ds = TensorDataset(Xtr, ytr)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    history = []
    for epoch in range(max(1, int(epochs))):
        model.train()
        losses = []
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        train_loss = float(np.mean(losses))
        history.append(train_loss)

        if verbose and (epoch % 500 == 0 or epoch == epochs - 1):
            print(f"[FULL] epoch={epoch} train={train_loss:.4e}")

    lbfgs_final_loss = np.nan
    if lbfgs_finetune_steps and lbfgs_finetune_steps > 0:
        # Full-batch LBFGS polish. This is often helpful for smooth low-dimensional
        # regression after AdamW reaches the right region.
        model.train()
        opt2 = torch.optim.LBFGS(
            model.parameters(),
            lr=1.0,
            max_iter=int(lbfgs_finetune_steps),
            history_size=50,
            line_search_fn="strong_wolfe",
        )

        def closure():
            opt2.zero_grad(set_to_none=True)
            loss = loss_fn(model(Xtr), ytr)
            loss.backward()
            return loss

        try:
            loss = opt2.step(closure)
            lbfgs_final_loss = float(loss.detach().cpu().item()) if hasattr(loss, "detach") else float(loss)
        except Exception as exc:
            print(f"[WARN] LBFGS finetune failed: {exc}")

    with torch.no_grad():
        final_train = float(loss_fn(model(Xtr), ytr).detach().cpu().item())

    info = {
        "best_val_mse": np.nan,
        "best_epoch": int(epochs) - 1,
        "epochs_run": int(epochs),
        "final_train_mse": final_train,
        "final_val_mse": np.nan,
        "lbfgs_final_loss": lbfgs_final_loss,
    }
    return model, info


# ============================================================
# Screening
# ============================================================

def rf_scores(X: np.ndarray, y: np.ndarray, seed: int, n_estimators: int = 300) -> np.ndarray:
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
    selected: List[int] = []
    seen = set()
    for v in must_include:
        v = int(v)
        if v not in seen:
            selected.append(v)
            seen.add(v)

    rest = [int(v) for v in candidate_pool if int(v) not in seen]
    need = max(0, top_m - len(selected))

    if need > 0:
        if scores is None:
            rng.shuffle(rest)
            selected.extend(rest[:need])
        else:
            rest = sorted(rest, key=lambda v: float(scores[v]), reverse=True)
            selected.extend(rest[:need])
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
    endpoints = set(interaction_endpoints(true_interactions))
    scores = np.zeros(d, dtype=float)

    if mode == "raw":
        return np.arange(d, dtype=int), scores, "none"

    if mode == "random":
        selected = np.array(sorted(rng.choice(d, size=min(top_m, d), replace=False).astype(int)), dtype=int)
        return selected, scores, "random"

    if mode == "oracle_support":
        selected = fill_with_support(true_vars, all_vars, top_m, rng, None)
        return selected, scores, "forced_true_support_random_fill"

    if mode == "exclude_interaction":
        pool = [v for v in all_vars if v not in endpoints]
        selected = np.array(sorted(rng.choice(pool, size=min(top_m, len(pool)), replace=False).astype(int)), dtype=int)
        return selected, scores, "random_excluding_interaction_endpoints"

    if mode == "rf":
        scores = rf_scores(X, y, seed=seed, n_estimators=rf_trees)
        selected = np.array(sorted(np.argsort(-scores)[:min(top_m, d)].astype(int)), dtype=int)
        return selected, scores, "rf"

    raise ValueError(f"Unknown screen mode={mode}")


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
# Recovery metrics
# ============================================================

def local_to_full_scores(local_scores: np.ndarray, selected_features: np.ndarray, d_full: int) -> np.ndarray:
    full = np.zeros(d_full, dtype=float)
    for local_idx, orig_idx in enumerate(selected_features):
        full[int(orig_idx)] = float(local_scores[local_idx])
    return full


def local_to_full_pair_scores(local_pair_scores: Dict[Pair, float], selected_features: np.ndarray, d_full: int) -> Dict[Pair, float]:
    selected_features = np.asarray(selected_features, dtype=int)
    full: Dict[Pair, float] = {}
    for (i_local, j_local), score in local_pair_scores.items():
        if i_local >= len(selected_features) or j_local >= len(selected_features):
            continue
        i = int(selected_features[i_local])
        j = int(selected_features[j_local])
        full[tuple(sorted((i, j)))] = float(score)
    for i, j in itertools.combinations(range(d_full), 2):
        full.setdefault((i, j), 0.0)
    return full


def gradient_importance(model: nn.Module, X_np: np.ndarray, device: str, points: int) -> np.ndarray:
    model.eval()
    n = min(points, len(X_np))
    X = to_tensor(X_np[:n], device)
    X.requires_grad_(True)
    y = model(X).sum()
    grad = torch.autograd.grad(y, X, create_graph=False)[0]
    return grad.abs().mean(dim=0).detach().cpu().numpy()


def integrated_gradients_importance(
    model: nn.Module,
    X_np: np.ndarray,
    device: str,
    points: int,
    steps: int,
) -> np.ndarray:
    model.eval()
    n = min(points, len(X_np))
    X = to_tensor(X_np[:n], device)
    baseline = torch.zeros_like(X)
    total_grad = torch.zeros_like(X)

    for a in torch.linspace(0, 1, steps, device=device):
        Z = baseline + a * (X - baseline)
        Z.requires_grad_(True)
        y = model(Z).sum()
        grad = torch.autograd.grad(y, Z, create_graph=False)[0]
        total_grad += grad.detach()

    ig = (X - baseline) * total_grad / max(steps, 1)
    return ig.abs().mean(dim=0).detach().cpu().numpy()


def finite_difference_interaction_scores(
    model: nn.Module,
    X_np: np.ndarray,
    device: str,
    points: int,
    h: float,
    batch_size: int,
) -> Dict[Pair, float]:
    d = X_np.shape[1]
    n = min(points, len(X_np))
    X = X_np[:n].copy()
    f0 = predict_model(model, X, device=device, batch_size=batch_size).reshape(-1)

    scores: Dict[Pair, float] = {}
    # Batch over pairs for speed.
    pairs = [(i, j) for i, j in itertools.combinations(range(d), 2)]
    for i, j in pairs:
        Xi = X.copy()
        Xj = X.copy()
        Xij = X.copy()
        Xi[:, i] += h
        Xj[:, j] += h
        Xij[:, i] += h
        Xij[:, j] += h

        fi = predict_model(model, Xi, device=device, batch_size=batch_size).reshape(-1)
        fj = predict_model(model, Xj, device=device, batch_size=batch_size).reshape(-1)
        fij = predict_model(model, Xij, device=device, batch_size=batch_size).reshape(-1)

        mixed = (fij - fi - fj + f0) / (h ** 2)
        scores[(i, j)] = float(np.mean(np.abs(mixed)))

    return scores


def hessian_interaction_scores(model: nn.Module, X_np: np.ndarray, device: str, points: int) -> Dict[Pair, float]:
    d = X_np.shape[1]
    n = min(points, len(X_np))
    X = to_tensor(X_np[:n], device)

    score_mat = torch.zeros((d, d), dtype=torch.float32, device=device)

    def scalar_func(z):
        return model(z.unsqueeze(0)).sum()

    model.eval()
    for idx in range(n):
        z = X[idx].clone().detach().requires_grad_(True)
        H = torch.autograd.functional.hessian(scalar_func, z)
        score_mat += H.abs().detach()

    score_mat = score_mat / max(n, 1)
    score_mat = score_mat.detach().cpu().numpy()
    return {(i, j): float(score_mat[i, j]) for i, j in itertools.combinations(range(d), 2)}


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


def evaluate_interaction_recovery(full_pair_scores: Dict[Pair, float], true_interactions: Sequence[Pair]) -> Dict:
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
    ranked = sorted(full_pair_scores.items(), key=lambda kv: kv[1], reverse=True)
    selected = {pair for pair, _ in ranked[:k]}
    precision, recall, f1 = f1_from_sets(selected, true_set)

    true_scores = [float(full_pair_scores.get(pair, 0.0)) for pair in true_set]
    nontrue_scores = [float(v) for pair, v in full_pair_scores.items() if pair not in true_set]

    return {
        "selected_interactions": sorted(selected),
        "interaction_precision": precision,
        "interaction_recall": recall,
        "interaction_f1": f1,
        "true_interaction_score_mean": float(np.mean(true_scores)) if true_scores else np.nan,
        "max_nontrue_interaction_score": float(np.max(nontrue_scores)) if nontrue_scores else np.nan,
    }


# ============================================================
# Hyperparameter search
# ============================================================

def sample_params(rng: np.random.Generator, args) -> Dict:
    hidden = int(rng.choice(args.hidden_choices))
    depth = int(rng.choice(args.depth_choices))
    activation = str(rng.choice(args.activation_choices))
    weight_decay = float(rng.choice(args.weight_decay_choices))
    dropout = float(rng.choice(args.dropout_choices))

    log_lr = rng.uniform(math.log10(args.lr_min), math.log10(args.lr_max))
    lr = float(10 ** log_lr)

    return {
        "hidden": hidden,
        "depth": depth,
        "activation": activation,
        "lr": lr,
        "weight_decay": weight_decay,
        "dropout": dropout,
    }


def train_val_split(X: np.ndarray, y: np.ndarray, val_frac: float, seed: int):
    rng = np.random.default_rng(seed)
    n = len(X)
    idx = rng.permutation(n)
    n_val = max(1, int(round(n * val_frac)))
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]
    return X[tr_idx], y[tr_idx], X[val_idx], y[val_idx]


def tune_one(args, function_name: str, screen_mode: str, seed: int, device: str) -> Tuple[List[Dict], Dict]:
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
    X = data["X_train"]
    y = data["y_train"]
    gt = data["ground_truth"]
    true_vars = tuple(int(v) for v in gt.active_variables)
    true_interactions = canonical_pairs(gt.interactions)

    selected_features, _, screen_desc = select_features(
        mode=screen_mode,
        X=X,
        y=y,
        top_m=args.top_m,
        true_vars=true_vars,
        true_interactions=true_interactions,
        seed=seed,
        rf_trees=args.rf_trees,
    )
    Xs = X[:, selected_features]

    Xtr, ytr, Xval, yval = train_val_split(Xs, y, args.val_frac, seed + 123)

    rng = np.random.default_rng(args.search_seed + seed * 1000 + abs(hash(function_name + screen_mode)) % 1000)
    rows = []
    best = None

    # Always include Adrian's provided params as trial 0 if requested.
    candidate_params = []
    if args.include_adrian_params:
        candidate_params.append({
            "hidden": 256,
            "depth": 4,
            "activation": args.adrian_activation,
            "lr": 1.7914823299257556e-05,
            "weight_decay": args.adrian_weight_decay,
            "dropout": 0.0,
        })

    while len(candidate_params) < args.n_trials:
        candidate_params.append(sample_params(rng, args))

    for trial_idx, params in enumerate(candidate_params):
        print(f"[TUNE] function={function_name} screen={screen_mode} seed={seed} trial={trial_idx}/{len(candidate_params)-1} params={params}")
        try:
            model, info = train_mlp(
                X_train=Xtr,
                y_train=ytr,
                X_val=Xval,
                y_val=yval,
                params=params,
                seed=seed + trial_idx * 17,
                device=device,
                max_epochs=args.tune_epochs,
                patience=args.patience,
                batch_size=args.batch_size,
                verbose=False,
            )
            row = {
                "function": function_name,
                "screen_mode": screen_mode,
                "seed": seed,
                "trial": trial_idx,
                "status": "ok",
                "error": "",
                "screen_score_type": screen_desc,
                "effective_dim": len(selected_features),
                **params,
                **info,
            }
            val = float(info["best_val_mse"])
            if best is None or val < best["best_val_mse"]:
                best = dict(row)
        except Exception as exc:
            row = {
                "function": function_name,
                "screen_mode": screen_mode,
                "seed": seed,
                "trial": trial_idx,
                "status": "failed",
                "error": repr(exc),
                **params,
            }
        rows.append(row)

    if best is None:
        raise RuntimeError(f"No successful tuning trial for {function_name}/{screen_mode}")

    best_params = {
        "hidden": int(best["hidden"]),
        "depth": int(best["depth"]),
        "activation": str(best["activation"]),
        "lr": float(best["lr"]),
        "weight_decay": float(best["weight_decay"]),
        "dropout": float(best["dropout"]),
        "best_val_mse": float(best["best_val_mse"]),
        "best_epoch": int(best["best_epoch"]),
        "screen_mode": screen_mode,
        "function": function_name,
    }
    print(f"[BEST] function={function_name} screen={screen_mode}: {best_params}")
    return rows, best_params


# ============================================================
# Evaluation
# ============================================================

def choose_params(best_params: Dict, function_name: str, screen_mode: str, fallback_params: Dict) -> Dict:
    # Try exact function/screen params.
    key = f"{function_name}::{screen_mode}"
    if key in best_params:
        return best_params[key]

    # Fall back to raw params for the same function.
    key_raw = f"{function_name}::raw"
    if key_raw in best_params:
        return best_params[key_raw]

    # Fall back to any function-independent params.
    if "__global__" in best_params:
        return best_params["__global__"]

    return fallback_params


def evaluate_one(args, function_name: str, screen_mode: str, seed: int, params: Dict, device: str) -> Dict:
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

    Xtr, ytr, Xval, yval = train_val_split(X_train_s, y_train, args.val_frac, seed + 999)

    if args.final_train_full:
        if args.use_best_epoch_for_eval and "best_epoch" in params:
            full_epochs = int(params.get("best_epoch", args.eval_epochs)) + int(args.final_extra_epochs)
            full_epochs = max(int(args.min_full_epochs), full_epochs)
            full_epochs = min(int(args.eval_epochs), full_epochs)
        else:
            full_epochs = int(args.eval_epochs)

        model, info = train_mlp_full_fixed_epochs(
            X_train=X_train_s,
            y_train=y_train,
            params=params,
            seed=seed,
            device=device,
            epochs=full_epochs,
            batch_size=args.batch_size,
            lbfgs_finetune_steps=args.lbfgs_finetune_steps,
            verbose=False,
        )
    else:
        model, info = train_mlp(
            X_train=Xtr,
            y_train=ytr,
            X_val=Xval,
            y_val=yval,
            params=params,
            seed=seed,
            device=device,
            max_epochs=args.eval_epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            verbose=False,
        )

    train_mse = mse_np(predict_model(model, X_train_s, device=device, batch_size=args.pred_batch_size), y_train)
    test_mse = mse_np(predict_model(model, X_test_s, device=device, batch_size=args.pred_batch_size), y_test)

    if args.variable_method == "grad":
        local_var_scores = gradient_importance(model, X_test_s, device=device, points=args.variable_points)
    elif args.variable_method == "ig":
        local_var_scores = integrated_gradients_importance(
            model, X_test_s, device=device, points=args.variable_points, steps=args.ig_steps
        )
    else:
        raise ValueError(f"Unknown variable_method={args.variable_method}")

    full_var_scores = local_to_full_scores(local_var_scores, selected_features, args.dimension)
    var_eval = evaluate_variable_recovery(full_var_scores, true_vars)

    if len(true_interactions) > 0:
        if args.interaction_method == "fd":
            local_pair_scores = finite_difference_interaction_scores(
                model=model,
                X_np=X_test_s,
                device=device,
                points=args.fd_points,
                h=args.fd_h,
                batch_size=args.pred_batch_size,
            )
        elif args.interaction_method == "hessian":
            local_pair_scores = hessian_interaction_scores(
                model=model,
                X_np=X_test_s,
                device=device,
                points=args.hessian_points,
            )
        else:
            raise ValueError(f"Unknown interaction_method={args.interaction_method}")
        full_pair_scores = local_to_full_pair_scores(local_pair_scores, selected_features, args.dimension)
    else:
        full_pair_scores = {}

    int_eval = evaluate_interaction_recovery(full_pair_scores, true_interactions)

    row = {
        "model": "MLP",
        "function": function_name,
        "screen_mode": screen_mode,
        "seed": seed,
        "samples": args.samples,
        "test_samples": args.test_samples,
        "dimension": args.dimension,
        "noise": args.noise,
        "top_m": args.top_m,
        "screen_score_type": screen_desc,
        "selected_screen_features": selected_features.tolist(),
        "screen_scores": screen_scores.tolist(),
        "true_variables": list(true_vars),
        "true_interactions": list(true_interactions),
        "num_true_variables": len(true_vars),
        "num_true_interactions": len(true_interactions),
        "formula": gt.formula,
        "status": "ok",
        "error": "",
        "traceback": "",
        "train_mse": train_mse,
        "test_mse": test_mse,
        "variable_method": args.variable_method,
        "interaction_method": args.interaction_method,
        "importance_scores": full_var_scores.tolist(),
        "hidden": int(params["hidden"]),
        "depth": int(params["depth"]),
        "activation": str(params["activation"]),
        "lr": float(params["lr"]),
        "weight_decay": float(params.get("weight_decay", 0.0)),
        "dropout": float(params.get("dropout", 0.0)),
        "final_train_full": int(args.final_train_full),
        "use_best_epoch_for_eval": int(args.use_best_epoch_for_eval),
        "final_extra_epochs": int(args.final_extra_epochs),
        "min_full_epochs": int(args.min_full_epochs),
        "lbfgs_finetune_steps": int(args.lbfgs_finetune_steps),
        **info,
    }
    row.update(support_stats(selected_features, true_vars, true_interactions))
    row.update(var_eval)
    row.update(int_eval)
    return row


def summarize_eval(df: pd.DataFrame) -> pd.DataFrame:
    ok = df[df["status"].astype(str) == "ok"].copy()
    if ok.empty:
        return pd.DataFrame()

    group_cols = [
        "model", "function", "screen_mode", "dimension", "samples", "noise",
        "variable_method", "interaction_method", "hidden", "depth", "activation",
        "final_train_full", "lbfgs_finetune_steps",
    ]
    numeric_cols = [
        "train_mse", "test_mse", "effective_dim",
        "screen_contains_all_true_vars", "screen_true_var_recall",
        "screen_contains_all_interaction_endpoints",
        "screen_interaction_endpoint_recall",
        "screen_contains_true_interactions",
        "variable_f1", "variable_auroc", "variable_auprc",
        "interaction_f1", "true_interaction_score_mean", "max_nontrue_interaction_score",
    ]
    for col in numeric_cols:
        if col in ok.columns:
            ok[col] = pd.to_numeric(ok[col], errors="coerce")

    agg = {}
    for col in numeric_cols:
        if col in ok.columns:
            if col in {"train_mse", "test_mse", "variable_f1", "variable_auroc", "variable_auprc",
                       "interaction_f1", "true_interaction_score_mean", "max_nontrue_interaction_score"}:
                agg[col] = ["mean", "std"]
            else:
                agg[col] = ["mean"]

    summary = ok.groupby(group_cols, dropna=False).agg(agg).reset_index()
    summary.columns = ["_".join(str(x) for x in col if x != "").rstrip("_") for col in summary.columns]
    return summary


def plot_eval_summary(summary: pd.DataFrame, fig_dir: Path):
    if summary.empty:
        return
    fig_dir.mkdir(parents=True, exist_ok=True)

    funcs = summary["function"].drop_duplicates().tolist()
    modes = ["raw", "rf", "oracle_support", "random", "exclude_interaction"]
    labels = {
        "raw": "Raw MLP",
        "rf": "RF-screened MLP",
        "oracle_support": "Oracle-support MLP",
        "random": "Random-screened MLP",
        "exclude_interaction": "Exclude-interaction MLP",
    }

    def plot_metric(col: str, ylabel: str, title: str, filename: str, log: bool = False):
        x = np.arange(len(funcs))
        width = 0.15
        plt.figure(figsize=(max(10, len(funcs) * 1.15), 5.5))
        for idx, mode in enumerate(modes):
            vals = []
            for fn in funcs:
                hit = summary[(summary["function"] == fn) & (summary["screen_mode"] == mode)]
                vals.append(float(hit[col].iloc[0]) if not hit.empty and col in hit.columns else np.nan)
            plt.bar(x + (idx - (len(modes)-1)/2) * width, vals, width=width, label=labels.get(mode, mode))
        if log:
            plt.yscale("log")
        else:
            plt.ylim(0, 1.08)
        plt.ylabel(ylabel)
        plt.xticks(x, funcs, rotation=35, ha="right")
        plt.title(title)
        plt.legend(ncol=2, fontsize=8)
        plt.tight_layout()
        plt.savefig(fig_dir / filename, dpi=250)
        plt.close()

    plot_metric("variable_f1_mean", "Variable F1", "MLP variable recovery", "mlp_variable_f1.png", log=False)
    plot_metric("interaction_f1_mean", "Interaction F1", "MLP interaction recovery", "mlp_interaction_f1.png", log=False)
    plot_metric("test_mse_mean", "Test MSE", "MLP prediction error", "mlp_test_mse.png", log=True)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", choices=["tune", "eval", "tune_and_eval"], default="tune_and_eval")
    parser.add_argument("--functions", nargs="+", default=[
        "core_interaction", "core_interaction_c5", "correlated_proxy", "pairwise_interaction"
    ])
    parser.add_argument("--tune_functions", nargs="+", default=None)
    parser.add_argument("--eval_functions", nargs="+", default=None)

    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--test_samples", type=int, default=2048)
    parser.add_argument("--dimension", type=int, default=100)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--tune_seed", type=int, default=0)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--val_frac", type=float, default=0.2)

    parser.add_argument("--tune_screen_modes", nargs="+", default=["raw"])
    parser.add_argument("--eval_screen_modes", nargs="+", default=["raw"])
    parser.add_argument("--top_m", type=int, default=20)
    parser.add_argument("--rf_trees", type=int, default=300)

    parser.add_argument("--n_trials", type=int, default=12)
    parser.add_argument("--search_seed", type=int, default=2026)
    parser.add_argument("--hidden_choices", nargs="+", type=int, default=[64, 128, 256])
    parser.add_argument("--depth_choices", nargs="+", type=int, default=[2, 3, 4])
    parser.add_argument("--activation_choices", nargs="+", default=["silu", "tanh"])
    parser.add_argument("--weight_decay_choices", nargs="+", type=float, default=[0.0, 1e-6, 1e-5, 1e-4])
    parser.add_argument("--dropout_choices", nargs="+", type=float, default=[0.0])
    parser.add_argument("--lr_min", type=float, default=1e-5)
    parser.add_argument("--lr_max", type=float, default=3e-3)

    parser.add_argument("--include_adrian_params", action="store_true")
    parser.add_argument("--adrian_activation", default="silu")
    parser.add_argument("--adrian_weight_decay", type=float, default=0.0)

    parser.add_argument("--tune_epochs", type=int, default=1500)
    parser.add_argument("--eval_epochs", type=int, default=2000)
    parser.add_argument("--patience", type=int, default=300)
    parser.add_argument("--final_train_full", action="store_true",
                        help="During eval, refit selected params on all training samples instead of only the train split.")
    parser.add_argument("--use_best_epoch_for_eval", action="store_true",
                        help="When final_train_full is set, use tuned best_epoch + final_extra_epochs, capped by eval_epochs.")
    parser.add_argument("--final_extra_epochs", type=int, default=500)
    parser.add_argument("--min_full_epochs", type=int, default=500)
    parser.add_argument("--lbfgs_finetune_steps", type=int, default=0,
                        help="Optional full-batch LBFGS polish after AdamW final training.")
    parser.add_argument("--batch_size", type=int, default=0, help="0 means full-batch")
    parser.add_argument("--pred_batch_size", type=int, default=8192)

    parser.add_argument("--variable_method", choices=["grad", "ig"], default="grad")
    parser.add_argument("--variable_points", type=int, default=2048)
    parser.add_argument("--ig_steps", type=int, default=32)

    parser.add_argument("--interaction_method", choices=["fd", "hessian"], default="fd")
    parser.add_argument("--fd_points", type=int, default=32)
    parser.add_argument("--fd_h", type=float, default=1e-2)
    parser.add_argument("--hessian_points", type=int, default=16)

    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])

    parser.add_argument("--tuning_out", default=None)
    parser.add_argument("--best_params_out", default=None)
    parser.add_argument("--eval_out", default=None)
    parser.add_argument("--summary_out", default=None)
    parser.add_argument("--fig_dir", default=None)

    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device={device}")

    tune_functions = [normalize_function_alias(f) for f in (args.tune_functions or args.functions)]
    eval_functions = [normalize_function_alias(f) for f in (args.eval_functions or args.functions)]

    best_params: Dict[str, Dict] = {}

    if args.mode in {"tune", "tune_and_eval"} and args.tuning_out is None:
        raise ValueError("--tuning_out is required when --mode is tune or tune_and_eval")
    if args.mode in {"tune", "tune_and_eval", "eval"} and args.best_params_out is None:
        raise ValueError("--best_params_out is required")
    if args.mode in {"tune_and_eval", "eval"} and args.eval_out is None:
        raise ValueError("--eval_out is required when --mode is eval or tune_and_eval")

    if args.mode in {"tune", "tune_and_eval"}:
        tuning_rows = []
        for fn in tune_functions:
            for screen_mode in args.tune_screen_modes:
                rows, best = tune_one(args, fn, screen_mode, args.tune_seed, device)
                tuning_rows.extend(rows)
                best_params[f"{fn}::{screen_mode}"] = best

        tuning_path = Path(args.tuning_out)
        tuning_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(tuning_rows).to_csv(tuning_path, index=False)
        print(f"Wrote tuning trials to {tuning_path}")

        best_path = Path(args.best_params_out)
        best_path.parent.mkdir(parents=True, exist_ok=True)
        best_path.write_text(json.dumps(best_params, indent=2), encoding="utf-8")
        print(f"Wrote best params to {best_path}")

    if args.mode == "eval":
        best_path = Path(args.best_params_out)
        if not best_path.exists():
            raise FileNotFoundError(f"Missing best params JSON: {best_path}")
        best_params = json.loads(best_path.read_text(encoding="utf-8"))

    if args.mode in {"eval", "tune_and_eval"}:
        fallback_params = {
            "hidden": 256,
            "depth": 4,
            "activation": "silu",
            "lr": 1.7914823299257556e-05,
            "weight_decay": 0.0,
            "dropout": 0.0,
        }

        eval_rows = []
        for fn in eval_functions:
            for screen_mode in args.eval_screen_modes:
                params = choose_params(best_params, fn, screen_mode, fallback_params)
                for seed in args.seeds:
                    print(f"[EVAL] function={fn} screen={screen_mode} seed={seed} params={params}")
                    try:
                        row = evaluate_one(args, fn, screen_mode, seed, params, device)
                    except Exception as exc:
                        row = {
                            "model": "MLP",
                            "function": fn,
                            "screen_mode": screen_mode,
                            "seed": seed,
                            "status": "failed",
                            "error": repr(exc),
                            "traceback": traceback.format_exc(),
                        }
                        print(f"[WARN] failed eval function={fn}, screen={screen_mode}, seed={seed}: {exc}")
                    eval_rows.append(row)

        eval_path = Path(args.eval_out)
        eval_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(eval_rows).to_csv(eval_path, index=False)
        print(f"Wrote eval rows to {eval_path}")

        summary = summarize_eval(pd.DataFrame(eval_rows))
        if args.summary_out is not None:
            summary_path = Path(args.summary_out)
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary.to_csv(summary_path, index=False)
            print(f"Wrote summary to {summary_path}")

        if args.fig_dir is not None:
            plot_eval_summary(summary, Path(args.fig_dir))
            print(f"Wrote figures to {args.fig_dir}")


if __name__ == "__main__":
    main()
