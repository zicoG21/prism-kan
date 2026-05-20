from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression

from src.data import make_synthetic


def parse_pair(pair_text: str) -> Tuple[int, int]:
    parts = pair_text.replace("(", "").replace(")", "").split(",")
    if len(parts) != 2:
        raise ValueError(f"Invalid pair format: {pair_text!r}. Use format like '2,3'.")
    i, j = int(parts[0].strip()), int(parts[1].strip())
    if i == j:
        raise ValueError(f"Pair must contain two different variables: {pair_text!r}")
    return tuple(sorted((i, j)))


def mse_np(pred: np.ndarray, target: np.ndarray) -> float:
    pred = pred.reshape(-1, 1)
    target = target.reshape(-1, 1)
    return float(np.mean((pred - target) ** 2))


def predict_model(model, X_np: np.ndarray) -> np.ndarray:
    X = torch.tensor(X_np, dtype=torch.float32)
    with torch.no_grad():
        y = model(X)
    if isinstance(y, torch.Tensor):
        return y.detach().cpu().numpy().reshape(-1, 1)
    return np.asarray(y).reshape(-1, 1)


def train_pykan(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    width_hidden: int,
    grid: int,
    k: int,
    steps: int,
    lamb: float,
    seed: int,
):
    try:
        from kan import KAN
    except Exception as exc:
        raise ImportError(
            "Could not import pykan. Install with `pip install pykan` "
            "or `pip install git+https://github.com/KindXiaoming/pykan.git`."
        ) from exc

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

    if hasattr(model, "fit"):
        model.fit(dataset, opt="LBFGS", steps=steps, lamb=lamb)
    else:
        raise RuntimeError("This pykan version does not expose model.fit(...).")

    return model


def correlation_screen(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    y_flat = y.reshape(-1)
    y_center = y_flat - y_flat.mean()
    y_norm = np.linalg.norm(y_center)
    scores = np.zeros(X.shape[1], dtype=float)

    if y_norm == 0:
        return scores

    for j in range(X.shape[1]):
        x = X[:, j]
        x_center = x - x.mean()
        denom = np.linalg.norm(x_center) * y_norm
        scores[j] = 0.0 if denom == 0 else abs(float(np.dot(x_center, y_center) / denom))
    return scores


def mutual_info_screen(X: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    return np.asarray(mutual_info_regression(X, y.reshape(-1), random_state=seed), dtype=float)


def random_forest_screen(X: np.ndarray, y: np.ndarray, seed: int, n_estimators: int = 300) -> np.ndarray:
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=seed,
        n_jobs=-1,
        min_samples_leaf=2,
    )
    rf.fit(X, y.reshape(-1))
    return np.asarray(rf.feature_importances_, dtype=float)


def marginal_permutation_screen(X: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 9981)
    y_flat = y.reshape(-1)
    y_center = y_flat - y_flat.mean()
    scores = np.zeros(X.shape[1], dtype=float)

    for j in range(X.shape[1]):
        x = X[:, j]
        x_center = x - x.mean()
        original = float(np.mean(x_center * y_center) ** 2)
        perm = rng.permutation(X.shape[0])
        x_perm = x[perm]
        x_perm_center = x_perm - x_perm.mean()
        shuffled = float(np.mean(x_perm_center * y_center) ** 2)
        scores[j] = max(0.0, original - shuffled)

    return scores


def screen_features(X: np.ndarray, y: np.ndarray, method: str, top_m: int, seed: int, rf_trees: int):
    if method == "correlation":
        scores = correlation_screen(X, y)
    elif method == "mutual_info":
        scores = mutual_info_screen(X, y, seed)
    elif method == "rf":
        scores = random_forest_screen(X, y, seed, n_estimators=rf_trees)
    elif method == "marginal_perm":
        scores = marginal_permutation_screen(X, y, seed)
    else:
        raise ValueError(f"Unknown screen method: {method}")

    top_m = min(top_m, X.shape[1])
    selected = np.argsort(-scores)[:top_m]
    selected = np.array(sorted(int(i) for i in selected), dtype=int)
    return selected, scores


def get_act_layers(model):
    if not hasattr(model, "act_fun"):
        raise RuntimeError("This KAN model does not have attribute `act_fun`.")
    layers = model.act_fun
    if not isinstance(layers, (list, torch.nn.ModuleList)):
        raise RuntimeError(f"`model.act_fun` has unexpected type: {type(layers)}")
    if len(layers) < 2:
        raise RuntimeError(f"Expected at least 2 KAN activation layers, got {len(layers)}.")
    return layers


def get_mask_tensor(layer):
    if not hasattr(layer, "mask"):
        raise RuntimeError(f"KAN layer {type(layer)} does not have `.mask`.")
    mask = layer.mask
    if not isinstance(mask, torch.Tensor):
        raise RuntimeError(f"Layer mask is not a torch.Tensor. Type: {type(mask)}")
    return mask


def save_masks(model) -> List[torch.Tensor]:
    return [get_mask_tensor(layer).detach().clone() for layer in get_act_layers(model)]


def restore_masks(model, saved_masks: List[torch.Tensor]) -> None:
    with torch.no_grad():
        for layer, saved in zip(get_act_layers(model), saved_masks):
            get_mask_tensor(layer).copy_(saved)


def zero_input_hidden_edge(model, input_idx: int, hidden_idx: int, input_dim: int, hidden_dim: int) -> None:
    mask = get_mask_tensor(get_act_layers(model)[0])
    with torch.no_grad():
        if mask.shape[0] == input_dim and mask.shape[1] == hidden_dim:
            mask[input_idx, hidden_idx] = 0.0
        elif mask.shape[0] == hidden_dim and mask.shape[1] == input_dim:
            mask[hidden_idx, input_idx] = 0.0
        else:
            raise RuntimeError(f"Unexpected first-layer mask shape {tuple(mask.shape)}")


def zero_all_edges_from_input(model, input_idx: int, input_dim: int, hidden_dim: int) -> None:
    mask = get_mask_tensor(get_act_layers(model)[0])
    with torch.no_grad():
        if mask.shape[0] == input_dim and mask.shape[1] == hidden_dim:
            mask[input_idx, :] = 0.0
        elif mask.shape[0] == hidden_dim and mask.shape[1] == input_dim:
            mask[:, input_idx] = 0.0
        else:
            raise RuntimeError(f"Unexpected first-layer mask shape {tuple(mask.shape)}")


def zero_hidden_output_edge(model, hidden_idx: int, hidden_dim: int) -> None:
    mask = get_mask_tensor(get_act_layers(model)[1])
    with torch.no_grad():
        if mask.shape[0] == hidden_dim and mask.shape[1] == 1:
            mask[hidden_idx, 0] = 0.0
        elif mask.shape[0] == 1 and mask.shape[1] == hidden_dim:
            mask[0, hidden_idx] = 0.0
        else:
            raise RuntimeError(f"Unexpected second-layer mask shape {tuple(mask.shape)}")


def evaluate_delta(model, X_test: np.ndarray, y_test: np.ndarray, baseline_mse: float):
    pred = predict_model(model, X_test)
    ablated_mse = mse_np(pred, y_test)
    return ablated_mse, ablated_mse - baseline_mse


def make_base_row(args, seed: int, gt, selected_features, screen_scores, train_mse, test_mse, screen_method, top_m):
    true_vars = [int(v) for v in gt.active_variables]
    true_interactions = [tuple(sorted((int(i), int(j)))) for i, j in gt.interactions]
    selected_set = set(int(v) for v in selected_features)
    true_var_set = set(true_vars)
    true_int_set = set(true_interactions)

    return {
        "model": "SCREENED_KAN",
        "function": args.function,
        "seed": seed,
        "samples": args.samples,
        "test_samples": args.test_samples,
        "dimension": args.dimension,
        "noise": args.noise,
        "screen_method": screen_method,
        "top_m": top_m,
        "selected_screen_features": selected_features.tolist(),
        "screen_scores": screen_scores.tolist(),
        "screen_contains_all_true_vars": int(true_var_set.issubset(selected_set)),
        "screen_true_var_recall": len(true_var_set & selected_set) / len(true_var_set) if true_var_set else np.nan,
        "screen_contains_true_interactions": int(all(i in selected_set and j in selected_set for i, j in true_int_set)) if true_int_set else np.nan,
        "kan_width": args.kan_width,
        "kan_grid": args.kan_grid,
        "kan_k": args.kan_k,
        "kan_steps": args.kan_steps,
        "kan_lamb": args.kan_lamb,
        "train_mse": train_mse,
        "test_mse": test_mse,
        "baseline_mse": test_mse,
        "formula": gt.formula,
        "true_variables": true_vars,
        "true_interactions": true_interactions,
    }


def run_one(args, seed: int, screen_method: str, top_m: int):
    data = make_synthetic(
        function_name=args.function,
        n_train=args.samples,
        n_test=args.test_samples,
        d=args.dimension,
        noise=args.noise,
        seed=seed,
        standardize_target=True,
    )
    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]
    gt = data["ground_truth"]

    selected_features, screen_scores = screen_features(X_train, y_train, screen_method, top_m, seed, args.rf_trees)
    original_to_local = {int(orig): local for local, orig in enumerate(selected_features)}

    X_train_s = X_train[:, selected_features]
    X_test_s = X_test[:, selected_features]

    model = train_pykan(
        X_train_s,
        y_train,
        X_test_s,
        y_test,
        args.kan_width,
        args.kan_grid,
        args.kan_k,
        args.kan_steps,
        args.kan_lamb,
        seed,
    )

    train_mse = mse_np(predict_model(model, X_train_s), y_train)
    test_mse = mse_np(predict_model(model, X_test_s), y_test)

    input_dim = len(selected_features)
    hidden_dim = args.kan_width
    saved = save_masks(model)

    base = make_base_row(args, seed, gt, selected_features, screen_scores, train_mse, test_mse, screen_method, top_m)
    rows: List[Dict] = []

    true_vars = [int(v) for v in gt.active_variables]
    variables_to_test = list(dict.fromkeys([int(v) for v in args.variables] + true_vars))
    variables_to_test = [v for v in variables_to_test if 0 <= v < args.dimension]

    path_delta: Dict[int, float] = {}
    edge_delta: Dict[Tuple[int, int], float] = {}

    for original_var in variables_to_test:
        row = dict(base)
        row.update({
            "row_type": "feature_path",
            "target": f"x{original_var}",
            "original_variable": original_var,
            "local_variable": original_to_local.get(original_var, ""),
            "hidden": "",
            "pair": "",
            "pair_in_screened_set": "",
            "ablated_mse": np.nan,
            "delta_mse": np.nan,
            "synergy": np.nan,
            "is_true_active": int(original_var in set(true_vars)),
            "is_true_interaction": "",
        })
        if original_var in original_to_local:
            restore_masks(model, saved)
            local_idx = original_to_local[original_var]
            zero_all_edges_from_input(model, local_idx, input_dim, hidden_dim)
            ablated_mse, delta = evaluate_delta(model, X_test_s, y_test, test_mse)
            path_delta[original_var] = delta
            row["ablated_mse"] = ablated_mse
            row["delta_mse"] = delta
        rows.append(row)

    if args.include_hidden_output:
        for hidden in range(hidden_dim):
            restore_masks(model, saved)
            zero_hidden_output_edge(model, hidden, hidden_dim)
            ablated_mse, delta = evaluate_delta(model, X_test_s, y_test, test_mse)
            row = dict(base)
            row.update({
                "row_type": "hidden_output_path",
                "target": f"h{hidden}->out",
                "original_variable": "",
                "local_variable": "",
                "hidden": hidden,
                "pair": "",
                "pair_in_screened_set": "",
                "ablated_mse": ablated_mse,
                "delta_mse": delta,
                "synergy": np.nan,
                "is_true_active": "",
                "is_true_interaction": "",
            })
            rows.append(row)

    if args.include_edges:
        for original_var in variables_to_test:
            if original_var not in original_to_local:
                continue
            local_idx = original_to_local[original_var]
            for hidden in range(hidden_dim):
                restore_masks(model, saved)
                zero_input_hidden_edge(model, local_idx, hidden, input_dim, hidden_dim)
                ablated_mse, delta = evaluate_delta(model, X_test_s, y_test, test_mse)
                edge_delta[(original_var, hidden)] = delta
                row = dict(base)
                row.update({
                    "row_type": "single_edge",
                    "target": f"x{original_var}->h{hidden}",
                    "original_variable": original_var,
                    "local_variable": local_idx,
                    "hidden": hidden,
                    "pair": "",
                    "pair_in_screened_set": "",
                    "ablated_mse": ablated_mse,
                    "delta_mse": delta,
                    "synergy": np.nan,
                    "is_true_active": int(original_var in set(true_vars)),
                    "is_true_interaction": "",
                })
                rows.append(row)

    true_int_set = {tuple(sorted((int(i), int(j)))) for i, j in gt.interactions}
    pairs_to_test = list(dict.fromkeys([tuple(sorted(p)) for p in args.pairs] + list(true_int_set)))

    for pair in pairs_to_test:
        i, j = pair
        in_screen = i in original_to_local and j in original_to_local
        row = dict(base)
        row.update({
            "row_type": "feature_pair_path",
            "target": f"x{i},x{j}",
            "original_variable": "",
            "local_variable": "",
            "hidden": "",
            "pair": str(pair),
            "pair_in_screened_set": int(in_screen),
            "ablated_mse": np.nan,
            "delta_mse": np.nan,
            "synergy": np.nan,
            "is_true_active": "",
            "is_true_interaction": int(pair in true_int_set),
        })
        if in_screen:
            restore_masks(model, saved)
            li, lj = original_to_local[i], original_to_local[j]
            zero_all_edges_from_input(model, li, input_dim, hidden_dim)
            zero_all_edges_from_input(model, lj, input_dim, hidden_dim)
            ablated_mse, delta = evaluate_delta(model, X_test_s, y_test, test_mse)
            row["ablated_mse"] = ablated_mse
            row["delta_mse"] = delta
            row["synergy"] = delta - path_delta.get(i, np.nan) - path_delta.get(j, np.nan)
        rows.append(row)

        if args.include_edges and in_screen:
            li, lj = original_to_local[i], original_to_local[j]
            for hidden in range(hidden_dim):
                restore_masks(model, saved)
                zero_input_hidden_edge(model, li, hidden, input_dim, hidden_dim)
                zero_input_hidden_edge(model, lj, hidden, input_dim, hidden_dim)
                ablated_mse, delta = evaluate_delta(model, X_test_s, y_test, test_mse)
                row = dict(base)
                row.update({
                    "row_type": "shared_hidden_pair_edges",
                    "target": f"x{i},x{j}->h{hidden}",
                    "original_variable": "",
                    "local_variable": "",
                    "hidden": hidden,
                    "pair": str(pair),
                    "pair_in_screened_set": 1,
                    "ablated_mse": ablated_mse,
                    "delta_mse": delta,
                    "synergy": delta - edge_delta.get((i, hidden), np.nan) - edge_delta.get((j, hidden), np.nan),
                    "is_true_active": "",
                    "is_true_interaction": int(pair in true_int_set),
                })
                rows.append(row)

    restore_masks(model, saved)
    return rows


def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "function",
        "dimension",
        "noise",
        "screen_method",
        "top_m",
        "row_type",
        "target",
        "original_variable",
        "local_variable",
        "hidden",
        "pair",
        "pair_in_screened_set",
        "is_true_active",
        "is_true_interaction",
    ]
    metric_cols = [
        "train_mse",
        "test_mse",
        "screen_contains_all_true_vars",
        "screen_true_var_recall",
        "screen_contains_true_interactions",
        "baseline_mse",
        "ablated_mse",
        "delta_mse",
        "synergy",
    ]
    summary = df.groupby(group_cols, dropna=False)[metric_cols].agg(["mean", "std", "min", "max"]).reset_index()
    summary.columns = ["_".join(str(x) for x in col if x != "").rstrip("_") for col in summary.columns]
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--function", type=str, required=True)
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--test_samples", type=int, default=2048)
    parser.add_argument("--dimension", type=int, default=100)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])

    parser.add_argument("--screen_methods", type=str, nargs="+", default=["rf"])
    parser.add_argument("--top_m", type=int, nargs="+", default=[10, 20])
    parser.add_argument("--rf_trees", type=int, default=300)

    parser.add_argument("--kan_width", type=int, default=8)
    parser.add_argument("--kan_grid", type=int, default=5)
    parser.add_argument("--kan_k", type=int, default=3)
    parser.add_argument("--kan_steps", type=int, default=50)
    parser.add_argument("--kan_lamb", type=float, default=0.0)

    parser.add_argument("--variables", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    parser.add_argument("--pairs", type=str, nargs="+", default=["2,3", "0,1", "0,4", "1,5"])
    parser.add_argument("--include_edges", action="store_true")
    parser.add_argument("--include_hidden_output", action="store_true")

    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--summary_out", type=str, default=None)
    args = parser.parse_args()
    args.pairs = [parse_pair(p) for p in args.pairs]

    all_rows: List[Dict] = []
    for seed in args.seeds:
        for method in args.screen_methods:
            for m in args.top_m:
                print(f"Running seed={seed}, screen_method={method}, top_m={m}")
                all_rows.extend(run_one(args, seed, method, m))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_rows)
    df.to_csv(out_path, index=False)
    print(f"Wrote rows to {out_path}")

    if args.summary_out is not None:
        summary_path = Path(args.summary_out)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summarize_results(df).to_csv(summary_path, index=False)
        print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
