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


def to_numpy_prediction(y) -> np.ndarray:
    if isinstance(y, torch.Tensor):
        return y.detach().cpu().numpy()
    return np.asarray(y)


def predict_model(model, X_np: np.ndarray) -> np.ndarray:
    X = torch.tensor(X_np, dtype=torch.float32)
    with torch.no_grad():
        y = model(X)
    return to_numpy_prediction(y).reshape(-1, 1)


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

    model = KAN(
        width=[d, width_hidden, 1],
        grid=grid,
        k=k,
        seed=seed,
    )

    dataset = {
        "train_input": torch.tensor(X_train, dtype=torch.float32),
        "train_label": torch.tensor(y_train, dtype=torch.float32),
        "test_input": torch.tensor(X_test, dtype=torch.float32),
        "test_label": torch.tensor(y_test, dtype=torch.float32),
    }

    if hasattr(model, "fit"):
        model.fit(dataset, opt="LBFGS", steps=steps, lamb=lamb)
    else:
        raise RuntimeError(
            "This pykan version does not expose model.fit(...). "
            "Check your installed pykan API."
        )

    return model


def get_act_layers(model):
    if not hasattr(model, "act_fun"):
        raise RuntimeError(
            "This KAN model does not have attribute `act_fun`. "
            "Run with --inspect_only and paste the printed model structure."
        )

    layers = model.act_fun
    if not isinstance(layers, (list, torch.nn.ModuleList)):
        raise RuntimeError(
            f"`model.act_fun` exists but is not a list/ModuleList. Type: {type(layers)}"
        )

    if len(layers) < 2:
        raise RuntimeError(
            f"Expected at least 2 KAN activation layers for width=[d,h,1], got {len(layers)}."
        )

    return layers


def get_mask_tensor(layer):
    if not hasattr(layer, "mask"):
        raise RuntimeError(
            f"KAN layer {type(layer)} does not have a `.mask` attribute. "
            "Need to inspect pykan internals."
        )

    mask = layer.mask
    if not isinstance(mask, torch.Tensor):
        raise RuntimeError(f"Layer mask is not a torch.Tensor. Type: {type(mask)}")

    return mask


def print_model_structure(model):
    print("\n=== Model ===")
    print(model)

    print("\n=== Named modules ===")
    for name, module in model.named_modules():
        print(name, type(module))

    print("\n=== Named parameters ===")
    for name, param in model.named_parameters():
        print(name, tuple(param.shape))

    if hasattr(model, "act_fun"):
        print("\n=== act_fun layers and masks ===")
        for ell, layer in enumerate(model.act_fun):
            print(f"act_fun[{ell}]: {type(layer)}")
            if hasattr(layer, "mask"):
                print(f"  mask shape: {tuple(layer.mask.shape)}")
            for attr in ["coef", "scale_base", "scale_sp", "grid"]:
                if hasattr(layer, attr):
                    value = getattr(layer, attr)
                    if isinstance(value, torch.Tensor):
                        print(f"  {attr} shape: {tuple(value.shape)}")
                    else:
                        print(f"  {attr}: {type(value)}")


def save_masks(model) -> List[torch.Tensor]:
    layers = get_act_layers(model)
    masks = []
    for layer in layers:
        mask = get_mask_tensor(layer)
        masks.append(mask.detach().clone())
    return masks


def restore_masks(model, saved_masks: List[torch.Tensor]) -> None:
    layers = get_act_layers(model)
    with torch.no_grad():
        for layer, saved in zip(layers, saved_masks):
            mask = get_mask_tensor(layer)
            mask.copy_(saved)


def zero_input_hidden_edge(
    model,
    input_idx: int,
    hidden_idx: int,
    input_dim: int,
    hidden_dim: int,
) -> None:
    layer0 = get_act_layers(model)[0]
    mask = get_mask_tensor(layer0)

    with torch.no_grad():
        if mask.shape[0] == input_dim and mask.shape[1] == hidden_dim:
            mask[input_idx, hidden_idx] = 0.0
        elif mask.shape[0] == hidden_dim and mask.shape[1] == input_dim:
            mask[hidden_idx, input_idx] = 0.0
        else:
            raise RuntimeError(
                f"Unexpected first-layer mask shape {tuple(mask.shape)} "
                f"for input_dim={input_dim}, hidden_dim={hidden_dim}."
            )


def zero_all_edges_from_input(
    model,
    input_idx: int,
    input_dim: int,
    hidden_dim: int,
) -> None:
    layer0 = get_act_layers(model)[0]
    mask = get_mask_tensor(layer0)

    with torch.no_grad():
        if mask.shape[0] == input_dim and mask.shape[1] == hidden_dim:
            mask[input_idx, :] = 0.0
        elif mask.shape[0] == hidden_dim and mask.shape[1] == input_dim:
            mask[:, input_idx] = 0.0
        else:
            raise RuntimeError(
                f"Unexpected first-layer mask shape {tuple(mask.shape)} "
                f"for input_dim={input_dim}, hidden_dim={hidden_dim}."
            )


def zero_hidden_output_edge(model, hidden_idx: int, hidden_dim: int) -> None:
    layer1 = get_act_layers(model)[1]
    mask = get_mask_tensor(layer1)

    with torch.no_grad():
        if mask.shape[0] == hidden_dim and mask.shape[1] == 1:
            mask[hidden_idx, 0] = 0.0
        elif mask.shape[0] == 1 and mask.shape[1] == hidden_dim:
            mask[0, hidden_idx] = 0.0
        else:
            raise RuntimeError(
                f"Unexpected second-layer mask shape {tuple(mask.shape)} "
                f"for hidden_dim={hidden_dim}."
            )


def zero_all_edges_for_pair(
    model,
    pair: Tuple[int, int],
    input_dim: int,
    hidden_dim: int,
) -> None:
    i, j = pair
    zero_all_edges_from_input(model, i, input_dim, hidden_dim)
    zero_all_edges_from_input(model, j, input_dim, hidden_dim)


def evaluate_delta(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    baseline_mse: float,
) -> Tuple[float, float]:
    pred = predict_model(model, X_test)
    ablated_mse = mse_np(pred, y_test)
    delta = ablated_mse - baseline_mse
    return ablated_mse, delta


def run_for_seed(args, seed: int) -> List[Dict]:
    print(f"\nTraining KAN for seed={seed}")

    data = make_synthetic(
        function_name=args.function,
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

    model = train_pykan(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        width_hidden=args.kan_width,
        grid=args.kan_grid,
        k=args.kan_k,
        steps=args.kan_steps,
        lamb=args.kan_lamb,
        seed=seed,
    )

    if args.inspect_only:
        print_model_structure(model)
        return []

    baseline_pred = predict_model(model, X_test)
    baseline_mse = mse_np(baseline_pred, y_test)
    print(f"seed={seed}, baseline test MSE={baseline_mse:.6g}")

    input_dim = args.dimension
    hidden_dim = args.kan_width
    rows: List[Dict] = []

    variables = [v for v in args.variables if 0 <= v < input_dim]
    pairs = [p for p in args.pairs if p[0] < input_dim and p[1] < input_dim]

    saved = save_masks(model)

    path_delta: Dict[int, float] = {}
    edge_delta: Dict[Tuple[int, int], float] = {}

    for variable in variables:
        restore_masks(model, saved)
        zero_all_edges_from_input(
            model=model,
            input_idx=variable,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
        )
        ablated_mse, delta = evaluate_delta(model, X_test, y_test, baseline_mse)
        path_delta[variable] = delta

        rows.append(
            {
                "row_type": "feature_path",
                "model": "KAN",
                "function": args.function,
                "seed": seed,
                "samples": args.samples,
                "test_samples": args.test_samples,
                "dimension": input_dim,
                "noise": args.noise,
                "kan_width": args.kan_width,
                "kan_grid": args.kan_grid,
                "kan_k": args.kan_k,
                "kan_steps": args.kan_steps,
                "kan_lamb": args.kan_lamb,
                "target": f"x{variable}",
                "variable": variable,
                "hidden": "",
                "pair": "",
                "baseline_mse": baseline_mse,
                "ablated_mse": ablated_mse,
                "delta_mse": delta,
                "synergy": np.nan,
                "is_true_active": int(variable in gt.active_variables),
                "is_true_interaction": "",
                "formula": gt.formula,
                "true_variables": list(gt.active_variables),
                "true_interactions": list(gt.interactions),
            }
        )

    for hidden in range(hidden_dim):
        restore_masks(model, saved)
        zero_hidden_output_edge(model=model, hidden_idx=hidden, hidden_dim=hidden_dim)
        ablated_mse, delta = evaluate_delta(model, X_test, y_test, baseline_mse)

        rows.append(
            {
                "row_type": "hidden_output_path",
                "model": "KAN",
                "function": args.function,
                "seed": seed,
                "samples": args.samples,
                "test_samples": args.test_samples,
                "dimension": input_dim,
                "noise": args.noise,
                "kan_width": args.kan_width,
                "kan_grid": args.kan_grid,
                "kan_k": args.kan_k,
                "kan_steps": args.kan_steps,
                "kan_lamb": args.kan_lamb,
                "target": f"h{hidden}->out",
                "variable": "",
                "hidden": hidden,
                "pair": "",
                "baseline_mse": baseline_mse,
                "ablated_mse": ablated_mse,
                "delta_mse": delta,
                "synergy": np.nan,
                "is_true_active": "",
                "is_true_interaction": "",
                "formula": gt.formula,
                "true_variables": list(gt.active_variables),
                "true_interactions": list(gt.interactions),
            }
        )

    if args.include_edges:
        for variable in variables:
            for hidden in range(hidden_dim):
                restore_masks(model, saved)
                zero_input_hidden_edge(
                    model=model,
                    input_idx=variable,
                    hidden_idx=hidden,
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                )
                ablated_mse, delta = evaluate_delta(model, X_test, y_test, baseline_mse)
                edge_delta[(variable, hidden)] = delta

                rows.append(
                    {
                        "row_type": "single_edge",
                        "model": "KAN",
                        "function": args.function,
                        "seed": seed,
                        "samples": args.samples,
                        "test_samples": args.test_samples,
                        "dimension": input_dim,
                        "noise": args.noise,
                        "kan_width": args.kan_width,
                        "kan_grid": args.kan_grid,
                        "kan_k": args.kan_k,
                        "kan_steps": args.kan_steps,
                        "kan_lamb": args.kan_lamb,
                        "target": f"x{variable}->h{hidden}",
                        "variable": variable,
                        "hidden": hidden,
                        "pair": "",
                        "baseline_mse": baseline_mse,
                        "ablated_mse": ablated_mse,
                        "delta_mse": delta,
                        "synergy": np.nan,
                        "is_true_active": int(variable in gt.active_variables),
                        "is_true_interaction": "",
                        "formula": gt.formula,
                        "true_variables": list(gt.active_variables),
                        "true_interactions": list(gt.interactions),
                    }
                )

    true_interaction_set = {tuple(sorted(p)) for p in gt.interactions}

    for pair in pairs:
        i, j = pair

        restore_masks(model, saved)
        zero_all_edges_for_pair(
            model=model,
            pair=pair,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
        )
        ablated_mse, delta = evaluate_delta(model, X_test, y_test, baseline_mse)
        synergy = delta - path_delta.get(i, np.nan) - path_delta.get(j, np.nan)

        rows.append(
            {
                "row_type": "feature_pair_path",
                "model": "KAN",
                "function": args.function,
                "seed": seed,
                "samples": args.samples,
                "test_samples": args.test_samples,
                "dimension": input_dim,
                "noise": args.noise,
                "kan_width": args.kan_width,
                "kan_grid": args.kan_grid,
                "kan_k": args.kan_k,
                "kan_steps": args.kan_steps,
                "kan_lamb": args.kan_lamb,
                "target": f"x{i},x{j}",
                "variable": "",
                "hidden": "",
                "pair": str(pair),
                "baseline_mse": baseline_mse,
                "ablated_mse": ablated_mse,
                "delta_mse": delta,
                "synergy": synergy,
                "is_true_active": "",
                "is_true_interaction": int(pair in true_interaction_set),
                "formula": gt.formula,
                "true_variables": list(gt.active_variables),
                "true_interactions": list(gt.interactions),
            }
        )

        if args.include_edges:
            for hidden in range(hidden_dim):
                restore_masks(model, saved)
                zero_input_hidden_edge(
                    model=model,
                    input_idx=i,
                    hidden_idx=hidden,
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                )
                zero_input_hidden_edge(
                    model=model,
                    input_idx=j,
                    hidden_idx=hidden,
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                )
                ablated_mse_h, delta_h = evaluate_delta(model, X_test, y_test, baseline_mse)
                edge_i = edge_delta.get((i, hidden), np.nan)
                edge_j = edge_delta.get((j, hidden), np.nan)
                synergy_h = delta_h - edge_i - edge_j

                rows.append(
                    {
                        "row_type": "shared_hidden_pair_edges",
                        "model": "KAN",
                        "function": args.function,
                        "seed": seed,
                        "samples": args.samples,
                        "test_samples": args.test_samples,
                        "dimension": input_dim,
                        "noise": args.noise,
                        "kan_width": args.kan_width,
                        "kan_grid": args.kan_grid,
                        "kan_k": args.kan_k,
                        "kan_steps": args.kan_steps,
                        "kan_lamb": args.kan_lamb,
                        "target": f"x{i},x{j}->h{hidden}",
                        "variable": "",
                        "hidden": hidden,
                        "pair": str(pair),
                        "baseline_mse": baseline_mse,
                        "ablated_mse": ablated_mse_h,
                        "delta_mse": delta_h,
                        "synergy": synergy_h,
                        "is_true_active": "",
                        "is_true_interaction": int(pair in true_interaction_set),
                        "formula": gt.formula,
                        "true_variables": list(gt.active_variables),
                        "true_interactions": list(gt.interactions),
                    }
                )

    restore_masks(model, saved)
    return rows


def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "function",
        "dimension",
        "noise",
        "row_type",
        "target",
        "variable",
        "hidden",
        "pair",
        "is_true_active",
        "is_true_interaction",
    ]

    metric_cols = ["baseline_mse", "ablated_mse", "delta_mse", "synergy"]

    summary = (
        df.groupby(group_cols, dropna=False)[metric_cols]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )

    summary.columns = [
        "_".join(str(x) for x in col if x != "").rstrip("_")
        for col in summary.columns
    ]
    return summary


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--function", type=str, required=True)
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--test_samples", type=int, default=2048)
    parser.add_argument("--dimension", type=int, default=20)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])

    parser.add_argument("--kan_width", type=int, default=8)
    parser.add_argument("--kan_grid", type=int, default=5)
    parser.add_argument("--kan_k", type=int, default=3)
    parser.add_argument("--kan_steps", type=int, default=50)
    parser.add_argument("--kan_lamb", type=float, default=0.0)

    parser.add_argument(
        "--variables",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4, 5],
        help="Variables whose KAN input-to-hidden paths should be deleted.",
    )

    parser.add_argument(
        "--pairs",
        type=str,
        nargs="+",
        default=["2,3", "0,1", "0,4", "1,5"],
        help="Pairs to intervene on. Use format like '2,3'.",
    )

    parser.add_argument(
        "--include_edges",
        action="store_true",
        help="Also run individual input-hidden edge deletions and shared-hidden pair edge deletions.",
    )

    parser.add_argument(
        "--inspect_only",
        action="store_true",
        help="Train one model and print pykan internal structure without running interventions.",
    )

    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--summary_out", type=str, default=None)

    args = parser.parse_args()
    args.pairs = [parse_pair(p) for p in args.pairs]

    all_rows: List[Dict] = []
    for seed in args.seeds:
        rows = run_for_seed(args, seed)
        all_rows.extend(rows)

    if args.inspect_only:
        return

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(all_rows)
    df.to_csv(out_path, index=False)
    print(f"Wrote KAN path intervention rows to {out_path}")

    if args.summary_out is not None:
        summary = summarize_results(df)
        summary_path = Path(args.summary_out)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(summary_path, index=False)
        print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
