from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.data import make_synthetic
from experiments.run_tuned_kan_recovery import (
    batch_predict,
    canonical_pairs,
    mse_np,
    train_kan,
)


def product_surface_sanity(
    model,
    X_background: np.ndarray,
    local_i: int,
    local_j: int,
    *,
    device: str,
    grid_size: int,
    background_size: int,
    seed: int,
    pred_batch_size: int,
) -> dict:
    rng = np.random.default_rng(int(seed) + 8803)
    if len(X_background) > background_size:
        idx = rng.choice(len(X_background), size=int(background_size), replace=False)
        background = X_background[idx].copy()
    else:
        background = X_background.copy()

    qi = np.linspace(0.05, 0.95, int(grid_size))
    grid_i = np.quantile(background[:, local_i], qi)
    grid_j = np.quantile(background[:, local_j], qi)
    grid_i = np.unique(grid_i)
    grid_j = np.unique(grid_j)
    if len(grid_i) < 2:
        grid_i = np.linspace(float(np.min(background[:, local_i])), float(np.max(background[:, local_i])), int(grid_size))
    if len(grid_j) < 2:
        grid_j = np.linspace(float(np.min(background[:, local_j])), float(np.max(background[:, local_j])), int(grid_size))

    f0 = float(np.mean(batch_predict(model, background, device=device, batch_size=pred_batch_size)))
    fi = []
    for a in grid_i:
        Xp = background.copy()
        Xp[:, local_i] = float(a)
        fi.append(float(np.mean(batch_predict(model, Xp, device=device, batch_size=pred_batch_size))))
    fj = []
    for b in grid_j:
        Xp = background.copy()
        Xp[:, local_j] = float(b)
        fj.append(float(np.mean(batch_predict(model, Xp, device=device, batch_size=pred_batch_size))))
    fi = np.asarray(fi, dtype=np.float64)
    fj = np.asarray(fj, dtype=np.float64)

    fij = np.empty((len(grid_i), len(grid_j)), dtype=np.float64)
    for ai, a in enumerate(grid_i):
        for bj, b in enumerate(grid_j):
            Xp = background.copy()
            Xp[:, local_i] = float(a)
            Xp[:, local_j] = float(b)
            fij[ai, bj] = float(np.mean(batch_predict(model, Xp, device=device, batch_size=pred_batch_size)))

    h = fij - fi[:, None] - fj[None, :] + f0
    product = np.outer(grid_i - np.mean(background[:, local_i]), grid_j - np.mean(background[:, local_j]))
    h_vec = h.reshape(-1)
    p_vec = product.reshape(-1)
    h_center = h_vec - h_vec.mean()
    p_center = p_vec - p_vec.mean()
    denom = float(np.linalg.norm(h_center) * np.linalg.norm(p_center)) + 1e-12
    corr = float((h_center @ p_center) / denom)
    slope = float((p_center @ h_center) / (p_center @ p_center + 1e-12))
    fitted = slope * p_center + h_vec.mean()
    ss_res = float(np.sum((h_vec - fitted) ** 2))
    ss_tot = float(np.sum((h_vec - h_vec.mean()) ** 2)) + 1e-12
    r2 = float(1.0 - ss_res / ss_tot)
    residual_ratio = float(np.sqrt(ss_res / (float(np.sum(h_center * h_center)) + 1e-12)))
    energy = float(np.mean(h_center * h_center))
    return {
        "surface_corr_product": corr,
        "surface_r2_product": r2,
        "surface_product_slope": slope,
        "surface_residual_ratio": residual_ratio,
        "surface_interaction_energy": energy,
        "surface_grid_i_min": float(np.min(grid_i)),
        "surface_grid_i_max": float(np.max(grid_i)),
        "surface_grid_j_min": float(np.min(grid_j)),
        "surface_grid_j_max": float(np.max(grid_j)),
    }


def run_one(args: argparse.Namespace, seed: int, device: str) -> dict:
    t0 = time.time()
    data = make_synthetic(
        function_name=args.function,
        n_train=args.samples,
        n_test=args.test_samples,
        d=args.dimension,
        noise=args.noise,
        seed=int(seed),
        standardize_target=True,
        nuisance_correlation=args.nuisance_correlation,
        n_correlated_proxies=args.n_correlated_proxies,
    )
    X_train_full = data["X_train"].astype(np.float32)
    y_train = data["y_train"].astype(np.float32)
    X_test_full = data["X_test"].astype(np.float32)
    y_test = data["y_test"].astype(np.float32)
    gt = data["ground_truth"]
    true_vars = np.asarray(sorted(int(v) for v in gt.active_variables), dtype=int)
    true_pairs = canonical_pairs(gt.interactions)
    pair = tuple(int(v) for v in args.pair)
    if pair not in true_pairs and tuple(reversed(pair)) not in true_pairs:
        raise ValueError(f"Requested pair {pair} is not in ground-truth pairs {true_pairs}")
    local_i = int(np.where(true_vars == pair[0])[0][0])
    local_j = int(np.where(true_vars == pair[1])[0][0])

    train_args = SimpleNamespace(
        width_hidden=args.width_hidden,
        grid=args.grid,
        k=args.k,
        steps=args.steps,
        lamb=args.lamb,
        opt=args.opt,
        update_grid=args.update_grid,
        grid_update_num=args.grid_update_num,
        batch=args.batch,
    )
    model = train_kan(
        X_train_full[:, true_vars],
        y_train,
        X_test_full[:, true_vars],
        y_test,
        train_args,
        seed=int(args.refit_seed_offset + seed),
        device=device,
    )
    pred = batch_predict(model, X_test_full[:, true_vars], device=device, batch_size=args.pred_batch_size)
    row = {
        "function": args.function,
        "seed": int(seed),
        "samples": int(args.samples),
        "test_samples": int(args.test_samples),
        "dimension": int(args.dimension),
        "noise": float(args.noise),
        "nuisance_correlation": float(args.nuisance_correlation),
        "n_correlated_proxies": int(args.n_correlated_proxies),
        "true_variables": true_vars.tolist(),
        "pair": tuple(pair),
        "width_hidden": int(args.width_hidden),
        "grid": int(args.grid),
        "k": int(args.k),
        "steps": int(args.steps),
        "lamb": float(args.lamb),
        "test_mse": mse_np(pred, y_test),
        "runtime_sec": float(time.time() - t0),
    }
    row.update(
        product_surface_sanity(
            model,
            X_test_full[:, true_vars],
            local_i,
            local_j,
            device=device,
            grid_size=args.surface_grid,
            background_size=args.surface_background,
            seed=int(seed),
            pred_batch_size=args.pred_batch_size,
        )
    )
    row["runtime_sec"] = float(time.time() - t0)
    return row


def summarize(detail: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "function",
        "samples",
        "test_samples",
        "dimension",
        "noise",
        "nuisance_correlation",
        "n_correlated_proxies",
        "pair",
        "width_hidden",
        "grid",
        "k",
        "steps",
        "lamb",
    ]
    numeric_cols = [
        "test_mse",
        "surface_corr_product",
        "surface_r2_product",
        "surface_product_slope",
        "surface_residual_ratio",
        "surface_interaction_energy",
        "runtime_sec",
    ]
    out = detail.groupby(group_cols, dropna=False)[numeric_cols].agg(["mean", "std"]).reset_index()
    out.columns = ["_".join(str(v) for v in col if v != "").rstrip("_") for col in out.columns]
    counts = detail.groupby(group_cols, dropna=False).size().reset_index(name="num_runs")
    high_shape = (
        detail.assign(surface_r2_ge_090=(detail["surface_r2_product"] >= 0.90).astype(int))
        .groupby(group_cols, dropna=False)["surface_r2_ge_090"]
        .sum()
        .reset_index(name="surface_r2_ge_090_successes")
    )
    return out.merge(counts, on=group_cols, how="left").merge(high_shape, on=group_cols, how="left")


def main() -> None:
    parser = argparse.ArgumentParser(description="True-support KAN bivariate formula-shape sanity check.")
    parser.add_argument("--function", default="core_interaction_c025")
    parser.add_argument("--samples", type=int, nargs="+", default=[1024])
    parser.add_argument("--test_samples", type=int, default=2048)
    parser.add_argument("--dimension", type=int, default=100)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--nuisance_correlation", type=float, default=0.0)
    parser.add_argument("--n_correlated_proxies", type=int, default=0)
    parser.add_argument("--pair", type=int, nargs=2, default=[2, 3])
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--width_hidden", type=int, default=8)
    parser.add_argument("--grid", type=int, default=5)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--lamb", type=float, default=1e-3)
    parser.add_argument("--opt", default="LBFGS")
    parser.add_argument("--update_grid", action="store_true")
    parser.add_argument("--grid_update_num", type=int, default=5)
    parser.add_argument("--batch", type=int, default=-1)
    parser.add_argument("--refit_seed_offset", type=int, default=91000)
    parser.add_argument("--surface_grid", type=int, default=17)
    parser.add_argument("--surface_background", type=int, default=256)
    parser.add_argument("--pred_batch_size", type=int, default=4096)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--out_dir", default="results/formula_surface_sanity/true_support_core_c025")
    args = parser.parse_args()

    if args.device == "auto":
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
    else:
        device = args.device

    rows = []
    for n in args.samples:
        for seed in args.seeds:
            local_args = argparse.Namespace(**vars(args))
            local_args.samples = int(n)
            print(f"Running n={n}, seed={seed}", flush=True)
            rows.append(run_one(local_args, int(seed), device))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    detail = pd.DataFrame(rows)
    detail.to_csv(out_dir / "true_support_formula_surface_detail.csv", index=False)
    summary = summarize(detail)
    summary.to_csv(out_dir / "true_support_formula_surface_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
