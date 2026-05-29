#!/usr/bin/env python3
"""Smoke-test pyKAN pruning/symbolic workflow as an audited readout layer.

This is not a full symbolic-recovery benchmark.  It asks a narrower reviewer
question: if we use pyKAN's exposed pruning APIs after training the same KAN
configuration, do the pruned input supports retain the known active variables
and interaction endpoints?  The script also records whether symbolic_formula()
can be called on the pruned model, but does not tune symbolic libraries or claim
formula equivalence.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import make_synthetic
from experiments.run_kan_native_innovation_loop import safe_feature_score, safe_edge_path_scores
from experiments.run_tuned_kan_recovery import batch_predict, canonical_pairs, mse_np, train_kan


def parse_seeds(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return sorted(dict.fromkeys(out))


def train_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        width_hidden=int(args.width_hidden),
        grid=int(args.grid),
        k=int(args.k),
        steps=int(args.steps),
        lamb=float(args.lamb),
        opt=args.opt,
        update_grid=bool(args.update_grid),
        grid_update_num=int(args.grid_update_num),
        batch=int(args.batch),
    )


def tensor_to_int_list(value) -> list[int]:
    if value is None:
        return []
    try:
        arr = value.detach().cpu().numpy().reshape(-1)
    except Exception:
        arr = np.asarray(value).reshape(-1)
    return [int(x) for x in arr.tolist()]


def rank_desc(scores: np.ndarray, idx: int) -> int:
    order = sorted(range(len(scores)), key=lambda j: (-float(scores[j]), int(j)))
    return int(order.index(int(idx)) + 1)


def call_quietly(fn, *args, **kwargs):
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
        return fn(*args, **kwargs), buffer.getvalue()


def symbolic_smoke(model) -> tuple[int, str]:
    try:
        formula, _ = call_quietly(model.symbolic_formula)
        text = str(formula)
        return 1, text[:500]
    except Exception as exc:
        return 0, f"{type(exc).__name__}: {exc}"[:500]


def run_one(args: argparse.Namespace, seed: int) -> list[dict]:
    data = make_synthetic(
        function_name=args.function,
        n_train=args.samples,
        n_test=args.test_samples,
        d=args.dimension,
        noise=args.noise,
        seed=seed,
        standardize_target=True,
        nuisance_correlation=args.nuisance_correlation,
        n_correlated_proxies=args.n_correlated_proxies,
    )
    X_train = data["X_train"].astype(np.float32)
    y_train = data["y_train"].astype(np.float32)
    X_test = data["X_test"].astype(np.float32)
    y_test = data["y_test"].astype(np.float32)
    gt = data["ground_truth"]
    true_vars = set(int(v) for v in gt.active_variables)
    true_pairs = canonical_pairs(gt.interactions)
    endpoints = set(int(v) for pair in true_pairs for v in pair)

    model = train_kan(
        X_train,
        y_train,
        X_test,
        y_test,
        train_args(args),
        seed=seed,
        device=args.device,
    )
    full_pred = batch_predict(model, X_test, device=args.device, batch_size=args.batch_size)
    full_mse = mse_np(full_pred, y_test)

    feature_scores = safe_feature_score(model, args.dimension)
    edge_scores, _, endpoint_mass = safe_edge_path_scores(model, args.dimension)
    hybrid_scores = feature_scores + edge_scores + endpoint_mass
    endpoint_rank_feature = max(rank_desc(feature_scores, v) for v in endpoints) if endpoints else -1
    endpoint_rank_hybrid = max(rank_desc(hybrid_scores, v) for v in endpoints) if endpoints else -1

    rows: list[dict] = []
    for threshold in args.thresholds:
        for workflow in args.workflows:
            pruned = None
            error = ""
            try:
                if workflow == "prune_input":
                    pruned, _ = call_quietly(model.prune_input, threshold=float(threshold), log_history=False)
                elif workflow == "prune":
                    pruned, _ = call_quietly(
                        model.prune,
                        node_th=float(args.node_threshold),
                        edge_th=float(threshold),
                    )
                else:
                    raise ValueError(f"unknown workflow={workflow}")
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"

            selected = tensor_to_int_list(getattr(pruned, "input_id", None)) if pruned is not None else []
            selected_set = set(selected)
            pruned_mse = np.nan
            if pruned is not None and selected:
                try:
                    # pyKAN pruned models keep an `input_id` mapping to the
                    # original feature coordinates, so the safest prediction
                    # path is to pass the full-dimensional input.  Some local
                    # versions instead expect the reduced input; keep that as
                    # a fallback.
                    pred = batch_predict(pruned, X_test, device=args.device, batch_size=args.batch_size)
                    pruned_mse = mse_np(pred, y_test)
                except Exception as exc:
                    try:
                        Xp = X_test[:, selected].astype(np.float32)
                        pred = batch_predict(pruned, Xp, device=args.device, batch_size=args.batch_size)
                        pruned_mse = mse_np(pred, y_test)
                        error = (error + "; " if error else "") + "predict_used_reduced_input_fallback"
                    except Exception as exc2:
                        error = (
                            (error + "; " if error else "")
                            + f"predict {type(exc).__name__}: {exc}; "
                            + f"fallback {type(exc2).__name__}: {exc2}"
                        )

            symbolic_ok = 0
            symbolic_text = ""
            if args.symbolic_smoke and pruned is not None:
                symbolic_ok, symbolic_text = symbolic_smoke(pruned)

            rows.append(
                {
                    "function": args.function,
                    "seed": int(seed),
                    "samples": int(args.samples),
                    "dimension": int(args.dimension),
                    "noise": float(args.noise),
                    "width_hidden": int(args.width_hidden),
                    "grid": int(args.grid),
                    "k": int(args.k),
                    "steps": int(args.steps),
                    "lamb": float(args.lamb),
                    "update_grid": int(bool(args.update_grid)),
                    "workflow": workflow,
                    "threshold": float(threshold),
                    "full_mse": float(full_mse),
                    "pruned_mse": float(pruned_mse) if np.isfinite(pruned_mse) else np.nan,
                    "support_size": int(len(selected)),
                    "selected_inputs": json.dumps(selected),
                    "contains_all_true_vars": int(true_vars.issubset(selected_set)) if true_vars else -1,
                    "endpoint_contains": int(endpoints.issubset(selected_set)) if endpoints else -1,
                    "endpoint_recall": len(endpoints & selected_set) / len(endpoints) if endpoints else np.nan,
                    "endpoint_rank_feature": int(endpoint_rank_feature),
                    "endpoint_rank_hybrid": int(endpoint_rank_hybrid),
                    "symbolic_formula_ok": int(symbolic_ok),
                    "symbolic_formula_text": symbolic_text,
                    "error": error,
                }
            )
    return rows


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["function", "samples", "dimension", "workflow", "threshold"]
    rows = []
    for key, sub in df.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, key))
        row.update(
            {
                "runs": int(len(sub)),
                "mean_full_mse": float(sub["full_mse"].mean()),
                "median_full_mse": float(sub["full_mse"].median()),
                "mean_pruned_mse": float(sub["pruned_mse"].mean()),
                "median_pruned_mse": float(sub["pruned_mse"].median()),
                "median_support_size": float(sub["support_size"].median()),
                "mean_support_size": float(sub["support_size"].mean()),
                "contains_all_true_vars": int(sub["contains_all_true_vars"].sum()),
                "endpoint_contains": int(sub["endpoint_contains"].sum()),
                "mean_endpoint_recall": float(sub["endpoint_recall"].mean()),
                "symbolic_formula_ok": int(sub["symbolic_formula_ok"].sum()),
                "errors": int((sub["error"].fillna("") != "").sum()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--function", default="core_interaction_c025")
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--test-samples", type=int, default=2048)
    parser.add_argument("--dimension", type=int, default=100)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--nuisance-correlation", type=float, default=0.0)
    parser.add_argument("--n-correlated-proxies", type=int, default=0)
    parser.add_argument("--seeds", default="0-9")
    parser.add_argument("--width-hidden", type=int, default=16)
    parser.add_argument("--grid", type=int, default=5)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--steps", type=int, default=75)
    parser.add_argument("--lamb", type=float, default=1e-3)
    parser.add_argument("--opt", default="LBFGS")
    parser.add_argument("--update-grid", action="store_true")
    parser.add_argument("--grid-update-num", type=int, default=5)
    parser.add_argument("--batch", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.01, 0.03, 0.05, 0.10])
    parser.add_argument("--node-threshold", type=float, default=0.01)
    parser.add_argument("--workflows", nargs="+", default=["prune_input", "prune"])
    parser.add_argument("--symbolic-smoke", action="store_true")
    parser.add_argument("--out-dir", default="results/revision/pykan_prune_symbolic_smoke")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for seed in parse_seeds(args.seeds):
        print(f"[pykan-prune-smoke] seed={seed}", flush=True)
        rows.extend(run_one(args, seed))
        detail = pd.DataFrame(rows)
        detail.to_csv(out_dir / "pykan_prune_symbolic_detail.csv", index=False)
        summarize(detail).to_csv(out_dir / "pykan_prune_symbolic_summary.csv", index=False)

    detail = pd.DataFrame(rows)
    detail.to_csv(out_dir / "pykan_prune_symbolic_detail.csv", index=False)
    summary = summarize(detail)
    summary.to_csv(out_dir / "pykan_prune_symbolic_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"Saved: {out_dir}")


if __name__ == "__main__":
    main()
