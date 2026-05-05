from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.data import make_synthetic
from src.metrics import (
    gradient_importance,
    permutation_importance,
    topk_set,
    precision_recall_f1,
    hessian_interaction_scores,
    topk_interactions,
)
from src.models import train_kan, train_mlp


def run_one(args, seed: int):
    data = make_synthetic(
        function_name=args.function,
        n_train=args.samples,
        n_test=args.test_samples,
        d=args.dimension,
        noise=args.noise,
        seed=seed,
    )
    gt = data["ground_truth"]
    k_active = len(gt.active_variables)

    results = []

    if not args.skip_kan:
        kan_res = train_kan(
            data,
            seed=seed,
            width_hidden=args.kan_width,
            grid=args.kan_grid,
            k=args.kan_k,
            steps=args.kan_steps,
            lamb=args.kan_lamb,
            prune=not args.no_prune,
            finetune_steps=args.kan_finetune_steps,
            device=args.device,
        )
        for method, scores in [
            ("grad", gradient_importance(kan_res.model, data["X_test"], device=args.device)),
            ("perm", permutation_importance(kan_res.model, data["X_test"], data["y_test"], device=args.device, seed=seed)),
        ]:
            pred_set = topk_set(scores, k_active)
            p, r, f1 = precision_recall_f1(gt.active_variables, pred_set)
            row = {
                "model": "KAN",
                "explain_method": method,
                "seed": seed,
                "function": args.function,
                "samples": args.samples,
                "dimension": args.dimension,
                "noise": args.noise,
                "test_mse": kan_res.test_mse,
                "train_mse": kan_res.train_mse,
                "selected_variables": sorted(pred_set),
                "true_variables": list(gt.active_variables),
                "variable_precision": p,
                "variable_recall": r,
                "variable_f1": f1,
                "formula": gt.formula,
                **{f"score_x{i}": float(scores[i]) for i in range(len(scores))},
            }

            if len(gt.interactions) > 0 and args.compute_interactions:
                H = hessian_interaction_scores(
                    kan_res.model,
                    data["X_test"],
                    device=args.device,
                    max_points=args.hessian_points,
                    candidate_variables=sorted(pred_set),
                )
                pred_interactions = topk_interactions(H, len(gt.interactions))
                ip, ir, if1 = precision_recall_f1(gt.interactions, pred_interactions)
                row.update({
                    "selected_interactions": sorted(pred_interactions),
                    "true_interactions": list(gt.interactions),
                    "interaction_precision": ip,
                    "interaction_recall": ir,
                    "interaction_f1": if1,
                })
            results.append(row)

    if not args.skip_mlp:
        mlp_res = train_mlp(
            data,
            seed=seed,
            hidden=args.mlp_hidden,
            depth=args.mlp_depth,
            epochs=args.mlp_epochs,
            device=args.device,
        )
        for method, scores in [
            ("grad", gradient_importance(mlp_res.model, data["X_test"], device=args.device)),
            ("perm", permutation_importance(mlp_res.model, data["X_test"], data["y_test"], device=args.device, seed=seed)),
        ]:
            pred_set = topk_set(scores, k_active)
            p, r, f1 = precision_recall_f1(gt.active_variables, pred_set)
            results.append({
                "model": "MLP",
                "explain_method": method,
                "seed": seed,
                "function": args.function,
                "samples": args.samples,
                "dimension": args.dimension,
                "noise": args.noise,
                "test_mse": mlp_res.test_mse,
                "train_mse": mlp_res.train_mse,
                "selected_variables": sorted(pred_set),
                "true_variables": list(gt.active_variables),
                "variable_precision": p,
                "variable_recall": r,
                "variable_f1": f1,
                "formula": gt.formula,
                **{f"score_x{i}": float(scores[i]) for i in range(len(scores))},
            })

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--function", default="core_interaction")
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--test_samples", type=int, default=2048)
    parser.add_argument("--dimension", type=int, default=20)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default="results/synthetic.csv")

    parser.add_argument("--kan_width", type=int, default=8)
    parser.add_argument("--kan_grid", type=int, default=5)
    parser.add_argument("--kan_k", type=int, default=3)
    parser.add_argument("--kan_steps", type=int, default=50)
    parser.add_argument("--kan_lamb", type=float, default=0.01)
    parser.add_argument("--kan_finetune_steps", type=int, default=20)
    parser.add_argument("--no_prune", action="store_true")
    parser.add_argument("--skip_kan", action="store_true")

    parser.add_argument("--mlp_hidden", type=int, default=64)
    parser.add_argument("--mlp_depth", type=int, default=2)
    parser.add_argument("--mlp_epochs", type=int, default=1500)
    parser.add_argument("--skip_mlp", action="store_true")

    parser.add_argument("--compute_interactions", action="store_true")
    parser.add_argument("--hessian_points", type=int, default=128)

    args = parser.parse_args()

    rows = []
    for seed in args.seeds:
        print(f"Running seed={seed}")
        rows.extend(run_one(args, seed))

    df = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(df[["model", "explain_method", "seed", "test_mse", "selected_variables", "variable_f1"]])
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
