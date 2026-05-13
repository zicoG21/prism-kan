from __future__ import annotations

import argparse
import ast
import json
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
    ranking_metrics,
    score_distribution_summary,
)
from src.models import train_kan, train_mlp


def build_explanation_row(*, model_name, method, seed, args, result, scores, gt):
    """Create one CSV row for a model/explanation-method pair."""
    scores = list(map(float, scores))
    k_active = len(gt.active_variables)
    pred_set = topk_set(scores, k_active)
    p, r, f1 = precision_recall_f1(gt.active_variables, pred_set)
    auroc, auprc = ranking_metrics(scores, gt.active_variables, args.dimension)
    score_summary = score_distribution_summary(scores, gt.active_variables, args.dimension)

    row = {
        "model": model_name,
        "explain_method": method,
        "seed": seed,
        "function": args.function,
        "samples": args.samples,
        "dimension": args.dimension,
        "noise": args.noise,
        "test_mse": result.test_mse,
        "train_mse": result.train_mse,
        "selected_variables": sorted(pred_set),
        "true_variables": list(gt.active_variables),
        "variable_precision": p,
        "variable_recall": r,
        "variable_f1": f1,
        "variable_auroc": auroc,
        "variable_auprc": auprc,
        "importance_scores": json.dumps(scores),
        "formula": gt.formula,
        **score_summary,
        **{f"score_x{i}": float(scores[i]) for i in range(len(scores))},
    }
    return row, pred_set


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
            row, pred_set = build_explanation_row(
                model_name="KAN",
                method=method,
                seed=seed,
                args=args,
                result=kan_res,
                scores=scores,
                gt=gt,
            )

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
            row, _ = build_explanation_row(
                model_name="MLP",
                method=method,
                seed=seed,
                args=args,
                result=mlp_res,
                scores=scores,
                gt=gt,
            )
            results.append(row)

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
    parser.add_argument(
        "--scores_out",
        default=None,
        help="Optional path for a long-form per-variable importance score CSV.",
    )

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

    if args.scores_out is not None:
        score_rows = []
        for _, row in df.iterrows():
            true_vars = set(row["true_variables"] if isinstance(row["true_variables"], list) else ast.literal_eval(row["true_variables"]))
            for j in range(args.dimension):
                col = f"score_x{j}"
                if col in row:
                    score_rows.append({
                        "model": row["model"],
                        "explain_method": row["explain_method"],
                        "seed": row["seed"],
                        "function": row["function"],
                        "samples": row["samples"],
                        "dimension": row["dimension"],
                        "noise": row["noise"],
                        "variable": j,
                        "score": row[col],
                        "is_active": int(j in true_vars),
                    })
        scores_out = Path(args.scores_out)
        scores_out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(score_rows).to_csv(scores_out, index=False)
        print(f"Saved per-variable scores: {scores_out}")

    print(df[["model", "explain_method", "seed", "test_mse", "selected_variables", "variable_f1", "variable_auroc", "variable_auprc"]])
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
