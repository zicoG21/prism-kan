from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.data import load_uci_energy
from src.metrics import gradient_importance, permutation_importance
from src.models import train_kan, train_mlp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="heating", choices=["heating", "cooling"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default="results/energy.csv")
    parser.add_argument("--kan_steps", type=int, default=50)
    parser.add_argument("--skip_mlp", action="store_true")
    args = parser.parse_args()

    data = load_uci_energy(seed=args.seed, target=args.target)
    rows = []

    kan_res = train_kan(
        data,
        seed=args.seed,
        width_hidden=6,
        grid=5,
        steps=args.kan_steps,
        lamb=0.01,
        prune=True,
        finetune_steps=20,
        device=args.device,
    )
    for method, scores in [
        ("grad", gradient_importance(kan_res.model, data["X_test"], device=args.device)),
        ("perm", permutation_importance(kan_res.model, data["X_test"], data["y_test"], device=args.device, seed=args.seed)),
    ]:
        for name, score in zip(data["feature_names"], scores):
            rows.append({
                "dataset": "uci_energy",
                "target": args.target,
                "model": "KAN",
                "method": method,
                "test_mse_standardized": kan_res.test_mse,
                "feature": name,
                "importance": float(score),
            })

    if not args.skip_mlp:
        mlp_res = train_mlp(data, seed=args.seed, epochs=1500, device=args.device)
        for method, scores in [
            ("grad", gradient_importance(mlp_res.model, data["X_test"], device=args.device)),
            ("perm", permutation_importance(mlp_res.model, data["X_test"], data["y_test"], device=args.device, seed=args.seed)),
        ]:
            for name, score in zip(data["feature_names"], scores):
                rows.append({
                    "dataset": "uci_energy",
                    "target": args.target,
                    "model": "MLP",
                    "method": method,
                    "test_mse_standardized": mlp_res.test_mse,
                    "feature": name,
                    "importance": float(score),
                })

    df = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(df.sort_values(["model", "method", "importance"], ascending=[True, True, False]))
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
