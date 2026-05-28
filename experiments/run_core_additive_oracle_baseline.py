from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.data import make_synthetic


COEFFS = {
    "core_interaction_c01": 0.10,
    "core_interaction_c025": 0.25,
    "core_interaction_c05": 0.50,
    "core_interaction_c1": 1.00,
}


def additive_clean(X: np.ndarray) -> np.ndarray:
    return np.sin(2 * np.pi * X[:, 0]) + X[:, 1] ** 2


def interaction_clean(X: np.ndarray, c: float) -> np.ndarray:
    return float(c) * X[:, 2] * X[:, 3]


def run_one(function_name: str, samples: int, test_samples: int, dimension: int, seed: int) -> dict:
    c = COEFFS[function_name]
    data = make_synthetic(
        function_name=function_name,
        n_train=samples,
        n_test=test_samples,
        d=dimension,
        noise=0.0,
        seed=seed,
        standardize_target=True,
    )
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_test = data["y_test"].reshape(-1)
    mean = float(data["target_mean"])
    std = float(data["target_std"])

    add_train = additive_clean(X_train)
    int_train = interaction_clean(X_train, c)
    add_test = additive_clean(X_test)
    int_test = interaction_clean(X_test, c)

    y_add_test = (add_test - mean) / std
    y_add_int_test = (add_test + int_test - mean) / std
    # y_add_int_test should match y_test up to dtype roundoff. Keeping it makes
    # the table self-checking if the generator changes.
    return {
        "function": function_name,
        "c": c,
        "samples": int(samples),
        "test_samples": int(test_samples),
        "dimension": int(dimension),
        "seed": int(seed),
        "target_std": std,
        "additive_only_test_mse": float(np.mean((y_add_test - y_test) ** 2)),
        "full_formula_selfcheck_mse": float(np.mean((y_add_int_test - y_test) ** 2)),
        "raw_interaction_var_train": float(np.var(int_train)),
        "raw_interaction_var_test": float(np.var(int_test)),
        "standardized_interaction_var_test": float(np.var(int_test / std)),
        "additive_var_train": float(np.var(add_train)),
        "full_var_train": float(np.var(add_train + int_train)),
        "interaction_var_fraction_train": float(np.var(int_train) / max(np.var(add_train + int_train), 1e-12)),
    }


def summarize(detail: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["function", "c", "samples", "test_samples", "dimension"]
    numeric_cols = [
        "target_std",
        "additive_only_test_mse",
        "full_formula_selfcheck_mse",
        "raw_interaction_var_train",
        "raw_interaction_var_test",
        "standardized_interaction_var_test",
        "additive_var_train",
        "full_var_train",
        "interaction_var_fraction_train",
    ]
    out = detail.groupby(group_cols)[numeric_cols].agg(["mean", "std"]).reset_index()
    out.columns = ["_".join(str(v) for v in col if v != "").rstrip("_") for col in out.columns]
    counts = detail.groupby(group_cols).size().reset_index(name="num_runs")
    return out.merge(counts, on=group_cols, how="left")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--functions", nargs="+", default=list(COEFFS))
    parser.add_argument("--samples", type=int, nargs="+", default=[512, 1024, 2048])
    parser.add_argument("--test_samples", type=int, default=2048)
    parser.add_argument("--dimension", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--out_dir", default="results/workshop_review_tables/core_additive_oracle_baseline")
    args = parser.parse_args()

    rows = []
    for fn in args.functions:
        if fn not in COEFFS:
            raise ValueError(f"Unsupported function {fn!r}. Use one of {sorted(COEFFS)}.")
        for n in args.samples:
            for seed in args.seeds:
                rows.append(run_one(fn, int(n), args.test_samples, args.dimension, int(seed)))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    detail = pd.DataFrame(rows)
    summary = summarize(detail)
    detail.to_csv(out_dir / "core_additive_oracle_detail.csv", index=False)
    summary.to_csv(out_dir / "core_additive_oracle_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
