from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/sweep.csv")
    parser.add_argument("--function", default="core_interaction")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--samples_grid", nargs="+", type=int, default=[256, 512, 1024, 2048])
    parser.add_argument("--noise_grid", nargs="+", type=float, default=[0.0, 0.01, 0.05, 0.1])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--kan_steps", type=int, default=30)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    tmp_dir = root / "results" / "_tmp_sweep"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    csvs = []
    for n in args.samples_grid:
        for noise in args.noise_grid:
            tmp_csv = tmp_dir / f"{args.function}_n{n}_noise{noise}.csv"
            cmd = [
                sys.executable,
                str(root / "experiments" / "run_synthetic.py"),
                "--function", args.function,
                "--samples", str(n),
                "--noise", str(noise),
                "--seeds", *map(str, args.seeds),
                "--device", args.device,
                "--kan_steps", str(args.kan_steps),
                "--out", str(tmp_csv),
            ]
            print(" ".join(cmd))
            subprocess.run(cmd, check=True)
            csvs.append(tmp_csv)

    df = pd.concat([pd.read_csv(c) for c in csvs], ignore_index=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    summary = (
        df.groupby(["model", "explain_method", "function", "samples", "noise"])
        .agg(
            mean_test_mse=("test_mse", "mean"),
            std_test_mse=("test_mse", "std"),
            mean_variable_f1=("variable_f1", "mean"),
            std_variable_f1=("variable_f1", "std"),
        )
        .reset_index()
    )
    summary_out = out.with_name(out.stem + "_summary.csv")
    summary.to_csv(summary_out, index=False)
    print(f"Saved raw: {out}")
    print(f"Saved summary: {summary_out}")


if __name__ == "__main__":
    main()
