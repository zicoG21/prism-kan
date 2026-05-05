from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="results/sweep.csv")
    parser.add_argument("--out", default="results/fig_variable_f1.png")
    parser.add_argument("--model", default="KAN")
    parser.add_argument("--method", default="perm")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    sub = df[(df["model"] == args.model) & (df["explain_method"] == args.method)]
    summary = sub.groupby(["samples", "noise"])["variable_f1"].mean().reset_index()

    plt.figure()
    for n, g in summary.groupby("samples"):
        g = g.sort_values("noise")
        plt.plot(g["noise"], g["variable_f1"], marker="o", label=f"n={n}")
    plt.xlabel("Noise level")
    plt.ylabel("Variable recovery F1")
    plt.title(f"{args.model} variable recovery with {args.method} importance")
    plt.legend()
    plt.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=200)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
