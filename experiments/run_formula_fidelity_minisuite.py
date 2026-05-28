from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_FUNCTIONS = [
    "formula_poly_additive",
    "formula_bilinear",
    "formula_weak_centered",
    "formula_trig_product",
    "formula_nested_trig",
    "formula_rational_product",
    "formula_division_mixed",
    "formula_exp_product",
    "formula_log_product",
    "formula_three_way_product",
    "formula_mixed_sparse",
    "formula_sqrt_energy",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the formula-fidelity mini-suite through the existing benchmark runner."
    )
    parser.add_argument("--out_dir", default="results/formula_fidelity_minisuite/default")
    parser.add_argument("--functions", nargs="+", default=DEFAULT_FUNCTIONS)
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--test_samples", type=int, default=2048)
    parser.add_argument("--dimension", type=int, default=100)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--nuisance_correlation", type=float, default=0.0)
    parser.add_argument("--n_correlated_proxies", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--screen_modes",
        nargs="+",
        default=["raw", "rf", "oracle_support", "random", "exclude_interaction"],
    )
    parser.add_argument("--top_m", type=int, default=20)
    parser.add_argument("--rf_trees", type=int, default=300)
    parser.add_argument("--kan_steps", type=int, default=30)
    parser.add_argument("--kan_width", type=int, default=8)
    parser.add_argument("--kan_grid", type=int, default=5)
    parser.add_argument("--kan_k", type=int, default=3)
    parser.add_argument("--kan_lamb", type=float, default=0.0)
    parser.add_argument("--opt", default="LBFGS")
    parser.add_argument("--compute_interactions", action="store_true")
    parser.add_argument("--hessian_points", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    detail = out_dir / "minisuite_detail.csv"
    summary = out_dir / "minisuite_summary.csv"
    fig_dir = out_dir / "figures"

    cmd = [
        sys.executable,
        "experiments/run_benchmark_suite.py",
        "--functions",
        *args.functions,
        "--samples",
        str(args.samples),
        "--test_samples",
        str(args.test_samples),
        "--dimension",
        str(args.dimension),
        "--noise",
        str(args.noise),
        "--nuisance_correlation",
        str(args.nuisance_correlation),
        "--n_correlated_proxies",
        str(args.n_correlated_proxies),
        "--seeds",
        *[str(seed) for seed in args.seeds],
        "--screen_modes",
        *args.screen_modes,
        "--top_m",
        str(args.top_m),
        "--rf_trees",
        str(args.rf_trees),
        "--kan_steps",
        str(args.kan_steps),
        "--kan_width",
        str(args.kan_width),
        "--kan_grid",
        str(args.kan_grid),
        "--kan_k",
        str(args.kan_k),
        "--kan_lamb",
        str(args.kan_lamb),
        "--opt",
        args.opt,
        "--out",
        str(detail),
        "--summary_out",
        str(summary),
        "--fig_dir",
        str(fig_dir),
    ]
    if args.compute_interactions:
        cmd.extend(["--compute_interactions", "--hessian_points", str(args.hessian_points)])
    if args.resume:
        cmd.append("--resume")

    print("Running:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
