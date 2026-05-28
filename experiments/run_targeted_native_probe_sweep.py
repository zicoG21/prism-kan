from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import make_synthetic
from experiments.run_kan_native_innovation_loop import (
    METHODS,
    Setting,
    aggregate_probe_scores,
    ensure_probes,
    select_support,
)
from experiments.run_tuned_kan_recovery import canonical_pairs, support_stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Targeted KAN-native support probe sweep without low-dimensional refits.")
    parser.add_argument("--out_dir", default="results/formula_aware_pair_scoring/targeted_native_probe_sweep")
    parser.add_argument("--base_probe_cache", default="")
    parser.add_argument("--functions", nargs="+", default=["core_interaction_c05"])
    parser.add_argument("--samples", nargs="+", type=int, default=[2048])
    parser.add_argument("--dimensions", nargs="+", type=int, default=[500])
    parser.add_argument("--top_ms", nargs="+", type=int, default=[4, 6, 8, 10, 12, 16, 20])
    parser.add_argument("--methods", nargs="+", default=["feature_stability_var", "feature_edge_hybrid", "edge_pair_hybrid"])
    parser.add_argument("--probe_seeds", nargs="+", type=int, default=[260, 261, 262, 263, 264, 265, 266, 267])
    parser.add_argument("--heldout_seeds", nargs="+", type=int, default=[])
    parser.add_argument("--no_leave_one_out", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--test_samples", type=int, default=2048)
    parser.add_argument("--grid", type=int, default=5)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--width_hidden", type=int, default=8)
    parser.add_argument("--lamb", type=float, default=0.001)
    parser.add_argument("--probe_steps", type=int, default=35)
    parser.add_argument("--opt", default="LBFGS")
    parser.add_argument("--update_grid", action="store_true")
    parser.add_argument("--no_update_grid", dest="update_grid", action="store_false")
    parser.set_defaults(update_grid=False)
    parser.add_argument("--grid_update_num", type=int, default=10)
    parser.add_argument("--batch", type=int, default=-1)
    parser.add_argument("--probe_variable_points", type=int, default=512)
    parser.add_argument("--pred_batch_size", type=int, default=4096)
    parser.add_argument("--keep_top_pairs", type=int, default=120)
    parser.add_argument("--force_probe", action="store_true")
    return parser.parse_args()


def device_from_arg(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


def seed_label(seed: int | None) -> str:
    return "all" if seed is None else str(int(seed))


def load_or_seed_cache(base_probe_cache: str, probe_path: Path) -> None:
    if probe_path.exists() or not base_probe_cache:
        return
    base = Path(base_probe_cache)
    if base.exists():
        probe_path.write_text(base.read_text(encoding="utf-8"), encoding="utf-8")


def true_structure(function_name: str, samples: int, dimension: int, noise: float, test_samples: int) -> tuple[tuple[int, ...], tuple[tuple[int, int], ...]]:
    data = make_synthetic(
        function_name=function_name,
        n_train=samples,
        n_test=test_samples,
        d=dimension,
        noise=noise,
        seed=0,
        standardize_target=True,
    )
    gt = data["ground_truth"]
    return tuple(int(v) for v in gt.active_variables), canonical_pairs(gt.interactions)


def main() -> None:
    args = parse_args()
    bad_methods = [m for m in args.methods if m not in METHODS]
    if bad_methods:
        raise ValueError(f"Unknown methods: {bad_methods}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    probe_path = out_dir / "probe_cache.csv"
    load_or_seed_cache(args.base_probe_cache, probe_path)
    device = device_from_arg(args.device)
    print(f"Using device={device}")
    print(f"Writing results under {out_dir}")

    probe_args = SimpleNamespace(**vars(args))
    rows: list[dict] = []
    for function_name in args.functions:
        for samples in args.samples:
            for dimension in args.dimensions:
                setting = Setting(function_name, int(samples), int(dimension), max(args.top_ms))
                probes = ensure_probes(
                    setting=setting,
                    probe_seeds=args.probe_seeds,
                    args=probe_args,
                    device=device,
                    probe_path=probe_path,
                )
                true_vars, true_interactions = true_structure(
                    function_name,
                    int(samples),
                    int(dimension),
                    args.noise,
                    args.test_samples,
                )
                heldouts: list[int | None] = [None]
                if not args.no_leave_one_out:
                    heldouts.extend(args.heldout_seeds or args.probe_seeds)
                for heldout in heldouts:
                    if heldout is None:
                        stable = probes.copy()
                    else:
                        stable = probes[pd.to_numeric(probes["seed"], errors="coerce") != int(heldout)].copy()
                    if stable.empty:
                        continue
                    agg = aggregate_probe_scores(stable, int(dimension))
                    for method in args.methods:
                        for top_m in args.top_ms:
                            support, meta = select_support(method, agg, int(top_m), int(dimension))
                            stats = support_stats(np.asarray(support, dtype=int), true_vars, true_interactions)
                            rows.append(
                                {
                                    "function": function_name,
                                    "samples": int(samples),
                                    "dimension": int(dimension),
                                    "method": method,
                                    "top_m": int(top_m),
                                    "heldout_seed": seed_label(heldout),
                                    "num_probe_rows": int(len(stable)),
                                    "selected_screen_features": json.dumps([int(v) for v in support]),
                                    "top_selection_variables": json.dumps([int(v) for v in meta.get("top_selection_variables", [])]),
                                    "top_edge_pairs": json.dumps(meta.get("top_edge_pairs", [])),
                                    **stats,
                                }
                            )

    detail = pd.DataFrame(rows)
    detail.to_csv(out_dir / "support_sweep_detail.csv", index=False)
    if not detail.empty:
        numeric = [
            "screen_contains_all_true_vars",
            "screen_true_var_recall",
            "screen_contains_all_interaction_endpoints",
            "screen_interaction_endpoint_recall",
            "screen_contains_true_interactions",
        ]
        for col in numeric:
            detail[col] = pd.to_numeric(detail[col], errors="coerce")
        summary_source = detail if args.no_leave_one_out else detail[detail["heldout_seed"].astype(str).ne("all")]
        summary = (
            summary_source
            .groupby(["function", "samples", "dimension", "method", "top_m"], dropna=False)[numeric]
            .agg(["mean", "std"])
            .reset_index()
        )
        summary.columns = ["_".join(str(x) for x in col if x != "").rstrip("_") for col in summary.columns]
        counts = (
            summary_source
            .groupby(["function", "samples", "dimension", "method", "top_m"], dropna=False)
            .size()
            .reset_index(name="num_support_evals")
        )
        summary = summary.merge(counts, on=["function", "samples", "dimension", "method", "top_m"], how="left")
    else:
        summary = pd.DataFrame()
    summary.to_csv(out_dir / "support_sweep_summary.csv", index=False)
    print(summary.to_string(index=False) if not summary.empty else "No rows.")


if __name__ == "__main__":
    main()
