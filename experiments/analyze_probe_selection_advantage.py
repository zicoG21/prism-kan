from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.data import make_synthetic


def parse_scores(value, d: int) -> np.ndarray:
    if isinstance(value, str):
        parsed = ast.literal_eval(value)
    else:
        parsed = value
    arr = np.asarray(parsed, dtype=float).reshape(-1)
    if len(arr) < d:
        out = np.zeros(d, dtype=float)
        out[: len(arr)] = arr
        return out
    return arr[:d]


def top_set(scores: np.ndarray, m: int) -> set[int]:
    return {int(i) for i in np.argsort(-np.nan_to_num(scores, nan=0.0))[: min(m, len(scores))]}


def interaction_endpoints(pairs) -> set[int]:
    out: set[int] = set()
    for i, j in pairs:
        out.add(int(i))
        out.add(int(j))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure empirical p1>p0 support advantage in KAN probe caches.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--score_columns", nargs="+", default=["grad_scores", "feature_scores", "edge_var_scores"])
    parser.add_argument("--top_ms", nargs="+", type=int, default=[4, 8, 12, 20])
    parser.add_argument("--function", default="")
    parser.add_argument("--samples", type=int, default=0)
    parser.add_argument("--dimension", type=int, default=0)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--nuisance_correlation", type=float, default=0.0)
    parser.add_argument("--n_correlated_proxies", type=int, default=0)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if "status" in df.columns:
        df = df[df["status"].astype(str).eq("ok")].copy()
    if args.function:
        df = df[df["function"].astype(str).eq(args.function)].copy()
    if args.samples:
        df = df[pd.to_numeric(df["samples"], errors="coerce").eq(args.samples)].copy()
    if args.dimension:
        df = df[pd.to_numeric(df["dimension"], errors="coerce").eq(args.dimension)].copy()
    if df.empty:
        raise ValueError("No probe rows after filtering.")

    first = df.iloc[0]
    function = args.function or str(first["function"])
    samples = args.samples or int(first["samples"])
    dimension = args.dimension or int(first["dimension"])
    noise = float(first.get("noise", args.noise))
    rho = float(first.get("nuisance_correlation", args.nuisance_correlation))
    n_proxies = int(first.get("n_correlated_proxies", args.n_correlated_proxies))
    data = make_synthetic(
        function_name=function,
        n_train=samples,
        n_test=16,
        d=dimension,
        noise=noise,
        seed=0,
        nuisance_correlation=rho,
        n_correlated_proxies=n_proxies,
    )
    active = {int(v) for v in data["ground_truth"].active_variables}
    endpoints = interaction_endpoints(data["ground_truth"].interactions)
    nuisance = set(range(dimension)) - active
    proxy_groups = {int(k): int(v) for k, v in data.get("proxy_groups", {}).items()}

    rows = []
    freq_rows = []
    for score_col in args.score_columns:
        if score_col not in df.columns:
            continue
        scores_by_probe = [parse_scores(v, dimension) for v in df[score_col]]
        for top_m in args.top_ms:
            selected_sets = [top_set(scores, top_m) for scores in scores_by_probe]
            freqs = np.zeros(dimension, dtype=float)
            for selected in selected_sets:
                for j in selected:
                    freqs[j] += 1.0
            freqs /= max(len(selected_sets), 1)
            true_endpoint_freqs = [float(freqs[j]) for j in sorted(endpoints)]
            active_freqs = [float(freqs[j]) for j in sorted(active)]
            nuisance_freqs = [float(freqs[j]) for j in sorted(nuisance)]
            proxy_freqs = [float(freqs[j]) for j in sorted(proxy_groups)]
            p1_endpoint_min = float(np.min(true_endpoint_freqs)) if true_endpoint_freqs else np.nan
            p1_active_min = float(np.min(active_freqs)) if active_freqs else np.nan
            p0_nuisance_max = float(np.max(nuisance_freqs)) if nuisance_freqs else np.nan
            p0_nuisance_p95 = float(np.quantile(nuisance_freqs, 0.95)) if nuisance_freqs else np.nan
            rows.append(
                {
                    "score_column": score_col,
                    "top_m": int(top_m),
                    "num_probes": int(len(selected_sets)),
                    "p1_endpoint_min": p1_endpoint_min,
                    "p1_active_min": p1_active_min,
                    "p0_nuisance_max": p0_nuisance_max,
                    "p0_nuisance_p95": p0_nuisance_p95,
                    "endpoint_advantage_over_max": p1_endpoint_min - p0_nuisance_max,
                    "endpoint_advantage_over_p95": p1_endpoint_min - p0_nuisance_p95,
                    "endpoint_freqs": [float(freqs[j]) for j in sorted(endpoints)],
                    "active_freqs": [float(freqs[j]) for j in sorted(active)],
                    "top_frequency_variables": [int(i) for i in np.argsort(-freqs)[: min(12, dimension)]],
                    "top_frequency_values": [float(freqs[i]) for i in np.argsort(-freqs)[: min(12, dimension)]],
                    "proxy_max_frequency": float(np.max(proxy_freqs)) if proxy_freqs else np.nan,
                }
            )
            for j in range(dimension):
                freq_rows.append(
                    {
                        "score_column": score_col,
                        "top_m": int(top_m),
                        "variable": int(j),
                        "frequency": float(freqs[j]),
                        "is_active": int(j in active),
                        "is_endpoint": int(j in endpoints),
                        "proxy_for": proxy_groups.get(j, np.nan),
                    }
                )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows)
    detail = pd.DataFrame(freq_rows)
    summary.to_csv(out_dir / "probe_selection_advantage_summary.csv", index=False)
    detail.to_csv(out_dir / "probe_selection_frequency_detail.csv", index=False)
    print(summary.to_string(index=False))
    print(f"Wrote {out_dir / 'probe_selection_advantage_summary.csv'}")
    print(f"Wrote {out_dir / 'probe_selection_frequency_detail.csv'}")


if __name__ == "__main__":
    main()
