#!/usr/bin/env python3
"""EPIM-propose / ANOVA-verify interaction audit for full-dimensional KANs.

This script is a method-shaped follow-up to the stage-record audit.  It treats
edge-path interaction mass (EPIM) as a KAN-native proposal score for candidate
pairs, then verifies only a small candidate/control set with functional ANOVA on
the fitted full KAN.  In the implementation, EPIM(i,j) is the normalized
downstream-weighted co-path mass

    sum_h (|a_hi| |b_h|) (|a_hj| |b_h|),

where a_hi is the input-to-hidden edge scale and b_h is the hidden-to-output
scale exposed by pyKAN attribution.  The goal is to test whether a cheap
architecture-native pair proposal can reduce the cost of all-pairs ANOVA while
preserving the true interaction under nuisance features.
"""

from __future__ import annotations

import argparse
import itertools
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import make_synthetic
from experiments.run_kan_native_innovation_loop import safe_edge_path_scores, safe_feature_score
from experiments.run_tuned_kan_recovery import batch_predict, canonical_pairs, mse_np, train_kan

Pair = tuple[int, int]


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


def rank_pair(pair_scores: dict[Pair, float], pair: Pair) -> int:
    ranked = sorted(pair_scores.items(), key=lambda kv: (-float(kv[1]), kv[0][0], kv[0][1]))
    lookup = {p: idx + 1 for idx, (p, _) in enumerate(ranked)}
    return int(lookup.get(pair, len(ranked) + 1))


def top_pairs(pair_scores: dict[Pair, float], q: int) -> list[Pair]:
    ranked = sorted(pair_scores.items(), key=lambda kv: (-float(kv[1]), kv[0][0], kv[0][1]))
    return [p for p, _ in ranked[: max(0, int(q))]]


def top_vars(scores: np.ndarray, k: int) -> list[int]:
    order = sorted(range(len(scores)), key=lambda j: (-float(scores[j]), int(j)))
    return [int(v) for v in order[: max(0, int(k))]]


def random_control_pairs(
    *,
    d: int,
    exclude: Iterable[Pair],
    count: int,
    seed: int,
) -> list[Pair]:
    exclude_set = {tuple(sorted((int(i), int(j)))) for i, j in exclude}
    all_pairs = [(i, j) for i, j in itertools.combinations(range(d), 2) if (i, j) not in exclude_set]
    rng = random.Random(int(seed))
    rng.shuffle(all_pairs)
    return all_pairs[: min(int(count), len(all_pairs))]


def batched_candidate_anova_pair_scores(
    model,
    X_np: np.ndarray,
    pairs: Sequence[Pair],
    device: str,
    *,
    points: int,
    background: int,
    batch_size: int,
    pair_chunk_size: int,
) -> dict[Pair, float]:
    """Functional-ANOVA pair scores over a small candidate set.

    This is the candidate-set counterpart to the vectorized all-pairs scorer:
    for each anchor row, main effects are computed once for all variables that
    appear in the candidate set, then pair interventions are batched in chunks.
    """

    pairs = sorted({(int(i), int(j)) if int(i) < int(j) else (int(j), int(i)) for i, j in pairs})
    if not pairs:
        return {}

    n_points = min(int(points), len(X_np))
    n_bg = min(int(background), len(X_np))
    anchors = X_np[:n_points].copy()
    bg = X_np[:n_bg].copy()
    d = int(bg.shape[1])
    pair_arr = np.asarray(pairs, dtype=int)
    used_vars = sorted(set(pair_arr.reshape(-1).tolist()))
    var_to_pos = {int(v): idx for idx, v in enumerate(used_vars)}

    f_mean = float(np.mean(batch_predict(model, bg, device=device, batch_size=batch_size).reshape(-1)))
    accum = np.zeros(len(pairs), dtype=np.float64)
    pair_chunk_size = int(pair_chunk_size)
    if pair_chunk_size <= 0:
        pair_chunk_size = len(pairs)

    for row in anchors:
        main_rows_3d = np.broadcast_to(bg[None, :, :], (len(used_vars), n_bg, d)).copy()
        for pos, var in enumerate(used_vars):
            main_rows_3d[pos, :, var] = row[var]
        main_rows = main_rows_3d.reshape(len(used_vars) * n_bg, d)
        main_pred = batch_predict(model, main_rows, device=device, batch_size=batch_size).reshape(len(used_vars), n_bg)
        main_mean = np.mean(main_pred, axis=1)

        for start_pair in range(0, len(pairs), pair_chunk_size):
            stop_pair = min(start_pair + pair_chunk_size, len(pairs))
            chunk_arr = pair_arr[start_pair:stop_pair]
            chunk_len = len(chunk_arr)
            pair_rows_3d = np.broadcast_to(bg[None, :, :], (chunk_len, n_bg, d)).copy()
            chunk_idx = np.arange(chunk_len)
            pair_rows_3d[chunk_idx, :, chunk_arr[:, 0]] = row[chunk_arr[:, 0], None]
            pair_rows_3d[chunk_idx, :, chunk_arr[:, 1]] = row[chunk_arr[:, 1], None]
            pair_rows = pair_rows_3d.reshape(chunk_len * n_bg, d)
            pair_pred = batch_predict(model, pair_rows, device=device, batch_size=batch_size).reshape(chunk_len, n_bg)
            pair_mean = np.mean(pair_pred, axis=1)

            left = np.asarray([var_to_pos[int(v)] for v in chunk_arr[:, 0]], dtype=int)
            right = np.asarray([var_to_pos[int(v)] for v in chunk_arr[:, 1]], dtype=int)
            comps = pair_mean - main_mean[left] - main_mean[right] + f_mean
            accum[start_pair:stop_pair] += np.abs(comps)

    scores = accum / float(max(n_points, 1))
    return {pair: float(score) for pair, score in zip(pairs, scores)}


def summarize(detail: pd.DataFrame) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame()
    numeric = [
        "train_mse",
        "test_mse",
        "epim_true_pair_rank",
        "epim_proposal_contains_true_pair",
        "epim_endpoint_contains_true_pair",
        "candidate_pairs_scored",
        "random_controls_scored",
        "verified_true_pair_rank",
        "verified_top_is_true_pair",
        "practical_verified_top_is_true_pair",
        "verified_true_beats_candidate_false",
        "practical_verified_true_beats_candidate_false",
        "verified_true_minus_max_candidate_false",
        "verified_true_minus_max_random_control",
        "runtime_sec",
    ]
    for col in numeric:
        if col in detail.columns:
            detail[col] = pd.to_numeric(detail[col], errors="coerce")
    group_cols = [
        "function",
        "samples",
        "dimension",
        "noise",
        "update_grid",
        "width_hidden",
        "proposal_q",
        "random_controls",
    ]
    agg = {col: ["mean", "std"] for col in numeric if col in detail.columns}
    out = detail.groupby(group_cols, dropna=False).agg(agg).reset_index()
    out.columns = ["_".join(str(v) for v in col if v != "").rstrip("_") for col in out.columns]
    counts = detail.groupby(group_cols, dropna=False).size().reset_index(name="num_runs")
    return out.merge(counts, on=group_cols, how="left")


def parse_int_list(spec: str | None, default: Sequence[int]) -> list[int]:
    if spec is None or str(spec).strip() == "":
        return sorted(dict.fromkeys(int(v) for v in default))
    out: list[int] = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return sorted(dict.fromkeys(out))


def restrict_scores(scores: dict[Pair, float], pairs: Iterable[Pair]) -> dict[Pair, float]:
    pair_set = {(int(i), int(j)) if int(i) < int(j) else (int(j), int(i)) for i, j in pairs}
    return {pair: float(score) for pair, score in scores.items() if pair in pair_set}


def q_specific_row(
    *,
    base: dict,
    q: int,
    proposals: list[Pair],
    controls: list[Pair],
    verified_all: dict[Pair, float],
    true_pair: Pair | None,
    true_endpoints: set[int],
    epim_pair_scores: dict[Pair, float],
    epim_endpoint_mass: np.ndarray,
    args: argparse.Namespace,
    runtime_sec: float,
) -> dict:
    q_proposals = proposals[: int(q)]
    q_pair_set = set(q_proposals) | set(controls)
    if args.score_true_pair_for_diagnostics and true_pair is not None:
        q_pair_set = q_pair_set | {true_pair}
    verified = restrict_scores(verified_all, q_pair_set)
    ranked_verified = sorted(verified.items(), key=lambda kv: (-float(kv[1]), kv[0][0], kv[0][1]))
    top_verified_pair = ranked_verified[0][0] if ranked_verified else None

    proposal_set = set(q_proposals)
    random_set = set(controls)
    candidate_false_scores = [
        float(score) for pair, score in verified.items() if pair != true_pair and pair in proposal_set
    ]
    random_scores = [float(score) for pair, score in verified.items() if pair in random_set]
    true_score = float(verified.get(true_pair, np.nan)) if true_pair is not None else np.nan
    max_candidate_false = float(np.max(candidate_false_scores)) if candidate_false_scores else np.nan
    max_random = float(np.max(random_scores)) if random_scores else np.nan
    endpoint_top = set(top_vars(epim_endpoint_mass, args.endpoint_top_m))

    row = dict(base)
    row.update(
        {
            "proposal_q": int(q),
            "max_proposal_q_scored": int(max(args.proposal_q_values)),
            "epim_top_endpoint_variables": str(top_vars(epim_endpoint_mass, min(12, args.dimension))),
            "epim_top_pairs": str(q_proposals[: min(12, len(q_proposals))]),
            "candidate_pairs_scored": len(q_pair_set),
            "random_controls_scored": len(controls),
            "verified_top_pair": str(top_verified_pair),
            "verified_top_score": float(ranked_verified[0][1]) if ranked_verified else np.nan,
            "verified_true_pair_score": true_score,
            "verified_max_candidate_false_score": max_candidate_false,
            "verified_max_random_control_score": max_random,
            "verified_true_pair_rank": rank_pair(verified, true_pair) if true_pair is not None else np.nan,
            "verified_top_is_true_pair": int(top_verified_pair == true_pair) if true_pair is not None else np.nan,
            "practical_verified_top_is_true_pair": (
                int((true_pair in proposal_set) and (top_verified_pair == true_pair))
                if true_pair is not None
                else np.nan
            ),
            "verified_true_beats_candidate_false": (
                int(true_score > max_candidate_false)
                if np.isfinite(true_score) and np.isfinite(max_candidate_false)
                else np.nan
            ),
            "practical_verified_true_beats_candidate_false": (
                int((true_pair in proposal_set) and (true_score > max_candidate_false))
                if np.isfinite(true_score) and np.isfinite(max_candidate_false) and true_pair is not None
                else np.nan
            ),
            "verified_true_minus_max_candidate_false": (
                true_score - max_candidate_false
                if np.isfinite(true_score) and np.isfinite(max_candidate_false)
                else np.nan
            ),
            "verified_true_minus_max_random_control": (
                true_score - max_random
                if np.isfinite(true_score) and np.isfinite(max_random)
                else np.nan
            ),
            "epim_true_pair_rank": rank_pair(epim_pair_scores, true_pair) if true_pair is not None else np.nan,
            "epim_proposal_contains_true_pair": (
                int(true_pair in proposal_set) if true_pair is not None else np.nan
            ),
            "epim_endpoint_contains_true_pair": (
                int(true_endpoints.issubset(endpoint_top)) if true_endpoints else np.nan
            ),
            "runtime_sec": float(runtime_sec),
        }
    )
    return row


def run_one(args: argparse.Namespace, seed: int) -> list[dict]:
    t0 = time.time()
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
    true_pairs = canonical_pairs(data["ground_truth"].interactions)
    true_pair = true_pairs[0] if true_pairs else None
    true_endpoints = set(true_pair) if true_pair is not None else set()

    base = {
        "function": args.function,
        "seed": int(seed),
        "samples": int(args.samples),
        "test_samples": int(args.test_samples),
        "dimension": int(args.dimension),
        "noise": float(args.noise),
        "nuisance_correlation": float(args.nuisance_correlation),
        "n_correlated_proxies": int(args.n_correlated_proxies),
        "update_grid": int(bool(args.update_grid)),
        "width_hidden": int(args.width_hidden),
        "grid": int(args.grid),
        "k": int(args.k),
        "lamb": float(args.lamb),
        "steps": int(args.steps),
        "proposal_q": np.nan,
        "random_controls": int(args.random_controls),
        "true_pair": str(true_pair),
        "status": "ok",
        "error": "",
    }

    try:
        model = train_kan(
            X_train,
            y_train,
            X_test,
            y_test,
            train_args(args),
            seed=seed,
            device=args.device,
        )
        train_pred = batch_predict(model, X_train, device=args.device, batch_size=args.batch_size)
        test_pred = batch_predict(model, X_test, device=args.device, batch_size=args.batch_size)

        feature_scores = safe_feature_score(model, args.dimension)
        _, epim_pair_scores, epim_endpoint_mass = safe_edge_path_scores(model, args.dimension)
        proposals = top_pairs(epim_pair_scores, max(args.proposal_q_values))
        controls = random_control_pairs(
            d=args.dimension,
            exclude=set(proposals) | set(true_pairs),
            count=args.random_controls,
            seed=10_000 + int(seed),
        )
        scored_pairs = sorted(set(proposals) | set(controls))
        if args.score_true_pair_for_diagnostics and true_pair is not None:
            scored_pairs = sorted(set(scored_pairs) | {true_pair})

        verified = batched_candidate_anova_pair_scores(
            model,
            X_test,
            scored_pairs,
            args.device,
            points=args.anova_points,
            background=args.anova_background,
            batch_size=args.batch_size,
            pair_chunk_size=args.pair_chunk_size,
        )
        base.update(
            {
                "train_mse": mse_np(train_pred, y_train),
                "test_mse": mse_np(test_pred, y_test),
                "top_feature_variables": str(top_vars(feature_scores, min(12, args.dimension))),
            }
        )
        runtime_sec = float(time.time() - t0)
        return [
            q_specific_row(
                base=base,
                q=q,
                proposals=proposals,
                controls=controls,
                verified_all=verified,
                true_pair=true_pair,
                true_endpoints=true_endpoints,
                epim_pair_scores=epim_pair_scores,
                epim_endpoint_mass=epim_endpoint_mass,
                args=args,
                runtime_sec=runtime_sec,
            )
            for q in args.proposal_q_values
        ]
    except Exception as exc:
        base.update({"status": "failed", "error": repr(exc), "runtime_sec": float(time.time() - t0)})
        return [{**base, "proposal_q": int(q), "max_proposal_q_scored": int(max(args.proposal_q_values))} for q in args.proposal_q_values]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--function", default="core_interaction_c025")
    parser.add_argument("--samples", type=int, default=1024)
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
    parser.add_argument("--batch", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--proposal-q", type=int, default=32)
    parser.add_argument(
        "--proposal-qs",
        default="",
        help="Comma-separated proposal budgets to emit from one fitted model; overrides --proposal-q.",
    )
    parser.add_argument("--random-controls", type=int, default=128)
    parser.add_argument("--endpoint-top-m", type=int, default=4)
    parser.add_argument("--anova-points", type=int, default=24)
    parser.add_argument("--anova-background", type=int, default=24)
    parser.add_argument("--pair-chunk-size", type=int, default=256)
    parser.add_argument("--score-true-pair-for-diagnostics", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-dir", type=Path, default=Path("results/revision/epim_pairverify"))
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    args.proposal_q_values = parse_int_list(args.proposal_qs, default=[args.proposal_q])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    detail_path = args.out_dir / "epim_pairverify_detail.csv"
    summary_path = args.out_dir / "epim_pairverify_summary.csv"

    rows: list[dict] = []
    completed: set[tuple[int, int]] = set()
    if detail_path.exists() and not args.no_resume:
        existing = pd.read_csv(detail_path)
        if len(existing):
            rows = existing.to_dict("records")
            completed = {
                (int(row["seed"]), int(row["proposal_q"]))
                for _, row in existing.dropna(subset=["seed", "proposal_q"]).iterrows()
            }
            print(f"[resume] loaded {len(existing)} rows from {detail_path}", flush=True)

    for seed in parse_seeds(args.seeds):
        needed = [(int(seed), int(q)) for q in args.proposal_q_values]
        if all(item in completed for item in needed):
            print(f"[resume] skip seed={seed}", flush=True)
            continue
        print(f"Running EPIM PairVerify seed={seed}", flush=True)
        new_rows = [row for row in run_one(args, int(seed)) if (int(row["seed"]), int(row["proposal_q"])) not in completed]
        rows.extend(new_rows)
        detail = pd.DataFrame(rows)
        detail.to_csv(detail_path, index=False)
        summarize(detail).to_csv(summary_path, index=False)

    detail = pd.DataFrame(rows)
    summary = summarize(detail)
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
