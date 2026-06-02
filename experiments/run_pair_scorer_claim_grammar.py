#!/usr/bin/env python3
"""Compare scorer-indexed pair claims on the same fitted KAN.

ClaimTransfer-Bench treats a pair-recovery statement as a typed claim:

    task card + workflow adapter + evidence object + scorer + predicate.

This runner makes that design testable.  For each seed it trains one
full-dimensional KAN, then ranks the declared interaction pairs with several
pair scorers on the same fitted function/readout:

* ``epim``: pyKAN edge-path interaction mass, used as a proposal/readout score.
* ``anova_abs``: functional-ANOVA pair score on the fitted KAN.
* ``fd``: finite-difference mixed-effect score on the fitted KAN.
* ``hessian``: autograd Hessian magnitude on the fitted KAN.
* ``hybrid_epim_anova``: average of max-normalized EPIM and ANOVA scores.

The output is deliberately scorer-indexed.  A disagreement is not an error; it
is the benchmark signal that "the interaction was recovered" needs a declared
claim grammar and scorer.
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import make_synthetic
from experiments.run_kan_native_innovation_loop import safe_edge_path_scores
from experiments.run_tuned_kan_recovery import (
    Pair,
    batch_predict,
    canonical_pairs,
    evaluate_interaction_recovery,
    finite_difference_pair_scores,
    hessian_pair_scores,
    hybrid_pair_scores,
    mse_np,
    train_kan,
)
from scripts.run_full_kan_pair_anova_probe import all_pair_anova_pair_scores


def parse_seeds(spec: str) -> list[int]:
    out: list[int] = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return sorted(dict.fromkeys(out))


def parse_scorers(spec: str) -> list[str]:
    allowed = {"epim", "anova_abs", "fd", "hessian", "hybrid_epim_anova"}
    out = []
    for item in str(spec).split(","):
        item = item.strip()
        if not item:
            continue
        if item not in allowed:
            raise ValueError(f"Unknown scorer {item!r}; allowed={sorted(allowed)}")
        out.append(item)
    return sorted(dict.fromkeys(out), key=out.index)


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


def canonical_pair(pair: Iterable[int]) -> Pair:
    i, j = [int(v) for v in pair]
    return (i, j) if i < j else (j, i)


def true_pair_rank_payload(pair_scores: dict[Pair, float], true_pairs: tuple[Pair, ...]) -> dict:
    ranked = sorted(pair_scores.items(), key=lambda kv: (-float(kv[1]), kv[0][0], kv[0][1]))
    top_pair, top_score = ranked[0] if ranked else ((-1, -1), np.nan)
    true_ranks = {pair: rank_pair(pair_scores, pair) for pair in true_pairs}
    true_scores = {pair: float(pair_scores.get(pair, np.nan)) for pair in true_pairs}
    false_scores = [
        float(score) for pair, score in pair_scores.items() if canonical_pair(pair) not in set(true_pairs)
    ]
    max_false = float(np.max(false_scores)) if false_scores else np.nan
    min_true = float(np.nanmin(list(true_scores.values()))) if true_scores else np.nan
    return {
        "top_pair": str(top_pair),
        "top_pair_score": float(top_score),
        "true_pair_ranks": str({str(k): int(v) for k, v in true_ranks.items()}),
        "true_pair_scores": str({str(k): float(v) for k, v in true_scores.items()}),
        "true_pair_best_rank": float(min(true_ranks.values())) if true_ranks else np.nan,
        "true_pair_worst_rank": float(max(true_ranks.values())) if true_ranks else np.nan,
        "min_true_pair_score": min_true,
        "max_false_pair_score": max_false,
        "min_true_minus_max_false": (
            min_true - max_false if np.isfinite(min_true) and np.isfinite(max_false) else np.nan
        ),
        "all_true_pairs_rank1_to_q": (
            int(all(rank <= len(true_pairs) for rank in true_ranks.values())) if true_ranks else np.nan
        ),
    }


def scorer_claim_grammar(function_name: str, true_pairs: tuple[Pair, ...]) -> str:
    if function_name in {"formula_nested_trig", "formula_three_way_product"}:
        return "pairwise_stress_card"
    if len(true_pairs) > 1:
        return "multi_pair_declared"
    return "single_pair_declared"


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
    gt = data["ground_truth"]
    true_pairs = tuple(canonical_pair(pair) for pair in canonical_pairs(gt.interactions))
    all_pair_count = int(args.dimension * (args.dimension - 1) // 2)
    if args.max_all_pairs > 0 and all_pair_count > args.max_all_pairs:
        raise ValueError(
            f"Requested all-pairs scoring over {all_pair_count} pairs; "
            f"increase --max-all-pairs if intentional."
        )

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

    scorers = parse_scorers(args.scorers)
    score_maps: dict[str, dict[Pair, float]] = {}

    if "epim" in scorers or "hybrid_epim_anova" in scorers:
        _, epim_pairs, _ = safe_edge_path_scores(model, args.dimension)
        score_maps["epim"] = {canonical_pair(pair): float(score) for pair, score in epim_pairs.items()}

    if "anova_abs" in scorers or "hybrid_epim_anova" in scorers:
        score_maps["anova_abs"] = all_pair_anova_pair_scores(
            model,
            X_test,
            device=args.device,
            points=args.anova_points,
            background=args.anova_background,
            batch_size=args.batch_size,
            pair_chunk_size=args.pair_chunk_size,
        )

    if "fd" in scorers:
        score_maps["fd"] = finite_difference_pair_scores(
            model,
            X_test,
            device=args.device,
            points=args.fd_points,
            h=args.fd_h,
            batch_size=args.batch_size,
        )

    if "hessian" in scorers:
        score_maps["hessian"] = hessian_pair_scores(
            model,
            X_test,
            device=args.device,
            points=args.hessian_points,
        )

    if "hybrid_epim_anova" in scorers:
        score_maps["hybrid_epim_anova"] = hybrid_pair_scores(
            score_maps.get("epim", {}),
            score_maps.get("anova_abs", {}),
        )

    base = {
        "function": args.function,
        "formula": gt.formula,
        "claim_grammar": scorer_claim_grammar(args.function, true_pairs),
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
        "true_pairs": str(true_pairs),
        "num_true_pairs": int(len(true_pairs)),
        "train_mse": mse_np(train_pred, y_train),
        "test_mse": mse_np(test_pred, y_test),
        "all_pairs_scored": all_pair_count,
    }

    rows: list[dict] = []
    for scorer in scorers:
        scores = score_maps.get(scorer, {})
        row = dict(base)
        row["pair_scorer"] = scorer
        row["status"] = "ok"
        row["runtime_sec"] = float(time.time() - t0)
        row.update(evaluate_interaction_recovery(scores, true_pairs))
        row.update(true_pair_rank_payload(scores, true_pairs))
        rows.append(row)
    return rows


def summarize(detail: pd.DataFrame) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame()
    numeric = [
        "train_mse",
        "test_mse",
        "interaction_f1",
        "true_interaction_best_rank",
        "true_interaction_worst_rank",
        "true_interaction_mean_score_margin",
        "true_interaction_beats_all_false",
        "true_pair_best_rank",
        "true_pair_worst_rank",
        "min_true_minus_max_false",
        "all_true_pairs_rank1_to_q",
        "runtime_sec",
    ]
    for col in numeric:
        if col in detail.columns:
            detail[col] = pd.to_numeric(detail[col], errors="coerce")
    group_cols = [
        "function",
        "claim_grammar",
        "samples",
        "dimension",
        "noise",
        "update_grid",
        "width_hidden",
        "pair_scorer",
    ]
    agg = {col: ["mean", "std"] for col in numeric if col in detail.columns}
    out = detail.groupby(group_cols, dropna=False).agg(agg).reset_index()
    out.columns = ["_".join(str(v) for v in col if v != "").rstrip("_") for col in out.columns]
    counts = detail.groupby(group_cols, dropna=False).size().reset_index(name="num_runs")
    return out.merge(counts, on=group_cols, how="left")


def scorer_disagreement(detail: pd.DataFrame) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame()
    rows = []
    for keys, group in detail.groupby(["function", "seed"], dropna=False):
        top_pairs = {str(r["pair_scorer"]): str(r["top_pair"]) for _, r in group.iterrows()}
        rank1 = {
            str(r["pair_scorer"]): int(float(r.get("all_true_pairs_rank1_to_q", 0)) == 1)
            for _, r in group.iterrows()
        }
        rows.append(
            {
                "function": keys[0],
                "seed": int(keys[1]),
                "num_scorers": int(len(group)),
                "top_pair_by_scorer": str(top_pairs),
                "rank1_by_scorer": str(rank1),
                "num_distinct_top_pairs": int(len(set(top_pairs.values()))),
                "num_rank1_scorers": int(sum(rank1.values())),
                "scorer_rank1_disagreement": int(len(set(rank1.values())) > 1),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--function", default="formula_bilinear")
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--test-samples", type=int, default=2048)
    parser.add_argument("--dimension", type=int, default=100)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--nuisance-correlation", type=float, default=0.0)
    parser.add_argument("--n-correlated-proxies", type=int, default=0)
    parser.add_argument("--seeds", default="0-3")
    parser.add_argument("--width-hidden", type=int, default=16)
    parser.add_argument("--grid", type=int, default=5)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--steps", type=int, default=90)
    parser.add_argument("--lamb", type=float, default=1e-3)
    parser.add_argument("--opt", default="LBFGS")
    parser.add_argument("--update-grid", action="store_true")
    parser.add_argument("--grid-update-num", type=int, default=5)
    parser.add_argument("--batch", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--scorers", default="epim,anova_abs,fd,hybrid_epim_anova")
    parser.add_argument("--anova-points", type=int, default=16)
    parser.add_argument("--anova-background", type=int, default=16)
    parser.add_argument("--pair-chunk-size", type=int, default=512)
    parser.add_argument("--fd-points", type=int, default=16)
    parser.add_argument("--fd-h", type=float, default=1e-2)
    parser.add_argument("--hessian-points", type=int, default=4)
    parser.add_argument("--max-all-pairs", type=int, default=15000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-dir", type=Path, default=Path("results/revision/pair_scorer_claim_grammar"))
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    detail_path = args.out_dir / "pair_scorer_claim_grammar_detail.csv"
    summary_path = args.out_dir / "pair_scorer_claim_grammar_summary.csv"
    disagreement_path = args.out_dir / "pair_scorer_claim_grammar_disagreement.csv"

    rows: list[dict] = []
    completed: set[int] = set()
    if detail_path.exists() and not args.no_resume:
        existing = pd.read_csv(detail_path)
        if len(existing):
            rows = existing.to_dict("records")
            completed = {int(v) for v in existing["seed"].dropna().astype(int).tolist()}
            print(f"[resume] loaded {len(existing)} rows from {detail_path}", flush=True)

    for seed in parse_seeds(args.seeds):
        if int(seed) in completed:
            print(f"[resume] skip seed={seed}", flush=True)
            continue
        print(f"Running scorer grammar seed={seed}", flush=True)
        try:
            rows.extend(run_one(args, int(seed)))
        except Exception as exc:
            rows.append(
                {
                    "function": args.function,
                    "seed": int(seed),
                    "pair_scorer": "all",
                    "status": "failed",
                    "error": repr(exc),
                }
            )
        detail = pd.DataFrame(rows)
        detail.to_csv(detail_path, index=False)
        summarize(detail[detail.get("status", "ok").astype(str).eq("ok")]).to_csv(summary_path, index=False)
        scorer_disagreement(detail[detail.get("status", "ok").astype(str).eq("ok")]).to_csv(disagreement_path, index=False)

    detail = pd.DataFrame(rows)
    ok = detail[detail.get("status", "ok").astype(str).eq("ok")].copy()
    summary = summarize(ok)
    disagreement = scorer_disagreement(ok)
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    disagreement.to_csv(disagreement_path, index=False)
    print(summary.to_string(index=False))
    if len(disagreement):
        print(disagreement.to_string(index=False))


if __name__ == "__main__":
    main()
