#!/usr/bin/env python3
"""Seed-aligned stage records for controlled KAN structure claims.

This script is intentionally small-table oriented.  Each output row trains one
full-dimensional pyKAN model for one setting/seed and then records the stages
that a reviewer would want to inspect on the *same trained workflow*:

  prediction -> full-model pair reliance -> exposed-readout endpoints
  -> selected-support refit -> pruning/symbolic provenance.

The goal is not a new estimator.  The goal is to make the stage-discordance
claim auditable at seed level instead of only through aggregate phase diagrams.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import itertools
import json
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import make_synthetic
from experiments.run_kan_native_innovation_loop import (
    combine_scores,
    safe_edge_path_scores,
    safe_feature_score,
    top_vars,
)
from experiments.run_tuned_kan_recovery import (
    anova_pair_scores,
    batch_predict,
    canonical_pairs,
    mse_np,
    train_kan,
)
from scripts.run_full_kan_pair_anova_probe import all_pair_anova_pair_scores


Pair = tuple[int, int]


@dataclass(frozen=True)
class Setting:
    label: str
    function: str
    samples: int
    dimension: int
    noise: float
    update_grid: bool
    width_hidden: int
    steps: int
    seeds: tuple[int, ...]
    anova_points: int
    anova_background: int


DEFAULT_SETTINGS = [
    "clean_w16_n512|core_interaction_c025|512|100|0|0|16|75|0,1,2,3,4,5|24|24",
    "clean_w16_n1024|core_interaction_c025|1024|100|0|0|16|75|0,1,2,3|24|24",
    "gridupdate_w16_n512|core_interaction_c025|512|100|0|1|16|75|0,1,2,3|24|24",
    "gridupdate_w16_n1024|core_interaction_c025|1024|100|0|1|16|75|0,1,2,3|24|24",
    "noise010_w16_n512|core_interaction_c025|512|100|0.10|0|16|75|0,1,2,3|24|24",
    "clean_w32_n768|core_interaction_c025|768|100|0|0|32|75|0,1,2,3|24|24",
]


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


def parse_setting(spec: str) -> Setting:
    fields = spec.split("|")
    if len(fields) != 11:
        raise ValueError(
            "setting must be label|function|samples|dimension|noise|update_grid|"
            "width|steps|seeds|anova_points|anova_background"
        )
    label, function, samples, dimension, noise, update_grid, width, steps, seeds, points, background = fields
    return Setting(
        label=label,
        function=function,
        samples=int(samples),
        dimension=int(dimension),
        noise=float(noise),
        update_grid=bool(int(update_grid)),
        width_hidden=int(width),
        steps=int(steps),
        seeds=tuple(parse_seeds(seeds)),
        anova_points=int(points),
        anova_background=int(background),
    )


def train_args(args: argparse.Namespace, setting: Setting, *, input_width: int | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        width_hidden=int(setting.width_hidden if input_width is None else args.refit_width_hidden),
        grid=int(args.grid),
        k=int(args.k),
        steps=int(setting.steps if input_width is None else args.refit_steps),
        lamb=float(args.lamb),
        opt=args.opt,
        update_grid=bool(setting.update_grid),
        grid_update_num=int(args.grid_update_num),
        batch=int(args.batch),
    )


def rank_desc(scores: np.ndarray, idx: int) -> int:
    order = sorted(range(len(scores)), key=lambda j: (-float(scores[j]), int(j)))
    return int(order.index(int(idx)) + 1)


def score_margin(scores: np.ndarray, positives: Iterable[int], exclude: Iterable[int]) -> float:
    scores = np.asarray(scores, dtype=float).reshape(-1)
    positives = sorted({int(v) for v in positives})
    exclude = set(int(v) for v in exclude)
    false = [j for j in range(len(scores)) if j not in exclude]
    if not positives or not false:
        return np.nan
    return float(np.min(scores[positives]) - np.max(scores[false]))


def json_list(values: Sequence[int]) -> str:
    return json.dumps([int(v) for v in values])


def canonical_pair(pair: Sequence[int]) -> Pair:
    i, j = int(pair[0]), int(pair[1])
    return (i, j) if i < j else (j, i)


def ranked_pair_stats(pair_scores: dict[Pair, float], true_pair: Pair | None) -> dict[str, object]:
    ranked = sorted(pair_scores.items(), key=lambda kv: (-float(kv[1]), int(kv[0][0]), int(kv[0][1])))
    if true_pair is None or true_pair not in pair_scores:
        return {
            "rank": -1,
            "score": np.nan,
            "max_false": np.nan,
            "margin": np.nan,
            "top_pair": "NA",
            "top_score": np.nan,
            "beats_false": 0,
        }
    rank_lookup = {pair: idx + 1 for idx, (pair, _) in enumerate(ranked)}
    true_score = float(pair_scores[true_pair])
    false_scores = [float(score) for pair, score in ranked if pair != true_pair]
    max_false = float(np.max(false_scores)) if false_scores else np.nan
    top_pair, top_score = ranked[0]
    margin = true_score - max_false if np.isfinite(max_false) else np.nan
    return {
        "rank": int(rank_lookup[true_pair]),
        "score": true_score,
        "max_false": max_false,
        "margin": float(margin),
        "top_pair": str(top_pair),
        "top_score": float(top_score),
        "beats_false": int(rank_lookup[true_pair] == 1 and margin > 0),
    }


def call_quietly(fn, *args, **kwargs):
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
        return fn(*args, **kwargs), buffer.getvalue()


def tensor_to_int_list(value) -> list[int]:
    if value is None:
        return []
    try:
        arr = value.detach().cpu().numpy().reshape(-1)
    except Exception:
        arr = np.asarray(value).reshape(-1)
    return [int(x) for x in arr.tolist()]


def symbolic_smoke(model) -> tuple[int, str]:
    try:
        formula, _ = call_quietly(model.symbolic_formula)
        return 1, str(formula)[:500]
    except Exception as exc:
        return 0, f"{type(exc).__name__}: {exc}"[:500]


def prune_stage(model, args: argparse.Namespace) -> dict[str, object]:
    row: dict[str, object] = {
        "prune_workflow": args.prune_workflow,
        "prune_threshold": float(args.prune_threshold),
        "prune_support_size": -1,
        "prune_selected_inputs": "[]",
        "prune_endpoint_contains": 0,
        "symbolic_formula_ok": 0,
        "symbolic_formula_text": "",
        "prune_error": "",
    }
    try:
        if args.prune_workflow == "prune_input":
            pruned, _ = call_quietly(model.prune_input, threshold=float(args.prune_threshold), log_history=False)
        elif args.prune_workflow == "prune":
            pruned, _ = call_quietly(
                model.prune,
                node_th=float(args.node_threshold),
                edge_th=float(args.prune_threshold),
            )
        else:
            raise ValueError(f"unknown prune workflow={args.prune_workflow}")
        selected = tensor_to_int_list(getattr(pruned, "input_id", None))
        row["prune_support_size"] = int(len(selected))
        row["prune_selected_inputs"] = json_list(selected)
        if args.symbolic_smoke:
            ok, text = symbolic_smoke(pruned)
            row["symbolic_formula_ok"] = int(ok)
            row["symbolic_formula_text"] = text
    except Exception as exc:
        row["prune_error"] = f"{type(exc).__name__}: {exc}"
    return row


def selected_refit_stage(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    selected: list[int],
    true_pair: Pair | None,
    setting: Setting,
    args: argparse.Namespace,
    seed: int,
) -> dict[str, object]:
    out: dict[str, object] = {
        "refit_test_mse": np.nan,
        "refit_pair_rank": -1,
        "refit_pair_margin": np.nan,
        "refit_pair_beats_false": 0,
        "refit_error": "",
    }
    if true_pair is None or any(v not in selected for v in true_pair):
        out["refit_error"] = "true_pair_not_in_selected_support"
        return out
    local_map = {int(v): idx for idx, v in enumerate(selected)}
    local_true_pair = canonical_pair((local_map[true_pair[0]], local_map[true_pair[1]]))
    try:
        Xtr = X_train[:, selected].astype(np.float32)
        Xte = X_test[:, selected].astype(np.float32)
        refit_model = train_kan(
            Xtr,
            y_train,
            Xte,
            y_test,
            train_args(args, setting, input_width=len(selected)),
            seed=seed + int(args.refit_seed_offset),
            device=args.device,
        )
        pred = batch_predict(refit_model, Xte, device=args.device, batch_size=args.batch_size)
        out["refit_test_mse"] = mse_np(pred, y_test)
        pair_scores = anova_pair_scores(
            refit_model,
            Xte,
            device=args.device,
            points=args.refit_anova_points,
            background=args.refit_anova_background,
            batch_size=args.batch_size,
            score="abs",
        )
        stats = ranked_pair_stats(pair_scores, local_true_pair)
        out.update(
            {
                "refit_pair_rank": int(stats["rank"]),
                "refit_pair_margin": float(stats["margin"]),
                "refit_pair_beats_false": int(stats["beats_false"]),
            }
        )
    except Exception as exc:
        out["refit_error"] = f"{type(exc).__name__}: {exc}"
    return out


def first_broken_stage(row: dict[str, object], *, mse_threshold: float, top_m: int) -> str:
    if not np.isfinite(float(row.get("test_mse", np.nan))) or float(row["test_mse"]) > mse_threshold:
        return "prediction"
    if int(row.get("full_pair_rank", -1)) != 1 or float(row.get("full_pair_margin", np.nan)) <= 0:
        return "full-model reliance"
    if int(row.get("readout_worst_endpoint_rank", 10**9)) > top_m or float(row.get("readout_endpoint_margin", np.nan)) <= 0:
        return "readout endpoints"
    if int(row.get("refit_pair_rank", -1)) != 1 or float(row.get("refit_pair_margin", np.nan)) <= 0:
        return "support refit"
    if int(row.get("prune_endpoint_contains", 0)) != 1:
        return "pruning"
    return "none"


def run_one(args: argparse.Namespace, setting: Setting, seed: int) -> dict[str, object]:
    data = make_synthetic(
        function_name=setting.function,
        n_train=setting.samples,
        n_test=args.test_samples,
        d=setting.dimension,
        noise=setting.noise,
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
    true_vars = tuple(int(v) for v in gt.active_variables)
    true_pairs = canonical_pairs(gt.interactions)
    true_pair = true_pairs[0] if true_pairs else None
    endpoints = sorted({int(v) for pair in true_pairs for v in pair})

    t0 = time.time()
    model = train_kan(
        X_train,
        y_train,
        X_test,
        y_test,
        train_args(args, setting),
        seed=seed,
        device=args.device,
    )
    train_pred = batch_predict(model, X_train, device=args.device, batch_size=args.batch_size)
    test_pred = batch_predict(model, X_test, device=args.device, batch_size=args.batch_size)

    feature_scores = safe_feature_score(model, setting.dimension)
    edge_scores, _, endpoint_mass = safe_edge_path_scores(model, setting.dimension)
    hybrid_scores = combine_scores(feature_scores, edge_scores, endpoint_mass)
    selected = top_vars(hybrid_scores, args.top_m)

    endpoint_ranks = [rank_desc(hybrid_scores, v) for v in endpoints]
    full_pair_scores = all_pair_anova_pair_scores(
        model,
        X_test,
        device=args.device,
        points=setting.anova_points,
        background=setting.anova_background,
        batch_size=args.batch_size,
        pair_chunk_size=args.pair_chunk_size,
    )
    full_stats = ranked_pair_stats(full_pair_scores, true_pair)
    prune = prune_stage(model, args)
    prune_selected = json.loads(str(prune.get("prune_selected_inputs", "[]")))
    prune["prune_endpoint_contains"] = int(set(endpoints).issubset(set(prune_selected))) if endpoints else 0

    refit = selected_refit_stage(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        selected=selected,
        true_pair=true_pair,
        setting=setting,
        args=args,
        seed=seed,
    )

    row: dict[str, object] = {
        "setting": setting.label,
        "function": setting.function,
        "seed": int(seed),
        "samples": int(setting.samples),
        "dimension": int(setting.dimension),
        "noise": float(setting.noise),
        "update_grid": int(setting.update_grid),
        "width_hidden": int(setting.width_hidden),
        "steps": int(setting.steps),
        "grid": int(args.grid),
        "lamb": float(args.lamb),
        "true_variables": json_list(true_vars),
        "true_pair": str(true_pair),
        "train_mse": mse_np(train_pred, y_train),
        "test_mse": mse_np(test_pred, y_test),
        "full_pair_rank": int(full_stats["rank"]),
        "full_pair_margin": float(full_stats["margin"]),
        "full_pair_top_pair": str(full_stats["top_pair"]),
        "full_pair_beats_false": int(full_stats["beats_false"]),
        "readout_endpoint_ranks": json_list(endpoint_ranks),
        "readout_worst_endpoint_rank": int(max(endpoint_ranks)) if endpoint_ranks else -1,
        "readout_endpoint_margin": score_margin(hybrid_scores, endpoints, true_vars),
        "selected_support": json_list(selected),
        "selected_contains_all_true_vars": int(set(true_vars).issubset(set(selected))) if true_vars else 0,
        "selected_contains_endpoints": int(set(endpoints).issubset(set(selected))) if endpoints else 0,
        "runtime_sec": float(time.time() - t0),
    }
    row.update(refit)
    row.update(prune)
    row["first_broken_stage"] = first_broken_stage(row, mse_threshold=args.mse_threshold, top_m=args.top_m)
    return row


def load_existing(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def compact_table(detail: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    if detail.empty:
        return detail
    df = detail.copy()
    if "first_broken_stage" not in df.columns:
        return df.head(max_rows)
    df = df[df["first_broken_stage"].notna()].copy()
    if df.empty:
        return detail.head(max_rows)
    priority = {
        "none": 0,
        "full-model reliance": 1,
        "readout endpoints": 2,
        "support refit": 3,
        "pruning": 4,
        "prediction": 5,
    }
    df["_priority"] = df["first_broken_stage"].map(priority).fillna(9)
    # Keep diversity across settings and first-broken stages, then fill by
    # smallest seed for determinism.
    rows = []
    seen = set()
    for _, row in df.sort_values(["_priority", "setting", "seed"]).iterrows():
        key = (row["setting"], row["first_broken_stage"])
        if key in seen:
            continue
        seen.add(key)
        rows.append(row)
        if len(rows) >= max_rows:
            break
    if len(rows) < max_rows:
        chosen = {(r["setting"], int(r["seed"])) for r in rows}
        for _, row in df.sort_values(["setting", "seed"]).iterrows():
            key = (row["setting"], int(row["seed"]))
            if key in chosen:
                continue
            rows.append(row)
            chosen.add(key)
            if len(rows) >= max_rows:
                break
    out = pd.DataFrame(rows).drop(columns=["_priority"], errors="ignore")
    return out


def dataframe_to_markdown(df: pd.DataFrame, index: bool = False) -> str:
    """Render a small dataframe as GitHub-style markdown without tabulate."""

    if df.empty:
        return ""
    table = df.reset_index() if index else df.copy()
    table = table.astype(str)
    headers = list(table.columns)
    rows = table.values.tolist()
    widths = [
        max(len(str(header)), *(len(str(row[i])) for row in rows))
        for i, header in enumerate(headers)
    ]

    def fmt_row(values: list[object]) -> str:
        cells = [str(value).ljust(widths[i]) for i, value in enumerate(values)]
        return "| " + " | ".join(cells) + " |"

    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    return "\n".join([fmt_row(headers), separator, *(fmt_row(row) for row in rows)])


def write_markdown(detail: pd.DataFrame, path: Path, max_rows: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = compact_table(detail, max_rows=max_rows)
    cols = [
        "setting",
        "seed",
        "test_mse",
        "full_pair_rank",
        "full_pair_margin",
        "readout_endpoint_ranks",
        "readout_endpoint_margin",
        "selected_support",
        "refit_pair_rank",
        "prune_support_size",
        "prune_endpoint_contains",
        "first_broken_stage",
    ]
    lines = [
        "# Seed-aligned stage records",
        "",
        "Each row trains one full-dimensional KAN and carries the same seed/model through full-model ANOVA, exposed readout, selected-support refit, and pruning/symbolic provenance.",
        "",
    ]
    if table.empty:
        lines.append("No completed rows.")
    else:
        for col in cols:
            if col not in table.columns:
                table[col] = np.nan
        show = table[cols].copy()
        for col in ["test_mse", "full_pair_margin", "readout_endpoint_margin"]:
            show[col] = pd.to_numeric(show[col], errors="coerce").map(lambda x: f"{x:.4g}")
        lines.append(dataframe_to_markdown(show, index=False))
        lines.extend(
            [
                "",
                "First-broken-stage counts:",
                "",
                (
                    dataframe_to_markdown(
                        detail["first_broken_stage"].value_counts().rename_axis("stage").reset_index(name="count"),
                        index=False,
                    )
                    if "first_broken_stage" in detail.columns
                    else "No completed first-broken-stage rows yet."
                ),
            ]
        )
    path.write_text("\n".join(lines) + "\n")
    print(f"wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", nargs="+", default=DEFAULT_SETTINGS)
    parser.add_argument("--test-samples", type=int, default=2048)
    parser.add_argument("--nuisance-correlation", type=float, default=0.0)
    parser.add_argument("--n-correlated-proxies", type=int, default=0)
    parser.add_argument("--top-m", type=int, default=4)
    parser.add_argument("--grid", type=int, default=5)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--lamb", type=float, default=1e-3)
    parser.add_argument("--opt", default="LBFGS")
    parser.add_argument("--grid-update-num", type=int, default=5)
    parser.add_argument("--batch", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--pair-chunk-size", type=int, default=1000)
    parser.add_argument("--refit-width-hidden", type=int, default=16)
    parser.add_argument("--refit-steps", type=int, default=75)
    parser.add_argument("--refit-seed-offset", type=int, default=10000)
    parser.add_argument("--refit-anova-points", type=int, default=32)
    parser.add_argument("--refit-anova-background", type=int, default=32)
    parser.add_argument("--prune-workflow", choices=["prune_input", "prune"], default="prune_input")
    parser.add_argument("--prune-threshold", type=float, default=0.03)
    parser.add_argument("--node-threshold", type=float, default=0.01)
    parser.add_argument("--symbolic-smoke", action="store_true")
    parser.add_argument("--mse-threshold", type=float, default=0.05)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", type=Path, default=Path("results/revision/seed_aligned_stage_records"))
    parser.add_argument("--max-table-rows", type=int, default=8)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    settings = [parse_setting(spec) for spec in args.settings]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    detail_path = args.out_dir / "seed_aligned_stage_records_detail.csv"
    compact_path = args.out_dir / "seed_aligned_stage_records_compact.csv"
    md_path = args.out_dir / "seed_aligned_stage_records.md"

    rows: list[dict[str, object]] = []
    completed: set[tuple[str, int]] = set()
    if detail_path.exists() and not args.no_resume:
        existing = load_existing(detail_path)
        if not existing.empty:
            rows = existing.to_dict("records")
            completed = {
                (str(row["setting"]), int(row["seed"]))
                for _, row in existing[["setting", "seed"]].dropna().iterrows()
            }
            print(f"[resume] loaded {len(existing)} rows from {detail_path}", flush=True)

    for setting in settings:
        for seed in setting.seeds:
            key = (setting.label, int(seed))
            if key in completed:
                print(f"[resume] skip {setting.label} seed={seed}", flush=True)
                continue
            print(f"[seed-stage] {setting.label} seed={seed}", flush=True)
            try:
                rows.append(run_one(args, setting, int(seed)))
            except Exception as exc:
                rows.append(
                    {
                        "setting": setting.label,
                        "function": setting.function,
                        "seed": int(seed),
                        "status": "failed",
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
            detail = pd.DataFrame(rows)
            detail.to_csv(detail_path, index=False)
            compact_table(detail, args.max_table_rows).to_csv(compact_path, index=False)
            write_markdown(detail, md_path, max_rows=args.max_table_rows)

    detail = pd.DataFrame(rows)
    detail.to_csv(detail_path, index=False)
    compact_table(detail, args.max_table_rows).to_csv(compact_path, index=False)
    write_markdown(detail, md_path, max_rows=args.max_table_rows)
    print(compact_table(detail, args.max_table_rows).to_string(index=False))


if __name__ == "__main__":
    main()
