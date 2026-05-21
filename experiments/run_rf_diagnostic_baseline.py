from __future__ import annotations

import argparse
import itertools
import sys
import traceback
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import average_precision_score, roc_auc_score

from src.data import make_synthetic


Pair = Tuple[int, int]


# ============================================================
# Metrics and utilities
# ============================================================

def mse_np(pred: np.ndarray, target: np.ndarray) -> float:
    pred = pred.reshape(-1, 1)
    target = target.reshape(-1, 1)
    return float(np.mean((pred - target) ** 2))


def f1_from_sets(pred: set, true: set) -> Tuple[float, float, float]:
    if len(true) == 0:
        return np.nan, np.nan, np.nan
    if len(pred) == 0:
        return 0.0, 0.0, 0.0

    tp = len(pred & true)
    precision = tp / len(pred)
    recall = tp / len(true)
    f1 = 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)
    return precision, recall, f1


def safe_auroc_auprc(y_true: np.ndarray, scores: np.ndarray) -> Tuple[float, float]:
    try:
        auroc = float(roc_auc_score(y_true, scores))
    except ValueError:
        auroc = np.nan

    try:
        auprc = float(average_precision_score(y_true, scores))
    except ValueError:
        auprc = np.nan

    return auroc, auprc


def normalize_function_alias(name: str) -> str:
    aliases = {
        "core_c5": "core_interaction_c5",
        "proxy": "correlated_proxy",
    }
    return aliases.get(name, name)


def canonical_pairs(pairs: Sequence[Tuple[int, int]]) -> Tuple[Pair, ...]:
    return tuple(tuple(sorted((int(i), int(j)))) for i, j in pairs)


def interaction_endpoints(true_interactions: Sequence[Pair]) -> Tuple[int, ...]:
    endpoints = set()
    for i, j in true_interactions:
        endpoints.add(int(i))
        endpoints.add(int(j))
    return tuple(sorted(endpoints))


def evaluate_variable_recovery(scores: np.ndarray, true_vars: Sequence[int]) -> Dict:
    true_set = set(int(v) for v in true_vars)

    if len(true_set) == 0:
        return {
            "selected_variables": [],
            "variable_precision": np.nan,
            "variable_recall": np.nan,
            "variable_f1": np.nan,
            "variable_auroc": np.nan,
            "variable_auprc": np.nan,
        }

    k = len(true_set)
    selected = set(int(i) for i in np.argsort(-scores)[:k])
    precision, recall, f1 = f1_from_sets(selected, true_set)

    y_true = np.array([1 if i in true_set else 0 for i in range(len(scores))], dtype=int)
    auroc, auprc = safe_auroc_auprc(y_true, scores)

    active_scores = scores[y_true == 1]
    inactive_scores = scores[y_true == 0]

    return {
        "selected_variables": sorted(selected),
        "variable_precision": precision,
        "variable_recall": recall,
        "variable_f1": f1,
        "variable_auroc": auroc,
        "variable_auprc": auprc,
        "active_score_mean": float(active_scores.mean()) if len(active_scores) else np.nan,
        "inactive_score_mean": float(inactive_scores.mean()) if len(inactive_scores) else np.nan,
        "active_score_min": float(active_scores.min()) if len(active_scores) else np.nan,
        "inactive_score_max": float(inactive_scores.max()) if len(inactive_scores) else np.nan,
    }


def evaluate_interaction_recovery(pair_scores: Dict[Pair, float], true_interactions: Sequence[Pair]) -> Dict:
    true_set = set(canonical_pairs(true_interactions))

    if len(true_set) == 0:
        return {
            "selected_interactions": [],
            "pair_scores_top50": [],
            "interaction_precision": np.nan,
            "interaction_recall": np.nan,
            "interaction_f1": np.nan,
            "true_interaction_score_mean": np.nan,
            "max_nontrue_interaction_score": np.nan,
        }

    k = len(true_set)
    ranked = sorted(pair_scores.items(), key=lambda kv: kv[1], reverse=True)
    selected = {pair for pair, _ in ranked[:k]}

    precision, recall, f1 = f1_from_sets(selected, true_set)

    true_scores = [float(pair_scores.get(pair, 0.0)) for pair in true_set]
    nontrue_scores = [float(v) for pair, v in pair_scores.items() if pair not in true_set]

    return {
        "selected_interactions": sorted(selected),
        "interaction_precision": precision,
        "interaction_recall": recall,
        "interaction_f1": f1,
        "true_interaction_score_mean": float(np.mean(true_scores)) if true_scores else np.nan,
        "max_nontrue_interaction_score": float(np.max(nontrue_scores)) if nontrue_scores else np.nan,
    }


def support_stats(scores: np.ndarray, top_m: int, true_vars: Sequence[int], true_interactions: Sequence[Pair]) -> Dict:
    true_var_set = set(int(v) for v in true_vars)
    endpoints = set(interaction_endpoints(true_interactions))
    selected_top_m = set(int(i) for i in np.argsort(-scores)[:top_m])

    return {
        "top_m": top_m,
        "rf_screen_selected_top_m": sorted(selected_top_m),
        "rf_screen_contains_all_true_vars": int(true_var_set.issubset(selected_top_m)) if true_var_set else np.nan,
        "rf_screen_true_var_recall": len(true_var_set & selected_top_m) / len(true_var_set) if true_var_set else np.nan,
        "rf_screen_contains_all_interaction_endpoints": int(endpoints.issubset(selected_top_m)) if endpoints else np.nan,
        "rf_screen_interaction_endpoint_recall": len(endpoints & selected_top_m) / len(endpoints) if endpoints else np.nan,
        "rf_screen_contains_true_interactions": int(
            all(int(i) in selected_top_m and int(j) in selected_top_m for i, j in true_interactions)
        ) if true_interactions else np.nan,
    }


# ============================================================
# RF predictor and interaction scores
# ============================================================

def train_rf_predictor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
    n_estimators: int,
    max_depth: int | None,
    min_samples_leaf: int,
) -> RandomForestRegressor:
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=seed,
        n_jobs=-1,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
    )
    rf.fit(X_train, y_train.reshape(-1))
    return rf


def batch_predict(predict_fn: Callable[[np.ndarray], np.ndarray], X: np.ndarray, batch_size: int) -> np.ndarray:
    if len(X) <= batch_size:
        return predict_fn(X).reshape(-1)

    outs = []
    for start in range(0, len(X), batch_size):
        outs.append(predict_fn(X[start:start + batch_size]).reshape(-1))
    return np.concatenate(outs, axis=0)


def pair_permutation_scores_predictor(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    X_np: np.ndarray,
    points: int,
    seed: int,
    batch_size: int,
    pair_chunk_size: int,
) -> Tuple[Dict[Pair, float], Dict[Pair, float]]:
    """Compute RF functional pair scores.

    joint score:
        E[(f(X with i,j jointly permuted) - f(X))^2]

    synergy score:
        | joint_delta(i,j) - single_delta(i) - single_delta(j) |

    The synergy score is used as the main RF interaction-recovery score because
    the joint score is often dominated by main effects.
    """
    d = X_np.shape[1]
    n = min(points, X_np.shape[0])
    X = X_np[:n].copy()
    rng = np.random.default_rng(seed + 9001)

    base = batch_predict(predict_fn, X, batch_size=batch_size)

    # Precompute one independent permutation for each feature.
    perms = {j: rng.permutation(n) for j in range(d)}

    # Single-feature deltas, batched over all features.
    single_delta = np.zeros(d, dtype=float)
    single_blocks = []
    single_features = []
    for j in range(d):
        X_j = X.copy()
        X_j[:, j] = X_j[perms[j], j]
        single_blocks.append(X_j)
        single_features.append(j)

    X_single_big = np.vstack(single_blocks)
    pred_single_big = batch_predict(predict_fn, X_single_big, batch_size=batch_size)
    pred_single_big = pred_single_big.reshape(len(single_blocks), n)

    for block_idx, j in enumerate(single_features):
        single_delta[j] = float(np.mean((pred_single_big[block_idx] - base) ** 2))

    # Joint pair deltas, batched over pair chunks.
    pairs = [(i, j) for i, j in itertools.combinations(range(d), 2)]
    joint_scores: Dict[Pair, float] = {}
    synergy_scores: Dict[Pair, float] = {}

    for start in range(0, len(pairs), pair_chunk_size):
        chunk = pairs[start:start + pair_chunk_size]
        joint_blocks = []

        for i, j in chunk:
            X_ij = X.copy()
            X_ij[:, i] = X_ij[perms[i], i]
            X_ij[:, j] = X_ij[perms[j], j]
            joint_blocks.append(X_ij)

        X_joint_big = np.vstack(joint_blocks)
        pred_joint_big = batch_predict(predict_fn, X_joint_big, batch_size=batch_size)
        pred_joint_big = pred_joint_big.reshape(len(chunk), n)

        for block_idx, (i, j) in enumerate(chunk):
            joint = float(np.mean((pred_joint_big[block_idx] - base) ** 2))
            synergy = abs(joint - single_delta[i] - single_delta[j])

            joint_scores[(i, j)] = joint
            synergy_scores[(i, j)] = float(synergy)

    return joint_scores, synergy_scores


# ============================================================
# Run one config
# ============================================================

def run_one(args, function_name: str, seed: int) -> List[Dict]:
    function_name = normalize_function_alias(function_name)

    data = make_synthetic(
        function_name=function_name,
        n_train=args.samples,
        n_test=args.test_samples,
        d=args.dimension,
        noise=args.noise,
        seed=seed,
        standardize_target=True,
    )

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]
    gt = data["ground_truth"]

    true_vars = tuple(int(v) for v in gt.active_variables)
    true_interactions = canonical_pairs(gt.interactions)

    base = {
        "model": "RF_PREDICTOR_DIAGNOSTIC",
        "function": function_name,
        "seed": seed,
        "samples": args.samples,
        "test_samples": args.test_samples,
        "dimension": args.dimension,
        "noise": args.noise,
        "rf_trees": args.rf_trees,
        "rf_max_depth": args.rf_max_depth,
        "rf_min_samples_leaf": args.rf_min_samples_leaf,
        "true_variables": list(true_vars),
        "true_interactions": list(true_interactions),
        "num_true_variables": len(true_vars),
        "num_true_interactions": len(true_interactions),
        "formula": gt.formula,
    }

    try:
        rf = train_rf_predictor(
            X_train=X_train,
            y_train=y_train,
            seed=seed,
            n_estimators=args.rf_trees,
            max_depth=args.rf_max_depth,
            min_samples_leaf=args.rf_min_samples_leaf,
        )

        train_pred = rf.predict(X_train).reshape(-1, 1)
        test_pred = rf.predict(X_test).reshape(-1, 1)
        train_mse = mse_np(train_pred, y_train)
        test_mse = mse_np(test_pred, y_test)

        var_scores = np.asarray(rf.feature_importances_, dtype=float)

        rows: List[Dict] = []

        # Variable recovery / RF-screen support quality row.
        row_var = dict(base)
        row_var.update({
            "status": "ok",
            "error": "",
            "traceback": "",
            "metric": "rf_feature_importance",
            "train_mse": train_mse,
            "test_mse": test_mse,
            "importance_scores": var_scores.tolist(),
            "pair_scores_top50": [],
        })
        row_var.update(evaluate_variable_recovery(var_scores, true_vars))
        row_var.update(support_stats(var_scores, args.top_m, true_vars, true_interactions))
        row_var.update(evaluate_interaction_recovery({}, true_interactions))
        rows.append(row_var)

        # RF predictor interaction scores.
        if len(true_interactions) > 0 and args.compute_pair_permutation:
            predict_fn = lambda X: rf.predict(X)
            joint_scores, synergy_scores = pair_permutation_scores_predictor(
                predict_fn=predict_fn,
                X_np=X_test,
                points=args.perm_points,
                seed=seed,
                batch_size=args.batch_size,
                pair_chunk_size=args.pair_chunk_size,
            )

            row_joint = dict(base)
            row_joint.update({
                "status": "ok",
                "error": "",
                "traceback": "",
                "metric": "rf_pair_permutation_joint",
                "train_mse": train_mse,
                "test_mse": test_mse,
                "importance_scores": var_scores.tolist(),
                "pair_scores_top50": sorted(
                    [(int(i), int(j), float(v)) for (i, j), v in joint_scores.items()],
                    key=lambda x: x[2],
                    reverse=True,
                )[:50],
            })
            row_joint.update(evaluate_variable_recovery(var_scores, true_vars))
            row_joint.update(support_stats(var_scores, args.top_m, true_vars, true_interactions))
            row_joint.update(evaluate_interaction_recovery(joint_scores, true_interactions))
            rows.append(row_joint)

            row_syn = dict(base)
            row_syn.update({
                "status": "ok",
                "error": "",
                "traceback": "",
                "metric": "rf_pair_permutation_synergy",
                "train_mse": train_mse,
                "test_mse": test_mse,
                "importance_scores": var_scores.tolist(),
                "pair_scores_top50": sorted(
                    [(int(i), int(j), float(v)) for (i, j), v in synergy_scores.items()],
                    key=lambda x: x[2],
                    reverse=True,
                )[:50],
            })
            row_syn.update(evaluate_variable_recovery(var_scores, true_vars))
            row_syn.update(support_stats(var_scores, args.top_m, true_vars, true_interactions))
            row_syn.update(evaluate_interaction_recovery(synergy_scores, true_interactions))
            rows.append(row_syn)

        return rows

    except Exception as exc:
        row = dict(base)
        row.update({
            "status": "failed",
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "metric": "failed",
            "train_mse": np.nan,
            "test_mse": np.nan,
            "selected_variables": [],
            "variable_precision": np.nan,
            "variable_recall": np.nan,
            "variable_f1": np.nan,
            "selected_interactions": [],
            "interaction_precision": np.nan,
            "interaction_recall": np.nan,
            "interaction_f1": np.nan,
        })
        print(f"[WARN] failed function={function_name}, seed={seed}: {exc}")
        return [row]


# ============================================================
# Save / summarize / merge with KAN benchmark
# ============================================================

def append_rows(path: Path, rows: List[Dict]) -> None:
    """Append rows while keeping a stable CSV schema.

    The first configuration can be additive and therefore have no pair-score
    column unless we force it. Later interaction rows include pair_scores_top50.
    A changing column set corrupts appended CSV files, so we explicitly align
    every append to the existing header when the file already exists.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)

    # Columns that may be absent for additive/no-interaction rows.
    for col, default in {
        "pair_scores_top50": [],
        "importance_scores": [],
        "selected_interactions": [],
        "selected_variables": [],
    }.items():
        if col not in df.columns:
            df[col] = [default for _ in range(len(df))]

    if path.exists() and path.stat().st_size > 0:
        # Read only the header to preserve the original schema.
        header_cols = list(pd.read_csv(path, nrows=0).columns)
        for col in header_cols:
            if col not in df.columns:
                df[col] = np.nan
        # Drop any accidental new columns not present in the existing file.
        df = df[header_cols]
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        df.to_csv(path, mode="w", header=True, index=False)


def summarize_rf(df: pd.DataFrame) -> pd.DataFrame:
    ok = df[df["status"].astype(str) == "ok"].copy()
    if ok.empty:
        return pd.DataFrame()

    group_cols = [
        "function",
        "metric",
        "dimension",
        "samples",
        "test_samples",
        "noise",
        "rf_trees",
        "rf_max_depth",
        "rf_min_samples_leaf",
    ]

    numeric_cols = [
        "train_mse",
        "test_mse",
        "num_true_variables",
        "num_true_interactions",
        "variable_f1",
        "variable_auroc",
        "variable_auprc",
        "interaction_f1",
        "true_interaction_score_mean",
        "max_nontrue_interaction_score",
        "rf_screen_contains_all_true_vars",
        "rf_screen_true_var_recall",
        "rf_screen_contains_all_interaction_endpoints",
        "rf_screen_interaction_endpoint_recall",
        "rf_screen_contains_true_interactions",
    ]

    for col in numeric_cols:
        if col in ok.columns:
            ok[col] = pd.to_numeric(ok[col], errors="coerce")

    agg = {}
    for col in numeric_cols:
        if col in ok.columns:
            if col in {
                "train_mse",
                "test_mse",
                "variable_f1",
                "variable_auroc",
                "variable_auprc",
                "interaction_f1",
                "true_interaction_score_mean",
                "max_nontrue_interaction_score",
            }:
                agg[col] = ["mean", "std"]
            else:
                agg[col] = ["mean"]

    summary = ok.groupby(group_cols, dropna=False).agg(agg).reset_index()
    summary.columns = ["_".join(str(x) for x in col if x != "").rstrip("_") for col in summary.columns]

    counts = df.groupby(["function", "metric"], dropna=False).agg(
        num_rows=("status", "size"),
        num_failed=("status", lambda s: int((s.astype(str) != "ok").sum())),
    ).reset_index()

    summary = summary.merge(counts, on=["function", "metric"], how="left")
    return summary


def make_rf_vs_kan_table(rf_summary: pd.DataFrame, kan_summary_path: str | None, out_path: Path | None) -> pd.DataFrame:
    rows = []

    # RF predictor rows.
    rf_var = rf_summary[rf_summary["metric"] == "rf_feature_importance"].copy()
    rf_syn = rf_summary[rf_summary["metric"] == "rf_pair_permutation_synergy"].copy()

    for fn in sorted(rf_summary["function"].dropna().unique()):
        hit_var = rf_var[rf_var["function"] == fn]
        hit_syn = rf_syn[rf_syn["function"] == fn]

        if not hit_var.empty:
            r = hit_var.iloc[0]
            row = {
                "function": fn,
                "method": "RF predictor",
                "test_mse_mean": r.get("test_mse_mean", np.nan),
                "variable_f1_mean": r.get("variable_f1_mean", np.nan),
                "interaction_f1_mean": np.nan,
                "interaction_metric": "N/A",
                "support_contains_true_interactions_mean": r.get("rf_screen_contains_true_interactions_mean", np.nan),
                "note": "RF feature importance for variables; interaction F1 from synergy row if available.",
            }
            if not hit_syn.empty:
                s = hit_syn.iloc[0]
                row["interaction_f1_mean"] = s.get("interaction_f1_mean", np.nan)
                row["interaction_metric"] = "pair_permutation_synergy"
            rows.append(row)

    # Optional KAN benchmark rows from run_benchmark_suite.py.
    if kan_summary_path is not None and Path(kan_summary_path).exists():
        kan = pd.read_csv(kan_summary_path)
        if "explain_method" in kan.columns:
            kan = kan[kan["explain_method"].astype(str) == "grad"].copy()

        method_map = {
            "raw": "Raw KAN",
            "rf": "RF-screened KAN",
            "oracle_support": "Oracle-support KAN",
            "random": "Random-screened KAN",
            "exclude_interaction": "Exclude-interaction KAN",
        }

        for _, r in kan.iterrows():
            mode = str(r.get("screen_mode", ""))
            if mode not in method_map:
                continue
            rows.append({
                "function": r.get("function", ""),
                "method": method_map[mode],
                "test_mse_mean": r.get("test_mse_mean", np.nan),
                "variable_f1_mean": r.get("variable_f1_mean", np.nan),
                "interaction_f1_mean": r.get("interaction_f1_mean", np.nan),
                "interaction_metric": "KAN Hessian from benchmark suite",
                "support_contains_true_interactions_mean": r.get("screen_contains_true_interactions_mean", np.nan),
                "note": "KAN benchmark result.",
            })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["function", "method"])
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_path, index=False)
    return out


def plot_outputs(rf_summary: pd.DataFrame, combined: pd.DataFrame | None, fig_dir: Path):
    fig_dir.mkdir(parents=True, exist_ok=True)

    # RF predictor diagnostic: variable F1 and interaction F1.
    rf_var = rf_summary[rf_summary["metric"] == "rf_feature_importance"].copy()
    rf_syn = rf_summary[rf_summary["metric"] == "rf_pair_permutation_synergy"].copy()

    fns = sorted(rf_summary["function"].dropna().unique())
    x = np.arange(len(fns))
    width = 0.32

    var_vals = []
    int_vals = []
    mse_vals = []
    support_vals = []

    for fn in fns:
        hv = rf_var[rf_var["function"] == fn]
        hs = rf_syn[rf_syn["function"] == fn]

        var_vals.append(float(hv["variable_f1_mean"].iloc[0]) if not hv.empty else np.nan)
        mse_vals.append(float(hv["test_mse_mean"].iloc[0]) if not hv.empty else np.nan)
        support_vals.append(float(hv["rf_screen_contains_true_interactions_mean"].iloc[0]) if not hv.empty else np.nan)
        int_vals.append(float(hs["interaction_f1_mean"].iloc[0]) if not hs.empty else np.nan)

    plt.figure(figsize=(max(10, len(fns) * 1.1), 5.5))
    plt.bar(x - width / 2, var_vals, width=width, label="RF variable F1")
    plt.bar(x + width / 2, int_vals, width=width, label="RF interaction F1")
    plt.ylim(0, 1.08)
    plt.ylabel("F1")
    plt.xticks(x, fns, rotation=35, ha="right")
    plt.title("RF predictor: support and interaction diagnostics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "rf_predictor_f1.png", dpi=250)
    plt.close()

    plt.figure(figsize=(max(10, len(fns) * 1.1), 5.5))
    plt.bar(x, mse_vals)
    plt.yscale("log")
    plt.ylabel("Test MSE")
    plt.xticks(x, fns, rotation=35, ha="right")
    plt.title("RF predictor test error")
    plt.tight_layout()
    plt.savefig(fig_dir / "rf_predictor_test_mse.png", dpi=250)
    plt.close()

    plt.figure(figsize=(max(10, len(fns) * 1.1), 5.5))
    plt.bar(x, support_vals)
    plt.ylim(0, 1.08)
    plt.ylabel("Fraction across seeds")
    plt.xticks(x, fns, rotation=35, ha="right")
    plt.title("Does RF top-M contain true interaction endpoints?")
    plt.tight_layout()
    plt.savefig(fig_dir / "rf_screen_contains_interaction_support.png", dpi=250)
    plt.close()

    # Optional combined RF vs KAN plots.
    if combined is not None and not combined.empty:
        # Keep a readable subset.
        methods = [
            "RF predictor",
            "Raw KAN",
            "RF-screened KAN",
            "Oracle-support KAN",
            "Random-screened KAN",
            "Exclude-interaction KAN",
        ]
        method_to_idx = {m: i for i, m in enumerate(methods)}
        combined = combined[combined["method"].isin(methods)].copy()
        combined["method_order"] = combined["method"].map(method_to_idx)
        combined = combined.sort_values(["function", "method_order"])

        fns = sorted(combined["function"].dropna().unique())
        x = np.arange(len(fns))
        width = 0.12

        def plot_combined_metric(metric_col: str, ylabel: str, title: str, filename: str, log: bool = False):
            plt.figure(figsize=(max(10, len(fns) * 1.1), 5.5))
            for idx, method in enumerate(methods):
                vals = []
                for fn in fns:
                    hit = combined[(combined["function"] == fn) & (combined["method"] == method)]
                    vals.append(float(hit[metric_col].iloc[0]) if not hit.empty else np.nan)
                plt.bar(x + (idx - (len(methods)-1)/2) * width, vals, width=width, label=method)
            if not log:
                plt.ylim(0, 1.08)
            else:
                plt.yscale("log")
            plt.ylabel(ylabel)
            plt.xticks(x, fns, rotation=35, ha="right")
            plt.title(title)
            plt.legend(ncol=2, fontsize=8)
            plt.tight_layout()
            plt.savefig(fig_dir / filename, dpi=250)
            plt.close()

        plot_combined_metric(
            metric_col="variable_f1_mean",
            ylabel="Variable F1",
            title="RF predictor vs KAN variants: variable recovery",
            filename="rf_vs_kan_variable_f1.png",
            log=False,
        )
        plot_combined_metric(
            metric_col="interaction_f1_mean",
            ylabel="Interaction F1",
            title="RF predictor vs KAN variants: interaction recovery",
            filename="rf_vs_kan_interaction_f1.png",
            log=False,
        )
        plot_combined_metric(
            metric_col="test_mse_mean",
            ylabel="Test MSE",
            title="RF predictor vs KAN variants: prediction error",
            filename="rf_vs_kan_test_mse.png",
            log=True,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--functions",
        type=str,
        nargs="+",
        default=[
            "additive_sparse",
            "core_interaction",
            "core_interaction_c5",
            "pairwise_interaction",
            "compositional",
            "dense_quadratic",
            "correlated_proxy",
        ],
    )
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--test_samples", type=int, default=2048)
    parser.add_argument("--dimension", type=int, default=100)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])

    parser.add_argument("--top_m", type=int, default=20)
    parser.add_argument("--rf_trees", type=int, default=500)
    parser.add_argument("--rf_max_depth", type=int, default=None)
    parser.add_argument("--rf_min_samples_leaf", type=int, default=2)

    parser.add_argument("--compute_pair_permutation", action="store_true")
    parser.add_argument("--perm_points", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--pair_chunk_size", type=int, default=64)

    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--summary_out", type=str, default=None)
    parser.add_argument("--kan_summary_csv", type=str, default=None)
    parser.add_argument("--combined_out", type=str, default=None)
    parser.add_argument("--fig_dir", type=str, default=None)
    parser.add_argument("--resume", action="store_true")

    args = parser.parse_args()

    out_path = Path(args.out)
    existing_keys = set()

    if args.resume and out_path.exists():
        try:
            old = pd.read_csv(out_path)
            for r in old[["function", "seed"]].drop_duplicates().itertuples(index=False):
                existing_keys.add((str(r.function), int(r.seed)))
            print(f"Resume mode: found {len(existing_keys)} completed function/seed configs.")
        except Exception as exc:
            print(f"[WARN] Could not read existing output for resume: {exc}")

    for function_name in args.functions:
        function_name = normalize_function_alias(function_name)
        for seed in args.seeds:
            key = (function_name, int(seed))
            if key in existing_keys:
                print(f"Skipping completed {key}")
                continue

            print(f"Running RF diagnostic function={function_name}, seed={seed}")
            rows = run_one(args, function_name=function_name, seed=seed)
            append_rows(out_path, rows)

    df = pd.read_csv(out_path)
    summary = summarize_rf(df)

    if args.summary_out is not None:
        summary_path = Path(args.summary_out)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(summary_path, index=False)
        print(f"Wrote RF summary to {summary_path}")

    combined = None
    if args.combined_out is not None or args.kan_summary_csv is not None:
        combined_out_path = Path(args.combined_out) if args.combined_out is not None else None
        combined = make_rf_vs_kan_table(summary, args.kan_summary_csv, combined_out_path)
        if combined_out_path is not None:
            print(f"Wrote combined RF/KAN diagnostic table to {combined_out_path}")

    if args.fig_dir is not None:
        plot_outputs(summary, combined, Path(args.fig_dir))
        print(f"Wrote figures to {args.fig_dir}")

    print(f"Wrote rows to {out_path}")


if __name__ == "__main__":
    main()
