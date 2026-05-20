from __future__ import annotations

import argparse
import ast
from pathlib import Path
from collections import Counter
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, roc_auc_score


def parse_list(value):
    if value is None or pd.isna(value):
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    try:
        parsed = ast.literal_eval(str(value))
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, tuple):
            return list(parsed)
    except Exception:
        pass
    return []


def parse_pair_list(value):
    raw = parse_list(value)
    pairs = []
    for item in raw:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            pairs.append(tuple(sorted((int(item[0]), int(item[1])))))
    return pairs


def infer_function_from_name(name: str) -> str:
    known = [
        "core_interaction_c05",
        "core_interaction_c1",
        "core_interaction_c2",
        "core_interaction_c5",
        "core_interaction",
        "highdim_sparse",
        "correlated_proxy",
        "additive_sparse",
        "pairwise_interaction",
        "compositional",
        "rational",
        "discontinuous",
        "dense_quadratic",
    ]
    for k in known:
        if k in name:
            return k
    return "unknown"


def infer_true_variables(function_name: str) -> Tuple[int, ...]:
    if function_name in {
        "core_interaction",
        "highdim_sparse",
        "correlated_proxy",
        "core_interaction_c05",
        "core_interaction_c1",
        "core_interaction_c2",
        "core_interaction_c5",
    }:
        return (0, 1, 2, 3)
    if function_name in {"additive_sparse", "pairwise_interaction", "compositional", "rational"}:
        return (0, 1, 2)
    if function_name == "discontinuous":
        return (0, 1)
    if function_name == "dense_quadratic":
        return (0, 1, 2, 3, 4)
    return ()


def infer_true_interactions(function_name: str) -> Tuple[Tuple[int, int], ...]:
    if function_name in {
        "core_interaction",
        "highdim_sparse",
        "correlated_proxy",
        "core_interaction_c05",
        "core_interaction_c1",
        "core_interaction_c2",
        "core_interaction_c5",
    }:
        return ((2, 3),)
    if function_name in {"pairwise_interaction", "compositional"}:
        return ((0, 1),)
    if function_name == "rational":
        return ((0, 1), (0, 2), (1, 2))
    if function_name == "dense_quadratic":
        pairs = []
        for i in range(5):
            for j in range(i + 1, 5):
                pairs.append((i, j))
        return tuple(pairs)
    return ()


def get_function_name(df: pd.DataFrame, source_file: str) -> str:
    if "function" in df.columns and df["function"].notna().any():
        return str(df["function"].dropna().iloc[0])
    if "function_name" in df.columns and df["function_name"].notna().any():
        return str(df["function_name"].dropna().iloc[0])
    return infer_function_from_name(source_file)


def get_true_variables(df: pd.DataFrame, function_name: str) -> Tuple[int, ...]:
    for col in ["true_variables", "active_variables"]:
        if col in df.columns and df[col].notna().any():
            vals = parse_list(df[col].dropna().iloc[0])
            if vals:
                return tuple(int(v) for v in vals)
    return infer_true_variables(function_name)


def get_true_interactions(df: pd.DataFrame, function_name: str) -> Tuple[Tuple[int, int], ...]:
    for col in ["true_interactions", "interactions"]:
        if col in df.columns and df[col].notna().any():
            vals = parse_pair_list(df[col].dropna().iloc[0])
            if vals:
                return tuple(vals)
    return infer_true_interactions(function_name)


def get_dimension(df: pd.DataFrame, source_file: str, true_vars: Sequence[int]) -> int:
    if "dimension" in df.columns and df["dimension"].notna().any():
        return int(pd.to_numeric(df["dimension"], errors="coerce").dropna().iloc[0])
    if "importance_scores" in df.columns and df["importance_scores"].notna().any():
        scores = parse_list(df["importance_scores"].dropna().iloc[0])
        if scores:
            return len(scores)
    if true_vars:
        return max(true_vars) + 1
    return 0


def f1_from_sets(pred: set, true: set) -> Tuple[float, float, float]:
    if len(pred) == 0 and len(true) == 0:
        return 1.0, 1.0, 1.0
    if len(pred) == 0:
        return 0.0, 0.0, 0.0
    tp = len(pred & true)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(true) if true else 0.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def ranking_metrics(score: np.ndarray, true_indices: Sequence[int]) -> Dict[str, float]:
    d = len(score)
    true_set = set(int(i) for i in true_indices)
    y_true = np.array([1 if i in true_set else 0 for i in range(d)], dtype=int)

    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        auroc = np.nan
        auprc = np.nan
    else:
        auroc = roc_auc_score(y_true, score)
        auprc = average_precision_score(y_true, score)

    return {"auroc": auroc, "auprc": auprc}


def topk_result(score: np.ndarray, true_indices: Sequence[int], k: int | None = None) -> Dict:
    true_set = set(int(i) for i in true_indices)
    if k is None:
        k = len(true_set)
    k = max(int(k), 0)

    if len(score) == 0 or k == 0:
        pred = set()
    else:
        order = np.argsort(-score)
        pred = set(int(i) for i in order[:k])

    precision, recall, f1 = f1_from_sets(pred, true_set)
    metrics = ranking_metrics(score, true_indices) if len(score) else {"auroc": np.nan, "auprc": np.nan}

    return {
        "selected": sorted(pred),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auroc": metrics["auroc"],
        "auprc": metrics["auprc"],
    }


def threshold_result(score: np.ndarray, true_indices: Sequence[int], threshold: float) -> Dict:
    true_set = set(int(i) for i in true_indices)
    pred = set(int(i) for i, s in enumerate(score) if s >= threshold)
    precision, recall, f1 = f1_from_sets(pred, true_set)
    return {
        "selected": sorted(pred),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "num_selected": len(pred),
    }


def parse_importance_scores(sub: pd.DataFrame, d: int) -> List[np.ndarray]:
    scores = []
    if "importance_scores" not in sub.columns:
        return scores

    for value in sub["importance_scores"]:
        arr = parse_list(value)
        if not arr:
            continue
        arr = np.array(arr, dtype=float)
        if len(arr) != d:
            continue
        scores.append(arr)

    return scores


def variable_stability_for_method(
    sub: pd.DataFrame,
    source_file: str,
    function_name: str,
    method: str,
    true_vars: Tuple[int, ...],
    d: int,
    thresholds: Sequence[float],
) -> Tuple[List[Dict], pd.DataFrame]:
    rows = []

    if "selected_variables" not in sub.columns:
        return rows, pd.DataFrame()

    counter = Counter()
    n_runs = 0

    for value in sub["selected_variables"]:
        selected = parse_list(value)
        if selected:
            n_runs += 1
            counter.update(int(v) for v in selected)

    freq_score = np.zeros(d, dtype=float)
    if n_runs > 0:
        for v, c in counter.items():
            if 0 <= int(v) < d:
                freq_score[int(v)] = c / n_runs

    single_seed_mean_f1 = float(sub["variable_f1"].mean()) if "variable_f1" in sub.columns else np.nan
    single_seed_std_f1 = float(sub["variable_f1"].std()) if "variable_f1" in sub.columns else np.nan
    single_seed_mean_auroc = float(sub["variable_auroc"].mean()) if "variable_auroc" in sub.columns else np.nan
    single_seed_mean_auprc = float(sub["variable_auprc"].mean()) if "variable_auprc" in sub.columns else np.nan

    freq_topk = topk_result(freq_score, true_vars)

    rows.append({
        "source_file": source_file,
        "function": function_name,
        "explain_method": method,
        "aggregation": "selection_frequency_topk",
        "threshold": np.nan,
        "num_runs": n_runs,
        "single_seed_mean_f1": single_seed_mean_f1,
        "single_seed_std_f1": single_seed_std_f1,
        "single_seed_mean_auroc": single_seed_mean_auroc,
        "single_seed_mean_auprc": single_seed_mean_auprc,
        "stable_precision": freq_topk["precision"],
        "stable_recall": freq_topk["recall"],
        "stable_f1": freq_topk["f1"],
        "stable_auroc": freq_topk["auroc"],
        "stable_auprc": freq_topk["auprc"],
        "stable_selected": freq_topk["selected"],
    })

    for tau in thresholds:
        thr = threshold_result(freq_score, true_vars, tau)
        rows.append({
            "source_file": source_file,
            "function": function_name,
            "explain_method": method,
            "aggregation": "selection_frequency_threshold",
            "threshold": tau,
            "num_runs": n_runs,
            "single_seed_mean_f1": single_seed_mean_f1,
            "single_seed_std_f1": single_seed_std_f1,
            "single_seed_mean_auroc": single_seed_mean_auroc,
            "single_seed_mean_auprc": single_seed_mean_auprc,
            "stable_precision": thr["precision"],
            "stable_recall": thr["recall"],
            "stable_f1": thr["f1"],
            "stable_auroc": np.nan,
            "stable_auprc": np.nan,
            "stable_num_selected": thr["num_selected"],
            "stable_selected": thr["selected"],
        })

    score_arrays = parse_importance_scores(sub, d)
    if score_arrays:
        mean_score = np.mean(np.stack(score_arrays, axis=0), axis=0)
        mean_topk = topk_result(mean_score, true_vars)
        rows.append({
            "source_file": source_file,
            "function": function_name,
            "explain_method": method,
            "aggregation": "mean_importance_topk",
            "threshold": np.nan,
            "num_runs": len(score_arrays),
            "single_seed_mean_f1": single_seed_mean_f1,
            "single_seed_std_f1": single_seed_std_f1,
            "single_seed_mean_auroc": single_seed_mean_auroc,
            "single_seed_mean_auprc": single_seed_mean_auprc,
            "stable_precision": mean_topk["precision"],
            "stable_recall": mean_topk["recall"],
            "stable_f1": mean_topk["f1"],
            "stable_auroc": mean_topk["auroc"],
            "stable_auprc": mean_topk["auprc"],
            "stable_selected": mean_topk["selected"],
        })
    else:
        mean_score = np.full(d, np.nan)

    variable_rows = []
    true_set = set(true_vars)
    for v in range(d):
        variable_rows.append({
            "source_file": source_file,
            "function": function_name,
            "explain_method": method,
            "variable": v,
            "is_true_active": int(v in true_set),
            "selection_frequency": freq_score[v],
            "mean_importance": mean_score[v] if not np.isnan(mean_score[v]) else np.nan,
        })

    return rows, pd.DataFrame(variable_rows)


def interaction_stability_for_method(
    sub: pd.DataFrame,
    source_file: str,
    function_name: str,
    method: str,
    true_interactions: Tuple[Tuple[int, int], ...],
    thresholds: Sequence[float],
) -> Tuple[List[Dict], pd.DataFrame]:
    rows = []

    if "selected_interactions" not in sub.columns:
        return rows, pd.DataFrame()

    counter = Counter()
    n_runs = 0

    for value in sub["selected_interactions"]:
        selected = parse_pair_list(value)
        if selected:
            n_runs += 1
            counter.update(selected)

    true_set = set(tuple(sorted(p)) for p in true_interactions)
    all_pairs = set(counter.keys()) | true_set

    if not all_pairs:
        return rows, pd.DataFrame()

    pair_list = sorted(all_pairs)
    freq_score = np.array([counter.get(p, 0) / n_runs if n_runs > 0 else 0.0 for p in pair_list])

    k = max(len(true_set), 1)
    order = np.argsort(-freq_score)
    pred = set(pair_list[i] for i in order[:k])
    precision, recall, f1 = f1_from_sets(pred, true_set)

    single_seed_mean_f1 = float(sub["interaction_f1"].mean()) if "interaction_f1" in sub.columns else np.nan
    single_seed_std_f1 = float(sub["interaction_f1"].std()) if "interaction_f1" in sub.columns else np.nan

    rows.append({
        "source_file": source_file,
        "function": function_name,
        "explain_method": method,
        "aggregation": "interaction_frequency_topk",
        "threshold": np.nan,
        "num_runs": n_runs,
        "single_seed_mean_interaction_f1": single_seed_mean_f1,
        "single_seed_std_interaction_f1": single_seed_std_f1,
        "stable_interaction_precision": precision,
        "stable_interaction_recall": recall,
        "stable_interaction_f1": f1,
        "stable_selected_interactions": sorted(pred),
    })

    for tau in thresholds:
        pred_tau = set(pair_list[i] for i, s in enumerate(freq_score) if s >= tau)
        p_tau, r_tau, f_tau = f1_from_sets(pred_tau, true_set)
        rows.append({
            "source_file": source_file,
            "function": function_name,
            "explain_method": method,
            "aggregation": "interaction_frequency_threshold",
            "threshold": tau,
            "num_runs": n_runs,
            "single_seed_mean_interaction_f1": single_seed_mean_f1,
            "single_seed_std_interaction_f1": single_seed_std_f1,
            "stable_interaction_precision": p_tau,
            "stable_interaction_recall": r_tau,
            "stable_interaction_f1": f_tau,
            "stable_num_selected_interactions": len(pred_tau),
            "stable_selected_interactions": sorted(pred_tau),
        })

    pair_rows = []
    for pair, score in zip(pair_list, freq_score):
        pair_rows.append({
            "source_file": source_file,
            "function": function_name,
            "explain_method": method,
            "pair": str(pair),
            "is_true_interaction": int(pair in true_set),
            "selection_frequency": score,
        })

    return rows, pd.DataFrame(pair_rows)


def load_result_files(input_dirs: Sequence[str]) -> List[Path]:
    files = []
    for d in input_dirs:
        p = Path(d)
        if p.is_file() and p.suffix == ".csv":
            files.append(p)
        elif p.is_dir():
            files.extend(sorted(p.rglob("*.csv")))

    result_files = []
    for f in files:
        name = f.name.lower()
        stem = f.stem.lower()
        if "scores" in stem:
            continue
        if "summary" in stem:
            continue
        if "selection_frequency" in stem:
            continue
        if "stability" in stem:
            continue
        result_files.append(f)

    return sorted(set(result_files))


def process_file(path: Path, thresholds: Sequence[float]):
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        print(f"[WARN] Could not read {path}: {exc}")
        return [], pd.DataFrame(), [], pd.DataFrame()

    if df.empty:
        return [], pd.DataFrame(), [], pd.DataFrame()

    if "model" in df.columns:
        df = df[df["model"].astype(str).str.upper() == "KAN"].copy()

    if df.empty:
        return [], pd.DataFrame(), [], pd.DataFrame()

    if "explain_method" not in df.columns:
        df["explain_method"] = "unknown"

    function_name = get_function_name(df, path.name)
    true_vars = get_true_variables(df, function_name)
    true_interactions = get_true_interactions(df, function_name)
    d = get_dimension(df, path.name, true_vars)

    var_rows_all = []
    var_freq_all = []
    int_rows_all = []
    int_freq_all = []

    for method, sub in df.groupby("explain_method"):
        method = str(method)

        rows, var_freq = variable_stability_for_method(
            sub=sub,
            source_file=path.name,
            function_name=function_name,
            method=method,
            true_vars=true_vars,
            d=d,
            thresholds=thresholds,
        )
        var_rows_all.extend(rows)
        if not var_freq.empty:
            var_freq_all.append(var_freq)

        irows, ifreq = interaction_stability_for_method(
            sub=sub,
            source_file=path.name,
            function_name=function_name,
            method=method,
            true_interactions=true_interactions,
            thresholds=thresholds,
        )
        int_rows_all.extend(irows)
        if not ifreq.empty:
            int_freq_all.append(ifreq)

    var_freq_df = pd.concat(var_freq_all, ignore_index=True) if var_freq_all else pd.DataFrame()
    int_freq_df = pd.concat(int_freq_all, ignore_index=True) if int_freq_all else pd.DataFrame()

    return var_rows_all, var_freq_df, int_rows_all, int_freq_df


def plot_variable_comparison(summary_df: pd.DataFrame, out_dir: Path):
    if summary_df.empty:
        return

    topk = summary_df[summary_df["aggregation"].isin(["selection_frequency_topk", "mean_importance_topk"])].copy()
    if topk.empty:
        return

    # Keep the most informative/high-dimensional settings if many files exist.
    topk["setting_label"] = topk["source_file"].str.replace(".csv", "", regex=False)
    topk = topk.sort_values(["setting_label", "explain_method", "aggregation"])

    # Limit plot to at most 20 rows for readability.
    plot_df = topk.head(20).copy()
    labels = [
        f"{r.setting_label}\n{r.explain_method}, {r.aggregation.replace('_topk','')}"
        for r in plot_df.itertuples()
    ]

    x = np.arange(len(plot_df))
    width = 0.35

    plt.figure(figsize=(max(10, len(plot_df) * 0.7), 5.2))
    plt.bar(x - width / 2, plot_df["single_seed_mean_f1"], width=width, label="single-seed mean F1")
    plt.bar(x + width / 2, plot_df["stable_f1"], width=width, label="stable aggregated F1")
    plt.ylim(0, 1.05)
    plt.ylabel("Variable recovery F1")
    plt.xticks(x, labels, rotation=55, ha="right")
    plt.title("Single-seed vs stability-selected variable recovery")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "variable_stability_selected_comparison.png", dpi=250)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--out", default="results/stability_selected")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.2, 0.4, 0.6, 0.8])
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    result_files = load_result_files(args.input)
    print(f"Found {len(result_files)} result files.")

    var_rows = []
    var_freqs = []
    int_rows = []
    int_freqs = []

    for f in result_files:
        print(f"Processing {f}")
        rows, var_freq, irows, int_freq = process_file(f, args.thresholds)
        var_rows.extend(rows)
        int_rows.extend(irows)
        if not var_freq.empty:
            var_freqs.append(var_freq)
        if not int_freq.empty:
            int_freqs.append(int_freq)

    var_summary = pd.DataFrame(var_rows)
    int_summary = pd.DataFrame(int_rows)
    var_freq_all = pd.concat(var_freqs, ignore_index=True) if var_freqs else pd.DataFrame()
    int_freq_all = pd.concat(int_freqs, ignore_index=True) if int_freqs else pd.DataFrame()

    var_summary.to_csv(out_dir / "stability_selected_variable_summary.csv", index=False)
    int_summary.to_csv(out_dir / "stability_selected_interaction_summary.csv", index=False)
    var_freq_all.to_csv(out_dir / "stability_selected_variable_frequencies.csv", index=False)
    int_freq_all.to_csv(out_dir / "stability_selected_interaction_frequencies.csv", index=False)

    with pd.ExcelWriter(out_dir / "stability_selected_summary.xlsx", engine="openpyxl") as writer:
        var_summary.to_excel(writer, sheet_name="Variable Stable Summary", index=False)
        int_summary.to_excel(writer, sheet_name="Interaction Stable Summary", index=False)
        var_freq_all.to_excel(writer, sheet_name="Variable Frequencies", index=False)
        int_freq_all.to_excel(writer, sheet_name="Interaction Frequencies", index=False)

    plot_variable_comparison(var_summary, out_dir)

    print(f"Saved outputs to {out_dir}")


if __name__ == "__main__":
    main()
