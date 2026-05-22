import argparse
import ast
import glob
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


SCREEN_LABELS = {
    "raw": "Raw KAN",
    "rf": "RF-screened",
    "oracle_support": "Oracle-support",
    "random": "Random-screened",
    "exclude_interaction": "Exclude-interaction",
}


def parse_int_list(x):
    """Parse selected_variables-like fields robustly."""
    if x is None:
        return []
    if isinstance(x, float) and math.isnan(x):
        return []
    if isinstance(x, (list, tuple, set, np.ndarray)):
        return [int(v) for v in x]

    s = str(x).strip()
    if s == "" or s.lower() == "nan" or s == "[]":
        return []

    # Try normal Python/JSON literal first.
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, (list, tuple, set)):
            return [int(v) for v in obj]
        if isinstance(obj, dict):
            return [int(k) for k in obj.keys()]
    except Exception:
        pass

    # Fallback: extract integers from strings such as "[np.int64(2), np.int64(3)]".
    nums = re.findall(r"-?\d+", s)
    return [int(v) for v in nums]


def parse_scores(x):
    """
    Parse variable score fields.
    Supports:
      - list: [0.1, 0.2, ...]
      - dict: {"0": 0.1, "1": 0.2, ...}
      - strings of either.
    Returns dict[int, float] or None.
    """
    if x is None:
        return None
    if isinstance(x, float) and math.isnan(x):
        return None

    obj = x
    if isinstance(x, str):
        s = x.strip()
        if s == "" or s.lower() == "nan":
            return None
        try:
            obj = json.loads(s)
        except Exception:
            try:
                obj = ast.literal_eval(s)
            except Exception:
                return None

    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            try:
                out[int(k)] = float(v)
            except Exception:
                continue
        return out if out else None

    if isinstance(obj, (list, tuple, np.ndarray)):
        out = {}
        for i, v in enumerate(obj):
            try:
                out[int(i)] = float(v)
            except Exception:
                continue
        return out if out else None

    return None


def infer_dimension_from_path(path):
    name = Path(path).name
    m = re.search(r"d(\d+)", name)
    if m:
        return int(m.group(1))
    return None


def find_selected_variable_column(df):
    candidates = [
        "selected_variables",
        "selected_vars",
        "selected_support",
        "support",
        "predicted_variables",
        "pred_variables",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def find_score_column(df):
    candidates = [
        "variable_scores",
        "var_scores",
        "feature_scores",
        "feature_importances",
        "variable_importances",
        "importance_scores",
        "attribution_scores",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def load_inputs(patterns):
    frames = []
    for pat in patterns:
        matched = sorted(glob.glob(pat))
        if not matched:
            print(f"[warn] no files matched: {pat}")
        for p in matched:
            df = pd.read_csv(p)
            df["source_file"] = p
            if "dimension" not in df.columns:
                dim = infer_dimension_from_path(p)
                if dim is not None:
                    df["dimension"] = dim
            frames.append(df)

    if not frames:
        raise FileNotFoundError("No input CSV files found.")

    return pd.concat(frames, ignore_index=True)


def pairwise_jaccard(sets):
    if len(sets) <= 1:
        return np.nan

    vals = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            a, b = sets[i], sets[j]
            union = a | b
            if len(union) == 0:
                vals.append(1.0)
            else:
                vals.append(len(a & b) / len(union))
    return float(np.mean(vals)) if vals else np.nan


def make_retention_and_stability(df, selected_col, true_support, interaction_pair):
    rows = []

    df = df.copy()
    df["selected_set"] = df[selected_col].apply(lambda x: set(parse_int_list(x)))

    for (function, dimension, screen_mode), g in df.groupby(["function", "dimension", "screen_mode"], dropna=False):
        selected_sets = list(g["selected_set"])

        endpoint_retention = np.mean([
            set(interaction_pair).issubset(s) for s in selected_sets
        ])

        full_support_retention = np.mean([
            set(true_support).issubset(s) for s in selected_sets
        ])

        at_least_one_endpoint = np.mean([
            len(set(interaction_pair) & s) >= 1 for s in selected_sets
        ])

        stability = pairwise_jaccard(selected_sets)

        avg_size = np.mean([len(s) for s in selected_sets])

        rows.append({
            "function": function,
            "dimension": int(dimension),
            "screen_mode": screen_mode,
            "n_seeds": len(g),
            "endpoint_retention": endpoint_retention,
            "full_support_retention": full_support_retention,
            "at_least_one_endpoint": at_least_one_endpoint,
            "support_stability": stability,
            "avg_selected_size": avg_size,
        })

    return pd.DataFrame(rows)


def make_score_gap(df, score_col, true_support, interaction_pair):
    rows = []

    for _, row in df.iterrows():
        scores = parse_scores(row.get(score_col))
        if not scores:
            continue

        dim = int(row["dimension"])
        all_vars = set(range(dim))
        true_support_set = set(true_support)
        nuisance = sorted(all_vars - true_support_set)

        if not all(v in scores for v in true_support):
            continue
        if not all(v in scores for v in interaction_pair):
            continue

        nuisance_scores = [scores[v] for v in nuisance if v in scores]
        if len(nuisance_scores) == 0:
            continue

        true_min = min(scores[v] for v in true_support)
        endpoint_min = min(scores[v] for v in interaction_pair)
        nuisance_max = max(nuisance_scores)

        rows.append({
            "function": row.get("function"),
            "dimension": dim,
            "screen_mode": row.get("screen_mode"),
            "seed": row.get("seed", np.nan),
            "true_min_score": true_min,
            "endpoint_min_score": endpoint_min,
            "nuisance_max_score": nuisance_max,
            "support_score_gap": true_min - nuisance_max,
            "endpoint_score_gap": endpoint_min - nuisance_max,
        })

    return pd.DataFrame(rows)


def plot_metric_lines(summary, metric, ylabel, title, out_path):
    plt.figure(figsize=(8, 5))

    for mode, g in summary.groupby("screen_mode"):
        g = g.sort_values("dimension")
        label = SCREEN_LABELS.get(mode, mode)
        plt.plot(g["dimension"], g[metric], marker="o", linewidth=2, label=label)

    plt.xlabel("Input dimension d")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.ylim(-0.05, 1.05) if "retention" in metric or "stability" in metric else None
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=250)
    plt.close()
    print(f"[saved] {out_path}")


def plot_score_gap(score_df, metric, ylabel, title, out_path):
    if score_df.empty:
        print(f"[skip] no score data available for {metric}")
        return

    agg = (
        score_df
        .groupby(["dimension", "screen_mode"], as_index=False)
        .agg(mean=(metric, "mean"), std=(metric, "std"), n=(metric, "count"))
    )

    plt.figure(figsize=(8, 5))

    for mode, g in agg.groupby("screen_mode"):
        g = g.sort_values("dimension")
        label = SCREEN_LABELS.get(mode, mode)
        plt.plot(g["dimension"], g["mean"], marker="o", linewidth=2, label=label)

        if len(g) > 1:
            std = g["std"].fillna(0).to_numpy()
            x = g["dimension"].to_numpy()
            y = g["mean"].to_numpy()
            plt.fill_between(x, y - std, y + std, alpha=0.15)

    plt.axhline(0.0, linestyle="--", linewidth=1)
    plt.xlabel("Input dimension d")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=250)
    plt.close()
    print(f"[saved] {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--function", default="core_interaction")
    parser.add_argument("--true_support", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--interaction_pair", nargs=2, type=int, default=[2, 3])
    parser.add_argument("--screen_modes", nargs="*", default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_inputs(args.inputs)

    if "function" not in df.columns:
        df["function"] = args.function

    df = df[df["function"] == args.function].copy()

    if args.screen_modes:
        df = df[df["screen_mode"].isin(args.screen_modes)].copy()

    if df.empty:
        raise ValueError("No rows left after filtering. Check --function and --screen_modes.")

    if "dimension" not in df.columns:
        raise ValueError("No dimension column found and could not infer dimension from filename.")

    selected_col = find_selected_variable_column(df)
    if selected_col is None:
        print("[error] Could not find selected variable column.")
        print("Available columns:")
        print(list(df.columns))
        raise SystemExit(1)

    print(f"[info] using selected variable column: {selected_col}")

    support_summary = make_retention_and_stability(
        df=df,
        selected_col=selected_col,
        true_support=args.true_support,
        interaction_pair=args.interaction_pair,
    )

    support_summary = support_summary.sort_values(["dimension", "screen_mode"])
    support_summary_path = out_dir / "support_retention_stability_summary.csv"
    support_summary.to_csv(support_summary_path, index=False)
    print(f"[saved] {support_summary_path}")
    print(support_summary.to_string(index=False))

    plot_metric_lines(
        support_summary,
        metric="endpoint_retention",
        ylabel="P(true interaction endpoints retained)",
        title=f"Interaction endpoint retention: {args.function}",
        out_path=out_dir / "endpoint_retention_vs_dimension.pdf",
    )

    plot_metric_lines(
        support_summary,
        metric="full_support_retention",
        ylabel="P(full true support retained)",
        title=f"Full support retention: {args.function}",
        out_path=out_dir / "full_support_retention_vs_dimension.pdf",
    )

    plot_metric_lines(
        support_summary,
        metric="support_stability",
        ylabel="Pairwise support Jaccard stability",
        title=f"Support stability across seeds: {args.function}",
        out_path=out_dir / "support_stability_vs_dimension.pdf",
    )

    score_col = find_score_column(df)
    if score_col is None:
        print("[warn] No variable score column found. Score-gap plots skipped.")
        print("[warn] Expected one of: variable_scores, var_scores, feature_scores, feature_importances, variable_importances, importance_scores, attribution_scores.")
        print("[warn] You can still use retention/stability plots now.")
        return

    print(f"[info] using score column: {score_col}")

    score_df = make_score_gap(
        df=df,
        score_col=score_col,
        true_support=args.true_support,
        interaction_pair=args.interaction_pair,
    )

    score_path = out_dir / "score_gap_rows.csv"
    score_df.to_csv(score_path, index=False)
    print(f"[saved] {score_path}")

    plot_score_gap(
        score_df,
        metric="support_score_gap",
        ylabel="min true support score - max nuisance score",
        title=f"Support score separation: {args.function}",
        out_path=out_dir / "support_score_gap_vs_dimension.pdf",
    )

    plot_score_gap(
        score_df,
        metric="endpoint_score_gap",
        ylabel="min interaction endpoint score - max nuisance score",
        title=f"Interaction-endpoint score separation: {args.function}",
        out_path=out_dir / "endpoint_score_gap_vs_dimension.pdf",
    )


if __name__ == "__main__":
    main()
