import argparse
import ast
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd
import matplotlib.pyplot as plt


def parse_list(value):
    if pd.isna(value):
        return []
    if isinstance(value, list):
        return value
    try:
        parsed = ast.literal_eval(str(value))
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, tuple):
            return list(parsed)
        return []
    except Exception:
        return []


def parse_pair_list(value):
    if pd.isna(value):
        return []
    try:
        parsed = ast.literal_eval(str(value))
        pairs = []
        for p in parsed:
            if isinstance(p, (list, tuple)) and len(p) == 2:
                pairs.append(tuple(sorted((int(p[0]), int(p[1])))))
        return pairs
    except Exception:
        return []


def infer_setting_name(path):
    stem = path.stem
    return stem.replace("_scores", "")


def load_result_files(input_dirs):
    files = []
    for d in input_dirs:
        d = Path(d)
        files.extend(sorted(d.rglob("*.csv")))

    result_files = [
        f for f in files
        if "scores" not in f.stem.lower()
        and "summary" not in f.stem.lower()
        and "selection_frequency" not in f.stem.lower()
    ]

    return result_files


def get_true_variables(row):
    for col in ["true_variables", "active_variables"]:
        if col in row and not pd.isna(row[col]):
            vals = parse_list(row[col])
            if vals:
                return set(int(v) for v in vals)

    # Default for all core_interaction / highdim_sparse / correlated_proxy experiments.
    return {0, 1, 2, 3}


def get_true_interactions(row):
    for col in ["true_interactions", "interactions"]:
        if col in row and not pd.isna(row[col]):
            vals = parse_pair_list(row[col])
            if vals:
                return set(vals)

    return {(2, 3)}


def variable_stability_for_file(path):
    df = pd.read_csv(path)

    if "selected_variables" not in df.columns:
        return pd.DataFrame()

    if "model" in df.columns:
        df = df[df["model"].astype(str).str.upper() == "KAN"].copy()

    if "explain_method" not in df.columns:
        df["explain_method"] = "unknown"

    setting = infer_setting_name(path)
    rows = []

    for method, sub in df.groupby("explain_method"):
        counter = Counter()
        seed_count = len(sub)

        if seed_count == 0:
            continue

        true_vars = get_true_variables(sub.iloc[0])

        for _, row in sub.iterrows():
            selected = parse_list(row["selected_variables"])
            selected = [int(v) for v in selected]
            counter.update(selected)

        all_seen = set(counter.keys()) | true_vars

        for v in sorted(all_seen):
            count = counter.get(v, 0)
            rows.append({
                "source_file": path.name,
                "setting": setting,
                "explain_method": method,
                "variable": v,
                "is_true_active": v in true_vars,
                "selection_count": count,
                "num_runs": seed_count,
                "selection_frequency": count / seed_count,
            })

    return pd.DataFrame(rows)


def interaction_stability_for_file(path):
    df = pd.read_csv(path)

    if "selected_interactions" not in df.columns:
        return pd.DataFrame()

    if "model" in df.columns:
        df = df[df["model"].astype(str).str.upper() == "KAN"].copy()

    if "explain_method" not in df.columns:
        df["explain_method"] = "unknown"

    setting = infer_setting_name(path)
    rows = []

    for method, sub in df.groupby("explain_method"):
        counter = Counter()
        seed_count = len(sub)

        if seed_count == 0:
            continue

        true_interactions = get_true_interactions(sub.iloc[0])

        for _, row in sub.iterrows():
            selected = parse_pair_list(row["selected_interactions"])
            counter.update(selected)

        all_seen = set(counter.keys()) | true_interactions

        for pair in sorted(all_seen):
            count = counter.get(pair, 0)
            rows.append({
                "source_file": path.name,
                "setting": setting,
                "explain_method": method,
                "interaction": str(pair),
                "is_true_interaction": pair in true_interactions,
                "selection_count": count,
                "num_runs": seed_count,
                "selection_frequency": count / seed_count,
            })

    return pd.DataFrame(rows)


def make_top_variable_tables(var_df, out_dir, top_k=15):
    out_dir = Path(out_dir)
    tables_dir = out_dir / "top_variables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    for (setting, method), sub in var_df.groupby(["setting", "explain_method"]):
        sub = sub.sort_values(
            ["selection_frequency", "is_true_active", "variable"],
            ascending=[False, False, True],
        )
        top = sub.head(top_k)

        safe = f"{setting}_{method}".replace("/", "_").replace(" ", "_")
        top.to_csv(tables_dir / f"top_variables_{safe}.csv", index=False)


def plot_top_variables(var_df, out_dir, top_k=12):
    out_dir = Path(out_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    for (setting, method), sub in var_df.groupby(["setting", "explain_method"]):
        sub = sub.sort_values(
            ["selection_frequency", "is_true_active", "variable"],
            ascending=[False, False, True],
        ).head(top_k)

        labels = [
            f"x{int(v)}*" if active else f"x{int(v)}"
            for v, active in zip(sub["variable"], sub["is_true_active"])
        ]

        plt.figure(figsize=(8, 4.5))
        bars = plt.bar(labels, sub["selection_frequency"])
        plt.ylim(0, 1.05)
        plt.ylabel("Selection frequency")
        plt.xlabel("Variable")
        plt.title(f"Top selected variables\n{setting}, {method}")

        for bar, active in zip(bars, sub["is_true_active"]):
            if active:
                bar.set_hatch("//")

        plt.tight_layout()
        safe = f"{setting}_{method}".replace("/", "_").replace(" ", "_")
        plt.savefig(fig_dir / f"top_variables_{safe}.png", dpi=220)
        plt.close()


def make_summary(var_df, int_df, out_dir):
    summaries = []

    if not var_df.empty:
        var_summary = (
            var_df.groupby(["setting", "explain_method"])
            .apply(lambda g: pd.Series({
                "true_active_mean_freq": g[g["is_true_active"]]["selection_frequency"].mean(),
                "true_active_min_freq": g[g["is_true_active"]]["selection_frequency"].min(),
                "inactive_max_freq": g[~g["is_true_active"]]["selection_frequency"].max(),
                "num_inactive_selected_at_least_once": int(
                    (g[~g["is_true_active"]]["selection_frequency"] > 0).sum()
                ),
            }))
            .reset_index()
        )
        summaries.append(("variable_stability_summary.csv", var_summary))

    if not int_df.empty:
        int_summary = (
            int_df.groupby(["setting", "explain_method"])
            .apply(lambda g: pd.Series({
                "true_interaction_mean_freq": g[g["is_true_interaction"]]["selection_frequency"].mean(),
                "true_interaction_max_freq": g[g["is_true_interaction"]]["selection_frequency"].max(),
                "false_interaction_max_freq": g[~g["is_true_interaction"]]["selection_frequency"].max(),
                "num_false_interactions_selected_at_least_once": int(
                    (g[~g["is_true_interaction"]]["selection_frequency"] > 0).sum()
                ),
            }))
            .reset_index()
        )
        summaries.append(("interaction_stability_summary.csv", int_summary))

    for name, df in summaries:
        df.to_csv(Path(out_dir) / name, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="One or more directories containing result CSV files.",
    )
    parser.add_argument(
        "--out",
        default="results/stability_selection",
        help="Output directory.",
    )
    parser.add_argument("--top_k", type=int, default=15)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    result_files = load_result_files(args.input)

    var_dfs = []
    int_dfs = []

    for f in result_files:
        var_df = variable_stability_for_file(f)
        if not var_df.empty:
            var_dfs.append(var_df)

        int_df = interaction_stability_for_file(f)
        if not int_df.empty:
            int_dfs.append(int_df)

    if var_dfs:
        var_all = pd.concat(var_dfs, ignore_index=True)
    else:
        var_all = pd.DataFrame()

    if int_dfs:
        int_all = pd.concat(int_dfs, ignore_index=True)
    else:
        int_all = pd.DataFrame()

    if not var_all.empty:
        var_all.to_csv(out_dir / "variable_selection_frequency.csv", index=False)
        make_top_variable_tables(var_all, out_dir, top_k=args.top_k)
        plot_top_variables(var_all, out_dir, top_k=args.top_k)

    if not int_all.empty:
        int_all.to_csv(out_dir / "interaction_selection_frequency.csv", index=False)

    make_summary(var_all, int_all, out_dir)

    with pd.ExcelWriter(out_dir / "stability_selection_summary.xlsx", engine="openpyxl") as writer:
        if not var_all.empty:
            var_all.to_excel(writer, sheet_name="Variable Frequency", index=False)
        if not int_all.empty:
            int_all.to_excel(writer, sheet_name="Interaction Frequency", index=False)

    print(f"Processed {len(result_files)} result files.")
    print(f"Saved outputs to {out_dir}")


if __name__ == "__main__":
    main()