import argparse
import ast
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_noise_from_name(name: str):
    m = re.search(r"noise([0-9.]+)", name)
    if not m:
        return None

    raw = m.group(1).rstrip(".")
    if "." in raw:
        return float(raw)

    if raw == "00":
        return 0.0
    if raw == "01":
        return 0.1
    if raw == "02":
        return 0.2
    if raw == "03":
        return 0.3
    if raw == "005":
        return 0.05
    if raw == "001":
        return 0.01

    try:
        return float(raw)
    except ValueError:
        return None


def parse_first_int(pattern: str, name: str):
    m = re.search(pattern, name)
    if not m:
        return None
    return int(m.group(1))


def parse_lamb_from_name(name: str):
    m = re.search(r"lamb([0-9.]+)", name)
    if not m:
        return None
    raw = m.group(1).rstrip(".")
    try:
        return float(raw)
    except ValueError:
        return None


def infer_function_name(stem: str):
    known = [
        "core_interaction",
        "compositional",
        "correlated_proxy",
        "additive_sparse",
        "pairwise_interaction",
        "rational",
        "discontinuous",
        "dense_quadratic",
        "highdim_sparse",
    ]
    for k in known:
        if k in stem:
            return k
    return stem


def infer_setting_label(row):
    function = row.get("function", None)
    dimension = row.get("dimension", None)
    samples = row.get("samples", None)
    noise = row.get("noise", None)
    lamb = row.get("lamb", None)

    parts = []
    if pd.notna(function):
        parts.append(str(function))
    if pd.notna(dimension):
        parts.append(f"d={int(dimension)}")
    if pd.notna(samples):
        parts.append(f"n={int(samples)}")
    if pd.notna(noise):
        parts.append(f"noise={noise:g}")
    if pd.notna(lamb):
        parts.append(f"lambda={lamb:g}")

    return ", ".join(parts)


def normalize_result_df(path: Path):
    df = pd.read_csv(path)
    stem = path.stem

    if "source_file" not in df.columns:
        df["source_file"] = path.name

    if "function" not in df.columns:
        if "function_name" in df.columns:
            df["function"] = df["function_name"]
        else:
            df["function"] = infer_function_name(stem)

    if "dimension" not in df.columns:
        dim = parse_first_int(r"d(\d+)", stem)
        if dim is None:
            dim = parse_first_int(r"dimension(\d+)", stem)
        df["dimension"] = dim

    if "samples" not in df.columns:
        samples = parse_first_int(r"n(\d+)", stem)
        if samples is None:
            samples = parse_first_int(r"samples(\d+)", stem)
        df["samples"] = samples

    if "noise" not in df.columns:
        df["noise"] = parse_noise_from_name(stem)

    if "lamb" not in df.columns:
        df["lamb"] = parse_lamb_from_name(stem)

    if "model" not in df.columns:
        df["model"] = "KAN"

    if "explain_method" not in df.columns:
        df["explain_method"] = "unknown"

    numeric_cols = [
        "train_mse",
        "test_mse",
        "variable_f1",
        "variable_precision",
        "variable_recall",
        "variable_auroc",
        "variable_auprc",
        "interaction_f1",
        "interaction_precision",
        "interaction_recall",
        "active_score_mean",
        "inactive_score_mean",
        "active_score_min",
        "inactive_score_max",
        "samples",
        "dimension",
        "noise",
        "lamb",
    ]

    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["setting"] = df.apply(infer_setting_label, axis=1)
    return df


def read_all_results(input_dir: Path):
    files = sorted(input_dir.glob("*.csv"))
    result_files = [
        f for f in files
        if "scores" not in f.stem.lower()
        and "summary" not in f.stem.lower()
    ]

    if not result_files:
        raise FileNotFoundError(f"No result CSV files found in {input_dir}")

    dfs = []
    for f in result_files:
        try:
            dfs.append(normalize_result_df(f))
        except Exception as e:
            print(f"[WARN] Failed to read {f}: {e}")

    if not dfs:
        raise RuntimeError("No result CSV files could be loaded.")

    return pd.concat(dfs, ignore_index=True)


def read_score_files(input_dir: Path):
    files = sorted(input_dir.glob("*scores*.csv"))
    dfs = []

    for f in files:
        try:
            df = pd.read_csv(f)
            df["source_file"] = f.name
            df["function"] = infer_function_name(f.stem)

            if "dimension" not in df.columns:
                df["dimension"] = parse_first_int(r"d(\d+)", f.stem)
            if "samples" not in df.columns:
                df["samples"] = parse_first_int(r"n(\d+)", f.stem)
            if "noise" not in df.columns:
                df["noise"] = parse_noise_from_name(f.stem)

            colmap = {}
            for c in df.columns:
                lc = c.lower()
                if lc in {"variable", "variable_index", "feature", "feature_index", "idx"}:
                    colmap[c] = "variable_index"
                if lc in {"score", "importance", "importance_score", "variable_score"}:
                    colmap[c] = "score"
                if lc in {"is_active", "active", "true_active", "label"}:
                    colmap[c] = "is_active"
                if lc in {"method", "explain_method"}:
                    colmap[c] = "explain_method"
                if lc in {"seed", "random_seed"}:
                    colmap[c] = "seed"

            df = df.rename(columns=colmap)

            needed = {"variable_index", "score", "is_active"}
            if not needed.issubset(df.columns):
                print(f"[WARN] Score file {f.name} missing columns {needed - set(df.columns)}; skipping.")
                continue

            df["score"] = pd.to_numeric(df["score"], errors="coerce")
            df["is_active"] = df["is_active"].astype(int)
            dfs.append(df)

        except Exception as e:
            print(f"[WARN] Failed to read score file {f}: {e}")

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)


def make_summary(df: pd.DataFrame):
    group_cols = [
        "source_file",
        "setting",
        "function",
        "dimension",
        "samples",
        "noise",
        "lamb",
        "model",
        "explain_method",
    ]
    group_cols = [c for c in group_cols if c in df.columns]

    metric_cols = [
        "train_mse",
        "test_mse",
        "variable_f1",
        "variable_auroc",
        "variable_auprc",
        "interaction_f1",
        "active_score_mean",
        "inactive_score_mean",
        "active_score_min",
        "inactive_score_max",
    ]
    metric_cols = [c for c in metric_cols if c in df.columns]

    summary = (
        df.groupby(group_cols, dropna=False)[metric_cols]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )

    summary.columns = [
        "_".join([str(x) for x in col if x != ""]).rstrip("_")
        for col in summary.columns
    ]

    return summary


def save_excel(all_df, summary_df, score_df, out_dir: Path):
    xlsx_path = out_dir / "stage1_summary.xlsx"

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        all_df.to_excel(writer, sheet_name="All Results", index=False)
        summary_df.to_excel(writer, sheet_name="Summary by Setting", index=False)

        if not score_df.empty:
            score_df.to_excel(writer, sheet_name="Variable Scores", index=False)

        key_cols = [
            "setting",
            "function",
            "dimension",
            "samples",
            "noise",
            "lamb",
            "model",
            "explain_method",
            "test_mse",
            "variable_f1",
            "variable_auroc",
            "variable_auprc",
            "interaction_f1",
        ]
        key_cols = [c for c in key_cols if c in all_df.columns]
        all_df[key_cols].to_excel(writer, sheet_name="Key Rows", index=False)

    print(f"[OK] Wrote {xlsx_path}")


def plot_variable_f1_by_dimension(df, out_dir: Path):
    if "variable_f1" not in df.columns or "dimension" not in df.columns:
        return

    sub = df.copy()
    sub = sub[sub["model"].astype(str).str.upper().eq("KAN")]
    sub = sub[pd.notna(sub["dimension"])]
    sub = sub[pd.notna(sub["variable_f1"])]

    if sub.empty:
        return

    grouped = (
        sub.groupby(["dimension", "explain_method"], dropna=False)["variable_f1"]
        .mean()
        .reset_index()
        .sort_values("dimension")
    )

    plt.figure(figsize=(7, 4.5))
    for method, g in grouped.groupby("explain_method"):
        plt.plot(g["dimension"], g["variable_f1"], marker="o", label=str(method))

    plt.xlabel("Input dimension")
    plt.ylabel("Mean variable recovery F1")
    plt.title("Variable recovery degrades with input dimension")
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "fig_variable_f1_by_dimension.png", dpi=200)
    plt.close()


def plot_test_mse_by_dimension(df, out_dir: Path):
    if "test_mse" not in df.columns or "dimension" not in df.columns:
        return

    sub = df.copy()
    sub = sub[sub["model"].astype(str).str.upper().eq("KAN")]
    sub = sub[pd.notna(sub["dimension"])]
    sub = sub[pd.notna(sub["test_mse"])]

    if sub.empty:
        return

    grouped = (
        sub.groupby(["dimension", "explain_method"], dropna=False)["test_mse"]
        .mean()
        .reset_index()
        .sort_values("dimension")
    )

    plt.figure(figsize=(7, 4.5))
    for method, g in grouped.groupby("explain_method"):
        plt.plot(g["dimension"], g["test_mse"], marker="o", label=str(method))

    plt.xlabel("Input dimension")
    plt.ylabel("Mean test MSE")
    plt.title("Prediction error under increasing input dimension")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "fig_test_mse_by_dimension.png", dpi=200)
    plt.close()


def plot_variable_f1_by_sample_noise(df, out_dir: Path):
    required = {"samples", "noise", "variable_f1"}
    if not required.issubset(df.columns):
        return

    sub = df.copy()
    sub = sub[sub["model"].astype(str).str.upper().eq("KAN")]
    sub = sub[pd.notna(sub["samples"])]
    sub = sub[pd.notna(sub["noise"])]
    sub = sub[pd.notna(sub["variable_f1"])]

    if sub.empty:
        return

    if "core_interaction" in set(sub["function"].astype(str)):
        sub = sub[sub["function"].astype(str).eq("core_interaction")]

    if "perm" in set(sub["explain_method"].astype(str)):
        sub = sub[sub["explain_method"].astype(str).eq("perm")]

    grouped = (
        sub.groupby(["samples", "noise"], dropna=False)["variable_f1"]
        .mean()
        .reset_index()
        .sort_values(["samples", "noise"])
    )

    plt.figure(figsize=(7, 4.5))
    for samples, g in grouped.groupby("samples"):
        plt.plot(g["noise"], g["variable_f1"], marker="o", label=f"n={int(samples)}")

    plt.xlabel("Noise level")
    plt.ylabel("Mean variable recovery F1")
    plt.title("Sample size / noise sweep")
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "fig_variable_f1_by_sample_noise.png", dpi=200)
    plt.close()


def plot_interaction_f1_by_setting(df, out_dir: Path):
    if "interaction_f1" not in df.columns:
        return

    sub = df.copy()
    sub = sub[sub["model"].astype(str).str.upper().eq("KAN")]
    sub = sub[pd.notna(sub["interaction_f1"])]

    if sub.empty:
        return

    grouped = (
        sub.groupby(["setting", "explain_method"], dropna=False)["interaction_f1"]
        .mean()
        .reset_index()
    )

    pivot = grouped.pivot(index="setting", columns="explain_method", values="interaction_f1")
    pivot = pivot.sort_index()

    ax = pivot.plot(kind="bar", figsize=(10, 5))
    ax.set_ylabel("Mean interaction recovery F1")
    ax.set_title("Interaction recovery by setting")
    ax.set_ylim(-0.05, 1.05)
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "fig_interaction_f1_by_setting.png", dpi=200)
    plt.close()


def plot_score_distributions(score_df, out_dir: Path):
    if score_df.empty:
        return

    for source_file, sub0 in score_df.groupby("source_file"):
        for method, sub in sub0.groupby("explain_method" if "explain_method" in sub0.columns else lambda x: "all"):
            active = sub[sub["is_active"] == 1]["score"].dropna()
            inactive = sub[sub["is_active"] == 0]["score"].dropna()

            if active.empty or inactive.empty:
                continue

            plt.figure(figsize=(6, 4.5))
            plt.boxplot([active, inactive], labels=["Active", "Inactive"], showfliers=False)
            plt.ylabel("Importance score")
            plt.title(f"Score distribution\n{source_file}, {method}")
            plt.tight_layout()

            safe_source = source_file.replace(".csv", "").replace("/", "_")
            safe_method = str(method).replace("/", "_")
            plt.savefig(out_dir / f"fig_score_distribution_{safe_source}_{safe_method}.png", dpi=200)
            plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="results/stage1")
    parser.add_argument("--out", type=str, default="results/stage1_report")
    args = parser.parse_args()

    input_dir = Path(args.input)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_df = read_all_results(input_dir)
    score_df = read_score_files(input_dir)
    summary_df = make_summary(all_df)

    all_df.to_csv(out_dir / "stage1_all_rows.csv", index=False)
    summary_df.to_csv(out_dir / "stage1_summary_by_setting.csv", index=False)
    if not score_df.empty:
        score_df.to_csv(out_dir / "stage1_variable_scores_all.csv", index=False)

    save_excel(all_df, summary_df, score_df, out_dir)

    plot_variable_f1_by_dimension(all_df, out_dir)
    plot_test_mse_by_dimension(all_df, out_dir)
    plot_variable_f1_by_sample_noise(all_df, out_dir)
    plot_interaction_f1_by_setting(all_df, out_dir)
    plot_score_distributions(score_df, out_dir)

    print(f"[OK] Done. Outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()