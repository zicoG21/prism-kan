from __future__ import annotations

from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def read(path):
    return pd.read_csv(path)


def first_float(df, col):
    if df.empty or col not in df.columns:
        return np.nan
    val = df[col].iloc[0]
    return float(val) if pd.notna(val) else np.nan


def mean_delta(df, row_type, target=None, pair=None, setting_label=None, screen_method=None, top_m=None):
    sub = df[df["row_type"] == row_type].copy()
    if target is not None:
        sub = sub[sub["target"].astype(str) == target]
    if pair is not None:
        sub = sub[sub["pair"].astype(str) == pair]
    if setting_label is not None and "setting_label" in sub.columns:
        sub = sub[sub["setting_label"].astype(str) == setting_label]
    if screen_method is not None and "screen_method" in sub.columns:
        sub = sub[sub["screen_method"].astype(str) == screen_method]
    if top_m is not None and "top_m" in sub.columns:
        sub = sub[pd.to_numeric(sub["top_m"], errors="coerce") == top_m]
    vals = pd.to_numeric(sub["delta_mse_mean"], errors="coerce").dropna()
    return float(vals.mean()) if len(vals) else np.nan


def load_data(root):
    return {
        "failure": read(root / "failure_attribution" / "failure_attribution_table.csv"),
        "stab_var": read(root / "stability_selected" / "stability_selected_variable_summary.csv"),
        "stab_int": read(root / "stability_selected" / "stability_selected_interaction_summary.csv"),
        "screened": read(root / "screened_kan" / "figures" / "combined_screened_summary.csv"),
        "raw_core_path": read(root / "path_intervention" / "core_d100_path_summary.csv"),
        "raw_proxy_path": read(root / "path_intervention" / "correlated_proxy_d100_path_summary.csv"),
        "screened_path": read(root / "screened_kan" / "figures" / "combined_screened_path_summary.csv"),
    }


def make_figures(data, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    failure = data["failure"]
    stab_var = data["stab_var"]
    stab_int = data["stab_int"]
    screened = data["screened"]
    raw_path_core = data["raw_core_path"]
    raw_path_proxy = data["raw_proxy_path"]
    screened_path = data["screened_path"]

    def get_raw(setting, metric):
        return first_float(failure[failure["setting"] == setting], metric)

    def get_stab_var(source):
        sub = stab_var[(stab_var["source_file"] == source) & (stab_var["explain_method"] == "grad") & (stab_var["aggregation"] == "selection_frequency_topk")]
        return first_float(sub, "stable_f1")

    def get_stab_int(source):
        sub = stab_int[(stab_int["source_file"] == source) & (stab_int["explain_method"] == "grad") & (stab_int["aggregation"] == "interaction_frequency_topk")]
        return first_float(sub, "stable_interaction_f1")

    def get_screened(label, top_m, metric):
        sub = screened[(screened["setting_label"] == label) & (screened["screen_method"] == "rf") & (screened["top_m"] == top_m) & (screened["explain_method"] == "grad")]
        return first_float(sub, metric)

    rows = []
    settings = [
        ("core d=100", "highdim_core_d100", "core_d100_noise00_interactions.csv", "core_d100"),
        ("core c=5 d=100", "strong_interaction_c5_d100", "core_d100_c5.csv", "core_c5_d100"),
        ("proxy d=100", "correlated_proxy_d100", "correlated_proxy_d100_n512_noise005.csv", "proxy_d100"),
    ]
    for label, raw_setting, stab_source, screened_label in settings:
        rows.append({"setting": label, "method": "Raw KAN", "variable_f1": get_raw(raw_setting, "kan_var_f1"), "interaction_f1": get_raw(raw_setting, "kan_int_f1")})
        rows.append({"setting": label, "method": "Stability-selected", "variable_f1": get_stab_var(stab_source), "interaction_f1": get_stab_int(stab_source)})
        for m in [10, 20]:
            rows.append({"setting": label, "method": f"RF-screened M={m}", "variable_f1": get_screened(screened_label, m, "variable_f1_mean"), "interaction_f1": get_screened(screened_label, m, "interaction_f1_mean")})
    recovery = pd.DataFrame(rows)
    recovery.to_csv(out_dir / "master_recovery_summary.csv", index=False)

    def plot_recovery(metric, title, outfile):
        settings_order = recovery["setting"].drop_duplicates().tolist()
        methods_order = ["Raw KAN", "Stability-selected", "RF-screened M=10", "RF-screened M=20"]
        x = np.arange(len(settings_order))
        width = 0.18
        plt.figure(figsize=(9, 5))
        for idx, method in enumerate(methods_order):
            vals = []
            for setting in settings_order:
                hit = recovery[(recovery["setting"] == setting) & (recovery["method"] == method)]
                vals.append(first_float(hit, metric))
            plt.bar(x + (idx - (len(methods_order)-1)/2)*width, vals, width=width, label=method)
        plt.ylim(0, 1.08)
        plt.ylabel("Mean F1")
        plt.xticks(x, settings_order)
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / outfile, dpi=250)
        plt.close()

    plot_recovery("variable_f1", "Variable recovery: raw, stability-selected, and screened KAN", "master_variable_recovery.png")
    plot_recovery("interaction_f1", "Interaction recovery: screening rescues structure when retained", "master_interaction_recovery.png")

    features = ["x0", "x1", "x2", "x3"]
    methods = ["Raw KAN", "RF-screened M=10", "RF-screened M=20"]
    vals = {m: [] for m in methods}
    for f in features:
        vals["Raw KAN"].append(mean_delta(raw_path_core, "feature_path", target=f))
        vals["RF-screened M=10"].append(mean_delta(screened_path, "feature_path", target=f, setting_label="core_d100_rf", screen_method="rf", top_m=10))
        vals["RF-screened M=20"].append(mean_delta(screened_path, "feature_path", target=f, setting_label="core_d100_rf", screen_method="rf", top_m=20))
    x = np.arange(len(features))
    width = 0.25
    plt.figure(figsize=(8,5))
    for idx, method in enumerate(methods):
        plt.bar(x + (idx-1)*width, vals[method], width=width, label=method)
    plt.axhline(0, linewidth=1)
    plt.xticks(x, features)
    plt.ylabel("Mean test MSE increase")
    plt.xlabel("Deleted KAN feature path")
    plt.title("Screening restores core d=100 feature-path reliance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "master_core_feature_path_reliance.png", dpi=250)
    plt.close()

    pairs = ["(2, 3)", "(0, 1)"]
    pair_vals = {m: [] for m in methods}
    for p in pairs:
        pair_vals["Raw KAN"].append(mean_delta(raw_path_core, "feature_pair_path", pair=p))
        pair_vals["RF-screened M=10"].append(mean_delta(screened_path, "feature_pair_path", pair=p, setting_label="core_d100_rf", screen_method="rf", top_m=10))
        pair_vals["RF-screened M=20"].append(mean_delta(screened_path, "feature_pair_path", pair=p, setting_label="core_d100_rf", screen_method="rf", top_m=20))
    x = np.arange(len(pairs))
    plt.figure(figsize=(7,5))
    for idx, method in enumerate(methods):
        plt.bar(x + (idx-1)*width, pair_vals[method], width=width, label=method)
    plt.axhline(0, linewidth=1)
    plt.xticks(x, pairs)
    plt.ylabel("Mean test MSE increase")
    plt.xlabel("Deleted KAN pair path")
    plt.title("Screening restores true interaction-path reliance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "master_core_pair_path_reliance.png", dpi=250)
    plt.close()

    proxy_features = ["x0", "x2", "x3", "x4"]
    proxy_methods = ["Raw proxy KAN", "RF-screened M=10", "RF-screened M=20"]
    proxy_vals = {m: [] for m in proxy_methods}
    for f in proxy_features:
        proxy_vals["Raw proxy KAN"].append(mean_delta(raw_path_proxy, "feature_path", target=f))
        proxy_vals["RF-screened M=10"].append(mean_delta(screened_path, "feature_path", target=f, setting_label="proxy_d100_rf", screen_method="rf", top_m=10))
        proxy_vals["RF-screened M=20"].append(mean_delta(screened_path, "feature_path", target=f, setting_label="proxy_d100_rf", screen_method="rf", top_m=20))
    x = np.arange(len(proxy_features))
    plt.figure(figsize=(8,5))
    for idx, method in enumerate(proxy_methods):
        plt.bar(x + (idx-1)*width, proxy_vals[method], width=width, label=method)
    plt.axhline(0, linewidth=1)
    plt.xticks(x, proxy_features)
    plt.ylabel("Mean test MSE increase")
    plt.xlabel("Deleted KAN feature path")
    plt.title("Proxy setting: true paths restored, proxy reliance remains")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "master_proxy_feature_path_reliance.png", dpi=250)
    plt.close()

    proxy_pairs = ["(2, 3)", "(0, 4)"]
    proxy_pair_vals = {m: [] for m in proxy_methods}
    for p in proxy_pairs:
        proxy_pair_vals["Raw proxy KAN"].append(mean_delta(raw_path_proxy, "feature_pair_path", pair=p))
        proxy_pair_vals["RF-screened M=10"].append(mean_delta(screened_path, "feature_pair_path", pair=p, setting_label="proxy_d100_rf", screen_method="rf", top_m=10))
        proxy_pair_vals["RF-screened M=20"].append(mean_delta(screened_path, "feature_pair_path", pair=p, setting_label="proxy_d100_rf", screen_method="rf", top_m=20))
    x = np.arange(len(proxy_pairs))
    plt.figure(figsize=(7,5))
    for idx, method in enumerate(proxy_methods):
        plt.bar(x + (idx-1)*width, proxy_pair_vals[method], width=width, label=method)
    plt.axhline(0, linewidth=1)
    plt.xticks(x, proxy_pairs)
    plt.ylabel("Mean test MSE increase")
    plt.xlabel("Deleted KAN pair path")
    plt.title("Proxy pair-path reliance after screening")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "master_proxy_pair_path_reliance.png", dpi=250)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="results")
    parser.add_argument("--out_dir", default="results/master_figures")
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    data = load_data(root)
    make_figures(data, out_dir)
    print(f"Wrote master figures to {out_dir}")


if __name__ == "__main__":
    main()
