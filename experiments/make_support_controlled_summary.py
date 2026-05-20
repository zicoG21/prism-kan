from __future__ import annotations

from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


MODE_ORDER = ["random", "oracle_support", "exclude_interaction", "rf", "rf_exclude_interaction"]
MODE_LABELS = {
    "random": "Random",
    "oracle_support": "Oracle support",
    "exclude_interaction": "Exclude interaction",
    "rf": "RF",
    "rf_exclude_interaction": "RF excl. interaction",
}


def load_detail(root: Path) -> pd.DataFrame:
    files = {
        "core d=100": root / "support_controlled" / "core_d100_support_controlled_summary.csv",
        "core c=5 d=100": root / "support_controlled" / "core_c5_d100_support_controlled_summary.csv",
        "proxy d=100": root / "support_controlled" / "correlated_proxy_d100_support_controlled_summary.csv",
    }

    rows = []
    for setting, path in files.items():
        df = pd.read_csv(path)
        if "explain_method" in df.columns:
            df = df[df["explain_method"].astype(str) == "grad"].copy()
        for _, r in df.iterrows():
            mode = str(r["screen_mode"])
            if mode not in MODE_ORDER:
                continue
            rows.append({
                "setting": setting,
                "screen_mode": mode,
                "screen_mode_label": MODE_LABELS[mode],
                "top_m": int(r["top_m"]),
                "support_retained": float(r.get("screen_contains_all_true_vars_mean", np.nan)),
                "interaction_endpoints_retained": float(r.get("screen_contains_true_interactions_mean", np.nan)),
                "test_mse": float(r.get("test_mse_mean", np.nan)),
                "variable_f1": float(r.get("variable_f1_mean", np.nan)),
                "interaction_f1": float(r.get("interaction_f1_mean", np.nan)),
            })
    return pd.DataFrame(rows).sort_values(["setting", "screen_mode", "top_m"])


def make_mechanism_table(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for setting in detail["setting"].drop_duplicates():
        row = {"setting": setting}
        for mode in MODE_ORDER:
            sub = detail[(detail["setting"] == setting) & (detail["screen_mode"] == mode)].copy()
            if sub.empty:
                for suffix in ["best_m", "support", "var_f1", "int_f1"]:
                    row[f"{mode}_{suffix}"] = np.nan
                continue
            sub = sub.sort_values(["interaction_f1", "variable_f1", "top_m"], ascending=[False, False, True])
            best = sub.iloc[0]
            row[f"{mode}_best_m"] = int(best["top_m"])
            row[f"{mode}_support"] = best["support_retained"]
            row[f"{mode}_var_f1"] = best["variable_f1"]
            row[f"{mode}_int_f1"] = best["interaction_f1"]
        rows.append(row)
    return pd.DataFrame(rows)


def plot_best(mechanism: pd.DataFrame, metric: str, title: str, ylabel: str, out_path: Path):
    settings = mechanism["setting"].tolist()
    x = np.arange(len(settings))
    width = 0.15

    plt.figure(figsize=(10, 5.5))
    for idx, mode in enumerate(MODE_ORDER):
        vals = [float(mechanism.loc[mechanism["setting"] == s, f"{mode}_{metric}"].iloc[0]) for s in settings]
        plt.bar(x + (idx - (len(MODE_ORDER)-1)/2) * width, vals, width=width, label=MODE_LABELS[mode])
    plt.ylim(0, 1.08)
    plt.ylabel(ylabel)
    plt.xticks(x, settings)
    plt.title(title)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=250)
    plt.close()


def plot_by_m(detail: pd.DataFrame, out_path: Path):
    settings = detail["setting"].drop_duplicates().tolist()
    fig, axes = plt.subplots(1, len(settings), figsize=(5 * len(settings), 4.8), sharey=True)
    if len(settings) == 1:
        axes = [axes]

    for ax, setting in zip(axes, settings):
        sub_setting = detail[detail["setting"] == setting].copy()
        for mode in MODE_ORDER:
            sub = sub_setting[sub_setting["screen_mode"] == mode].sort_values("top_m")
            if not sub.empty:
                ax.plot(sub["top_m"], sub["interaction_f1"], marker="o", label=MODE_LABELS[mode])
        ax.set_title(setting)
        ax.set_xlabel("M")
        ax.set_xticks([10, 20, 30])
        ax.set_ylim(0, 1.08)
    axes[0].set_ylabel("Interaction F1")
    axes[-1].legend(fontsize=8, loc="lower right")
    fig.suptitle("Interaction recovery across screening budget M")
    fig.tight_layout()
    fig.savefig(out_path, dpi=250)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="results")
    parser.add_argument("--out_dir", default="results/support_controlled/clean_summary")
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    detail = load_detail(root)
    mechanism = make_mechanism_table(detail)

    detail.to_csv(out_dir / "support_controlled_detail_table.csv", index=False)
    mechanism.to_csv(out_dir / "support_controlled_mechanism_table.csv", index=False)

    compact = []
    for _, r in mechanism.iterrows():
        compact.append({
            "Setting": r["setting"],
            "Random Int F1": r["random_int_f1"],
            "Oracle Support Int F1": r["oracle_support_int_f1"],
            "Exclude Interaction Int F1": r["exclude_interaction_int_f1"],
            "RF Int F1": r["rf_int_f1"],
            "RF Excl. Interaction Int F1": r["rf_exclude_interaction_int_f1"],
            "Main takeaway": (
                "Support retention is sufficient and necessary"
                if r["oracle_support_int_f1"] >= 0.65 and r["exclude_interaction_int_f1"] <= 0.05
                else "Support retention helps, but proxy/confounding remains"
            ),
        })
    pd.DataFrame(compact).to_csv(out_dir / "support_controlled_compact_table.csv", index=False)

    plot_best(
        mechanism,
        metric="int_f1",
        title="Support-controlled ablation: interaction recovery requires true interaction variables",
        ylabel="Best interaction F1 over M",
        out_path=out_dir / "support_controlled_interaction_f1_clean.png",
    )
    plot_best(
        mechanism,
        metric="var_f1",
        title="Support-controlled ablation: variable recovery",
        ylabel="Best variable F1 over M",
        out_path=out_dir / "support_controlled_variable_f1_clean.png",
    )
    plot_by_m(detail, out_dir / "support_controlled_interaction_f1_by_m.png")

    print(f"Wrote clean support-controlled summary to {out_dir}")


if __name__ == "__main__":
    main()
