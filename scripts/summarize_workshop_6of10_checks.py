from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path("results/workshop_review_tables")
OUT = ROOT / "workshop_6of10_checks"
SENS_ROOT = ROOT / "kan_probe_hparam_sensitivity_c1_d1000_n1024"
ROBUST_ROOT = ROOT / "kan_probe_noise_corr_c025_d100_n1024"
RESID_ROOT = ROOT / "residual_pair_screen_noise_corr_c025_d100_n1024"


def load_sensitivity(root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(root.glob("*/support_sensitivity_summary.csv")):
        df = pd.read_csv(path)
        if df.empty:
            continue
        df.insert(0, "config", path.parent.name)
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def load_residual(root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(root.glob("*/residual_pair_screen_summary.csv")):
        df = pd.read_csv(path)
        if df.empty:
            continue
        df.insert(0, "config", path.parent.name)
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def select_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df[[c for c in cols if c in df.columns]].copy()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    hparam = load_sensitivity(SENS_ROOT)
    hparam.to_csv(OUT / "kan_hparam_sensitivity_all.csv", index=False)
    if not hparam.empty:
        hparam_focus = hparam[
            (hparam["method"].astype(str) == "feature_edge_hybrid")
            & (pd.to_numeric(hparam["top_m"], errors="coerce").isin([6, 20, 100, 250, 500, 1000]))
        ].copy()
        hparam_focus = select_cols(
            hparam_focus,
            [
                "config",
                "width_hidden",
                "grid",
                "lamb",
                "probe_steps",
                "top_m",
                "screen_contains_all_interaction_endpoints_mean",
                "screen_interaction_endpoint_recall_mean",
                "screen_contains_true_interactions_mean",
                "true_endpoint_rank_worst_mean",
                "endpoint_minus_max_nuisance_mean",
                "num_support_evals",
            ],
        )
        hparam_focus.to_csv(OUT / "kan_hparam_sensitivity_focus.csv", index=False)

    robust = load_sensitivity(ROBUST_ROOT)
    robust.to_csv(OUT / "kan_noise_corr_robustness_all.csv", index=False)
    if not robust.empty:
        robust_focus = robust[
            (robust["method"].astype(str) == "feature_edge_hybrid")
            & (pd.to_numeric(robust["top_m"], errors="coerce").isin([4, 8, 20]))
        ].copy()
        robust_focus = select_cols(
            robust_focus,
            [
                "config",
                "noise",
                "nuisance_correlation",
                "n_correlated_proxies",
                "top_m",
                "screen_contains_all_true_vars_mean",
                "screen_contains_all_interaction_endpoints_mean",
                "screen_interaction_endpoint_recall_mean",
                "screen_contains_true_interactions_mean",
                "true_endpoint_rank_worst_mean",
                "endpoint_minus_max_nuisance_mean",
                "num_support_evals",
            ],
        )
        robust_focus.to_csv(OUT / "kan_noise_corr_robustness_focus.csv", index=False)

    residual = load_residual(RESID_ROOT)
    residual.to_csv(OUT / "residual_noise_corr_robustness_all.csv", index=False)
    if not residual.empty:
        residual_focus = select_cols(
            residual,
            [
                "config",
                "noise",
                "nuisance_correlation",
                "n_correlated_proxies",
                "top1_pair_accuracy_mean",
                "top1_successes",
                "num_runs",
                "true_interaction_rank_worst_mean",
                "max_false_pair_score_mean",
                "runtime_sec_mean",
            ],
        )
        residual_focus.to_csv(OUT / "residual_noise_corr_robustness_focus.csv", index=False)

    lines = ["# Workshop 6/10 Checks", ""]
    if not hparam.empty:
        h6 = hparam[
            (hparam["method"].astype(str) == "feature_edge_hybrid")
            & (pd.to_numeric(hparam["top_m"], errors="coerce") == 6)
        ].copy()
        h6 = h6.sort_values(["width_hidden", "grid", "lamb"])
        lines.extend(["## KAN hyperparameter sensitivity, d=1000,c=1,n=1024, top-m=6", ""])
        for _, row in h6.iterrows():
            lines.append(
                "- {config}: endpoint all={ep:.2f}, pair-retained={pair:.2f}, worst-rank={rank:.1f}, margin={margin:.3f}, evals={n}".format(
                    config=row["config"],
                    ep=float(row["screen_contains_all_interaction_endpoints_mean"]),
                    pair=float(row["screen_contains_true_interactions_mean"]),
                    rank=float(row["true_endpoint_rank_worst_mean"]),
                    margin=float(row["endpoint_minus_max_nuisance_mean"]),
                    n=int(row["num_support_evals"]),
                )
            )
        lines.append("")
    if not robust.empty:
        r4 = robust[
            (robust["method"].astype(str) == "feature_edge_hybrid")
            & (pd.to_numeric(robust["top_m"], errors="coerce") == 4)
        ].copy()
        r4 = r4.sort_values(["noise", "nuisance_correlation"])
        lines.extend(["## KAN-FE robustness, d=100,c=.25,n=1024, top-m=4", ""])
        for _, row in r4.iterrows():
            lines.append(
                "- noise={noise:g}, rho={rho:g}: exact-vars={var:.2f}, endpoints={ep:.2f}, worst-rank={rank:.1f}, margin={margin:.3f}".format(
                    noise=float(row["noise"]),
                    rho=float(row["nuisance_correlation"]),
                    var=float(row["screen_contains_all_true_vars_mean"]),
                    ep=float(row["screen_contains_all_interaction_endpoints_mean"]),
                    rank=float(row["true_endpoint_rank_worst_mean"]),
                    margin=float(row["endpoint_minus_max_nuisance_mean"]),
                )
            )
        lines.append("")
    if not residual.empty:
        res = residual.sort_values(["noise", "nuisance_correlation"])
        lines.extend(["## Residual spline pair screen robustness, d=100,c=.25,n=1024", ""])
        for _, row in res.iterrows():
            lines.append(
                "- noise={noise:g}, rho={rho:g}: top-1={succ:.0f}/{runs:.0f}, rank={rank:.1f}, runtime={rt:.2f}s".format(
                    noise=float(row["noise"]),
                    rho=float(row["nuisance_correlation"]),
                    succ=float(row["top1_successes"]),
                    runs=float(row["num_runs"]),
                    rank=float(row["true_interaction_rank_worst_mean"]),
                    rt=float(row["runtime_sec_mean"]),
                )
            )
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((OUT / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
