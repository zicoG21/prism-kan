from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def read_csv(path: str) -> pd.DataFrame:
    p = ROOT / path
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def combine_kan_keygrid() -> pd.DataFrame:
    parts = []
    for path in [
        "results/workshop_review_tables/same_data_kan_stability_c025_d100_bootstrap_R20_10seed/same_data_kan_stability_detail.csv",
        "results/workshop_review_tables/same_data_kan_stability_c025_d100_n_grid_R20_10seed/same_data_kan_stability_detail.csv",
        "results/workshop_review_tables/same_data_kan_stability_c025_d100_anchor_R20_seeds10_29/same_data_kan_stability_detail.csv",
        "results/workshop_review_tables/same_data_kan_stability_c025_d100_keygrid_R20_seeds10_29/same_data_kan_stability_detail.csv",
        "results/workshop_review_tables/same_data_kan_stability_c025_d100_keygrid_R20_seeds10_49/same_data_kan_stability_detail.csv",
        "results/workshop_review_tables/same_data_kan_stability_c025_d100_n640_R20_seeds10_29/same_data_kan_stability_detail.csv",
        "results/workshop_review_tables/same_data_kan_stability_c025_d100_n768_R20_seeds10_29/same_data_kan_stability_detail.csv",
        "results/workshop_review_tables/same_data_kan_stability_c025_d100_n896_R20_seeds10_29/same_data_kan_stability_detail.csv",
        "results/workshop_review_tables/same_data_kan_stability_c025_d100_n1280_R20_seeds10_29/same_data_kan_stability_detail.csv",
    ]:
        df = read_csv(path)
        if len(df):
            parts.append(df)
    if not parts:
        return pd.DataFrame()
    detail = pd.concat(parts, ignore_index=True)
    detail = detail[detail["samples"].isin([512, 640, 768, 896, 1024, 1280])].copy()
    # If the same seed/sample is present in multiple partial runs, keep the
    # newest row from later files.
    detail = detail.drop_duplicates(["samples", "outer_seed", "method"], keep="last")
    rows = []
    for (samples, method), g in detail.groupby(["samples", "method"], dropna=False):
        rows.append(
            {
                "samples": int(samples),
                "method": method,
                "num_runs": int(len(g)),
                "exact_support_successes": int(g["screen_contains_all_true_vars"].sum()),
                "endpoint_successes": int(g["screen_contains_all_interaction_endpoints"].sum()),
                "top1_successes": int(g["interaction_f1"].fillna(0).sum()),
                "endpoint_recall_mean": float(g["screen_interaction_endpoint_recall"].mean()),
                "var_recall_mean": float(g["screen_true_var_recall"].mean()),
                "refit_mse_mean": float(g["refit_test_mse"].mean()),
                "runtime_sec_mean": float(g["runtime_sec"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["samples", "method"])


def main() -> None:
    out_dir = ROOT / "results/workshop_review_tables/reinforcement_summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    kan = combine_kan_keygrid()
    if len(kan):
        kan.to_csv(out_dir / "kan_same_data_keygrid_combined.csv", index=False)
        print("\nKAN same-data keygrid")
        print(kan.to_string(index=False))

    hsic = read_csv(
        "results/interaction_baselines/residual_rff_hsic_pair_screen_c025_d100_keygrid_50seed/residual_rff_hsic_pair_screen_summary.csv"
    )
    if len(hsic):
        cols = [
            "samples",
            "num_runs",
            "top1_successes",
            "endpoint_recall_at_top_pairs_mean",
            "true_interaction_rank_mean_mean",
            "runtime_sec_mean",
        ]
        view = hsic[[c for c in cols if c in hsic.columns]].copy()
        view.to_csv(out_dir / "residual_rff_hsic_keygrid.csv", index=False)
        print("\nResidual RFF-HSIC keygrid")
        print(view.to_string(index=False))

    surf = read_csv(
        "results/formula_surface_sanity/true_support_core_c025_n512_1024_30seed/true_support_formula_surface_summary.csv"
    )
    if len(surf):
        cols = [
            "samples",
            "num_runs",
            "test_mse_mean",
            "surface_corr_product_mean",
            "surface_r2_product_mean",
            "surface_r2_ge_090_successes",
            "surface_residual_ratio_mean",
        ]
        view = surf[[c for c in cols if c in surf.columns]].copy()
        view.to_csv(out_dir / "true_support_surface_sanity.csv", index=False)
        print("\nTrue-support formula surface sanity")
        print(view.to_string(index=False))


if __name__ == "__main__":
    main()
