from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "workshop_review_tables" / "highdim_prediction_clean_case"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    probe_path = ROOT / "results" / "innovation_loop" / "strict_validation_20260526_011917" / "probe_cache.csv"
    support_path = ROOT / "results" / "innovation_loop" / "final_candidate_figures" / "highdim_support_failure_diagnostics_summary.csv"
    residual_path = ROOT / "results" / "interaction_baselines" / "residual_pair_screen_c025_d1000_n2048_10seed" / "residual_pair_screen_summary.csv"

    probes = pd.read_csv(probe_path)
    support = pd.read_csv(support_path)
    residual = pd.read_csv(residual_path)

    setting = {
        "function": "core_interaction_c025",
        "samples": 2048,
        "dimension": 1000,
    }

    probe_rows = probes[
        (probes["function"] == setting["function"])
        & (probes["samples"] == setting["samples"])
        & (probes["dimension"] == setting["dimension"])
        & (probes["status"] == "ok")
    ].copy()
    support_rows = support[
        (support["function"] == setting["function"])
        & (support["samples"] == setting["samples"])
        & (support["dimension"] == setting["dimension"])
        & (support["method"] == "feature_edge_hybrid")
        & (support["top_m"] == 6)
    ].copy()
    residual_rows = residual[
        (residual["function"] == setting["function"])
        & (residual["samples"] == setting["samples"])
        & (residual["dimension"] == setting["dimension"])
    ].copy()

    if probe_rows.empty:
        raise RuntimeError(f"No probe rows found in {probe_path}")
    if support_rows.empty:
        raise RuntimeError(f"No support rows found in {support_path}")
    if residual_rows.empty:
        raise RuntimeError(f"No residual rows found in {residual_path}")

    srow = support_rows.iloc[0]
    rrow = residual_rows.iloc[0]
    out = pd.DataFrame(
        [
            {
                **setting,
                "c": 0.25,
                "kan_full_probe_count": int(len(probe_rows)),
                "kan_full_probe_mse_mean": float(probe_rows["test_mse"].mean()),
                "kan_full_probe_mse_std": float(probe_rows["test_mse"].std()),
                "kan_full_probe_mse_min": float(probe_rows["test_mse"].min()),
                "kan_full_probe_mse_max": float(probe_rows["test_mse"].max()),
                "kan_fe_support_evals": int(srow["num_runs"]),
                "kan_fe_top_m": int(srow["top_m"]),
                "kan_fe_endpoint_successes": int(round(float(srow["pair_retained_mean"]) * float(srow["num_runs"]))),
                "kan_fe_endpoint_success_rate": float(srow["pair_retained_mean"]),
                "kan_fe_top1_success_rate": float(srow["top1_pair_accuracy_mean"]),
                "kan_fe_worst_endpoint_rank": float(srow["endpoint_rank_worst_mean"]),
                "kan_fe_endpoint_score_margin": float(srow["support_score_margin_mean"]),
                "residual_screen_runs": int(rrow["num_runs"]),
                "residual_screen_top1_successes": int(rrow["top1_successes"]),
                "residual_screen_top1_success_rate": float(rrow["top1_pair_accuracy_mean"]),
            }
        ]
    )
    out.to_csv(OUT_DIR / "highdim_prediction_clean_case_summary.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
