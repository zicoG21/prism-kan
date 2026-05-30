#!/usr/bin/env python3
"""Build a compact revision result digest from completed experiment CSVs.

The digest is intentionally paper-facing: each row states what role a result can
play in the manuscript instead of dumping every experimental output.  It is
safe to rerun after local or Great Lakes jobs finish.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    return df if len(df) else None


def fullkan_row(label: str, root: Path, role: str) -> dict | None:
    detail = read_csv(root / "full_kan_pair_anova_detail.csv")
    if detail is None:
        return None
    row = {
        "label": label,
        "source": str(root),
        "result_type": "full_model_anova",
        "paper_role": role,
        "runs": int(len(detail)),
        "samples": int(detail["samples"].iloc[0]) if "samples" in detail else np.nan,
        "dimension": int(detail["dimension"].iloc[0]) if "dimension" in detail else np.nan,
        "noise": float(detail["noise"].iloc[0]) if "noise" in detail else np.nan,
        "width_hidden": int(detail["width_hidden"].iloc[0]) if "width_hidden" in detail else np.nan,
        "grid": int(detail["grid"].iloc[0]) if "grid" in detail else np.nan,
        "steps": int(detail["steps"].iloc[0]) if "steps" in detail else np.nan,
        "mean_test_mse": float(detail["test_mse"].mean()) if "test_mse" in detail else np.nan,
        "median_test_mse": float(detail["test_mse"].median()) if "test_mse" in detail else np.nan,
        "true_pair_rank1": int(detail["true_pair_beats_candidates"].sum())
        if "true_pair_beats_candidates" in detail
        else np.nan,
        "mean_true_pair_rank": float(detail["true_pair_rank"].mean()) if "true_pair_rank" in detail else np.nan,
        "median_true_pair_rank": float(detail["true_pair_rank"].median()) if "true_pair_rank" in detail else np.nan,
        "mean_margin": float(detail["true_minus_max_false"].mean())
        if "true_minus_max_false" in detail
        else np.nan,
        "median_margin": float(detail["true_minus_max_false"].median())
        if "true_minus_max_false" in detail
        else np.nan,
    }
    for th in [0.005, 0.01, 0.02, 0.05]:
        if "test_mse" in detail and "true_pair_beats_candidates" in detail:
            sub = detail[detail["test_mse"] < th]
            row[f"mse_lt_{th:g}_runs"] = int(len(sub))
            row[f"mse_lt_{th:g}_rank1"] = int(sub["true_pair_beats_candidates"].sum()) if len(sub) else 0
    return row


def readout_row(label: str, path: Path, role: str) -> list[dict]:
    df = read_csv(path)
    if df is None:
        return []
    rows = []
    for _, rec in df.iterrows():
        method = str(rec.get("method", rec.get("selection_method", "")))
        rows.append(
            {
                "label": f"{label}:{method}",
                "source": str(path),
                "result_type": "exposed_readout",
                "paper_role": role,
                "runs": int(rec.get("trials", rec.get("runs", rec.get("num_outer_seeds", np.nan))))
                if not pd.isna(rec.get("trials", rec.get("runs", rec.get("num_outer_seeds", np.nan))))
                else np.nan,
                "samples": rec.get("samples", np.nan),
                "dimension": rec.get("dimension", np.nan),
                "noise": rec.get("noise", np.nan),
                "width_hidden": rec.get("width_hidden", np.nan),
                "grid": rec.get("grid", np.nan),
                "endpoint_successes": rec.get(
                    "endpoint_successes",
                    rec.get("screen_contains_all_interaction_endpoints_sum", rec.get("endpoints_retained", np.nan)),
                ),
                "endpoint_rate": rec.get(
                    "endpoint_rate",
                    rec.get("screen_contains_all_interaction_endpoints_mean", np.nan),
                ),
                "worst_endpoint_rank": rec.get(
                    "worst_endpoint_rank_mean",
                    rec.get("endpoint_rank_worst", rec.get("worst_rank", np.nan)),
                ),
                "endpoint_margin": rec.get(
                    "endpoint_margin_mean",
                    rec.get("margin", rec.get("endpoint_minus_max_nuisance", np.nan)),
                ),
            }
        )
    return rows


def semisynthetic_rows(path: Path, role: str) -> list[dict]:
    df = read_csv(path)
    if df is None:
        return []
    rows = []
    for _, rec in df.iterrows():
        dataset = rec.get("dataset", "")
        method = rec.get("method", "")
        rows.append(
            {
                "label": f"semisyn:{dataset}:c{rec.get('c', '')}:n{rec.get('samples', '')}:{method}",
                "source": str(path),
                "result_type": "semisynthetic_readout",
                "paper_role": role,
                "runs": rec.get("num_outer_seeds", np.nan),
                "samples": rec.get("samples", np.nan),
                "dimension": rec.get("dimension", np.nan),
                "noise": rec.get("noise", np.nan),
                "width_hidden": rec.get("width_hidden", np.nan),
                "grid": rec.get("grid", np.nan),
                "endpoint_successes": rec.get("endpoint_successes", np.nan),
                "endpoint_rate": rec.get("screen_contains_all_interaction_endpoints_mean", np.nan),
                "pair_successes": rec.get("support_pair_successes", np.nan),
                "residual_top1_successes": rec.get("residual_top1_successes", np.nan),
            }
        )
    return rows


def prune_rows(path: Path, role: str) -> list[dict]:
    df = read_csv(path)
    if df is None:
        return []
    rows = []
    for _, rec in df.iterrows():
        rows.append(
            {
                "label": f"prune:{rec.get('workflow', '')}:thr{rec.get('threshold', '')}:n{rec.get('samples', '')}",
                "source": str(path),
                "result_type": "prune_symbolic_smoke",
                "paper_role": role,
                "runs": rec.get("runs", np.nan),
                "samples": rec.get("samples", np.nan),
                "dimension": rec.get("dimension", np.nan),
                "mean_test_mse": rec.get("mean_full_mse", np.nan),
                "median_test_mse": rec.get("median_full_mse", np.nan),
                "median_support_size": rec.get("median_support_size", np.nan),
                "contains_all_true_vars": rec.get("contains_all_true_vars", np.nan),
                "endpoint_successes": rec.get("endpoint_contains", np.nan),
                "symbolic_formula_ok": rec.get("symbolic_formula_ok", np.nan),
                "errors": rec.get("errors", np.nan),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/revision"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/revision/final_digest"))
    args = parser.parse_args()

    rows: list[dict] = []
    fullkan_specs = [
        ("fullkan_clean_n512", args.root / "fullkan_anova_boundary/clean_w16_n512", "main_if_complete"),
        ("fullkan_clean_n640_60", args.root / "fullkan_anova_boundary/clean_w16_n640_60seed", "main_if_complete"),
        ("fullkan_clean_n768", args.root / "fullkan_anova_boundary/clean_w16_n768_fast", "main_or_appendix"),
        ("fullkan_clean_n896_60", args.root / "fullkan_anova_boundary/clean_w16_n896_60seed", "main_if_complete"),
        ("fullkan_clean_n1024", args.root / "fullkan_anova_boundary/clean_w16_n1024_fast", "main"),
        ("fullkan_grid_n512", args.root / "fullkan_anova_boundary/gridupdate_w16_n512", "main"),
        ("fullkan_grid_n640_60", args.root / "fullkan_anova_boundary/gridupdate_w16_n640_60seed", "main_if_complete"),
        ("fullkan_grid_n768", args.root / "fullkan_anova_boundary/gridupdate_w16_n768_fast", "main_or_appendix"),
        ("fullkan_grid_n896_60", args.root / "fullkan_anova_boundary/gridupdate_w16_n896_60seed", "main_if_complete"),
        ("fullkan_grid_n1024", args.root / "fullkan_anova_boundary/gridupdate_w16_n1024_fast", "main_or_appendix"),
        ("fullkan_noise010_n512", args.root / "fullkan_anova_boundary/noise010_w16_n512_fast", "appendix"),
        ("fullkan_noise010_n768", args.root / "fullkan_anova_boundary/noise010_w16_n768_fast", "appendix"),
        ("fullkan_noise010_n1024", args.root / "fullkan_anova_boundary/noise010_w16_n1024_fast", "appendix"),
    ]
    for label, root, role in fullkan_specs:
        row = fullkan_row(label, root, role)
        if row is not None:
            rows.append(row)

    readout_paths = [
        ("focused", args.root / "gpu_only_safe_queue/summary/focused_30seed_main_rows.csv", "main"),
        ("capacity", args.root / "gpu_parallel_c01_capacity/summary/focused_30seed_main_rows.csv", "appendix"),
        ("hparam", args.root / "d100_c025_hparam_sensitivity/summary/d100_c025_hparam_sensitivity_focus.csv", "appendix"),
        ("lowdim", args.root / "lowdim_phase_grid/summary/lowdim_phase_grid_focus.csv", "appendix"),
    ]
    for label, path, role in readout_paths:
        rows.extend(readout_row(label, path, role))

    semisyn_paths = [
        args.root / "semisynthetic_covariates_3h/semisynthetic_covariate_audit_summary.csv",
        args.root / "boundary_overnight_12h/semisynthetic/noise0/semisynthetic_covariate_audit_summary.csv",
        args.root / "gpu_4h_todo_pack/semisynthetic/dataset_split_10seed/semisynthetic_covariate_audit_summary.csv",
    ]
    for path in semisyn_paths:
        rows.extend(semisynthetic_rows(path, "secondary"))

    for path in sorted(args.root.glob("**/pykan_prune_symbolic_summary.csv")):
        rows.extend(prune_rows(path, "appendix_smoke"))

    digest = pd.DataFrame(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    digest_path = args.out_dir / "revision_result_digest.csv"
    digest.to_csv(digest_path, index=False)

    compact_cols = [
        col
        for col in [
            "label",
            "result_type",
            "paper_role",
            "runs",
            "samples",
            "dimension",
            "noise",
            "width_hidden",
            "endpoint_successes",
            "true_pair_rank1",
            "mean_test_mse",
            "median_test_mse",
            "median_true_pair_rank",
            "median_margin",
            "source",
        ]
        if col in digest.columns
    ]
    compact = digest[compact_cols].copy() if len(digest) else digest
    compact_path = args.out_dir / "revision_result_digest_compact.csv"
    compact.to_csv(compact_path, index=False)

    md_path = args.out_dir / "summary.md"
    with md_path.open("w", encoding="utf-8") as fh:
        fh.write("# Revision Result Digest\n\n")
        fh.write(f"Rows: {len(digest)}\n\n")
        if len(digest):
            fh.write("## By Result Type\n\n")
            by_type = digest.groupby(["result_type", "paper_role"]).size().reset_index(name="rows")
            fh.write("```text\n")
            fh.write(by_type.to_string(index=False))
            fh.write("\n```")
            fh.write("\n\n")
            fh.write("## Compact Preview\n\n")
            fh.write("```text\n")
            fh.write(compact.head(40).to_string(index=False))
            fh.write("\n```\n")

    print(f"Wrote {digest_path}")
    print(f"Wrote {compact_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
