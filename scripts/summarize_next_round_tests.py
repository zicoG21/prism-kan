from __future__ import annotations

from pathlib import Path
import pandas as pd

ROOT = Path("results/next_round")

def print_table(path: Path, title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    if not path.exists():
        print(f"Missing: {path}")
        return
    df = pd.read_csv(path)
    cols = [
        "function",
        "screen_mode",
        "dimension",
        "test_mse_mean",
        "test_mse_std",
        "variable_f1_mean",
        "interaction_f1_mean",
        "screen_contains_true_interactions_mean",
    ]
    cols = [c for c in cols if c in df.columns]
    if not cols:
        print(df.head().to_string(index=False))
    else:
        print(df[cols].to_string(index=False))

def main():
    print_table(
        ROOT / "feynman_interaction" / "feynman_interaction_summary.csv",
        "Stage 1: Feynman interaction validation",
    )

    pieces = []
    for p in sorted((ROOT / "dim_transition").glob("dim_*_summary.csv")):
        try:
            df = pd.read_csv(p)
            # Ensure dimension column exists even if script grouping changed.
            if "dimension" not in df.columns:
                d = int(p.stem.split("_")[1])
                df["dimension"] = d
            pieces.append(df)
        except Exception as e:
            print(f"Could not read {p}: {e}")
    if pieces:
        dim = pd.concat(pieces, ignore_index=True)
        out = ROOT / "dim_transition" / "dim_transition_all_summary.csv"
        dim.to_csv(out, index=False)
        cols = [
            "function",
            "screen_mode",
            "dimension",
            "test_mse_mean",
            "variable_f1_mean",
            "interaction_f1_mean",
            "screen_contains_true_interactions_mean",
        ]
        cols = [c for c in cols if c in dim.columns]
        print("\n" + "=" * 80)
        print("Stage 2: dimension transition")
        print("=" * 80)
        print(dim[cols].sort_values(["function", "dimension", "screen_mode"]).to_string(index=False))
        print(f"\nWrote combined dimension summary to {out}")
    else:
        print("\nNo dimension summaries found.")

    print_table(
        ROOT / "tuned_screened" / "tuned_screened_summary.csv",
        "Stage 3: tuned raw vs tuned screened KAN",
    )

    print("\nKey interpretation checklist:")
    print("1. Feynman interaction validation: selected_interactions should no longer be empty.")
    print("2. Dimension transition: does raw KAN degrade as nuisance dimensions increase?")
    print("3. Tuned screened KAN: does the same tuned configuration work when support is retained?")
    print("4. If tuned raw fails but tuned RF/oracle succeeds, support retention is a cleaner mechanism.")

if __name__ == "__main__":
    main()
