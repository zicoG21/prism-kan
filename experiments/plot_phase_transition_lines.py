import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


C_MAP = {
    "core_interaction_c01": 0.10,
    "core_interaction_c025": 0.25,
    "core_interaction_c05": 0.50,
    "core_interaction_c1": 1.00,
}

METHODS = ["raw", "rf", "oracle_support", "random", "exclude_interaction"]

LABELS = {
    "raw": "Raw KAN",
    "rf": "RF-screened KAN",
    "oracle_support": "Oracle-support KAN",
    "random": "Random support",
    "exclude_interaction": "Exclude interaction",
}

def parse_file(path):
    name = Path(path).name
    m = re.search(r"(core_interaction_c(?:01|025|05|1))_n(\d+)_d(\d+)", name)
    if not m:
        return None
    return m.group(1), int(m.group(2)), int(m.group(3))

rows = []
for p in glob.glob("results/hard_regime/summaries/*_summary.csv"):
    parsed = parse_file(p)
    if parsed is None:
        continue
    fn, n, d = parsed
    df = pd.read_csv(p)
    df["function_tag"] = fn
    df["c"] = C_MAP[fn]
    df["n"] = n
    df["d"] = d

    for col in ["test_mse", "variable_f1", "interaction_f1"]:
        mean_col = col + "_mean"
        if mean_col in df.columns:
            df[col] = df[mean_col]

    rows.append(df)

out = pd.concat(rows, ignore_index=True)
Path("results/hard_regime/paper_figures").mkdir(parents=True, exist_ok=True)
out.to_csv("results/hard_regime/paper_figures/phase_transition_source.csv", index=False)

for d in sorted(out["d"].unique()):
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.8), sharey=True)

    for ax, method in zip(axes, ["raw", "rf", "oracle_support"]):
        sub_m = out[(out["d"] == d) & (out["screen_mode"] == method)]

        for c in sorted(sub_m["c"].unique()):
            sub = sub_m[sub_m["c"] == c].sort_values("n")
            ax.plot(
                sub["n"],
                sub["interaction_f1"],
                marker="o",
                linewidth=2.2,
                label=f"c={c:g}",
            )

        ax.set_title(LABELS[method])
        ax.set_xlabel("Sample size $n$")
        ax.set_xscale("log", base=2)
        ax.set_xticks([128, 256, 512, 1024])
        ax.set_xticklabels(["128", "256", "512", "1024"])
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Interaction F1")
    axes[-1].legend(title="Interaction strength", loc="lower right", frameon=True)

    fig.suptitle(f"Formula-fidelity phase transition on core interaction, d={d}", y=1.04)
    fig.tight_layout()

    pdf = f"results/hard_regime/paper_figures/phase_transition_lines_d{d}.pdf"
    png = f"results/hard_regime/paper_figures/phase_transition_lines_d{d}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=400)
    print("saved", pdf)
    plt.close(fig)
