from __future__ import annotations

import csv
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PY = os.environ.get("PYTHON", "/home/perzival/anaconda3/envs/prism/bin/python")
DEVICE = os.environ.get("FOCUSED_DEVICE", "cpu")
ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "results/revision/focused_30seed_core"
SEEDS = list(range(1200, 1230))


def stamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def write_line(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(line + "\n")


def base_cmd(setting: dict, seeds: list[int], summarize_only: bool = False) -> list[str]:
    cmd = [
        PY,
        "experiments/run_kan_probe_sensitivity.py",
        "--out_dir",
        str(OUT_ROOT / setting["label"]),
        "--function",
        setting["function"],
        "--samples",
        str(setting["samples"]),
        "--dimension",
        str(setting["dimension"]),
        "--test_samples",
        "2048",
        "--noise",
        str(setting.get("noise", 0.0)),
        "--seeds",
        *[str(s) for s in seeds],
        "--methods",
        "feature_stability_var",
        "feature_edge_hybrid",
        "--top_ms",
        "4",
        "6",
        "10",
        "20",
        "--width_hidden",
        str(setting["width"]),
        "--grid",
        "5",
        "--k",
        "3",
        "--lamb",
        "0.001",
        "--probe_steps",
        str(setting["steps"]),
        "--probe_variable_points",
        "512",
        "--pred_batch_size",
        "4096",
        "--device",
        DEVICE,
    ]
    if setting.get("update_grid", False):
        cmd.extend(["--update_grid", "--grid_update_num", "5"])
    if summarize_only:
        cmd.append("--summarize_existing_only")
    return cmd


def run_cmd(cmd: list[str], log_path: Path) -> int:
    with log_path.open("a") as log:
        proc = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    return int(proc.returncode)


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "STARTED_AT").write_text(stamp() + "\n")
    (OUT_ROOT / "PY_CONTROLLER_PID").write_text(str(os.getpid()) + "\n")
    (OUT_ROOT / "PY_CONTROLLER_DEVICE").write_text(DEVICE + "\n")
    progress = OUT_ROOT / "python_controller_progress.log"
    failed = OUT_ROOT / "python_controller_failed.csv"

    settings = []
    for n in (512, 1024):
        settings.extend(
            [
                dict(label=f"core_c025_d100_clean_w8_n{n}", function="core_interaction_c025", samples=n, dimension=100, width=8, steps=35),
                dict(label=f"core_c025_d100_clean_w16_n{n}", function="core_interaction_c025", samples=n, dimension=100, width=16, steps=75),
                dict(label=f"core_c025_d100_noise010_w16_n{n}", function="core_interaction_c025", samples=n, dimension=100, width=16, steps=75, noise=0.10),
                dict(label=f"core_c025_d100_gridupdate_w16_n{n}", function="core_interaction_c025", samples=n, dimension=100, width=16, steps=75, update_grid=True),
            ]
        )
    for n in (256, 512, 1024):
        settings.append(
            dict(label=f"nonmonotone_c01_d20_w8_n{n}", function="core_interaction_c01", samples=n, dimension=20, width=8, steps=35)
        )

    for setting in settings:
        label = setting["label"]
        write_line(progress, f"[{stamp()}] BEGIN {label}")
        log_path = OUT_ROOT / f"{label}.log"
        for seed in SEEDS:
            code = run_cmd(base_cmd(setting, [seed]), log_path)
            if code != 0:
                with failed.open("a", newline="") as f:
                    csv.writer(f).writerow([stamp(), label, seed, code])
        code = run_cmd(base_cmd(setting, SEEDS, summarize_only=True), log_path)
        if code != 0:
            with failed.open("a", newline="") as f:
                csv.writer(f).writerow([stamp(), label, "summarize_existing_only", code])
        write_line(progress, f"[{stamp()}] END {label}")

    code = run_cmd([PY, "scripts/summarize_revision_focused_30seed_core.py", "--root", str(OUT_ROOT)], OUT_ROOT / "summarize.log")
    if code != 0:
        with failed.open("a", newline="") as f:
            csv.writer(f).writerow([stamp(), "final_summary", "all", code])
    write_line(progress, f"[{stamp()}] DONE focused_30seed_core")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
