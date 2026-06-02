#!/usr/bin/env bash
#SBATCH --job-name=kan-a40-smoke
#SBATCH --partition=spgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

# A40 / spgpu CUDA smoke test.
#
# Submit from the Great Lakes repo root:
#   sbatch --account=jaabell0 \
#     --export=ALL,PYTHON_BIN=$PWD/.venv/bin/python \
#     scripts/greatlakes_a40_cuda_smoke.sbatch
#
# For engin1, use the same script with --account=engin1.  This only requests
# one GPU and two CPU cores, so it is friendly to the engin1 quota.

cd "${SLURM_SUBMIT_DIR:-$PWD}"
PY="${PYTHON_BIN:-python}"

echo "node=${SLURM_JOB_NODELIST:-unknown}"
date
nvidia-smi

"$PY" - <<'PY'
import torch

print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    x = torch.randn(2048, 2048, device="cuda")
    y = x @ x.T
    torch.cuda.synchronize()
    print("check:", float(y[0, 0].detach().cpu()))
PY

