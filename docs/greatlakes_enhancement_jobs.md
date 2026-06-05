# Great Lakes enhancement jobs

These jobs are optional strengthening runs for ClaimTransfer-Bench.  They are
not required for the current alpha P0/P1/P2 readiness gate.

## One-command enhancement queue

From the Great Lakes checkout:

```bash
cd /home/zicong/prism-kan
git pull
source .venv/bin/activate
bash scripts/submit_claimtransfer_enhancement_gl.sh
```

Useful variants:

```bash
# CPU-only jobs: symbolic expression, cross-method gapfill, TreeGate gapfill.
SUBMIT_SPGPU=0 SUBMIT_GPU=0 bash scripts/submit_claimtransfer_enhancement_gl.sh

# A40-only jobs: scorer grammar and EPIM/TreeGate breadth.
SUBMIT_STANDARD=0 SUBMIT_GPU=0 ACCOUNT=jaabell0 bash scripts/submit_claimtransfer_enhancement_gl.sh

# Slow pyKAN GPU-only robustness rows.
SUBMIT_STANDARD=0 SUBMIT_SPGPU=0 ACCOUNT=jaabell0 bash scripts/submit_claimtransfer_enhancement_gl.sh

# engin1 A40 queue only.
ACCOUNT=engin1 SUBMIT_STANDARD=0 SUBMIT_GPU=0 bash scripts/submit_claimtransfer_enhancement_gl.sh
```

The wrapper intentionally does not add Slurm array throttles.  Let Slurm and the
account quotas decide how many tasks run.

## Coverage-gap-only queue

For the current optional symbolic-expression gap and CPU gapfill rows:

```bash
ACCOUNT=jaabell0 bash scripts/submit_claimtransfer_gapfill_gl.sh
```

Skip parts with:

```bash
SUBMIT_XFER=0 SUBMIT_TREEGATE=0 ACCOUNT=jaabell0 bash scripts/submit_claimtransfer_gapfill_gl.sh
SUBMIT_SYMEXPR=0 ACCOUNT=jaabell0 bash scripts/submit_claimtransfer_gapfill_gl.sh
```

## New symbolic-expression diagnostic

The new standard-partition job:

```bash
sbatch --account=jaabell0 --array=0-3 \
  --export=ALL,PYTHON_BIN=$PWD/.venv/bin/python \
  scripts/greatlakes_symbolic_expression_operator_recall_standard.sbatch
```

It writes:

```text
results/revision/symbolic_expression_operator_recall/*/symbolic_expression_detail.csv
results/revision/symbolic_expression_operator_recall/*/symbolic_expression_summary.csv
```

After pulling results back, run:

```bash
python scripts/refresh_from_greatlakes_results.py --latest
python scripts/run_benchmark.py --quick --rebuild-adapter-outputs
```

The expected effect is to add `symbolic_operator_recall` rows for the
`scientific_expression` task family.  This is an expression-level diagnostic
control, not a claim that the benchmark now contains a full symbolic-regression
leaderboard.
