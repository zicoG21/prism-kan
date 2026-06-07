# ClaimTransfer-Bench 1.0 Benchmark Card

## Purpose

ClaimTransfer-Bench measures structural overclaim risk in formula-recovery
workflows.  The benchmark unit is a task card x adapter x evidence object x
typed structural claim, not a single model-level score.

## Released Scope

- Official claim rows: 247,990
- Public diagnostic task ids: 71
- Standard-formula settings: 90
- Adapter families: 13
- Methods/adapters: 33

## Official Claims

Primary structural-transfer claims cover prediction, support, endpoints, pair,
candidate pair, pruning/extraction, and symbolic status.  The expression-quality
track decomposes formula quality into variable/support recall, pair-term recall,
operator recall, and complexity-budget claims.

## Public and Hidden Use

The public suite supports reproducible diagnostic analysis.  The same adapter
format and scorer support offline hidden/private cards or private seeds for
maintainer-run evaluation.

## Missingness Policy

Missing evidence is explicit.  A method is not penalized for unsupported native
fields, but it is not authorized to make claims whose evidence object is absent.

## Intended Use

Use ClaimTransfer-Bench to identify which claim-transfer edge fails for a method:
prediction -> pair, support -> prediction, candidate -> verified pair, fitted
pair -> readout/pruning, or symbolic status -> expression quality.

## Non-Goals

The benchmark does not collapse all formula-recovery behavior into one scalar
leaderboard.  Exact algebraic equivalence, coefficient error, dimensional
consistency, and extrapolation are task-card-specific fields rather than a
universal primary score.

## Minimal Commands

```bash
python scripts/run_benchmark.py --quick
python scripts/check_benchmark_artifact.py
python scripts/build_full_benchmark_analysis_reports.py
```
