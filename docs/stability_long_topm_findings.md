# Long SS-KAN Top-M Sweep Findings

Source: `results/stability_kan/long_topm_long_boundary_v1/topm_sweep_summary.csv`

The sweep is complete: 120 stability-selected KAN summary rows, with 10 successful seeds for every configuration and no warning/error lines in the logs.

## Main Takeaways

1. SS-KAN is effective in the medium/high-signal regimes.
   - For `c=0.5`, SS-KAN reaches interaction F1 `1.0` in most `d=50/100`, `n=512/1024` settings.
   - For `c=1.0`, SS-KAN reaches interaction F1 `1.0` for all tested settings except the deliberately tight `d=100,n=512,top_m=4` pair-stability case.

2. The hard boundary is now sharply localized.
   - `c=0.25,d=100,n=512` remains unrecovered: interaction F1 is `0.0` for all tested `top_m` and both SS-KAN variants.
   - `c=0.25,d=100,n=1024` is a partial recovery regime: endpoint recall reaches about `0.95-1.0`, but interaction F1 peaks at `0.7`.

3. `top_m=5/6` is the best default range.
   - Across all settings, `SS-KAN-P, top_m=6` has the highest mean interaction F1 (`0.825`).
   - `SS-KAN-V, top_m=5/6` is very close (`0.808`).
   - Larger support budgets (`top_m=8/10`) do not consistently improve recovery and sometimes reduce pair ranking stability in the weak-signal regime.

4. Pair-stability helps at the hardest recoverable boundary, but is not uniformly better.
   - At `c=0.25,d=100,n=1024`, `SS-KAN-P, top_m=6` reaches interaction F1 `0.7`, versus `SS-KAN-V, top_m=5` at `0.7` and `SS-KAN-V, top_m=6` at `0.6`.
   - At `c=1.0,d=100,n=512`, `SS-KAN-P, top_m=4` fails badly because it often keeps only one true interaction endpoint, while `top_m=6` fixes it.

5. Prediction accuracy and formula fidelity are still separated.
   - In `c=0.25,d=100,n=1024`, selected support often contains all true variables and MSE is near zero, but the finite-difference pair ranking still often selects `(0,1)` or `(0,3)` instead of `(2,3)`.
   - This is exactly the paper's central claim: low prediction error and correct variable support do not guarantee formula-level interaction fidelity.

## Best SS-KAN Setting by Regime

| c | d | n | best SS method | top_m | interaction F1 | endpoint recall | test MSE |
|---:|---:|---:|---|---:|---:|---:|---:|
| 0.25 | 50 | 512 | SS-KAN-P | 5 | 0.9 | 1.0 | 0.000 |
| 0.25 | 50 | 1024 | SS-KAN-V | 5 | 0.8 | 1.0 | 0.001 |
| 0.25 | 100 | 512 | SS-KAN-V | 4 | 0.0 | 0.4 | 0.013 |
| 0.25 | 100 | 1024 | SS-KAN-P | 6 | 0.7 | 0.95 | 0.002 |
| 0.50 | 50 | 512 | SS-KAN-V | 8 | 1.0 | 1.0 | 0.000 |
| 0.50 | 50 | 1024 | SS-KAN-P | 8 | 1.0 | 1.0 | 0.000 |
| 0.50 | 100 | 512 | SS-KAN-V | 8 | 1.0 | 1.0 | 0.000 |
| 0.50 | 100 | 1024 | SS-KAN-V | 4 | 1.0 | 1.0 | 0.000 |
| 1.00 | 50 | 512 | SS-KAN-P | 6 | 1.0 | 1.0 | 0.000 |
| 1.00 | 50 | 1024 | SS-KAN-V | 6 | 1.0 | 1.0 | 0.000 |
| 1.00 | 100 | 512 | SS-KAN-V | 10 | 1.0 | 1.0 | 0.000 |
| 1.00 | 100 | 1024 | SS-KAN-P | 6 | 1.0 | 1.0 | 0.000 |

## Recommended Paper Positioning

Use `SS-KAN-V, top_m=6` as the conservative main method and report `SS-KAN-P` as a KAN-native variant/ablation. The main story should not claim universal recovery. The stronger and more defensible claim is:

> Stability-selected KAN shifts the formula-fidelity recovery boundary in medium/high-signal regimes, while the weakest high-dimensional regime remains a genuine structure-recovery bottleneck.

This keeps the paper honest and actually makes the diagnostic contribution sharper.
