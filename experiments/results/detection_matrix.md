Test set: seeds 20 to 39 on 3 datasets (60 runs per generator). Cells are the fraction of runs in which the warning fired. Columns for codes that never fired are omitted.

| generator | n | privacy_memorisation | diversity_mode_collapse | diversity_excess_variance | diversity_low_coverage | numeric_marginal_shift | categorical_mean_l1 | categorical_mean_js | categorical_worst_shift | dependence_shift | detectability_high_auc | utility_degradation | utility_calibration_degradation | utility_synthetic_single_class | intended warning fired | any warning fired |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `heldout_real` | 60 | 0.02 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.02 | 0.02 | 0.00 | 0.00 | 0.02 | 0.00 | - | 0.07 |
| `gaussian_copula` | 60 | 0.00 | 0.00 | 0.00 | 0.62 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.72 | 0.40 | 0.07 | 0.00 | - | 1.00 |
| `exact_copy` | 60 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 |
| `near_copy` | 60 | 1.00 | 0.00 | 0.00 | 0.00 | 0.70 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 |
| `near_copy_5pct` | 60 | 1.00 | 0.00 | 0.00 | 0.00 | 0.70 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 |
| `column_shuffle` | 60 | 0.00 | 0.00 | 0.00 | 1.00 | 0.03 | 0.12 | 0.10 | 0.07 | 1.00 | 1.00 | 0.93 | 0.18 | 0.00 | 1.00 | 1.00 |
| `mode_collapse` | 60 | 0.37 | 0.58 | 0.08 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.80 | 1.00 | 0.12 | 1.00 | 1.00 |
| `category_oversample` | 60 | 0.00 | 0.00 | 0.00 | 0.90 | 0.03 | 1.00 | 1.00 | 1.00 | 0.42 | 1.00 | 0.00 | 0.08 | 0.00 | 1.00 | 1.00 |
| `label_shuffle` | 60 | 0.07 | 0.00 | 0.00 | 0.22 | 0.03 | 0.12 | 0.10 | 0.07 | 0.47 | 1.00 | 0.92 | 0.10 | 0.00 | 0.92 | 1.00 |

False-alarm rate on `heldout_real` (any warning): 4 of 60 test runs (0.067).

Warnings on `gaussian_copula` (any warning): 60 of 60 test runs (1.000). A copula legitimately differs from real data, so these are not strictly false alarms.

Failure modes whose intended warning fires in at least 90% of test runs: 6 of 6.

Headline: detects 6 of 6 injected failure modes (intended warning in at least 90% of held-out runs) with 4 of 60 held-out-real runs raising any warning.

False-alarm rate of each check on the `heldout_real` test runs (k of 60 runs):

| warning code | k of n | rate |
|---|---|---|
| `privacy_memorisation` | 1 of 60 | 0.017 |
| `diversity_mode_collapse` | 0 of 60 | 0.000 |
| `diversity_excess_variance` | 0 of 60 | 0.000 |
| `diversity_low_coverage` | 0 of 60 | 0.000 |
| `numeric_marginal_shift` | 0 of 60 | 0.000 |
| `categorical_mean_l1` | 0 of 60 | 0.000 |
| `categorical_mean_js` | 0 of 60 | 0.000 |
| `categorical_worst_shift` | 1 of 60 | 0.017 |
| `dependence_shift` | 1 of 60 | 0.017 |
| `detectability_high_auc` | 0 of 60 | 0.000 |
| `utility_degradation` | 0 of 60 | 0.000 |
| `utility_calibration_degradation` | 1 of 60 | 0.017 |
| `utility_synthetic_single_class` | 0 of 60 | 0.000 |
