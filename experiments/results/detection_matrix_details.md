Per dataset (intended warning fired / any warning fired, 10 test runs per cell):

| generator | adult: intended / any | german_credit: intended / any | cmc: intended / any |
|---|---|---|---|
| `heldout_real` | - / 0.00 | - / 0.00 | - / 0.00 |
| `gaussian_copula` | - / 1.00 | - / 1.00 | - / 1.00 |
| `exact_copy` | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| `near_copy` | 1.00 / 1.00 | 1.00 / 1.00 | 0.00 / 1.00 |
| `near_copy_5pct` | 1.00 / 1.00 | 1.00 / 1.00 | 0.00 / 1.00 |
| `column_shuffle` | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| `mode_collapse` | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| `category_oversample` | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| `label_shuffle` | 0.00 / 1.00 | 0.30 / 1.00 | 0.90 / 1.00 |

Leave-one-dataset-out (thresholds learned from the other two datasets, scored on the held-out dataset's test seeds):

| generator | n | privacy_memorisation | diversity_mode_collapse | diversity_excess_variance | diversity_low_coverage | numeric_marginal_shift | categorical_mean_l1 | categorical_mean_js | categorical_worst_shift | dependence_shift | detectability_high_auc | utility_degradation | utility_calibration_degradation | utility_synthetic_single_class | intended warning fired | any warning fired |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `heldout_real` | 30 | 0.07 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | - | 0.07 |
| `gaussian_copula` | 30 | 0.00 | 0.00 | 0.00 | 0.63 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.67 | 0.13 | 0.00 | 0.00 | - | 1.00 |
| `exact_copy` | 30 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 |
| `near_copy` | 30 | 0.67 | 0.00 | 0.00 | 0.00 | 0.67 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.67 | 1.00 |
| `near_copy_5pct` | 30 | 0.67 | 0.00 | 0.00 | 0.00 | 0.67 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.67 | 1.00 |
| `column_shuffle` | 30 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 0.43 | 0.23 | 0.00 | 1.00 | 1.00 |
| `mode_collapse` | 30 | 0.67 | 0.67 | 0.10 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.50 | 1.00 | 0.10 | 1.00 | 1.00 |
| `category_oversample` | 30 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.93 | 1.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 |
| `label_shuffle` | 30 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.40 | 0.17 | 0.00 | 0.40 | 1.00 |

LODO false-alarm rate on `heldout_real`: 2 of 30 runs. Failure modes detected in at least 90% of runs: 4 of 6.
