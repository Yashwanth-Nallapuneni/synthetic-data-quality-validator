Per dataset (intended warning fired / any warning fired, 20 test runs per cell):

| generator | adult: intended / any | german_credit: intended / any | cmc: intended / any |
|---|---|---|---|
| `heldout_real` | - / 0.00 | - / 0.10 | - / 0.10 |
| `gaussian_copula` | - / 1.00 | - / 1.00 | - / 1.00 |
| `exact_copy` | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| `near_copy` | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| `near_copy_5pct` | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| `column_shuffle` | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| `mode_collapse` | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| `category_oversample` | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| `label_shuffle` | 1.00 / 1.00 | 0.85 / 1.00 | 0.90 / 1.00 |

Leave-one-dataset-out (thresholds learned from the other two datasets, scored on the held-out dataset's test seeds):

| generator | n | privacy_memorisation | diversity_mode_collapse | diversity_excess_variance | diversity_low_coverage | numeric_marginal_shift | categorical_mean_l1 | categorical_mean_js | categorical_worst_shift | dependence_shift | detectability_high_auc | utility_degradation | utility_calibration_degradation | utility_synthetic_single_class | intended warning fired | any warning fired |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `heldout_real` | 60 | 0.10 | 0.00 | 0.00 | 0.00 | 0.03 | 0.00 | 0.00 | 0.07 | 0.02 | 0.00 | 0.00 | 0.03 | 0.00 | - | 0.23 |
| `gaussian_copula` | 60 | 0.00 | 0.00 | 0.00 | 0.70 | 0.02 | 0.00 | 0.00 | 0.00 | 1.00 | 0.73 | 0.42 | 0.08 | 0.00 | - | 1.00 |
| `exact_copy` | 60 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 |
| `near_copy` | 60 | 1.00 | 0.00 | 0.00 | 0.00 | 0.70 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 |
| `near_copy_5pct` | 60 | 1.00 | 0.00 | 0.00 | 0.00 | 0.70 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 |
| `column_shuffle` | 60 | 0.00 | 0.00 | 0.00 | 1.00 | 0.08 | 0.18 | 0.22 | 0.20 | 1.00 | 1.00 | 0.93 | 0.23 | 0.00 | 1.00 | 1.00 |
| `mode_collapse` | 60 | 0.37 | 0.58 | 0.10 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.80 | 1.00 | 0.12 | 1.00 | 1.00 |
| `category_oversample` | 60 | 0.00 | 0.00 | 0.00 | 0.95 | 0.08 | 1.00 | 1.00 | 1.00 | 0.50 | 1.00 | 0.00 | 0.22 | 0.00 | 1.00 | 1.00 |
| `label_shuffle` | 60 | 0.15 | 0.00 | 0.00 | 0.33 | 0.08 | 0.18 | 0.22 | 0.20 | 0.50 | 1.00 | 0.92 | 0.12 | 0.00 | 0.92 | 1.00 |

LODO false-alarm rate on `heldout_real`: 14 of 60 runs. Failure modes detected in at least 90% of runs: 6 of 6.
