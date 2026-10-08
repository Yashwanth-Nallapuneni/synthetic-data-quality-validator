Test half: odd seeds 1 to 19 on 3 datasets (30 runs per generator). Cells are the fraction of runs in which the warning fired. Columns for codes that never fired are omitted.

| generator | n | privacy_memorisation | diversity_mode_collapse | diversity_excess_variance | diversity_low_coverage | numeric_marginal_shift | categorical_mean_l1 | categorical_mean_js | categorical_worst_shift | dependence_shift | detectability_high_auc | utility_degradation | utility_calibration_degradation | utility_synthetic_single_class | intended warning fired | any warning fired |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `heldout_real` | 30 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | - | 0.00 |
| `gaussian_copula` | 30 | 0.00 | 0.00 | 0.00 | 0.57 | 0.00 | 0.00 | 0.00 | 0.00 | 0.97 | 0.70 | 0.10 | 0.00 | 0.00 | - | 1.00 |
| `exact_copy` | 30 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 |
| `near_copy` | 30 | 0.67 | 0.00 | 0.00 | 0.00 | 0.67 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.67 | 1.00 |
| `near_copy_5pct` | 30 | 0.67 | 0.00 | 0.00 | 0.00 | 0.67 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.67 | 1.00 |
| `column_shuffle` | 30 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 0.43 | 0.13 | 0.00 | 1.00 | 1.00 |
| `mode_collapse` | 30 | 0.67 | 0.67 | 0.10 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.47 | 1.00 | 0.10 | 1.00 | 1.00 |
| `category_oversample` | 30 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.90 | 0.90 | 1.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 |
| `label_shuffle` | 30 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.40 | 0.17 | 0.00 | 0.40 | 1.00 |

False-alarm rate on `heldout_real` (any warning): 0 of 30 test runs (0.000).

Warnings on `gaussian_copula` (any warning): 30 of 30 test runs (1.000). A copula legitimately differs from real data, so these are not strictly false alarms.

Failure modes whose intended warning fires in at least 90% of test runs: 4 of 6.
