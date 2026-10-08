Test seeds 20 to 39 (60 runs per generator, 3 datasets). Thresholds for both tools come from `heldout_real` runs of seeds 0 to 19 with the same rule. A mode counts as detected when its intended check fires in at least 90% of test runs. Cells are the fraction of test runs in which the intended check fired. SDMetrics strict uses the analogous SDMetrics check and has none for `mode_collapse` (SDMetrics has no diversity check). SDMetrics lenient counts any individual SDMetrics check. The last column uses only the overall Quality score.

| Failure mode | Ours (intended warning) | SDMetrics strict | SDMetrics lenient | SDMetrics overall score only |
|---|---|---|---|---|
| `exact_copy` | 1.00 detected | 1.00 detected | 1.00 detected | 0.00 |
| `near_copy` | 1.00 detected | 1.00 detected | 1.00 detected | 0.00 |
| `column_shuffle` | 1.00 detected | 0.55 | 0.78 | 0.00 |
| `mode_collapse` | 1.00 detected | 0.00 | 1.00 detected | 1.00 detected |
| `category_oversample` | 1.00 detected | 0.00 | 0.67 | 0.17 |
| `label_shuffle` | 0.92 detected | 0.38 | 0.43 | 0.00 |
| `near_copy_5pct` (not in count) | 1.00 | 1.00 | 1.00 | 0.00 |
| Failure modes detected | 6 of 6 | 2 of 6 | 3 of 6 | 1 of 6 |
| `heldout_real` runs with any warning | 4 of 60 | 0 of 60 | 0 of 60 | 0 of 60 |
| `gaussian_copula` runs with any warning | 60 of 60 | 15 of 60 | 15 of 60 | 0 of 60 |

Per dataset (intended check fired, then any-warning counts):

| Failure mode | View | adult | cmc | german_credit |
|---|---|---|---|---|
| `exact_copy` | Ours (intended warning) | 1.00 | 1.00 | 1.00 |
| `exact_copy` | SDMetrics strict | 1.00 | 1.00 | 1.00 |
| `exact_copy` | SDMetrics lenient | 1.00 | 1.00 | 1.00 |
| `exact_copy` | SDMetrics overall score only | 0.00 | 0.00 | 0.00 |
| `near_copy` | Ours (intended warning) | 1.00 | 1.00 | 1.00 |
| `near_copy` | SDMetrics strict | 1.00 | 1.00 | 1.00 |
| `near_copy` | SDMetrics lenient | 1.00 | 1.00 | 1.00 |
| `near_copy` | SDMetrics overall score only | 0.00 | 0.00 | 0.00 |
| `column_shuffle` | Ours (intended warning) | 1.00 | 1.00 | 1.00 |
| `column_shuffle` | SDMetrics strict | 1.00 | 0.10 | 0.55 |
| `column_shuffle` | SDMetrics lenient | 1.00 | 0.55 | 0.80 |
| `column_shuffle` | SDMetrics overall score only | 0.00 | 0.00 | 0.00 |
| `mode_collapse` | Ours (intended warning) | 1.00 | 1.00 | 1.00 |
| `mode_collapse` | SDMetrics strict | 0.00 | 0.00 | 0.00 |
| `mode_collapse` | SDMetrics lenient | 1.00 | 1.00 | 1.00 |
| `mode_collapse` | SDMetrics overall score only | 1.00 | 1.00 | 1.00 |
| `category_oversample` | Ours (intended warning) | 1.00 | 1.00 | 1.00 |
| `category_oversample` | SDMetrics strict | 0.00 | 0.00 | 0.00 |
| `category_oversample` | SDMetrics lenient | 0.00 | 1.00 | 1.00 |
| `category_oversample` | SDMetrics overall score only | 0.00 | 0.50 | 0.00 |
| `label_shuffle` | Ours (intended warning) | 1.00 | 0.90 | 0.85 |
| `label_shuffle` | SDMetrics strict | 0.40 | 0.45 | 0.30 |
| `label_shuffle` | SDMetrics lenient | 0.45 | 0.45 | 0.40 |
| `label_shuffle` | SDMetrics overall score only | 0.00 | 0.00 | 0.00 |
| `near_copy_5pct` | Ours (intended warning) | 1.00 | 1.00 | 1.00 |
| `near_copy_5pct` | SDMetrics strict | 1.00 | 1.00 | 1.00 |
| `near_copy_5pct` | SDMetrics lenient | 1.00 | 1.00 | 1.00 |
| `near_copy_5pct` | SDMetrics overall score only | 0.00 | 0.00 | 0.00 |
| `heldout_real` any warning | Ours (intended warning) | 0 of 20 | 2 of 20 | 2 of 20 |
| `heldout_real` any warning | SDMetrics strict | 0 of 20 | 0 of 20 | 0 of 20 |
| `heldout_real` any warning | SDMetrics lenient | 0 of 20 | 0 of 20 | 0 of 20 |
| `heldout_real` any warning | SDMetrics overall score only | 0 of 20 | 0 of 20 | 0 of 20 |
| `gaussian_copula` any warning | Ours (intended warning) | 20 of 20 | 20 of 20 | 20 of 20 |
| `gaussian_copula` any warning | SDMetrics strict | 6 of 20 | 6 of 20 | 3 of 20 |
| `gaussian_copula` any warning | SDMetrics lenient | 6 of 20 | 6 of 20 | 3 of 20 |
| `gaussian_copula` any warning | SDMetrics overall score only | 0 of 20 | 0 of 20 | 0 of 20 |

SDMetrics thresholds and false alarms of each score on the `heldout_real` test runs:

| SDMetrics score | Warns when | Calibrated threshold | Null extreme | Null values | `heldout_real` test false alarms |
|---|---|---|---|---|---|
| `column_shapes` | below | 0.8543 | 0.9492 | 60 | 0 of 60 |
| `column_pair_trends` | below | 0.7660 | 0.8511 | 60 | 0 of 60 |
| `logistic_detection` | below | 0.7521 | 0.8357 | 60 | 0 of 60 |
| `dcr_overfitting` | below | 0.4536 | 0.5040 | 60 | 0 of 60 |
| `new_row_synthesis` | below | 0.8584 | 0.9538 | 60 | 0 of 60 |
| `efficacy_ratio` | below | 0.7298 | 0.8109 | 60 | 0 of 60 |
| `quality_score` | below | 0.8132 | 0.9035 | 60 | 0 of 60 |

Runtime (all runs; both tools timed inside the same parallel job, one thread each):

| Tool | Mean seconds per run |
|---|---|
| Ours (`generate_trust_report`, validator only) | 1.91 |
| SDMetrics (all metrics below) | 11.80 |
| SDMetrics `quality_score` | 0.21 |
| SDMetrics `logistic_detection` | 0.09 |
| SDMetrics `dcr_overfitting` | 8.27 |
| SDMetrics `new_row_synthesis` | 3.14 |
| SDMetrics `efficacy_ratio` | 0.10 |

SDMetrics errors: none.
Runs where our re-run did not reproduce the stored statistics: 0.
