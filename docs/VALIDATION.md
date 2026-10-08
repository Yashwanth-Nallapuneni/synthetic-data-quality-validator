# Validation against injected failures

Back to the [README](../README.md).

A tool that grades synthetic data is only credible if it catches known failures and stays quiet on good data. `experiments/` tests that. The results below are from protocol **v2**. Everything was produced by `python -m experiments.benchmark && python -m experiments.calibrate --update-readme docs/VALIDATION.md` (about 6 minutes on 8 cores) and the raw results are in `experiments/results/`. See [`experiments/README.md`](../experiments/README.md) for the details.

**Headline (v2, test seeds 20 to 39): the validator detects 6 of 6 injected failure modes (intended warning in at least 90% of held-out runs) with 4 of 60 held-out-real runs raising any warning.**

**v1 to v2.** The first run (v1, archived in `experiments/results/v1/`) found 4 of 6 failure modes and exposed a flaw in how four failures were injected plus two metric weaknesses. v2 fixes those and tests on new seeds that nobody had seen. The v1 test seeds were used to motivate the fixes, so they now count as calibration data. The full list is in [Changes between v1 and v2](../experiments/README.md#changes-between-v1-and-v2).

**Method**
- Datasets, from OpenML with pinned ids: Adult (id 1590, 48,842 rows, subsampled to 6,000 rows per seed for runtime), German credit (id 31, 1,000 rows) and CMC (id 23, 1,473 rows, 3 classes). All three list the licence as Public on OpenML
- 40 seeds per dataset (0 to 39). Per seed the rows are split at random, stratified by label, into the generator training set T (50%), a held-out gold set H_gold (25%) and a held-out baseline set (25%). The baseline set is the `holdout_df` for the privacy check. Every generator produces as many rows as H_gold
- Good generators: `heldout_real` (H_gold itself, the null) and `gaussian_copula` (written from scratch in `experiments/generators.py`; categorical columns use interval jitter, so it attenuates dependence)
- Failure injections. `exact_copy` (rows of T without replacement) and `near_copy` (the same plus Gaussian noise of 1% of each numeric column's std; `near_copy_5pct` is a stronger variant that is not counted in the headline) draw from T. The other four start from H_gold so that they contain no training rows: `column_shuffle` (every column permuted independently), `mode_collapse` (a bootstrap of 5 random rows plus 1% numeric jitter), `category_oversample` (50% of rows of the categorical feature with the most categories set to its rarest category) and `label_shuffle` (label column permuted)
- Calibration: seeds 0 to 19 calibrate and seeds 20 to 39 test. Each threshold is set from the `heldout_real` calibration runs only: for a check that warns when a statistic is high, threshold = max over those runs + max(10% of the max, 0.01), and for a check that warns when it is low, threshold = min over those runs - max(10% of the min, 0.01). The broken generators and the copula play no part in choosing thresholds. With n pooled null values a fresh null value falls beyond the observed maximum with probability 1 / (n + 1) per check, which is about 1.6% for n = 60, before the margin. Several checks run on each report, so the any-warning false-alarm rate is higher than any one check's
- Evaluation: the thresholds are applied to the recorded statistics of the test seeds. A hit means the intended warning fired. Intended warnings: `exact_copy` and `near_copy` need `privacy_memorisation`; `column_shuffle` needs `dependence_shift` or `detectability_high_auc`; `mode_collapse` needs `diversity_mode_collapse` or `diversity_low_coverage`; `category_oversample` needs any `categorical_*` code; `label_shuffle` needs `utility_degradation`. A failure mode counts as detected when its intended warning fires in at least 90% of test runs

**Calibrated thresholds** (from `app/calibrated_thresholds.json`, which also records the rule, seeds and date)

| Threshold | Provisional | Calibrated | Null values pooled |
|---|---|---|---|
| `privacy_high_share` | 0.15 | 0.099674 | 60 |
| `privacy_medium_share` | 0.1 | 0.089674 | derived (display only) |
| `privacy_exact_match_margin` | 0.05 | 0.031864 | 60 |
| `privacy_closeness_auc` | 0.6 | 0.620937 | 60 |
| `variance_ratio_low` | 0.5 | 0.766848 | 60 |
| `variance_ratio_high` | 1.5 | 1.385892 | 60 |
| `coverage_min` | 0.5 | 0.7362 | 60 |
| `numeric_max_ks` | 0.2 | 0.1364 | 60 |
| `utility_balanced_accuracy_ratio` | 0.8 | 0.771132 | 120 |
| `utility_r2_drop` | 0.2 | 0.2 | 0 (no regression dataset, provisional value kept) |
| `utility_ece_increase` | 0.1 | 0.105622 | 120 |
| `categorical_mean_l1` | 0.3 | 0.126585 | 60 |
| `categorical_mean_js` | 0.2 | 0.062091 | 60 |
| `categorical_worst_shift` | 0.25 | 0.11 | 60 |
| `dependence_mean_abs_diff` | 0.1 | 0.074027 | 60 |
| `c2st_auc` | 0.7 | 0.612485 | 60 |

**Results on the test seeds**

<!-- detection-matrix:start -->

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

<!-- detection-matrix:end -->

Per-dataset rates and a leave-one-dataset-out variant (thresholds learned from the other two datasets only) are in `experiments/results/detection_matrix_details.md`. The machine-readable versions are `detection_matrix.json` and `summary.json`.

**Reading the results**
- Five of six failure modes fire their intended warning in every test run on all three datasets. `label_shuffle` fires `utility_degradation` in 0.92 of runs overall (1.00 on Adult, 0.85 on German credit and 0.90 on CMC), which clears the 90% bar overall but not on German credit alone (CMC is exactly 0.90)
- `heldout_real` raised a warning in 4 of 60 test runs. They came from four different checks (one run each), and all four were on German credit or CMC (2 of 20 runs each). The expected per-check rate was about 1.6%, so this is in line with having 13 checks and 60 runs, and it means the any-warning false-alarm rate is not zero
- The copula triggered a warning in every test run, mostly `dependence_shift` and `detectability_high_auc`. That is expected for a weak copula and is not a false alarm
- The leave-one-dataset-out thresholds are much less safe: 14 of 60 `heldout_real` test runs raised a warning, so thresholds learned from other datasets do not transfer well
- `mode_collapse` also raised `privacy_memorisation` in some runs even though it contains no training rows (see the matrix). I did not investigate why

**Limitations of this evidence**
- Only three datasets, all small or subsampled, and no regression dataset (so `utility_r2_drop` stays at its provisional value)
- The failure injections are simple and their strength was fixed before the v2 test seeds were run. Detection rates depend on that strength: a milder oversample or a smaller shuffled fraction would be harder to catch
- v2 fixed the weaknesses v1 showed, so a 6 of 6 result partly reflects fixes aimed at those same datasets and failure modes. The new seeds are unseen, but the datasets and injection types are not
- The copula is a weak generator. No modern deep generator (CTGAN, diffusion) was tested, so the false-alarm rate on a good-but-imperfect generator is not measured
- The 4 of 60 false-alarm count has a small denominator, so the true rate is uncertain
- The thresholds are tuned to the dataset sizes used here (a few hundred to a few thousand rows). Other sizes may need recalibration with `python -m experiments.calibrate`

## Comparison with SDMetrics

SDMetrics 0.32.0 (DataCebo, MIT licence) is the incumbent open-source library for this job. I ran it on the same failure-injection benchmark: the same tables for every (dataset, seed, generator), the same column types, and its thresholds calibrated with exactly the rule used for this tool (`heldout_real` seeds 0 to 19, null extreme minus or plus max(10%, 0.01)) and then scored on seeds 20 to 39. SDMetrics checks are mapped to the failure modes as memorisation to DCR overfitting protection or new row synthesis, categorical shift to Column Shapes, dependence to Column Pair Trends or LogisticDetection, and label damage to ML efficacy. SDMetrics has no diversity or collapse check, so the strict view gives `mode_collapse` no intended check, while the lenient view accepts any SDMetrics check. The last column uses only the single overall Quality score, how it is often used. The two "worst column/pair" columns were added after I had seen the first results, following a fairness review: our marginal checks use the worst column while Column Shapes and Column Pair Trends are averages, so these columns use the smallest per-column and per-pair score of the same QualityReport instead (same calibration protocol). The change can only help SDMetrics, and it does.

<!-- sdmetrics-comparison:start -->

Test seeds 20 to 39, 60 runs per generator. Cells are the fraction of test runs in which the intended check fired ("detected" means at least 90%). Both tools' thresholds come from `heldout_real` seeds 0 to 19 with the same rule.

| Failure mode | Ours (intended warning) | SDMetrics strict | SDMetrics strict (worst column/pair) | SDMetrics lenient | SDMetrics lenient (worst column/pair) | SDMetrics overall score only |
|---|---|---|---|---|---|---|
| `exact_copy` | 1.00 detected | 1.00 detected | 1.00 detected | 1.00 detected | 1.00 detected | 0.00 |
| `near_copy` | 1.00 detected | 1.00 detected | 1.00 detected | 1.00 detected | 1.00 detected | 0.00 |
| `column_shuffle` | 1.00 detected | 0.55 | 0.40 | 0.78 | 0.78 | 0.00 |
| `mode_collapse` | 1.00 detected | 0.00 | 0.00 | 1.00 detected | 1.00 detected | 1.00 detected |
| `category_oversample` | 1.00 detected | 0.00 | 1.00 detected | 0.67 | 1.00 detected | 0.17 |
| `label_shuffle` | 0.92 detected | 0.38 | 0.38 | 0.43 | 0.48 | 0.00 |
| `near_copy_5pct` (not in count) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 |
| Failure modes detected | 6 of 6 | 2 of 6 | 3 of 6 | 3 of 6 | 4 of 6 | 1 of 6 |
| `heldout_real` runs with any warning | 4 of 60 | 0 of 60 | 0 of 60 | 0 of 60 | 0 of 60 | 0 of 60 |
| `gaussian_copula` runs with any warning | 60 of 60 | 15 of 60 | 30 of 60 | 15 of 60 | 30 of 60 | 0 of 60 |

<!-- sdmetrics-comparison:end -->

**Takeaway.** On this benchmark this validator detects 6 of 6 failure modes and SDMetrics 2 to 4 of 6, depending on how it is used. Using the worst column instead of the average (a change made after seeing the first results) lifts SDMetrics from 2 to 3 of 6 in the strict view and from 3 to 4 of 6 in the lenient one: `category_oversample` is then caught in every run, so the first gap on that mode was an averaging effect and not a lack of capability. Exact and near copies are caught by both tools in every run. Three gaps remain: `column_shuffle` (SDMetrics at most 78%, and only 40% for the strict worst-pair mapping, because the worst pair is a noisy statistic), `label_shuffle` (38% to 48%) and `mode_collapse` (no SDMetrics diversity check; the lenient view catches it only because fidelity scores happen to drop). The price is a higher false-alarm count (4 of 60 `heldout_real` runs against 0 of 60 for every SDMetrics view), partly because SDMetrics' pooled thresholds are wide, and the worst-column checks raise SDMetrics' warnings on `gaussian_copula` from 15 to 30 of 60 (ours: 60 of 60). SDMetrics is faster only on small data: its DCR metric makes it about 6 times slower per run here. Caveats: the benchmark was built around this tool's checks, the SDMetrics mapping and ML efficacy ratio are my choices, and per-dataset SDMetrics thresholds might do better. The other differences are not about detection: per-check calibrated thresholds with a stated false-alarm target, warnings that name the metric and value, a holdout privacy baseline with a tie-robust rank statistic, and an API and CLI. Method details, per-dataset numbers and runtimes are in [Comparison with SDMetrics](../experiments/README.md#comparison-with-sdmetrics). To reproduce: `pip install -e ".[compare]" && python -m experiments.compare_sdmetrics && python -m experiments.compare_sdmetrics --worst-only && python -m experiments.compare_report --update-readme docs/VALIDATION.md --update-full experiments/README.md` (about 80 minutes on 3 jobs, plus 1 minute for the worst-column pass).
