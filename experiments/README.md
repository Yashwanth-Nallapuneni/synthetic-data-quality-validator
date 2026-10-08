# Experiments: validating the validator

This folder holds two protocol versions. **v2** (current) is described first. **v1 (first run)** is archived unchanged in `results/v1/` and described at the end.

Reproduce everything with two commands from the repo root (with the dev environment active):

```bash
python -m experiments.benchmark                          # 40 seeds, several minutes on 8 cores, needs network once
python -m experiments.calibrate --update-readme README.md
```

`benchmark` downloads the three datasets from OpenML into `experiments/cache/` (git-ignored), runs every generator on every dataset and seed and writes the raw statistics to `results/runs.jsonl` plus the environment to `results/meta.json`. `calibrate` reads those, learns the thresholds, writes `app/calibrated_thresholds.json` and the evaluation files in `results/`, and pastes the detection matrix into the main README between its `detection-matrix` markers. Only the date inside the calibrated file and the run times change when the commands are repeated on the same machine and package versions.

## Files

| File | Purpose |
|---|---|
| `generators.py` | Gaussian copula and the six failure injections (`fresh` rows for the four isolated ones) |
| `benchmark.py` | Splits, runs, records raw statistics (`stats`) and provisional-threshold warning codes per run |
| `calibrate.py` | Threshold learning, test-set evaluation, leave-one-dataset-out variant |
| `results/runs.jsonl` | One line per (dataset, seed, generator): statistics, runtime, errors |
| `results/meta.json` | Datasets (id, name, version, licence), seeds, split, package versions, total runtime |
| `results/detection_matrix.md` and `.json` | Hit rate per generator and warning code on the test seeds, intended and any-warning rates, per-dataset breakdown |
| `results/detection_matrix_details.md` | Per-dataset table and the leave-one-dataset-out matrix |
| `results/summary.json` | Headline: failure modes detected, `heldout_real` false-alarm count, copula rate, leave-one-dataset-out headline |

## Design

- Datasets: Adult (OpenML id 1590, version 2), German credit (id 31, `credit-g` version 1) and CMC (id 23, version 1). Adult is subsampled to 6,000 rows per seed (stratified by label, seeded) so a full run takes minutes. All three list the licence as Public on OpenML
- Per (dataset, seed): a stratified 50 / 25 / 25 split into T (generator training set and `real_df`), H_gold (output of the `heldout_real` generator) and H_base (`holdout_df`, the privacy baseline). Every generator makes `len(H_gold)` rows, so row counts are equal across generators. `exact_copy`, `near_copy` and the copula use T. The other four injections start from H_gold (see Changes between v1 and v2)
- Seeds 0 to 39. Seeds 0 to 19 are the calibration set and seeds 20 to 39 are the test set. All headline numbers come from seeds 20 to 39. The seed controls the subsample, the split, the generator and the validator's own randomness
- Statistics recorded per run are exactly the numbers `app/checks.py` compares with thresholds (`extract_stats`), so thresholds can be recalibrated without re-running any metric. `tests/test_benchmark.py` checks that re-applying thresholds to recorded statistics gives the same warning codes as `generate_trust_report` on the same inputs
- Threshold rule and the intended-warning mapping are written in the `calibrate.py` module docstring and in the main README

## Failure injections

| Name | Construction |
|---|---|
| `exact_copy` | `len(H_gold)` rows of T sampled without replacement |
| `near_copy` | the same, plus Gaussian noise with std 1% of each numeric column's std; categoricals unchanged |
| `near_copy_5pct` | as `near_copy` with 5% (reported, not in the headline count) |
| `column_shuffle` | bootstrap of H_gold, then every column (label included) permuted independently |
| `mode_collapse` | 5 random rows of H_gold, sampled with replacement, plus 1% numeric jitter (scaled by the training column's std) |
| `category_oversample` | bootstrap of H_gold; the categorical feature with the most categories in T has 50% of its rows set to its rarest category in T |
| `label_shuffle` | bootstrap of H_gold with the label column permuted |

The Gaussian copula maps numeric columns through the empirical CDF to normal scores and categorical columns (label included) to jittered intervals by frequency, fits a multivariate normal to the scores and inverts the mapping. Integer-like columns are rounded back. Missing categoricals are their own category and missing numerics are re-created at the training missing rate.

## Changes between v1 and v2

v1 (seeds 0 to 19, even seeds calibration and odd seeds test) exposed a design flaw and two metric weaknesses. These changes were motivated by looking at the v1 test-half results, so the v1 test seeds were seen and are now part of the calibration set. To keep the evidence clean, v2 uses seeds 0 to 19 for calibration only and reports every headline number from new seeds 20 to 39, which nobody had looked at. All choices below were fixed before seeds 20 to 39 were run and are not to be re-tuned after seeing v2 results. Anything wrong found after the v2 run is documented, not fixed.

1. **Isolated failure modes.** In v1, `column_shuffle`, `mode_collapse`, `category_oversample` and `label_shuffle` bootstrapped from T, the rows the privacy check compares against, so `privacy_memorisation` fired on all of them for reasons unrelated to the injected failure and the any-warning rate was inflated. In v2 these four start from H_gold (real rows not in T). Only `exact_copy` and `near_copy` use T. The row count is still `len(H_gold)`
2. **Tie-robust privacy statistic.** The v1 privacy rule counts synthetic rows below the 5th percentile of holdout DCR. On data with many duplicate rows (CMC) that percentile is exactly 0, so a noisy copy is never counted and `near_copy` was missed on CMC. v2 adds `dcr_closeness_auc` = P(synthetic DCR < holdout DCR) + 0.5 P(equal), computed from ranks. It is 0.5 without memorisation. `privacy_memorisation` now fires if the AUC, the share or the exact-match excess exceeds its calibrated threshold. The share and exact-match statistics are still reported
3. **Balanced accuracy for utility.** Plain accuracy hides label damage on imbalanced labels (a majority-class predictor still scores about 0.76 on Adult). For classification `utility_degradation` now uses the TSTR / TRTR balanced-accuracy ratio, whose chance level is 1/k whatever the class balance. Accuracy and F1 are still reported. The threshold field was renamed `utility_balanced_accuracy_ratio`. Regression still uses the R2 drop
4. **Calibration rule unchanged.** Null extreme over the `heldout_real` calibration runs plus a margin of max(10%, 0.01). Calibration now uses 20 seeds per dataset (60 null runs) instead of 10 (30). The empirical false-alarm rate of each check on the 60 test null runs is reported
5. **Leave-one-dataset-out variant kept.** Its thresholds now use only calibration-seed null runs of the other datasets
6. **Layout.** The v1 results are archived in `results/v1/` unchanged, together with the v1 calibrated thresholds

## Changes made inside v1 (after its first run)

- `privacy_medium_share` was initially left at the provisional value, which ended up above the calibrated `privacy_high_share`. It is now set to the largest null value of the privacy statistic with no margin. It only changes the displayed risk level (MEDIUM versus LOW), never a warning, so no result in the detection matrix changed
- The shape of the generated markdown was fixed (the leave-one-dataset-out sentence had landed above its table). No numbers changed

Injection strengths, thresholds rule, seeds and intended-warning mapping were not changed after the results were seen. The weaknesses in the results (near copies on CMC, shuffled labels) are reported as they are.

## v1 (first run)

Archived unchanged in `results/v1/` (runs, matrix, summary and the v1 calibrated thresholds). Seeds 0 to 19, calibration on even seeds and test on odd seeds. The text below is the README section as it stood after that run. Its numbers are v1 numbers and are superseded by v2 above.

**Method**
- Datasets, from OpenML with pinned ids: Adult (id 1590, 48,842 rows, subsampled to 6,000 rows per seed for runtime), German credit (id 31, 1,000 rows) and CMC (id 23, 1,473 rows, 3 classes). All three list the licence as Public on OpenML
- 20 seeds per dataset (0 to 19). Per seed the rows are split at random, stratified by label, into the generator training set T (50%), a held-out gold set (25%) and a held-out baseline set (25%). The baseline set is the `holdout_df` for the privacy check. Every generator produces as many rows as the gold set, from T only
- Good generators: `heldout_real` (the gold set itself, the null) and `gaussian_copula` (written from scratch in `experiments/generators.py`; categorical columns use interval jitter, so it attenuates dependence)
- Failure injections, each from T: `exact_copy` (rows without replacement), `near_copy` (plus Gaussian noise of 1% of each numeric column's std, with `near_copy_5pct` as a stronger variant that is not counted in the headline), `column_shuffle` (every column permuted independently), `mode_collapse` (a bootstrap of 5 random rows plus 1% numeric jitter), `category_oversample` (50% of rows of the categorical feature with the most categories set to its rarest category) and `label_shuffle` (bootstrap with the label column permuted)
- Calibration: even seeds are the calibration half and odd seeds the test half. Each threshold is set from the `heldout_real` calibration runs only: for a check that warns when a statistic is high, threshold = max over those runs + max(10% of the max, 0.01), and for a check that warns when it is low, threshold = min over those runs - max(10% of the min, 0.01). The broken generators and the copula play no part in choosing thresholds. With n pooled null values a fresh null value falls beyond the observed maximum with probability 1 / (n + 1) per check, which is about 3% for n = 30, before the margin. Several checks run on each report, so the any-warning false-alarm rate is higher than any one check's
- Evaluation: the thresholds are applied to the recorded statistics of the test half. A hit means the intended warning fired. Intended warnings: `exact_copy` and `near_copy` need `privacy_memorisation`; `column_shuffle` needs `dependence_shift` or `detectability_high_auc`; `mode_collapse` needs `diversity_mode_collapse` or `diversity_low_coverage`; `category_oversample` needs any `categorical_*` code; `label_shuffle` needs `utility_degradation`. A failure mode counts as detected when its intended warning fires in at least 90% of test runs

**Calibrated thresholds** (from `app/calibrated_thresholds.json`, which also records the rule, seeds and date)

| Threshold | Provisional | Calibrated | Null values pooled |
|---|---|---|---|
| `privacy_high_share` | 0.15 | 0.099674 | 30 |
| `privacy_medium_share` | 0.1 | 0.089674 | derived (display only) |
| `privacy_exact_match_margin` | 0.05 | 0.023771 | 30 |
| `variance_ratio_low` | 0.5 | 0.769212 | 30 |
| `variance_ratio_high` | 1.5 | 1.385892 | 30 |
| `coverage_min` | 0.5 | 0.7362 | 30 |
| `numeric_max_ks` | 0.2 | 0.1364 | 30 |
| `utility_accuracy_ratio` | 0.8 | 0.800847 | 60 |
| `utility_r2_drop` | 0.2 | 0.2 | 0 (no regression dataset, provisional value kept) |
| `utility_ece_increase` | 0.1 | 0.105622 | 60 |
| `categorical_mean_l1` | 0.3 | 0.126585 | 30 |
| `categorical_mean_js` | 0.2 | 0.061616 | 30 |
| `categorical_worst_shift` | 0.25 | 0.11 | 30 |
| `dependence_mean_abs_diff` | 0.1 | 0.074027 | 30 |
| `c2st_auc` | 0.7 | 0.606936 | 30 |

**Results on the test half**


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


Per-dataset rates and a leave-one-dataset-out variant are in `results/v1/detection_matrix_details.md`.

**Reading the results**
- Exact copies, independently shuffled columns, mode collapse and a heavily oversampled category are caught in every test run on all three datasets
- Near copies are caught on Adult and German credit but never on CMC. The privacy check flags rows that sit closer to training rows than the 5th percentile of the holdout distance. In CMC many rows are duplicated (for seed 1, 8.4% of holdout rows have an exact twin in the training set), so that percentile is exactly 0 and a noisy copy, whose distance is small but not 0, is never closer. This is a weakness of the check on data with many duplicate rows
- Shuffled labels are caught by `utility_degradation` in only part of the runs (see the table), mostly on CMC. The accuracy ratio of a label-shuffled bootstrap is often not far enough below the real-data ratio to cross the calibrated threshold. The other warnings that fire on those runs are a side effect of the injection, which copies feature rows from T, and do not show the label damage
- `heldout_real` produced no warnings in the test half. The copula triggered warnings in every test run, mostly `dependence_shift` and `detectability_high_auc`. That is expected for a weak copula and says the thresholds are strict enough to separate it from real data, not that the validator misfires
- The leave-one-dataset-out thresholds are less safe: they raised a warning on some `heldout_real` test runs, so thresholds learned from other datasets do not transfer perfectly

**Limitations of this evidence**
- Only three datasets, all small or subsampled, and no regression dataset (so `utility_r2_drop` stays at its provisional value)
- The failure injections are simple and their strength was fixed before any results were seen. Detection rates depend on that strength: a milder oversample or a smaller shuffled fraction would be harder to catch
- The copula is a weak generator. No modern deep generator (CTGAN, diffusion) was tested, so the false-alarm rate on a good-but-imperfect generator is not measured
- The false-alarm rate on `heldout_real` has a small denominator (30 test runs), so a rate of 0 does not mean the true rate is 0
- The thresholds are tuned to the dataset sizes used here (a few hundred to a few thousand rows). Other sizes may need recalibration with `python -m experiments.calibrate`
