# Metrics and warnings

Full description of what the validator computes and when it warns. Back to the [README](../README.md).

## Motivation

Synthetic data is used to reduce privacy risk, address data scarcity and balance biased datasets. It can also fail quietly by:
- memorising real rows
- distorting categorical distributions
- degrading downstream model performance
- producing poorly calibrated predictions

This project computes a set of metrics that surface those failures before the data is used.

## What it computes

#### Column type detection
`app/utils/column_types.py` is the single source of truth. Columns are numeric, categorical or empty (no values). Strings, categories and booleans are categorical. A numeric column is treated as categorical only when it has at most 20 unique values, holds integers and spans a small range relative to its mean (for example a year column). Detection runs once on the real features and the resulting lists are passed to every metric, so a column has the same type everywhere.

#### Fidelity
- Numeric, per column:
  - Kolmogorov-Smirnov statistic (asymptotic method, so no scipy exact-calculation warning on large samples)
  - Wasserstein distance, raw and standardised (divided by the real column's standard deviation, so it is in units of real standard deviations and comparable across columns)
  - Jensen-Shannon distance (base 2) on 10 bins built from the real column's quantiles and shared by both samples. Counts are normalised directly with no epsilon. A fresh sample from the same distribution scores near 0 with a small positive bias that shrinks as the sample grows. The two end bins are open, so disjoint supports score below 1
- Categorical, per column: L1 distance, Jensen-Shannon distance and the worst-category shift

#### Pairwise dependence
Marginal checks alone pass data whose columns were shuffled independently, so associations are compared pair by pair. The label is included as a column. Features and label are typed by the same detector (a classification label counts as categorical, a regression label as numeric).
- Numeric pairs: Pearson correlation
- Categorical pairs: bias-corrected Cramer's V (Bergsma)
- Numeric by categorical pairs: correlation ratio eta, adjusted for the number of groups
- The report gives the association matrices for real and synthetic data and their absolute difference (for a heatmap), the mean and maximum absolute difference, the worst pair and the 20 largest pair differences. Matrices are left out above 50 columns per axis
- The overall mean pools the three kinds. Their scales are similar (about 0 to 1 in magnitude) but not identical

#### Detectability (C2ST)
A classifier tries to tell real rows (label 0) from synthetic rows (label 1).
- Both sides are subsampled with the seed to the same size, at most 10,000 rows each, so the classes are balanced and the cost is bounded
- The shared preprocessor (median imputation, standardisation, one-hot) sits inside a sklearn Pipeline with a HistGradientBoostingClassifier and is refit on every fold
- Stratified 5-fold cross-validation (fewer folds only for very small inputs). The report gives the mean and standard deviation of ROC AUC. Near 0.5 means the classifier cannot separate the data
- Fewer than 30 rows on either side gives `status: "insufficient_data"`
- Exact copies of real rows are not detected by this test (a copy looks like its original). The privacy check covers that case

#### Utility
- TRTR (train on real, test on real) and TSTR (train on synthetic, test on real) for each model, scored on the same held-out real split (70/30, fixed seed)
- Numeric columns are standardised and categorical columns are one-hot encoded inside an sklearn Pipeline
- The label can be any column. The task is detected from the label: small integer-coded or string labels are classification, many-valued numeric labels are regression
- Classification models: logistic regression and HistGradientBoostingClassifier. Metrics: accuracy, balanced accuracy, weighted F1 and top-label Expected Calibration Error (binary and multiclass)
- Regression models: ridge regression and HistGradientBoostingRegressor. Metrics: R2 and MAE
- The output is `models.<name>.trtr`, `models.<name>.tstr` and `models.<name>.tstr_vs_trtr` (accuracy ratio, balanced-accuracy ratio and ECE increase, or R2 drop)
- Splits are stratified only when every class has at least 2 rows and the split is feasible
- Fewer than 10 labelled real rows, or a single-class real label, gives `status: "insufficient_data"` with a reason. A single-class synthetic label is scored as a constant predictor and flagged

#### Privacy
Distance to closest record (DCR) with a holdout baseline. Numeric columns are standardised with train statistics and categorical columns are one-hot encoded. The report gives:
1. the synthetic to train DCR distribution (mean, median, min, 5th percentile)
2. the holdout to train DCR distribution
3. the share of synthetic rows closer to the train set than the 5th percentile of holdout DCR (about 5% is expected without memorisation)
4. the exact-match rate, next to the same rate for holdout rows
5. the closeness AUC: P(synthetic DCR < holdout DCR) + 0.5 P(equal), computed from ranks. It is 0.5 without memorisation and above 0.5 when synthetic rows sit closer to train than unseen real rows. Unlike item 3 it still works when many rows are duplicated and the holdout 5th percentile is 0

Risk level is HIGH, MEDIUM or LOW from the share in item 3, the exact-match rate and the closeness AUC. Fewer than 20 real rows gives `insufficient_data`. Query sets above 20,000 rows are subsampled with the seed and the report says so.

Two modes:
- **Holdout provided** (`holdout_df`, `--holdout` or the `holdout` upload): the real file is the generator's training set and the holdout is real data the generator never saw. This is the sharper test. It assumes the generator really did not see the holdout
- **No holdout**: the real rows are split 50/50 (fixed seed). If the generator was trained on all of the real rows, the holdout half is not unseen, which dilutes the signal. The report adds a notice saying so

#### Diversity
- Per-numeric-column variance ratio (synthetic over real) and its mean. Constant real columns are skipped
- PCA variance ratio: synthetic variance over real variance in the real data's top principal components (up to 10), after the real-fitted preprocessor (median imputation, standardisation, one-hot)
- Coverage: the fraction of real rows whose nearest synthetic row is within that real row's own distance to its 5th nearest other real row, in the same preprocessed space. Low coverage means parts of the real data have no synthetic counterpart, as in mode collapse. Only the query rows are subsampled (at most 5,000, seeded), so the estimate stays unbiased

#### Warnings
Every threshold lives in `app/thresholds.py`. `Thresholds()` holds the provisional hand-picked defaults. By default the validator uses the calibrated values shipped in `app/calibrated_thresholds.json` (see below) and falls back to the provisional defaults if that file is missing. `load_thresholds(path)` and `--thresholds` accept a JSON file that overrides any of them, and the thresholds used are included in the report.
Each warning is an object `{"code", "message", "metric", "value", "threshold"}`. The code is a stable snake_case id, `metric` is the path of the statistic that crossed the threshold and `value` and `threshold` are the numbers compared. Any warning makes the API severity `issues` and the CLI exit code 1. The checks live in `app/checks.py`.

<!-- warnings-table:start -->

| Code | Fires when | Calibrated default | Provisional fallback |
|---|---|---|---|
| `privacy_memorisation` | any of: the share of synthetic rows closer to train than the holdout 5th percentile is above `privacy_high_share`; the exact-match rate exceeds the holdout rate by more than `privacy_exact_match_margin`; the closeness AUC (P(synthetic DCR < holdout DCR) with ties counted half) is above `privacy_closeness_auc` | `privacy_high_share`: 0.09967<br>`privacy_exact_match_margin`: 0.03186<br>`privacy_closeness_auc`: 0.6209 | `privacy_high_share`: 0.15<br>`privacy_exact_match_margin`: 0.05<br>`privacy_closeness_auc`: 0.6 |
| `diversity_mode_collapse` | mean variance ratio is below `variance_ratio_low` | `variance_ratio_low`: 0.7668 | `variance_ratio_low`: 0.5 |
| `diversity_excess_variance` | mean variance ratio is above `variance_ratio_high` | `variance_ratio_high`: 1.386 | `variance_ratio_high`: 1.5 |
| `diversity_low_coverage` | k-NN coverage is below `coverage_min` | `coverage_min`: 0.7362 | `coverage_min`: 0.5 |
| `numeric_marginal_shift` | the largest per-column KS statistic is above `numeric_max_ks` | `numeric_max_ks`: 0.1364 | `numeric_max_ks`: 0.2 |
| `categorical_mean_l1` | mean categorical L1 is above `categorical_mean_l1` | `categorical_mean_l1`: 0.1266 | `categorical_mean_l1`: 0.3 |
| `categorical_mean_js` | mean categorical JS is above `categorical_mean_js` | `categorical_mean_js`: 0.06209 | `categorical_mean_js`: 0.2 |
| `categorical_worst_shift` | the largest single-category share shift is above `categorical_worst_shift` | `categorical_worst_shift`: 0.11 | `categorical_worst_shift`: 0.25 |
| `dependence_shift` | the mean absolute dependence difference is above `dependence_mean_abs_diff` | `dependence_mean_abs_diff`: 0.07403 | `dependence_mean_abs_diff`: 0.1 |
| `detectability_high_auc` | the C2ST AUC is above `c2st_auc` | `c2st_auc`: 0.6125 | `c2st_auc`: 0.7 |
| `utility_degradation` | for any model, TSTR balanced accuracy is below `utility_balanced_accuracy_ratio` times TRTR balanced accuracy (balanced accuracy has chance level 1/k whatever the class balance, so a majority-class predictor does not hide label damage), or the TSTR R2 drop is above `utility_r2_drop`. The model is named in the message | `utility_balanced_accuracy_ratio`: 0.7711<br>`utility_r2_drop`: 0.2 (not calibrated) | `utility_balanced_accuracy_ratio`: 0.8<br>`utility_r2_drop`: 0.2 |
| `utility_calibration_degradation` | for any model, TSTR ECE is more than `utility_ece_increase` above TRTR ECE | `utility_ece_increase`: 0.1056 | `utility_ece_increase`: 0.1 |
| `utility_synthetic_single_class` | synthetic labels contain a single class | n/a | n/a |

<!-- warnings-table:end -->

The table names the threshold keys each warning uses. The two value columns are generated by `python -m experiments.calibrate --update-warnings-table docs/METRICS.md` and checked by a test, so they cannot drift from the code. "Calibrated default" is the value in `app/calibrated_thresholds.json`, which the validator uses unless you pass `--thresholds`. "Provisional fallback" is the hand-picked `Thresholds()` default, used only if that file is missing. Values are rounded to 4 significant digits. `utility_r2_drop` is not calibrated because there is no regression dataset, so its calibrated default is the provisional value.

The report also has a `notices` list for things that are not warnings, such as a metric skipped because the data was too small, a subsample being used or the privacy baseline mode.
