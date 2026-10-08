# Synthetic Data Quality Validator

A framework to evaluate synthetic tabular data across fidelity, dependence, detectability, utility, privacy and diversity. It detects numeric and categorical columns automatically and raises warnings when a metric crosses a threshold.

The warning thresholds are calibrated on held-out real data and the validator is tested against injected failures (see [Validation against injected failures](#validation-against-injected-failures)). The test shows it misses some failure modes, so treat the warnings as screening signals.

## Live demo and deployment

- **API (deployed on Render free tier, cold start takes about a minute):** https://synthetic-data-quality-validator.onrender.com/docs
- **Streamlit UI:** not deployed. It runs locally with Docker Compose (see [Run with Docker](#run-with-docker)).

## Motivation

Synthetic data is used to reduce privacy risk, address data scarcity and balance biased datasets. It can also fail quietly by:
- memorising real rows
- distorting categorical distributions
- degrading downstream model performance
- producing poorly calibrated predictions

This project computes a set of metrics that surface those failures before the data is used.

## What it computes

### Column type detection
`app/utils/column_types.py` is the single source of truth. Columns are numeric, categorical or empty (no values). Strings, categories and booleans are categorical. A numeric column is treated as categorical only when it has at most 20 unique values, holds integers and spans a small range relative to its mean (for example a year column). Detection runs once on the real features and the resulting lists are passed to every metric, so a column has the same type everywhere.

### Fidelity
- Numeric, per column:
  - Kolmogorov-Smirnov statistic (asymptotic method, so no scipy exact-calculation warning on large samples)
  - Wasserstein distance, raw and standardised (divided by the real column's standard deviation, so it is in units of real standard deviations and comparable across columns)
  - Jensen-Shannon distance (base 2) on 10 bins built from the real column's quantiles and shared by both samples. Counts are normalised directly with no epsilon. A fresh sample from the same distribution scores near 0 with a small positive bias that shrinks as the sample grows. The two end bins are open, so disjoint supports score below 1
- Categorical, per column: L1 distance, Jensen-Shannon distance and the worst-category shift

### Pairwise dependence
Marginal checks alone pass data whose columns were shuffled independently, so associations are compared pair by pair. The label is included as a column. Features and label are typed by the same detector (a classification label counts as categorical, a regression label as numeric).
- Numeric pairs: Pearson correlation
- Categorical pairs: bias-corrected Cramer's V (Bergsma)
- Numeric by categorical pairs: correlation ratio eta, adjusted for the number of groups
- The report gives the association matrices for real and synthetic data and their absolute difference (for a heatmap), the mean and maximum absolute difference, the worst pair and the 20 largest pair differences. Matrices are left out above 50 columns per axis
- The overall mean pools the three kinds. Their scales are similar (about 0 to 1 in magnitude) but not identical

### Detectability (C2ST)
A classifier tries to tell real rows (label 0) from synthetic rows (label 1).
- Both sides are subsampled with the seed to the same size, at most 10,000 rows each, so the classes are balanced and the cost is bounded
- The shared preprocessor (median imputation, standardisation, one-hot) sits inside a sklearn Pipeline with a HistGradientBoostingClassifier and is refit on every fold
- Stratified 5-fold cross-validation (fewer folds only for very small inputs). The report gives the mean and standard deviation of ROC AUC. Near 0.5 means the classifier cannot separate the data
- Fewer than 30 rows on either side gives `status: "insufficient_data"`
- Exact copies of real rows are not detected by this test (a copy looks like its original). The privacy check covers that case

### Utility
- TRTR (train on real, test on real) and TSTR (train on synthetic, test on real) for each model, scored on the same held-out real split (70/30, fixed seed)
- Numeric columns are standardised and categorical columns are one-hot encoded inside an sklearn Pipeline
- The label can be any column. The task is detected from the label: small integer-coded or string labels are classification, many-valued numeric labels are regression
- Classification models: logistic regression and HistGradientBoostingClassifier. Metrics: accuracy, weighted F1 and top-label Expected Calibration Error (binary and multiclass)
- Regression models: ridge regression and HistGradientBoostingRegressor. Metrics: R2 and MAE
- The output is `models.<name>.trtr`, `models.<name>.tstr` and `models.<name>.tstr_vs_trtr` (accuracy ratio and ECE increase, or R2 drop)
- Splits are stratified only when every class has at least 2 rows and the split is feasible
- Fewer than 10 labelled real rows, or a single-class real label, gives `status: "insufficient_data"` with a reason. A single-class synthetic label is scored as a constant predictor and flagged

### Privacy
Distance to closest record (DCR) with a holdout baseline. Numeric columns are standardised with train statistics and categorical columns are one-hot encoded. The report gives:
1. the synthetic to train DCR distribution (mean, median, min, 5th percentile)
2. the holdout to train DCR distribution
3. the share of synthetic rows closer to the train set than the 5th percentile of holdout DCR (about 5% is expected without memorisation)
4. the exact-match rate, next to the same rate for holdout rows

Risk level is HIGH, MEDIUM or LOW from the share in item 3 and the exact-match rate. Fewer than 20 real rows gives `insufficient_data`. Query sets above 20,000 rows are subsampled with the seed and the report says so.

Two modes:
- **Holdout provided** (`holdout_df`, `--holdout` or the `holdout` upload): the real file is the generator's training set and the holdout is real data the generator never saw. This is the sharper test. It assumes the generator really did not see the holdout
- **No holdout**: the real rows are split 50/50 (fixed seed). If the generator was trained on all of the real rows, the holdout half is not unseen, which dilutes the signal. The report adds a notice saying so

### Diversity
- Per-numeric-column variance ratio (synthetic over real) and its mean. Constant real columns are skipped
- PCA variance ratio: synthetic variance over real variance in the real data's top principal components (up to 10), after the real-fitted preprocessor (median imputation, standardisation, one-hot)
- Coverage: the fraction of real rows whose nearest synthetic row is within that real row's own distance to its 5th nearest other real row, in the same preprocessed space. Low coverage means parts of the real data have no synthetic counterpart, as in mode collapse. Only the query rows are subsampled (at most 5,000, seeded), so the estimate stays unbiased

### Warnings
Every threshold lives in `app/thresholds.py`. `Thresholds()` holds the provisional hand-picked defaults. By default the validator uses the calibrated values shipped in `app/calibrated_thresholds.json` (see below) and falls back to the provisional defaults if that file is missing. `load_thresholds(path)` and `--thresholds` accept a JSON file that overrides any of them, and the thresholds used are included in the report. Warnings are raised when:
Each warning is an object `{"code", "message", "metric", "value", "threshold"}`. The code is a stable snake_case id, `metric` is the path of the statistic that crossed the threshold and `value` and `threshold` are the numbers compared. Any warning makes the API severity `issues` and the CLI exit code 1. The checks live in `app/checks.py`.

| Code | Fires when |
|---|---|
| `privacy_memorisation` | the share of synthetic rows closer to train than the holdout 5th percentile is above `privacy_high_share` (0.15), or the exact-match rate exceeds the holdout rate by more than `privacy_exact_match_margin` (0.05) |
| `diversity_mode_collapse` | mean variance ratio is below `variance_ratio_low` (0.5) |
| `diversity_excess_variance` | mean variance ratio is above `variance_ratio_high` (1.5) |
| `diversity_low_coverage` | k-NN coverage is below `coverage_min` (0.5) |
| `numeric_marginal_shift` | the largest per-column KS statistic is above `numeric_max_ks` (0.2) |
| `categorical_mean_l1` | mean categorical L1 is above `categorical_mean_l1` (0.3) |
| `categorical_mean_js` | mean categorical JS is above `categorical_mean_js` (0.2) |
| `categorical_worst_shift` | the largest single-category share shift is above `categorical_worst_shift` (0.25) |
| `dependence_shift` | the mean absolute dependence difference is above `dependence_mean_abs_diff` (0.1) |
| `detectability_high_auc` | the C2ST AUC is above `c2st_auc` (0.7) |
| `utility_degradation` | for any model, TSTR accuracy is below `utility_accuracy_ratio` (0.8) times TRTR accuracy, or the TSTR R2 drop is above `utility_r2_drop` (0.2). The model is named in the message |
| `utility_calibration_degradation` | for any model, TSTR ECE is more than `utility_ece_increase` (0.1) above TRTR ECE |
| `utility_synthetic_single_class` | synthetic labels contain a single class |

The numbers in brackets are the provisional defaults, not the calibrated values.

The report also has a `notices` list for things that are not warnings, such as a metric skipped because the data was too small, a subsample being used or the privacy baseline mode.

## Project structure

```
synthetic-validator/
├── app/
│   ├── api.py                    # FastAPI app (POST /validate, GET /health)
│   ├── cli.py                    # Command-line interface
│   ├── errors.py                 # InputError for user-data problems
│   ├── report.py                 # Builds the trust report and warnings
│   ├── checks.py                 # Warning checks: raw statistics in, structured warnings out
│   ├── thresholds.py             # Every warning threshold and the loader
│   ├── calibrated_thresholds.json  # Thresholds learned by experiments/calibrate.py
│   ├── schemas.py                # Reserved for response models
│   ├── metrics/
│   │   ├── fidelity.py           # Numeric fidelity
│   │   ├── categorical_fidelity.py
│   │   ├── utility.py            # Downstream ML utility
│   │   ├── dependence.py         # Pairwise association differences
│   │   ├── detectability.py      # C2ST real-vs-synthetic classifier
│   │   ├── privacy.py            # DCR privacy check
│   │   └── diversity.py          # Variance, PCA and coverage
│   └── utils/
│       ├── column_types.py       # Numeric vs categorical detection
│       └── preprocessing.py      # Shared coercion and sklearn preprocessing
├── experiments/                  # Phase 3 benchmark, calibration and results
├── tests/                        # pytest suite
├── data/                         # 5-row sample files
├── ui.py                         # Streamlit front end
├── Dockerfile                    # API image
├── Dockerfile.streamlit          # UI image
├── docker-compose.yml
├── start.sh                      # Render start command
├── pyproject.toml
├── requirements.txt              # Pinned runtime dependencies
└── .github/workflows/ci.yml      # Lint and tests on push and PR
```

## Install and test

```bash
python -m venv venv && source venv/bin/activate
pip install -e ".[dev,ui]"
ruff check .
pytest -q
```

Python 3.11 or newer is required. `requirements.txt` pins the exact versions used by the Docker images.

## Usage

### Python

```python
import pandas as pd
from app.report import generate_trust_report

real = pd.read_csv("data/real.csv")
synth = pd.read_csv("data/synth.csv")

report = generate_trust_report(real, synth, label_column="label", seed=42)
print(report["warnings"])  # list of {code, message, metric, value, threshold}
print(report["notices"])
```

Pass `holdout_df=` (real rows the generator never saw) and `thresholds=load_thresholds("t.json")` for the sharper privacy baseline and custom thresholds.

The two bundled files have 5 rows each, which is too few for utility and privacy, so the report lists those as notices.

### Command line

```bash
synth-validate --real data/real.csv --synth data/synth.csv --label label --out report.json
# optional: --holdout holdout.csv --thresholds thresholds.json --seed 42
# or: python -m app.cli ...
```

The command prints a short summary (dependence, C2ST AUC, privacy, coverage, utility), then the warnings and notices.

Exit codes, for use in CI:

| Code | Meaning |
|---|---|
| 0 | No warnings |
| 1 | The report contains warnings |
| 2 | Usage or input error (bad arguments, unreadable file, missing label, column mismatch) |

### API

```bash
uvicorn app.api:app --port 8000
curl -F real=@data/real.csv -F synth=@data/synth.csv -F label_column=label \
  http://localhost:8000/validate
# add -F holdout=@holdout.csv for the sharper privacy baseline
```

`GET /health` returns `{"status": "ok"}`. `POST /validate` returns `{"severity": "clean" | "issues", "report": {...}}`. Errors:

| Status | Cause |
|---|---|
| 400 | Unparseable or empty CSV, label column missing, real and synthetic (or holdout) columns differ (the differing columns are listed). Only user-data problems (`InputError`) map to 400 |
| 413 | File larger than `MAX_UPLOAD_BYTES` (default 10 MB) or more than `MAX_ROWS` rows (default 50,000). Both apply to the optional `holdout` file too |
| 500 | Unexpected internal error |

### Run with Docker

```bash
docker compose up --build
```

The API is on http://localhost:8000 and the UI on http://localhost:8502. The compose file sets `API_URL=http://api:8000/validate` for the UI. Outside compose, the UI reads `API_URL` from the environment and defaults to the Render deployment.

## Validation against injected failures

A tool that grades synthetic data is only credible if it catches known failures and stays quiet on good data. `experiments/` tests that. Everything below was produced by `python -m experiments.benchmark && python -m experiments.calibrate --update-readme README.md` (about 3 minutes on 8 cores) and the raw results are in `experiments/results/`. See `experiments/README.md` for details.

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

<!-- detection-matrix:start -->

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

<!-- detection-matrix:end -->

Per-dataset rates and a leave-one-dataset-out variant (thresholds learned from the other two datasets only) are in `experiments/results/detection_matrix_details.md`. The machine-readable versions are `detection_matrix.json` and `summary.json`.

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

## Limitations

- The thresholds are calibrated on three datasets of a few hundred to a few thousand rows and may not transfer to other sizes or domains (see [Validation against injected failures](#validation-against-injected-failures))
- Of six injected failure modes, two are not reliably detected: near copies on data with many duplicate rows and shuffled labels
- The holdout privacy mode assumes the generator never saw the holdout. Without a holdout the baseline splits the real data in half, which dilutes the signal when the generator trained on all of it
- C2ST does not detect exact copies of real rows, and its AUC has sampling noise that is larger on small inputs
- The dependence score pools Pearson, Cramer's V and eta, which are similar in scale but not identical. Pearson only sees linear association
- Numeric JS uses the real data's quantile bins, so it is insensitive to differences inside a single bin and has a small positive sampling bias
- Utility reports two models per task and warns per model, so the chance of a spurious warning is higher than with one
- Large inputs are subsampled in places (C2ST rows, privacy query rows, coverage query rows). The report lists this in `notices`
- Integer-coded columns with few values but a wide range relative to their mean (for example codes 0 to 4) are treated as numeric
- The Streamlit UI shows the new sections as raw JSON for now and is not deployed

Planned work: a richer UI and a response schema.
