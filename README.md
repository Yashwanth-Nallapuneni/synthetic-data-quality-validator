# Synthetic Data Quality Validator

[![CI](https://github.com/Yashwanth-Nallapuneni/synthetic-data-quality-validator/actions/workflows/ci.yml/badge.svg)](https://github.com/Yashwanth-Nallapuneni/synthetic-data-quality-validator/actions/workflows/ci.yml)
![Python 3.11 | 3.12 | 3.13 | 3.14](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue)
![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-261230)
![Types: mypy strict](https://img.shields.io/badge/types-mypy%20strict-2a6db2)

Test coverage of `app/` and `experiments/`: 96% (measured 2026-10-08 on Python 3.13, enforced floor 92% in CI).

A framework to evaluate synthetic tabular data across fidelity, dependence, detectability, utility, privacy and diversity. It detects numeric and categorical columns automatically and raises warnings when a metric crosses a threshold.

The warning thresholds are calibrated on held-out real data and the validator is tested against injected failures (see [Validation against injected failures](#validation-against-injected-failures)). On held-out real data it still raises a warning in some runs, so treat the warnings as screening signals.

## Live demo and deployment

- **API (deployed on Render free tier, cold start takes about a minute):** https://synthetic-data-quality-validator.onrender.com/docs
- **Streamlit UI:** not deployed. It runs locally (see [Streamlit UI](#streamlit-ui) and [Run with Docker](#run-with-docker)).

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
- Classification models: logistic regression and HistGradientBoostingClassifier. Metrics: accuracy, balanced accuracy, weighted F1 and top-label Expected Calibration Error (binary and multiclass)
- Regression models: ridge regression and HistGradientBoostingRegressor. Metrics: R2 and MAE
- The output is `models.<name>.trtr`, `models.<name>.tstr` and `models.<name>.tstr_vs_trtr` (accuracy ratio, balanced-accuracy ratio and ECE increase, or R2 drop)
- Splits are stratified only when every class has at least 2 rows and the split is feasible
- Fewer than 10 labelled real rows, or a single-class real label, gives `status: "insufficient_data"` with a reason. A single-class synthetic label is scored as a constant predictor and flagged

### Privacy
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

### Diversity
- Per-numeric-column variance ratio (synthetic over real) and its mean. Constant real columns are skipped
- PCA variance ratio: synthetic variance over real variance in the real data's top principal components (up to 10), after the real-fitted preprocessor (median imputation, standardisation, one-hot)
- Coverage: the fraction of real rows whose nearest synthetic row is within that real row's own distance to its 5th nearest other real row, in the same preprocessed space. Low coverage means parts of the real data have no synthetic counterpart, as in mode collapse. Only the query rows are subsampled (at most 5,000, seeded), so the estimate stays unbiased

### Warnings
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

The table names the threshold keys each warning uses. The two value columns are generated by `python -m experiments.calibrate --update-warnings-table README.md` and checked by a test, so they cannot drift from the code. "Calibrated default" is the value in `app/calibrated_thresholds.json`, which the validator uses unless you pass `--thresholds`. "Provisional fallback" is the hand-picked `Thresholds()` default, used only if that file is missing. Values are rounded to 4 significant digits. `utility_r2_drop` is not calibrated because there is no regression dataset, so its calibrated default is the provisional value.

The report also has a `notices` list for things that are not warnings, such as a metric skipped because the data was too small, a subsample being used or the privacy baseline mode.

## Project structure

```
synthetic-validator/
├── app/
│   ├── __init__.py               # __version__, the single source of the package version
│   ├── api.py                    # FastAPI app (POST /validate, GET /health)
│   ├── cli.py                    # Command-line interface
│   ├── errors.py                 # InputError for user-data problems
│   ├── report.py                 # Builds the trust report and warnings
│   ├── checks.py                 # Warning checks: raw statistics in, structured warnings out
│   ├── thresholds.py             # Every warning threshold and the loader
│   ├── calibrated_thresholds.json  # Thresholds learned by experiments/calibrate.py
│   ├── schemas.py                # Pydantic response models and SCHEMA_VERSION
│   ├── ui_helpers.py             # Pure table and plot-data builders for the UI
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
├── ui.py                         # Streamlit front end (imports app/ui_helpers.py)
├── Dockerfile                    # API image
├── Dockerfile.streamlit          # UI image
├── docker-compose.yml
├── start.sh                      # Render start command
├── pyproject.toml
├── requirements.txt              # Pinned runtime dependencies
└── .github/workflows/ci.yml      # Lint, type check and tests with coverage on push and PR
```

## Install and test

```bash
python -m venv venv && source venv/bin/activate
pip install -e ".[dev,ui]"
ruff check .
ruff format --check .
mypy
pytest -q --cov
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

The command prints a short summary (dependence, C2ST AUC, privacy, coverage, utility), then the warnings and notices. `--out` writes the same versioned JSON envelope the API returns (`schema_version`, `app_version`, `severity`, `report`).

Exit codes, for use in CI:

| Code | Meaning |
|---|---|
| 0 | No warnings |
| 1 | The report contains warnings |
| 2 | Usage or input error (bad arguments, unreadable file, missing label, column mismatch) |

Gate a pipeline on the exit code. Warnings are screening signals, so decide whether code 1 should fail the build or only be reported:

```yaml
# GitHub Actions step
- name: Validate synthetic data
  run: |
    synth-validate --real data/real.csv --synth data/synth.csv --label label --out report.json
  # exit 0 passes the step, 1 (warnings) and 2 (input error) fail it
```

```bash
# Fail only on input errors and keep warnings as a report
synth-validate --real real.csv --synth synth.csv --label label --out report.json
status=$?
if [ "$status" -eq 2 ]; then exit 2; fi
```

### API

```bash
uvicorn app.api:app --port 8000
curl -F real=@data/real.csv -F synth=@data/synth.csv -F label_column=label \
  http://localhost:8000/validate
# add -F holdout=@holdout.csv for the sharper privacy baseline
```

`GET /health` returns `{"status": "ok"}`. `POST /validate` returns a versioned envelope:

```json
{
  "schema_version": "2.0",
  "app_version": "2.0.0",
  "severity": "clean",
  "report": { "seed": 42, "fidelity": {}, "dependence": {}, "detectability": {},
              "utility": {}, "privacy": {}, "diversity": {},
              "warnings": [], "notices": [], "thresholds": {} }
}
```

- `schema_version` is the version of the JSON layout (`SCHEMA_VERSION` in `app/schemas.py`). A key that is removed or changes meaning bumps the major part and a new key bumps the minor part. `app_version` is the package version, defined once in `app/__init__.py` and read by `pyproject.toml`
- `severity` is `issues` when `report.warnings` is not empty and `clean` otherwise
- The response models are in `app/schemas.py`, wired into the routes with `response_model=` and published at `/docs` and `/openapi.json`. They are strict: an unknown key fails validation, and a test checks real reports from binary, multiclass, regression, tiny, holdout and categorical-only data against them. Only the maps keyed by data (per-feature statistics by column name and per-model utility statistics by model name) are open-ended
- Each of `dependence`, `detectability`, `utility` and `privacy` has `status` (`ok` or `insufficient_data`) and `reason`. When the status is `insufficient_data` the metric fields are absent. `privacy_risk_level` is then `UNKNOWN`
- `warnings` is a list of `{code, message, metric, value, threshold}`
- From version 2.0 the CLI `--out` file is this same envelope. Earlier files held the bare report

Errors:

| Status | Cause |
|---|---|
| 400 | Unparseable or empty CSV, label column missing, real and synthetic (or holdout) columns differ (the differing columns are listed). Only user-data problems (`InputError`) map to 400 |
| 413 | File larger than `MAX_UPLOAD_BYTES` (default 10 MB) or more than `MAX_ROWS` rows (default 50,000). Both apply to the optional `holdout` file too |
| 500 | Unexpected internal error |

### Streamlit UI

```bash
pip install -e ".[ui]"
API_URL=http://localhost:8000/validate streamlit run ui.py
```

The upload widgets are limited to 10 MB per file (`.streamlit/config.toml`, `maxUploadSize = 10`) to match the API default for `MAX_UPLOAD_BYTES`. If you raise `MAX_UPLOAD_BYTES`, raise `maxUploadSize` (in MB) too.

Upload the real and synthetic CSV files, optionally a holdout CSV, pick the label column from a selectbox filled from the real file and press Validate. The metrics are computed by the API. The UI only reads the uploaded files itself to draw the marginal plot. It shows:
- a verdict banner (clean or issues), the notices and a table of warnings with code, message, value and threshold
- a per-dimension verdict table (pass, warn or not computed) that lists which checks fired
- per-feature numeric fidelity (KS, standardised Wasserstein, JS) and categorical fidelity tables
- a real versus synthetic marginal plot for a chosen column (histogram for numeric columns and grouped bars for categorical columns)
- the dependence section with the absolute association-difference heatmap and the worst pairs
- the detectability AUC, the utility table (TRTR versus TSTR for every model and metric), the privacy table with a plain-language explanation of DCR, and the diversity table
- a JSON download button and a collapsible raw JSON view

There is no aggregate trust score, by design. A single number hides which dimension failed and would need weights that this project has no data to justify, so each dimension gets its own verdict instead. The table logic is in `app/ui_helpers.py` and is unit-tested without Streamlit.

### Run with Docker

```bash
docker compose up --build
```

The API is on http://localhost:8000 and the UI on http://localhost:8502. The compose file sets `API_URL=http://api:8000/validate` for the UI. Outside compose, the UI reads `API_URL` from the environment and defaults to the Render deployment. The UI image copies `app/` as well as `ui.py` because the UI imports its table builders from `app/ui_helpers.py`.

## Validation against injected failures

A tool that grades synthetic data is only credible if it catches known failures and stays quiet on good data. `experiments/` tests that. The results below are from protocol **v2**. Everything was produced by `python -m experiments.benchmark && python -m experiments.calibrate --update-readme README.md` (about 6 minutes on 8 cores) and the raw results are in `experiments/results/`. See `experiments/README.md` for the details.

**Headline (v2, test seeds 20 to 39): the validator detects 6 of 6 injected failure modes (intended warning in at least 90% of held-out runs) with 4 of 60 held-out-real runs raising any warning.**

**v1 to v2.** The first run (v1, archived in `experiments/results/v1/`) found 4 of 6 failure modes and exposed a flaw in how four failures were injected plus two metric weaknesses. v2 fixes those and tests on new seeds that nobody had seen. The v1 test seeds were used to motivate the fixes, so they now count as calibration data. The full list is in [Changes between v1 and v2](experiments/README.md#changes-between-v1-and-v2).

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

SDMetrics 0.32.0 (DataCebo, MIT licence) is the incumbent open-source library for this job. I ran it on the same failure-injection benchmark: the same tables for every (dataset, seed, generator), the same column types, and its thresholds calibrated with exactly the rule used for this tool (`heldout_real` seeds 0 to 19, null extreme minus or plus max(10%, 0.01)) and then scored on seeds 20 to 39. SDMetrics checks are mapped to the failure modes as memorisation to DCR overfitting protection or new row synthesis, categorical shift to Column Shapes, dependence to Column Pair Trends or LogisticDetection, and label damage to ML efficacy. SDMetrics has no diversity or collapse check, so the strict view gives `mode_collapse` no intended check, while the lenient view accepts any SDMetrics check. The last column uses only the single overall Quality score, how it is often used.

<!-- sdmetrics-comparison:start -->

Test seeds 20 to 39, 60 runs per generator. Cells are the fraction of test runs in which the intended check fired ("detected" means at least 90%). Both tools' thresholds come from `heldout_real` seeds 0 to 19 with the same rule.

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

<!-- sdmetrics-comparison:end -->

**Takeaway.** On this benchmark this validator detects more failure modes than SDMetrics, with the largest gaps on `category_oversample`, `mode_collapse` (no SDMetrics diversity check), `label_shuffle` and `column_shuffle`, and both tools catch exact and near copies. The price is a higher false-alarm count (4 of 60 `heldout_real` runs against 0 of 60), partly because SDMetrics' pooled thresholds are wide (its worst real-data score sets the cut-off for all three datasets). SDMetrics is faster only on small data: its DCR metric makes it about 6 times slower per run here. Caveats: the benchmark was built around this tool's checks, the SDMetrics mapping and ML efficacy ratio are my choices, and per-dataset SDMetrics thresholds might do better. The other differences are not about detection: per-check calibrated thresholds with a stated false-alarm target, warnings that name the metric and value, a holdout privacy baseline with a tie-robust rank statistic, and an API and CLI. Method details, per-dataset numbers and runtimes are in [Comparison with SDMetrics](experiments/README.md#comparison-with-sdmetrics). To reproduce: `pip install -e ".[compare]" && python -m experiments.compare_sdmetrics && python -m experiments.compare_report --update-readme README.md --update-full experiments/README.md` (about 80 minutes on 3 jobs).

## Limitations

- The thresholds are calibrated on three datasets of a few hundred to a few thousand rows and may not transfer to other sizes or domains (see [Validation against injected failures](#validation-against-injected-failures)). Leave-one-dataset-out thresholds raised many more warnings on real held-out data
- The privacy share rule degenerates on data with many duplicate rows (the holdout 5th percentile becomes 0). The closeness AUC covers that case
- The holdout privacy mode assumes the generator never saw the holdout. Without a holdout the baseline splits the real data in half, which dilutes the signal when the generator trained on all of it
- C2ST does not detect exact copies of real rows, and its AUC has sampling noise that is larger on small inputs
- The dependence score pools Pearson, Cramer's V and eta, which are similar in scale but not identical. Pearson only sees linear association
- Numeric JS uses the real data's quantile bins, so it is insensitive to differences inside a single bin and has a small positive sampling bias
- Utility reports two models per task and warns per model, so the chance of a spurious warning is higher than with one
- Large inputs are subsampled in places (C2ST rows, privacy query rows, coverage query rows). The report lists this in `notices`
- Integer-coded columns with few values but a wide range relative to their mean (for example codes 0 to 4) are treated as numeric
- The Streamlit UI is not deployed. It is tested by rendering the results view from real reports with Streamlit's app tester, not by a browser test
- There is no aggregate trust score by design (see [Streamlit UI](#streamlit-ui))
- The package is not published to PyPI
