# Synthetic Data Quality Validator

A framework to evaluate synthetic tabular data across fidelity, dependence, detectability, utility, privacy and diversity. It detects numeric and categorical columns automatically and raises warnings when a metric crosses a threshold.

The thresholds are not calibrated yet (see [Limitations](#limitations)). Treat the warnings as screening signals.

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
Every threshold lives in `app/thresholds.py`. They are provisional hand-picked values, not calibrated. `load_thresholds(path)` and `--thresholds` accept a JSON file that overrides any of them, and the thresholds used are included in the report. Warnings are raised when:
- privacy risk is HIGH
- mean variance ratio is outside `variance_ratio_low` to `variance_ratio_high` (0.5 to 1.5)
- for any utility model, TSTR accuracy is below `utility_accuracy_ratio` (0.8) times TRTR accuracy, or the TSTR R2 drop is above `utility_r2_drop` (0.2), or TSTR ECE is more than `utility_ece_increase` (0.1) above TRTR ECE
- synthetic labels contain a single class
- mean categorical L1 is above `categorical_mean_l1` (0.3) or mean categorical JS is above `categorical_mean_js` (0.2)
- any single category shifts by more than `categorical_worst_shift` (0.25)
- the mean absolute dependence difference is above `dependence_mean_abs_diff` (0.1)
- the C2ST AUC is above `c2st_auc` (0.7)

The report also has a `notices` list for things that are not warnings, such as a metric skipped because the data was too small, a subsample being used or the privacy baseline mode.

## Project structure

```
synthetic-validator/
├── app/
│   ├── api.py                    # FastAPI app (POST /validate, GET /health)
│   ├── cli.py                    # Command-line interface
│   ├── errors.py                 # InputError for user-data problems
│   ├── report.py                 # Builds the trust report and warnings
│   ├── thresholds.py             # Every warning threshold (provisional)
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
print(report["warnings"])
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

## Limitations

- The warning thresholds are provisional hand-picked values (see `app/thresholds.py`). They have not been calibrated against known-good and known-bad synthetic data
- No benchmark of the validator against injected failures exists yet, so detection and false-alarm rates are not measured
- The holdout privacy mode assumes the generator never saw the holdout. Without a holdout the baseline splits the real data in half, which dilutes the signal when the generator trained on all of it
- C2ST does not detect exact copies of real rows, and its AUC has sampling noise that is larger on small inputs
- The dependence score pools Pearson, Cramer's V and eta, which are similar in scale but not identical. Pearson only sees linear association
- Numeric JS uses the real data's quantile bins, so it is insensitive to differences inside a single bin and has a small positive sampling bias
- Utility reports two models per task and warns per model, so the chance of a spurious warning is higher than with one
- Large inputs are subsampled in places (C2ST rows, privacy query rows, coverage query rows). The report lists this in `notices`
- Integer-coded columns with few values but a wide range relative to their mean (for example codes 0 to 4) are treated as numeric
- The Streamlit UI shows the new sections as raw JSON for now and is not deployed

Planned work: a failure-injection benchmark with calibrated thresholds, followed by a richer UI and response schema.
