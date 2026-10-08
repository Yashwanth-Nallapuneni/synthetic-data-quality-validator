# Synthetic Data Quality Validator

A framework to evaluate synthetic tabular data across fidelity, utility, privacy and diversity. It detects numeric and categorical columns automatically and raises warnings when a metric crosses a threshold.

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
- Numeric: Kolmogorov-Smirnov statistic, Wasserstein distance and a 30-bin histogram Jensen-Shannon divergence per column
- Categorical: L1 distance, Jensen-Shannon divergence and the worst-category shift per column

Marginals only. Dependencies between columns are not checked yet.

### Utility
- Trains a model on real data and a model on synthetic data, then scores both on a held-out real split (70/30, fixed seed)
- Numeric columns are standardised and categorical columns are one-hot encoded inside an sklearn Pipeline
- The label can be any column. The task is detected from the label: small integer-coded or string labels are classification, many-valued numeric labels are regression
- Classification: logistic regression, reporting accuracy, weighted F1 and top-label Expected Calibration Error (works for binary and multiclass)
- Regression: ridge regression, reporting R2 and MAE
- Splits are stratified only when every class has at least 2 rows and the split is feasible
- Fewer than 10 labelled real rows, or a single-class real label, gives `status: "insufficient_data"` with a reason instead of an error. A single-class synthetic label is scored as a constant predictor and flagged

### Privacy
Distance to closest record (DCR) with a holdout baseline:
1. Real rows are split 50/50 into a train half and a holdout half (fixed seed)
2. Numeric columns are standardised with train statistics and categorical columns are one-hot encoded
3. For each synthetic row and each holdout row, the distance to the nearest train row is computed
4. The report gives the share of synthetic rows closer than the 5th percentile of the holdout distances (about 5% is expected without memorisation) and the exact-match rate
5. Risk level is HIGH, MEDIUM or LOW from that share. Fewer than 20 real rows gives `insufficient_data`

The baseline assumes the generator trained on the train half only. A generator trained on all real rows is still flagged when it copies rows, but the signal is weaker.

### Diversity
- Per-column variance ratio (synthetic over real) and its mean
- PCA coverage ratio on the numeric columns (not standardised yet)

### Warnings
Raised when:
- privacy risk is HIGH
- mean variance ratio is below 0.5 or above 1.5
- synthetic-trained accuracy is below 0.8 times real-trained accuracy (classification), or R2 drops by more than 0.2 (regression)
- synthetic-trained ECE is more than 0.1 above real-trained ECE
- synthetic labels contain a single class
- mean categorical L1 is above 0.3 or mean categorical JS is above 0.2
- any single category shifts by more than 0.25

The report also has a `notices` list for things that are not warnings, such as a metric skipped because the data was too small.

## Project structure

```
synthetic-validator/
├── app/
│   ├── api.py                    # FastAPI app (POST /validate, GET /health)
│   ├── cli.py                    # Command-line interface
│   ├── report.py                 # Builds the trust report and warnings
│   ├── schemas.py                # Reserved for response models
│   ├── metrics/
│   │   ├── fidelity.py           # Numeric fidelity
│   │   ├── categorical_fidelity.py
│   │   ├── utility.py            # Downstream ML utility
│   │   ├── privacy.py            # DCR privacy check
│   │   └── diversity.py
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

The two bundled files have 5 rows each, which is too few for utility and privacy, so the report lists those as notices.

### Command line

```bash
synth-validate --real data/real.csv --synth data/synth.csv --label label --out report.json
# or: python -m app.cli ...
```

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
```

`GET /health` returns `{"status": "ok"}`. `POST /validate` returns `{"severity": "clean" | "issues", "report": {...}}`. Errors:

| Status | Cause |
|---|---|
| 400 | Unparseable or empty CSV, label column missing, real and synthetic columns differ (the differing columns are listed) |
| 413 | File larger than `MAX_UPLOAD_BYTES` (default 10 MB) or more than `MAX_ROWS` rows (default 50,000) |
| 500 | Unexpected internal error |

### Run with Docker

```bash
docker compose up --build
```

The API is on http://localhost:8000 and the UI on http://localhost:8502. The compose file sets `API_URL=http://api:8000/validate` for the UI. Outside compose, the UI reads `API_URL` from the environment and defaults to the Render deployment.

## Limitations

- The warning thresholds above are fixed values that have not been calibrated against known-good and known-bad synthetic data
- Pairwise dependencies between columns are not measured, so independently shuffled columns can pass the fidelity checks
- There is no real-versus-synthetic detectability test (classifier two-sample test)
- Numeric histogram Jensen-Shannon depends on bin count and sample size
- Wasserstein distance is not standardised, so it is not comparable across columns
- PCA coverage does not standardise columns first
- Utility uses one simple model per task (logistic or ridge regression)
- The DCR baseline assumes the generator trained on the train half of the real data only
- Integer-coded columns with few values but a wide range relative to their mean (for example codes 0 to 4) are treated as numeric
- No benchmark of the validator against injected failures exists yet, so detection rates are not measured
- The Streamlit UI is not deployed

Planned work: statistically sounder metrics, a failure-injection benchmark with calibrated thresholds, and a richer UI and response schema.
