# Usage, UI and Docker

Back to the [README](../README.md). Endpoints and the response schema are in [API.md](API.md).

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

## Python

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

## Command line

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


## Streamlit UI

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

## Run with Docker

```bash
docker compose up --build
```

The API is on http://localhost:8000 and the UI on http://localhost:8502. The compose file sets `API_URL=http://api:8000/validate` for the UI. Outside compose, the UI reads `API_URL` from the environment and defaults to the Render deployment. The UI image copies `app/` as well as `ui.py` because the UI imports its table builders from `app/ui_helpers.py`.

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
