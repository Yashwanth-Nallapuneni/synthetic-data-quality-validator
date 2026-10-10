# Synthetic Data Quality Validator

[![CI](https://github.com/Yashwanth-Nallapuneni/synthetic-data-quality-validator/actions/workflows/ci.yml/badge.svg)](https://github.com/Yashwanth-Nallapuneni/synthetic-data-quality-validator/actions/workflows/ci.yml)
![Python 3.11 | 3.12 | 3.13 | 3.14](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue)
![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-261230)
![Types: mypy strict](https://img.shields.io/badge/types-mypy%20strict-2a6db2)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Test coverage of `app/` and `experiments/`: 96% (measured 2026-10-08 on Python 3.13, enforced floor 92% in CI).

A framework to evaluate synthetic tabular data across fidelity, dependence, detectability, utility, privacy and diversity. It detects numeric and categorical columns automatically and raises warnings when a metric crosses a threshold, so failures such as memorised rows, distorted categories or damaged downstream models surface before the data is used. Thresholds are calibrated on held-out real data and tested against injected failures. On held-out real data it still raises a warning in some runs, so treat the warnings as screening signals.

## Results

**Headline (v2, test seeds 20 to 39): the validator detects 6 of 6 injected failure modes (intended warning in at least 90% of held-out runs) with 4 of 60 held-out-real runs raising any warning.**

| Failure mode | Intended warning fired (fraction of 60 test runs) |
|---|---|
| `exact_copy` | 1.00 |
| `near_copy` | 1.00 |
| `column_shuffle` | 1.00 |
| `mode_collapse` | 1.00 |
| `category_oversample` | 1.00 |
| `label_shuffle` | 0.92 |
| `heldout_real` runs with any warning (false alarms) | 4 of 60 |

**Against SDMetrics** (same benchmark, same calibration rule, thresholds from seeds 0 to 19, scored on seeds 20 to 39): this tool detects 6 of 6 failure modes. SDMetrics detects 4 of 6 in its best configuration (lenient, worst column and pair) and 2 to 3 of 6 in the strict views. The trade-off is that this tool raises more false alarms: 4 of 60 `heldout_real` runs against 0 of 60 for every SDMetrics view. Mean runtime per run is 1.91 s for this tool against 11.80 s for SDMetrics.

Full detection matrix, calibration method and caveats: [docs/VALIDATION.md](docs/VALIDATION.md). Comparison details: [Comparison with SDMetrics](docs/VALIDATION.md#comparison-with-sdmetrics).

## Live demo and quickstart

- **API (Render free tier, cold start takes about a minute):** https://synthetic-data-quality-validator.onrender.com/docs
- **Streamlit UI:** not deployed. It runs locally (see [docs/USAGE.md](docs/USAGE.md#streamlit-ui)).
- **Docker:** `docker compose up --build` serves the API on http://localhost:8000 and the UI on http://localhost:8502.

Command line, with exit codes for CI (0 no warnings, 1 the report contains warnings, 2 usage or input error):

```bash
pip install -e ".[dev,ui]"
synth-validate --real data/real.csv --synth data/synth.csv --label label --out report.json
# optional: --holdout holdout.csv --thresholds thresholds.json --seed 42
echo $?
```

Python:

```python
import pandas as pd
from app.report import generate_trust_report

real = pd.read_csv("data/real.csv")
synth = pd.read_csv("data/synth.csv")

report = generate_trust_report(real, synth, label_column="label", seed=42)
print(report["warnings"])  # list of {code, message, metric, value, threshold}
```

API: `uvicorn app.api:app --port 8000`, then `curl -F real=@data/real.csv -F synth=@data/synth.csv -F label_column=label http://localhost:8000/validate`. Endpoints, response schema and limits are in [docs/API.md](docs/API.md). CI gating examples and the Streamlit UI are in [docs/USAGE.md](docs/USAGE.md).

## What it checks

| Dimension | Method | Warning codes |
|---|---|---|
| Fidelity | KS, Wasserstein and Jensen-Shannon per numeric column; L1, JS and worst-category shift per categorical column | `numeric_marginal_shift`, `categorical_mean_l1`, `categorical_mean_js`, `categorical_worst_shift` |
| Dependence | Pearson, bias-corrected Cramer's V and correlation ratio eta, compared pair by pair | `dependence_shift` |
| Detectability | C2ST: gradient boosting tries to tell real from synthetic rows, 5-fold ROC AUC | `detectability_high_auc` |
| Utility | TRTR versus TSTR for two models per task, with accuracy, balanced accuracy, ECE, R2 and MAE | `utility_degradation`, `utility_calibration_degradation`, `utility_synthetic_single_class` |
| Privacy | Distance to closest record against a holdout baseline, plus a rank-based closeness AUC | `privacy_memorisation` |
| Diversity | Variance ratio, PCA variance ratio and k-NN coverage | `diversity_mode_collapse`, `diversity_excess_variance`, `diversity_low_coverage` |

Every metric, threshold and the calibrated versus provisional values are in [docs/METRICS.md](docs/METRICS.md).

## Architecture

Column types are detected once on the real features and shared by every metric. Each metric module in `app/metrics/` returns raw statistics. `app/checks.py` turns statistics into structured warnings using the thresholds in `app/thresholds.py` (calibrated values from `app/calibrated_thresholds.json`). `app/report.py` assembles the trust report, which the CLI, the FastAPI app and the Streamlit UI all share. There is no aggregate trust score by design. Details: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Limitations

- The thresholds are calibrated on three datasets of a few hundred to a few thousand rows and may not transfer to other sizes or domains (see [docs/VALIDATION.md](docs/VALIDATION.md)). Leave-one-dataset-out thresholds raised many more warnings on real held-out data
- The privacy share rule degenerates on data with many duplicate rows (the holdout 5th percentile becomes 0). The closeness AUC covers that case
- The holdout privacy mode assumes the generator never saw the holdout. Without a holdout the baseline splits the real data in half, which dilutes the signal when the generator trained on all of it
- C2ST does not detect exact copies of real rows, and its AUC has sampling noise that is larger on small inputs
- The dependence score pools Pearson, Cramer's V and eta, which are similar in scale but not identical. Pearson only sees linear association
- Numeric JS uses the real data's quantile bins, so it is insensitive to differences inside a single bin and has a small positive sampling bias
- Utility reports two models per task and warns per model, so the chance of a spurious warning is higher than with one
- Large inputs are subsampled in places (C2ST rows, privacy query rows, coverage query rows). The report lists this in `notices`
- Integer-coded columns with few values but a wide range relative to their mean (for example codes 0 to 4) are treated as numeric
- The Streamlit UI is not deployed. It is tested by rendering the results view from real reports with Streamlit's app tester, not by a browser test
- There is no aggregate trust score by design (see [docs/USAGE.md](docs/USAGE.md#streamlit-ui))
- The package is not published to PyPI

## Project layout

- `app/`: API, CLI, report builder, checks, thresholds, schemas and the metric modules (`app/metrics/`)
- `experiments/`: benchmark, calibration, SDMetrics comparison and results
- `tests/`: pytest suite
- `docs/`: [METRICS](docs/METRICS.md), [API](docs/API.md), [USAGE](docs/USAGE.md), [VALIDATION](docs/VALIDATION.md) and [ARCHITECTURE](docs/ARCHITECTURE.md)
- `ui.py`, `Dockerfile`, `Dockerfile.streamlit`, `docker-compose.yml`, `start.sh`, `pyproject.toml`, `requirements.txt`

The full tree is in [docs/USAGE.md](docs/USAGE.md#project-structure).

## Development

Python 3.11 or newer is required. `requirements.txt` pins the exact versions used by the Docker images.

```bash
python -m venv venv && source venv/bin/activate
pip install -e ".[dev,ui]"
ruff check . && ruff format --check . && mypy && pytest -q --cov
```

## Reproduce the benchmark

```bash
python -m experiments.benchmark                          # 40 seeds, about 6 minutes on 8 cores, needs network once
python -m experiments.calibrate --update-readme docs/VALIDATION.md
python -m experiments.calibrate --update-warnings-table docs/METRICS.md
pip install -e ".[compare]" && python -m experiments.compare_sdmetrics && python -m experiments.compare_sdmetrics --worst-only \
  && python -m experiments.compare_report --update-readme docs/VALIDATION.md --update-full experiments/README.md  # about 80 minutes
```

See [experiments/README.md](experiments/README.md) for protocol details.

## License

MIT, see [LICENSE](LICENSE).
