# API reference

Back to the [README](../README.md).

## Endpoints

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
