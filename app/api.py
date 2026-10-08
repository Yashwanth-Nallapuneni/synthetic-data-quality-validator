import io
import logging
import os

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.report import generate_trust_report

logger = logging.getLogger("synthetic_validator")

DEFAULT_MAX_UPLOAD_BYTES = 10 * 1024 * 1024
DEFAULT_MAX_ROWS = 50_000

app = FastAPI(
    title="Synthetic Data Quality Validator",
    description=(
        "API to evaluate fidelity, utility, privacy, and diversity of synthetic tabular data"
    ),
    version="1.1.0",
)


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def max_upload_bytes() -> int:
    """Per-file upload limit in bytes (env MAX_UPLOAD_BYTES, default 10 MB)."""
    return _int_env("MAX_UPLOAD_BYTES", DEFAULT_MAX_UPLOAD_BYTES)


def max_rows() -> int:
    """Maximum rows per dataset (env MAX_ROWS, default 50,000)."""
    return _int_env("MAX_ROWS", DEFAULT_MAX_ROWS)


async def _read_csv(upload: UploadFile, name: str) -> pd.DataFrame:
    limit = max_upload_bytes()
    raw = await upload.read(limit + 1)
    if len(raw) > limit:
        raise HTTPException(
            status_code=413,
            detail=f"The {name} file is larger than the {limit} byte upload limit",
        )
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as e:
        raise HTTPException(
            status_code=400, detail=f"The {name} file is not a valid CSV: {e}"
        ) from e
    if df.empty:
        raise HTTPException(status_code=400, detail=f"The {name} file has no data rows")
    if len(df) > max_rows():
        raise HTTPException(
            status_code=413,
            detail=f"The {name} file has {len(df)} rows; the limit is {max_rows()}",
        )
    return df


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/validate")
async def validate_synthetic_data(
    real: UploadFile = File(...),
    synth: UploadFile = File(...),
    label_column: str = Form(...),
):
    real_df = await _read_csv(real, "real")
    synth_df = await _read_csv(synth, "synthetic")

    for name, df in (("real", real_df), ("synthetic", synth_df)):
        if label_column not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Label column '{label_column}' not found in {name} data",
            )

    only_real = sorted(set(real_df.columns) - set(synth_df.columns))
    only_synth = sorted(set(synth_df.columns) - set(real_df.columns))
    if only_real or only_synth:
        raise HTTPException(
            status_code=400,
            detail=(
                "Real and synthetic data must have the same columns. "
                f"Only in real: {only_real}. Only in synthetic: {only_synth}."
            ),
        )

    try:
        report = await run_in_threadpool(
            generate_trust_report, real_df, synth_df, label_column=label_column
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception("Unexpected error while generating the report")
        raise HTTPException(
            status_code=500, detail="Internal error while generating the report"
        ) from e

    return {
        "severity": "issues" if report["warnings"] else "clean",
        "report": report,
    }
