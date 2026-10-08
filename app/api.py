import io
import logging
import os

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from app import __version__
from app.errors import InputError
from app.report import build_response, generate_trust_report
from app.schemas import HealthResponse, ValidationResponse

logger = logging.getLogger("synthetic_validator")

DEFAULT_MAX_UPLOAD_BYTES = 10 * 1024 * 1024
DEFAULT_MAX_ROWS = 50_000

app = FastAPI(
    title="Synthetic Data Quality Validator",
    description=(
        "API to evaluate fidelity, utility, privacy, and diversity of synthetic tabular data"
    ),
    version=__version__,
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


@app.get("/health", response_model=HealthResponse)
def health():
    return {"status": "ok"}


@app.post("/validate", response_model=ValidationResponse)
async def validate_synthetic_data(
    real: UploadFile = File(...),
    synth: UploadFile = File(...),
    label_column: str = Form(...),
    holdout: UploadFile | None = File(None),
):
    real_df = await _read_csv(real, "real")
    synth_df = await _read_csv(synth, "synthetic")
    holdout_df = await _read_csv(holdout, "holdout") if holdout is not None else None

    try:
        report = await run_in_threadpool(
            generate_trust_report,
            real_df,
            synth_df,
            label_column=label_column,
            holdout_df=holdout_df,
        )
    except InputError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception("Unexpected error while generating the report")
        raise HTTPException(
            status_code=500, detail="Internal error while generating the report"
        ) from e

    return build_response(report)
