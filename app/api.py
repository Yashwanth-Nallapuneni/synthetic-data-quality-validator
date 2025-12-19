from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import pandas as pd
import io

from app.report import generate_trust_report

app = FastAPI(
    title="Synthetic Data Quality Validator",
    description="API to evaluate fidelity, utility, privacy, and diversity of synthetic tabular data",
    version="1.0.0",
)


@app.post("/validate")
async def validate_synthetic_data(
    real: UploadFile = File(...),
    synth: UploadFile = File(...),
    label_column: str = Form(...),
):
    try:
        real_bytes = await real.read()
        synth_bytes = await synth.read()

        real_df = pd.read_csv(io.BytesIO(real_bytes))
        synth_df = pd.read_csv(io.BytesIO(synth_bytes))

        if label_column not in real_df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Label column '{label_column}' not found in real data",
            )

        if label_column not in synth_df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Label column '{label_column}' not found in synthetic data",
            )

        report = generate_trust_report(
            real_df,
            synth_df,
            label_column=label_column,
        )

        severity = "clean"
        if report["warnings"]:
            severity = "issues"

        return {
            "severity": severity,
            "report": report,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))