from fastapi import FastAPI
from app.report import generate_trust_report
import pandas as pd

app = FastAPI(
    title="Synthetic Data Quality Validator",
    description="Evaluate fidelity, utility, privacy, diversity, and calibration of synthetic tabular data",
    version="1.0.0",
)