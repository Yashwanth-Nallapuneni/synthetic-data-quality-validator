import os

import requests
import streamlit as st

API_URL = os.environ.get(
    "API_URL", "https://synthetic-data-quality-validator.onrender.com/validate"
)

st.set_page_config(
    page_title="Synthetic Data Quality Validator",
    layout="centered",
)

st.title("🧪 Synthetic Data Quality Validator")

st.markdown(
    """
Upload **real** and **synthetic** tabular datasets to evaluate:
- Fidelity
- Utility
- Privacy
- Diversity
- Calibration
- Pairwise dependence
- Detectability (real-vs-synthetic classifier)
"""
)

# Inputs
real_file = st.file_uploader("Upload REAL dataset (CSV)", type=["csv"])
synth_file = st.file_uploader("Upload SYNTHETIC dataset (CSV)", type=["csv"])
holdout_file = st.file_uploader(
    "Optional: HOLDOUT real rows the generator never saw (CSV, same columns as the real file)",
    type=["csv"],
    help="Used as the privacy baseline. Without it the real data is split 50/50, which "
    "dilutes the memorisation signal if the generator trained on all of it.",
)
label_column = st.text_input("Label column name", value="label")

validate_btn = st.button("Validate")

if validate_btn:
    if not real_file or not synth_file or not label_column:
        st.error("Please upload both files and provide the label column.")
    else:
        with st.spinner("Validating synthetic data..."):
            try:
                files = {
                    "real": real_file,
                    "synth": synth_file,
                }
                if holdout_file:
                    files["holdout"] = holdout_file

                data = {
                    "label_column": label_column,
                }

                response = requests.post(API_URL, files=files, data=data, timeout=120)

                if response.status_code != 200:
                    st.error(f"API error: {response.text}")
                else:
                    result = response.json()
                    report = result["report"]

                    # Severity
                    severity = result.get("severity", "unknown")
                    if severity == "issues":
                        st.error("⚠️ Issues detected in synthetic data")
                    else:
                        st.success("✅ No major issues detected")

                    # Warnings
                    st.subheader("Warnings")
                    if report["warnings"]:
                        for w in report["warnings"]:
                            st.warning(w)
                    else:
                        st.write("No warnings")

                    for notice in report.get("notices", []):
                        st.info(notice)

                    # Key Metrics
                    st.subheader("Utility")
                    st.json(report["utility"])

                    st.subheader("Dependence")
                    st.json(report["dependence"])

                    st.subheader("Detectability")
                    st.json(report["detectability"])

                    st.subheader("Privacy")
                    st.json(report["privacy"])

                    st.subheader("Diversity")
                    st.json(report["diversity"])

                    st.subheader("Fidelity Summary")
                    st.json(report["fidelity"]["summary"])

            except Exception as e:
                st.error(str(e))