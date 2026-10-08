import io
import json
import os

import altair as alt
import pandas as pd
import requests
import streamlit as st

from app.ui_helpers import (
    PRIVACY_EXPLANATION,
    categorical_fidelity_table,
    category_share_frame,
    column_kind,
    dependence_heatmap_frame,
    detectability_table,
    dimension_verdicts,
    diversity_table,
    histogram_frame,
    numeric_fidelity_table,
    privacy_table,
    utility_table,
    warnings_table,
)

API_URL = os.environ.get(
    "API_URL", "https://synthetic-data-quality-validator.onrender.com/validate"
)

st.set_page_config(page_title="Synthetic Data Quality Validator", layout="wide")
st.title("Synthetic Data Quality Validator")
st.caption(
    "Compares a synthetic table with the real table it imitates across fidelity, "
    "dependence, detectability, utility, privacy and diversity. Metrics are computed by the "
    f"API at {API_URL}."
)


def read_csv(upload) -> pd.DataFrame | None:
    try:
        return pd.read_csv(io.BytesIO(upload.getvalue()))
    except Exception as e:  # unreadable upload: show it next to the uploader
        st.error(f"Could not read {upload.name} as CSV: {e}")
        return None


def table(df: pd.DataFrame, empty_message: str) -> None:
    if df.empty:
        st.caption(empty_message)
    else:
        st.dataframe(df, hide_index=True, width="stretch")


# Inputs

left, right = st.columns(2)
with left:
    real_file = st.file_uploader("Real dataset (CSV)", type=["csv"])
with right:
    synth_file = st.file_uploader("Synthetic dataset (CSV)", type=["csv"])
holdout_file = st.file_uploader(
    "Optional: holdout real rows the generator never saw (CSV, same columns as the real file)",
    type=["csv"],
    help="Used as the privacy baseline. Without it the real data is split 50/50, which "
    "dilutes the memorisation signal if the generator trained on all of it.",
)

real_df = read_csv(real_file) if real_file else None
synth_df = read_csv(synth_file) if synth_file else None

label_column = None
if real_df is not None:
    options = list(real_df.columns)
    default = options.index("label") if "label" in options else len(options) - 1
    label_column = st.selectbox("Label column", options, index=default)

if st.button("Validate", type="primary"):
    if real_df is None or synth_df is None or not label_column:
        st.error("Upload both files and choose the label column.")
    else:
        files = {
            "real": (real_file.name, real_file.getvalue(), "text/csv"),
            "synth": (synth_file.name, synth_file.getvalue(), "text/csv"),
        }
        if holdout_file:
            files["holdout"] = (holdout_file.name, holdout_file.getvalue(), "text/csv")
        with st.spinner("Validating synthetic data..."):
            try:
                response = requests.post(
                    API_URL, files=files, data={"label_column": label_column}, timeout=180
                )
            except requests.RequestException as e:
                st.session_state.pop("result", None)
                st.error(f"Could not reach the API: {e}")
            else:
                if response.status_code != 200:
                    st.session_state.pop("result", None)
                    st.error(f"API error ({response.status_code}): {response.text}")
                else:
                    st.session_state["result"] = {
                        "response": response.json(),
                        "real": real_df,
                        "synth": synth_df,
                        "label": label_column,
                    }

state = st.session_state.get("result")
if state:
    result, real, synth = state["response"], state["real"], state["synth"]
    report = result["report"]

    # Verdict

    n_warn = len(report["warnings"])
    if result["severity"] == "issues":
        st.error(f"Issues detected: {n_warn} warning(s). Review the table below.")
    else:
        st.success("No warnings. Every computed check is within its threshold.")
    st.caption(
        f"Schema {result.get('schema_version')}, app {result.get('app_version')}, "
        f"seed {report.get('seed')}. Warnings are screening signals, not proof of a defect."
    )
    for notice in report.get("notices", []):
        st.info(notice)

    st.subheader("Warnings")
    table(warnings_table(report), "No warnings.")

    st.subheader("Verdict by dimension")
    st.caption(
        "No aggregate score is provided by design. Each dimension is judged on its own "
        "checks. Not computed means the data was too small for that dimension."
    )
    table(dimension_verdicts(report), "")

    # Dimensions

    st.subheader("Fidelity")
    st.markdown("Numeric columns")
    table(numeric_fidelity_table(report), "No numeric feature columns.")
    st.markdown("Categorical columns")
    table(categorical_fidelity_table(report), "No categorical feature columns.")

    st.markdown("Real versus synthetic marginal")
    shared = [c for c in real.columns if c in synth.columns]
    column = st.selectbox("Column", shared, key="plot_column")
    if column:
        if column_kind(report, column, real[column]) == "numeric":
            frame = histogram_frame(real[column], synth[column])
            chart = (
                alt.Chart(frame)
                .mark_bar(opacity=0.55)
                .encode(
                    x=alt.X("bin_start:Q", title=column),
                    x2="bin_end:Q",
                    y=alt.Y("density:Q", title="Share of rows", stack=None),
                    color=alt.Color("dataset:N", title=None),
                    tooltip=["dataset", "bin_start", "bin_end", "density"],
                )
            )
        else:
            frame = category_share_frame(real[column], synth[column])
            chart = (
                alt.Chart(frame)
                .mark_bar()
                .encode(
                    x=alt.X("category:N", title=column, sort=None),
                    xOffset="dataset:N",
                    y=alt.Y("share:Q", title="Share of rows"),
                    color=alt.Color("dataset:N", title=None),
                    tooltip=["dataset", "category", "share"],
                )
            )
        st.altair_chart(chart, width="stretch")

    st.subheader("Dependence")
    dep = report["dependence"]
    if dep["status"] != "ok":
        st.caption(f"Not computed: {dep['reason']}")
    else:
        s = dep["summary"]
        st.write(
            f"Mean absolute difference {s['mean_abs_diff']:.3f}, largest {s['max_abs_diff']:.3f} "
            f"for {s['worst_pair'][0]} and {s['worst_pair'][1]}."
        )
        heat = dependence_heatmap_frame(report)
        if heat.empty:
            st.caption("Matrices were left out because the data has too many columns.")
        else:
            st.caption(
                "Absolute difference between the real and synthetic association of each "
                "column pair (darker means the pair is less faithfully reproduced)."
            )
            order = sorted(set(heat["x"]) | set(heat["y"]))
            st.altair_chart(
                alt.Chart(heat)
                .mark_rect()
                .encode(
                    x=alt.X("x:N", sort=order, title=None),
                    y=alt.Y("y:N", sort=order, title=None),
                    color=alt.Color("abs_diff:Q", title="Abs. difference"),
                    tooltip=["x", "y", "abs_diff"],
                ),
                width="stretch",
            )
        table(
            pd.DataFrame(dep["worst_pairs"]).head(10), "No column pair has a defined association."
        )

    st.subheader("Detectability")
    st.caption(
        "A classifier tries to tell real rows from synthetic rows. An AUC near 0.5 means it cannot."
    )
    det = report["detectability"]
    if det["status"] != "ok":
        st.caption(f"Not computed: {det['reason']}")
    else:
        st.caption(f"Classifier: {det['model']}.")
        table(detectability_table(report), "")

    st.subheader("Utility")
    st.caption(
        "TRTR trains on real data and TSTR trains on synthetic data. Both are scored on the "
        "same held-out real rows."
    )
    if report["utility"]["status"] != "ok":
        st.caption(f"Not computed: {report['utility']['reason']}")
    else:
        table(utility_table(report), "")

    st.subheader("Privacy")
    st.write(PRIVACY_EXPLANATION)
    priv = report["privacy"]
    if priv["status"] != "ok":
        st.caption(f"Not computed: {priv['reason']}")
    else:
        st.write(f"Risk level: {priv['privacy_risk_level']} (baseline: {priv['mode']}).")
        table(privacy_table(report), "")

    st.subheader("Diversity")
    table(diversity_table(report), "Not computed: not enough data.")

    # Raw report

    st.subheader("Report")
    payload = json.dumps(result, indent=2)
    st.download_button(
        "Download JSON report", payload, file_name="validation_report.json", mime="application/json"
    )
    with st.expander("Raw JSON"):
        st.json(result, expanded=False)
