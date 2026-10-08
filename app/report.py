from typing import Any

import pandas as pd

from app import __version__
from app.checks import evaluate_checks, extract_context, extract_stats
from app.errors import InputError
from app.metrics.categorical_fidelity import compute_categorical_fidelity
from app.metrics.dependence import compute_dependence
from app.metrics.detectability import compute_detectability
from app.metrics.diversity import compute_diversity
from app.metrics.fidelity import compute_fidelity
from app.metrics.privacy import compute_privacy_risk
from app.metrics.utility import compute_utility, detect_task
from app.schemas import SCHEMA_VERSION
from app.thresholds import Thresholds, load_thresholds
from app.utils.column_types import detect_column_types
from app.utils.preprocessing import coerce_features, to_jsonable


def _check_columns(
    left: pd.DataFrame, right: pd.DataFrame, left_name: str, right_name: str
) -> None:
    only_left = sorted(set(left.columns) - set(right.columns))
    only_right = sorted(set(right.columns) - set(left.columns))
    if only_left or only_right:
        raise InputError(
            f"{left_name.capitalize()} and {right_name} data must have the same columns. "
            f"Only in {left_name}: {only_left}. Only in {right_name}: {only_right}."
        )


def generate_trust_report(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    label_column: str,
    seed: int = 42,
    thresholds: Thresholds | None = None,
    holdout_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """
    Generates a comprehensive trust report for synthetic data.

    `holdout_df` is optional real data the generator never saw. When given, `real_df` is
    treated as the generator's training set and the holdout is the privacy baseline. When
    omitted, the privacy baseline is made by splitting `real_df` in half (see the notice in
    the report).

    Raises InputError for problems with the supplied data. Warning thresholds come from
    `thresholds` (the calibrated file when omitted, else the provisional defaults, see
    app/thresholds.py).
    """
    th = thresholds or load_thresholds()

    named = [("real", real_df), ("synthetic", synth_df)]
    if holdout_df is not None:
        named.append(("holdout", holdout_df))
    for name, df in named:
        if label_column not in df.columns:
            raise InputError(f"Label column '{label_column}' not found in {name} data")
    _check_columns(real_df, synth_df, "real", "synthetic")
    if holdout_df is not None:
        _check_columns(real_df, holdout_df, "real", "holdout")

    real_features = real_df.drop(columns=[label_column])
    synth_features = synth_df.drop(columns=[label_column])

    # One source of truth for column types, computed on the real features.
    col_types = detect_column_types(real_features)
    numeric_cols = col_types["numeric"]
    categorical_cols = col_types["categorical"]

    real_features = coerce_features(real_features, numeric_cols, categorical_cols)
    synth_features = coerce_features(synth_features, numeric_cols, categorical_cols)
    holdout_features = (
        coerce_features(holdout_df.drop(columns=[label_column]), numeric_cols, categorical_cols)
        if holdout_df is not None
        else None
    )

    # Features plus the label, for the metrics that look at the joint distribution.
    joint_numeric, joint_categorical = list(numeric_cols), list(categorical_cols)
    real_label = real_df[label_column].dropna()
    if not real_label.empty:
        if detect_task(real_label) == "regression":
            joint_numeric.append(label_column)
        else:
            joint_categorical.append(label_column)
    real_joint = coerce_features(real_df, joint_numeric, joint_categorical)
    synth_joint = coerce_features(synth_df, joint_numeric, joint_categorical)

    report: dict[str, Any] = {"seed": seed, "notices": []}
    notices: list[str] = report["notices"]
    if col_types["empty"]:
        notices.append(f"Ignored columns with no values in the real data: {col_types['empty']}")

    # 1. Fidelity (numeric + categorical)

    # Numeric fidelity
    numeric_fidelity = (
        compute_fidelity(
            real_features[numeric_cols],
            synth_features[numeric_cols],
        )
        if numeric_cols
        else None
    )

    # Categorical fidelity
    categorical_fidelity = (
        compute_categorical_fidelity(
            real_features,
            synth_features,
            categorical_cols,
        )
        if categorical_cols
        else None
    )

    report["fidelity"] = {
        "numeric": numeric_fidelity,
        "categorical": categorical_fidelity,
        "summary": {
            "num_numeric_features": len(numeric_cols),
            "num_categorical_features": len(categorical_cols),
        },
    }

    # 2. Dependence (feature and label associations)

    report["dependence"] = compute_dependence(
        real_joint, synth_joint, joint_numeric, joint_categorical
    )
    if report["dependence"]["status"] != "ok":
        notices.append(f"Dependence not computed: {report['dependence']['reason']}")

    # 3. Detectability (C2ST)

    report["detectability"] = compute_detectability(
        real_joint, synth_joint, joint_numeric, joint_categorical, seed=seed
    )
    if report["detectability"]["status"] != "ok":
        notices.append(f"Detectability not computed: {report['detectability']['reason']}")
    elif report["detectability"]["subsampled"]:
        d = report["detectability"]
        notices.append(
            f"Detectability used a seeded subsample of {d['n_per_side']} rows per side "
            f"(real {d['n_real_available']}, synthetic {d['n_synthetic_available']})."
        )

    # 4. Utility

    report["utility"] = compute_utility(
        real_df,
        synth_df,
        label_column,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        seed=seed,
    )
    if report["utility"]["status"] != "ok":
        notices.append(f"Utility not computed: {report['utility']['reason']}")

    # 5. Privacy

    report["privacy"] = compute_privacy_risk(
        real_features,
        synth_features,
        numeric_cols,
        categorical_cols,
        seed=seed,
        thresholds=th,
        holdout_df=holdout_features,
    )
    privacy = report["privacy"]
    if privacy["status"] == "ok":
        if privacy["mode"] == "split_real_50_50":
            notices.append(
                "No holdout was provided, so the privacy baseline splits the real data 50/50. "
                "If the generator was trained on all of the real data, the holdout half is "
                "not unseen and the memorisation signal is diluted. Provide a holdout the "
                "generator never saw for a sharper test."
            )
        else:
            notices.append(
                "Privacy used the real data as the generator's training set and the provided "
                "holdout as the baseline. This assumes the generator never saw the holdout."
            )
        if privacy["subsampled"]:
            notices.append(
                f"Privacy used seeded subsamples of {privacy['n_synthetic_queried']} synthetic "
                f"and {privacy['n_holdout_queried']} holdout query rows."
            )
    else:
        notices.append(f"Privacy not computed: {privacy['reason']}")

    # 6. Diversity

    report["diversity"] = compute_diversity(
        real_features, synth_features, numeric_cols, categorical_cols, seed=seed
    )
    coverage = report["diversity"]["coverage"]
    if coverage and coverage["subsampled"]:
        notices.append(
            f"Coverage queried a seeded subsample of {coverage['n_real_queried']} real rows."
        )

    # 7. Warnings (see app/checks.py)

    warnings = evaluate_checks(extract_stats(report), th, extract_context(report))

    report["warnings"] = warnings
    report["thresholds"] = th.to_dict()
    jsonable: dict[str, Any] = to_jsonable(report)
    return jsonable


def build_response(report: dict[str, Any]) -> dict[str, Any]:
    """Wrap a report in the versioned envelope shared by the API and the CLI JSON output."""
    return {
        "schema_version": SCHEMA_VERSION,
        "app_version": __version__,
        "severity": "issues" if report["warnings"] else "clean",
        "report": report,
    }
