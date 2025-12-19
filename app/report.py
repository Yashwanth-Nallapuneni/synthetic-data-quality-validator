import pandas as pd

from app.utils.column_types import detect_column_types
from app.metrics.categorical_fidelity import compute_categorical_fidelity

from app.metrics.fidelity import compute_fidelity
from app.metrics.utility import compute_utility
from app.metrics.privacy import compute_privacy_risk
from app.metrics.diversity import compute_diversity


def generate_trust_report(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    label_column: str,
):
    """
    Generates a comprehensive trust report for synthetic data.
    """

    # Split features / labels
    real_features = real_df.drop(columns=[label_column])
    synth_features = synth_df.drop(columns=[label_column])

    report = {}


    # 1. Fidelity (numeric + categorical)

    col_types = detect_column_types(real_features)

    numeric_cols = col_types["numeric"]
    categorical_cols = col_types["categorical"]

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


    # 2. Utility

    report["utility"] = compute_utility(real_df, synth_df)


    # 3. Privacy

    report["privacy"] = compute_privacy_risk(real_features, synth_features)


    # 4. Diversity

    report["diversity"] = compute_diversity(real_features, synth_features)


    # 5. Warnings

    warnings = []

    # Privacy warning
    if report["privacy"]["privacy_risk_level"] == "HIGH":
        warnings.append("High memorization risk detected")

    # Diversity warnings
    var_ratio = report["diversity"]["mean_variance_ratio"]
    if var_ratio < 0.5:
        warnings.append("Synthetic data shows signs of mode collapse")
    elif var_ratio > 1.5:
        warnings.append("Synthetic data shows excessive variance / instability")

    # Utility degradation
    if (
        report["utility"]["synthetic_train_accuracy"]
        < 0.8 * report["utility"]["real_train_accuracy"]
    ):
        warnings.append("Significant downstream utility degradation")

    # Calibration warning
    real_ece = report["utility"].get("real_train_ece", 0.0)
    synth_ece = report["utility"].get("synthetic_train_ece", 0.0)
    if synth_ece > real_ece + 0.1:
        warnings.append(
            "Synthetic data significantly degrades probability calibration"
        )


    # 6. Categorical distortion warnings (B2.4)

    cat_fidelity = report["fidelity"].get("categorical")

    if cat_fidelity:
        summary = cat_fidelity["summary"]
        per_feature = cat_fidelity["per_feature"]

        # Overall categorical shift
        if summary["mean_l1"] > 0.3:
            warnings.append(
                "Significant categorical distribution shift detected"
            )

        # High JS divergence
        if summary["mean_js"] > 0.2:
            warnings.append(
                "Categorical distributions differ substantially (high JS divergence)"
            )

        # Severe single-category distortion
        for col, stats in per_feature.items():
            if abs(stats["worst_shift"]) > 0.25:
                warnings.append(
                    f"Severe distortion in category '{stats['worst_category']}' "
                    f"of feature '{col}'"
                )

    report["warnings"] = warnings
    return report