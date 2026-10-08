import pandas as pd

from app.errors import InputError
from app.metrics.categorical_fidelity import compute_categorical_fidelity
from app.metrics.diversity import compute_diversity
from app.metrics.fidelity import compute_fidelity
from app.metrics.privacy import compute_privacy_risk
from app.metrics.utility import compute_utility
from app.thresholds import Thresholds
from app.utils.column_types import detect_column_types
from app.utils.preprocessing import coerce_features, to_jsonable


def generate_trust_report(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    label_column: str,
    seed: int = 42,
    thresholds: Thresholds | None = None,
):
    """
    Generates a comprehensive trust report for synthetic data.

    Raises InputError for problems with the supplied data. Warning thresholds come from
    `thresholds` (provisional defaults when omitted, see app/thresholds.py).
    """
    th = thresholds or Thresholds()

    for name, df in (("real", real_df), ("synthetic", synth_df)):
        if label_column not in df.columns:
            raise InputError(f"Label column '{label_column}' not found in {name} data")

    real_features = real_df.drop(columns=[label_column])
    synth_features = synth_df.drop(columns=[label_column])

    only_real = sorted(set(real_features.columns) - set(synth_features.columns))
    only_synth = sorted(set(synth_features.columns) - set(real_features.columns))
    if only_real or only_synth:
        raise InputError(
            "Real and synthetic data must have the same columns. "
            f"Only in real: {only_real}. Only in synthetic: {only_synth}."
        )

    # One source of truth for column types, computed on the real features.
    col_types = detect_column_types(real_features)
    numeric_cols = col_types["numeric"]
    categorical_cols = col_types["categorical"]

    real_features = coerce_features(real_features, numeric_cols, categorical_cols)
    synth_features = coerce_features(synth_features, numeric_cols, categorical_cols)

    report = {"notices": []}
    if col_types["empty"]:
        report["notices"].append(
            f"Ignored columns with no values in the real data: {col_types['empty']}"
        )

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

    # 2. Utility

    report["utility"] = compute_utility(
        real_df,
        synth_df,
        label_column,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        seed=seed,
    )
    if report["utility"]["status"] != "ok":
        report["notices"].append(f"Utility not computed: {report['utility']['reason']}")

    # 3. Privacy

    report["privacy"] = compute_privacy_risk(
        real_features, synth_features, numeric_cols, categorical_cols, seed=seed, thresholds=th
    )

    # 4. Diversity

    report["diversity"] = compute_diversity(real_features, synth_features, numeric_cols)

    # 5. Warnings

    warnings = []

    # Privacy warning
    if report["privacy"]["status"] != "ok":
        report["notices"].append(f"Privacy not computed: {report['privacy']['reason']}")
    elif report["privacy"]["privacy_risk_level"] == "HIGH":
        warnings.append("High memorization risk detected")

    # Diversity warnings
    var_ratio = report["diversity"]["mean_variance_ratio"]
    if var_ratio is None:
        pass
    elif var_ratio < th.variance_ratio_low:
        warnings.append("Synthetic data shows signs of mode collapse")
    elif var_ratio > th.variance_ratio_high:
        warnings.append("Synthetic data shows excessive variance / instability")

    warnings.extend(_utility_warnings(report["utility"], th))

    # 6. Categorical distortion warnings (B2.4)

    cat_fidelity = report["fidelity"].get("categorical")

    if cat_fidelity:
        summary = cat_fidelity["summary"]
        per_feature = cat_fidelity["per_feature"]

        # Overall categorical shift
        if summary["mean_l1"] > th.categorical_mean_l1:
            warnings.append("Significant categorical distribution shift detected")

        # High JS divergence
        if summary["mean_js"] > th.categorical_mean_js:
            warnings.append("Categorical distributions differ substantially (high JS divergence)")

        # Severe single-category distortion
        for col, stats in per_feature.items():
            if abs(stats["worst_shift"]) > th.categorical_worst_shift:
                warnings.append(
                    f"Severe distortion in category '{stats['worst_category']}' of feature '{col}'"
                )

    report["warnings"] = warnings
    return to_jsonable(report)


def _utility_warnings(utility: dict, th: Thresholds) -> list[str]:
    """Warnings derived from the utility block (provisional thresholds)."""
    out: list[str] = []
    if utility["status"] != "ok":
        return out
    if utility["task"] == "regression":
        if utility["synthetic_train_r2"] < utility["real_train_r2"] - th.utility_r2_drop:
            out.append("Significant downstream utility degradation")
        return out
    if utility.get("synthetic_single_class"):
        out.append("Synthetic labels contain a single class")
    acc_floor = th.utility_accuracy_ratio * utility["real_train_accuracy"]
    if utility["synthetic_train_accuracy"] < acc_floor:
        out.append("Significant downstream utility degradation")
    if utility["synthetic_train_ece"] > utility["real_train_ece"] + th.utility_ece_increase:
        out.append("Synthetic data significantly degrades probability calibration")
    return out
