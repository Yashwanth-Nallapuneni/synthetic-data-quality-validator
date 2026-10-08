"""Downstream utility: TSTR (train on synthetic) vs TRTR (train on real), tested on real."""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from app.utils.column_types import detect_column_types
from app.utils.preprocessing import (
    as_categorical,
    as_numeric,
    build_preprocessor,
    coerce_features,
    limit_threads_for_small_data,
)

MIN_ROWS = 10
TEST_SIZE = 0.3
REGRESSION_MIN_UNIQUE = 20


def expected_calibration_error(confidences, correct, n_bins: int = 10) -> float:
    """Top-label Expected Calibration Error.

    confidences: the predicted probability of the predicted class (max over classes).
    correct: 1 if the predicted class equals the true class, else 0.
    ECE = sum over bins of |accuracy(bin) - mean confidence(bin)| * share of samples in bin.
    Works for binary and multiclass models.
    """
    confidences = np.asarray(confidences, dtype=float)
    correct = np.asarray(correct, dtype=float)
    if confidences.size == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.clip(np.digitize(confidences, edges) - 1, 0, n_bins - 1)
    ece = 0.0
    for i in range(n_bins):
        mask = bin_ids == i
        if mask.any():
            ece += abs(correct[mask].mean() - confidences[mask].mean()) * mask.mean()
    return float(ece)


def detect_task(label: pd.Series) -> str:
    """Return "regression" or "classification" for a label series.

    Uses the shared column-type detector. A numeric label is regression only if it has
    many unique values or is not integer-coded; small integer-coded labels such as 0/1
    are class labels.
    """
    clean = label.dropna()
    types = detect_column_types(clean.to_frame("y"))
    if types["numeric"]:
        values = as_numeric(clean)
        integer_like = bool(np.allclose(values, np.round(values)))
        if values.nunique() > REGRESSION_MIN_UNIQUE or not integer_like:
            return "regression"
    return "classification"


def _insufficient(reason: str, task: str | None = None) -> dict:
    return {"status": "insufficient_data", "reason": reason, "task": task}


def _can_stratify(y: pd.Series, test_size: float) -> bool:
    counts = y.value_counts()
    n_classes = len(counts)
    n_test = int(np.ceil(test_size * len(y)))
    n_train = len(y) - n_test
    return bool(counts.min() >= 2 and n_test >= n_classes and n_train >= n_classes)


def _classification_metrics(model, X_test, y_test) -> dict:
    proba = model.predict_proba(X_test)
    classes = model.classes_
    pred = classes[np.argmax(proba, axis=1)]
    conf = proba.max(axis=1)
    correct = pred == np.asarray(y_test)
    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "f1": float(f1_score(y_test, pred, average="weighted", zero_division=0)),
        "ece": expected_calibration_error(conf, correct),
    }


def _constant_classification_metrics(label, y_test) -> dict:
    """Metrics for a degenerate model that always predicts one class with probability 1."""
    pred = np.full(len(y_test), label, dtype=object)
    correct = pred == np.asarray(y_test)
    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "f1": float(f1_score(y_test, pred, average="weighted", zero_division=0)),
        "ece": expected_calibration_error(np.ones(len(y_test)), correct),
    }


def _make_models(task: str, seed: int) -> dict:
    """Unfitted estimators by report name. Each model gets its own fresh preprocessing."""
    if task == "regression":
        return {
            "ridge_regression": Ridge(alpha=1.0),
            "hist_gradient_boosting": HistGradientBoostingRegressor(
                max_iter=100, random_state=seed
            ),
        }
    return {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=seed),
        "hist_gradient_boosting": HistGradientBoostingClassifier(max_iter=100, random_state=seed),
    }


def _regression_metrics(model, X_test, y_test) -> dict:
    pred = model.predict(X_test)
    return {
        "r2": float(r2_score(y_test, pred)),
        "mae": float(mean_absolute_error(y_test, pred)),
    }


def _comparison(task: str, trtr: dict, tstr: dict) -> dict:
    """Summary of how much TSTR falls short of TRTR."""
    if task == "regression":
        return {"r2_drop": trtr["r2"] - tstr["r2"]}
    ratio = tstr["accuracy"] / trtr["accuracy"] if trtr["accuracy"] > 0 else None
    return {"accuracy_ratio": ratio, "ece_increase": tstr["ece"] - trtr["ece"]}


def compute_utility(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    label_column: str,
    numeric_cols: list[str] | None = None,
    categorical_cols: list[str] | None = None,
    seed: int = 42,
) -> dict:
    """TRTR vs TSTR downstream utility, per model.

    The real rows are split once (70/30, seeded). For each model, TRTR trains on the real
    train split and TSTR trains on all synthetic rows; both are scored on the same held-out
    real test split. Features are the given numeric and categorical columns (default:
    detected from the real features). Numeric columns are standardised and categorical
    columns one-hot encoded inside an sklearn Pipeline. The label may be any column.

    Models: logistic regression and histogram gradient boosting for classification (accuracy,
    weighted F1, top-label ECE; binary and multiclass), ridge regression and histogram
    gradient boosting for regression (R2, MAE). The result has `models[name]["trtr"]`,
    `["tstr"]` and `["tstr_vs_trtr"]`. When the data are too small the result has status
    "insufficient_data" and a reason instead of metrics.
    """
    if label_column not in real_df.columns or label_column not in synth_df.columns:
        raise ValueError(f"Label column '{label_column}' must exist in both datasets")

    if numeric_cols is None or categorical_cols is None:
        types = detect_column_types(real_df.drop(columns=[label_column]))
        numeric_cols, categorical_cols = types["numeric"], types["categorical"]
    feature_cols = list(numeric_cols) + list(categorical_cols)
    if not feature_cols:
        return _insufficient("No usable feature columns")

    real_label = real_df[label_column]
    real_label = real_label[real_label.notna()]
    if len(real_label) < MIN_ROWS:
        return _insufficient(f"Need at least {MIN_ROWS} labelled real rows, got {len(real_label)}")

    task = detect_task(real_label)

    def prepare(df: pd.DataFrame):
        df = df[df[label_column].notna()]
        X = coerce_features(df, list(numeric_cols), list(categorical_cols))
        y = (
            as_numeric(df[label_column])
            if task == "regression"
            else as_categorical(df[label_column])
        )
        return X, y

    X_real, y_real = prepare(real_df)
    X_synth, y_synth = prepare(synth_df)

    if task == "classification" and y_real.nunique() < 2:
        return _insufficient("Real label has a single class", task)
    if len(y_synth) == 0:
        return _insufficient("Synthetic data has no labelled rows", task)

    stratify = y_real if task == "classification" and _can_stratify(y_real, TEST_SIZE) else None
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        X_real, y_real, test_size=TEST_SIZE, random_state=seed, stratify=stratify
    )
    if task == "classification" and yr_train.nunique() < 2:
        return _insufficient("Real training split has a single class", task)

    def fit(estimator, X, y) -> Pipeline:
        pre = build_preprocessor(list(numeric_cols), list(categorical_cols))
        return Pipeline([("pre", pre), ("model", estimator)]).fit(X, y)

    def score(model, X_test, y_test) -> dict:
        if task == "regression":
            return _regression_metrics(model, X_test, y_test)
        return _classification_metrics(model, X_test, y_test)

    result: dict = {
        "status": "ok",
        "reason": None,
        "task": task,
        "n_real_train": int(len(Xr_train)),
        "n_real_test": int(len(Xr_test)),
        "n_synthetic": int(len(X_synth)),
        "models": {},
    }
    synthetic_single_class = task == "classification" and y_synth.nunique() < 2
    if task == "classification":
        result["n_classes"] = int(y_real.nunique())
        result["synthetic_single_class"] = bool(synthetic_single_class)

    with limit_threads_for_small_data(len(X_real) + len(X_synth)):
        for name, estimator in _make_models(task, seed).items():
            trtr = score(fit(estimator, Xr_train, yr_train), Xr_test, yr_test)
            if synthetic_single_class:
                # A model trained on one class can only predict that class.
                tstr = _constant_classification_metrics(y_synth.iloc[0], yr_test)
            else:
                tstr = score(fit(clone(estimator), X_synth, y_synth), Xr_test, yr_test)
            result["models"][name] = {
                "trtr": trtr,
                "tstr": tstr,
                "tstr_vs_trtr": _comparison(task, trtr, tstr),
            }
    return result
