"""Single source of truth for numeric vs categorical column typing."""

import numpy as np
import pandas as pd


def _is_categorical_numeric(
    series: pd.Series, categorical_threshold: int, range_ratio_threshold: float
) -> bool:
    """Heuristic for integer-coded categoricals among numeric columns.

    A numeric column is treated as categorical only if it has low cardinality, holds
    integer-like values and spans a small range relative to its magnitude.
    """
    unique_vals = series.nunique()
    if unique_vals > categorical_threshold:
        return False
    is_integer_like = bool(np.allclose(series.to_numpy(dtype=float), np.round(series)))
    if not is_integer_like:
        return False
    value_range = float(series.max() - series.min())
    scale = max(abs(float(series.mean())), 1.0)
    return (value_range / scale) < range_ratio_threshold


def detect_column_types(
    df: pd.DataFrame,
    categorical_threshold: int = 20,
    range_ratio_threshold: float = 0.05,
) -> dict[str, list[str]]:
    """Detect numeric vs categorical columns in a tabular dataset.

    Returns a dict with three lists of column names:
    - "numeric": continuous or high-cardinality numeric columns
    - "categorical": strings, categories, booleans and integer-coded categoricals
    - "empty": columns with no non-missing values (ignored by every metric)

    Missing values and infinities are ignored when deciding the type.
    """
    numeric_cols: list[str] = []
    categorical_cols: list[str] = []
    empty_cols: list[str] = []

    for col in df.columns:
        series = df[col]
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            series = series.replace([np.inf, -np.inf], np.nan)
        series = series.dropna()

        if series.empty:
            empty_cols.append(col)
        elif pd.api.types.is_bool_dtype(series) or not pd.api.types.is_numeric_dtype(series):
            categorical_cols.append(col)
        elif _is_categorical_numeric(series, categorical_threshold, range_ratio_threshold):
            categorical_cols.append(col)
        else:
            numeric_cols.append(col)

    return {"numeric": numeric_cols, "categorical": categorical_cols, "empty": empty_cols}
