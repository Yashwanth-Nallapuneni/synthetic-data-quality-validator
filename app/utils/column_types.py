import pandas as pd
import numpy as np


def detect_column_types(
    df: pd.DataFrame,
    categorical_threshold: int = 20,
    range_ratio_threshold: float = 0.05,
):
    """
    Detect numeric vs categorical columns in a tabular dataset.

    A numeric column is considered categorical ONLY IF:
    - Low cardinality
    - Integer-like
    - Small value range relative to magnitude
    """

    numeric_cols = []
    categorical_cols = []

    for col in df.columns:
        series = df[col].dropna()

        # Explicit categorical
        if series.dtype == "object" or str(series.dtype).startswith("category"):
            categorical_cols.append(col)
            continue

        if pd.api.types.is_numeric_dtype(series):
            unique_vals = series.nunique()

            is_integer_like = np.allclose(series, series.astype(int))
            value_range = series.max() - series.min()
            scale = max(abs(series.mean()), 1.0)
            range_ratio = value_range / scale

            if (
                unique_vals <= categorical_threshold
                and is_integer_like
                and range_ratio < range_ratio_threshold
            ):
                categorical_cols.append(col)
            else:
                numeric_cols.append(col)

    return {
        "numeric": numeric_cols,
        "categorical": categorical_cols,
    }