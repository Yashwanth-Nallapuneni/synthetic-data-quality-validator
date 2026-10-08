"""Shared helpers to coerce columns to the types chosen by detect_column_types."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _normalise_value(v):
    """Stringify a categorical value so 1, 1.0 and "1" all map to "1"."""
    if isinstance(v, (float, np.floating)) and float(v).is_integer():
        return str(int(v))
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    return str(v)


def as_categorical(series: pd.Series) -> pd.Series:
    """Cast a series to strings, keeping missing values as NaN."""
    out = series.astype("object")
    mask = out.notna()
    out = out.where(mask, np.nan)
    out[mask] = out[mask].map(_normalise_value)
    return out


def as_numeric(series: pd.Series) -> pd.Series:
    """Cast a series to float, turning unparseable values and infinities into NaN."""
    out = pd.to_numeric(series, errors="coerce").astype(float)
    return out.replace([np.inf, -np.inf], np.nan)


def coerce_features(
    df: pd.DataFrame, numeric_cols: list[str], categorical_cols: list[str]
) -> pd.DataFrame:
    """Return a copy of df restricted to the typed columns, in a consistent dtype."""
    out = pd.DataFrame(index=df.index)
    for col in numeric_cols:
        out[col] = as_numeric(df[col])
    for col in categorical_cols:
        out[col] = as_categorical(df[col])
    return out


def build_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    """Impute, standardise numeric columns and one-hot encode categorical columns."""
    transformers = []
    if numeric_cols:
        transformers.append(
            (
                "num",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric_cols,
            )
        )
    if categorical_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        (
                            "impute",
                            SimpleImputer(
                                strategy="most_frequent",
                                keep_empty_features=True,
                            ),
                        ),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_cols,
            )
        )
    return ColumnTransformer(transformers, sparse_threshold=0.0)
