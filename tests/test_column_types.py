import numpy as np
import pandas as pd

from app.utils.column_types import detect_column_types


def test_basic_types():
    df = pd.DataFrame(
        {
            "income": np.linspace(20000, 90000, 50),
            "city": ["a", "b"] * 25,
            "flag": [True, False] * 25,
            "year": [2020, 2021, 2022, 2023, 2024] * 10,
        }
    )
    t = detect_column_types(df)
    assert t["numeric"] == ["income"]
    assert set(t["categorical"]) == {"city", "flag", "year"}
    assert t["empty"] == []


def test_nan_and_empty_columns():
    df = pd.DataFrame(
        {
            "x": [1.5, np.nan, 3.2, 4.1, 5.9, 6.3],
            "ints_with_nan": [2020.0, np.nan, 2021.0, 2022.0, 2020.0, 2021.0],
            "all_nan": [np.nan] * 6,
        }
    )
    t = detect_column_types(df)
    assert t["numeric"] == ["x"]
    assert t["categorical"] == ["ints_with_nan"]
    assert t["empty"] == ["all_nan"]


def test_infinite_values_do_not_crash():
    df = pd.DataFrame({"x": [1.0, np.inf, 2.5, -np.inf, 3.7]})
    assert detect_column_types(df)["numeric"] == ["x"]


def test_empty_frame():
    t = detect_column_types(pd.DataFrame())
    assert t == {"numeric": [], "categorical": [], "empty": []}
