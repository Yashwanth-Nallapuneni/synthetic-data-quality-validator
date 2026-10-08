import numpy as np
import pandas as pd
import pytest


def make_dataset(n=400, seed=0, n_classes=2, label="label", label_position="last"):
    """Mixed-type dataset where the label depends on income, age and city."""
    rng = np.random.default_rng(seed)
    age = rng.normal(40, 10, n)
    income = rng.normal(60000, 15000, n)
    city = rng.choice(["nyc", "sf", "la"], n)
    score = (income - 60000) / 15000 + (age - 40) / 10 + (city == "nyc") * 1.0
    score = score + rng.normal(0, 0.3, n)
    if n_classes == 2:
        y = (score > 0.5).astype(int)
    else:
        y = np.digitize(score, np.quantile(score, np.linspace(0, 1, n_classes + 1)[1:-1]))
    df = pd.DataFrame({"age": age, "income": income, "city": city})
    if label_position == "first":
        df.insert(0, label, y)
    elif label_position == "middle":
        df.insert(1, label, y)
    else:
        df[label] = y
    return df


@pytest.fixture
def make():
    return make_dataset
