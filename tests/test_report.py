import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.metrics.diversity import compute_diversity
from app.report import generate_trust_report
from tests.conftest import make_dataset

DATA = Path(__file__).resolve().parent.parent / "data"


def test_report_is_json_serialisable_and_strict():
    report = generate_trust_report(make_dataset(300, 1), make_dataset(300, 2), "label")
    text = json.dumps(report, allow_nan=False)
    assert json.loads(text)["utility"]["status"] == "ok"


def test_report_is_deterministic():
    real, synth = make_dataset(300, 1), make_dataset(300, 2)
    assert generate_trust_report(real, synth, "label") == generate_trust_report(
        real, synth, "label"
    )


def test_bundled_sample_does_not_crash():
    real = pd.read_csv(DATA / "real.csv")
    synth = pd.read_csv(DATA / "synth.csv")
    report = generate_trust_report(real, synth, "label")
    assert report["utility"]["status"] == "insufficient_data"
    assert report["privacy"]["status"] == "insufficient_data"
    json.dumps(report, allow_nan=False)


def test_integer_coded_categorical_is_consistent_across_modules():
    """A year-like integer column is categorical everywhere, not numeric in some modules."""

    def gen(seed):
        df = make_dataset(300, seed)
        df["year"] = np.random.default_rng(seed).choice([2020, 2021, 2022, 2023], 300)
        return df

    report = generate_trust_report(gen(1), gen(2), "label")
    assert "year" in report["fidelity"]["categorical"]["per_feature"]
    assert "year" not in report["fidelity"]["numeric"]["per_feature"]
    assert "year" not in report["diversity"]["variance_ratio"]
    assert report["fidelity"]["summary"]["num_categorical_features"] == 2


def test_missing_values_do_not_crash():
    real, synth = make_dataset(300, 1), make_dataset(300, 2)
    real.loc[real.index[:20], "age"] = np.nan
    synth.loc[synth.index[:20], "city"] = np.nan
    real["empty"] = np.nan
    synth["empty"] = np.nan
    report = generate_trust_report(real, synth, "label")
    json.dumps(report, allow_nan=False)
    assert any("empty" in n for n in report["notices"])


def test_column_mismatch_raises_with_names():
    real, synth = make_dataset(100, 1), make_dataset(100, 2)
    synth = synth.rename(columns={"age": "years"})
    with pytest.raises(ValueError, match="age.*years"):
        generate_trust_report(real, synth, "label")


def test_missing_label_raises():
    real = make_dataset(100, 1)
    with pytest.raises(ValueError, match="nope"):
        generate_trust_report(real, real, "nope")


def test_diversity_uses_given_numeric_columns_only():
    real, synth = make_dataset(200, 1), make_dataset(200, 2)
    d = compute_diversity(real, synth, ["age"], [])
    assert list(d["variance_ratio"]) == ["age"]
