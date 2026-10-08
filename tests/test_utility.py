from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.metrics.utility import compute_utility, detect_task, expected_calibration_error
from app.report import generate_trust_report
from tests.conftest import make_dataset

DATA = Path(__file__).resolve().parent.parent / "data"


def test_ece_known_answer():
    conf = np.full(10, 0.9)
    correct = np.array([1] * 7 + [0] * 3)
    assert expected_calibration_error(conf, correct) == pytest.approx(0.2)


def test_ece_perfectly_calibrated_and_confidence_one():
    assert expected_calibration_error(np.ones(5), np.ones(5)) == pytest.approx(0.0)
    assert expected_calibration_error(np.ones(4), np.array([1, 1, 0, 0])) == pytest.approx(0.5)


@pytest.mark.parametrize("position", ["first", "middle", "last"])
def test_label_in_any_column(position):
    real = make_dataset(400, seed=1, label_position=position)
    synth = make_dataset(400, seed=2, label_position=position)
    u = compute_utility(real, synth, "label")
    assert u["status"] == "ok"
    assert u["task"] == "classification"
    assert u["real_train_accuracy"] > 0.8


def test_label_position_does_not_change_result():
    last = compute_utility(make_dataset(400, 1), make_dataset(400, 2), "label")
    first = compute_utility(
        make_dataset(400, 1, label_position="first"),
        make_dataset(400, 2, label_position="first"),
        "label",
    )
    assert last == first


def test_label_first_through_report_does_not_crash():
    real = make_dataset(300, 1, label_position="first")
    synth = make_dataset(300, 2, label_position="first")
    report = generate_trust_report(real, synth, "label")
    assert report["utility"]["status"] == "ok"


def test_multiclass_ece_is_top_label_and_bounded():
    real = make_dataset(600, 1, n_classes=3)
    synth = make_dataset(600, 2, n_classes=3)
    u = compute_utility(real, synth, "label")
    assert u["n_classes"] == 3
    assert 0.0 <= u["real_train_ece"] <= 1.0
    assert 0.0 <= u["synthetic_train_ece"] <= 1.0
    assert u["real_train_accuracy"] > 0.6


def test_string_labels():
    real = make_dataset(300, 1)
    synth = make_dataset(300, 2)
    for df in (real, synth):
        df["label"] = df["label"].map({0: "low", 1: "high"})
    u = compute_utility(real, synth, "label")
    assert u["status"] == "ok" and u["task"] == "classification"
    assert u["real_train_accuracy"] > 0.8


def test_regression_label_reports_r2_and_mae():
    def gen(seed):
        r = np.random.default_rng(seed)
        x = r.normal(size=400)
        c = r.choice(["a", "b"], 400)
        return pd.DataFrame({"x": x, "c": c, "y": 3 * x + (c == "a") * 2 + r.normal(0, 0.1, 400)})

    u = compute_utility(gen(1), gen(2), "y")
    assert u["task"] == "regression"
    assert u["real_train_r2"] > 0.95
    assert u["synthetic_train_r2"] > 0.95
    assert "real_train_accuracy" not in u
    assert u["real_train_mae"] < 0.5


def test_categorical_features_are_used():
    """The label depends only on a categorical column, so dropping it would fail."""
    def gen(seed):
        r = np.random.default_rng(seed)
        c = r.choice(["a", "b"], 400)
        return pd.DataFrame({"noise": r.normal(size=400), "c": c, "label": (c == "a").astype(int)})

    u = compute_utility(gen(1), gen(2), "label")
    assert u["real_train_accuracy"] > 0.95


def test_tiny_data_returns_insufficient_status():
    real = pd.read_csv(DATA / "real.csv")
    synth = pd.read_csv(DATA / "synth.csv")
    u = compute_utility(real, synth, "label")
    assert u["status"] == "insufficient_data"
    assert "at least" in u["reason"]
    report = generate_trust_report(real, synth, "label")
    assert report["utility"]["status"] == "insufficient_data"
    assert any("Utility not computed" in n for n in report["notices"])


def test_rare_class_disables_stratify_without_crashing():
    real = make_dataset(100, 1)
    real.loc[real.index[:1], "label"] = 7  # a class with a single member
    synth = make_dataset(100, 2)
    u = compute_utility(real, synth, "label")
    assert u["status"] == "ok"


def test_single_class_synthetic_labels_do_not_crash():
    real = make_dataset(300, 1)
    synth = make_dataset(300, 2)
    synth["label"] = 1
    u = compute_utility(real, synth, "label")
    assert u["status"] == "ok"
    assert u["synthetic_single_class"] is True
    report = generate_trust_report(real, synth, "label")
    assert "Synthetic labels contain a single class" in report["warnings"]


def test_single_class_real_labels_are_insufficient():
    real = make_dataset(100, 1)
    real["label"] = 0
    u = compute_utility(real, make_dataset(100, 2), "label")
    assert u["status"] == "insufficient_data"


def test_detect_task():
    assert detect_task(pd.Series([0, 1, 1, 0] * 5)) == "classification"
    assert detect_task(pd.Series(["a", "b"] * 10)) == "classification"
    assert detect_task(pd.Series(np.linspace(0, 1, 100))) == "regression"
    assert detect_task(pd.Series(np.arange(100))) == "regression"
