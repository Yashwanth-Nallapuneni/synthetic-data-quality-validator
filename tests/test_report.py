import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.errors import InputError
from app.metrics.diversity import compute_diversity
from app.report import generate_trust_report
from app.thresholds import Thresholds
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


def _shuffle_columns(df, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({c: rng.permutation(df[c].to_numpy()) for c in df.columns})


def test_report_has_new_sections_and_thresholds():
    report = generate_trust_report(make_dataset(500, 1), make_dataset(500, 2), "label")
    assert report["dependence"]["status"] == "ok"
    assert report["detectability"]["status"] == "ok"
    assert report["diversity"]["coverage"]["value"] is not None
    assert set(report["utility"]["models"]) == {"logistic_regression", "hist_gradient_boosting"}
    assert report["thresholds"]["c2st_auc"] == 0.7
    json.dumps(report, allow_nan=False)


def test_independently_shuffled_columns_trigger_dependence_and_c2st_warnings():
    real = make_dataset(1500, 1)
    report = generate_trust_report(real, _shuffle_columns(real), "label")
    assert "Pairwise dependencies differ from the real data" in report["warnings"]
    assert report["dependence"]["summary"]["mean_abs_diff"] > 0.1


def test_clean_data_has_neither_new_warning():
    report = generate_trust_report(make_dataset(1500, 1), make_dataset(1500, 2), "label")
    assert "Pairwise dependencies differ from the real data" not in report["warnings"]
    assert "Synthetic rows are easily distinguishable from real rows" not in report["warnings"]


def test_c2st_warning_fires_for_shifted_data_and_respects_custom_threshold():
    real, synth = make_dataset(1000, 1), make_dataset(1000, 2)
    synth["income"] += 30000
    msg = "Synthetic rows are easily distinguishable from real rows"
    assert msg in generate_trust_report(real, synth, "label")["warnings"]
    lax = Thresholds(c2st_auc=0.999)
    assert msg not in generate_trust_report(real, synth, "label", thresholds=lax)["warnings"]


def test_label_in_dependence_so_shuffled_labels_are_seen():
    real = make_dataset(1000, 1)
    synth = make_dataset(1000, 2)
    synth["label"] = np.random.default_rng(0).permutation(synth["label"].to_numpy())
    report = generate_trust_report(real, synth, "label")
    pairs = report["dependence"]["worst_pairs"]
    assert any("label" in (p["a"], p["b"]) for p in pairs[:3])


def test_split_mode_notice_and_holdout_mode_notice():
    real, synth = make_dataset(400, 1), make_dataset(400, 2)
    split = generate_trust_report(real, synth, "label")
    assert split["privacy"]["mode"] == "split_real_50_50"
    assert any("diluted" in n for n in split["notices"])
    held = generate_trust_report(real, synth, "label", holdout_df=make_dataset(400, 3))
    assert held["privacy"]["mode"] == "provided_holdout"
    assert held["privacy"]["n_train"] == 400
    assert not any("diluted" in n for n in held["notices"])
    assert any("provided holdout" in n for n in held["notices"])


def test_holdout_mode_catches_copy_of_all_real_rows_more_sharply():
    real = make_dataset(500, 1)
    synth = real.copy()
    synth["income"] += np.random.default_rng(0).normal(0, 1, len(synth))
    report = generate_trust_report(real, synth, "label", holdout_df=make_dataset(500, 9))
    assert "High memorization risk detected" in report["warnings"]
    assert report["privacy"]["share_closer_than_holdout_p5"] > 0.9


def test_holdout_validation():
    real, synth = make_dataset(100, 1), make_dataset(100, 2)
    with pytest.raises(InputError, match="holdout"):
        generate_trust_report(real, synth, "label", holdout_df=real.drop(columns=["age"]))
    with pytest.raises(InputError, match="holdout"):
        generate_trust_report(real, synth, "label", holdout_df=real.drop(columns=["label"]))


def test_tiny_holdout_is_a_notice_not_an_error():
    real, synth = make_dataset(200, 1), make_dataset(200, 2)
    report = generate_trust_report(real, synth, "label", holdout_df=make_dataset(5, 3))
    assert report["privacy"]["status"] == "insufficient_data"
    assert any("holdout" in n for n in report["notices"])


def test_detectability_subsampling_is_noted(monkeypatch):
    import app.metrics.detectability as det

    monkeypatch.setattr(det, "MAX_ROWS_PER_SIDE", 100)
    report = generate_trust_report(make_dataset(300, 1), make_dataset(300, 2), "label")
    assert report["detectability"]["n_per_side"] == 100
    assert any("subsample of 100 rows per side" in n for n in report["notices"])
