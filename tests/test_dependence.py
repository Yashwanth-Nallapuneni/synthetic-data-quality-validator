import numpy as np
import pandas as pd
import pytest

from app.metrics.dependence import (
    compute_dependence,
    correlation_ratio,
    cramers_v_bias_corrected,
    pearson,
)
from tests.conftest import make_dataset


def _correlated(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = 0.9 * x + rng.normal(0, 0.4, n)
    c = np.where(x > 0, "pos", "neg")
    flip = rng.random(n) < 0.1
    c = np.where(flip, np.where(c == "pos", "neg", "pos"), c)
    d = np.where(x + rng.normal(0, 0.5, n) > 0, "u", "v")
    return pd.DataFrame({"x": x, "y": y, "c": c, "d": d})


COLS = (["x", "y"], ["c", "d"])


def test_pearson_known_answers():
    x = np.arange(10.0)
    assert pearson(x, 2 * x + 1) == pytest.approx(1.0)
    assert pearson(x, -x) == pytest.approx(-1.0)
    assert np.isnan(pearson(x, np.ones(10)))


def test_cramers_v_known_answers():
    a = np.array(list("aabb") * 50)
    assert cramers_v_bias_corrected(a, a) == pytest.approx(1.0, abs=0.01)
    rng = np.random.default_rng(0)
    indep = cramers_v_bias_corrected(rng.choice(list("abc"), 5000), rng.choice(list("xyz"), 5000))
    assert indep < 0.05
    one_level = np.array(["a"] * 10)
    assert np.isnan(cramers_v_bias_corrected(one_level, np.array(list("ab") * 5)))


def test_cramers_v_bias_correction_removes_small_sample_inflation():
    rng = np.random.default_rng(1)
    vals = [
        cramers_v_bias_corrected(rng.choice(list("abcde"), 30), rng.choice(list("vwxyz"), 30))
        for _ in range(200)
    ]
    assert np.mean(vals) < 0.15  # uncorrected V is typically 0.3+ here


def test_correlation_ratio_known_answers():
    groups = np.array(["a"] * 50 + ["b"] * 50)
    values = np.where(groups == "a", 0.0, 10.0) + np.tile([-0.01, 0.01], 50)
    assert correlation_ratio(values, groups) > 0.99
    rng = np.random.default_rng(2)
    assert correlation_ratio(rng.normal(size=3000), rng.choice(list("abc"), 3000)) < 0.1
    assert np.isnan(correlation_ratio(np.ones(10), np.array(list("ab") * 5)))


def test_identical_data_has_zero_difference():
    df = _correlated()
    out = compute_dependence(df, df.copy(), *COLS)
    assert out["status"] == "ok"
    assert out["summary"]["mean_abs_diff"] == pytest.approx(0.0, abs=1e-12)


def test_fresh_sample_gives_small_difference():
    out = compute_dependence(_correlated(3000, 1), _correlated(3000, 2), *COLS)
    assert out["summary"]["mean_abs_diff"] < 0.05


def test_independently_shuffled_columns_give_large_difference():
    real = _correlated(3000, 1)
    rng = np.random.default_rng(5)
    shuffled = pd.DataFrame({c: rng.permutation(real[c].to_numpy()) for c in real.columns})
    out = compute_dependence(real, shuffled, *COLS)
    fresh = compute_dependence(real, _correlated(3000, 2), *COLS)
    assert out["summary"]["mean_abs_diff"] > 0.3
    assert out["summary"]["mean_abs_diff"] > 5 * fresh["summary"]["mean_abs_diff"]
    assert out["numeric_numeric"]["max_abs_diff"] > 0.8  # x-y correlation is about 0.9 in real
    assert set(out["summary"]["worst_pair"]) <= {"x", "y", "c", "d"}


def test_worst_pair_and_matrices_for_hand_built_case():
    n = 400
    rng = np.random.default_rng(3)
    a = rng.normal(size=n)
    real = pd.DataFrame({"a": a, "b": a, "z": rng.normal(size=n)})
    synth = pd.DataFrame({"a": a, "b": -a, "z": real["z"]})  # only the a-b pair flips sign
    out = compute_dependence(real, synth, ["a", "b", "z"], [])
    assert out["summary"]["worst_pair"] == ["a", "b"]
    assert out["summary"]["max_abs_diff"] == pytest.approx(2.0)
    m = out["numeric_numeric"]
    assert m["rows"] == ["a", "b", "z"]
    assert m["real"][0][1] == pytest.approx(1.0)
    assert m["synthetic"][0][1] == pytest.approx(-1.0)
    assert m["abs_diff"][0][1] == pytest.approx(2.0)


def test_worst_pairs_list_is_capped_and_sorted():
    rng = np.random.default_rng(4)
    cols = [f"c{i}" for i in range(12)]  # 66 pairs
    real = pd.DataFrame(rng.normal(size=(300, 12)), columns=cols)
    synth = pd.DataFrame(rng.normal(size=(300, 12)), columns=cols)
    out = compute_dependence(real, synth, cols, [])
    diffs = [p["abs_diff"] for p in out["worst_pairs"]]
    assert len(diffs) == 20
    assert diffs == sorted(diffs, reverse=True)
    assert out["summary"]["n_pairs"] == 66


def test_missing_values_and_small_inputs_do_not_crash():
    df = _correlated(200)
    df.loc[df.index[:50], "x"] = np.nan
    assert compute_dependence(df, df.copy(), *COLS)["status"] == "ok"
    assert compute_dependence(df.head(5), df.head(5), *COLS)["status"] == "insufficient_data"
    assert compute_dependence(df, df, ["x"], [])["status"] == "insufficient_data"


def test_mixed_dataset_with_conftest_generator():
    real, synth = make_dataset(500, 1), make_dataset(500, 2)
    out = compute_dependence(real, synth, ["age", "income", "label"], ["city"])
    assert out["status"] == "ok"
    assert "numeric_categorical" in out
