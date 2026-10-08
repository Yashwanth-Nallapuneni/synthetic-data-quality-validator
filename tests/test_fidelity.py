import warnings

import numpy as np
import pandas as pd
import pytest

from app.metrics.fidelity import compute_fidelity, quantile_js_distance, standardised_wasserstein


def test_js_identical_samples_is_zero():
    x = np.random.default_rng(0).normal(size=1000)
    assert quantile_js_distance(x, x) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("n", [2000, 20000, 50000])
def test_js_fresh_sample_is_near_zero_regardless_of_size(n):
    rng = np.random.default_rng(1)
    real = rng.normal(size=n)
    fresh = rng.normal(size=n)
    assert quantile_js_distance(real, fresh) < 0.08


def test_js_shifted_sample_scores_clearly_higher():
    rng = np.random.default_rng(2)
    real = rng.normal(size=5000)
    fresh = rng.normal(size=5000)
    shifted = rng.normal(loc=1.0, size=5000)
    assert quantile_js_distance(real, shifted) > 0.3
    assert quantile_js_distance(real, shifted) > 5 * quantile_js_distance(real, fresh)


def test_js_all_synthetic_mass_in_one_bin_known_answer_and_scale_free():
    from scipy.spatial.distance import jensenshannon

    real = np.linspace(0, 1, 1000)
    synth = real - 100  # everything lands in the open-ended first bin
    p = np.full(10, 0.1)
    q = np.zeros(10)
    q[0] = 1.0
    expected = jensenshannon(p, q, base=2)
    assert quantile_js_distance(real, synth) == pytest.approx(expected, abs=1e-9)
    assert quantile_js_distance(real * 1e6, synth * 1e6) == pytest.approx(expected, abs=1e-9)


def test_js_handles_constant_and_discrete_columns():
    assert quantile_js_distance(np.ones(100), np.ones(50)) == pytest.approx(0.0)
    assert quantile_js_distance(np.ones(100), np.full(50, 2.0)) == pytest.approx(1.0)
    r = np.repeat([0.0, 1.0, 2.0], 100)
    assert quantile_js_distance(r, r.copy()) == pytest.approx(0.0)


def test_standardised_wasserstein_known_answer_and_scale_invariance():
    real = np.array([0.0, 2.0, 4.0, 6.0])  # std = sqrt(5)
    synth = real + 1.0  # raw W1 = 1
    assert standardised_wasserstein(real, synth) == pytest.approx(1 / np.sqrt(5))
    assert standardised_wasserstein(real * 1000, synth * 1000) == pytest.approx(1 / np.sqrt(5))
    assert standardised_wasserstein(np.ones(5), np.zeros(5)) is None


def test_compute_fidelity_columns_comparable_across_scales():
    rng = np.random.default_rng(3)
    real = pd.DataFrame({"small": rng.normal(0, 1, 2000), "big": rng.normal(0, 1000, 2000)})
    synth = real.copy()
    synth["small"] += 0.5
    synth["big"] += 500
    out = compute_fidelity(real, synth)["per_feature"]
    assert out["small"]["wasserstein_standardised"] == pytest.approx(
        out["big"]["wasserstein_standardised"], rel=0.05
    )
    assert out["big"]["wasserstein_distance"] > 100 * out["small"]["wasserstein_distance"]


def test_large_samples_emit_no_scipy_runtime_warning():
    rng = np.random.default_rng(4)
    real = pd.DataFrame({"x": rng.normal(size=30000)})
    synth = pd.DataFrame({"x": rng.normal(size=30000)})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = compute_fidelity(real, synth)
    assert out["per_feature"]["x"]["ks_statistic"] < 0.03
