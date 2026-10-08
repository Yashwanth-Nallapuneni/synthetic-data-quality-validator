import numpy as np
import pandas as pd
import pytest

from app.metrics.privacy import compute_privacy_risk
from app.report import generate_trust_report
from tests.conftest import codes


def _real(n=500, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "income": rng.normal(60000, 15000, n),
            "age": rng.normal(40, 10, n),
            "label": rng.integers(0, 2, n),
        }
    )


COLS = ["income", "age"]


def test_noisy_near_copy_is_flagged_high():
    """Regression for bug 2: sigma=5 on income and 0.5 on age was reported LOW."""
    real = _real()
    rng = np.random.default_rng(1)
    synth = real.copy()
    synth["income"] += rng.normal(0, 5, len(synth))
    synth["age"] += rng.normal(0, 0.5, len(synth))
    p = compute_privacy_risk(real, synth, COLS, [])
    assert p["privacy_risk_level"] == "HIGH"
    assert p["share_closer_than_holdout_p5"] > 0.15
    report = generate_trust_report(real, synth, "label")
    assert "privacy_memorisation" in codes(report)


def test_exact_copy_is_flagged_high():
    real = _real()
    p = compute_privacy_risk(real, real.copy(), COLS, [])
    assert p["privacy_risk_level"] == "HIGH"
    assert p["exact_match_rate"] == pytest.approx(1.0)


@pytest.mark.parametrize("seed", range(5))
def test_fresh_sample_is_not_flagged(seed):
    real = _real(500, seed=100 + seed)
    synth = _real(500, seed=200 + seed)
    p = compute_privacy_risk(real, synth, COLS, [], seed=seed)
    assert p["privacy_risk_level"] != "HIGH"
    assert p["share_closer_than_holdout_p5"] < 0.15
    report = generate_trust_report(real, synth, "label", seed=seed)
    assert "privacy_memorisation" not in codes(report)


def test_scale_invariance_large_column_does_not_dominate():
    """Changing the units of income must not change the verdict."""
    real = _real()
    synth = _real(500, seed=5)
    a = compute_privacy_risk(real, synth, COLS, [])
    real2, synth2 = real.copy(), synth.copy()
    real2["income"] *= 1000
    synth2["income"] *= 1000
    b = compute_privacy_risk(real2, synth2, COLS, [])
    assert a["share_closer_than_holdout_p5"] == pytest.approx(b["share_closer_than_holdout_p5"])


def test_categorical_columns_contribute():
    rng = np.random.default_rng(0)
    real = pd.DataFrame(
        {"c": rng.choice(list("abcdefghij"), 400), "d": rng.choice(list("pqrstuvwxy"), 400)}
    )
    p = compute_privacy_risk(real, real.copy(), [], ["c", "d"])
    assert p["status"] == "ok"
    assert p["privacy_risk_level"] != "UNKNOWN"


def test_tiny_data_is_insufficient_not_an_error():
    p = compute_privacy_risk(_real(5), _real(5, seed=1), COLS, [])
    assert p["status"] == "insufficient_data"
    assert p["privacy_risk_level"] == "UNKNOWN"


def test_deterministic_with_seed():
    real, synth = _real(), _real(500, seed=9)
    assert compute_privacy_risk(real, synth, COLS, [], seed=3) == compute_privacy_risk(
        real, synth, COLS, [], seed=3
    )


def _near_copy(real, seed=1):
    rng = np.random.default_rng(seed)
    synth = real.copy()
    synth["income"] += rng.normal(0, 0.5, len(synth))
    synth["age"] += rng.normal(0, 0.05, len(synth))
    return synth


def test_holdout_mode_is_sharper_than_split_for_generator_trained_on_all_real():
    real = _real(600, seed=1)
    holdout = _real(600, seed=2)
    synth = _near_copy(real)  # the generator memorised ALL of the real rows
    split = compute_privacy_risk(real, synth, COLS, [])
    held = compute_privacy_risk(real, synth, COLS, [], holdout_df=holdout)
    assert split["mode"] == "split_real_50_50" and held["mode"] == "provided_holdout"
    assert held["n_train"] == 600 and held["n_holdout"] == 600
    assert held["share_closer_than_holdout_p5"] > 0.95
    assert split["share_closer_than_holdout_p5"] < held["share_closer_than_holdout_p5"] - 0.2
    assert held["privacy_risk_level"] == "HIGH"


@pytest.mark.parametrize("seed", range(3))
def test_holdout_mode_fresh_sample_is_low(seed):
    real, holdout, synth = _real(500, 10 + seed), _real(500, 20 + seed), _real(500, 30 + seed)
    p = compute_privacy_risk(real, synth, COLS, [], seed=seed, holdout_df=holdout)
    assert p["privacy_risk_level"] == "LOW"
    assert p["share_closer_than_holdout_p5"] < 0.1


def test_holdout_mode_exact_copy_reports_all_four_quantities():
    real, holdout = _real(), _real(500, seed=2)
    p = compute_privacy_risk(real, real.copy(), COLS, [], holdout_df=holdout)
    assert p["exact_match_rate"] == pytest.approx(1.0)
    assert p["synthetic_dcr"]["median"] == 0.0
    assert p["holdout_dcr"]["median"] > 0.0
    assert p["share_closer_than_holdout_p5"] == pytest.approx(1.0)


def test_holdout_too_small_is_insufficient():
    p = compute_privacy_risk(_real(), _real(500, 1), COLS, [], holdout_df=_real(5, 2))
    assert p["status"] == "insufficient_data"
    assert "holdout" in p["reason"]


def test_query_rows_are_capped_and_recorded():
    real, synth, holdout = _real(300), _real(300, 1), _real(300, 2)
    p = compute_privacy_risk(real, synth, COLS, [], holdout_df=holdout, max_query_rows=100)
    assert p["subsampled"] is True
    assert p["n_synthetic_queried"] == 100 and p["n_holdout_queried"] == 100
    assert p == compute_privacy_risk(real, synth, COLS, [], holdout_df=holdout, max_query_rows=100)
    assert compute_privacy_risk(real, synth, COLS, [])["subsampled"] is False
