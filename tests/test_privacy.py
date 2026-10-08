import numpy as np
import pandas as pd
import pytest

from app.metrics.privacy import compute_privacy_risk
from app.report import generate_trust_report


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
    assert "High memorization risk detected" in report["warnings"]


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
    assert "High memorization risk detected" not in report["warnings"]


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
