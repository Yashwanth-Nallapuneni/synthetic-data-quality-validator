import numpy as np
import pandas as pd

from app.metrics.detectability import compute_detectability
from tests.conftest import make_dataset

NUM, CAT = ["age", "income", "label"], ["city"]


def test_fresh_sample_auc_near_chance():
    out = compute_detectability(make_dataset(1500, 1), make_dataset(1500, 2), NUM, CAT)
    assert out["status"] == "ok"
    assert 0.44 < out["auc_mean"] < 0.58
    assert out["n_per_side"] == 1500
    assert len(out["auc_per_fold"]) == out["n_splits"] == 5


def test_shifted_sample_is_clearly_detectable():
    real = make_dataset(1500, 1)
    synth = make_dataset(1500, 2)
    synth["income"] += 15000  # one standard deviation
    out = compute_detectability(real, synth, NUM, CAT)
    assert out["auc_mean"] > 0.7


def test_independently_shuffled_columns_are_detectable():
    real = make_dataset(2000, 1)
    rng = np.random.default_rng(7)
    shuffled = pd.DataFrame({c: rng.permutation(real[c].to_numpy()) for c in real.columns})
    out = compute_detectability(real, shuffled, NUM, CAT)
    fresh = compute_detectability(real, make_dataset(2000, 2), NUM, CAT)
    assert out["auc_mean"] > 0.62
    assert out["auc_mean"] > fresh["auc_mean"] + 0.1


def test_tiny_data_is_insufficient():
    out = compute_detectability(make_dataset(10, 1), make_dataset(10, 2), NUM, CAT)
    assert out["status"] == "insufficient_data"
    assert compute_detectability(make_dataset(100, 1), make_dataset(100, 2), [], [])[
        "status"
    ] == "insufficient_data"


def test_rows_are_capped_and_balanced():
    out = compute_detectability(
        make_dataset(600, 1), make_dataset(900, 2), NUM, CAT, max_rows_per_side=200
    )
    assert out["n_per_side"] == 200
    assert out["subsampled"] is True
    out = compute_detectability(make_dataset(300, 1), make_dataset(500, 2), NUM, CAT)
    assert out["n_per_side"] == 300  # larger side subsampled down to the smaller


def test_deterministic_with_seed():
    a, b = make_dataset(300, 1), make_dataset(300, 2)
    assert compute_detectability(a, b, NUM, CAT, seed=3) == compute_detectability(
        a, b, NUM, CAT, seed=3
    )
