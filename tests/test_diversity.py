import numpy as np
import pandas as pd
import pytest

from app.metrics.diversity import compute_diversity, knn_coverage
from tests.conftest import make_dataset

NUM, CAT = ["age", "income"], ["city"]


def test_coverage_known_answer_on_a_line():
    real = np.arange(10, dtype=float).reshape(-1, 1)
    # k=2: interior rows have two others at distance 1, so radius 1; the end rows have radius 2
    synth_full = real.copy()
    assert knn_coverage(real, synth_full, k=2)["value"] == pytest.approx(1.0)
    synth_left = np.array([[0.0], [1.0]])
    # rows 0, 1 and 2 have a synthetic row within their radius; row 3 is 2 away with radius 1
    assert knn_coverage(real, synth_left, k=2)["value"] == pytest.approx(0.3)


def test_fresh_sample_has_high_coverage_and_mode_collapse_low():
    real, fresh = make_dataset(1500, 1), make_dataset(1500, 2)
    good = compute_diversity(real, fresh, NUM, CAT)["coverage"]["value"]
    rng = np.random.default_rng(0)
    subset = real.iloc[:5]
    collapsed = subset.iloc[rng.integers(0, 5, 20)]  # 20 rows resampled from 5 rows
    bad = compute_diversity(real, collapsed, NUM, CAT)["coverage"]["value"]
    assert good > 0.85
    assert bad < 0.1


def test_coverage_query_subsampling_is_recorded_and_deterministic():
    rng = np.random.default_rng(0)
    real, synth = rng.normal(size=(600, 3)), rng.normal(size=(600, 3))
    a = knn_coverage(real, synth, max_queries=100, seed=1)
    assert a["subsampled"] is True and a["n_real_queried"] == 100
    assert a == knn_coverage(real, synth, max_queries=100, seed=1)
    full = knn_coverage(real, synth)
    assert full["subsampled"] is False
    assert abs(a["value"] - full["value"]) < 0.1


def test_pca_is_standardised_so_scale_does_not_matter():
    real, synth = make_dataset(500, 1), make_dataset(500, 2)
    a = compute_diversity(real, synth, NUM, CAT)
    real2, synth2 = real.copy(), synth.copy()
    real2["income"] *= 1e6
    synth2["income"] *= 1e6
    b = compute_diversity(real2, synth2, NUM, CAT)
    assert a["pca_coverage_ratio"] == pytest.approx(b["pca_coverage_ratio"], rel=1e-6)
    assert a["coverage"]["value"] == pytest.approx(b["coverage"]["value"])
    assert 0.8 < a["pca_coverage_ratio"] < 1.2


def test_pca_ratio_drops_for_collapsed_data():
    real = make_dataset(800, 1)
    collapsed = pd.concat([real.iloc[:1]] * 800)  # one row repeated
    out = compute_diversity(real, collapsed, NUM, CAT)
    assert out["pca_coverage_ratio"] == pytest.approx(0.0, abs=1e-12)


def test_variance_ratio_known_answer_and_constant_column():
    real = pd.DataFrame({"a": [0.0, 2.0, 4.0, 6.0], "k": [1.0] * 4})
    synth = pd.DataFrame({"a": [0.0, 1.0, 2.0, 3.0], "k": [1.0] * 4})
    out = compute_diversity(real, synth, ["a", "k"], [])
    assert out["variance_ratio"] == {"a": pytest.approx(0.25)}
    assert out["mean_variance_ratio"] == pytest.approx(0.25)


def test_categorical_only_and_tiny_inputs():
    real, synth = make_dataset(200, 1), make_dataset(200, 2)
    out = compute_diversity(real, synth, [], ["city"])
    assert out["variance_ratio"] is None
    assert out["coverage"]["value"] is not None
    tiny = compute_diversity(real.head(3), synth.head(3), NUM, CAT)
    assert tiny["coverage"]["value"] is None
