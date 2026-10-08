import numpy as np
import pandas as pd
import pytest

from experiments.generators import (
    BROKEN,
    GENERATORS,
    GaussianCopula,
    oversample_target,
    split_columns,
)
from tests.conftest import make_dataset


def _frame(seed):
    df = make_dataset(400, seed, n_classes=3)
    df["city"] = pd.Categorical(df["city"])
    df.loc[::17, "city"] = np.nan  # missing categoricals
    df["label"] = df["label"].astype(str)
    return df.reset_index(drop=True)


@pytest.fixture
def train():
    return _frame(3)


@pytest.fixture
def fresh():
    """Real rows that are not in train (what the benchmark passes as H_gold)."""
    return _frame(30)


FRESH_BASED = ("column_shuffle", "mode_collapse", "category_oversample", "label_shuffle")


@pytest.mark.parametrize("name", list(GENERATORS))
def test_shape_columns_and_determinism(train, fresh, name):
    a = GENERATORS[name](train, 120, "label", 5, fresh=fresh)
    b = GENERATORS[name](train, 120, "label", 5, fresh=fresh)
    assert a.shape == (120, train.shape[1])
    assert list(a.columns) == list(train.columns)
    pd.testing.assert_frame_equal(a, b)


def test_exact_copy_is_distinct_subset_of_train(train):
    s = GENERATORS["exact_copy"](train, 150, "label", 1)
    assert len(s.drop_duplicates()) == 150
    merged = s.merge(train.drop_duplicates(), how="left", indicator=True)
    assert (merged["_merge"] == "both").all()


def _rows_in(s, train):
    return len(s.drop_duplicates().merge(train.drop_duplicates(), how="inner"))


@pytest.mark.parametrize("name", FRESH_BASED)
def test_isolated_injections_contain_no_training_rows(train, fresh, name):
    s = GENERATORS[name](train, 200, "label", 1, fresh=fresh)
    assert _rows_in(s, train) == 0


@pytest.mark.parametrize("name", FRESH_BASED)
def test_isolated_injections_require_fresh_rows(train, name):
    with pytest.raises(ValueError):
        GENERATORS[name](train, 50, "label", 1)


def test_copy_injections_do_use_training_rows(train, fresh):
    assert _rows_in(GENERATORS["exact_copy"](train, 150, "label", 1, fresh=fresh), train) == 150
    near = GENERATORS["near_copy"](train, 150, "label", 1, fresh=fresh)
    assert _rows_in(near, train) == 0  # perturbed, but each row is a noisy training row


def test_near_copy_noise_is_small_and_only_numeric(train):
    s = GENERATORS["near_copy"](train, 200, "label", 1)
    assert set(s["city"].dropna()) <= {"nyc", "sf", "la"}
    assert s["label"].isin(train["label"]).all()
    numeric, _ = split_columns(train, "label")
    assert set(numeric) == {"age", "income"}
    # each synthetic row is within ~5 noise sigma of some training row, per column
    for c in numeric:
        d = np.abs(s[c].to_numpy()[:, None] - train[c].to_numpy()[None, :]).min(axis=1)
        assert d.max() < 5 * 0.01 * train[c].std()


def test_column_shuffle_preserves_marginals_but_breaks_dependence(train):
    big = make_dataset(2000, 4)
    s = GENERATORS["column_shuffle"](train, 2000, "label", 2, fresh=big)
    # exact marginals of the bootstrap: same multiset per column as some bootstrap, so
    # means agree closely with train
    assert abs(s["income"].mean() - big["income"].mean()) < 0.1 * big["income"].std()
    assert (
        s["city"].value_counts(normalize=True) - big["city"].value_counts(normalize=True)
    ).abs().max() < 0.05
    real_corr = abs(np.corrcoef(big["income"], big["label"])[0, 1])
    synth_corr = abs(np.corrcoef(s["income"], s["label"])[0, 1])
    assert real_corr > 0.3 and synth_corr < 0.1


def test_mode_collapse_has_few_clusters(train, fresh):
    s = GENERATORS["mode_collapse"](train, 300, "label", 1, fresh=fresh)
    assert s["city"].nunique() <= 3  # at most 5 source rows
    assert s["income"].std() < train["income"].std()


def test_category_oversample_target_and_share(train, fresh):
    col, rarest = oversample_target(train, "label")
    assert col == "city"
    counts = train["city"].value_counts()
    assert rarest == counts.idxmin()
    s = GENERATORS["category_oversample"](train, 400, "label", 1, fresh=fresh)
    assert (s["city"] == rarest).mean() >= 0.5
    assert list(s["city"].cat.categories) == list(train["city"].cat.categories)


def test_label_shuffle_keeps_label_marginal_and_features(train, fresh):
    s = GENERATORS["label_shuffle"](train, 400, "label", 1, fresh=fresh)
    boot = fresh.sample(400, replace=True, random_state=1).reset_index(drop=True)
    pd.testing.assert_frame_equal(s.drop(columns="label"), boot.drop(columns="label"))
    assert sorted(s["label"]) == sorted(boot["label"])
    assert (s["label"] != boot["label"]).any()


def test_gaussian_copula_matches_marginals_and_dependence():
    real = make_dataset(3000, 8)
    s = GaussianCopula("label", 0).fit(real).sample(3000, 1)
    assert s["age"].mean() == pytest.approx(real["age"].mean(), abs=0.8)
    assert s["income"].std() == pytest.approx(real["income"].std(), rel=0.1)
    assert (s["city"] == "nyc").mean() == pytest.approx((real["city"] == "nyc").mean(), abs=0.04)
    assert set(s["label"].unique()) == {0, 1}
    r = np.corrcoef(real["income"], real["label"])[0, 1]
    c = np.corrcoef(s["income"], s["label"])[0, 1]
    # Interval jitter on the categorical label attenuates the dependence; it is kept in part.
    assert 0.4 * r < c < 1.05 * r


def test_gaussian_copula_integers_and_missing_rate():
    rng = np.random.default_rng(0)
    n = 1500
    df = pd.DataFrame(
        {
            "count": rng.poisson(30, n).astype(float),
            "x": np.where(rng.uniform(size=n) < 0.2, np.nan, rng.normal(size=n)),
            "c": rng.choice(["a", "b", None], n, p=[0.5, 0.3, 0.2]),
            "label": rng.choice(["u", "v"], n),
        }
    )
    s = GaussianCopula("label", 0).fit(df).sample(n, 2)
    assert (s["count"] == s["count"].round()).all()
    assert s["x"].isna().mean() == pytest.approx(0.2, abs=0.04)
    assert s["c"].isna().mean() == pytest.approx(0.2, abs=0.04)


def test_broken_list_excludes_copula():
    assert "gaussian_copula" not in BROKEN and "exact_copy" in BROKEN
