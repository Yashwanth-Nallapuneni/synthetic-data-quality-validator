"""Smoke test of the benchmark on a small synthetic frame (not the real datasets)."""

import pytest

from app.checks import extract_stats
from app.report import generate_trust_report
from app.thresholds import Thresholds
from experiments.benchmark import Dataset, run_unit, split_three
from experiments.calibrate import fired_codes
from experiments.generators import GENERATORS, HELDOUT
from tests.conftest import make_dataset


@pytest.fixture(scope="module")
def toy_runs():
    ds = Dataset("toy", 0, "label")
    df = make_dataset(400, 5)
    return df, ds, run_unit(df, ds, 0)


def test_splits_are_disjoint_and_sized():
    df = make_dataset(400, 5)
    train, gold, base = split_three(df, "label", 0, None)
    assert (len(train), len(gold), len(base)) == (200, 100, 100)
    key = lambda d: set(map(tuple, d.round(6).to_numpy()))  # noqa: E731
    assert not key(train) & key(gold) and not key(train) & key(base) and not key(gold) & key(base)


def test_max_rows_subsamples_before_splitting():
    train, gold, base = split_three(make_dataset(400, 5), "label", 0, 200)
    assert len(train) + len(gold) + len(base) == 200


def test_run_unit_covers_every_generator_without_errors(toy_runs):
    _, _, runs = toy_runs
    assert [r["generator"] for r in runs] == [HELDOUT, *GENERATORS]
    assert all(r["error"] is None and r["stats"] for r in runs)
    assert all(r["n_synth"] == 100 and r["n_train"] == 200 for r in runs)


def test_exact_copy_is_caught_by_privacy_in_the_benchmark(toy_runs):
    _, _, runs = toy_runs
    by = {r["generator"]: r for r in runs}
    assert "privacy_memorisation" in fired_codes(by["exact_copy"], Thresholds())


def test_reapplying_thresholds_to_recorded_stats_matches_the_report(toy_runs):
    df, ds, runs = toy_runs
    train, gold, base = split_three(df, ds.label, 0, None)
    custom = Thresholds(c2st_auc=0.55, variance_ratio_low=0.9, privacy_high_share=0.3)
    for name in (HELDOUT, "column_shuffle", "exact_copy"):
        synth = gold if name == HELDOUT else GENERATORS[name](train, len(gold), ds.label, 0)
        for th in (Thresholds(), custom):
            report = generate_trust_report(
                train, synth, ds.label, seed=0, thresholds=th, holdout_df=base
            )
            recorded = next(r for r in runs if r["generator"] == name)
            assert recorded["stats"] == extract_stats(report)
            assert fired_codes(recorded, th) == {w["code"] for w in report["warnings"]}
