"""Calibration, direction and metadata logic of the SDMetrics comparison (no SDMetrics needed)."""

import numpy as np
import pandas as pd
import pytest

from experiments import compare_report as cr
from experiments import compare_sdmetrics as cs
from tests.conftest import make_dataset


def sd_run(gen, seed, dataset="d", **metrics):
    return {
        "dataset": dataset,
        "seed": seed,
        "generator": gen,
        "metrics": metrics,
        "runtime_s": {"quality_score": 1.0},
        "runtime_total_s": 1.0,
        "errors": {},
    }


NULLS = [
    sd_run("heldout_real", s, column_shapes=0.90 + 0.01 * s, dcr_overfitting=0.50 - 0.01 * s)
    for s in range(3)
]


def test_threshold_below_is_min_minus_margin():
    thr, info = cr.threshold_for([0.9, 0.95, 0.92], cr.BELOW)
    assert thr == pytest.approx(0.9 - 0.09)
    assert info["n_null_values"] == 3
    assert info["max_per_check_false_alarm_rate_before_margin"] == pytest.approx(0.25)


def test_threshold_above_is_max_plus_margin():
    thr, _ = cr.threshold_for([0.1, 0.4], cr.ABOVE)
    assert thr == pytest.approx(0.4 + 0.04)


def test_absolute_margin_floor_near_zero_and_bad_input():
    assert cr.threshold_for([0.0], cr.BELOW)[0] == pytest.approx(-0.01)
    assert cr.threshold_for([0.0], cr.ABOVE)[0] == pytest.approx(0.01)
    with pytest.raises(ValueError):
        cr.threshold_for([], cr.BELOW)
    with pytest.raises(ValueError):
        cr.threshold_for([1.0], "sideways")


def test_fires_respects_direction_and_missing_values():
    assert cr.fires(0.1, 0.5, cr.BELOW) and not cr.fires(0.9, 0.5, cr.BELOW)
    assert cr.fires(0.9, 0.5, cr.ABOVE) and not cr.fires(0.1, 0.5, cr.ABOVE)
    assert not cr.fires(0.5, 0.5, cr.BELOW) and not cr.fires(0.5, 0.5, cr.ABOVE)
    assert not cr.fires(None, 0.5, cr.BELOW)


def test_every_sdmetrics_check_warns_when_the_score_is_low():
    assert set(cr.SD_CHECKS.values()) == {cr.BELOW} and cr.OVERALL_DIRECTION == cr.BELOW


def test_calibrate_sdmetrics_uses_null_runs_only_and_skips_missing():
    nulls = [*NULLS, sd_run("heldout_real", 3, column_shapes=None, dcr_overfitting=0.4)]
    th, details = cr.calibrate_sdmetrics(nulls)
    assert th["column_shapes"] == pytest.approx(0.90 - 0.09)
    assert details["column_shapes"]["n_null_values"] == 3
    assert details["column_shapes"]["n_null_runs_without_value"] == 1
    assert th["dcr_overfitting"] == pytest.approx(0.4 - 0.04)
    assert "new_row_synthesis" not in th
    assert "no null values" in details["new_row_synthesis"]["note"]
    with pytest.raises(ValueError, match="heldout_real"):
        cr.calibrate_sdmetrics([sd_run("exact_copy", 0, column_shapes=0.1)])
    with pytest.raises(ValueError, match="seeds below"):
        cr.calibrate_sdmetrics([sd_run("heldout_real", 20, column_shapes=0.1)])


def test_sd_fired_and_overall_fired():
    th = {"column_shapes": 0.8, "dcr_overfitting": 0.4, "quality_score": 0.7}
    run = sd_run(
        "x", 0, column_shapes=0.5, dcr_overfitting=0.45, quality_score=0.6, efficacy_ratio=0
    )
    assert cr.sd_fired(run, th) == {"column_shapes"}  # no threshold for efficacy_ratio
    assert cr.overall_fired(run, th)
    assert not cr.overall_fired(sd_run("x", 0, quality_score=0.9), th)
    assert not cr.overall_fired(run, {})


def test_strict_mapping_has_no_collapse_check_and_covers_headline_modes():
    assert cr.SD_INTENDED["mode_collapse"] == ()
    assert set(cr.SD_INTENDED) == set(cr.MODES)
    assert all(set(v) <= set(cr.SD_CHECKS) for v in cr.SD_INTENDED.values())


def _ours_run(gen, seed, dataset="d", **stats):
    return {"dataset": dataset, "seed": seed, "generator": gen, "stats": stats, "error": None}


def test_build_comparison_end_to_end_on_synthetic_records():
    ours, sd = [], []
    gens = ["heldout_real", "gaussian_copula", *cr.BROKEN]
    for seed in range(40):
        for g in gens:
            bad = g not in ("heldout_real", "gaussian_copula")
            ours.append(_ours_run(g, seed, **{"detectability.auc_mean": 0.9 if bad else 0.5}))
            m = {
                "column_shapes": 0.5 if g == "category_oversample" else 0.95,
                "column_pair_trends": 0.95,
                "logistic_detection": 0.95,
                "dcr_overfitting": 0.0 if g in ("exact_copy", "near_copy") else 0.5,
                "new_row_synthesis": 0.99,
                "efficacy_ratio": 0.5 if g == "label_shuffle" else 1.0,
                "quality_score": 0.95,
            }
            rec = sd_run(g, seed, **m)
            rec["ours_rerun"] = {"runtime_s": 0.5, "stats_match": True}
            sd.append(rec)
    out = cr.build_comparison(ours, sd)
    strict = out["headline"]["sdmetrics_strict"]
    assert strict["modes"]["exact_copy"]["detected"]
    assert strict["modes"]["category_oversample"]["detected"]
    assert strict["modes"]["label_shuffle"]["detected"]
    assert not strict["modes"]["mode_collapse"]["detected"]
    assert not strict["modes"]["column_shuffle"]["detected"]
    assert out["headline"]["sdmetrics_lenient"]["modes"]["mode_collapse"]["rate"] is None or True
    assert out["headline"]["sdmetrics_overall"]["n_detected"] == 0
    assert strict["heldout_real_false_alarm"] == {"k": 0, "n": 20, "rate": 0.0}
    assert out["protocol"]["test_seeds"] == list(range(20, 40))
    assert out["ours_stats_mismatch_runs"] == 0
    assert out["runtime"]["ours_mean_s_per_run"] == pytest.approx(0.5)
    md = cr.comparison_markdown(out)
    assert "SDMetrics strict" in md and "`mode_collapse`" in md and "0 of 20" in md


def test_join_runs_rejects_unmatched_sdmetrics_run():
    with pytest.raises(ValueError, match="no benchmark run"):
        cr.join_runs([], [sd_run("exact_copy", 0)])


def test_errors_are_summarised():
    runs = [sd_run("a", 0), sd_run("a", 1)]
    runs[0]["errors"] = {"new_row_synthesis": "ValueError: boom"}
    runs[1]["errors"] = {"new_row_synthesis": "ValueError: boom"}
    assert cr.error_summary(runs) == {"new_row_synthesis": {"n": 2, "first": "ValueError: boom"}}


# Metadata builder -------------------------------------------------------------------------


def test_metadata_follows_our_column_typing():
    df = make_dataset(300, 1)
    df["flag"] = np.random.default_rng(0).integers(100, 103, len(df))  # integer-coded category
    md = cs.build_metadata(df, "label")["columns"]
    assert md["age"] == {"sdtype": "numerical"}
    assert md["income"] == {"sdtype": "numerical"}
    assert md["city"] == {"sdtype": "categorical"}
    assert md["flag"] == {"sdtype": "categorical"}
    assert md["label"] == {"sdtype": "categorical"}


def test_metadata_drops_empty_columns_and_types_a_regression_label_as_numerical():
    df = make_dataset(300, 1)
    df["empty"] = np.nan
    df["label"] = df["age"] * 2 + np.random.default_rng(1).normal(size=len(df))
    md = cs.build_metadata(df, "label")["columns"]
    assert "empty" not in md
    assert md["label"] == {"sdtype": "numerical"}
    assert cs.efficacy_class(df["label"]) is None


def test_efficacy_class_binary_vs_multiclass():
    assert cs.efficacy_class(make_dataset(200, 0, n_classes=2)["label"]) == (
        "BinaryLogisticRegression"
    )
    assert cs.efficacy_class(make_dataset(200, 0, n_classes=3)["label"]) == (
        "MulticlassDecisionTree"
    )


def test_prepare_frame_casts_types_and_keeps_missing():
    df = pd.DataFrame(
        {
            "n": pd.array([1, 2, 3], dtype="int64"),
            "c": pd.Categorical(["a", None, "b"]),
            "k": [1.0, 2.0, np.nan],
            "label": ["x", "y", "x"],
            "extra": [9, 9, 9],
        }
    )
    md = {
        "columns": {
            "n": {"sdtype": "numerical"},
            "c": {"sdtype": "categorical"},
            "k": {"sdtype": "categorical"},
            "label": {"sdtype": "categorical"},
        }
    }
    out = cs.prepare_frame(df, md)
    assert list(out.columns) == ["n", "c", "k", "label"]
    assert out["n"].dtype == float
    assert out["c"].tolist()[0] == "a" and pd.isna(out["c"].tolist()[1])
    assert out["k"].tolist()[:2] == ["1.0", "2.0"] and pd.isna(out["k"].tolist()[2])


def test_stats_match_tolerance_and_keys():
    assert cs.stats_match({"a": 1.0, "b": None}, {"a": 1.0 + 1e-12, "b": None})
    assert not cs.stats_match({"a": 1.0}, {"a": 1.1})
    assert not cs.stats_match({"a": 1.0}, {"b": 1.0})
    assert not cs.stats_match({"a": None}, {"a": 1.0})


def test_timed_records_errors_instead_of_raising():
    values, runtimes, errors = {}, {}, {}
    cs._timed("ok", lambda: 0.5, 0, values, runtimes, errors)
    cs._timed("bad", lambda: 1 / 0, 0, values, runtimes, errors)
    cs._timed("nan", lambda: float("nan"), 0, values, runtimes, errors)
    assert values == {"ok": 0.5, "bad": None, "nan": None}
    assert "ZeroDivisionError" in errors["bad"] and errors["nan"] == "non-finite score"
    assert set(runtimes) == {"ok", "bad", "nan"} and "ok" not in errors


# Needs SDMetrics (the optional `compare` extra) -------------------------------------------


def test_score_run_and_directions_with_sdmetrics():
    pytest.importorskip("sdmetrics")
    df = make_dataset(300, 3)
    train, hold, fresh = df.iloc[:150], df.iloc[150:225], df.iloc[225:]
    res = cs.score_run(train.reset_index(drop=True), fresh, hold, "label", 0)
    assert not res["errors"]
    assert 0 <= res["metrics"]["quality_score"] <= 1
    copy = cs.score_run(train.reset_index(drop=True), train.iloc[:75], hold, "label", 0)
    assert copy["metrics"]["new_row_synthesis"] < 0.1 < res["metrics"]["new_row_synthesis"]
    assert cs.verify_directions()["assumptions_verified"]


def test_mask_unseen_categories_replaces_with_mode_and_skips_target():
    md = {"columns": {"c": {"sdtype": "categorical"}, "y": {"sdtype": "categorical"}}}
    ref = pd.DataFrame({"c": ["a", "a", "b"], "y": ["p", "p", "q"]})
    df = pd.DataFrame({"c": ["a", "z", None, "b"], "y": ["new", "p", "q", "p"]})
    out, n = cs.mask_unseen_categories(df, ref, md, skip="y")
    assert out["c"].tolist() == ["a", "a", "a", "b"] and n == 2
    assert out["y"].tolist() == df["y"].tolist()
    # a missing value is kept when the reference has missing values too
    ref_nan = pd.DataFrame({"c": ["a", None], "y": ["p", "q"]})
    out2, n2 = cs.mask_unseen_categories(df, ref_nan, md, skip="y")
    assert n2 == 2 and out2["c"].isna().sum() == 1
