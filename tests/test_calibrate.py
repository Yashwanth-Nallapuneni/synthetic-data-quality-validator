import json

import pytest

from app.checks import evaluate_checks
from app.thresholds import Thresholds
from experiments import calibrate as cal


def run(gen, seed, dataset="d", **stats):
    return {"dataset": dataset, "seed": seed, "generator": gen, "stats": stats, "error": None}


NULLS = [
    run(
        "heldout_real",
        s,
        **{
            "detectability.auc_mean": 0.50 + 0.01 * s,
            "diversity.coverage": 0.80 - 0.01 * s,
            "diversity.mean_variance_ratio": 1.0 + 0.02 * (s - 2),
            "utility.logistic_regression.accuracy_ratio": 0.99 - 0.01 * s,
            "utility.hist_gradient_boosting.accuracy_ratio": 0.98,
        },
    )
    for s in (0, 2, 4)
]


def test_upper_tail_threshold_is_max_plus_margin():
    th, details = cal.calibrate_thresholds(NULLS)
    assert th.c2st_auc == pytest.approx(0.54 + 0.054)
    assert details["c2st_auc"]["n_null_values"] == 3
    assert details["c2st_auc"]["max_per_check_false_alarm_rate_before_margin"] == pytest.approx(
        0.25
    )


def test_lower_tail_threshold_is_min_minus_margin_and_pools_models():
    th, _ = cal.calibrate_thresholds(NULLS)
    assert th.coverage_min == pytest.approx(0.76 - 0.076)
    assert th.utility_accuracy_ratio == pytest.approx(0.95 - 0.095)


def test_two_sided_variance_ratio_uses_both_ends():
    th, _ = cal.calibrate_thresholds(NULLS)
    assert th.variance_ratio_low == pytest.approx(0.96 - 0.096)
    assert th.variance_ratio_high == pytest.approx(1.04 + 0.104)
    assert th.variance_ratio_low < 1.0 < th.variance_ratio_high


def test_absolute_margin_floor_applies_near_zero():
    nulls = [run("heldout_real", 0, **{"dependence.mean_abs_diff": 0.0})]
    th, _ = cal.calibrate_thresholds(nulls)
    assert th.dependence_mean_abs_diff == pytest.approx(0.01)


def test_null_values_do_not_fire_with_calibrated_thresholds():
    th, _ = cal.calibrate_thresholds(NULLS)
    for r in NULLS:
        assert evaluate_checks(r["stats"], th) == []


def test_missing_statistics_keep_provisional_defaults():
    th, details = cal.calibrate_thresholds(NULLS)
    assert th.utility_r2_drop == Thresholds().utility_r2_drop
    assert "note" in details["utility_r2_drop"]


def test_calibration_rejects_non_null_runs():
    with pytest.raises(ValueError):
        cal.calibrate_thresholds([*NULLS, run("exact_copy", 0)])


def test_broken_runs_do_not_change_thresholds():
    th1, _ = cal.calibrate_thresholds(NULLS)
    broken = [run("column_shuffle", 0, **{"detectability.auc_mean": 0.99})]
    all_runs = NULLS + broken
    null_only = [r for r in all_runs if r["generator"] == "heldout_real"]
    th2, _ = cal.calibrate_thresholds(null_only)
    assert th1 == th2


def test_split_runs_even_calibration_odd_test():
    runs = [run("heldout_real", s) for s in range(6)]
    c, t = cal.split_runs(runs)
    assert {r["seed"] for r in c} == {0, 2, 4} and {r["seed"] for r in t} == {1, 3, 5}


def test_detection_table_intended_and_any():
    th = Thresholds()
    runs = [
        run("column_shuffle", 1, **{"dependence.mean_abs_diff": 0.5}),
        run("column_shuffle", 3, **{"dependence.mean_abs_diff": 0.0}),
        run("heldout_real", 1, **{"dependence.mean_abs_diff": 0.0}),
        run("label_shuffle", 1, **{"detectability.auc_mean": 0.99}),
    ]
    t = cal.detection_table(runs, th)
    assert t["column_shuffle"]["intended_fired"] == 0.5
    assert t["column_shuffle"]["code_hit_rate"]["dependence_shift"] == 0.5
    assert t["heldout_real"]["any_warning"] == 0.0 and t["heldout_real"]["intended_fired"] is None
    # label_shuffle's intended code is utility_degradation only; other warnings do not count
    assert t["label_shuffle"]["intended_fired"] == 0.0
    assert t["label_shuffle"]["any_warning"] == 1.0
    assert "dependence_shift" in cal.matrix_markdown(t)


def test_lodo_never_uses_held_out_datasets_null_runs():
    runs = []
    for d, auc in (("a", 0.5), ("b", 0.6), ("c", 0.9)):
        runs += [
            run("heldout_real", s, dataset=d, **{"detectability.auc_mean": auc}) for s in (0, 1)
        ]
    out = cal.lodo_table(runs, ["a", "b", "c"])
    # held out "c": thresholds come from a and b only, so below 0.9
    assert out["thresholds"]["c"]["c2st_auc"] < 0.9
    # held out "a": thresholds include c's 0.9
    assert out["thresholds"]["a"]["c2st_auc"] > 0.9


def test_main_writes_outputs(tmp_path):
    runs = []
    for d in ("a", "b"):
        for g in cal.ORDER:
            for s in range(4):
                runs.append(
                    run(
                        g,
                        s,
                        dataset=d,
                        **{"detectability.auc_mean": 0.5 + 0.4 * (g != "heldout_real")},
                    )
                )
    (tmp_path / "runs.jsonl").write_text("\n".join(json.dumps(r) for r in runs))
    (tmp_path / "meta.json").write_text(json.dumps({"datasets": [{"name": "a"}, {"name": "b"}]}))
    out = tmp_path / "thr.json"
    assert cal.main(["--results", str(tmp_path), "--thresholds-out", str(out)]) == 0
    saved = json.loads(out.read_text())
    assert saved["calibration"]["calibration_seeds"] == [0, 2]
    assert saved["calibration"]["generators_used_for_calibration"] == ["heldout_real"]
    assert saved["thresholds"]["c2st_auc"] == pytest.approx(0.55)
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["heldout_real_false_alarm"]["k"] == 0
    assert (tmp_path / "detection_matrix.md").exists()


def test_privacy_medium_share_is_the_null_max_below_high():
    nulls = [
        run("heldout_real", s, **{"privacy.share_closer_than_holdout_p5": v})
        for s, v in ((0, 0.04), (2, 0.07))
    ]
    th, _ = cal.calibrate_thresholds(nulls)
    assert th.privacy_medium_share == pytest.approx(0.07)
    assert th.privacy_high_share == pytest.approx(0.08)
