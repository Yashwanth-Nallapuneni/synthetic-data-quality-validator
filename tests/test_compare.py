"""Calibration, direction and metadata logic of the SDMetrics comparison (no SDMetrics needed)."""

import sys

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


def _records():
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
    return ours, sd


def test_build_comparison_end_to_end_on_synthetic_records():
    ours, sd = _records()
    out = cr.build_comparison(ours, sd)
    strict = out["headline"]["sdmetrics_strict"]
    assert strict["modes"]["exact_copy"]["detected"]
    assert strict["modes"]["category_oversample"]["detected"]
    assert strict["modes"]["label_shuffle"]["detected"]
    assert not strict["modes"]["mode_collapse"]["detected"]
    assert not strict["modes"]["column_shuffle"]["detected"]
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


def test_main_writes_files_and_updates_the_readme(tmp_path):
    import json

    ours, sd = _records()
    for name, rows in (("runs.jsonl", ours), ("sdmetrics_runs.jsonl", sd)):
        (tmp_path / name).write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    readme = tmp_path / "README.md"
    readme.write_text(f"intro\n{cr.START_MARK}\nSTALE\n{cr.END_MARK}\noutro\n")
    full = tmp_path / "full.md"
    full.write_text(f"{cr.FULL_START_MARK}\n{cr.FULL_END_MARK}\n")
    argv = ["--results", str(tmp_path), "--update-readme", str(readme), "--update-full", str(full)]
    assert cr.main(argv) == 0
    assert "Per dataset" in full.read_text()
    assert json.loads((tmp_path / "comparison.json").read_text())["headline"]
    assert "SDMetrics strict" in (tmp_path / "comparison.md").read_text()
    body = readme.read_text()
    assert "STALE" not in body and "SDMetrics strict" in body and body.startswith("intro")


# The runner with a stand-in for SDMetrics (so it is covered where the extra is not installed)


@pytest.fixture
def fake_sdmetrics(monkeypatch):
    import sys
    import types

    class Report:
        def generate(self, real, fake, md, verbose=False):
            self.n = len(real)

        def get_properties(self):
            return pd.DataFrame(
                {"Property": ["Column Shapes", "Column Pair Trends"], "Score": [0.9, 0.8]}
            )

        def get_score(self):
            return 0.85

        def get_details(self, prop):
            if prop == "Column Shapes":
                return pd.DataFrame(
                    {
                        "Column": ["a", "b", "c"],
                        "Metric": ["KSComplement", "TVComplement", "TVComplement"],
                        "Score": [0.95, 0.4, 0.9],
                    }
                )
            return pd.DataFrame(
                {
                    "Column 1": ["a", "a"],
                    "Column 2": ["b", "c"],
                    "Metric": ["CorrelationSimilarity", "ContingencySimilarity"],
                    "Score": [np.nan, 0.7],
                }
            )

    def metric(value):
        class M:
            @staticmethod
            def compute(*args, **kwargs):
                if value is None:
                    raise ValueError("boom")
                return value

        return M

    root = types.ModuleType("sdmetrics")
    root.__version__ = "0.0-fake"
    reports = types.ModuleType("sdmetrics.reports")
    single = types.ModuleType("sdmetrics.reports.single_table")
    single.QualityReport = Report
    tables = types.ModuleType("sdmetrics.single_table")
    tables.BinaryLogisticRegression = metric(0.6)
    tables.MulticlassDecisionTreeClassifier = metric(0.5)
    tables.DCROverfittingProtection = metric(0.5)
    tables.LogisticDetection = metric(None)  # errors are recorded, not raised
    tables.NewRowSynthesis = metric(0.9)
    for name, mod in {
        "sdmetrics": root,
        "sdmetrics.reports": reports,
        "sdmetrics.reports.single_table": single,
        "sdmetrics.single_table": tables,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)
    return tables


def test_score_run_records_values_runtimes_and_errors(fake_sdmetrics):
    df = make_dataset(200, 1)
    res = cs.score_run(
        df.iloc[:100].reset_index(drop=True), df.iloc[100:150], df.iloc[150:], "label", 0
    )
    m = res["metrics"]
    assert m["column_shapes"] == 0.9 and m["column_pair_trends"] == 0.8
    assert m["quality_score"] == 0.85 and m["new_row_synthesis"] == 0.9
    assert m["efficacy_ratio"] == pytest.approx(1.0) and m["logistic_detection"] is None
    assert (
        set(res["errors"]) == {"logistic_detection"}
        and "boom" in res["errors"]["logistic_detection"]
    )
    assert res["runtime_total_s"] >= 0 and res["efficacy_model"] == "BinaryLogisticRegression"


def test_score_run_handles_a_zero_trtr_and_regression_labels(fake_sdmetrics):
    fake_sdmetrics.BinaryLogisticRegression = type(
        "M", (), {"compute": staticmethod(lambda *a, **k: 0.0)}
    )
    df = make_dataset(200, 1)
    args = (df.iloc[:100].reset_index(drop=True), df.iloc[100:150], df.iloc[150:], "label", 0)
    res = cs.score_run(*args)
    assert (
        res["metrics"]["efficacy_ratio"] is None and "undefined" in res["errors"]["efficacy_ratio"]
    )
    reg = df.assign(label=df["age"] * 2 + np.random.default_rng(0).normal(size=len(df)))
    res = cs.score_run(
        reg.iloc[:100].reset_index(drop=True), reg.iloc[100:150], reg.iloc[150:], "label", 0
    )
    assert res["efficacy_model"] is None and "efficacy_ratio" not in res["metrics"]


def test_main_writes_runs_meta_and_directions(fake_sdmetrics, monkeypatch, tmp_path):
    import json

    from experiments.benchmark import Dataset, run_unit

    ds = Dataset("toy", 0, "label")
    df = make_dataset(400, 5)
    results = tmp_path / "bench"
    results.mkdir()
    (results / "runs.jsonl").write_text(
        "\n".join(json.dumps(r) for r in run_unit(df, ds, 0)) + "\n"
    )
    monkeypatch.setattr(cs, "RESULTS", results)
    monkeypatch.setattr(cs, "DATASETS", (ds,))
    monkeypatch.setattr(cs, "load_dataset", lambda d: (df, {}))
    monkeypatch.setattr(cs, "verify_directions", lambda: {"scores": {}, "assumptions_verified": []})
    out = tmp_path / "out"
    assert cs.main(["--seeds", "1", "--jobs", "1", "--out", str(out)]) == 0
    runs = [json.loads(line) for line in (out / "sdmetrics_runs.jsonl").read_text().splitlines()]
    assert len(runs) == 9 and all(r["ours_rerun"]["stats_match"] for r in runs)
    meta = json.loads((out / "sdmetrics_meta.json").read_text())
    assert meta["n_runs"] == 9 and meta["n_ours_stats_mismatch"] == 0
    assert meta["versions"]["sdmetrics"] == "0.0-fake"
    assert (out / "sdmetrics_directions.json").exists()
    # without re-running our validator no timing is recorded
    assert cs.main(["--seeds", "1", "--jobs", "1", "--out", str(out), "--no-rerun-ours"]) == 0
    assert "ours_rerun" not in json.loads(
        (out / "sdmetrics_runs.jsonl").read_text().splitlines()[0]
    )


# Worst-column statistics ------------------------------------------------------------------


def test_worst_scores_split_by_metric_and_skip_unscored_pairs():
    shapes = pd.DataFrame(
        {
            "Column": ["a", "b", "c", "d"],
            "Metric": ["KSComplement", "KSComplement", "TVComplement", "TVComplement"],
            "Score": [0.9, 0.8, 0.95, 0.6],
        }
    )
    pairs = pd.DataFrame({"Score": [np.nan, 0.7, 0.9]})
    assert cs.worst_scores(shapes, pairs) == {
        "shape_min": 0.6,
        "shape_min_numeric": 0.8,
        "shape_min_categorical": 0.6,
        "pair_min": 0.7,
    }


def test_worst_scores_none_when_a_group_has_no_scored_member():
    shapes = pd.DataFrame({"Column": ["a"], "Metric": ["KSComplement"], "Score": [0.9]})
    out = cs.worst_scores(shapes, pd.DataFrame({"Score": [np.nan, np.nan]}))
    assert out["shape_min_categorical"] is None and out["pair_min"] is None
    assert out["shape_min_numeric"] == 0.9
    assert cs._min_score(pd.Series([], dtype=float)) is None


def test_score_worst_records_minima_and_matches_averages(fake_sdmetrics):
    df = make_dataset(200, 1)
    res = cs.score_worst(df.iloc[:100].reset_index(drop=True), df.iloc[100:150], "label", 0)
    m = res["metrics"]
    assert m["column_shapes"] == 0.9 and m["column_pair_trends"] == 0.8
    assert m["shape_min"] == 0.4 and m["shape_min_numeric"] == 0.95
    assert m["shape_min_categorical"] == 0.4 and m["pair_min"] == 0.7
    assert not res["errors"] and set(cs.WORST_KEYS) <= set(m)


def test_score_worst_records_a_quality_error_as_missing_values(fake_sdmetrics, monkeypatch):
    def boom(self, *a, **k):
        raise ValueError("boom")

    monkeypatch.setattr(
        sys.modules["sdmetrics.reports.single_table"].QualityReport, "generate", boom
    )
    df = make_dataset(200, 1)
    res = cs.score_worst(df.iloc[:100].reset_index(drop=True), df.iloc[100:150], "label", 0)
    assert "boom" in res["errors"]["quality_score"]
    assert all(res["metrics"][k] is None for k in cs.WORST_KEYS)


def test_main_worst_only_writes_a_sidecar_and_leaves_existing_files(
    fake_sdmetrics, monkeypatch, tmp_path
):
    import json

    from experiments.benchmark import Dataset

    ds = Dataset("toy", 0, "label")
    df = make_dataset(400, 5)
    monkeypatch.setattr(cs, "DATASETS", (ds,))
    monkeypatch.setattr(cs, "load_dataset", lambda d: (df, {}))
    (tmp_path / "sdmetrics_runs.jsonl").write_text("keep\n")
    assert cs.main(["--worst-only", "--seeds", "1", "--jobs", "1", "--out", str(tmp_path)]) == 0
    rows = [json.loads(x) for x in (tmp_path / cs.WORST_FILE).read_text().splitlines()]
    assert len(rows) == 9 and all(r["metrics"]["pair_min"] == 0.7 for r in rows)
    assert (tmp_path / "sdmetrics_runs.jsonl").read_text() == "keep\n"


def test_run_worst_unit_records_a_failure_outside_the_metric(monkeypatch):
    from experiments.benchmark import Dataset

    monkeypatch.setattr(cs, "build_synth", lambda *a: 1 / 0)
    rows = cs.run_worst_unit(make_dataset(400, 5), Dataset("toy", 0, "label"), 0)
    assert len(rows) == 9 and "ZeroDivisionError" in rows[0]["errors"]["run"]


def _worst_records(oversample_min=0.1):
    ours, sd = _records()
    worst = []
    for r in sd:
        g = r["generator"]
        w = {k: r[k] for k in ("dataset", "seed", "generator")}
        w["metrics"] = {
            "column_shapes": r["metrics"]["column_shapes"],
            "column_pair_trends": r["metrics"]["column_pair_trends"],
            "shape_min": oversample_min if g == "category_oversample" else 0.9,
            "shape_min_numeric": 0.9,
            "shape_min_categorical": 0.9,
            "pair_min": 0.1 if g == "column_shuffle" else 0.8,
        }
        worst.append(w)
    return ours, sd, worst


def test_merge_worst_adds_fields_and_counts_recompute_mismatches():
    _, sd, worst = _worst_records()
    merged, bad = cr.merge_worst(sd, worst)
    assert bad == 0 and merged[0]["metrics"]["pair_min"] == 0.8
    assert merged[0]["metrics"]["column_shapes"] == sd[0]["metrics"]["column_shapes"]
    assert "pair_min" not in sd[0]["metrics"]  # inputs are not modified
    worst[0]["metrics"]["column_shapes"] += 0.1
    worst[1]["metrics"]["column_pair_trends"] = None
    assert cr.merge_worst(sd, worst)[1] == 2
    assert cr.merge_worst(sd, [])[0] == sd  # no sidecar: nothing is added


def test_worst_variant_detects_what_the_averages_miss():
    ours, sd, worst = _worst_records()
    out = cr.build_comparison(ours, sd, worst)
    head = out["headline"]
    assert not head["sdmetrics_strict"]["modes"]["column_shuffle"]["detected"]
    assert head["sdmetrics_worst_strict"]["modes"]["column_shuffle"]["detected"]
    assert head["sdmetrics_worst_strict"]["modes"]["category_oversample"]["detected"]
    assert not head["sdmetrics_worst_strict"]["modes"]["mode_collapse"]["detected"]
    assert head["sdmetrics_worst_lenient"]["modes"]["mode_collapse"]["detected"] is False
    # the original strict and lenient views do not see the new statistics
    assert not head["sdmetrics_lenient"]["modes"]["column_shuffle"]["detected"]
    assert head["sdmetrics_worst_strict"]["heldout_real_false_alarm"]["k"] == 0
    assert out["worst_recompute_mismatches"] == 0
    assert out["sdmetrics_thresholds"]["pair_min"] == pytest.approx(0.8 - 0.08)
    assert "SDMetrics strict (worst column/pair)" in cr.comparison_markdown(out)


def test_worst_mapping_changes_only_two_modes_and_uses_known_checks():
    changed = {m for m in cr.SD_INTENDED if cr.SD_INTENDED[m] != cr.SD_INTENDED_WORST[m]}
    assert changed == {"column_shuffle", "category_oversample"}
    assert cr.SD_INTENDED_WORST["mode_collapse"] == ()
    assert all(
        set(v) <= set(cr.SD_CHECKS) | set(cr.WORST_CHECKS) for v in cr.SD_INTENDED_WORST.values()
    )
    assert set(cr.WORST_CHECKS.values()) == {cr.BELOW}
