import json

import pytest

from app import thresholds as thresholds_module
from app.errors import InputError
from app.report import generate_trust_report
from app.thresholds import Thresholds, load_thresholds
from tests.conftest import codes, make_dataset


def test_provisional_defaults_when_no_calibrated_file(tmp_path, monkeypatch):
    monkeypatch.setattr(thresholds_module, "CALIBRATED_PATH", tmp_path / "missing.json")
    assert load_thresholds() == Thresholds()


def test_no_path_loads_calibrated_file(tmp_path, monkeypatch):
    p = tmp_path / "cal.json"
    p.write_text(json.dumps({"thresholds": {"c2st_auc": 0.61}, "calibration": {"rule": "x"}}))
    monkeypatch.setattr(thresholds_module, "CALIBRATED_PATH", p)
    t = load_thresholds()
    assert t.c2st_auc == 0.61
    assert t.privacy_high_share == Thresholds().privacy_high_share
    real, synth = make_dataset(200, 1), make_dataset(200, 2)
    assert generate_trust_report(real, synth, "label")["thresholds"]["c2st_auc"] == 0.61


def test_shipped_calibrated_file_is_valid_and_documented():
    if not thresholds_module.CALIBRATED_PATH.exists():
        pytest.skip("v2 calibrated file not generated yet")
    assert thresholds_module.CALIBRATED_PATH.exists()
    saved = json.loads(thresholds_module.CALIBRATED_PATH.read_text())
    assert set(saved) == {"thresholds", "calibration"}
    assert set(saved["thresholds"]) <= set(Thresholds().to_dict())
    loaded = load_thresholds().to_dict()
    assert {k: loaded[k] for k in saved["thresholds"]} == saved["thresholds"]
    c = saved["calibration"]
    assert {"rule", "date", "calibration_seeds", "datasets"} <= set(c)
    assert c["generators_used_for_calibration"] == ["heldout_real"]


def test_json_overrides_selected_values(tmp_path):
    p = tmp_path / "t.json"
    p.write_text(json.dumps({"c2st_auc": 0.6, "categorical_mean_l1": 1}))
    t = load_thresholds(p)
    assert t.c2st_auc == 0.6
    assert t.categorical_mean_l1 == 1.0
    assert t.privacy_high_share == Thresholds().privacy_high_share


@pytest.mark.parametrize("content", ['{"nope": 1}', "[1]", "not json", '{"c2st_auc": "x"}'])
def test_bad_threshold_files_raise_input_error(tmp_path, content):
    p = tmp_path / "t.json"
    p.write_text(content)
    with pytest.raises(InputError):
        load_thresholds(p)


def test_missing_file_raises_input_error(tmp_path):
    with pytest.raises(InputError):
        load_thresholds(tmp_path / "missing.json")


def test_package_data_includes_calibrated_file():
    import tomllib
    from pathlib import Path

    cfg = tomllib.loads((Path(__file__).resolve().parent.parent / "pyproject.toml").read_text())
    assert "calibrated_thresholds.json" in cfg["tool"]["setuptools"]["package-data"]["app"]


def test_report_uses_given_thresholds():
    real, synth = make_dataset(400, 1), make_dataset(400, 2)
    strict = Thresholds(variance_ratio_low=5.0)
    report = generate_trust_report(real, synth, "label", thresholds=strict)
    assert "diversity_mode_collapse" in codes(report)
