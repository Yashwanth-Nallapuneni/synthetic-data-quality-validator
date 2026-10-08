import json

import pytest

from app.errors import InputError
from app.report import generate_trust_report
from app.thresholds import Thresholds, load_thresholds
from tests.conftest import codes, make_dataset


def test_defaults_when_no_path():
    assert load_thresholds() == Thresholds()


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


def test_report_uses_given_thresholds():
    real, synth = make_dataset(400, 1), make_dataset(400, 2)
    strict = Thresholds(variance_ratio_low=5.0)
    report = generate_trust_report(real, synth, "label", thresholds=strict)
    assert "diversity_mode_collapse" in codes(report)
