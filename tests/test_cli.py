import json

from app.cli import main
from tests.conftest import make_dataset


def write(tmp_path, name, df):
    path = tmp_path / name
    df.to_csv(path, index=False)
    return str(path)


def test_exit_1_on_warnings(tmp_path, capsys):
    real = make_dataset(300, 1)
    synth = real.copy()  # exact copy: privacy warning
    code = main(
        [
            "--real",
            write(tmp_path, "r.csv", real),
            "--synth",
            write(tmp_path, "s.csv", synth),
            "--label",
            "label",
        ]
    )
    assert code == 1
    assert "[privacy_memorisation]" in capsys.readouterr().out


def test_exit_0_when_clean(tmp_path):
    out = tmp_path / "report.json"
    real, synth = make_dataset(600, 1), make_dataset(600, 2)
    code = main(
        [
            "--real",
            write(tmp_path, "r.csv", real),
            "--synth",
            write(tmp_path, "s.csv", synth),
            "--label",
            "label",
            "--out",
            str(out),
        ]
    )
    assert code == 0
    assert json.loads(out.read_text())["warnings"] == []


def test_exit_2_on_missing_file(tmp_path):
    assert main(["--real", str(tmp_path / "nope.csv"), "--synth", "x.csv", "--label", "l"]) == 2


def test_exit_2_on_missing_label(tmp_path):
    df = make_dataset(100, 1)
    p = write(tmp_path, "r.csv", df)
    assert main(["--real", p, "--synth", p, "--label", "nope"]) == 2


def test_exit_2_on_column_mismatch(tmp_path):
    a = make_dataset(100, 1)
    b = a.rename(columns={"age": "years"})
    assert (
        main(
            [
                "--real",
                write(tmp_path, "a.csv", a),
                "--synth",
                write(tmp_path, "b.csv", b),
                "--label",
                "label",
            ]
        )
        == 2
    )


def test_exit_2_on_usage_error():
    import pytest

    with pytest.raises(SystemExit) as e:
        main([])
    assert e.value.code == 2


def test_holdout_and_thresholds_options_and_summary(tmp_path, capsys):
    real, synth, holdout = make_dataset(400, 1), make_dataset(400, 2), make_dataset(400, 3)
    thresholds = tmp_path / "t.json"
    thresholds.write_text(json.dumps({"variance_ratio_low": 5.0}))
    out = tmp_path / "report.json"
    code = main(
        [
            "--real",
            write(tmp_path, "r.csv", real),
            "--synth",
            write(tmp_path, "s.csv", synth),
            "--holdout",
            write(tmp_path, "h.csv", holdout),
            "--thresholds",
            str(thresholds),
            "--label",
            "label",
            "--out",
            str(out),
        ]
    )
    text = capsys.readouterr().out
    report = json.loads(out.read_text())
    assert code == 1
    assert report["privacy"]["mode"] == "provided_holdout"
    assert "mode collapse" in text
    assert "Detectability (C2ST): AUC" in text
    assert "Dependence: mean abs difference" in text
    assert "Utility logistic_regression" in text
    assert report["thresholds"]["variance_ratio_low"] == 5.0


def test_exit_2_on_bad_thresholds_file_and_bad_holdout(tmp_path):
    df = make_dataset(100, 1)
    p = write(tmp_path, "r.csv", df)
    bad = tmp_path / "t.json"
    bad.write_text('{"nope": 1}')
    base = ["--real", p, "--synth", p, "--label", "label"]
    assert main(base + ["--thresholds", str(bad)]) == 2
    holdout = write(tmp_path, "h.csv", df.rename(columns={"age": "years"}))
    assert main(base + ["--holdout", holdout]) == 2
