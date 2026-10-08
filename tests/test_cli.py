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
    assert "memorization" in capsys.readouterr().out


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
