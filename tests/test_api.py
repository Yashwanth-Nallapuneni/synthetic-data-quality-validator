import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api import app
from tests.conftest import make_dataset

client = TestClient(app)
DATA = Path(__file__).resolve().parent.parent / "data"


def csv_bytes(df) -> bytes:
    return df.to_csv(index=False).encode()


def post(real, synth, label="label", holdout=None):
    files = {
        "real": ("real.csv", io.BytesIO(real), "text/csv"),
        "synth": ("synth.csv", io.BytesIO(synth), "text/csv"),
    }
    if holdout is not None:
        files["holdout"] = ("holdout.csv", io.BytesIO(holdout), "text/csv")
    return client.post("/validate", files=files, data={"label_column": label})


@pytest.fixture
def frames():
    return make_dataset(300, 1), make_dataset(300, 2)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_happy_path(frames):
    r = post(csv_bytes(frames[0]), csv_bytes(frames[1]))
    assert r.status_code == 200
    body = r.json()
    assert body["severity"] in {"clean", "issues"}
    assert body["report"]["utility"]["status"] == "ok"
    assert set(body["report"]) >= {
        "fidelity",
        "utility",
        "privacy",
        "diversity",
        "dependence",
        "detectability",
        "warnings",
    }
    assert body["report"]["privacy"]["mode"] == "split_real_50_50"


def test_label_in_first_column(frames):
    real = make_dataset(300, 1, label_position="first")
    synth = make_dataset(300, 2, label_position="first")
    r = post(csv_bytes(real), csv_bytes(synth))
    assert r.status_code == 200
    assert r.json()["report"]["utility"]["status"] == "ok"


def test_missing_label_is_400_not_500(frames):
    r = post(csv_bytes(frames[0]), csv_bytes(frames[1]), label="nope")
    assert r.status_code == 400
    assert "nope" in r.json()["detail"]


def test_label_missing_only_in_synth_is_400(frames):
    r = post(csv_bytes(frames[0]), csv_bytes(frames[1].drop(columns=["label"])))
    assert r.status_code == 400
    assert "synthetic" in r.json()["detail"]


def test_column_mismatch_lists_columns(frames):
    synth = frames[1].rename(columns={"age": "years"})
    r = post(csv_bytes(frames[0]), csv_bytes(synth))
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert "age" in detail and "years" in detail


def test_unparseable_csv_is_400(frames):
    r = post(b"", csv_bytes(frames[1]))
    assert r.status_code == 400
    r = post(b"a,b\n1,2,3,4\n5\n\x00\x01", csv_bytes(frames[1]))
    assert r.status_code == 400


def test_header_only_csv_is_400(frames):
    r = post(b"age,income,label\n", csv_bytes(frames[1]))
    assert r.status_code == 400


def test_oversize_upload_is_413(frames, monkeypatch):
    monkeypatch.setenv("MAX_UPLOAD_BYTES", "1000")
    r = post(csv_bytes(frames[0]), csv_bytes(frames[1]))
    assert r.status_code == 413


def test_row_cap_is_413(frames, monkeypatch):
    monkeypatch.setenv("MAX_ROWS", "50")
    r = post(csv_bytes(frames[0]), csv_bytes(frames[1]))
    assert r.status_code == 413
    assert "rows" in r.json()["detail"]


def test_tiny_sample_returns_200_with_insufficient_status():
    real = (DATA / "real.csv").read_bytes()
    synth = (DATA / "synth.csv").read_bytes()
    r = post(real, synth)
    assert r.status_code == 200
    assert r.json()["report"]["utility"]["status"] == "insufficient_data"


def test_internal_value_error_is_500_not_400(frames, monkeypatch):
    """Only InputError is a client error; a stray ValueError is an internal bug."""
    import app.api as api_module

    def boom(*args, **kwargs):
        raise ValueError("internal bug")

    monkeypatch.setattr(api_module, "generate_trust_report", boom)
    r = post(csv_bytes(frames[0]), csv_bytes(frames[1]))
    assert r.status_code == 500
    assert "internal bug" not in r.json()["detail"]


def test_optional_holdout_upload_switches_privacy_mode(frames):
    holdout = make_dataset(300, 3)
    r = post(csv_bytes(frames[0]), csv_bytes(frames[1]), holdout=csv_bytes(holdout))
    assert r.status_code == 200
    privacy = r.json()["report"]["privacy"]
    assert privacy["mode"] == "provided_holdout"
    assert privacy["n_holdout"] == 300


def test_holdout_with_different_columns_is_400(frames):
    holdout = make_dataset(300, 3).rename(columns={"age": "years"})
    r = post(csv_bytes(frames[0]), csv_bytes(frames[1]), holdout=csv_bytes(holdout))
    assert r.status_code == 400
    assert "holdout" in r.json()["detail"]


def test_holdout_row_cap_is_413(frames, monkeypatch):
    monkeypatch.setenv("MAX_ROWS", "400")
    big_holdout = make_dataset(500, 3)
    r = post(csv_bytes(frames[0]), csv_bytes(frames[1]), holdout=csv_bytes(big_holdout))
    assert r.status_code == 413
    assert "holdout" in r.json()["detail"]
