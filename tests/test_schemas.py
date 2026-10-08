import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import __version__
from app.api import app
from app.report import build_response, generate_trust_report
from app.schemas import SCHEMA_VERSION, ValidationResponse
from tests.conftest import make_dataset


def _regression(n, seed):
    rng = np.random.default_rng(seed)
    df = make_dataset(n, seed)
    df["label"] = df["income"] * 0.001 + df["age"] + rng.normal(0, 1, n)
    return df


SCENARIOS = {
    "binary": lambda: (make_dataset(300, 1), make_dataset(300, 2), None),
    "multiclass": lambda: (
        make_dataset(400, 1, n_classes=3),
        make_dataset(400, 2, n_classes=3),
        None,
    ),
    "regression": lambda: (_regression(300, 1), _regression(300, 2), None),
    "tiny": lambda: (make_dataset(8, 1), make_dataset(8, 2), None),
    "holdout": lambda: (make_dataset(300, 1), make_dataset(300, 2), make_dataset(200, 3)),
    "label_first_exact_copy": lambda: (
        make_dataset(200, 1, label_position="first"),
        make_dataset(200, 1, label_position="first"),
        None,
    ),
    "categorical_only": lambda: (
        make_dataset(200, 1).drop(columns=["age", "income"]),
        make_dataset(200, 2).drop(columns=["age", "income"]),
        None,
    ),
}


@pytest.mark.parametrize("name", SCENARIOS)
def test_real_reports_match_schema(name):
    real, synth, holdout = SCENARIOS[name]()
    report = generate_trust_report(real, synth, "label", holdout_df=holdout)
    body = ValidationResponse.model_validate(build_response(report))
    assert body.schema_version == SCHEMA_VERSION
    assert body.app_version == __version__
    assert body.severity == ("issues" if report["warnings"] else "clean")
    assert body.report.seed == 42


def test_schema_covers_expected_statuses():
    tiny = ValidationResponse.model_validate(
        build_response(generate_trust_report(make_dataset(8, 1), make_dataset(8, 2), "label"))
    )
    assert tiny.report.utility.status == "insufficient_data"
    assert tiny.report.detectability.status == "insufficient_data"
    assert tiny.report.privacy.privacy_risk_level == "UNKNOWN"
    reg = ValidationResponse.model_validate(
        build_response(generate_trust_report(_regression(300, 1), _regression(300, 2), "label"))
    )
    assert reg.report.utility.task == "regression"
    assert "r2_drop" in next(iter(reg.report.utility.models.values())).tstr_vs_trtr.model_dump()


def test_schema_rejects_unknown_keys():
    envelope = build_response(
        generate_trust_report(make_dataset(300, 1), make_dataset(300, 2), "label")
    )
    envelope["report"]["privacy"]["surprise"] = 1
    with pytest.raises(ValueError):
        ValidationResponse.model_validate(envelope)


def test_openapi_contains_models():
    spec = TestClient(app).get("/openapi.json").json()
    schemas = spec["components"]["schemas"]
    for name in ("ValidationResponse", "Report", "Warning", "HealthResponse", "Privacy"):
        assert name in schemas
    ok = spec["paths"]["/validate"]["post"]["responses"]["200"]["content"]["application/json"]
    assert ok["schema"]["$ref"].endswith("/ValidationResponse")
    assert spec["info"]["version"] == __version__
    assert "schema_version" in schemas["ValidationResponse"]["properties"]


def test_version_is_single_source():
    import tomllib
    from pathlib import Path

    text = (Path(__file__).resolve().parent.parent / "pyproject.toml").read_text()
    cfg = tomllib.loads(text)
    assert "version" in cfg["project"]["dynamic"]
    assert cfg["tool"]["setuptools"]["dynamic"]["version"]["attr"] == "app.__version__"
