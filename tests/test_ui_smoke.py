"""Render the Streamlit results view from real reports and check nothing raises."""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from app.report import build_response, generate_trust_report  # noqa: E402
from tests.conftest import make_dataset  # noqa: E402

UI = str(Path(__file__).resolve().parent.parent / "ui.py")


def render(real, synth):
    response = build_response(generate_trust_report(real, synth, "label"))
    at = AppTest.from_file(UI, default_timeout=60)
    at.session_state["result"] = {
        "response": response,
        "real": real,
        "synth": synth,
        "label": "label",
    }
    at.run()
    return at


def test_empty_page_renders():
    at = AppTest.from_file(UI, default_timeout=60).run()
    assert not at.exception


def test_clean_report_view_and_column_selection():
    at = render(make_dataset(300, 1), make_dataset(300, 2))
    assert not at.exception
    assert [s.value for s in at.subheader][:2] == ["Warnings", "Verdict by dimension"]
    for column in ("age", "city", "label"):
        at.selectbox(key="plot_column").set_value(column).run()
        assert not at.exception


def test_issues_and_tiny_data_views():
    real = make_dataset(300, 1)
    at = render(real, real.copy())
    assert not at.exception
    assert at.error, "an exact copy must show the issues banner"
    assert not render(make_dataset(8, 1), make_dataset(8, 2)).exception
