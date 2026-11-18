"""API <-> UI contract for GET /health (KeyError 'predictor_mode' regression).

Run: python -m pytest tests/test_health_contract.py -q
"""
import os

import pytest
from fastapi.testclient import TestClient

from api.config import Settings
from api.main import create_app
from ui.health import DEFAULTS, MODES, SERVICE_ID, HealthContractError, parse_health


def _app(trips_dir, **kw):
    s = Settings(trips_dir=str(trips_dir), predictor_path="tests.test_api:FakePredictor", warmup=False,
                 log_file=os.path.join(str(trips_dir), "req.log"), **kw)
    return TestClient(create_app(s))


@pytest.fixture
def trips_dir(tmp_path):
    d = tmp_path / "00000001"
    d.mkdir()
    (d / "00000001.track").write_text("8.6\t49.8\t1\n")
    return tmp_path


@pytest.mark.parametrize("kw,mode", [({}, "real"), ({"mock": True}, "mock"), ({"artifacts_dir": "missing"}, "unavailable")])
def test_api_health_satisfies_ui_contract(trips_dir, kw, mode):
    """Every field the UI reads is present in the real /health response, for all three modes."""
    with _app(trips_dir, **kw) as c:
        body = c.get("/health").json()
    assert body["service"] == SERVICE_ID and body["predictor_mode"] == mode
    assert set(DEFAULTS) <= set(body)
    h = parse_health(body, "http://x")
    assert h["predictor_mode"] == mode and h["n_trips"] == 1


def test_no_model_error_is_shown_not_hidden(trips_dir):
    with _app(trips_dir, artifacts_dir="missing") as c:
        h = parse_health(c.get("/health").json())
    assert h["predictor_mode"] == "unavailable" and "FileNotFoundError" in h["predictor_error"]


def test_foreign_service_on_same_port_is_reported_clearly():
    # what a different server (e.g. a Docker container on 127.0.0.1:8000) answered in the bug report
    with pytest.raises(HealthContractError, match="not the PathSnap API.*fields: status"):
        parse_health({"status": "ok"}, "http://127.0.0.1:8000")


@pytest.mark.parametrize("body", [None, [], "ok"])
def test_non_object_body_is_rejected(body):
    with pytest.raises(HealthContractError):
        parse_health(body)


def test_unknown_mode_is_an_error_not_a_guess():
    with pytest.raises(HealthContractError, match="unknown predictor_mode"):
        parse_health({"service": SERVICE_ID, "predictor_mode": "turbo"})


def test_missing_optional_fields_get_safe_defaults():
    h = parse_health({"predictor_mode": "real", "model_name": "xgb", "n_trips": None})
    assert h["model_name"] == "xgb" and h["n_trips"] == 0 and h["artifact_version"] is None
    assert h["predictor_error"] is None and h["status"] == "degraded"
    assert set(MODES) == {"real", "mock", "unavailable"}


# ---------------- Streamlit UI handling (headless, streamlit.testing AppTest) ----------------
class _Resp:
    def __init__(self, body, code=200):
        self._b, self.status_code = body, code

    def json(self):
        return self._b


def _run_ui(monkeypatch, health_body):
    import requests
    import streamlit as st
    from streamlit.testing.v1 import AppTest

    st.cache_data.clear()  # the UI caches /health per URL; each test needs a fresh body

    def fake_request(method, url, **kw):
        if url.endswith("/health"):
            return _Resp(health_body)
        if url.endswith("/trips"):
            return _Resp({"trips": ["00000001"], "count": 1, "trips_dir": "x"})
        raise AssertionError(url)

    monkeypatch.setattr(requests, "request", fake_request)
    at = AppTest.from_file(os.path.join(os.path.dirname(__file__), "..", "ui", "app.py"), default_timeout=30)
    at.run()
    assert not at.exception, at.exception
    return at


def _texts(elements):
    return " ".join(str(e.value) for e in elements)


def test_ui_does_not_crash_on_foreign_health_body(monkeypatch):
    at = _run_ui(monkeypatch, {"status": "ok"})
    assert "not the PathSnap API" in _texts(at.sidebar.error)


@pytest.mark.parametrize("mode,widget,text", [("real", "success", "Connected · model **xgb**"),
                                               ("mock", "warning", "MOCK MODE"),
                                               ("unavailable", "error", "no model is loaded")])
def test_ui_shows_each_predictor_mode(monkeypatch, mode, widget, text):
    body = {"service": SERVICE_ID, "status": "ok", "predictor_mode": mode, "model_name": "xgb" if mode == "real" else None,
            "predictor_error": "FileNotFoundError: x" if mode == "unavailable" else None, "n_trips": 1}
    at = _run_ui(monkeypatch, body)
    assert text in _texts(getattr(at.sidebar, widget))
