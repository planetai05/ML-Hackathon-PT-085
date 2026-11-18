"""API tests (issue #24). Run: python -m pytest tests/test_api.py -q

Uses a tiny synthetic trips folder and a FakePredictor loaded through PATHSNAP_PREDICTOR,
which also proves the API works with any Predictor that follows docs/API_CONTRACT.md.
An optional test runs the real predictor when artifacts + dataset are present.
"""
import os

import pytest
from fastapi.testclient import TestClient

from api.config import Settings
from api.main import create_app


class FakePredictor:
    """Contract-compliant stand-in (NOT an ML model)."""

    def __init__(self, artifacts_dir="artifacts", mock=False):
        if artifacts_dir == "missing":
            raise FileNotFoundError("artifacts/model_meta.json")
        self.model_name, self.version = "fake", "test"

    def predict_trip(self, trip_dir):
        tid = os.path.basename(trip_dir)
        if tid == "boom":
            raise RuntimeError("model exploded")
        return {"trip_id": tid, "n_points": 2, "gps_points": [[8.6, 49.8, 1.0], [8.61, 49.81, 2.0]],
                "per_point_arc": [5, None], "per_point_prob": [0.9, None], "predicted_arcs": [5],
                "predicted_route_geojson": {"type": "Feature", "properties": {}, "geometry": {"type": "MultiLineString", "coordinates": [[[8.6, 49.8], [8.61, 49.81]]]}},
                "metadata": {"model_name": "fake", "decoder": "test"}, "timings_ms": {"total": 1.0}, "_internal": [1]}

    def score(self, pred, trip_dir):
        if not os.path.exists(os.path.join(trip_dir, f"{pred['trip_id']}.route")):
            raise FileNotFoundError("no .route")
        return {"top1_accuracy": 1.0, "classifier_only_top1": 0.5, "route_f1": 1.0, "n_points_aligned": 1}


@pytest.fixture
def trips_dir(tmp_path):
    for tid, route in (("00000001", True), ("00000002", False), ("boom", True)):
        d = tmp_path / tid
        d.mkdir()
        (d / f"{tid}.track").write_text("8.6\t49.8\t1\n")
        if route:
            (d / f"{tid}.route").write_text("5\n")
    (tmp_path / "not_a_trip").mkdir()
    return str(tmp_path)


def client(trips_dir, **kw):
    s = Settings(trips_dir=trips_dir, predictor_path="tests.test_api:FakePredictor", warmup=False,
                 log_file=os.path.join(trips_dir, "req.log"), **kw)
    return TestClient(create_app(s))


def test_health_and_trips(trips_dir):
    with client(trips_dir) as c:
        h = c.get("/health").json()
        assert h["predictor_mode"] == "real" and h["model_loaded"] and h["n_trips"] == 3 and h["status"] == "ok"
        t = c.get("/trips").json()
        assert t["trips"] == ["00000001", "00000002", "boom"] and t["count"] == 3


def test_predict_ok_strips_internal_fields(trips_dir):
    with client(trips_dir) as c:
        r = c.post("/predict", json={"trip_id": "00000001"})
        assert r.status_code == 200, r.text
        j = r.json()
        assert j["mode"] == "real" and j["predicted_arcs"] == [5] and j["metrics"]["top1_accuracy"] == 1.0
        assert "_internal" not in j and j["api_latency_ms"] >= 0


def test_predict_without_route_returns_prediction_and_metrics_error(trips_dir):
    with client(trips_dir) as c:
        j = c.post("/predict", json={"trip_id": "00000002"}).json()
        assert j["metrics"] is None and "no .route" in j["metrics_error"]


@pytest.mark.parametrize("body,code", [({"trip_id": "99999999"}, 404), ({"trip_id": "../etc"}, 422),
                                       ({}, 422), ({"trip_id": "00000001", "x": 1}, 422)])
def test_predict_bad_input(trips_dir, body, code):
    with client(trips_dir) as c:
        r = c.post("/predict", json=body)
        assert r.status_code == code and "error" in r.json()


def test_prediction_failure_is_500(trips_dir):
    with client(trips_dir) as c:
        r = c.post("/predict", json={"trip_id": "boom"})
        assert r.status_code == 500 and r.json()["error"] == "prediction_failed"


def test_missing_model_is_503_and_health_degraded(trips_dir):
    with client(trips_dir, artifacts_dir="missing") as c:
        assert c.get("/health").json()["predictor_mode"] == "unavailable"
        r = c.post("/predict", json={"trip_id": "00000001"})
        assert r.status_code == 503 and r.json()["error"] == "predictor_unavailable"


def test_missing_model_with_mock_fallback_is_labelled(trips_dir):
    s = Settings(trips_dir=trips_dir, predictor_path="models.predictor:Predictor", artifacts_dir="missing_dir",
                 allow_mock_fallback=True, warmup=False, log_file=os.path.join(trips_dir, "req.log"))
    with TestClient(create_app(s)) as c:
        h = c.get("/health").json()
        if h["predictor_mode"] == "real":
            pytest.skip("this Predictor version does not need artifacts at construction")
        assert h["predictor_mode"] == "mock"
        assert c.post("/predict", json={"trip_id": "00000001"}).json()["mode"] == "mock"


def test_missing_trips_dir_is_503(tmp_path):
    with client(str(tmp_path / "nope")) as c:
        r = c.get("/trips")
        assert r.status_code == 503 and r.json()["error"] == "trips_dir_missing"


REAL = os.path.isfile(os.path.join("artifacts", "model_meta.json")) and os.path.isdir(Settings().trips_dir)


@pytest.mark.skipif(not REAL, reason="real artifacts or dataset not present")
def test_real_predictor_end_to_end():
    with TestClient(create_app(Settings(warmup=False))) as c:
        if c.get("/health").json()["predictor_mode"] != "real":
            pytest.skip("installed Predictor has no trained model on this branch")
        tid = c.get("/trips").json()["trips"][0]
        j = c.post("/predict", json={"trip_id": tid}).json()
        assert j["mode"] == "real" and len(j["predicted_arcs"]) > 0 and j["metrics"]["top1_accuracy"] is not None


def test_predict_with_missing_trips_dir_is_503(tmp_path):
    with client(str(tmp_path / "nope")) as c:
        r = c.post("/predict", json={"trip_id": "00000001"})
        assert r.status_code == 503 and r.json()["error"] == "trips_dir_missing"


def test_swagger_and_openapi_document_all_endpoints(trips_dir):
    with client(trips_dir) as c:
        assert c.get("/docs").status_code == 200
        assert c.get("/", follow_redirects=False).headers["location"] == "/docs"
        spec = c.get("/openapi.json").json()
        assert {"/health", "/trips", "/predict"} <= set(spec["paths"])
        assert {"PredictRequest", "PredictResponse", "HealthResponse", "ErrorResponse"} <= set(spec["components"]["schemas"])


def test_e2e_checker_reports_unreachable_api_clearly():
    import subprocess
    import sys
    r = subprocess.run([sys.executable, "scripts/e2e_check.py", "--api", "http://127.0.0.1:9", "--out", os.devnull],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode != 0 and "Cannot reach the API at http://127.0.0.1:9" in r.stderr and "Traceback" not in r.stderr
