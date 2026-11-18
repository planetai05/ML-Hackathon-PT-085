"""Tests for the /evaluate-zip background job endpoints (Two-Laptop Evaluation System)."""
import os
import time
import pytest
from fastapi.testclient import TestClient

from api.config import Settings
from api.main import create_app

EVAL_ZIP = os.path.join("data", "uploads", "evaluator_test.zip")
ZIP_EXISTS = os.path.isfile(EVAL_ZIP)


@pytest.fixture
def api_client():
    s = Settings(warmup=False)
    with TestClient(create_app(s)) as client:
        yield client


@pytest.mark.skipif(not ZIP_EXISTS, reason="data/uploads/evaluator_test.zip not found")
def test_evaluate_zip_info(api_client):
    r = api_client.get("/evaluate-zip/info")
    assert r.status_code == 200
    data = r.json()
    assert data["exists"] is True
    assert data["total_trips"] == 10
    assert len(data["trips"]) == 10
    assert "00000015" in data["trips"]
    assert "execution_device" in data
    assert "CPU" in data["execution_device"] or "GPU" in data["execution_device"]


def test_evaluate_zip_missing_zip():
    s = Settings(evaluator_zip="data/uploads/missing.zip", warmup=False)
    with TestClient(create_app(s)) as client:
        r = client.post("/evaluate-zip")
        assert r.status_code == 404
        assert r.json()["error"] == "zip_not_found"


def test_evaluate_zip_job_not_found(api_client):
    r = api_client.get("/evaluate-zip/nonexistent_job")
    assert r.status_code == 404
    assert r.json()["error"] == "job_not_found"

    r_res = api_client.get("/evaluate-zip/nonexistent_job/results")
    assert r_res.status_code == 404
    assert r_res.json()["error"] == "job_not_found"

    r_down = api_client.get("/evaluate-zip/nonexistent_job/download")
    assert r_down.status_code == 404
    assert r_down.json()["error"] == "csv_not_found"


@pytest.mark.skipif(not ZIP_EXISTS, reason="data/uploads/evaluator_test.zip not found")
def test_evaluate_zip_lifecycle_and_results(api_client):
    # 1. Start job
    r_start = api_client.post("/evaluate-zip")
    assert r_start.status_code == 200
    job_id = r_start.json()["job_id"]
    assert job_id is not None
    assert r_start.json()["status"] in ("queued", "running")

    # 2. Duplicate start while running should return 409 Conflict
    r_dup = api_client.post("/evaluate-zip")
    # If the job finishes super fast or is still running
    if r_dup.status_code == 409:
        assert r_dup.json()["error"] == "job_already_running"

    # 3. Poll until completed (timeout 120s)
    t0 = time.time()
    completed = False
    while time.time() - t0 < 120:
        r_poll = api_client.get(f"/evaluate-zip/{job_id}")
        assert r_poll.status_code == 200
        poll_data = r_poll.json()
        if poll_data["status"] == "completed":
            completed = True
            break
        elif poll_data["status"] == "failed":
            pytest.fail(f"Job failed: {poll_data['error']}")
        time.sleep(1.0)

    assert completed, f"Job {job_id} timed out before completing"

    # 4. Fetch results
    r_res = api_client.get(f"/evaluate-zip/{job_id}/results")
    assert r_res.status_code == 200
    res = r_res.json()
    assert res["status"] == "completed"
    assert res["total_trips"] == 10
    assert res["successful_trips"] == 10
    assert res["failed_trips"] == 0
    assert res["mean_viterbi_top1"] is not None
    assert res["mean_route_continuity"] is not None
    assert res["mean_route_f1"] is not None
    assert len(res["per_trip"]) == 10

    # 5. Download CSV
    r_down = api_client.get(f"/evaluate-zip/{job_id}/download")
    assert r_down.status_code == 200
    assert "text/csv" in r_down.headers["content-type"]
    csv_text = r_down.text
    assert "trip_id,n_points,classifier_only_top1" in csv_text
    assert "MEAN" in csv_text
    assert "00000015" in csv_text
