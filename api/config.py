"""API configuration, read from environment variables (see docs/API_UI.md)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    return default if v is None else v.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class Settings:
    # Folder holding the evaluation trips: <TRIPS_DIR>/<trip_id>/<trip_id>.{track,nodes,arcs,route}
    trips_dir: str = field(default_factory=lambda: os.environ.get(
        "PATHSNAP_TRIPS_DIR", os.path.join("data", "Student folder", "student_test")))
    # Predictor class implementing docs/API_CONTRACT.md, as "module:ClassName"
    predictor_path: str = field(default_factory=lambda: os.environ.get("PATHSNAP_PREDICTOR", "models.predictor:Predictor"))
    artifacts_dir: str = field(default_factory=lambda: os.environ.get("PATHSNAP_ARTIFACTS_DIR", "artifacts"))
    # mock = development only; responses are labelled "mode": "mock" and never presented as ML predictions
    mock: bool = field(default_factory=lambda: _bool("PATHSNAP_MOCK", False))
    # fall back to mock if the real predictor cannot be loaded (off by default for the demo)
    allow_mock_fallback: bool = field(default_factory=lambda: _bool("PATHSNAP_ALLOW_MOCK_FALLBACK", False))
    warmup: bool = field(default_factory=lambda: _bool("PATHSNAP_WARMUP", True))
    evaluator_zip: str = field(default_factory=lambda: os.environ.get(
        "PATHSNAP_EVALUATOR_ZIP", os.path.join("data", "uploads", "evaluator_test.zip")))
    cors_origins: list = field(default_factory=lambda: os.environ.get("PATHSNAP_CORS_ORIGINS", "*").split(","))
    log_file: str = field(default_factory=lambda: os.environ.get("PATHSNAP_LOG_FILE", os.path.join("api", "logs", "requests.log")))
