"""Pydantic request/response models. Mirrors docs/API_CONTRACT.md (issue #2)."""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

TRIP_ID_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [{"trip_id": "00000000"}]})
    trip_id: str = Field(..., pattern=TRIP_ID_PATTERN, description="Trip folder name, e.g. 00000000")


class GeoJSONFeature(BaseModel):
    type: Literal["Feature"] = "Feature"
    properties: dict = Field(default_factory=dict)
    geometry: dict


class Metrics(BaseModel):
    """Server-side scoring against the reference .route (computed after prediction)."""
    model_config = ConfigDict(extra="allow")
    top1_accuracy: Optional[float] = None
    classifier_only_top1: Optional[float] = None
    route_precision: Optional[float] = None
    route_recall: Optional[float] = None
    route_f1: Optional[float] = None
    route_continuity: Optional[float] = None
    n_points_aligned: Optional[int] = None
    reference_route_geojson: Optional[GeoJSONFeature] = None


class PredictResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    mode: Literal["real", "mock"] = Field(..., description="'mock' responses are NOT ML predictions")
    trip_id: str
    n_points: int
    gps_points: list[list[float]]
    per_point_arc: list[Optional[int]]
    per_point_prob: list[Optional[float]]
    predicted_arcs: list[int]
    predicted_route_geojson: GeoJSONFeature
    metadata: dict[str, Any]
    timings_ms: dict[str, float]
    metrics: Optional[Metrics] = None
    metrics_error: Optional[str] = None
    api_latency_ms: float


class TripsResponse(BaseModel):
    trips: list[str]
    count: int
    trips_dir: str


class HealthResponse(BaseModel):
    service: Literal["pathsnap-api"] = Field("pathsnap-api", description="Fixed identity so clients can tell this API apart from other services on the same port")
    status: Literal["ok", "degraded"]
    predictor_mode: Literal["real", "mock", "unavailable"]
    model_loaded: bool
    model_name: Optional[str] = None
    artifact_version: Optional[str] = None
    predictor: str
    predictor_error: Optional[str] = None
    trips_dir: str
    trips_dir_exists: bool
    n_trips: int


class ErrorResponse(BaseModel):
    error: str
    detail: str
