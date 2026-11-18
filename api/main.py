"""PathSnap FastAPI backend (Laptop 1).

Run:  uvicorn api.main:app --host 0.0.0.0 --port 8000
Docs: http://<host>:8000/docs

The API loads a Predictor implementing docs/API_CONTRACT.md via PATHSNAP_PREDICTOR
("module:ClassName", default models.predictor:Predictor) so the ML owner can swap the selected
model / implementation without any API or UI change. `.route` reference files never leave the
server except as the reference geometry returned AFTER prediction for display.
"""
from __future__ import annotations

import importlib
import logging
import os
import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse

from api.config import Settings
from api.schemas import ErrorResponse, HealthResponse, PredictRequest, PredictResponse, TripsResponse

log = logging.getLogger("pathsnap.api")


def _setup_logging(path: str):
    if log.handlers:
        return
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    log.addHandler(sh)
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fh = logging.FileHandler(path, encoding="utf-8")
        fh.setFormatter(fmt)
        log.addHandler(fh)
    except OSError as e:  # logging to file is best-effort
        log.warning("cannot open log file %s: %s", path, e)


class PredictorState:
    """Holds the loaded predictor and its mode ('real' | 'mock' | 'unavailable')."""

    def __init__(self, settings: Settings):
        self.s = settings
        self.predictor = None
        self.mode = "unavailable"
        self.error: str | None = None
        self.lock = threading.Lock()  # one prediction at a time (CPU-bound, shared caches)

    def _cls(self):
        mod, _, name = self.s.predictor_path.partition(":")
        return getattr(importlib.import_module(mod), name or "Predictor")

    def load(self):
        if self.s.mock:
            self.predictor, self.mode = self._cls()(artifacts_dir=self.s.artifacts_dir, mock=True), "mock"
            log.warning("PREDICTOR MODE = MOCK (PATHSNAP_MOCK=1): responses are NOT ML predictions")
            return
        try:
            pred = self._cls()(artifacts_dir=self.s.artifacts_dir)
            if not getattr(pred, "model_name", None):
                raise RuntimeError("predictor loaded but reports no model (model_name is empty) - is it the #2 stub?")
            self.predictor, self.mode, self.error = pred, "real", None
            log.info("predictor loaded: %s (%s)", self.s.predictor_path, self.model_name)
        except Exception as e:  # missing artifacts, import errors, ...
            self.error = f"{type(e).__name__}: {e}"
            log.error("predictor failed to load: %s", self.error)
            if self.s.allow_mock_fallback:
                self.predictor, self.mode = self._cls()(artifacts_dir=self.s.artifacts_dir, mock=True), "mock"
                log.warning("falling back to MOCK predictor (PATHSNAP_ALLOW_MOCK_FALLBACK=1)")
            else:
                self.predictor, self.mode = None, "unavailable"

    @property
    def model_name(self):
        return getattr(self.predictor, "model_name", None)

    @property
    def version(self):
        return getattr(self.predictor, "version", None)


def list_trip_ids(trips_dir: str) -> list[str]:
    if not os.path.isdir(trips_dir):
        return []
    out = []
    for name in sorted(os.listdir(trips_dir)):
        if os.path.isfile(os.path.join(trips_dir, name, f"{name}.track")):
            out.append(name)
    return out


def create_app(settings: Settings | None = None) -> FastAPI:
    s = settings or Settings()
    _setup_logging(s.log_file)
    state = PredictorState(s)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        state.load()
        log.info("trips_dir=%s (%d trips)", os.path.abspath(s.trips_dir), len(list_trip_ids(s.trips_dir)))
        if state.mode == "real" and s.warmup and hasattr(state.predictor, "warmup"):
            dirs = [os.path.join(s.trips_dir, t) for t in list_trip_ids(s.trips_dir)]
            threading.Thread(target=_warm, args=(state, dirs), daemon=True).start()
        yield

    app = FastAPI(title="PathSnap API", version="1.0",
                  description="Map-matching of noisy GPS trips: candidates → classifier → HMM/Viterbi. "
                              "Contract: docs/API_CONTRACT.md", lifespan=lifespan)
    app.state.pathsnap = state
    app.add_middleware(CORSMiddleware, allow_origins=s.cors_origins, allow_methods=["*"], allow_headers=["*"])

    @app.middleware("http")
    async def access_log(request: Request, call_next):
        t0 = time.perf_counter()
        response = await call_next(request)
        client = request.client.host if request.client else "-"
        log.info("%s %s %s -> %d (%.0f ms)", client, request.method, request.url.path, response.status_code,
                 (time.perf_counter() - t0) * 1000)
        return response

    @app.exception_handler(HTTPException)
    async def http_error(_, exc: HTTPException):
        return JSONResponse(status_code=exc.status_code, content=exc.detail if isinstance(exc.detail, dict)
                            else {"error": "http_error", "detail": str(exc.detail)})

    @app.exception_handler(RequestValidationError)
    async def validation_error(_, exc: RequestValidationError):
        msgs = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
        return JSONResponse(status_code=422, content={"error": "validation_error", "detail": msgs})

    def _err(code: int, error: str, detail: str):
        raise HTTPException(status_code=code, detail={"error": error, "detail": detail})

    @app.get("/", include_in_schema=False)
    def root():
        return RedirectResponse("/docs")

    @app.get("/health", response_model=HealthResponse, tags=["service"])
    def health():
        trips = list_trip_ids(s.trips_dir)
        return HealthResponse(status="ok" if state.mode == "real" and trips else "degraded", predictor_mode=state.mode,
                              model_loaded=state.mode == "real", model_name=state.model_name, artifact_version=state.version,
                              predictor=s.predictor_path, predictor_error=state.error, trips_dir=os.path.abspath(s.trips_dir),
                              trips_dir_exists=os.path.isdir(s.trips_dir), n_trips=len(trips))

    @app.get("/trips", response_model=TripsResponse, responses={503: {"model": ErrorResponse}}, tags=["trips"])
    def trips():
        if not os.path.isdir(s.trips_dir):
            _err(503, "trips_dir_missing", f"Trips folder not found: {os.path.abspath(s.trips_dir)} (set PATHSNAP_TRIPS_DIR)")
        ids = list_trip_ids(s.trips_dir)
        return TripsResponse(trips=ids, count=len(ids), trips_dir=os.path.abspath(s.trips_dir))

    @app.post("/predict", response_model=PredictResponse, response_model_exclude_none=False,
              responses={404: {"model": ErrorResponse}, 422: {"model": ErrorResponse},
                         500: {"model": ErrorResponse}, 503: {"model": ErrorResponse}}, tags=["prediction"])
    def predict(req: PredictRequest, request: Request):
        t0 = time.perf_counter()
        if state.predictor is None:
            _err(503, "predictor_unavailable", f"No predictor loaded ({state.error or 'unknown error'}). "
                 "Export artifacts or start with PATHSNAP_MOCK=1 for UI development.")
        if req.trip_id not in list_trip_ids(s.trips_dir):
            _err(404, "unknown_trip", f"Trip '{req.trip_id}' not found in {os.path.abspath(s.trips_dir)}")
        trip_dir = os.path.join(s.trips_dir, req.trip_id)
        with state.lock:
            try:
                pred = state.predictor.predict_trip(trip_dir)
            except Exception as e:
                log.exception("prediction failed for %s", req.trip_id)
                _err(500, "prediction_failed", f"{type(e).__name__}: {e}")
            metrics, metrics_error = None, None
            try:  # scoring needs the server-side .route; prediction is already done at this point
                metrics = state.predictor.score(pred, trip_dir)
            except Exception as e:
                metrics_error = f"{type(e).__name__}: {e}"
                log.warning("scoring failed for %s: %s", req.trip_id, metrics_error)
        pred = {k: v for k, v in pred.items() if not k.startswith("_")}
        latency = (time.perf_counter() - t0) * 1000
        m = metrics or {}
        log.info("PREDICT trip=%s mode=%s model=%s points=%s top1=%s clf_top1=%s route_f1=%s latency=%.0fms client=%s",
                 req.trip_id, state.mode, state.model_name, pred.get("n_points"), m.get("top1_accuracy"),
                 m.get("classifier_only_top1"), m.get("route_f1"), latency, request.client.host if request.client else "-")
        return PredictResponse(mode=state.mode, metrics=metrics, metrics_error=metrics_error, api_latency_ms=round(latency, 1), **pred)

    return app


def _warm(state: PredictorState, dirs):
    t0 = time.perf_counter()
    try:
        with state.lock:
            state.predictor.warmup(dirs)
        log.info("warmup done for %d trips in %.1fs", len(dirs), time.perf_counter() - t0)
    except Exception as e:
        log.warning("warmup failed: %s", e)


app = create_app()
