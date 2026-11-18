# PathSnap API and UI (Person C)

```
Laptop 2                         Laptop 1
Streamlit UI  --HTTP/JSON-->  FastAPI (api/main.py) --> Predictor (docs/API_CONTRACT.md) --> artifacts/
(ui/app.py)                   GET /health, /trips        predict_trip(trip_dir)  (.track/.nodes/.arcs only)
                              POST /predict              score(pred, trip_dir)   (.route, server-side, after prediction)
```

The UI never imports ML code, models or `.route` files. It only calls the API.
The API never imports a specific model. It loads whatever Predictor class `PATHSNAP_PREDICTOR` names, so the ML owner can change the selected model (or replace the whole implementation) without changing the API or UI.

## 1. Install
```bash
pip install -r requirements.txt
pip install streamlit streamlit-folium requests   # UI laptop
```

## 2. Run (two laptops on the same Wi-Fi/hotspot)
**Laptop 1 (API + model):**
```bat
scripts\run_api.bat
```
or `python -m uvicorn api.main:app --host 0.0.0.0 --port 8000`.
- Find the IP with `ipconfig` (IPv4 address), e.g. `192.168.1.20`.
- Allow the port once, from an **admin** prompt: `netsh advfirewall firewall add rule name="PathSnap API" dir=in action=allow protocol=TCP localport=8000`
- Check it from Laptop 2's browser: `http://192.168.1.20:8000/health`. Swagger docs are at `/docs`.

**Laptop 2 (UI):**
```bat
scripts\run_ui.bat http://192.168.1.20:8000
```
Then open `http://localhost:8501`. You can also change the API URL in the sidebar at any time.

On a single machine, use `http://127.0.0.1:8000`. On Windows, `localhost` first tries IPv6 and adds about 2 s to every request.

## 3. Configuration (environment variables, API laptop)
| Variable | Default | Meaning |
|---|---|---|
| `PATHSNAP_TRIPS_DIR` | `data/Student folder/student_test` | Folder with the evaluation trips: `<dir>/<trip_id>/<trip_id>.{track,nodes,arcs,route}`. Point it at the evaluator trips on demo day. |
| `PATHSNAP_PREDICTOR` | `models.predictor:Predictor` | Predictor class (`module:Class`) implementing `docs/API_CONTRACT.md` |
| `PATHSNAP_ARTIFACTS_DIR` | `artifacts` | Passed to `Predictor(artifacts_dir=...)` |
| `PATHSNAP_MOCK` | `0` | `1` = development mock (responses carry `"mode": "mock"`, and the UI shows a warning banner) |
| `PATHSNAP_ALLOW_MOCK_FALLBACK` | `0` | `1` = use mock if the real predictor fails to load. Keep `0` for the demo. |
| `PATHSNAP_WARMUP` | `1` | Pre-build spatial index and graph for all trips at startup (`Predictor.warmup`, if present) |
| `PATHSNAP_CORS_ORIGINS` | `*` | Comma-separated allowed origins |
| `PATHSNAP_LOG_FILE` | `api/logs/requests.log` | Request log (also printed to the console for the CMD screenshots) |
| `PATHSNAP_API_URL` (UI) | `http://127.0.0.1:8000` | Default API URL shown in the UI sidebar |

## 4. Endpoints
### `GET /health`
```json
{"status": "ok", "predictor_mode": "real", "model_loaded": true, "model_name": "lgbm",
 "artifact_version": "v20261118-0630", "predictor": "models.predictor:Predictor", "predictor_error": null,
 "trips_dir": "...\\student_test", "trips_dir_exists": true, "n_trips": 10}
```
`predictor_mode` is `real`, `mock` or `unavailable`, and `status` is `degraded` unless a real model and trips are present. A predictor that loads but reports no `model_name`, such as the #2 stub, counts as `unavailable`.

### `GET /trips`
`{"trips": ["00000000", ...], "count": 10, "trips_dir": "..."}`. A folder is a trip if it contains `<id>.track`. Returns **503** `trips_dir_missing` if the folder does not exist.

### `POST /predict`
Request: `{"trip_id": "00000000"}`. Pydantic rules: `trip_id` must match `^[A-Za-z0-9_-]{1,64}$`, and no extra fields are allowed.
Response: every `predict_trip` field from `docs/API_CONTRACT.md`, plus:
- `mode`: `real` or `mock`
- `metrics`: the output of `score()`, with `reference_route_geojson` returned only after prediction
- `metrics_error`: why there are no metrics, if scoring failed (for example, no `.route`)
- `api_latency_ms`

Keys that start with `_` in the predictor output are internal and are removed.

| Status | `error` | When |
|---|---|---|
| 404 | `unknown_trip` | `trip_id` not in `/trips` |
| 422 | `validation_error` | Bad or missing `trip_id`, or extra fields |
| 500 | `prediction_failed` | The predictor raised an exception (message included) |
| 503 | `predictor_unavailable` | No model loaded (see `/health` → `predictor_error`) |

All errors use the shape `{"error": "...", "detail": "..."}`.

Every request is logged as `client METHOD path -> status (ms)`. Every prediction also logs `PREDICT trip=... mode=... model=... top1=... route_f1=... latency=...`.

## 5. UI features (`ui/app.py`)
- Sidebar: API URL, Reconnect, connection and model status (a green real model, a yellow **MOCK MODE** warning, or a red "no model" error), and map-layer toggles.
- Trip dropdown from `/trips`, a **Predict route** button and a spinner while the pipeline runs.
- Six metric tiles: HMM point accuracy (with the gain over classifier-only), classifier-only accuracy, route F1, route continuity, GPS points and API latency.
- Folium map: GPS points (blue), predicted route (red), reference route (green) after scoring, start and end markers, a layer control and a scale.
- Prediction details table (model, decoder, arcs, segments, candidates, artifact version, route precision and recall), per-stage timings, and the raw JSON in an expander.
- Clear messages for an unreachable API, timeouts, HTTP errors, a missing trip selection and missing metrics.

## 6. For the ML owner: plugging in the selected model
1. Implement (or keep) a class with `__init__(artifacts_dir, mock=False)`, `predict_trip(trip_dir) -> dict` and `score(pred, trip_dir) -> dict`, following `docs/API_CONTRACT.md`. Set a non-empty `model_name` attribute and optionally `version` and `warmup(trip_dirs)`. The constructor should **raise** if artifacts are missing.
2. Put the artifacts in `artifacts/`, or point `PATHSNAP_ARTIFACTS_DIR` at them.
3. Restart the API. If the class lives elsewhere, set `PATHSNAP_PREDICTOR=yourpkg.module:YourPredictor`.
4. Verify with `python scripts/e2e_check.py --api http://127.0.0.1:8000`, which runs every trip through the HTTP API and writes a CSV.

The UI and API need no change when the selected model changes.

## 7. Tests
```bash
python -m pytest tests/test_api.py -q
```
Twelve tests use a synthetic trip folder and a contract-compliant `FakePredictor`, which proves the predictor is swappable. They cover health, trips, success, missing `.route`, 404, 422 (bad id, path traversal, empty body, extra field), 500, 503, labelled mock fallback and a missing trips folder. One more test runs the real predictor when artifacts and data are present.

## 8. Troubleshooting
| Symptom | Fix |
|---|---|
| UI: "Cannot reach the API" | Wrong IP or port, API not started with `--host 0.0.0.0`, or the firewall rule is missing. Test `/health` in Laptop 2's browser. |
| `/health` shows `predictor_mode: unavailable` | Read `predictor_error`: usually missing `artifacts/model_meta.json`, or `feature_config.json` doesn't match the code. Export the artifacts. |
| `/trips` returns 503 | Set `PATHSNAP_TRIPS_DIR` to the folder that contains the trip folders. |
| Every request about 2 s slower than `api_latency_ms` | Use `127.0.0.1`, not `localhost`, on Windows. |
| First prediction is slow | Warm-up builds the spatial index and graph in the background at startup. Wait for the `warmup done` log line. |
