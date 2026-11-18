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

## 2. Run (two Windows laptops on the same Wi-Fi or phone hotspot)
All commands are for **PowerShell**, run from the repo root. Both scripts take an optional port, so you can avoid ports that other programs already use.

### Laptop 1: API + model
1. Find the Wi-Fi address. Run `ipconfig` and use the IPv4 address under **Wireless LAN adapter Wi-Fi** (for example `10.218.43.14`). Ignore `192.168.56.x` (VirtualBox) and `172.x` (WSL/Hyper-V) adapters; the other laptop cannot reach those.
2. Check the port is free. If this prints anything, pick another port (Docker Desktop, for example, can hold `127.0.0.1:8000`):
   ```powershell
   Get-NetTCPConnection -State Listen -LocalPort 8000 -ErrorAction SilentlyContinue
   ```
3. Start the API on all network interfaces:
   ```powershell
   .\scripts\run_api.bat 8000
   ```
   Expect `predictor loaded: models.predictor:Predictor (xgb)`, `trips_dir=... (10 trips)` and `Uvicorn running on http://0.0.0.0:8000`.
4. Allow inbound connections (once). If Windows asks whether Python may communicate on the network, tick **both** Private and Public. The Wi-Fi profile is often *Public*. Or, from an **administrator** PowerShell:
   ```powershell
   New-NetFirewallRule -DisplayName "PathSnap API 8000" -Direction Inbound -Action Allow -Protocol TCP -LocalPort 8000 -Profile Any
   ```
5. Optional, to use the demo trips folder:
   ```powershell
   $env:PATHSNAP_TRIPS_DIR = "D:\eval\trips"; .\scripts\run_api.bat 8000
   ```

### Laptop 2: UI
1. Open `http://<laptop1-ip>:8000/health` in the browser. It must show `"service":"pathsnap-api"` and `"predictor_mode":"real"`. If it doesn't load, fix the network or firewall first.
2. Start the UI (second argument = UI port, default 8501):
   ```powershell
   .\scripts\run_ui.bat http://<laptop1-ip>:8000 8501
   ```
3. Open `http://localhost:8501`. The sidebar must say **Connected · model xgb**. You can change the API URL in the sidebar at any time, then click **Reconnect**.
4. Optional full check from Laptop 2:
   ```powershell
   python scripts\e2e_check.py --api http://<laptop1-ip>:8000 --out results\e2e_two_laptop.csv
   ```

**Single machine:** use `http://127.0.0.1:<port>`. On Windows, `localhost` first tries IPv6 and adds about 2 s to every request. In PowerShell, set environment variables with `$env:NAME = "value"`; `set NAME=value` only works in cmd.

### What was verified (18 Nov 2026, one laptop)
- API bound to `0.0.0.0` and reached through the Wi-Fi address `10.218.43.14` (requests logged as `client=10.218.43.14`).
- `run_api.bat <port>` and `run_ui.bat <url> <port>` both start correctly on custom ports.
- Firewall state on Laptop 1: Wi-Fi profile *Public*, firewall on, inbound allow rules for `python.exe` exist on *Public*.
- **Not yet verified:** a request from a physically separate second laptop. Requests to your own IP do not pass through the Windows Firewall, so only a second machine can confirm step 4.

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
{"service": "pathsnap-api", "status": "ok", "predictor_mode": "real", "model_loaded": true, "model_name": "xgb",
 "artifact_version": "1.0.0", "predictor": "models.predictor:Predictor", "predictor_error": null,
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
| 503 | `trips_dir_missing` | The configured trips folder does not exist |

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
```powershell
python -m pytest tests -q
```
- `tests/test_api.py`: a synthetic trip folder and a contract-compliant `FakePredictor`, which proves the predictor is swappable. Covers health, trips, success, missing `.route`, 404, 422 (bad id, path traversal, empty body, extra field), 500, 503 (no model, missing trips folder), labelled mock fallback, Swagger/OpenAPI, the e2e checker's message when the API is down, and the real predictor when artifacts and data are present.
- `tests/test_health_contract.py`: `/health` has every field the UI reads in real, mock and unavailable modes. Also covers foreign or malformed `/health` bodies, safe defaults, details-table formatting, and headless Streamlit tests (`AppTest`) for each sidebar state.

Run the end-to-end check against a running API:
```powershell
python scripts\e2e_check.py --api http://<ip>:<port>
```
Results from 18 Nov 2026 are in `results/e2e_api_check.csv`, with the matching server log in `docs/evidence/api_console_2026-11-18.log`.

## 8. Troubleshooting
| Symptom | Fix |
|---|---|
| UI: "Cannot reach the API" | Wrong IP or port, API not started with `--host 0.0.0.0`, or the firewall rule is missing. Test `/health` in Laptop 2's browser. Make sure you used the **Wi-Fi** IPv4, not a VirtualBox/WSL adapter address. |
| UI: "… is not the PathSnap API" | Another program answers on that host/port (for example a Docker container on `127.0.0.1:8000`). Use another port: `.\scripts\run_api.bat 8010`. |
| `[Errno 10048] error while attempting to bind` at API start | The port is already in use (maybe an API you started earlier). Stop it or pass another port. |
| `Port 8501 is not available` at UI start | Another Streamlit is running. Use `.\scripts\run_ui.bat <api-url> 8502`. |
| The UI ignores the API URL you set | In PowerShell use `$env:PATHSNAP_API_URL = "..."` (not `set`), or pass the URL to `run_ui.bat`, or type it in the sidebar. |
| UI shows `ImportError` after `git pull` | Streamlit reloads `app.py` but not modules it imports. Stop the UI (Ctrl+C) and start it again after pulling new code. |
| Warnings at startup about XGBoost/scikit-learn versions | The saved model was created with older library versions. Predictions were verified on these versions (10/10 trips), so the warnings are not errors. |
| `/health` shows `predictor_mode: unavailable` | Read `predictor_error`: usually missing `artifacts/model_meta.json`, or `feature_config.json` doesn't match the code. Export the artifacts. |
| `/trips` returns 503 | Set `PATHSNAP_TRIPS_DIR` to the folder that contains the trip folders. |
| Every request about 2 s slower than `api_latency_ms` | Use `127.0.0.1`, not `localhost`, on Windows. |
| Long trips take several seconds | Measured server time is 0.15–3.6 s per trip in a batch run (up to 6.2 s for the longest trip when the machine is busy). Most of it is in Viterbi decoding (see the UI's Timings table). The map for a 4,000-point trip takes about 10 s more to draw. |
