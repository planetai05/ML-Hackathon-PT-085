# 7. API, UI and Deployment (draft, Person C)

Author: Person C (`AryamanDK07`). This is a standalone draft for the consolidated report. All numbers come from runs on 18 Nov 2026, recorded in `results/e2e_api_check.csv` and `docs/evidence/api_console_2026-11-18.log`. Screenshots are in `docs/screenshots/`.

## 7.1 System architecture
PathSnap runs as two programs on two laptops that talk JSON over HTTP on the local network.

```
Laptop 2 (evaluator)                      Laptop 1 (model host)
Streamlit UI (ui/app.py)  --HTTP/JSON-->  FastAPI (api/main.py)
  trip dropdown, map, metrics               GET /health, GET /trips, POST /predict
                                              |
                                              v
                                            Predictor (models/predictor.py, contract docs/API_CONTRACT.md)
                                              candidates -> 24 features -> XGBoost -> HMM/Viterbi -> route
                                              score(): compares with the server-side .route AFTER prediction
```

- **Laptop 1** hosts the trained XGBoost classifier, the HMM/Viterbi decoder, the per-trip road networks and the reference routes used for scoring.
- **Laptop 2** only runs the user interface. It never loads a model, the dataset or any `.route` file. Everything it shows comes from the API.
- **Swappable model:** the API loads the predictor named by `PATHSNAP_PREDICTOR` and calls only the agreed contract (`predict_trip`, `score`). The selected model can change without any API or UI change.

## 7.2 FastAPI endpoints
| Endpoint | Purpose | Main fields returned |
|---|---|---|
| `GET /health` | Service and model status | `service: "pathsnap-api"`, `predictor_mode` (real/mock/unavailable), `model_name`, `artifact_version`, `predictor_error`, `n_trips` |
| `GET /trips` | Evaluation trips available on the server | `trips`, `count` |
| `POST /predict` `{"trip_id": "00000056"}` | Runs the full pipeline live and scores it | GPS points, per-point arc and probability, predicted arcs, route GeoJSON, stage timings, metrics (HMM and classifier-only Top-1, route precision/recall/F1, continuity), reference route GeoJSON, `mode`, `api_latency_ms` |

- **Validation:** Pydantic checks requests. `trip_id` must match `^[A-Za-z0-9_-]{1,64}$`, and extra fields are rejected.
- **Errors:** all use one JSON shape, `{"error", "detail"}`: 404 `unknown_trip`, 422 `validation_error`, 500 `prediction_failed`, 503 `predictor_unavailable` or `trips_dir_missing`.
- **Logging:** every request is logged to the console and `api/logs/requests.log`, and each prediction adds a `PREDICT trip=... mode=... model=... top1=... latency=... client=...` line.
- **Docs:** interactive Swagger documentation is served at `/docs` (Figure 7.7).

## 7.3 Streamlit interface
- **Sidebar:** the API URL (editable at run time), connection and model status, and map-layer toggles. A real model shows green "Connected · model xgb" (Figure 7.1); mock mode shows a yellow warning; a missing model shows the reason in red.
- **Trip selection and prediction:** a dropdown filled from `GET /trips` (Figure 7.2), then **Predict route** with a spinner while the pipeline runs (Figure 7.3).
- **Results** (Figure 7.4): six metric tiles (HMM point accuracy and its gain over classifier-only, classifier-only accuracy, route F1, route continuity, GPS points, API latency) and an interactive Folium map. The map draws GPS points (blue), the predicted route (red), the reference route (green), start and end markers, a scale and a layer control (Figure 7.5).
- **Details** (Figure 7.6): model, decoder, route arcs, candidates per point, route precision and recall, and per-stage timings in milliseconds.

## 7.4 API–UI communication and prediction workflow
1. On load, the UI calls `GET /health` and checks the reply is from PathSnap (`service` / `predictor_mode`). Then it calls `GET /trips`.
2. The evaluator selects a trip and clicks **Predict route**. The UI sends `POST /predict {"trip_id": ...}`.
3. The API checks the trip exists, then calls `Predictor.predict_trip(trip_dir)`, which reads only `.track`, `.nodes` and `.arcs`: candidate generation (40 m radius, up to 12 arcs), 24 features, XGBoost probabilities, then Viterbi decoding.
4. Only after prediction does the API call `Predictor.score(...)`, which reads the server-side `.route` and computes the metrics. The labels never reach the model or the UI before the prediction is made.
5. The API returns one JSON response. The UI draws the tiles, map and tables from it.

## 7.5 Error handling
| Situation | Behaviour (verified) |
|---|---|
| API not running or wrong IP | Sidebar: "Cannot reach the API at …", plus startup guidance (Figure 7.8) |
| Another program on the same port | Sidebar: "… is not the PathSnap API" (automated UI test) |
| No model loaded | `/health` → `unavailable` with the reason; `/predict` → 503 `predictor_unavailable` |
| Mock mode (development only) | `/health` → `mock`; every response has `"mode": "mock"`; UI warning banner; the server log says responses are not ML predictions |
| Unknown or invalid trip id | 404 `unknown_trip` / 422 `validation_error` |
| Trips folder missing | `/trips` and `/predict` → 503 `trips_dir_missing` |
| Scoring fails after a prediction | The route is still returned, with `metrics_error` explaining why |

## 7.6 Testing results
- **Automated tests:** `python -m pytest tests -q` → **55 passed** on 18 Nov 2026. This is the whole team suite. The API and UI tests cover every endpoint and error code, the `/health` contract in all three modes, and headless Streamlit tests for each sidebar state.
- **Consistency check:** for trips 00000052 and 00000056, the metrics and predicted arcs returned over HTTP were identical to calling the Predictor directly in Python.
- **End-to-end over the network address:** API bound to `0.0.0.0:8020`, requests sent to the Wi-Fi address `10.218.43.14` with `scripts/e2e_check.py`. **10/10 trips returned HTTP 200 with `mode=real`, `model=xgb`.**

| Trip | GPS points | Top-1 (HMM) | Top-1 (classifier only) | Route F1 | Server time (ms) |
|---|---:|---:|---:|---:|---:|
| 00000000 | 2,503 | 0.955 | 0.945 | 0.983 | 3,079 |
| 00000007 | 1,354 | 0.922 | 0.924 | 0.987 | 1,496 |
| 00000009 | 1,049 | 0.870 | 0.881 | 0.920 | 1,284 |
| 00000040 | 2,995 | 0.930 | 0.916 | 0.960 | 2,533 |
| 00000052 | 262 | 0.966 | 0.973 | 0.979 | 151 |
| 00000056 | 3,958 | 0.931 | 0.907 | 0.969 | 2,671 |
| 00000065 | 768 | 0.861 | 0.836 | 0.943 | 380 |
| 00000076 | 1,486 | 0.898 | 0.886 | 0.922 | 984 |
| 00000077 | 3,911 | 0.734 | 0.716 | 0.848 | 3,617 |
| 00000091 | 874 | 0.890 | 0.863 | 0.949 | 1,099 |
| **All** | 19,160 | **0.883** (point-weighted) | **0.869** | **0.946** (mean) | median 1,390, max 3,617 |

- **Latency:** varies with trip length and machine load. The longest trip (00000056) took 2.7 s in the batch run and 5.8–6.2 s while other work ran on the same laptop. The UI's timing table shows that most of the time is Viterbi decoding (4.9 s of 5.2 s in Figure 7.6).

## 7.7 Deployment instructions
Laptop 1 (PowerShell, repo root):
```powershell
.\scripts\run_api.bat 8000
```
Laptop 2:
```powershell
.\scripts\run_ui.bat http://<laptop1-wifi-ip>:8000 8501
```
Then open `http://localhost:8501`.

Use the IPv4 address of the **Wi-Fi** adapter from `ipconfig`, allow inbound TCP on the API port in Windows Firewall, and check `http://<ip>:8000/health` from Laptop 2 first. Full steps and troubleshooting are in `docs/API_UI.md`.

**Status of the two-laptop test:** network readiness was verified on one laptop (binding to `0.0.0.0`, access through its Wi-Fi address, run scripts, firewall rules present). *[Add here: the result of the physical two-laptop run and Figures 7.9–7.10 once performed.]*

## Figures
| Figure | File | Shows |
|---|---|---|
| 7.1 | `docs/screenshots/01_ui_connected_real_model.jpg` | UI connected to the API on the LAN address, real model xgb, 10 trips |
| 7.2 | `docs/screenshots/02_ui_trip_selection.jpg` | Trip dropdown filled from `GET /trips` |
| 7.3 | `docs/screenshots/03_ui_prediction_running.jpg` | Spinner while the pipeline runs |
| 7.4 | `docs/screenshots/04_ui_result_metrics_map_trip56.jpg` | Metrics and predicted vs reference route, trip 00000056 |
| 7.5 | `docs/screenshots/05_ui_map_layer_control.png` | Map layer control |
| 7.6 | `docs/screenshots/06_ui_details_and_timings.jpg` | Prediction details and stage timings |
| 7.6b | `docs/screenshots/06b_ui_trip52_details_formatted.jpg` | Short trip 00000052: details table; HMM 0.8 pts below classifier-only (HMM does not always help) |
| 7.7 | `docs/screenshots/07_api_swagger_docs.jpg` | Swagger documentation of the three endpoints |
| 7.8 | `docs/screenshots/08_ui_error_api_unreachable.jpg` | Error message when the API is unreachable |
| 7.9 | *to capture* | API terminal (CMD/PowerShell) on Laptop 1 showing `PREDICT ... mode=real model=xgb` lines |
| 7.10 | *to capture* | UI on Laptop 2 connected to Laptop 1's IP during the two-laptop test |
