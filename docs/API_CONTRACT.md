# PathSnap ML ↔ API ↔ UI contract

Owner: Person C (`AryamanDK07`). Consumers: Person B (`planetai05`, FastAPI + Streamlit) and Person A (`ashvinrlobo`, Folium maps).
Tracked in issue #2. Any change to this file needs a heads-up on #2.

## 1. Python interface (`models/predictor.py`)

```python
from models.predictor import Predictor

predictor = Predictor(artifacts_dir="artifacts")  # loads model, scaler, configs once (call at API startup)
pred = predictor.predict_trip(trip_dir)          # uses ONLY .track, .nodes, .arcs
metrics = predictor.score(pred, trip_dir)        # server-side only: reads .route AFTER prediction
```

- `trip_dir`: folder containing `<id>.track`, `<id>.nodes`, `<id>.arcs` (and `<id>.route` for scoring), e.g. `data/Student folder/student_test/00000000`.
- `predict_trip` never opens `.route`. Reference labels stay on the API laptop and are used only by `score`.
- `Predictor(mock=True)` returns a schema-valid response without a trained model (GPS polyline as the "route"), so the API and UI can be built before the model is ready.
- Both methods are synchronous and CPU-only. They are safe to call from a FastAPI `def` endpoint (FastAPI runs it in a threadpool).

## 2. `predict_trip` response (JSON-serialisable dict)

| Field | Type | Meaning |
|---|---|---|
| `trip_id` | str | Folder name, e.g. `"00000000"` |
| `n_points` | int | Number of GPS observations |
| `gps_points` | `[[lon, lat, t], ...]` | Raw observations (lon/lat order, WGS84) |
| `per_point_arc` | `[int or null, ...]` | Decoded arc id per GPS point (`null` = no candidate) |
| `per_point_prob` | `[float or null, ...]` | Classifier probability of the chosen arc |
| `predicted_arcs` | `[int, ...]` | Ordered, de-duplicated, gap-filled route (arc ids = row index in `.arcs`) |
| `predicted_route_geojson` | GeoJSON `Feature` | `MultiLineString` in `[lon, lat]`; one line per continuous segment |
| `metadata` | object | `model_name`, `decoder` (`"hmm_viterbi"` or `"classifier_only"`), `n_segments`, `mean_candidates`, `empty_candidate_points`, `artifact_version` |
| `timings_ms` | object | `load`, `candidates`, `features`, `model`, `viterbi`, `total` |

## 3. `score` response

| Field | Meaning |
|---|---|
| `top1_accuracy` | Share of aligned GPS points whose decoded arc equals the reference-aligned arc (HMM output) |
| `classifier_only_top1` | Same metric using the classifier argmax without HMM |
| `route_precision` / `route_recall` / `route_f1` | Length-weighted overlap of `predicted_arcs` and the `.route` arcs |
| `route_continuity` | Share of consecutive decoded arcs that are connected (`to_node == from_node`) |
| `n_points_aligned` | Points with a reliable reference label |
| `reference_route_geojson` | Reference route (`MultiLineString`). **Not returned by default** (ground truth stays server-side); only included when the API is started with `PATHSNAP_EXPOSE_REFERENCE=1` (development only) |

See `docs/sample_predict_response.json` for a full example.

## 4. HTTP API (implemented by Person B, issue #22)

- `GET /health` → `{"status": "ok", "model_loaded": true}`
- `GET /trips` → `{"trips": ["00000000", ...]}` (the 10 evaluation trip ids found in the configured folder)
- `POST /predict` body `{"trip_id": "00000000"}` → `{**predict_trip(...), "metrics": score(...)}`
