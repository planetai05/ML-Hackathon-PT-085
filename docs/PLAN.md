# PathSnap: Understanding and Plan

## 1. The problem in one paragraph
Noisy 1 Hz GPS points (`.track`) must be matched to road segments (arcs) of a road network (`.nodes` + `.arcs`). The pipeline is **candidate generation → ML classifier that scores candidates → HMM/Viterbi to get a connected route**. Everything is served through a **FastAPI backend (Laptop 1)** and a **UI (Laptop 2)**. At evaluation the UI shows 10 hidden trips, and the evaluator picks one. The API runs the full pipeline, scores itself against server-side ground truth, and the UI shows the predicted route and metrics.

## 2. What I found in the data
- `Student folder/Student folder/train/` has **80 trips**, `student_test/` has **10 trips**, and each trip has 4 files: `.track`, `.nodes`, `.arcs`, `.route`. The 10 evaluator trips are **not given**.
- Checked on trip 1: 1,315 GPS points, 88,809 nodes, 165,790 arcs, and a route of 525 arcs.
  - Lon/lat is around (8.59, 49.84), which is Darmstadt, Germany. The files are in **lon, lat** order.
  - **Each trip has its own road-network subgraph.** The arcs and nodes are per trip, so don't assume ids are shared across trips.
  - The `.route` is an ordered list of arc ids. It is **not aligned one-to-one with GPS points**, because there are 525 arcs for 1,315 points. You have to derive per-point labels yourself.
  - Arcs are directed, from a node id to a node id, and the row index is the arc id.
- The `.route` is only "reference" data and may be imperfect, so state this as a limitation in the report.
- I did not confirm whether the library imports (xgboost, lightgbm, optuna, fastapi, folium) work. That check was still running.

## 3. Key design decision: how to frame the ML task
The docs imply a **candidate-ranking classifier**, using Top-1 accuracy, candidate miss rate and log loss.

1. For each GPS point, take all arcs within radius R (for example 30–50 m) using a spatial index (KD-tree/STRtree on arc midpoints or geometry, in a projected CRS).
2. Compute features per (point, candidate arc) pair:
   - Perpendicular distance, projection offset along the arc, and heading difference between GPS bearing and arc bearing.
   - Speed, acceleration, time gap, and distance rank among the candidates.
   - Arc length and arc bearing.
   - Optional context from neighbouring points, such as the previous and next points' distance to this arc.
3. Label the pair 1 if the candidate arc is in the point's true arc. The true arc must be derived by aligning the points to the `.route` (the nearest arc **within the route**, with monotonic progress along the route).
4. Train a binary classifier on these pairs. For each point, the highest-scoring candidate gives the Top-1 prediction.
5. Report the **candidate miss rate**: the share of points whose true arc is not in the candidate set. This caps the achievable accuracy.

> [!IMPORTANT]
> All spatial maths must use a **projected CRS** (for example UTM zone 32N, EPSG:32632) or haversine distance. Don't use Euclidean distance in degrees, because the docs call this out explicitly.

## 4. Step-by-step plan, mapped to the 12 stages in the guide

| # | Stage | What to do | Output |
|---|---|---|---|
| 1 | Understand | Read the docs (done). | This plan |
| 2 | EDA | Inventory notebook covering trip counts, points per trip, file sizes, NaNs, timestamp gaps (about 3% are not 1 s), extent, node and arc counts, connectivity and isolated components (networkx), anomalies, and route structure. Make plots and **Folium maps** (GPS in blue, route in green, network). | `01_eda.ipynb`, plots, a data-quality log |
| 3 | Preprocessing and features | Project to UTM, build the spatial index, generate candidates, engineer features, align the route to points to get labels, and write the candidate-miss-rate check. Fit scalers on train only. | `src/data.py`, `src/features.py`, cached parquet files |
| 4 | Train 5 or more models | Use different families, for example Logistic Regression, KNN, Random Forest, HistGradientBoosting, XGBoost/LightGBM, and an MLP. Use **GroupKFold by trip** on the 80 train trips. Log config, seed, train and inference time, CV macro F1, Top-1 accuracy, log loss, precision, recall, and route continuity. Evaluate on student_test. | `results/model_comparison.csv`, plots and confusion matrices |
| 5 | Select | Pick a primary metric, for example **Top-1 accuracy and macro F1 in CV, combined with log loss and compatibility with the HMM**. Justify the choice. | Selection rationale |
| 6 | Bayesian HPO | Use Optuna (TPE) on the chosen model with GroupKFold CV. Keep the trial history (study CSV or SQLite). Compare tuned and untuned models on CV F1, student-test Top-1 and F1, log loss and inference time. | `results/optuna_trials.csv` |
| 7 | HMM and Viterbi | **States** are the candidate arcs at each time step. **Emission** is the classifier probability, or a log combination with a distance term. **Transition** depends on whether the arcs are connected (same arc, adjacent via `to_node = from_node`, or reachable within k hops through shortest path). It is penalised by the mismatch between the route distance and the GPS displacement. Handle empty candidate sets, GPS gaps (reset or bridge), and invalid transitions (small epsilon). Tune decoder parameters on train and student_test only. Then compare **classifier-only against classifier+HMM** on pointwise accuracy, sequence-level metrics and route continuity. | `src/hmm.py`, comparison table |
| 8 | Export artifacts | Save the model (joblib), scaler, feature config and HMM config. | `artifacts/` |
| 9 | FastAPI | Implement `GET /trips` (the list of 10 trip ids) and `POST /predict {trip_id}`. It loads the trip files, runs candidates, features, model and Viterbi, **scores against server-side `.route` data**, and returns the predicted route, points and metrics. **Logs every request** (needed for the demo). | `api/main.py` |
| 10 | UI | Use Streamlit, or plain HTML+JS with Leaflet. It shows a dropdown of 10 trips, calls the API **over the network with a configurable API URL**, and displays a map (GPS, predicted route, optionally the reference route) and the accuracy. | `ui/` |
| 11 | Integrate | Test with 2 machines on one LAN (firewall, CORS, IP). Test **all 10** student_test trips through the API. Measure latency per trip, because large networks of about 90k nodes and 165k arcs may be slow. Precompute or cache the spatial index and graph. | Screenshots and logs |
| 12 | Report | Write the PDF using the structure in `Deliverables.pdf` (sections 2.2 to 7), with real results only. | Report PDF |

## 5. Deliverables checklist
- [ ] **Public GitHub repo** with folders `data_processing/ features/ models/ hpo/ hmm/ api/ ui/ visualisations/ results/ artifacts/`, plus `requirements.txt` and a README (install, run, deploy). No secrets. Commit regularly, because repo history is reviewed.
- [ ] **One consolidated PDF report**: EDA (with partition named for each analysis), a model comparison table, per-model Q&A (A, B and C), CV F1, student-test Top-1, F1 and log loss comparison plots, HPO, HMM comparison, **Folium map screenshot** (GPS, network, reference, prediction, legend, layer control), and UI and API (CMD log) screenshots.
- [ ] First-phase evaluation is **before lunch**, so check the schedule early.

## 6. Rules to avoid penalties
- **Never touch evaluator-test data.** Don't tune thresholds or models on it. Keep their labels server-side only (the API can hold the `.route` files, but the UI and model must never see them).
- **No hardcoded predictions.** The API must actually run the model live.
- Split **by trip** and fit scalers on train only. Any CV must use `GroupKFold`.
- Record seeds, configs and logs, since "metrics are verified against the repo".
- Don't fabricate numbers. Every team member must be able to explain everything in a viva (HMM, Viterbi, Bayesian optimisation, features and so on).
- Disclose AI assistance as per the event policy.
- Independent work: no sharing code, models or results with other teams.

## 7. Risks and time sinks
1. **Label alignment** (route arcs to points) is the hardest and most error-prone step. Validate it visually on Folium maps.
2. **Scale**: each trip's graph has about 90k nodes and 165k arcs, and a 80-trip candidate table could reach millions of rows. Use KD-tree/STRtree, a radius cap, and parquet files. Consider subsampling negatives only if needed.
3. **Class imbalance**: there is one true arc and many wrong candidates. Use macro F1, rank-based Top-1, and class weights.
4. **HMM transition cost**: use bounded Dijkstra (cutoff), cached per arc pair.
5. **API latency in the live demo**: preload artifacts at startup and test on the demo laptop.

## 8. Team split (3 people, deadline 14:00, UI = plain HTML/JS)
| Person | Owns | Hands off |
|---|---|---|
| A: Data and features | EDA, UTM projection, candidate generation, route-to-point label alignment, feature parquet | Feature parquet (train and student_test) and `features.py` by about 11:30 |
| B: Models, Bayesian optimisation, report | 5+ models with GroupKFold, comparison table and plots, Optuna, report skeleton | Final model, scaler, config in `artifacts/` |
| C: HMM, API, UI | Viterbi decoder, FastAPI (`/trips`, `/predict`, request logging), plain HTML/JS + Leaflet UI | Skeleton with mock data until A is done |

## 9. Timeline
| Time | Milestone |
|---|---|
| 10:40-11:30 | A: candidates, labels, features, Folium check. B: training and CV harness. C: API and UI skeleton, Viterbi code |
| 11:30-12:15 | B: train and compare 5+ models. C: real features into HMM. A: EDA plots and Folium maps |
| 12:15-12:45 | B: Optuna (about 30 trials). C: classifier-only vs classifier+HMM comparison |
| 12:45-13:15 | Export artifacts, integrate, test all 10 trips via API, two-laptop LAN test |
| 13:15-13:50 | Report PDF, README, GitHub push |
| 13:50-14:00 | Buffer |

## 10. Shortcuts
- Candidate radius about 40 m, cap 10-15 nearest arcs per point.
- Core features only first (distance, offset, heading difference, speed, rank).
- Models: logistic regression, KNN, random forest, HistGradientBoosting, XGBoost or LightGBM, MLP.
- HMM transitions: same or connected arc gets full probability, others a small penalty.
- Validate label alignment on a Folium map early.
- Commit often (repo history is reviewed).
