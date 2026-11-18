# PathSnap: Distributed Map-Matching & Route Reconstruction (Team PT-085)

PathSnap matches noisy 1 Hz GPS points to a road network and reconstructs the vehicle's route.

**Pipeline:** candidate road generation → ML classifier scores each (GPS point, candidate arc) → HMM/Viterbi builds a connected route → FastAPI backend (Laptop 1) + Streamlit UI (Laptop 2).

**Run the application:** Laptop 1 `scripts\run_api.bat`, Laptop 2 `scripts\run_ui.bat http://<laptop1-ip>:8000`. See `docs/API_UI.md` for endpoints, configuration, and how to plug in the selected model.

## Project status
| Part | Owner | Status |
|---|---|---|
| A: EDA, candidates, features, labels | Person A | Done (see `docs/HANDOFF_PERSON_A.md`) |
| B: Models, Bayesian optimisation, report | Person B | To do |
| C: HMM/Viterbi, FastAPI, UI | Person C | To do |

## Repository layout
```
data_processing/   loader.py (read raw trip files), eda.py (inventory, plots, Folium map)
features/          features.py (projection, candidates, features), build_dataset.py (labels + parquet)
data/              generated parquet files (git-ignored, regenerate, see below)
results/           eda_inventory.csv, candidate_miss_rate.csv
visualisations/    EDA plot and Folium HTML maps
docs/              PLAN.md (overall plan), PERSON_A_PLAN.md, HANDOFF_PERSON_A.md
```
Planned (to be added by B and C): `models/`, `hpo/`, `hmm/`, `api/`, `ui/`, `artifacts/`.

## Setup
```bash
pip install -r requirements.txt
```
Python 3.11 was used.

## Raw data location
The code expects the dataset at `<repo parent>/Student folder/Student folder/{train,student_test}/<trip_id>/`.
This is set in `data_processing/loader.py` (`DEFAULT_DATA_ROOT`). Change it if your data is elsewhere.
The raw data is not committed to this repo.

## Reproduce Part A outputs
Run all commands from the repo root.
```bash
python -m data_processing.eda          # results/eda_inventory.csv, visualisations/eda_summary.png, Folium map
python -m features.build_dataset       # data/train_features.parquet, data/student_test_features.parquet, results/candidate_miss_rate.csv
```
`build_dataset` takes about 2 minutes. The parquet files are git-ignored (the train file is about 90 MB), so each teammate regenerates them with the command above.

## Use the feature extractor (for the API)
```python
from pathlib import Path
from features.features import extract_trip_features, FEATURE_COLS

df, meta = extract_trip_features(Path("<path to trip folder>"))
```
This does not read `.route`, so it is safe for live inference. See `docs/HANDOFF_PERSON_A.md` for details.

## Rules to remember
- Never use the evaluator-test data for training, tuning or model selection.
- Ground-truth `.route` labels stay server-side only. No hardcoded predictions.
- Split by trip (`GroupKFold` on `trip_id`). Fit scalers on train only.
- No secrets in the repo.
