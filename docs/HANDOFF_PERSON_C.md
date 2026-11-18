# PathSnap: ML Model Handoff to Person C (HMM & API Integration)

**Author:** Person B (Senior Machine Learning Engineer)  
**Target:** Person C (HMM / Viterbi, FastAPI, and UI Integration)  
**Deliverable for Issues:** #19, #25  
**Artifact Directory:** `artifacts/`  

---

## 1. Final Trained Model Artifacts

All final model artifacts are stored in `artifacts/`:

| Artifact | File Path | Format / Size | Description |
|---|---|---|---|
| **Champion Model (Tuned)** | `artifacts/best_model.joblib` | `joblib` (~248 KB) | Final tuned XGBoost classifier trained on all 80 training trips. |
| **Tuned Model Copy** | `artifacts/best_model_tuned.joblib` | `joblib` (~248 KB) | Identical to `best_model.joblib`. |
| **Untuned Baseline** | `artifacts/best_model_untuned.joblib` | `joblib` (~464 KB) | Untuned default XGBoost baseline for ablation comparison. |
| **Feature Scaler** | `artifacts/scaler.joblib` | `joblib` (~1.6 KB) | `StandardScaler` fitted on training features (optional; XGBoost uses raw float32). |
| **Feature Configuration** | `artifacts/feature_config.json` | JSON | Exact feature column list (24 features), candidate parameters, and CRS metadata. |
| **Model Metadata** | `artifacts/model_metadata.json` | JSON | Complete model specifications, cross-validation metrics, and class interpretations. |
| **Model Metadata (Alias)** | `artifacts/model_meta.json` | JSON | Duplicate for backwards compatibility with API test suites. |
| **Best Hyperparameters** | `artifacts/best_hyperparameters.json` | JSON | Optimal hyperparameters discovered via Optuna TPE Bayesian search. |
| **HMM Transition Config** | `artifacts/hmm_config.json` | JSON | Baseline transition and emission parameters for sequential Viterbi decoding. |

---

## 2. Feature Schema and Exact Ordering

The model expects exactly **24 engineered features** in float32 format, in the following exact order:

```python
FEATURES = [
    # Spatial distance
    "dist",               # Perpendicular distance from GPS observation to candidate arc (metres)
    "dist_rank",          # Rank of candidate distance among all candidates for this observation (0 = nearest)
    "dist_minus_min",     # Candidate distance minus the nearest candidate distance (metres)
    "dist_ratio",         # Ratio of candidate distance to minimum candidate distance
    "n_cand",             # Total number of generated candidates for this GPS observation
    # Position along the road segment
    "proj_t",             # Clipped projection parameter t in [0, 1] along the arc vector
    "beyond_end",         # Distance beyond segment endpoints in metres (0 if projected inside)
    "arc_len",            # Physical arc length in metres
    "log_arc_len",        # Log-transformed arc length: log(1 + arc_len)
    # Heading & Bearing
    "heading_diff",       # Angular difference between vehicle trajectory heading and road segment bearing (degrees, [0, 180])
    "cos_heading",        # Cosine of the angular heading difference
    "heading_rank",       # Rank of heading alignment among candidates
    "heading_valid",      # Binary flag (1 if vehicle displacement > 3m allowing reliable bearing, else 0)
    # Speed & Kinematics
    "speed",              # Estimated vehicle speed (m/s)
    "accel",              # Estimated vehicle acceleration (m/s^2)
    "dt_prev",            # Time interval to preceding GPS observation (seconds)
    "dt_next",            # Time interval to succeeding GPS observation (seconds)
    # Spatial context with neighbouring observations
    "dist_prev_pt",       # Distance from previous GPS observation to this same arc (metres)
    "dist_next_pt",       # Distance from next GPS observation to this same arc (metres)
    "dist_ctx_mean",      # Mean distance across (t-1, t, t+1) observations to this arc
    "conn_prev_nearest",  # Binary indicator: does this arc connect to the nearest candidate of t-1?
    "conn_next_nearest",  # Binary indicator: does this arc connect to the nearest candidate of t+1?
    # Road network topology
    "out_deg_to",         # Out-degree of the destination node (`to_node`)
    "in_deg_from",        # In-degree of the origin node (`from_node`)
]
```

---

## 3. Positive-Class Interpretation and Probability Extraction

- **Target Definition:** Binary label (`label \in {0, 1}`).
  - `0`: The candidate arc is an incorrect / off-route road segment.
  - `1`: The candidate arc lies on the true reference vehicle route.
- **`predict_proba()` output structure:**
  - `model.predict_proba(X)` returns an array of shape `(N, 2)`.
  - Column `0`: Probability that candidate is wrong: $P(\text{incorrect} \mid \mathbf{x})$.
  - Column `1`: Probability that candidate is true: $P(\text{correct} \mid \mathbf{x})$.
- **Extract positive emission probabilities:**
  ```python
  p_candidates = model.predict_proba(X)[:, 1]
  ```

---

## 4. How to Load and Run Inference (Code Example)

```python
import joblib
import numpy as np
from features.trip_io import load_trip
from features.candidates import generate_candidates
from features.build_features import trip_features, FEATURES

# 1. Load trained model once at startup
model = joblib.load("artifacts/best_model.joblib")

# 2. Process an incoming trip (e.g. at FastAPI endpoint)
trip = load_trip("data/Student folder/student_test/00000000", with_route=False)

# 3. Generate candidate road segments within 40m radius
cands = generate_candidates(trip)

# 4. Extract identical 24 features
df_features = trip_features(trip, cands)

# 5. Predict posterior probabilities
X = df_features[FEATURES].to_numpy(dtype=np.float32)
df_features["p"] = model.predict_proba(X)[:, 1]

# 6. Example probability output for point 0 candidates
pt0 = df_features[df_features.point_idx == 0][["point_idx", "arc_id", "dist", "heading_diff", "p"]]
print(pt0.sort_values("p", ascending=False).to_string())
```

---

## 5. Seamless API and Predictor Integration

The agreed public `Predictor` interface in `models/predictor.py` already encapsulates the entire workflow:

```python
from models.predictor import Predictor

# Initialise at FastAPI startup (in lifespan or state)
predictor = Predictor(artifacts_dir="artifacts", mock=False)

# Live prediction (never touches .route)
pred = predictor.predict_trip("data/Student folder/student_test/00000000")

# Server-side scoring (reads .route after prediction)
metrics = predictor.score(pred, "data/Student folder/student_test/00000000")
```

The response matches `docs/API_CONTRACT.md` and passes all `tests/test_api.py` endpoint validations.
