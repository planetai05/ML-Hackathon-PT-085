# Person A Handoff: What Was Done and How to Continue

For Persons B and C. Read this first. All commands run from the repo root.

## 1. What Person A built
| File | What it does |
|---|---|
| `data_processing/loader.py` | `list_trips(dir)` and `load_trip(trip_dir, load_route=True)` return the track, nodes, arcs and route for one trip. |
| `data_processing/eda.py` | Scans all 90 trips, then writes `results/eda_inventory.csv`, `visualisations/eda_summary.png` and a Folium map. |
| `features/features.py` | Projects coordinates to meters, finds candidate arcs and computes features. Main function: `extract_trip_features`. |
| `features/build_dataset.py` | Aligns `.route` to GPS points to make labels, then writes the two parquet files and `results/candidate_miss_rate.csv`. |

## 2. How the data was turned into a table
1. **Projection to meters:** each trip is projected with its own local azimuthal-equidistant projection, centred on the trip's median GPS point. One trip (`00000057`) is in Ecuador, so a fixed German UTM zone broke.
2. **Candidates:** for every GPS point, take the up to 15 closest arcs within 50 m (fallback radius 150 m if none). Arcs are indexed by sampled points along their length, then the exact point-to-segment distance is computed.
3. **One row = one (GPS point, candidate arc) pair.**
4. **Labels:** the `.route` is aligned to the GPS points by dynamic programming. Points move forward along the route and the total distance is minimised. `label = 1` if the candidate arc is the aligned route arc for that point, else 0.

## 3. The parquet files
Regenerate them with `python -m features.build_dataset` (about 2 minutes). They are git-ignored.

| File | Trips | GPS points | Rows |
|---|---|---|---|
| `data/train_features.parquet` | 80 | 208,417 | 1,740,981 |
| `data/student_test_features.parquet` | 10 | 19,160 | 164,349 |

### Columns
- **Identifiers:** `trip_id`, `point_idx` (index of the GPS point in its trip), `arc_id`, `from_node`, `to_node`, `timestamp`.
- **Model inputs:** these 13 columns are listed in `FEATURE_COLS` in `features/features.py`.
  - `dist_m`: distance from the GPS point to the arc, in meters.
  - `dist_rank`: 1 means closest arc for that point.
  - `dist_diff_from_min`: `dist_m` minus the closest candidate's distance.
  - `proj_t`: where the point projects onto the arc (0 start, 1 end, clamped).
  - `proj_t_unclamped`: same, without clamping.
  - `arc_len_m`: arc length in meters.
  - `heading_diff_deg`: angle between the vehicle's heading and the arc direction (0–180).
  - `speed_mps`, `dt_prev`, `step_dist_m`: movement since the previous GPS point.
  - `prev_dist_m`, `next_dist_m`: distance from the previous and next GPS point to this same arc.
  - `n_candidates`: number of candidates for this point.
- **Labels and quality (never use as model inputs):** `label`, `true_arc_id`, `aligned_dist_m`, `label_valid`.

## 4. Important: `label_valid`
For some trips the GPS track runs past the end of the reference route. The alignment then pins the leftover points to the last route arcs, which are far away. Those points get all-zero labels, but the zero labels are wrong.

So `label_valid` is true only when the aligned arc is within 50 m of the GPS point.

- **Train and score only on rows where `label_valid == True`.** Skipping this teaches the model wrong negatives and makes accuracy look worse than it is.
- Train: 88.84% of points are valid. Student test: 99.86%.
- Trips `00000033` (about 59% of points invalid) and `00000054` (about 89% invalid) are mostly unusable. Consider excluding them from training entirely.

## 5. Candidate miss rate
The share of GPS points where the true arc is not among the candidates. The best achievable top-1 accuracy is 100% minus this.

| Partition | Raw miss rate | Miss rate on valid points |
|---|---|---|
| train | 6.81% | 0.47% |
| student_test | 0.19% | 0.05% |

Per-trip numbers are in `results/candidate_miss_rate.csv`. Even among valid points, trip `00000033` still misses 19.4% and trip `00000054` misses 10.0%, so these two trips are the weakest. Trip `00000002` is fine on valid points (0.02% miss, 88% of points valid).

Baseline: the closest arc (`dist_rank == 1`) is the true arc only about 54% of the time, and the true arc is in the closest 3 about 91–95% of the time. So models need to beat 54% top-1.

## 6. Data findings (for the report)
- No NaNs and no backwards timestamps in train or student_test.
- About 3% of time steps are not 1 s. Some trips have large GPS gaps: trip `90` has a 153,064 s gap and trip `78` has 52,810 s. Handle large `dt` in the HMM.
- The `.route` has small breaks (non-connected consecutive arcs) in 13 of 80 train trips and 1 of 10 student_test trips.
- Each trip has its own road network. Arc ids are not shared between trips.
- Road networks have about 20 components (median) in train. About 98% of nodes are in the largest one.
- Trip `00000057` is in Ecuador, the rest are near Darmstadt, Germany.

## 7. Person B: how to continue
```python
import pandas as pd
from features.features import FEATURE_COLS
from sklearn.model_selection import GroupKFold

train = pd.read_parquet("data/train_features.parquet")
test = pd.read_parquet("data/student_test_features.parquet")
train = train[train.label_valid]   # drop unreliable labels
test = test[test.label_valid]

X, y, groups = train[FEATURE_COLS], train["label"], train["trip_id"]
for tr_idx, va_idx in GroupKFold(n_splits=5).split(X, y, groups):
    ...
```
- Always split by `trip_id`. Fit any scaler on train only.
- **Top-1 accuracy:** for each (`trip_id`, `point_idx`), take the candidate with the highest predicted probability and check `label == 1`. Points where the true arc is not in the candidates count as wrong, or report them as miss rate.
- The classes are imbalanced (about 12% positives). Use class weights or macro F1, and also report log loss.
- Save the final model, scaler and feature list to `artifacts/` for Person C.

## 8. Person C: how to continue
```python
from pathlib import Path
from features.features import extract_trip_features, FEATURE_COLS

df, meta = extract_trip_features(Path("<trip folder>"))
# df: one row per (point_idx, arc_id) with the 13 FEATURE_COLS
# meta: arrays "gx", "gy" (meters), "nx", "ny" (node coords, meters), "ts", "u", "v" (arc from/to nodes), "arc_len"
```
- It does **not** read `.route`, so it is safe for live API inference. Keep labels server-side only.
- It takes about 0.3 s for a 1,300-point trip, but the largest trips are slower. Load once at startup if possible.
- Hidden states for the HMM: the candidate arcs at each time step (`df` rows per `point_idx`). Use the model's probability as the emission score.
- Transitions: use `meta["u"]` and `meta["v"]`. Arcs `a -> b` are connected when `v[a] == u[b]`. Same arc is also allowed.
- Handle points with no candidates (they are missing from `df`) and large `dt_prev` gaps.
- For scoring on the server, `build_dataset.align_gps_to_route` shows how to turn a `.route` into per-point true arcs.

## 9. Known limitations
- Labels come from alignment to the reference `.route`, not independently verified ground truth. They can be wrong where the route is truncated or has breaks.
- A fixed 50 m radius and 15 candidates can miss the true arc at very noisy points or dense intersections.
- Heading uses the previous GPS point, so it is noisy when the vehicle is nearly stationary (the last bearing is reused).
