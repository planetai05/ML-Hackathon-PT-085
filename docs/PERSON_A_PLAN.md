# Person A Plan: Data Exploration, Features & Labeling

**Goal:** Understand and clean the raw GPS + road-network data, convert coordinates to meters, generate nearby road candidates, build features, attach ground-truth labels from `.route`, and hand off clean `.parquet` datasets (to Person B) + a reusable feature function (to Person C).

---

## 6 Simple Steps

### Step 1: Check & Map the Raw Data (EDA)
- Scan all **80 `train`** trips and **10 `student_test`** trips.
- Check trip counts, GPS points per trip, node/arc counts, route lengths, and file sizes.
- Check for missing/NaN values, out-of-range coordinates, duplicate/non-increasing timestamps, and GPS time gaps (`dt > 1s`).
- Check road-network health (connected components and isolated nodes).
- Save summary CSVs, EDA plots, and an interactive **Folium map** (blue dots = GPS, green line = `.route`, grey lines = nearby road network).

### Step 2: Convert GPS Coordinates to Meters
- Raw coordinates are `(longitude, latitude)` in degrees (around Darmstadt, Germany).
- Project all GPS points and road nodes into **meters** using `pyproj` with a **per-trip local azimuthal-equidistant projection** centred on the trip's median GPS point (a fixed UTM zone broke on trip `00000057`, which is in Ecuador, not Germany), so distances and angles are physically accurate anywhere.

### Step 3: Find Nearby Road Candidates
- For each trip's road network, compute each arc's start/end coordinates, midpoint, length, and direction (bearing).
- Use `scipy.spatial.cKDTree` to quickly find nearby arcs, then compute the exact point-to-segment distance in meters.
- Keep the **top 15 closest arcs** within **50 meters** for each GPS point.

### Step 4: Build Clues (Features) for Each Candidate
Compute features for every `(GPS point, candidate arc)` pair:
- **Distance to road** (meters) and **distance rank** (`1` = closest, `2` = 2nd closest, etc.).
- **Projection offset** along the road segment (`0` = start, `1` = end).
- **Heading difference**: angle between the vehicle's movement direction and the road segment's direction.
- **Vehicle kinematics**: speed (m/s), time gap `dt` (s), step distance (m).
- **Road geometry**: arc length (m).
- **Neighbor context**: distance from previous/next GPS point to the same candidate arc.

Put Steps 2–4 into a clean, reusable Python function (`features/features.py`) that works **without** `.route` so Person C's API can call it on unseen evaluation trips.

### Step 5: Attach Labels Using `.route` & Check Quality
- Align each GPS point in chronological order to the nearest arc **in the `.route` sequence**, enforcing forward progress along the route.
- Mark `label = 1` for the matching candidate arc and `label = 0` for all other candidates.
- Check **Candidate Miss Rate** (% of GPS points where the true arc was missed by the candidate search). Tune radius/top-K if miss rate is high.
- Visually sanity-check labeled points on a Folium map.

### Step 6: Save & Hand Off
- Save `data/train_features.parquet` (80 `train` trips) and `data/student_test_features.parquet` (10 `student_test` trips), keeping `trip_id` and `point_idx` columns for `GroupKFold` and per-point evaluation.
- Commit code, plots, and maps to GitHub.

---

## Folder Layout
- `data_processing/` → `loader.py` (read raw files), `eda.py` (inventory, quality checks, plots, Folium maps)
- `features/` → `features.py` (projection, candidates, feature extraction), `build_dataset.py` (label alignment + export `.parquet`)
- `visualisations/` → EDA charts + Folium HTML maps
- `results/` → EDA inventory & data-quality CSVs
- `data/` → generated `.parquet` files (gitignored if large)

---

## To-Do Task List

- [x] **1. Create folder structure** (`data_processing/`, `features/`, `visualisations/`, `results/`, `data/`) and `.gitignore` (ignore `data/*.parquet`).
- [x] **2. Write `data_processing/loader.py`** to cleanly load `.track`, `.nodes`, `.arcs`, and `.route` for any trip.
- [x] **3. Write & run `data_processing/eda.py`** across `train` (80) and `student_test` (10) partitions:
  - [x] Generate trip inventory CSV (`results/eda_inventory.csv`) with counts, sizes, missing values, time gaps, and network connectivity stats.
  - [x] Generate EDA plots (`visualisations/eda_summary.png`) for GPS points, time gaps, and network sizes.
  - [x] Generate interactive Folium map (`visualisations/eda_trip_00000001.html`) showing GPS points, `.route`, and nearby road network.
- [x] **4. Write `features/features.py`**:
  - [x] Coordinate projection to meters (per-trip local AEQD projection).
  - [x] Fast `cKDTree` candidate search + exact point-to-segment distance in meters.
  - [x] Feature extraction for each `(point, candidate_arc)` pair.
  - [x] Expose `extract_trip_features(trip_dir)` for Person C's API.
- [x] **5. Write & run `features/build_dataset.py`**:
  - [x] Implement monotonic `.route`-to-GPS alignment to assign ground-truth `arc_id` per GPS point.
  - [x] Compute and log **Candidate Miss Rate** on `train` and `student_test`.
  - [x] Generate a label-verification Folium map to visually confirm labels match `.route`.
  - [x] Export `data/train_features.parquet` and `data/student_test_features.parquet`.
- [ ] **6. Hand off & Commit**:
  - [x] Share `train_features.parquet`, `student_test_features.parquet`, and `features/features.py` with Persons B and C.
  - [ ] Commit all scripts, CSVs, plots, and Folium HTML maps to GitHub.


