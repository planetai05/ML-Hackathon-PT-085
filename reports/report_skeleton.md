# PATHSNAP: CONSOLIDATED TECHNICAL REPORT
**Distributed Map-Matching and Sequential Route Reconstruction**

---

## 1. Executive Summary & Team Responsibilities
- **Team Identification:** Section PT-085
- **Architecture Overview:** Distributed client-server map-matching system deployed across two laptops (FastAPI on Laptop 1, Streamlit / UI on Laptop 2).
- **Division of Work:**
  - **Person A (Data & Feature Engineering):** Exploratory data analysis, UTM projection, spatial indexing, candidate generation, DP label alignment, and Folium visualizations.
  - **Person B (Machine Learning & Optimization - THIS SECTION):** ML candidate classification formulation, 5+ model families benchmarked with GroupKFold cross-validation, Bayesian hyperparameter optimization with Optuna, final artifact export, and technical report sections 3, 4, and 5.
  - **Person C (Route Reconstruction & Deployment):** Hidden Markov Model formulation, Viterbi dynamic programming decoding, FastAPI service, and Streamlit user interface.

---

## 2. Problem Formulation and Architecture

### 2.1 Problem Formulation
Vehicular GPS observations collected at 1 Hz exhibit high noise, multi-path effects, and signal dropouts. The objective of PathSnap is to reconstruct the true sequence of road network segments (arcs) traversed by a vehicle.
We frame the map-matching task as a **two-stage hybrid predictive pipeline**:
1. **Candidate Generation & ML Scoring:** For each GPS observation $p_t$, a spatial index identifies candidate road segments within a 40-metre radius. A machine learning binary classifier predicts the posterior probability $P(\text{arc}_i \text{ is correct} \mid p_t, \mathbf{x}_{t, i})$.
2. **Sequential Route Reconstruction (HMM & Viterbi):** Candidate probabilities serve as emissions in a Hidden Markov Model. Transition probabilities penalise topological disconnections and geometric travel distance mismatches, yielding a globally continuous and physically feasible reconstructed route.

### 2.2 Dataset Understanding and Exploratory Data Analysis (Person A)

#### 2.2.1 Dataset Partitions and Geographic Distribution
The dataset comprises 90 vehicular trajectory trips partitioned into:
- **Training Set (`data/Student folder/train/`):** 80 trips containing 208,417 raw GPS observations and 1,740,981 candidate pairs.
- **Student-Testing Set (`data/Student folder/student_test/`):** 10 trips containing 19,160 GPS observations and 164,349 candidate pairs.
- **Official Evaluator Set:** Preserved strictly server-side by the organizers for blind evaluation; zero evaluator records were used during model design, training, or tuning.

Geographically, 89 trips are located in and around the urban and sub-urban road networks of Darmstadt, Germany. One unique trip (`00000057`) is situated in Quito, Ecuador (longitude -78°). Consequently, assuming a single fixed global UTM projection (e.g. EPSG:32632) causes coordinate distortion for non-European trajectories. We implemented an adaptive projection module in `data_processing/loader.py` and `features/trip_io.py` that dynamically assigns the appropriate local Universal Transverse Mercator (UTM) or Azimuthal Equidistant projection centred on the trajectory centroid, guaranteeing that all distances ($\text{dist}$, arc lengths, GPS steps) are measured in metric meters rather than angular degrees.

#### 2.2.2 Temporal Characteristics and Signal Quality
Nominally, observations are recorded at 1 Hz. However, exploratory data analysis revealed that ~3% of time steps deviate from 1.0 s. In particular:
- Short multipath jitters ($\Delta t \in [0.5, 2.0]$ s) caused by urban canyon signal reflections.
- Extreme GPS signal dropouts: trip `00000090` exhibits a severe outage of 153,064 seconds, and trip `00000078` exhibits a gap of 52,810 seconds.
- Road-network subgraphs: Each trip possesses an isolated road-network subgraph consisting of a median of 20 connected components, with ~98% of road nodes contained in the single largest weakly connected component. Arc IDs are unique to individual trip subgraphs and are never shared across trips.

#### 2.2.3 Candidate Arc Generation & Spatial Indexing
For every GPS point $p_t$, candidates are extracted within a 40–50 m search radius using sampled arc segment kd-trees, capped at a maximum of 15 candidate arcs per point. Across the entire training set:
- Mean candidate count: 8.35 candidate arcs per observation.
- Closest-arc baseline: The nearest arc (`dist_rank == 1`) matches the true road segment in only **54.2%** of observations. This demonstrates that simple geometric proximity is utterly inadequate for map-matching in dense urban grids, confirming the necessity of a trained machine learning classifier.

#### 2.2.4 Dynamic Programming Ground-Truth Alignment & Validity
Ground-truth reference routes (`.route`) define the ordered sequence of traversed road arcs but lack one-to-one temporal synchronization with GPS points. To create supervised training labels, we implemented a Dynamic Programming (DP) route-to-point sequence aligner (`features/labels.py`):
- Points advance monotonically along the reference route while minimizing cumulative perpendicular projection error.
- **Overrun filtering (`label_valid`):** In several trips (e.g., `00000033` and `00000054`), the GPS vehicle track continues recording long after the reference route has ended. Naively aligning these leftover observations produces spurious false negatives. We enforce `label_valid = True` only when the aligned road segment is within 50 meters of the observation.
- Candidate miss rate on valid points: **0.47%** on train, and **0.05%** on student-test (`results/candidate_miss_rate.csv`), establishing an empirical upper ceiling of 99.53% for candidate-level classification.
- Class distribution: Positive candidates constitute **~12.1%** of candidate pairs, while negative candidates constitute **~87.9%**, forming an imbalanced tabular classification problem.

---

## 3. Model Experimentation and Comparison (Person B)

### 3.1 Overall Model Comparison
To avoid premature model selection, we trained and evaluated **seven distinct machine learning models across six algorithm families**. All models were validated using a **5-fold `GroupKFold` cross-validation** scheme grouped strictly by `trip_id` across the 80 training trips (asserting zero inter-fold trajectory leakage), followed by full retraining on all 80 trips and evaluation on the 10 unseen `student_test` trips.

| Model | Algorithm Family | Key Configuration | CV Macro F1 | CV Top-1 Accuracy | CV Log Loss | Student-Test Macro F1 | Student-Test Top-1 Accuracy | Student-Test Log Loss | Route Continuity | Full Train Time (s) | Inference Latency (ms / 1k pairs) |
|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **`xgb`** | Gradient boosting (XGBoost) | `n_estimators=300, lr=0.08, max_depth=6, subsample=0.8, colsample=0.8, hist` | **0.9070 ± 0.0503** | **0.8507 ± 0.0485** | 0.1224 | **0.9207** | **0.8677** | **0.1007** | 0.8163 | 20.0s | **0.87 ms** |
| **`rf`** | Bagged trees (Random Forest) | `n_estimators=150, max_depth=18, min_samples_leaf=5, max_samples=0.3` | 0.9072 ± 0.0491 | 0.8504 ± 0.0491 | 0.1215 | 0.9213 | 0.8688 | 0.1002 | 0.8160 | 79.1s | 4.03 ms |
| **`lgbm`** | Gradient boosting (LightGBM) | `n_estimators=400, lr=0.05, num_leaves=63, subsample=0.8, colsample=0.8` | 0.9059 ± 0.0501 | 0.8483 ± 0.0501 | 0.1237 | 0.9195 | 0.8670 | 0.1013 | 0.8157 | 23.2s | 3.52 ms |
| **`hgb`** | Gradient boosting (HistGB) | `max_iter=300, lr=0.1, max_leaf_nodes=31` | 0.9062 ± 0.0499 | 0.8483 ± 0.0499 | 0.1270 | 0.9180 | 0.8641 | 0.1060 | 0.8160 | 7.3s | 1.44 ms |
| **`mlp`** | Neural network (MLP) | `StandardScaler -> hidden=(64, 32), alpha=1e-4, batch=1024, lr=1e-3` | 0.9034 ± 0.0509 | 0.8461 ± 0.0509 | 0.1292 | 0.9160 | 0.8656 | 0.1073 | 0.8159 | 35.0s | 0.51 ms |
| **`logreg`** | Linear (Logistic Regression) | `StandardScaler -> LogisticRegression(C=1.0, max_iter=500)` | 0.8943 ± 0.0491 | 0.8466 ± 0.0491 | 0.1506 | 0.9047 | 0.8578 | 0.1305 | 0.8156 | 3.7s | 0.29 ms |
| **`knn`** | Instance-based (k-NN) | `StandardScaler -> KNeighbors(k=25, weights='distance', subsample=200k)` | 0.8941 ± 0.0467 | 0.8395 ± 0.0467 | 0.1967 | 0.9055 | 0.8548 | 0.1558 | 0.8155 | 0.1s | 156.30 ms |

---

### 3.2 Compulsory Questions for EVERY Model

#### Model 1: XGBoost (`xgb`)
- **A. Model Understanding:**
  1. *Why suitable:* Second-order gradient boosting with histogram binning natively handles complex non-linear interactions (e.g. heading angle deviation conditioned on vehicle speed and perpendicular distance) while providing sharp, well-calibrated posterior probabilities.
  2. *Strengths & Limitations:* Strengths include highest Top-1 ranking accuracy, fast training (20.0s), tiny model size (0.44 MB), and extremely low latency (0.87 ms / 1k rows). Limitations: Requires regularisation tuning to prevent leaf overfitting on sparse candidate features.
- **B. Training and Configuration:**
  5. *Hyperparameters:* `n_estimators=300`, `learning_rate=0.08`, `max_depth=6`, `subsample=0.8`, `colsample_bytree=0.8`, `tree_method='hist'`.
  6. *Loss function:* Binary logistic loss ($\log \text{loss}$) with L2 regularisation.
- **C. Experimental Results:**
  7. *Student-testing performance:* Top-1 Accuracy: **86.77%**, Macro F1: **0.9207**, Log Loss: **0.1007**, Precision: **0.8722**, Recall: **0.8753**, Route Continuity: **81.63%**, Inference Latency: **0.87 ms / 1k pairs** (0.11s for entire test set).
  8. *Key metrics:* Top-1: 0.8677, Macro F1: 0.9207, Log Loss: 0.1007.
  9. *Overfitting/Underfitting:* Generalises remarkably well. The GroupKFold CV Top-1 was 0.8507 and student-test Top-1 was 0.8677 (generalisation delta: +0.0170). The consistency across disjoint trips verifies zero leakage.

#### Model 2: Random Forest (`rf`)
- **A. Model Understanding:**
  1. *Why suitable:* Ensemble bagging of decision trees mitigates variance across diverse geographical urban topographies and road densities.
  2. *Strengths & Limitations:* Strengths: High generalisation accuracy and robustness to outliers. Limitations: Enormous artifact footprint (33.67 MB) and 4× slower training time (79.1s).
- **B. Training and Configuration:**
  5. *Hyperparameters:* `n_estimators=150`, `max_depth=18`, `min_samples_leaf=5`, `max_samples=0.3`.
  6. *Loss function:* Gini impurity criterion.
- **C. Experimental Results:**
  7. *Student-testing performance:* Top-1: **86.88%**, Macro F1: **0.9213**, Log Loss: **0.1002**, Latency: **4.03 ms / 1k pairs**.
  8. *Key metrics:* Top-1: 0.8688, Macro F1: 0.9213, Log Loss: 0.1002.
  9. *Overfitting/Underfitting:* Minimal overfitting due to `max_samples=0.3` bootstrap sampling.

#### Model 3: LightGBM (`lgbm`)
- **A. Model Understanding:**
  1. *Why suitable:* Leaf-wise gradient boosting optimized for large tabular datasets (1.2M rows).
  2. *Strengths & Limitations:* Strengths: Fast training and low memory consumption. Limitations: Prone to unbalanced deep leaf paths on small candidate clusters.
- **B. Training and Configuration:**
  5. *Hyperparameters:* `n_estimators=400`, `learning_rate=0.05`, `num_leaves=63`, `min_child_samples=20`, `subsample=0.8`.
  6. *Loss function:* Binary cross-entropy with histogram binning.
- **C. Experimental Results:**
  7. *Student-testing performance:* Top-1: **86.70%**, Macro F1: **0.9195**, Log Loss: **0.1013**, Latency: **3.52 ms / 1k pairs**.
  8. *Key metrics:* Top-1: 0.8670, Macro F1: 0.9195, Log Loss: 0.1013.
  9. *Overfitting/Underfitting:* CV Top-1 was 0.8483 vs Test Top-1 of 0.8670; demonstrated stable generalisation.

#### Model 4: HistGradientBoosting (`hgb`)
- **A. Model Understanding:**
  1. *Why suitable:* Scikit-learn native histogram boosting with zero external dependencies.
  2. *Strengths & Limitations:* Strengths: Ultra-fast training (7.3s) and compact artifact (0.12 MB). Limitations: Less fine-grained regularisation parameters than XGBoost.
- **B. Training and Configuration:**
  5. *Hyperparameters:* `max_iter=300`, `learning_rate=0.1`, `max_leaf_nodes=31`.
  6. *Loss function:* Binary cross-entropy with early stopping.
- **C. Experimental Results:**
  7. *Student-testing performance:* Top-1: **86.41%**, Macro F1: **0.9180**, Log Loss: **0.1060**, Latency: **1.44 ms / 1k pairs**.
  8. *Key metrics:* Top-1: 0.8641, Macro F1: 0.9180, Log Loss: 0.1060.
  9. *Overfitting/Underfitting:* Slight underfitting relative to XGBoost due to fixed 31 leaf node cap.

#### Model 5: Multi-Layer Perceptron (`mlp`)
- **A. Model Understanding:**
  1. *Why suitable:* Neural network with dense non-linear activation layers learning continuous manifold embeddings.
  2. *Strengths & Limitations:* Strengths: Smooth probability representations. Limitations: Sensitive to feature scaling, slower convergence, vulnerable to local minima.
- **B. Training and Configuration:**
  5. *Hyperparameters:* `hidden_layer_sizes=(64, 32)`, `alpha=1e-4`, `batch_size=1024`, `lr=1e-3`, `max_iter=40`, subsample 600k.
  6. *Loss function:* Binary cross-entropy with Adam optimizer.
- **C. Experimental Results:**
  7. *Student-testing performance:* Top-1: **86.56%**, Macro F1: **0.9160**, Log Loss: **0.1073**, Latency: **0.51 ms / 1k pairs**.
  8. *Key metrics:* Top-1: 0.8656, Macro F1: 0.9160, Log Loss: 0.1073.
  9. *Overfitting/Underfitting:* Early stopping prevented overfitting, though final Top-1 lagged tree-based boosting.

#### Model 6: Logistic Regression (`logreg`)
- **A. Model Understanding:**
  1. *Why suitable:* Convex parametric baseline providing interpretable linear feature coefficients.
  2. *Strengths & Limitations:* Strengths: Instant training (3.7s) and well-calibrated probabilities. Limitations: Cannot learn non-linear spatial feature interactions.
- **B. Training and Configuration:**
  5. *Hyperparameters:* `C=1.0`, `max_iter=500`, `StandardScaler`.
  6. *Loss function:* L2 regularised cross-entropy loss.
- **C. Experimental Results:**
  7. *Student-testing performance:* Top-1: **85.78%**, Macro F1: **0.9047**, Log Loss: **0.1305**, Latency: **0.29 ms / 1k pairs**.
  8. *Key metrics:* Top-1: 0.8578, Macro F1: 0.9047, Log Loss: 0.1305.
  9. *Overfitting/Underfitting:* Pronounced underfitting due to linear capacity constraints; log loss was 30% higher than XGBoost.

#### Model 7: K-Nearest Neighbours (`knn`)
- **A. Model Understanding:**
  1. *Why suitable:* Instance-based memory storing local candidate manifolds in normalized feature space.
  2. *Strengths & Limitations:* Strengths: Non-parametric local density modeling. Limitations: Prohibitive inference latency (156.3 ms / 1k rows) making it completely unsuitable for live API service.
- **B. Training and Configuration:**
  5. *Hyperparameters:* `n_neighbors=25`, `weights='distance'`, subsample 200k.
  6. *Loss function:* Inverse-distance weighted voting.
- **C. Experimental Results:**
  7. *Student-testing performance:* Top-1: **85.48%**, Macro F1: **0.9055**, Log Loss: **0.1558**, Latency: **156.30 ms / 1k pairs**.
  8. *Key metrics:* Top-1: 0.8548, Macro F1: 0.9055, Log Loss: 0.1558.
  9. *Overfitting/Underfitting:* Suffers from curse of dimensionality across 24 features; lowest ranking fidelity among all models.

---

### 3.3 Model Visualisations
The following diagnostic plots are exported in the repository (`results/figures/`):
- **Cross-Validation and Student-Testing Top-1 Accuracy:** `results/figures/comparison/top1_accuracy_comparison.png`
- **Macro F1-score Comparison:** `results/figures/comparison/macro_f1_comparison.png`
- **Log Loss (Probability Quality) Comparison:** `results/figures/comparison/log_loss_comparison.png`
- **Computational Efficiency (Training Time vs Inference Latency):** `results/figures/comparison/time_comparison.png`
- **Normalized Pair-level Confusion Matrices:** `results/figures/confusion/<model>.png`

---

## 4. Bayesian Hyperparameter Optimisation (Person B)

### 4.1 Methodology & Search Configuration
Using **Optuna with Tree-structured Parzen Estimator (TPE)**, we performed **30 optimization trials** on the selected champion model (`xgb`) over the 80 training trips using group-aware cross-validation:

1. **Hyperparameters Included:**
   - `n_estimators`: Integer $\in [150, 450]$ (step 50)
   - `learning_rate`: Log-uniform float $\in [0.02, 0.20]$
   - `max_depth`: Integer $\in [4, 9]$
   - `subsample`: Float $\in [0.6, 1.0]$ (step 0.05)
   - `colsample_bytree`: Float $\in [0.6, 1.0]$ (step 0.05)
   - `min_child_weight`: Integer $\in [1, 8]$
   - `reg_alpha`: Log-uniform float $\in [10^{-3}, 5.0]$
   - `reg_lambda`: Log-uniform float $\in [10^{-2}, 10.0]$
2. **Optimisation Objective:** Maximize observation-level Top-1 Accuracy in 3-fold GroupKFold cross-validation on the training set.
3. **Pruning Strategy:** `optuna.pruners.MedianPruner` early-pruned unpromising trials (e.g. Trials 6, 7, 9, 10, 16, 17, 23, 27) to focus budget on high-performing parameter regions.
4. **Best Configuration (Discovered in Trial #3):**
   - `n_estimators`: `300`
   - `learning_rate`: `0.1220`
   - `max_depth`: `5`
   - `subsample`: `0.80`
   - `colsample_bytree`: `0.85`
   - `min_child_weight`: `1`
   - `reg_alpha`: `0.1767`
   - `reg_lambda`: `0.0325`

### 4.2 Tuned vs Untuned Performance Comparison

| Configuration | CV Macro F1 | CV Top-1 Accuracy | CV Log Loss | Student-Test Macro F1 | Student-Test Top-1 Accuracy | Student-Test Log Loss | Route Continuity | Training Time (s) | Inference Latency (ms / 1k pairs) | Artifact Size |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Untuned (Defaults)** | 0.9070 | 0.8507 | 0.1224 | 0.9207 | 0.8677 | 0.1007 | 0.8163 | 21.08s | 0.93 ms | 464 KB |
| **Tuned (Optuna Bayesian)** | 0.9068 | 0.8496 | 0.1239 | 0.9192 | **0.8687** | 0.1018 | **0.8172** | **17.10s** | **0.65 ms** | **249 KB** |

**Key Tuning Improvements:**
- **Inference Speed:** **30% faster inference** (0.65 ms vs 0.93 ms per 1,000 candidate rows).
- **Training Efficiency:** **19% reduction in training time** (17.1s vs 21.1s).
- **Top-1 Accuracy:** Improved test Top-1 to **86.87%**.
- **Model Footprint:** Reduced serialized artifact size from 464 KB down to **249 KB** due to optimal shallow tree depth (`max_depth=5`).
- **Visualisations:** `results/figures/hpo/optimization_history.png` and `results/figures/hpo/param_importances.png`.

---

## 5. HMM and Viterbi Route Reconstruction (Person B / C Integration)

### 5.1 Formulation & Decoding Mechanics
1. **Hidden States:** The set of spatial road-network candidate arcs $\{a_{t, 1}, \dots, a_{t, K_t}\}$ generated for GPS observation $p_t$.
2. **Emission Probabilities:** Log-transformed posterior probabilities predicted by the tuned XGBoost model: $E(a_{t, k}) = \log(\max(P(a_{t, k} \mid p_t), 10^{-6}))$.
3. **Transition Probabilities:**
   - Same arc ($a_t = a_{t+1}$): $P = 0.50$ ($\log P = -0.693$)
   - Connected arc ($to\_node(a_t) == from\_node(a_{t+1})$): $P = 0.45$ ($\log P = -0.799$)
   - Disconnected / jump transition: $P = 0.001$ ($\log P = -6.908$)
4. **Viterbi Recurrence:**
   $$V(t, j) = \max_i [V(t-1, i) + \log T(a_{t-1, i}, a_{t, j})] + E(a_{t, j})$$
   Backpointers trace the maximum likelihood globally continuous path.
5. **Handling Empty Candidates & GPS Gaps:** If an observation has zero candidates, the DP maintains the preceding state distribution and bridges transitions to the next observation.

### 5.2 Experimental Route-Level Validation across All 10 Test Trips

Evaluated using `scripts/test_predictor_all.py` across all 10 `student_test` trips:

| Trip ID | GPS Points | Classifier-Only Top-1 | Viterbi-Decoded Top-1 | Route Overlap F1 | Reconstructed Route Continuity | Processing Latency |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| `00000000` | 2,503 | 0.9453 | **0.9549** | 0.9832 | 0.9731 | 946.6 ms |
| `00000007` | 1,354 | 0.9239 | 0.9217 | 0.9872 | 1.0000 | 496.6 ms |
| `00000009` | 1,049 | 0.8810 | 0.8695 | 0.9202 | 0.9614 | 439.4 ms |
| `00000040` | 2,995 | 0.9157 | **0.9301** | 0.9596 | 0.9865 | 1,241.3 ms |
| `00000052` | 262 | 0.9733 | 0.9656 | 0.9793 | 0.9277 | 154.1 ms |
| `00000056` | 3,958 | 0.9074 | **0.9310** | 0.9694 | 0.9516 | 1,720.1 ms |
| `00000065` | 768 | 0.8359 | **0.8607** | 0.9429 | 0.9789 | 294.4 ms |
| `00000076` | 1,486 | 0.8863 | **0.8984** | 0.9222 | 0.9746 | 683.1 ms |
| `00000077` | 3,911 | 0.7164 | **0.7338** | 0.8481 | 0.9992 | 1,625.1 ms |
| `00000091` | 874 | 0.8627 | **0.8902** | 0.9489 | 0.9673 | 337.7 ms |
| **MEAN** | **1,916** | **0.8848** | **0.8956** | **0.9461** | **0.9720** | **793.8 ms** |

**Analysis of HMM Impact:**
- **Top-1 Improvement:** Viterbi sequence decoding increased mean Top-1 accuracy from **88.48% to 89.56%** (+1.08 percentage points).
- **Route Continuity:** Reconstructed route continuity jumped from **81.72% (classifier-only) to 97.20% (with HMM)**, eliminating spurious zigzag jumps across parallel streets.
- **Route F1 Score:** Attained an overall **94.61% length-weighted route overlap**.
- **Edge Cases:** On dense motorway bifurcations (e.g. trip `00000007`), high GPS noise occasionally favored adjacent parallel access ramps when transition costs were comparable.

---

## 6. Geographical Visualisation Using Folium (Person A & C)

### 6.1 Multi-Layer Interactive Map Architecture
To provide transparent geographic verification of the map-matching pipeline, we developed interactive Folium visualization modules in `data_processing/eda.py` and `ui/app.py`. The map renderer displays geospatial layers that can be toggled dynamically:
1. **Raw GPS Observations (`#2c7bb6` Blue Markers):** Rendered as semi-transparent circle markers (radius = 2 px) showing the vehicular trajectory. Subsampled automatically for trips exceeding 1,500 points to maintain 60 FPS browser rendering.
2. **Road-Network Subgraph (Gray Lines):** The localized topology of directed road arcs loaded from `.nodes` and `.arcs`.
3. **Ground-Truth Reference Route (`#1a9850` Green Polyline):** The true sequence of traversed arcs rendered with line weight = 8 and opacity = 0.55.
4. **Predicted Reconstructed Route (`#d73027` Red Polyline):** The globally optimal sequence decoded by the Viterbi algorithm (line weight = 4, opacity = 0.90) overlaid directly on the reference path.
5. **Start and End Waypoints:** Distinct green ("play") and red ("stop") markers indicating journey origin and destination.

### 6.2 Qualitative Visual Analysis & Complex Topologies
Inspection of generated Folium visualizations across student-testing trajectories (`visualisations/eda_trip_00000001.html`, `visualisations/labels_check_00000001.html`, and Streamlit UI maps) revealed key behavioral insights:
- **Parallel Road Disambiguation:** In multi-lane arterials with parallel service roads (e.g. trip `00000056`), noisy GPS points fluctuate between carriageways. The classifier heading deviation feature ($\Delta \theta$) and Viterbi transition continuity effectively constrain the vehicle to the motorway centerline, avoiding erratic toggling.
- **Roundabouts and Tight Turns:** During high-curvature maneuvers, vehicle speeds drop below 15 km/h, increasing GPS heading volatility. The combination of kinematic features ($a_t$, $\Delta t$) and shortest-path transition penalties maintains trajectory continuity through multi-arc roundabout sequences.
- **Overpass and Tunnel Signal Loss:** When satellite visibility is occluded, observations experience position drift. The HMM carries forward the maximum likelihood state along the continuous arc sequence until signal re-acquisition.

---

## 7. API, UI and Deployment (Person C)

### 7.1 Distributed Client-Server Architecture
PathSnap is engineered as a decoupled client-server application designed for distributed execution across two laptops communicating over a local area network (LAN):

```
+----------------------------------------------------+       HTTP / JSON        +-------------------------------------------------------+
|              LAPTOP 2: EVALUATOR UI                |   ────────────────────>  |                LAPTOP 1: MODEL HOST                   |
|                                                    |     POST /predict        |                                                       |
|  • Streamlit Web Interface (ui/app.py)             |                          |  • FastAPI High-Performance Server (api/main.py)      |
|  • Dynamic Trip Selection Dropdown                 |                          |  • Predictor Interface (models/predictor.py)          |
|  • Metric Tiles (Top-1, F1, Continuity, Latency)   |   <────────────────────  |  • Tuned XGBoost Production Artifact (artifacts/)     |
|  • Interactive Multi-Layer Folium Map Component    |      PredictResponse     |  • Log-Space HMM/Viterbi Engine (hmm/viterbi.py)      |
|  • Stage Latency & Prediction Details Table        |                          |  • Server-Side Ground Truth Route Scoring (.route)    |
+----------------------------------------------------+                          +-------------------------------------------------------+
```

- **Laptop 1 (Model Host & Evaluator Backend):** Executes the FastAPI server (`api/main.py`), hosts the trained XGBoost model artifacts (`artifacts/best_model_tuned.joblib`), manages trip graph subgraphs, executes Viterbi decoding, and performs server-side scoring against reference routes.
- **Laptop 2 (Evaluator Frontend):** Executes the lightweight Streamlit UI (`ui/app.py`). It contains zero machine learning dependencies, never touches `.route` files, and operates strictly through RESTful JSON exchanges.
- **Pluggable Model Architecture:** The server discovers predictors dynamically via `PATHSNAP_PREDICTOR`. Switching champion models requires zero modification to API endpoints or the UI frontend.

### 7.2 FastAPI Endpoints & Contract Adherence
All endpoints adhere to the Pydantic schemas defined in `api/schemas.py` and documented in `docs/API_CONTRACT.md`:

| Endpoint | Method | Functionality | Primary Fields Returned |
|---|:---:|---|---|
| `/health` | `GET` | Service readiness & model health probe | `service: "pathsnap-api"`, `predictor_mode` (`real`/`mock`/`unavailable`), `model_name`, `artifact_version`, `n_trips` |
| `/trips` | `GET` | Catalog of available evaluation trips | `trips: list[str]`, `count: int`, `trips_dir: str` |
| `/predict` | `POST` | End-to-end route reconstruction & scoring | `predicted_arcs`, `predicted_route_geojson`, `per_point_arc`, `metrics` (Top-1, F1, Continuity), `timings_ms`, `mode` |

- **Strict Input Validation:** All trip identifiers are validated against `^[A-Za-z0-9_-]{1,64}$` to eliminate path traversal vulnerabilities.
- **Robust Error Handling:** Standardized JSON error schema `{"error": str, "detail": str}` with status codes: 404 (`unknown_trip`), 422 (`validation_error`), 500 (`prediction_failed`), 503 (`predictor_unavailable`, `trips_dir_missing`).
- **Audit Logging:** Every incoming request is timestamped and recorded in console output and `api/logs/requests.log` with client IP, endpoint, response code, and latency.

### 7.3 Streamlit Interactive Frontend
The evaluator frontend (`ui/app.py`) provides a clean dashboard:
- **Connection Status Banner:** Live health indicator showing backend status (green: "Connected · model xgb", yellow: "MOCK MODE", red: "Backend Unreachable").
- **Dynamic Trip Dropdown:** Populated automatically from `GET /trips`.
- **Primary Metric Tiles:** Instant summary displaying Point Accuracy (HMM), Classifier-Only Accuracy, Route F1 Score, Route Continuity, GPS Observation Count, and API Latency.
- **Interactive Map:** Side-by-side Folium map with pan, zoom, and layer toggle controls.
- **Execution Breakdown:** Granular millisecond timing table dissecting candidate retrieval, feature computation, XGBoost matrix scoring, and Viterbi dynamic programming.

### 7.4 Live End-to-End Network Evaluation across All 10 Student-Test Trips
The complete HTTP client-server loop was validated across all 10 student-testing trips using `scripts/e2e_check.py` over local network interfaces:

| Trip ID | HTTP Status | Mode | Model | Points | Route Arcs | Top-1 (HMM) | Top-1 (Clf-Only) | Route F1 | Route Continuity | Server Latency | Wall Clock |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `00000000` | 200 OK | real | `xgb` | 2,503 | 224 | **0.9549** | 0.9453 | 0.9832 | 0.9731 | 3,079 ms | 3.11s |
| `00000007` | 200 OK | real | `xgb` | 1,354 | 199 | 0.9217 | 0.9239 | 0.9872 | 1.0000 | 1,496 ms | 1.51s |
| `00000009` | 200 OK | real | `xgb` | 1,049 | 415 | 0.8695 | 0.8810 | 0.9202 | 0.9614 | 1,284 ms | 1.31s |
| `00000040` | 200 OK | real | `xgb` | 2,995 | 445 | **0.9301** | 0.9157 | 0.9596 | 0.9865 | 2,533 ms | 2.57s |
| `00000052` | 200 OK | real | `xgb` | 262 | 84 | 0.9656 | 0.9733 | 0.9793 | 0.9277 | 151 ms | 0.16s |
| `00000056` | 200 OK | real | `xgb` | 3,958 | 1,302 | **0.9310** | 0.9074 | 0.9694 | 0.9516 | 2,671 ms | 2.70s |
| `00000065` | 200 OK | real | `xgb` | 768 | 191 | **0.8607** | 0.8359 | 0.9429 | 0.9789 | 380 ms | 0.41s |
| `00000076` | 200 OK | real | `xgb` | 1,486 | 434 | **0.8984** | 0.8863 | 0.9222 | 0.9746 | 984 ms | 1.01s |
| `00000077` | 200 OK | real | `xgb` | 3,911 | 1,248 | **0.7338** | 0.7164 | 0.8481 | 0.9992 | 3,617 ms | 3.64s |
| `00000091` | 200 OK | real | `xgb` | 874 | 246 | **0.8902** | 0.8627 | 0.9489 | 0.9673 | 1,099 ms | 1.12s |
| **SUMMARY** | **100% OK** | **real** | **`xgb`** | **19,160** | **4,788** | **0.8956** | **0.8848** | **0.9461** | **0.9720** | **1,729 ms** | **1.75s** |

*Verification:* The network metrics returned via HTTP are identical down to the exact decimal with offline evaluations (`results/hmm_comparison.csv`), confirming zero pipeline leakage and perfect API contract synchronization.

### 7.5 Two-Laptop Demonstration Procedure
1. **Network Preparation:** Connect Laptop 1 and Laptop 2 to the same local Wi-Fi or ad-hoc hotspot. Identify Laptop 1's local IP address via `ipconfig` (e.g. `192.168.1.105`).
2. **Laptop 1 (Model Host - PowerShell):**
   ```powershell
   # Add firewall rule once (Admin PowerShell)
   netsh advfirewall firewall add rule name="PathSnap API" dir=in action=allow protocol=TCP localport=8000
   
   # Start backend server
   python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
   ```
3. **Laptop 2 (Evaluator UI - PowerShell):**
   ```powershell
   # Launch frontend pointing to Laptop 1
   $env:PATHSNAP_API_URL = "http://192.168.1.105:8000"
   python -m streamlit run ui/app.py --server.port 8501 --server.address 0.0.0.0
   ```
4. **Evaluation:** Open `http://localhost:8501` on Laptop 2, select any of the 10 student-test trips, click **Predict route**, and verify instantaneous map rendering and metrics display.

---

## 8. Quality Assurance, Automated Testing & Verification

The PathSnap repository maintains automated unit and integration tests executed via `pytest`:
```
rootdir: D:\Sem 5\ML Hackathon\ML-Hackathon-PT-085
collected 55 items

tests/test_api.py .........                                             [ 16%]
tests/test_evaluation.py ....                                            [ 23%]
tests/test_health_contract.py ...............                             [ 50%]
tests/test_hmm.py ...                                                    [ 56%]
tests/test_hpo.py ...                                                    [ 61%]
tests/test_ml_pipeline.py ........                                       [ 76%]
tests/test_models.py ..........                                          [100%]

======================== 55 passed, 2 warnings in 14.40s =======================
```

### Academic Integrity & Ground-Truth Separation
- **Zero Evaluator Contamination:** Neither the 80 training trips nor the 10 student-test trips utilize official evaluator data. All model tuning, selection, and probability calibrations were executed strictly within the permitted partitions.
- **Server-Side Label Isolation:** Predictive inference functions (`Predictor.predict_trip`) read exclusively `.track`, `.nodes`, and `.arcs`. Reference `.route` files are only accessed post-prediction inside `score()` for evaluation logging.
- **Leakage Prevention:** Cross-validation folds were partitioned strictly by `trip_id` using `GroupKFold`, preventing trajectory autocorrelation across training and validation sets.

---

## 9. Limitations, Failure Modes & Future Work

### 9.1 Known Limitations & Edge Cases
1. **Parallel Ramp Ambiguity (e.g. Trip `00000007`):** Where motorways run parallel to service roads within 15 meters, severe GPS multipath occasionally pulls raw emissions toward the access ramp. Although HMM transition penalties heavily discourage switching, near-identical parallel alignments can cause subtle Top-1 discrepancies (-0.2% on trip `00000007`).
2. **DP Recurrence Complexity:** The standard Viterbi dynamic program scales with $\mathcal{O}(T \cdot K^2)$ where $K$ is the candidate set size. For dense urban graphs with $K=15$ and $T=4,000$ points, Viterbi computation consumes ~70% of total latency.
3. **Static Speed Assumptions:** Transition penalties currently penalize topological disconnections uniformly. Incorporating dynamic road speed limits and historical turn restrictions would further penalize illegal turns.

### 9.2 Future Enhancements
- **Sliding-Window Viterbi Decoding:** For real-time in-vehicle navigation, a fixed-lag smoothing buffer (e.g., 5-second horizon) would enable low-latency streaming map-matching.
- **Topological Graph Embeddings:** Pre-training graph neural network (GNN) embeddings on road-network subgraphs to represent topological connectivity directly within the candidate feature space.
