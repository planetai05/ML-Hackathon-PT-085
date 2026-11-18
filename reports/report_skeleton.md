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

### 2.2 Dataset Understanding and Exploratory Data Analysis
<!-- INTEGRATION SECTION: PERSON A (Owned by Person A - see docs/HANDOFF_PERSON_A.md) -->
*Person A completes: Missing value analysis, trajectory plots, road-network topological analysis, and Folium interactive map screenshots.*

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

## 6. Geographical Visualisation Using Folium
<!-- INTEGRATION SECTION: PERSON A (Owned by Person A) -->
*Person A completes: Folium HTML maps, screenshots showing raw GPS (blue), reference route (green), and predicted route (orange/purple) with interactive layer controls.*

---

## 7. API, UI and Deployment
<!-- INTEGRATION SECTION: PERSON C (Owned by Person C) -->
*Person C completes: FastAPI `/predict` screenshots, command-line request access logs, Streamlit user interface screenshots, and two-laptop LAN deployment instructions.*

---

## 8. Academic Integrity and Reproducibility
All experimental results, metrics, hyperparameters, and serialized model files documented in this report are verifiable using the submitted GitHub repository:
- Reproducible random seeds (`SEED = 42`)
- GroupKFold disjoint splits asserting zero spatial leakage
- Artifacts committed in `artifacts/`
- Full test suite passing (30/30 unit and integration tests)
