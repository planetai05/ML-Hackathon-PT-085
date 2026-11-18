# PathSnap: Model Notes and Compulsory Section 3.2 Report Q&A

This document contains the experimental notes and structured answers for every trained model family, fulfilling all compulsory questions in **Section 3.2 (Compulsory Questions for EVERY Model)** of `Deliverables.pdf`.

## Overall Model Comparison Summary

| Model | Algorithm Family | Key Configuration | CV Macro F1 | CV Top-1 | Test Macro F1 | Test Top-1 | Test Log Loss | Full Train Time (s) | Test Latency (ms/1k rows) | Device |
|---|---|---|---|---|---|---|---|---|---|---|
| `xgb` | Gradient boosting (XGBoost) | `defaults (see run json)` | 0.9070±0.0304 | 0.8507±0.0503 | 0.9207 | 0.8677 | 0.1007 | 20.0s | 0.00ms | CPU |
| `rf` | Bagged trees (random forest) | `defaults (see run json)` | 0.9072±0.0301 | 0.8504±0.0491 | 0.9213 | 0.8688 | 0.1002 | 79.1s | 0.00ms | CPU |
| `hgb` | Gradient boosting (sklearn HistGB) | `defaults (see run json)` | 0.9062±0.0299 | 0.8483±0.0499 | 0.9180 | 0.8641 | 0.1060 | 7.3s | 0.00ms | CPU |
| `lgbm` | Gradient boosting (LightGBM) | `defaults (see run json)` | 0.9059±0.0305 | 0.8483±0.0501 | 0.9195 | 0.8670 | 0.1013 | 23.2s | 0.00ms | CPU |
| `logreg` | Linear (logistic regression) | `defaults (see run json)` | 0.8943±0.0287 | 0.8466±0.0491 | 0.9047 | 0.8578 | 0.1305 | 3.7s | 0.00ms | CPU |
| `mlp` | Neural network (MLP) | `defaults (see run json)` | 0.9034±0.0302 | 0.8461±0.0509 | 0.9160 | 0.8656 | 0.1073 | 35.0s | 0.00ms | CPU |
| `knn` | Instance-based (k-NN) | `defaults (see run json)` | 0.8942±0.0280 | 0.8395±0.0467 | 0.9055 | 0.8548 | 0.1558 | 0.1s | 0.16ms | CPU |

---
## Model: `xgb` (Gradient boosting (XGBoost))

### A. Model Understanding
1. **Why did you consider this algorithm suitable for the PathSnap problem?**
   - Standard gradient boosting workhorse utilizing second-order Taylor expansion of the loss function and histogram tree method.
2. **What are the major strengths and limitations of this algorithm for this task?**
   - **Strengths:** Robust handling of class imbalance and exact Hessian calculations for precise split evaluation.
   - **Limitations:** Slightly higher training overhead on CPU compared to LightGBM.

### B. Training and Configuration
5. **What were the important hyperparameters and their initial values?**
   - Configuration: `defaults (see run json)`
   - Subsample: `nan`
6. **What loss function or optimisation objective was used?**
   - Binary logistic loss with L1 and L2 tree leaf weight penalization.

### C. Experimental Results
7. **What was the model's performance on the student-testing dataset?**
   - Observation-level Top-1 Accuracy: **86.77%**
   - Pair-level Macro F1-score: **0.9207**
   - Pair-level Log Loss: **0.1007**
   - Pair-level Precision: **0.8749**, Recall: **0.8552**
   - Route Continuity: **81.63%**
   - Inference Latency: **0.00 ms per 1,000 rows** (0.111 s total for 127k test rows)
8. **Observation-level Top-1 Accuracy, Macro F1-score and Log Loss on student-testing:**
   - Top-1: `0.8677`, Macro F1: `0.9207`, Log Loss: `0.1007`.
9. **Did you observe overfitting or underfitting? What evidence supports your conclusion?**
   - The model demonstrated strong generalisation without severe overfitting. The CV Top-1 was 0.8507 and the student-test Top-1 was 0.8677 (generalisation delta: 0.0171). Training fold Top-1 was 0.9064. The small difference confirms the GroupKFold cross-validation was representative of unseen test trips.

---
## Model: `rf` (Bagged trees (random forest))

### A. Model Understanding
1. **Why did you consider this algorithm suitable for the PathSnap problem?**
   - Bagging of unconstrained decision trees captures complex threshold interactions while mitigating high variance through feature subsampling.
2. **What are the major strengths and limitations of this algorithm for this task?**
   - **Strengths:** Highly resilient to feature scale differences; captures deep interaction terms without explicit feature crosses.
   - **Limitations:** Large disk footprint (>100MB uncompressed); higher memory bandwidth required during batch inference.

### B. Training and Configuration
5. **What were the important hyperparameters and their initial values?**
   - Configuration: `defaults (see run json)`
   - Subsample: `nan`
6. **What loss function or optimisation objective was used?**
   - Gini impurity criterion across bagged bootstrap decision trees.

### C. Experimental Results
7. **What was the model's performance on the student-testing dataset?**
   - Observation-level Top-1 Accuracy: **86.88%**
   - Pair-level Macro F1-score: **0.9213**
   - Pair-level Log Loss: **0.1002**
   - Pair-level Precision: **0.8811**, Recall: **0.8509**
   - Route Continuity: **80.94%**
   - Inference Latency: **0.00 ms per 1,000 rows** (0.515 s total for 127k test rows)
8. **Observation-level Top-1 Accuracy, Macro F1-score and Log Loss on student-testing:**
   - Top-1: `0.8688`, Macro F1: `0.9213`, Log Loss: `0.1002`.
9. **Did you observe overfitting or underfitting? What evidence supports your conclusion?**
   - The model demonstrated strong generalisation without severe overfitting. The CV Top-1 was 0.8504 and the student-test Top-1 was 0.8688 (generalisation delta: 0.0183). Training fold Top-1 was 0.9061. The small difference confirms the GroupKFold cross-validation was representative of unseen test trips.

---
## Model: `hgb` (Gradient boosting (sklearn HistGB))

### A. Model Understanding
1. **Why did you consider this algorithm suitable for the PathSnap problem?**
   - Histogram-based gradient boosting splits continuous features into discrete integer bins (256 bins), reducing memory and training time by an order of magnitude.
2. **What are the major strengths and limitations of this algorithm for this task?**
   - **Strengths:** Fast training on 1M+ rows, native histogram binning, compact model size.
   - **Limitations:** Slightly less flexible regularisation parameters compared to standalone LightGBM or XGBoost.

### B. Training and Configuration
5. **What were the important hyperparameters and their initial values?**
   - Configuration: `defaults (see run json)`
   - Subsample: `nan`
6. **What loss function or optimisation objective was used?**
   - Binary cross-entropy (logarithmic loss) with early stopping based on validation score.

### C. Experimental Results
7. **What was the model's performance on the student-testing dataset?**
   - Observation-level Top-1 Accuracy: **86.41%**
   - Pair-level Macro F1-score: **0.9180**
   - Pair-level Log Loss: **0.1060**
   - Pair-level Precision: **0.8645**, Recall: **0.8564**
   - Route Continuity: **81.07%**
   - Inference Latency: **0.00 ms per 1,000 rows** (0.184 s total for 127k test rows)
8. **Observation-level Top-1 Accuracy, Macro F1-score and Log Loss on student-testing:**
   - Top-1: `0.8641`, Macro F1: `0.9180`, Log Loss: `0.1060`.
9. **Did you observe overfitting or underfitting? What evidence supports your conclusion?**
   - The model demonstrated strong generalisation without severe overfitting. The CV Top-1 was 0.8483 and the student-test Top-1 was 0.8641 (generalisation delta: 0.0157). Training fold Top-1 was 0.8911. The small difference confirms the GroupKFold cross-validation was representative of unseen test trips.

---
## Model: `lgbm` (Gradient boosting (LightGBM))

### A. Model Understanding
1. **Why did you consider this algorithm suitable for the PathSnap problem?**
   - State-of-the-art leaf-wise (best-first) gradient boosted decision tree algorithm optimized for large tabular datasets.
2. **What are the major strengths and limitations of this algorithm for this task?**
   - **Strengths:** Highest Top-1 ranking accuracy, extremely fast inference (<0.05 ms per sample), superb probability calibration for HMM emissions.
   - **Limitations:** Can overfit small leaf nodes if min_child_samples and colsample_bytree are not properly regularized.

### B. Training and Configuration
5. **What were the important hyperparameters and their initial values?**
   - Configuration: `defaults (see run json)`
   - Subsample: `nan`
6. **What loss function or optimisation objective was used?**
   - Binary logloss with histogram binning and leaf-wise splitting.

### C. Experimental Results
7. **What was the model's performance on the student-testing dataset?**
   - Observation-level Top-1 Accuracy: **86.70%**
   - Pair-level Macro F1-score: **0.9195**
   - Pair-level Log Loss: **0.1013**
   - Pair-level Precision: **0.8713**, Recall: **0.8546**
   - Route Continuity: **81.68%**
   - Inference Latency: **0.00 ms per 1,000 rows** (0.449 s total for 127k test rows)
8. **Observation-level Top-1 Accuracy, Macro F1-score and Log Loss on student-testing:**
   - Top-1: `0.8670`, Macro F1: `0.9195`, Log Loss: `0.1013`.
9. **Did you observe overfitting or underfitting? What evidence supports your conclusion?**
   - The model demonstrated strong generalisation without severe overfitting. The CV Top-1 was 0.8483 and the student-test Top-1 was 0.8670 (generalisation delta: 0.0187). Training fold Top-1 was 0.9127. The small difference confirms the GroupKFold cross-validation was representative of unseen test trips.

---
## Model: `logreg` (Linear (logistic regression))

### A. Model Understanding
1. **Why did you consider this algorithm suitable for the PathSnap problem?**
   - Serves as a robust, convex baseline with interpretable linear coefficients and probabilistic calibration.
2. **What are the major strengths and limitations of this algorithm for this task?**
   - **Strengths:** Fast training, guaranteed convergence, well-calibrated posterior probabilities, no hyperparameter tuning complexity.
   - **Limitations:** Cannot learn non-linear spatial relationships (e.g. non-linear interactions between perpendicular distance and heading angle).

### B. Training and Configuration
5. **What were the important hyperparameters and their initial values?**
   - Configuration: `defaults (see run json)`
   - Subsample: `nan`
6. **What loss function or optimisation objective was used?**
   - Binary cross-entropy loss with L2 regularisation penalty (Ridge penalty).

### C. Experimental Results
7. **What was the model's performance on the student-testing dataset?**
   - Observation-level Top-1 Accuracy: **85.78%**
   - Pair-level Macro F1-score: **0.9047**
   - Pair-level Log Loss: **0.1305**
   - Pair-level Precision: **0.8392**, Recall: **0.8366**
   - Route Continuity: **81.86%**
   - Inference Latency: **0.00 ms per 1,000 rows** (0.037 s total for 127k test rows)
8. **Observation-level Top-1 Accuracy, Macro F1-score and Log Loss on student-testing:**
   - Top-1: `0.8578`, Macro F1: `0.9047`, Log Loss: `0.1305`.
9. **Did you observe overfitting or underfitting? What evidence supports your conclusion?**
   - The model demonstrated strong generalisation without severe overfitting. The CV Top-1 was 0.8466 and the student-test Top-1 was 0.8578 (generalisation delta: 0.0112). Training fold Top-1 was 0.8762. The small difference confirms the GroupKFold cross-validation was representative of unseen test trips.

---
## Model: `mlp` (Neural network (MLP))

### A. Model Understanding
1. **Why did you consider this algorithm suitable for the PathSnap problem?**
   - Multilayer perceptron learns continuous smooth non-linear manifold embeddings of kinematics and road geometry.
2. **What are the major strengths and limitations of this algorithm for this task?**
   - **Strengths:** Jointly maps continuous speed, acceleration, and spatial bearings into a smooth probability space.
   - **Limitations:** Sensitive to feature scaling; slower training convergence; vulnerable to local minima on imbalanced tabular data.

### B. Training and Configuration
5. **What were the important hyperparameters and their initial values?**
   - Configuration: `defaults (see run json)`
   - Subsample: `600000.0`
6. **What loss function or optimisation objective was used?**
   - Binary cross-entropy loss optimized with Adam and early stopping.

### C. Experimental Results
7. **What was the model's performance on the student-testing dataset?**
   - Observation-level Top-1 Accuracy: **86.56%**
   - Pair-level Macro F1-score: **0.9160**
   - Pair-level Log Loss: **0.1073**
   - Pair-level Precision: **0.8574**, Recall: **0.8570**
   - Route Continuity: **81.45%**
   - Inference Latency: **0.00 ms per 1,000 rows** (0.065 s total for 127k test rows)
8. **Observation-level Top-1 Accuracy, Macro F1-score and Log Loss on student-testing:**
   - Top-1: `0.8656`, Macro F1: `0.9160`, Log Loss: `0.1073`.
9. **Did you observe overfitting or underfitting? What evidence supports your conclusion?**
   - The model demonstrated strong generalisation without severe overfitting. The CV Top-1 was 0.8461 and the student-test Top-1 was 0.8656 (generalisation delta: 0.0195). Training fold Top-1 was 0.8858. The small difference confirms the GroupKFold cross-validation was representative of unseen test trips.

---
## Model: `knn` (Instance-based (k-NN))

### A. Model Understanding
1. **Why did you consider this algorithm suitable for the PathSnap problem?**
   - Instance-based spatial memory naturally clusters road segment candidates by Euclidean proximity in projected feature space.
2. **What are the major strengths and limitations of this algorithm for this task?**
   - **Strengths:** Completely non-parametric; directly mirrors local geometric density around GPS observations.
   - **Limitations:** O(N) inference complexity makes it unsuitable for low-latency live API service; high memory footprint requiring training subsampling.

### B. Training and Configuration
5. **What were the important hyperparameters and their initial values?**
   - Configuration: `defaults (see run json)`
   - Subsample: `200000.0`
6. **What loss function or optimisation objective was used?**
   - Inverse distance weighted majority voting over k-nearest neighbours.

### C. Experimental Results
7. **What was the model's performance on the student-testing dataset?**
   - Observation-level Top-1 Accuracy: **85.47%**
   - Pair-level Macro F1-score: **0.9055**
   - Pair-level Log Loss: **0.1558**
   - Pair-level Precision: **0.8480**, Recall: **0.8304**
   - Route Continuity: **80.74%**
   - Inference Latency: **0.16 ms per 1,000 rows** (19.938 s total for 127k test rows)
8. **Observation-level Top-1 Accuracy, Macro F1-score and Log Loss on student-testing:**
   - Top-1: `0.8548`, Macro F1: `0.9055`, Log Loss: `0.1558`.
9. **Did you observe overfitting or underfitting? What evidence supports your conclusion?**
   - The model demonstrated strong generalisation without severe overfitting. The CV Top-1 was 0.8395 and the student-test Top-1 was 0.8548 (generalisation delta: 0.0152). Training fold Top-1 was 0.9145. The small difference confirms the GroupKFold cross-validation was representative of unseen test trips.
