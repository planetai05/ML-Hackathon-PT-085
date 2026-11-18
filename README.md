# PathSnap: Distributed Map-Matching & Sequential Route Reconstruction (Team PT-085)

PathSnap accurately matches noisy 1 Hz vehicular GPS trajectories to a road network and reconstructs physically feasible, continuous driving routes.

**Production Pipeline:**
Raw GPS Observations → Spatial Index Candidate Generation (40m radius) → 24 Spatial-Kinematic Features → Champion Tuned XGBoost Classifier → Log-Space Hidden Markov Model & Viterbi Decoding → Continuous Reconstructed Route → High-Performance FastAPI Backend (Laptop 1) + Interactive Folium Streamlit Evaluator Dashboard (Laptop 2).

---

## 1. Project Status & Team Ownership

| Part | Component & Responsibilities | Owner | Status |
|:---:|---|:---:|:---:|
| **Part A** | EDA, UTM Projection, Spatial Indexing, Candidate Generation, DP Label Alignment, Folium Maps | **Person A** | **Completed** (`docs/HANDOFF_PERSON_A.md`) |
| **Part B** | ML Model Benchmark (7 models), GroupKFold, Model Selection, Optuna Bayesian HPO, Production Artifacts | **Person B** | **Completed** (`docs/HANDOFF_PERSON_C.md`) |
| **Part C** | HMM/Viterbi Decoder, FastAPI Server (`/health`, `/trips`, `/predict`), Streamlit Dashboard, LAN Deployment | **Person C** | **Completed** (`docs/API_UI.md`) |

---

## 2. Key Empirical Results (Validated Across All 10 Student-Test Trips)

All figures verified against actual runs recorded in `results/hmm_comparison.csv` and `results/e2e_api_check.csv`:

- **Candidate Top-1 Accuracy (XGBoost Classifier):** **86.87%**
- **Reconstructed Route Top-1 Accuracy (Viterbi HMM):** **88.30%** point-weighted vs 86.87% classifier-only (89.56% vs 88.48% as a mean of per-trip values)
- **Route Continuity:** Jumped from 81.72% (classifier-only) to **97.20% (with HMM)**
- **Length-Weighted Route Overlap F1 Score:** **94.61%**
- **Mean Single-Trip Processing Latency:** **813.7 ms** (offline) / **1,729 ms** (live HTTP over LAN)
- **Automated Test Suite:** **57 passed, 2 skipped** (`python -m pytest tests -q`; the skipped tests need the evaluator ZIP)

---

## 3. Repository Architecture

```
PathSnap/
├── api/                   # FastAPI backend implementation
│   ├── config.py          # Environment settings (PATHSNAP_TRIPS_DIR, PREDICTOR, CORS)
│   ├── main.py            # Endpoints: GET /health, GET /trips, POST /predict
│   └── schemas.py         # Pydantic v2 contract schemas
├── artifacts/             # Serialized production model artifacts
│   ├── best_model_tuned.joblib    # Champion Tuned XGBoost (249 KB)
│   ├── best_model_untuned.joblib  # Baseline XGBoost (464 KB)
│   ├── best_hyperparameters.json  # Optuna Trial #3 optimal parameters
│   ├── scaler.joblib              # StandardScaler fitted on training folds
│   ├── hmm_config.json            # Viterbi transition probabilities and penalties
│   └── model_metadata.json        # Feature column order and metadata
├── data_processing/       # Raw dataset loading & EDA modules
│   ├── loader.py          # Trip loader & UTM projection
│   └── eda.py             # Dataset inventory, stats, and data quality logs
├── docs/                  # Project specifications, handoffs, and evidence
│   ├── API_CONTRACT.md    # REST API request/response specification
│   ├── API_UI.md          # Deployment & integration guide
│   ├── HANDOFF_PERSON_A.md # Feature schema & EDA notes
│   ├── HANDOFF_PERSON_C.md # Backend/frontend handoff documentation
│   └── screenshots/       # Live UI dashboard and Swagger documentation captures
├── evaluation/            # Offline and server-side route scoring
│   └── metrics.py         # score_route, continuity, route_overlap, pair_metrics
├── features/              # Feature engineering & candidate generation
│   ├── build_features.py  # 24 spatial-kinematic feature extractors
│   ├── candidates.py      # Spatial kd-tree candidate generation
│   ├── labels.py          # Dynamic programming route-to-point alignment
│   └── trip_io.py         # High-speed memory-mapped trip I/O
├── hmm/                   # Sequential reconstruction
│   └── viterbi.py         # Log-space dynamic programming sequence decoder
├── hpo/                   # Bayesian hyperparameter optimization
│   └── optuna_search.py   # 30-trial TPE study with MedianPruner
├── models/                # Machine learning classifiers & Predictor
│   ├── predictor.py       # Production Predictor adhering to docs/API_CONTRACT.md
│   └── train_models.py    # 7-model GroupKFold cross-validation suite
├── reports/               # Technical report documents
│   ├── PathSnap_Report_PT-085.pdf     # Consolidated technical report (submission PDF)
│   └── PathSnap_Report.md             # Report source (built by scripts/build_report_pdf.py)
├── results/               # Committed empirical CSV records and figures
│   ├── e2e_api_check.csv              # Full HTTP verification log for 10 trips
│   ├── hmm_comparison.csv             # Offline HMM vs Classifier-only comparison
│   ├── hpo_comparison.csv             # Tuned vs untuned performance
│   ├── model_comparison.csv           # 7-model cross-validation benchmark
│   └── figures/report/                # High-resolution publication plots
├── scripts/               # Launcher scripts and verification utilities
│   ├── e2e_check.py       # Live HTTP end-to-end evaluation
│   ├── run_api.bat / .ps1 # Laptop 1 API launch script
│   └── run_ui.bat / .ps1  # Laptop 2 Streamlit launch script
├── tests/                 # Automated pytest suite (55 test cases)
└── ui/                    # Streamlit dashboard
    ├── app.py             # Evaluator user interface with interactive Folium map
    └── health.py          # /health probe parser & status formatter
```

---

## 4. Quick Start & Setup

### Requirements
- Python 3.11 or 3.12 (Tested on Python 3.12.4 Windows 11).

```powershell
pip install -r requirements.txt
```

### Run Full Test Suite
```powershell
python -m pytest tests -q
```
*Expected: 57 passed, 2 skipped (the 2 need `data/uploads/evaluator_test.zip`).*

---

## 5. Distributed Demonstration (Two-Laptop Workflow)

### Machine 1: Laptop 1 (Model Host & Evaluator Backend)
- **Role:** Hosts trained XGBoost model, HMM/Viterbi engine, FastAPI server, and the evaluation archive (`data/uploads/evaluator_test.zip`).
- **Hardware:** NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM) / 12-thread CPU fallback.
1. Ensure inbound traffic is permitted through the Windows Firewall:
   ```powershell
   netsh advfirewall firewall add rule name="PathSnap API" dir=in action=allow protocol=TCP localport=8000
   ```
2. Find Laptop 1's local IPv4 address:
   ```powershell
   ipconfig
   ```
   *(Look for "IPv4 Address", e.g. `192.168.1.15` or `10.x.x.x`)*
3. Launch the FastAPI server:
   ```powershell
   # Option A: PowerShell helper
   .\scripts\run_api.ps1 -Port 8000

   # Option B: Batch helper
   .\scripts\run_api.bat 8000

   # Option C: Direct uvicorn
   uvicorn api.main:app --host 0.0.0.0 --port 8000
   ```
   *The server binds to `0.0.0.0:8000` and displays its local IP address.*

### Machine 2: Laptop 2 (Streamlit Evaluator Dashboard)
- **Role:** Pure client UI running on Laptop 2. Connects to Laptop 1 over Wi-Fi/LAN. Does not require the dataset archive or ML dependencies.
1. Launch the Streamlit evaluator dashboard pointing to Laptop 1:
   ```powershell
   # Option A: PowerShell helper
   .\scripts\run_ui.ps1 -ApiUrl "http://<laptop1-ip>:8000" -Port 8501

   # Option B: Direct streamlit run
   $env:PATHSNAP_API_URL="http://<laptop1-ip>:8000"; streamlit run ui/app.py --server.port 8501
   ```
2. Open `http://localhost:8501` in your browser.
3. In the sidebar, verify the connection status to Laptop 1.
4. **Tab 1 ("⚡ Evaluator — Full Dataset Evaluation"):**
   - Click **🚀 Evaluate ZIP**.
   - Laptop 1 sequentially processes all 10 trips in `data/uploads/evaluator_test.zip` in a thread-safe background job.
   - Watch the live progress bar update in real-time.
   - Upon completion, view:
     - 6 Aggregate benchmark performance cards (Top-1, Continuity, Overlap F1, Latency, Total Time).
     - Execution device transparency badge.
     - **📥 Download CSV Evaluation Report** button.
     - Per-trip metric breakdown table.
     - 3 Diagnostic visualization charts (Classifier vs Viterbi Top-1, Continuity & F1, Latency).
     - Interactive Folium map displaying GPS observations and ML-predicted route (strictly isolated ground truth).
5. **Tab 2 ("🔍 Single Trip Inspection"):**
   - Select any trip from the library and click **Predict route** to inspect candidate counts, probabilities, latency breakdowns, and reference route overlays.

---

## 6. Offline Evaluation & Reproducibility
- **End-to-End HTTP Evaluation:** `python scripts/e2e_check.py --api http://127.0.0.1:8000`
- **HMM Route Comparison:** `python scripts/evaluate_hmm.py`
- **Report Visualizations:** `python visualisations/result_plots.py`
- **Compile Report PDF:** `python scripts/build_report_pdf.py`

---

## 7. Submission Deliverables
- **Consolidated Technical Report:** [`reports/PathSnap_Report_PT-085.pdf`](reports/PathSnap_Report_PT-085.pdf) (source: `reports/PathSnap_Report.md`).
- **GitHub Repository:** Clean, tagged, with zero untracked runtime artifacts and zero evaluator contamination.
