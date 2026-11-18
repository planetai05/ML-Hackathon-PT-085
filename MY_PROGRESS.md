# MY_PROGRESS – Person C (AryamanDK07)

Branch: `ml-pipeline`. Statuses: Not Started · In Progress · Blocked · Implemented · Tested · Completed.
Tests: `python -m pytest tests -q` (7 unit tests: toy graph + one real trip).

| Issue | Title | Status | Files | Implementation | Tests | Commit | Remaining |
|---|---|---|---|---|---|---|---|
| #2 | ML prediction interface and API contract | Tested | `docs/API_CONTRACT.md`, `docs/sample_predict_response.json`, `models/predictor.py` | Contract doc; `Predictor(mock=True)` | Mock run on student_test 00000052 | 61c6e45 | 👍 from planetai05 and ashvinrlobo |
| #8 | Candidate generation | Completed | `features/trip_io.py`, `features/candidates.py` | cKDTree over densified arcs, R=40 m, k=12, 2R retry | toy + real-trip tests; miss rate train 0.28% / test 0.06% | 588e362 | – |
| #9 | Route-to-point label alignment | Completed | `features/labels.py`, `features/plot_alignment.py` | Monotonic sparse DP; unaligned flag | toy test, 3 validation plots | 2b9586e | – |
| #10 | Feature engineering and parquet cache | Completed | `features/build_features.py` | 28 features incl. windowed heading (d5ab858) | toy heading test, no NaNs | 552d49f, d5ab858 | – |
| #11 | GroupKFold harness | Completed | `models/train.py`, `evaluation/metrics.py` | 5-fold GroupKFold by trip, all metrics, run JSONs | logreg run | 464b0c4 | – |
| #13 | Train 5+ models | In Progress | `models/train.py` | logreg, knn, rf, hgb, lgbm, mlp | retraining after feature update | – | finish run, commit results |
| #14 | Model selection | Not Started | | | | | needs #13 |
| #15 | Optuna HPO | Implemented | `hpo/optuna_search.py` | TPE, SQLite study, tuned vs untuned | – | 71c5183 | run on selected model |
| #16 | HMM | Completed | `hmm/model.py` | Emission/transition, bounded Dijkstra cache | 2 toy tests, timing on largest trips | 8472593 | – |
| #17 | Viterbi | Tested | `hmm/viterbi.py` | Log-space Viterbi, restarts, gap filling | toy test; decodes student_test | 66f055c | confirm continuity in #18 |
| #18 | Route-level evaluation | Implemented | `evaluation/route_eval.py` | OOF-based decoder tuning, test comparison | – | 44cf3a2 | run with final model |
| #19 | Export artifacts + real Predictor | Implemented | `models/predictor.py`, `models/export.py` | Full pipeline, per-trip cache, server-side score | ran on 3 student_test trips | e38f7d6 | export final model |
| #25 | Wire ML into FastAPI | Blocked | | | | | Needs #22 (planetai05): no `api/` pushed yet |
| #31 | Report sections 3–5 | Not Started | | | | | needs results |
| #34 | Demo rehearsal and submission | Not Started | | | | | end of day |
