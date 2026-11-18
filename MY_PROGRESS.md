# MY_PROGRESS – Person C (AryamanDK07)

Role (final team split): FastAPI backend, Streamlit UI, API–UI integration, application testing, deployment readiness, API/UI docs, report section 7.
Statuses: Not Started · In Progress · Blocked · Implemented · Tested · Completed. Updated 18 Nov 2026.

## Person C work
| Item | Status | Where | Evidence | Remaining |
|---|---|---|---|---|
| API contract (#2) | Completed (closed) | `docs/API_CONTRACT.md` | Used unchanged by B's Predictor | – |
| FastAPI `/health`, `/trips`, `/predict` | Tested | `api/` (PR #37, #41, #44) | Status codes 200/307/404/422/503 checked over the LAN address; Swagger at `/docs` | – |
| Streamlit UI | Tested | `ui/app.py`, `ui/health.py` | Live run against the LAN API; headless AppTest tests; screenshots 01–08 | – |
| Real model in the API (#25) | Tested | – | 10/10 trips, API metrics identical to direct Predictor calls | Closes when PR #44 merges |
| Automated tests | Tested | `tests/test_api.py`, `tests/test_health_contract.py` | `python -m pytest tests -q` → 55 passed | – |
| E2E check | Tested | `scripts/e2e_check.py`, `results/e2e_api_check.csv` | 10/10 HTTP 200, mode=real, model=xgb | – |
| Deployment readiness | Implemented | `scripts/run_*.bat`, `docs/API_UI.md` | `0.0.0.0` bind + Wi-Fi IP access + scripts with ports verified on one laptop | **Physical two-laptop test** |
| Report section 7 | Implemented (draft) | `docs/report_section7_api_ui.md` | Real numbers and figures | Add the two-laptop result and screenshots 9–10 |
| Screenshots | In Progress | `docs/screenshots/` | 9 captured | 09 API terminal, 10 Laptop-2 UI (need second laptop) |
| Demo and submission (#34) | In Progress | – | – | Two-laptop dry run, viva prep, submit; repo must be made public by its owner (#33) |

## Earlier ML work (before the role change; ML is now owned by Person B)
#8, #9, #10, #11 and #16 were implemented and closed by me (candidates, label alignment, features, CV harness, HMM). The `ml-pipeline` branch holds my HMM/Viterbi and Predictor experiments; it was not merged, and B's implementation is the one on `main`.
