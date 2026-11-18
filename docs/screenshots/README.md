# Screenshots for report section 7 (API, UI and Deployment)

Captured on 18 Nov 2026 from the real application (API on `0.0.0.0:8020`, UI pointed at the Wi-Fi address `http://10.218.43.14:8020`, model `xgb`). They were taken in the browser automation pane at 800 px width. Retake them at full resolution if the report needs sharper images.

| # | File | Status |
|---|---|---|
| 1 | `01_ui_connected_real_model.jpg`: backend connection status, model availability | Captured |
| 2 | `02_ui_trip_selection.jpg`: trip selection | Captured |
| 3 | `03_ui_prediction_running.jpg`: loading indicator | Captured |
| 4 | `04_ui_result_metrics_map_trip56.jpg`: successful prediction, metrics, route map | Captured |
| 5 | `05_ui_map_layer_control.png`: map layers | Captured |
| 6 | `06_ui_details_and_timings.jpg`: details table and stage timings (taken before the number-formatting fix; candidates/point unrounded) | Captured |
| 6b | `06b_ui_trip52_details_formatted.jpg`: trip 00000052 with the fixed details table; also shows a trip where the HMM is 0.8 pts *below* classifier-only | Captured |
| 7 | `07_api_swagger_docs.jpg`: Swagger API documentation | Captured |
| 8 | `08_ui_error_api_unreachable.jpg`: error handling | Captured |
| 9 | `09_api_terminal_predict_log.png`: API terminal showing prediction requests | **To capture manually** |
| 10 | `10_two_laptop_ui.jpg`: UI on Laptop 2 connected to Laptop 1 | **To capture during the two-laptop test** |

## How to capture 9 and 10
1. Laptop 1: `.\scripts\run_api.bat 8000` in a PowerShell or CMD window. Keep that window visible.
2. Laptop 2: `.\scripts\run_ui.bat http://<laptop1-wifi-ip>:8000`, open `http://localhost:8501`, pick a trip and click **Predict route**.
3. On Laptop 1, press **Win + Shift + S** and capture the terminal window showing the `PREDICT trip=... mode=real model=xgb ... client=<laptop2-ip>` line. Save it as `09_api_terminal_predict_log.png`. The `client=` address proves the request came from the other laptop.
4. On Laptop 2, capture the full browser window with the result. Make sure the sidebar API URL (Laptop 1's IP) is visible. Save it as `10_two_laptop_ui.jpg`.

The real server log from the one-laptop run is in `docs/evidence/api_console_2026-11-18.log`. Use it only as supporting text, not as a substitute for screenshot 9.
