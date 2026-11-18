@echo off
REM Laptop 2: PathSnap UI.   Usage: scripts\run_ui.bat http://<laptop1-ip>:<api-port> [ui-port]   (default UI port 8501)
if not "%~1"=="" set PATHSNAP_API_URL=%~1
set UI_PORT=%~2
if "%UI_PORT%"=="" set UI_PORT=8501
python -m streamlit run ui/app.py --server.port %UI_PORT% --server.address 0.0.0.0
