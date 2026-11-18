@echo off
REM Laptop 2: PathSnap UI. Usage: scripts\run_ui.bat http://<laptop1-ip>:8000
if not "%~1"=="" set PATHSNAP_API_URL=%~1
python -m streamlit run ui/app.py --server.port 8501 --server.address 0.0.0.0
