@echo off
REM Laptop 1: PathSnap API. Run from the repo root.   Usage: scripts\run_api.bat [port]   (default 8000)
REM Optional before running: set PATHSNAP_TRIPS_DIR=<folder with the evaluation trips>
REM Allow the port once in Windows Firewall (admin prompt):
REM   netsh advfirewall firewall add rule name="PathSnap API" dir=in action=allow protocol=TCP localport=8000
set PORT=%~1
if "%PORT%"=="" set PORT=8000
python -m uvicorn api.main:app --host 0.0.0.0 --port %PORT%
