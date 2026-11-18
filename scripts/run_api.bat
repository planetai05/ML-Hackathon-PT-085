@echo off
REM Laptop 1: PathSnap API. Run from the repo root. Optional: set PATHSNAP_TRIPS_DIR / PATHSNAP_ARTIFACTS_DIR first.
REM Allow the port once in Windows Firewall (admin):  netsh advfirewall firewall add rule name="PathSnap API" dir=in action=allow protocol=TCP localport=8000
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
