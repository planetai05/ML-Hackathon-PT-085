#!/usr/bin/env sh
# Laptop 1: PathSnap API (run from the repo root). Usage: scripts/run_api.sh [port]
exec python -m uvicorn api.main:app --host 0.0.0.0 --port "${1:-${PORT:-8000}}"
