#!/usr/bin/env sh
# Laptop 1: PathSnap API (run from the repo root)
exec python -m uvicorn api.main:app --host 0.0.0.0 --port "${PORT:-8000}"
