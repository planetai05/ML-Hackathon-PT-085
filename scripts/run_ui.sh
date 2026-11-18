#!/usr/bin/env sh
# Laptop 2: PathSnap UI. Usage: scripts/run_ui.sh http://<laptop1-ip>:8000
[ -n "$1" ] && export PATHSNAP_API_URL="$1"
exec python -m streamlit run ui/app.py --server.port 8501 --server.address 0.0.0.0
