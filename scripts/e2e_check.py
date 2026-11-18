"""End-to-end check over HTTP: every trip from GET /trips through POST /predict (issue #28).

python scripts/e2e_check.py --api http://<laptop1-ip>:8000 [--out results/e2e_api_check.csv]
Uses only the public API (exactly what the Streamlit UI does).
"""
import argparse
import csv
import sys
import time

import requests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://127.0.0.1:8000")
    ap.add_argument("--out", default="results/e2e_api_check.csv")
    a = ap.parse_args()
    try:
        h = requests.get(f"{a.api}/health", timeout=10).json()
    except requests.RequestException as e:
        sys.exit(f"Cannot reach the API at {a.api} ({type(e).__name__}). Is it running, and are the IP/port and firewall correct?")
    if not isinstance(h, dict) or "predictor_mode" not in h:
        sys.exit(f"{a.api}/health is not the PathSnap API (got {h!r}); another service may be using this port")
    print("health:", {k: h.get(k) for k in ("service", "status", "predictor_mode", "model_name", "n_trips")})
    trips = requests.get(f"{a.api}/trips", timeout=10).json()["trips"]
    rows, ok = [], True
    for t in trips:
        t0 = time.perf_counter()
        r = requests.post(f"{a.api}/predict", json={"trip_id": t}, timeout=300)
        wall = time.perf_counter() - t0
        j = r.json()
        m = j.get("metrics") or {}
        row = dict(trip_id=t, http=r.status_code, mode=j.get("mode"), model=(j.get("metadata") or {}).get("model_name"),
                   n_points=j.get("n_points"), route_arcs=len(j.get("predicted_arcs") or []),
                   segments=(j.get("metadata") or {}).get("n_segments"), top1=m.get("top1_accuracy"),
                   clf_top1=m.get("classifier_only_top1"), route_f1=m.get("route_f1"), continuity=m.get("route_continuity"),
                   api_ms=j.get("api_latency_ms"), wall_s=round(wall, 2), error=j.get("error") or j.get("metrics_error"))
        ok &= r.status_code == 200 and j.get("mode") == "real" and row["route_arcs"] > 0
        rows.append(row)
        print(row)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("ALL OK" if ok else "FAILURES PRESENT", "->", a.out)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
