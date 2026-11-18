"""PathSnap evaluator UI (Laptop 2).

Run:  streamlit run ui/app.py            (API URL from PATHSNAP_API_URL or the sidebar)
The UI talks ONLY to the FastAPI backend over HTTP; it never loads ML code, models or .route files.
"""
from __future__ import annotations

import os
import sys
import time

import folium
import pandas as pd
import requests
import streamlit as st
from streamlit_folium import st_folium

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for `ui.health`
from ui.health import HealthContractError, fmt_value, parse_health  # noqa: E402

DEFAULT_API = os.environ.get("PATHSNAP_API_URL", "http://127.0.0.1:8000")
TIMEOUT_S = float(os.environ.get("PATHSNAP_API_TIMEOUT", "120"))

st.set_page_config(page_title="PathSnap Evaluator", page_icon="🗺️", layout="wide")


# ---------------- API client ----------------
class ApiError(Exception):
    pass


def api_call(method: str, base: str, path: str, **kw):
    url = base.rstrip("/") + path
    try:
        r = requests.request(method, url, timeout=kw.pop("timeout", TIMEOUT_S), **kw)
    except requests.ConnectionError:
        raise ApiError(f"Cannot reach the API at {base}. Is the backend running and the URL/IP correct?")
    except requests.Timeout:
        raise ApiError(f"The API at {base} did not answer within {TIMEOUT_S:.0f} s.")
    try:
        body = r.json()
    except ValueError:
        raise ApiError(f"Unexpected non-JSON response from {url} (HTTP {r.status_code}).")
    if r.status_code >= 400:
        raise ApiError(f"HTTP {r.status_code} – {body.get('error', 'error')}: {body.get('detail', body)}")
    return body


def download_eval_csv(base: str, job_id: str) -> str:
    url = base.rstrip("/") + f"/evaluate-zip/{job_id}/download"
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 200:
            return r.text
    except Exception:
        pass
    return ""


@st.cache_data(ttl=10, show_spinner=False)
def get_health(base):
    return api_call("GET", base, "/health", timeout=5)


@st.cache_data(ttl=30, show_spinner=False)
def get_trips(base):
    return api_call("GET", base, "/trips", timeout=10)["trips"]


def get_eval_info(base):
    try:
        url = base.rstrip("/") + "/evaluate-zip/info"
        r = requests.get(url, timeout=4)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


# ---------------- map ----------------
def _lines(feature):
    if not feature:
        return []
    g = feature.get("geometry") or {}
    coords = g.get("coordinates") or []
    if g.get("type") == "LineString":
        coords = [coords]
    return [[(lat, lon) for lon, lat in line] for line in coords if len(line) > 1]


def build_map(res: dict, show_pts: bool) -> folium.Map:
    pts = [(p[1], p[0]) for p in res["gps_points"]]
    m = folium.Map(tiles="OpenStreetMap", control_scale=True)
    if show_pts:
        fg = folium.FeatureGroup(name="GPS observations (blue)")
        step = max(1, len(pts) // 1500)  # keep the map responsive for long trips
        for lat, lon in pts[::step]:
            folium.CircleMarker((lat, lon), radius=2, color="#2c7bb6", fill=True, fill_opacity=0.8, weight=0).add_to(fg)
        fg.add_to(m)
    fg = folium.FeatureGroup(name="Predicted route (red)")
    for line in _lines(res["predicted_route_geojson"]):
        folium.PolyLine(line, color="#d73027", weight=4, opacity=0.9).add_to(fg)
    fg.add_to(m)
    if pts:
        folium.Marker(pts[0], tooltip="Start", icon=folium.Icon(color="green", icon="play")).add_to(m)
        folium.Marker(pts[-1], tooltip="End", icon=folium.Icon(color="red", icon="stop")).add_to(m)
        lats, lons = [p[0] for p in pts], [p[1] for p in pts]
        m.fit_bounds([(min(lats), min(lons)), (max(lats), max(lons))])
    folium.LayerControl(collapsed=True).add_to(m)
    return m


def build_eval_map(trip_data: dict) -> folium.Map:
    pts = [(p[1], p[0]) for p in trip_data.get("gps_points", [])]
    m = folium.Map(tiles="OpenStreetMap", control_scale=True)
    if pts:
        fg_pts = folium.FeatureGroup(name="GPS observations (blue)")
        for lat, lon in pts:
            folium.CircleMarker((lat, lon), radius=2, color="#2c7bb6", fill=True, fill_opacity=0.8, weight=0).add_to(fg_pts)
        fg_pts.add_to(m)

    pred_geo = trip_data.get("predicted_route_geojson")
    if pred_geo:
        fg_pred = folium.FeatureGroup(name="Predicted route (red)")
        for line in _lines(pred_geo):
            folium.PolyLine(line, color="#d73027", weight=4, opacity=0.9).add_to(fg_pred)
        fg_pred.add_to(m)

    if pts:
        folium.Marker(pts[0], tooltip="Start", icon=folium.Icon(color="green", icon="play")).add_to(m)
        folium.Marker(pts[-1], tooltip="End", icon=folium.Icon(color="red", icon="stop")).add_to(m)
        lats, lons = [p[0] for p in pts], [p[1] for p in pts]
        m.fit_bounds([(min(lats), min(lons)), (max(lats), max(lons))])
    folium.LayerControl(collapsed=True).add_to(m)
    return m


def pct(v):
    return "–" if v is None else f"{100 * v:.1f}%"


# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Backend Connection")
    api = st.text_input("API URL", value=st.session_state.get("api", DEFAULT_API),
                        help="Laptop 1 address, e.g. http://192.168.1.20:8000").strip()
    st.session_state["api"] = api
    if st.button("Reconnect", use_container_width=True):
        get_health.clear()
        get_trips.clear()
    health = None
    try:
        health = parse_health(get_health(api), api)
        mode = health["predictor_mode"]
        if mode == "real":
            st.success(f"Connected · model **{health['model_name'] or 'unknown'}**")
        elif mode == "mock":
            st.warning("Connected · **MOCK MODE** – responses are placeholders, not ML predictions")
        else:
            st.error(f"Connected, but no model is loaded: {health['predictor_error'] or 'no reason reported'}")
        st.caption(f"Trips folder: {health['n_trips']} trips · artifact {health['artifact_version'] or '–'}")
    except (ApiError, HealthContractError) as e:
        health = None
        st.error(str(e))
    st.header("Map Layers")
    show_pts = st.checkbox("GPS points", value=True)

if health is None:
    st.title("🗺️ PathSnap – GPS Map-Matching")
    st.info("Start the backend on Laptop 1 (`uvicorn api.main:app --host 0.0.0.0 --port 8000`) "
            "and enter its IP/URL in the sidebar.")
    st.stop()


# ---------------- Main Page Tabs ----------------
tab_eval, tab_single = st.tabs(["⚡ Evaluator — Full Dataset Evaluation", "🔍 Single Trip Inspection"])

# =========================================================================
# TAB 1: EVALUATOR — TWO-LAPTOP EVALUATION SUITE
# =========================================================================
with tab_eval:
    st.title("⚡ PathSnap Evaluator")
    st.caption("End-to-end evaluation suite across 10 evaluation trips stored on Laptop 1. "
               "The backend executes candidate generation, trained XGBoost, and HMM/Viterbi with strict ground-truth isolation.")

    eval_info = get_eval_info(api)
    exists = eval_info.get("exists", False) if eval_info else False
    dev_info = eval_info.get("execution_device", "Unknown") if eval_info else "Unknown"
    tot_trips = eval_info.get("total_trips", 0) if eval_info else 0

    col_info1, col_info2, col_info3 = st.columns([2, 1, 1])
    with col_info1:
        st.markdown(f"**Dataset Archive (Laptop 1):** `{eval_info.get('dataset_name', 'evaluator_test.zip') if eval_info else 'evaluator_test.zip'}`")
        st.caption(f"Status: {'✅ Ready (' + str(tot_trips) + ' trips found)' if exists else '❌ Not found on backend'}")
    with col_info2:
        st.markdown(f"**Total Trips:** {tot_trips}")
        st.caption("Benchmark suite")
    with col_info3:
        st.markdown("**Execution Device:**")
        st.caption(f"{dev_info}")

    c_btn, c_space = st.columns([1, 3])
    eval_btn = c_btn.button("🚀 Evaluate ZIP", type="primary", use_container_width=True, disabled=not exists)

    if eval_btn:
        with st.spinner("Submitting evaluation job to Laptop 1 backend..."):
            try:
                start_resp = api_call("POST", api, "/evaluate-zip")
                job_id = start_resp["job_id"]
                st.session_state["eval_job_id"] = job_id
            except ApiError as e:
                st.error(f"Failed to start evaluation: {e}")
                job_id = None

        if job_id:
            prog_bar = st.progress(0.0)
            status_text = st.empty()
            failed = False

            while True:
                try:
                    job_status = api_call("GET", api, f"/evaluate-zip/{job_id}")
                except ApiError as e:
                    st.error(f"Error checking job status: {e}")
                    failed = True
                    break

                st_val = job_status.get("status")
                comp = job_status.get("completed_trips", 0)
                tot = job_status.get("total_trips", 10)
                curr = job_status.get("current_trip") or "initializing"
                pct_val = job_status.get("progress_pct", 0.0)

                prog_bar.progress(min(pct_val / 100.0, 1.0))
                status_text.info(f"⏳ Evaluating trip **{curr}** ({comp}/{tot} completed)...")

                if st_val == "completed":
                    prog_bar.progress(1.0)
                    status_text.success(f"✅ Evaluation complete! All {tot} trips successfully evaluated.")
                    break
                elif st_val == "failed":
                    prog_bar.empty()
                    status_text.error(f"❌ Evaluation failed: {job_status.get('error')}")
                    failed = True
                    break

                time.sleep(0.8)

            if not failed:
                try:
                    results = api_call("GET", api, f"/evaluate-zip/{job_id}/results")
                    st.session_state["eval_results"] = results
                    st.rerun()
                except ApiError as e:
                    st.error(f"Could not retrieve evaluation results: {e}")

    # Render completed results if available
    eval_results = st.session_state.get("eval_results")
    eval_job_id = st.session_state.get("eval_job_id", "")

    if eval_results:
        st.divider()
        agg = eval_results.get("aggregate_metrics") or eval_results
        trip_list = eval_results.get("per_trip") or eval_results.get("trip_results") or []

        st.subheader("📊 Aggregate Benchmark Performance")
        k = st.columns(3) + st.columns(3)
        k[0].metric("Viterbi Top-1 Accuracy", pct(agg.get("mean_viterbi_top1")),
                    None if agg.get("mean_classifier_top1") is None or agg.get("mean_viterbi_top1") is None
                    else f"{100 * (agg['mean_viterbi_top1'] - agg['mean_classifier_top1']):+.1f} pts vs classifier")
        k[1].metric("Classifier-Only Top-1", pct(agg.get("mean_classifier_top1")))
        k[2].metric("Route Continuity", pct(agg.get("mean_route_continuity")))
        k[3].metric("Route F1 Score", pct(agg.get("mean_route_f1")))
        mean_lat_ms = agg.get("mean_latency_ms") if agg.get("mean_latency_ms") is not None else agg.get("mean_single_trip_latency_ms")
        k[4].metric("Mean Latency / Trip", f"{(mean_lat_ms or 0):.1f} ms")
        k[5].metric("Total Run Time", f"{(agg.get('total_evaluation_time_s') or 0):.1f} s")

        st.info(f"💻 **Execution Device:** {agg.get('execution_device', dev_info)}")

        # Download CSV button
        csv_data = download_eval_csv(api, eval_job_id)
        if csv_data:
            st.download_button(
                label="📥 Download CSV Evaluation Report",
                data=csv_data,
                file_name=f"evaluator_test_results_{eval_job_id}.csv",
                mime="text/csv",
                use_container_width=False,
            )

        # Per-trip table
        st.subheader("📋 Per-Trip Evaluation Breakdown")
        table_rows = []
        for t in trip_list:
            table_rows.append({
                "Trip ID": t["trip_id"],
                "GPS Points": f"{t.get('n_points', 0):,}",
                "Classifier Top-1": pct(t.get("classifier_only_top1")),
                "Viterbi Top-1": pct(t.get("viterbi_top1")),
                "Route Continuity": pct(t.get("route_continuity")),
                "Route F1": pct(t.get("route_f1")),
                "Latency (ms)": f"{t.get('latency_ms', 0):.1f}",
                "Status": t.get("status", "ok"),
            })
        df_trips = pd.DataFrame(table_rows)
        st.dataframe(df_trips, use_container_width=True, hide_index=True)

        # Diagnostic Charts
        st.subheader("📈 Diagnostic Visualizations")
        chart_rows = []
        for t in trip_list:
            if t.get("status") == "ok":
                chart_rows.append({
                    "Trip": t["trip_id"],
                    "Classifier Top-1 (%)": (t.get("classifier_only_top1") or 0) * 100,
                    "Viterbi Top-1 (%)": (t.get("viterbi_top1") or 0) * 100,
                    "Route Continuity (%)": (t.get("route_continuity") or 0) * 100,
                    "Route F1 (%)": (t.get("route_f1") or 0) * 100,
                    "Latency (ms)": t.get("latency_ms") or 0,
                })

        if chart_rows:
            df_chart = pd.DataFrame(chart_rows).set_index("Trip")
            c_ch1, c_ch2 = st.columns(2)
            with c_ch1:
                st.markdown("**Accuracy: Classifier-Only vs Viterbi Decoder (%)**")
                st.bar_chart(df_chart[["Classifier Top-1 (%)", "Viterbi Top-1 (%)"]])
            with c_ch2:
                st.markdown("**Route Quality: Continuity & Overlap F1 (%)**")
                st.bar_chart(df_chart[["Route Continuity (%)", "Route F1 (%)"]])

            st.markdown("**Single-Trip Processing Latency (ms)**")
            st.bar_chart(df_chart["Latency (ms)"])

        # Interactive Map for evaluated trips
        st.subheader("🗺️ Trip Route Visualizer (Evaluated Archive)")
        trip_ids = [t["trip_id"] for t in trip_list]
        if trip_ids:
            c_sel, _ = st.columns([2, 2])
            selected_eval_trip_id = c_sel.selectbox("Select trip to inspect on map", trip_ids, index=0)
            selected_trip = next((t for t in trip_list if t["trip_id"] == selected_eval_trip_id), None)
            if selected_trip:
                st.caption("Map displays GPS observations (blue) and ML-predicted route (red). "
                           "Strict ground-truth isolation is enforced; reference routes are evaluated server-side and never exposed.")
                st_folium(build_eval_map(selected_trip), height=520, use_container_width=True, returned_objects=[])


# =========================================================================
# TAB 2: SINGLE TRIP INSPECTION (PRESERVED DEMO WORKFLOW)
# =========================================================================
with tab_single:
    st.title("🔍 Single Trip Inspection")
    st.caption("Pick a trip from the backend's standard trips library to inspect intermediate candidate generation, "
               "probabilities, and reference vs predicted route overlay.")

    try:
        trips = get_trips(api)
    except ApiError as e:
        st.error(f"Could not load the trip list: {e}")
        trips = []

    if not trips:
        st.warning("The backend has no trips in its trips folder (PATHSNAP_TRIPS_DIR).")
    else:
        c1, c2 = st.columns([3, 1])
        trip = c1.selectbox("Evaluation trip", trips, index=None, placeholder="Select one of the trips…")
        c2.write("")
        c2.write("")
        go = c2.button("Predict route", type="primary", use_container_width=True)

        if go and not trip:
            st.warning("Select a trip first.")
        if go and trip:
            with st.spinner(f"Running the PathSnap pipeline on trip {trip}…"):
                try:
                    st.session_state["result"] = api_call("POST", api, "/predict", json={"trip_id": trip})
                    st.session_state.pop("error", None)
                except ApiError as e:
                    st.session_state["error"] = str(e)
                    st.session_state.pop("result", None)

        if st.session_state.get("error"):
            st.error(f"Prediction failed – {st.session_state['error']}")

        res = st.session_state.get("result")
        if res:
            if res.get("mode") == "mock":
                st.warning("⚠️ MOCK RESPONSE – the backend is in development mode. This is not an ML prediction.")
            meta, met = res.get("metadata", {}), res.get("metrics") or {}
            st.subheader(f"Trip {res['trip_id']}")
            k = st.columns(3) + st.columns(3)
            k[0].metric("Point accuracy (HMM)", pct(met.get("top1_accuracy")),
                        None if met.get("classifier_only_top1") is None or met.get("top1_accuracy") is None
                        else f"{100 * (met['top1_accuracy'] - met['classifier_only_top1']):+.1f} pts vs classifier-only")
            k[1].metric("Classifier-only accuracy", pct(met.get("classifier_only_top1")))
            k[2].metric("Route F1", pct(met.get("route_f1")))
            k[3].metric("Route continuity", pct(met.get("route_continuity")))
            k[4].metric("GPS points", f"{res['n_points']:,}")
            k[5].metric("API latency", f"{res['api_latency_ms'] / 1000:.2f} s")
            if res.get("metrics_error"):
                st.info(f"Metrics unavailable for this trip: {res['metrics_error']}")

            left, right = st.columns([5, 2])
            with left:
                st_folium(build_map(res, show_pts), height=560, use_container_width=True, returned_objects=[])
            with right:
                rows = [("Model", meta.get("model_name")), ("Decoder", meta.get("decoder")),
                        ("Route arcs", len(res.get("predicted_arcs", []))), ("Route segments", meta.get("n_segments")),
                        ("Candidates / point", meta.get("mean_candidates")), ("Points w/o candidates", meta.get("empty_candidate_points")),
                        ("Artifact", meta.get("artifact_version"))]
                if met:
                    rows += [("Route precision", pct(met.get("route_precision"))), ("Route recall", pct(met.get("route_recall"))),
                             ("Scored points", met.get("n_points_aligned"))]
                st.markdown("**Prediction details**")
                st.markdown("\n".join(["| Detail | Value |", "|---|---|"] + [f"| {a} | {fmt_value(b)} |" for a, b in rows]))
                st.markdown("**Timings (ms)**")
                st.markdown("\n".join(["| stage | ms |", "|---|---:|"] + [f"| {k_} | {v:,.0f} |" for k_, v in res.get("timings_ms", {}).items()]))
            with st.expander("Raw API response"):
                st.json({k_: v for k_, v in res.items() if k_ not in ("gps_points", "per_point_arc", "per_point_prob")}, expanded=False)
