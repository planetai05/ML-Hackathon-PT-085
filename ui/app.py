"""PathSnap evaluator UI (Laptop 2).

Run:  streamlit run ui/app.py            (API URL from PATHSNAP_API_URL or the sidebar)
The UI talks ONLY to the FastAPI backend over HTTP; it never loads ML code, models or .route files.
"""
from __future__ import annotations

import os

import folium
import requests
import streamlit as st
from streamlit_folium import st_folium

DEFAULT_API = os.environ.get("PATHSNAP_API_URL", "http://localhost:8000")
TIMEOUT_S = float(os.environ.get("PATHSNAP_API_TIMEOUT", "120"))

st.set_page_config(page_title="PathSnap", page_icon="🗺️", layout="wide")


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


@st.cache_data(ttl=10, show_spinner=False)
def get_health(base):
    return api_call("GET", base, "/health", timeout=5)


@st.cache_data(ttl=30, show_spinner=False)
def get_trips(base):
    return api_call("GET", base, "/trips", timeout=10)["trips"]


# ---------------- map ----------------
def _lines(feature):
    if not feature:
        return []
    g = feature.get("geometry") or {}
    coords = g.get("coordinates") or []
    if g.get("type") == "LineString":
        coords = [coords]
    return [[(lat, lon) for lon, lat in line] for line in coords if len(line) > 1]


def build_map(res: dict, show_ref: bool, show_pts: bool) -> folium.Map:
    pts = [(p[1], p[0]) for p in res["gps_points"]]
    m = folium.Map(tiles="cartodbpositron", control_scale=True)
    if show_ref and res.get("metrics") and res["metrics"].get("reference_route_geojson"):
        fg = folium.FeatureGroup(name="Reference route (green)")
        for line in _lines(res["metrics"]["reference_route_geojson"]):
            folium.PolyLine(line, color="#1a9850", weight=8, opacity=0.55).add_to(fg)
        fg.add_to(m)
    fg = folium.FeatureGroup(name="Predicted route (red)")
    for line in _lines(res["predicted_route_geojson"]):
        folium.PolyLine(line, color="#d73027", weight=4, opacity=0.9).add_to(fg)
    fg.add_to(m)
    if show_pts:
        fg = folium.FeatureGroup(name="GPS observations (blue)")
        step = max(1, len(pts) // 1500)  # keep the map responsive for long trips
        for lat, lon in pts[::step]:
            folium.CircleMarker((lat, lon), radius=2, color="#2c7bb6", fill=True, fill_opacity=0.8, weight=0).add_to(fg)
        fg.add_to(m)
    if pts:
        folium.Marker(pts[0], tooltip="Start", icon=folium.Icon(color="green", icon="play")).add_to(m)
        folium.Marker(pts[-1], tooltip="End", icon=folium.Icon(color="red", icon="stop")).add_to(m)
        lats, lons = [p[0] for p in pts], [p[1] for p in pts]
        m.fit_bounds([(min(lats), min(lons)), (max(lats), max(lons))])
    folium.LayerControl(collapsed=False).add_to(m)
    return m


def pct(v):
    return "–" if v is None else f"{100 * v:.1f}%"


# ---------------- layout ----------------
st.title("🗺️ PathSnap – GPS map-matching")
st.caption("Pick an evaluation trip. The backend runs candidate generation → classifier → HMM/Viterbi live "
           "and scores the result against the reference route held on the server.")

with st.sidebar:
    st.header("Backend")
    api = st.text_input("API URL", value=st.session_state.get("api", DEFAULT_API),
                        help="Laptop 1 address, e.g. http://192.168.1.20:8000").strip()
    st.session_state["api"] = api
    if st.button("Reconnect", use_container_width=True):
        get_health.clear(); get_trips.clear()
    health = None
    try:
        health = get_health(api)
        mode = health["predictor_mode"]
        if mode == "real":
            st.success(f"Connected · model **{health.get('model_name')}**")
        elif mode == "mock":
            st.warning("Connected · **MOCK MODE** – responses are placeholders, not ML predictions")
        else:
            st.error(f"Connected, but no model is loaded: {health.get('predictor_error')}")
        st.caption(f"Trips folder: {health['n_trips']} trips · artifact {health.get('artifact_version') or '–'}")
    except ApiError as e:
        st.error(str(e))
    st.header("Map layers")
    show_ref = st.checkbox("Reference route", value=True)
    show_pts = st.checkbox("GPS points", value=True)

if health is None:
    st.info("Start the backend on Laptop 1 (`uvicorn api.main:app --host 0.0.0.0 --port 8000`) "
            "and enter its URL in the sidebar.")
    st.stop()

try:
    trips = get_trips(api)
except ApiError as e:
    st.error(f"Could not load the trip list: {e}")
    st.stop()
if not trips:
    st.warning("The backend has no trips in its trips folder (PATHSNAP_TRIPS_DIR).")
    st.stop()

c1, c2 = st.columns([3, 1])
trip = c1.selectbox("Evaluation trip", trips, index=None, placeholder="Select one of the trips…")
c2.write(""); c2.write("")
go = c2.button("Predict route", type="primary", use_container_width=True, disabled=trip is None)

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
    k = st.columns(6)
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

    left, right = st.columns([3, 1])
    with left:
        st_folium(build_map(res, show_ref, show_pts), height=560, use_container_width=True, returned_objects=[])
    with right:
        st.markdown("**Prediction**")
        st.write({"model": meta.get("model_name"), "decoder": meta.get("decoder"),
                  "route arcs": len(res.get("predicted_arcs", [])), "route segments": meta.get("n_segments"),
                  "mean candidates / point": meta.get("mean_candidates"),
                  "points without candidates": meta.get("empty_candidate_points"),
                  "artifact": meta.get("artifact_version")})
        if met:
            st.markdown("**Route overlap (length-weighted)**")
            st.write({"precision": pct(met.get("route_precision")), "recall": pct(met.get("route_recall")),
                      "scored points": met.get("n_points_aligned")})
        st.markdown("**Timings (ms)**")
        st.bar_chart({k_: v for k_, v in res.get("timings_ms", {}).items() if k_ != "total"}, horizontal=True, height=200)
    with st.expander("Raw API response"):
        st.json({k_: v for k_, v in res.items() if k_ not in ("gps_points", "per_point_arc", "per_point_prob")}, expanded=False)
