"""Exploratory Data Analysis (EDA) for PathSnap.

Inspects all trips in `train` (80) and `student_test` (10), saves a trip-level
inventory CSV, generates summary plots, and builds an interactive Folium map.
"""

from pathlib import Path
from typing import Dict, List

import folium
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from data_processing.loader import DEFAULT_DATA_ROOT, list_trips, load_trip


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
VIS_DIR = REPO_ROOT / "visualisations"


def analyse_trip(partition: str, trip_dir: Path) -> Dict[str, object]:
    """Compute inventory, data-quality, connectivity, and route checks for one trip."""
    data = load_trip(trip_dir, load_route=True)
    trip_id = data["trip_id"]
    track: pd.DataFrame = data["track"]
    nodes: pd.DataFrame = data["nodes"]
    arcs: pd.DataFrame = data["arcs"]
    route: np.ndarray = data["route"]

    # File sizes
    track_bytes = (trip_dir / f"{trip_id}.track").stat().st_size
    nodes_bytes = (trip_dir / f"{trip_id}.nodes").stat().st_size
    arcs_bytes = (trip_dir / f"{trip_id}.arcs").stat().st_size
    route_bytes = (trip_dir / f"{trip_id}.route").stat().st_size

    # Missing / invalid values
    track_nan = int(track.isna().sum().sum())
    nodes_nan = int(nodes.isna().sum().sum())
    arcs_nan = int(arcs.isna().sum().sum())

    # Timestamp checks
    ts = track["timestamp"].to_numpy()
    dt = np.diff(ts) if len(ts) > 1 else np.array([], dtype=np.float64)
    n_non_increasing_ts = int(np.sum(dt <= 0))
    n_gaps_gt_1s = int(np.sum(dt > 1.0))
    pct_1s = float(np.mean(dt == 1.0) * 100.0) if len(dt) > 0 else 0.0
    max_dt = float(np.max(dt)) if len(dt) > 0 else 0.0
    duration_s = float(ts[-1] - ts[0]) if len(ts) > 1 else 0.0

    # Coordinate plausibility
    valid_track_coords = bool(
        (track["lon"].between(-180, 180) & track["lat"].between(-90, 90)).all()
    )

    # Road network connectivity (undirected weakly connected components)
    n_nodes = len(nodes)
    n_arcs = len(arcs)
    u = arcs["from_node"].to_numpy()
    v = arcs["to_node"].to_numpy()
    adj = csr_matrix(
        (np.ones(n_arcs, dtype=np.int8), (u, v)),
        shape=(n_nodes, n_nodes),
    )
    n_components, labels = connected_components(adj, directed=False, return_labels=True)
    comp_sizes = np.bincount(labels)
    largest_comp_nodes = int(comp_sizes.max()) if len(comp_sizes) > 0 else 0

    deg = np.bincount(u, minlength=n_nodes) + np.bincount(v, minlength=n_nodes)
    n_isolated_nodes = int(np.sum(deg == 0))

    # Reference route checks
    n_route = int(len(route)) if route is not None else 0
    route_valid_ids = bool(np.all((route >= 0) & (route < n_arcs))) if n_route > 0 else False
    if n_route > 1 and route_valid_ids:
        route_from = u[route]
        route_to = v[route]
        n_route_discontinuities = int(np.sum(route_to[:-1] != route_from[1:]))
    else:
        n_route_discontinuities = 0

    return {
        "partition": partition,
        "trip_id": trip_id,
        "n_gps": len(track),
        "n_nodes": n_nodes,
        "n_arcs": n_arcs,
        "n_route": n_route,
        "track_kb": round(track_bytes / 1024, 2),
        "nodes_kb": round(nodes_bytes / 1024, 2),
        "arcs_kb": round(arcs_bytes / 1024, 2),
        "route_kb": round(route_bytes / 1024, 2),
        "track_nan": track_nan,
        "nodes_nan": nodes_nan,
        "arcs_nan": arcs_nan,
        "valid_track_coords": valid_track_coords,
        "duration_s": round(duration_s, 1),
        "pct_dt_1s": round(pct_1s, 2),
        "n_non_increasing_ts": n_non_increasing_ts,
        "n_gaps_gt_1s": n_gaps_gt_1s,
        "max_dt_s": round(max_dt, 2),
        "lon_min": round(float(track["lon"].min()), 6),
        "lon_max": round(float(track["lon"].max()), 6),
        "lat_min": round(float(track["lat"].min()), 6),
        "lat_max": round(float(track["lat"].max()), 6),
        "n_components": int(n_components),
        "largest_comp_pct": round(100.0 * largest_comp_nodes / max(n_nodes, 1), 2),
        "n_isolated_nodes": n_isolated_nodes,
        "route_valid_ids": route_valid_ids,
        "n_route_discontinuities": n_route_discontinuities,
    }


def make_eda_plots(df: pd.DataFrame, all_dts: Dict[str, np.ndarray], out_path: Path) -> None:
    """Create a 2x2 summary figure covering both train and student_test partitions."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    colors = {"train": "#2563eb", "student_test": "#ea580c"}

    # 1. GPS observations per trip
    ax = axes[0, 0]
    for part in ["train", "student_test"]:
        sub = df[df["partition"] == part]
        ax.hist(sub["n_gps"], bins=15, alpha=0.65, label=f"{part} (n={len(sub)})", color=colors[part])
    ax.set_title("GPS Observations per Trip")
    ax.set_xlabel("Number of GPS points")
    ax.set_ylabel("Number of trips")
    ax.legend()

    # 2. GPS time gaps (dt > 1s)
    ax = axes[0, 1]
    for part in ["train", "student_test"]:
        gaps = all_dts[part][all_dts[part] > 1.0]
        if len(gaps) > 0:
            ax.hist(np.clip(gaps, 0, 30), bins=25, alpha=0.65, label=f"{part} (gaps={len(gaps)})", color=colors[part])
    ax.set_title("GPS Time Gaps (dt > 1s, clipped at 30s)")
    ax.set_xlabel("Time gap dt (seconds)")
    ax.set_ylabel("Count")
    ax.legend()

    # 3. Road network size (nodes vs arcs)
    ax = axes[1, 0]
    for part in ["train", "student_test"]:
        sub = df[df["partition"] == part]
        ax.scatter(sub["n_nodes"], sub["n_arcs"], alpha=0.75, label=part, color=colors[part])
    ax.set_title("Road Network Size per Trip")
    ax.set_xlabel("Number of nodes")
    ax.set_ylabel("Number of arcs")
    ax.legend()

    # 4. Route length vs GPS count
    ax = axes[1, 1]
    for part in ["train", "student_test"]:
        sub = df[df["partition"] == part]
        ax.scatter(sub["n_gps"], sub["n_route"], alpha=0.75, label=part, color=colors[part])
    ax.set_title("GPS Points vs Reference Route Arcs")
    ax.set_xlabel("Number of GPS points")
    ax.set_ylabel("Number of arcs in .route")
    ax.legend()

    fig.suptitle("PathSnap EDA Summary (Train: 80 trips, Student Test: 10 trips)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def make_folium_map(trip_dir: Path, out_html: Path, max_network_arcs: int = 4000) -> None:
    """Create an interactive Folium map of a trip: road network, reference route, and raw GPS."""
    data = load_trip(trip_dir, load_route=True)
    track: pd.DataFrame = data["track"]
    nodes: pd.DataFrame = data["nodes"]
    arcs: pd.DataFrame = data["arcs"]
    route: np.ndarray = data["route"]

    center_lat = float(track["lat"].mean())
    center_lon = float(track["lon"].mean())
    m = folium.Map(location=[center_lat, center_lon], zoom_start=14, tiles="OpenStreetMap")

    # Filter road network arcs to the GPS bounding box (+ small pad) so HTML stays fast
    pad = 0.005
    lon_min, lon_max = track["lon"].min() - pad, track["lon"].max() + pad
    lat_min, lat_max = track["lat"].min() - pad, track["lat"].max() + pad

    node_lon = nodes["lon"].to_numpy()
    node_lat = nodes["lat"].to_numpy()
    u = arcs["from_node"].to_numpy()
    v = arcs["to_node"].to_numpy()

    in_box = (
        (node_lon[u] >= lon_min) & (node_lon[u] <= lon_max) &
        (node_lat[u] >= lat_min) & (node_lat[u] <= lat_max)
    )
    box_idx = np.where(in_box)[0][:max_network_arcs]

    net_group = folium.FeatureGroup(name="Road Network (nearby)", show=True)
    for idx in box_idx:
        p1 = [float(node_lat[u[idx]]), float(node_lon[u[idx]])]
        p2 = [float(node_lat[v[idx]]), float(node_lon[v[idx]])]
        folium.PolyLine([p1, p2], color="#9ca3af", weight=1.5, opacity=0.6).add_to(net_group)
    net_group.add_to(m)

    # Reference route (green)
    if route is not None and len(route) > 0:
        route_group = folium.FeatureGroup(name="Reference Route (.route)", show=True)
        for aid in route:
            p1 = [float(node_lat[u[aid]]), float(node_lon[u[aid]])]
            p2 = [float(node_lat[v[aid]]), float(node_lon[v[aid]])]
            folium.PolyLine([p1, p2], color="#16a34a", weight=5, opacity=0.85).add_to(route_group)
        route_group.add_to(m)

    # Raw GPS observations (blue dots)
    gps_group = folium.FeatureGroup(name="Raw GPS Points (.track)", show=True)
    for i, row in track.iterrows():
        folium.CircleMarker(
            location=[float(row["lat"]), float(row["lon"])],
            radius=3,
            color="#2563eb",
            fill=True,
            fill_color="#2563eb",
            fill_opacity=0.85,
            popup=f"pt={i}, t={row['timestamp']}s",
        ).add_to(gps_group)
    gps_group.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    m.save(str(out_html))


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    VIS_DIR.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    all_dts: Dict[str, List[np.ndarray]] = {"train": [], "student_test": []}

    for partition in ["train", "student_test"]:
        trips = list_trips(DEFAULT_DATA_ROOT / partition)
        for trip_dir in trips:
            rows.append(analyse_trip(partition, trip_dir))
            tr = load_trip(trip_dir, load_route=False)["track"]
            if len(tr) > 1:
                all_dts[partition].append(np.diff(tr["timestamp"].to_numpy()))

    df = pd.DataFrame(rows)
    inv_path = RESULTS_DIR / "eda_inventory.csv"
    df.to_csv(inv_path, index=False)

    dt_arrays = {k: np.concatenate(v) if v else np.array([]) for k, v in all_dts.items()}
    plot_path = VIS_DIR / "eda_summary.png"
    make_eda_plots(df, dt_arrays, plot_path)

    sample_trip = list_trips(DEFAULT_DATA_ROOT / "train")[0]
    map_path = VIS_DIR / f"eda_trip_{sample_trip.name}.html"
    make_folium_map(sample_trip, map_path)

    print("=== EDA Summary by Partition ===")
    summary = df.groupby("partition").agg(
        trips=("trip_id", "count"),
        gps_min=("n_gps", "min"),
        gps_median=("n_gps", "median"),
        gps_max=("n_gps", "max"),
        nodes_median=("n_nodes", "median"),
        arcs_median=("n_arcs", "median"),
        route_median=("n_route", "median"),
        total_nan=("track_nan", "sum"),
        non_inc_ts=("n_non_increasing_ts", "sum"),
        gaps_gt_1s=("n_gaps_gt_1s", "sum"),
        max_dt_s=("max_dt_s", "max"),
        route_discont=("n_route_discontinuities", "sum"),
    )
    print(summary.to_string())
    print(f"\nSaved inventory: {inv_path}")
    print(f"Saved plot:      {plot_path}")
    print(f"Saved Folium:    {map_path}")


if __name__ == "__main__":
    main()
