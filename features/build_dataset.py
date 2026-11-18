"""Align .route to GPS points, label candidates, check miss rate, and export parquet."""

from pathlib import Path
from typing import Dict, List

import folium
import numpy as np
import pandas as pd

from data_processing.loader import DEFAULT_DATA_ROOT, list_trips, load_trip
from features.features import (
    extract_trip_features,
    point_to_segment_distance_and_proj,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_OUT_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"
VIS_DIR = REPO_ROOT / "visualisations"

# Max distance (m) between a GPS point and its DP-aligned reference arc for the label to be trusted
LABEL_VALID_MAX_DIST_M = 50.0


def align_gps_to_route(
    gx: np.ndarray,
    gy: np.ndarray,
    ax: np.ndarray,
    ay: np.ndarray,
    bx: np.ndarray,
    by: np.ndarray,
    route: np.ndarray,
) -> np.ndarray:
    """Find the globally optimal monotonic alignment of GPS points to .route arcs via DP.

    Returns an array of shape (len(gx),) with the aligned ground-truth arc_id per GPS point.
    """
    n_pts = len(gx)
    n_route = len(route)
    if n_route == 0 or n_pts == 0:
        return np.full(n_pts, -1, dtype=np.int64)

    rax = ax[route]
    ray = ay[route]
    rbx = bx[route]
    rby = by[route]

    # Distance matrix: shape (n_pts, n_route)
    dist_mat = np.empty((n_pts, n_route), dtype=np.float64)
    for j in range(n_route):
        d, _, _ = point_to_segment_distance_and_proj(
            gx,
            gy,
            np.full(n_pts, rax[j]),
            np.full(n_pts, ray[j]),
            np.full(n_pts, rbx[j]),
            np.full(n_pts, rby[j]),
        )
        dist_mat[:, j] = d

    # Monotonic DP: j_0 <= j_1 <= ... <= j_{n_pts-1}
    dp = np.empty_like(dist_mat)
    back = np.empty((n_pts, n_route), dtype=np.int32)
    dp[0] = dist_mat[0]
    back[0] = np.arange(n_route, dtype=np.int32)

    for i in range(1, n_pts):
        # Running prefix minimum and argmin over prev row
        prev = dp[i - 1]
        best_val = np.minimum.accumulate(prev)
        # Compute running argmin
        best_idx = np.empty(n_route, dtype=np.int32)
        cur_min = prev[0]
        cur_arg = 0
        best_idx[0] = 0
        for j in range(1, n_route):
            if prev[j] < cur_min:
                cur_min = prev[j]
                cur_arg = j
            best_idx[j] = cur_arg
        dp[i] = dist_mat[i] + best_val
        back[i] = best_idx

    aligned_route_pos = np.empty(n_pts, dtype=np.int32)
    aligned_route_pos[-1] = int(np.argmin(dp[-1]))
    for i in range(n_pts - 2, -1, -1):
        aligned_route_pos[i] = back[i + 1, aligned_route_pos[i + 1]]

    return route[aligned_route_pos]


def make_label_check_map(
    trip_dir: Path,
    true_arcs: np.ndarray,
    df_trip: pd.DataFrame,
    out_html: Path,
) -> None:
    """Visualise .route, raw GPS, and any missed-candidate GPS points on Folium."""
    data = load_trip(trip_dir, load_route=True)
    track: pd.DataFrame = data["track"]
    nodes: pd.DataFrame = data["nodes"]
    arcs: pd.DataFrame = data["arcs"]
    route: np.ndarray = data["route"]

    node_lon = nodes["lon"].to_numpy()
    node_lat = nodes["lat"].to_numpy()
    u = arcs["from_node"].to_numpy()
    v = arcs["to_node"].to_numpy()

    m = folium.Map(
        location=[float(track["lat"].mean()), float(track["lon"].mean())],
        zoom_start=14,
        tiles="OpenStreetMap",
    )

    route_group = folium.FeatureGroup(name="Reference Route (.route)", show=True)
    for aid in route:
        folium.PolyLine(
            [[float(node_lat[u[aid]]), float(node_lon[u[aid]])],
             [float(node_lat[v[aid]]), float(node_lon[v[aid]])]],
            color="#16a34a",
            weight=5,
            opacity=0.8,
        ).add_to(route_group)
    route_group.add_to(m)

    hit_pts = set(df_trip.loc[df_trip["label"] == 1, "point_idx"].tolist())
    ok_group = folium.FeatureGroup(name="Matched GPS (in candidates)", show=True)
    miss_group = folium.FeatureGroup(name="Missed GPS (true arc not in candidates)", show=True)

    for i, row in track.iterrows():
        is_hit = i in hit_pts
        target_group = ok_group if is_hit else miss_group
        folium.CircleMarker(
            location=[float(row["lat"]), float(row["lon"])],
            radius=3 if is_hit else 5,
            color="#2563eb" if is_hit else "#dc2626",
            fill=True,
            fill_color="#2563eb" if is_hit else "#dc2626",
            fill_opacity=0.85,
            popup=f"pt={i}, true_arc={true_arcs[i]}, hit={is_hit}",
        ).add_to(target_group)

    ok_group.add_to(m)
    miss_group.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.save(str(out_html))


def process_partition(partition: str) -> pd.DataFrame:
    """Build labeled candidate feature dataframe for a whole partition."""
    trips = list_trips(DEFAULT_DATA_ROOT / partition)
    dfs: List[pd.DataFrame] = []
    miss_rows: List[Dict[str, object]] = []

    for idx, trip_dir in enumerate(trips):
        trip_data = load_trip(trip_dir, load_route=True)
        route = trip_data["route"]
        nodes = trip_data["nodes"]
        arcs = trip_data["arcs"]

        df_trip, meta = extract_trip_features(trip_dir, radius_m=50.0, max_candidates=15)
        nx, ny = meta["nx"], meta["ny"]
        u = meta["u"]
        v = meta["v"]

        true_arcs = align_gps_to_route(
            meta["gx"], meta["gy"], nx[u], ny[u], nx[v], ny[v], route
        )

        pt_idx = df_trip["point_idx"].to_numpy()
        df_trip["true_arc_id"] = true_arcs[pt_idx]
        df_trip["label"] = (df_trip["arc_id"].to_numpy() == df_trip["true_arc_id"].to_numpy()).astype(np.int8)

        # A point's label is only trustworthy if the aligned reference arc is physically close
        # to the GPS point. Otherwise the route ended/was truncated and the DP pinned the point
        # to the last route arcs, so the all-zero labels would be wrong.
        aligned_dist, _, _ = point_to_segment_distance_and_proj(
            meta["gx"], meta["gy"],
            nx[u[true_arcs]], ny[u[true_arcs]], nx[v[true_arcs]], ny[v[true_arcs]],
        )
        point_valid = aligned_dist <= LABEL_VALID_MAX_DIST_M
        df_trip["aligned_dist_m"] = aligned_dist[pt_idx]
        df_trip["label_valid"] = point_valid[pt_idx]

        n_pts = len(meta["gx"])
        n_valid = int(point_valid.sum())
        hit_pts = np.unique(df_trip.loc[df_trip["label"] == 1, "point_idx"].to_numpy())
        n_hits = int(len(hit_pts))
        n_hits_valid = int(point_valid[hit_pts].sum())
        miss_rate = 100.0 * (n_pts - n_hits) / max(n_pts, 1)
        miss_rate_valid = 100.0 * (n_valid - n_hits_valid) / max(n_valid, 1)

        miss_rows.append(
            {
                "partition": partition,
                "trip_id": trip_data["trip_id"],
                "n_gps": n_pts,
                "n_candidate_rows": len(df_trip),
                "n_hits": n_hits,
                "n_misses": n_pts - n_hits,
                "miss_rate_pct": round(miss_rate, 3),
                "n_valid_pts": n_valid,
                "pct_valid_pts": round(100.0 * n_valid / max(n_pts, 1), 2),
                "n_misses_valid": n_valid - n_hits_valid,
                "miss_rate_valid_pct": round(miss_rate_valid, 3),
            }
        )
        dfs.append(df_trip)

        if partition == "train" and idx == 0:
            make_label_check_map(
                trip_dir,
                true_arcs,
                df_trip,
                VIS_DIR / f"labels_check_{trip_data['trip_id']}.html",
            )

    full_df = pd.concat(dfs, ignore_index=True)
    return full_df, pd.DataFrame(miss_rows)


def main() -> None:
    DATA_OUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    VIS_DIR.mkdir(parents=True, exist_ok=True)

    train_df, train_miss = process_partition("train")
    test_df, test_miss = process_partition("student_test")

    train_path = DATA_OUT_DIR / "train_features.parquet"
    test_path = DATA_OUT_DIR / "student_test_features.parquet"
    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)

    miss_df = pd.concat([train_miss, test_miss], ignore_index=True)
    miss_path = RESULTS_DIR / "candidate_miss_rate.csv"
    miss_df.to_csv(miss_path, index=False)

    for part, mdf, fdf in [("train", train_miss, train_df), ("student_test", test_miss, test_df)]:
        tot_pts = mdf["n_gps"].sum()
        tot_miss = mdf["n_misses"].sum()
        tot_valid = mdf["n_valid_pts"].sum()
        tot_miss_valid = mdf["n_misses_valid"].sum()
        vdf = fdf[fdf["label_valid"]]
        print(
            f"[{part}] rows={len(fdf):,}, pts={tot_pts:,}, "
            f"raw miss rate={100.0 * tot_miss / tot_pts:.2f}% | "
            f"valid pts={tot_valid:,} ({100.0 * tot_valid / tot_pts:.2f}%), "
            f"miss rate on valid pts={100.0 * tot_miss_valid / max(tot_valid, 1):.2f}%, "
            f"positive_label (valid rows)={100.0 * vdf['label'].mean():.2f}%"
        )

    print(f"\nSaved: {train_path}")
    print(f"Saved: {test_path}")
    print(f"Saved: {miss_path}")


if __name__ == "__main__":
    main()
