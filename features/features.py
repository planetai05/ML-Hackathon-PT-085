"""Coordinate projection, candidate generation, and feature extraction for PathSnap."""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree

from data_processing.loader import load_trip


def project_to_local_meters(
    lon: np.ndarray, lat: np.ndarray, lon0: float, lat0: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Project (lon, lat) degrees to (x, y) meters using a local AEQD CRS centered at (lon0, lat0)."""
    proj_str = f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m +no_defs"
    transformer = Transformer.from_crs("EPSG:4326", proj_str, always_xy=True)
    x, y = transformer.transform(lon, lat)
    return np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)


FEATURE_COLS: List[str] = [
    "dist_m",
    "dist_rank",
    "dist_diff_from_min",
    "proj_t",
    "proj_t_unclamped",
    "arc_len_m",
    "heading_diff_deg",
    "speed_mps",
    "dt_prev",
    "step_dist_m",
    "prev_dist_m",
    "next_dist_m",
    "n_candidates",
]


def point_to_segment_distance_and_proj(
    px: np.ndarray,
    py: np.ndarray,
    ax: np.ndarray,
    ay: np.ndarray,
    bx: np.ndarray,
    by: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized exact distance (m) and projection parameter t from P to segment AB."""
    dx = bx - ax
    dy = by - ay
    len_sq = dx * dx + dy * dy
    safe_len_sq = np.where(len_sq > 1e-12, len_sq, 1.0)
    t_unclamped = ((px - ax) * dx + (py - ay) * dy) / safe_len_sq
    t_unclamped = np.where(len_sq > 1e-12, t_unclamped, 0.0)
    t_clamped = np.clip(t_unclamped, 0.0, 1.0)
    cx = ax + t_clamped * dx
    cy = ay + t_clamped * dy
    dist = np.hypot(px - cx, py - cy)
    return dist, t_clamped, t_unclamped


def _build_arc_sample_index(
    ax: np.ndarray,
    ay: np.ndarray,
    bx: np.ndarray,
    by: np.ndarray,
    arc_len: np.ndarray,
    step_m: float = 50.0,
) -> Tuple[cKDTree, np.ndarray]:
    """Build a cKDTree of points sampled along every arc so long arcs are never missed."""
    n_arcs = len(ax)
    # At least 3 samples (start, mid, end) per arc; more for long arcs
    counts = np.clip(np.ceil(arc_len / step_m).astype(np.int64) + 1, 3, 50)
    arc_ids = np.repeat(np.arange(n_arcs, dtype=np.int64), counts)

    # Compute normalized positions t in [0, 1] for each sample
    offsets = np.cumsum(counts) - counts
    local_idx = np.arange(len(arc_ids), dtype=np.float64) - np.repeat(offsets, counts)
    denom = np.repeat(counts - 1, counts).astype(np.float64)
    t = local_idx / denom

    sx = np.repeat(ax, counts) + t * np.repeat(bx - ax, counts)
    sy = np.repeat(ay, counts) + t * np.repeat(by - ay, counts)
    tree = cKDTree(np.column_stack([sx, sy]))
    return tree, arc_ids


def _compute_gps_kinematics(
    gx: np.ndarray, gy: np.ndarray, ts: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute dt_prev, step_dist_m, speed_mps, and gps_bearing_rad per GPS point."""
    n = len(gx)
    dt_prev = np.ones(n, dtype=np.float64)
    step_dist = np.zeros(n, dtype=np.float64)
    bearing = np.zeros(n, dtype=np.float64)

    if n > 1:
        dt = np.diff(ts)
        dt_prev[1:] = np.maximum(dt, 1e-3)
        dx = np.diff(gx)
        dy = np.diff(gy)
        dists = np.hypot(dx, dy)
        step_dist[1:] = dists

        seg_bearings = np.arctan2(dy, dx)
        bearing[0] = seg_bearings[0]
        bearing[1:] = seg_bearings
        # Carry forward last valid bearing when vehicle is stationary (< 1m step)
        for i in range(1, n):
            if step_dist[i] < 1.0:
                bearing[i] = bearing[i - 1]

    speed = step_dist / dt_prev
    return dt_prev, step_dist, speed, bearing


def extract_trip_features(
    trip_dir: Path,
    radius_m: float = 50.0,
    max_candidates: int = 15,
    fallback_radius_m: float = 150.0,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """Generate candidate road arcs and ML features for every GPS point in a trip.

    Does NOT read `.route`, so it can be used directly in live inference/API.
    Returns (features_df, trip_arrays_dict).
    """
    data = load_trip(trip_dir, load_route=False)
    trip_id = str(data["trip_id"])
    track: pd.DataFrame = data["track"]
    nodes: pd.DataFrame = data["nodes"]
    arcs: pd.DataFrame = data["arcs"]

    lon0 = float(np.median(track["lon"].to_numpy()))
    lat0 = float(np.median(track["lat"].to_numpy()))
    gx, gy = project_to_local_meters(track["lon"].to_numpy(), track["lat"].to_numpy(), lon0, lat0)
    nx, ny = project_to_local_meters(nodes["lon"].to_numpy(), nodes["lat"].to_numpy(), lon0, lat0)
    ts = track["timestamp"].to_numpy()

    u = arcs["from_node"].to_numpy()
    v = arcs["to_node"].to_numpy()
    ax, ay = nx[u], ny[u]
    bx, by = nx[v], ny[v]
    arc_dx = bx - ax
    arc_dy = by - ay
    arc_len = np.hypot(arc_dx, arc_dy)
    arc_bearing = np.arctan2(arc_dy, arc_dx)

    dt_prev, step_dist, speed, gps_bearing = _compute_gps_kinematics(gx, gy, ts)

    tree, sample_arc_ids = _build_arc_sample_index(ax, ay, bx, by, arc_len, step_m=50.0)
    gps_xy = np.column_stack([gx, gy])
    # Buffer by 30m so half-step between samples never misses an arc within radius_m
    hits_list = tree.query_ball_point(gps_xy, r=radius_m + 30.0)

    pt_idx_parts: List[np.ndarray] = []
    arc_id_parts: List[np.ndarray] = []
    dist_parts: List[np.ndarray] = []
    t_clamp_parts: List[np.ndarray] = []
    t_unclamp_parts: List[np.ndarray] = []
    rank_parts: List[np.ndarray] = []
    diff_min_parts: List[np.ndarray] = []
    n_cand_parts: List[np.ndarray] = []

    n_pts = len(gx)
    for i in range(n_pts):
        raw_hits = hits_list[i]
        cand_arcs = np.unique(sample_arc_ids[raw_hits]) if raw_hits else np.empty(0, dtype=np.int64)

        if len(cand_arcs) > 0:
            d, tc, tu = point_to_segment_distance_and_proj(
                np.full(len(cand_arcs), gx[i]),
                np.full(len(cand_arcs), gy[i]),
                ax[cand_arcs],
                ay[cand_arcs],
                bx[cand_arcs],
                by[cand_arcs],
            )
            mask = d <= radius_m
            cand_arcs, d, tc, tu = cand_arcs[mask], d[mask], tc[mask], tu[mask]
        else:
            d = tc = tu = np.empty(0, dtype=np.float64)

        # Fallback if no arc within radius_m (e.g. severe GPS noise)
        if len(cand_arcs) == 0:
            fb_hits = tree.query_ball_point(gps_xy[i], r=fallback_radius_m + 30.0)
            if fb_hits:
                cand_arcs = np.unique(sample_arc_ids[fb_hits])
                d, tc, tu = point_to_segment_distance_and_proj(
                    np.full(len(cand_arcs), gx[i]),
                    np.full(len(cand_arcs), gy[i]),
                    ax[cand_arcs],
                    ay[cand_arcs],
                    bx[cand_arcs],
                    by[cand_arcs],
                )

        if len(cand_arcs) == 0:
            continue

        order = np.argsort(d)[:max_candidates]
        cand_arcs = cand_arcs[order]
        d = d[order]
        tc = tc[order]
        tu = tu[order]

        k = len(cand_arcs)
        pt_idx_parts.append(np.full(k, i, dtype=np.int64))
        arc_id_parts.append(cand_arcs)
        dist_parts.append(d)
        t_clamp_parts.append(tc)
        t_unclamp_parts.append(tu)
        rank_parts.append(np.arange(1, k + 1, dtype=np.int64))
        diff_min_parts.append(d - d[0])
        n_cand_parts.append(np.full(k, k, dtype=np.int64))

    if not pt_idx_parts:
        return pd.DataFrame(), {"gx": gx, "gy": gy, "ts": ts, "u": u, "v": v, "arc_len": arc_len}

    pt_idx = np.concatenate(pt_idx_parts)
    arc_ids = np.concatenate(arc_id_parts)
    dist_m = np.concatenate(dist_parts)

    # Heading difference in degrees [0, 180]
    angle_diff = np.abs(gps_bearing[pt_idx] - arc_bearing[arc_ids])
    angle_diff = np.where(angle_diff > np.pi, 2.0 * np.pi - angle_diff, angle_diff)
    heading_diff_deg = np.degrees(angle_diff)

    # Neighbor GPS point distances to the same candidate arc
    prev_i = np.maximum(pt_idx - 1, 0)
    next_i = np.minimum(pt_idx + 1, n_pts - 1)
    prev_dist_m, _, _ = point_to_segment_distance_and_proj(
        gx[prev_i], gy[prev_i], ax[arc_ids], ay[arc_ids], bx[arc_ids], by[arc_ids]
    )
    next_dist_m, _, _ = point_to_segment_distance_and_proj(
        gx[next_i], gy[next_i], ax[arc_ids], ay[arc_ids], bx[arc_ids], by[arc_ids]
    )

    df = pd.DataFrame(
        {
            "trip_id": trip_id,
            "point_idx": pt_idx,
            "arc_id": arc_ids,
            "from_node": u[arc_ids],
            "to_node": v[arc_ids],
            "timestamp": ts[pt_idx],
            "dist_m": dist_m,
            "dist_rank": np.concatenate(rank_parts),
            "dist_diff_from_min": np.concatenate(diff_min_parts),
            "proj_t": np.concatenate(t_clamp_parts),
            "proj_t_unclamped": np.concatenate(t_unclamp_parts),
            "arc_len_m": arc_len[arc_ids],
            "heading_diff_deg": heading_diff_deg,
            "speed_mps": speed[pt_idx],
            "dt_prev": dt_prev[pt_idx],
            "step_dist_m": step_dist[pt_idx],
            "prev_dist_m": prev_dist_m,
            "next_dist_m": next_dist_m,
            "n_candidates": np.concatenate(n_cand_parts),
        }
    )

    meta = {
        "gx": gx,
        "gy": gy,
        "nx": nx,
        "ny": ny,
        "ts": ts,
        "u": u,
        "v": v,
        "arc_len": arc_len,
    }
    return df, meta
