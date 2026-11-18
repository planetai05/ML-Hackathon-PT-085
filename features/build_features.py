"""Feature engineering + cached parquet (issue #10).

One row per (GPS point, candidate arc). The SAME `trip_features` function is used for training
and at inference (models/predictor.py), so preprocessing is identical. Features never use the
reference route. Scaling is done inside sklearn Pipelines fitted on training folds only.

Run:  python -m features.build_features
Writes cache/features_train.parquet, cache/features_student_test.parquet,
results/candidate_stats.csv, artifacts/feature_config.json
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from features.candidates import DEFAULTS, generate_candidates, miss_rate
from features.labels import align_route
from features.trip_io import Trip, list_trips, load_trip, point_segment

FEATURES = [
    # distance
    "dist", "dist_rank", "dist_minus_min", "dist_ratio", "n_cand",
    # position along the road segment
    "proj_t", "beyond_end", "arc_len", "log_arc_len",
    # heading
    "heading_diff", "cos_heading", "heading_rank", "heading_valid",
    "heading_diff_w", "along_w", "along_w_m", "disp_w",
    # speed / time
    "speed", "accel", "dt_prev", "dt_next",
    # spatial consistency with neighbouring observations
    "dist_prev_pt", "dist_next_pt", "dist_ctx_mean", "conn_prev_nearest", "conn_next_nearest",
    # road-network topology
    "out_deg_to", "in_deg_from",
]
GAP_S = 10.0  # time gap (s) treated as a break for context features
WIN = 5       # +/- observations for the robust (windowed) heading


def _bearing(dx, dy):
    return np.degrees(np.arctan2(dx, dy)) % 360


def point_kinematics(trip: Trip):
    xy, t = trip.xy, trip.track[:, 2]
    n = len(xy)
    dt = np.diff(t)
    step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    dt_prev = np.r_[np.nan, dt]
    dt_next = np.r_[dt, np.nan]
    v_seg = step / np.maximum(dt, 1e-3)
    speed = np.empty(n)
    speed[0] = v_seg[0] if n > 1 else 0
    speed[-1] = v_seg[-1] if n > 1 else 0
    if n > 2:
        speed[1:-1] = (v_seg[:-1] + v_seg[1:]) / 2
    accel = np.r_[0, np.diff(speed) / np.maximum(dt, 1e-3)] if n > 1 else np.zeros(n)
    # heading from previous to next point (central difference), fallback to one-sided
    prv = np.r_[0, np.arange(n - 1)]
    nxt = np.r_[np.arange(1, n), n - 1]
    d = xy[nxt] - xy[prv]
    heading = _bearing(d[:, 0], d[:, 1])
    disp = np.linalg.norm(d, axis=1)
    # robust heading over a +/-WIN window (stationary GPS jitter makes the 1-step heading random)
    t_ = trip.track[:, 2]
    pw = np.clip(np.arange(n) - WIN, 0, n - 1)
    nw = np.clip(np.arange(n) + WIN, 0, n - 1)
    # shrink the window so it never spans a time gap
    pw = np.where(t_ - t_[pw] > WIN * 3, np.arange(n), pw)
    nw = np.where(t_[nw] - t_ > WIN * 3, np.arange(n), nw)
    dw = xy[nw] - xy[pw]
    return dict(speed=np.nan_to_num(speed), accel=np.clip(np.nan_to_num(accel), -20, 20),
                dwx=dw[:, 0], dwy=dw[:, 1], disp_w=np.linalg.norm(dw, axis=1),
                dt_prev=np.nan_to_num(dt_prev, nan=1.0), dt_next=np.nan_to_num(dt_next, nan=1.0),
                heading=heading, heading_valid=(disp > 3.0).astype(float))


def trip_features(trip: Trip, cands: pd.DataFrame | None = None, radius=DEFAULTS["radius"], k=DEFAULTS["k"]):
    """Return candidate DataFrame with FEATURES columns (no labels)."""
    if cands is None:
        cands = generate_candidates(trip, radius, k)
    df = cands.copy()
    if len(df) == 0:
        for f in FEATURES:
            df[f] = []
        return df
    kin = point_kinematics(trip)
    pi = df.point_idx.to_numpy()
    arc = df.arc_id.to_numpy()
    n = len(trip.xy)
    g = df.groupby("point_idx")
    df["dist_rank"] = df["rank"].astype(float)
    mn = g.dist.transform("min").to_numpy()
    df["dist_minus_min"] = df.dist.to_numpy() - mn
    df["dist_ratio"] = (df.dist.to_numpy() + 1) / (mn + 1)
    df["n_cand"] = g.dist.transform("size").astype(float)
    df["proj_t"] = df.t
    tr = df.t_raw.to_numpy()
    L = trip.arc_len[arc]
    df["beyond_end"] = np.maximum(0, np.maximum(-tr, tr - 1)) * L  # metres beyond segment end
    df["arc_len"] = L
    df["log_arc_len"] = np.log1p(L)
    hd = np.abs(kin["heading"][pi] - trip.arc_bearing[arc]) % 360
    hd = np.minimum(hd, 360 - hd)
    df["heading_valid"] = kin["heading_valid"][pi]
    df["heading_diff"] = np.where(df.heading_valid > 0, hd, 90.0)  # unknown heading -> neutral value
    df["cos_heading"] = np.where(df.heading_valid > 0, np.cos(np.radians(hd)), 0.0)
    df["heading_rank"] = df.groupby("point_idx").heading_diff.rank(method="first") - 1
    # windowed heading / signed progress along the arc direction
    ab = trip.arc_b[arc] - trip.arc_a[arc]
    ux, uy = ab[:, 0] / np.maximum(L, 1e-6), ab[:, 1] / np.maximum(L, 1e-6)
    dwx, dwy, dsw = kin["dwx"][pi], kin["dwy"][pi], kin["disp_w"][pi]
    along = dwx * ux + dwy * uy
    df["disp_w"] = dsw
    df["along_w_m"] = along
    df["along_w"] = along / np.maximum(dsw, 1.0)
    hw = np.abs(_bearing(dwx, dwy) - trip.arc_bearing[arc]) % 360
    hw = np.minimum(hw, 360 - hw)
    df["heading_diff_w"] = np.where(dsw > 5.0, hw, 90.0)
    for f in ("speed", "accel", "dt_prev", "dt_next"):
        df[f] = kin[f][pi]
    # spatial consistency: distance from neighbouring observations to the same arc
    a, b = trip.arc_a[arc], trip.arc_b[arc]
    big = 100.0
    for name, off, dtk in (("dist_prev_pt", -1, "dt_prev"), ("dist_next_pt", 1, "dt_next")):
        j = np.clip(pi + off, 0, n - 1)
        dd, _, _ = point_segment(trip.xy[j, 0], trip.xy[j, 1], a[:, 0], a[:, 1], b[:, 0], b[:, 1])
        valid = (j != pi) & (kin[dtk][pi] <= GAP_S)
        df[name] = np.where(valid, np.minimum(dd, big), df.dist.to_numpy())
    df["dist_ctx_mean"] = (df.dist + df.dist_prev_pt + df.dist_next_pt) / 3
    # connectivity to the nearest candidate of neighbouring points
    nearest = df[df["rank"] == 0].set_index("point_idx").arc_id
    near_arr = np.full(n, -1, dtype=np.int64)
    near_arr[nearest.index.to_numpy()] = nearest.to_numpy()
    p_near = near_arr[np.clip(pi - 1, 0, n - 1)]
    n_near = near_arr[np.clip(pi + 1, 0, n - 1)]
    af, at = trip.arc_from, trip.arc_to
    df["conn_prev_nearest"] = ((p_near >= 0) & ((p_near == arc) | (at[np.maximum(p_near, 0)] == af[arc]))).astype(float)
    df["conn_next_nearest"] = ((n_near >= 0) & ((n_near == arc) | (af[np.maximum(n_near, 0)] == at[arc]))).astype(float)
    outd, ind = trip.degrees()
    df["out_deg_to"] = outd[at[arc]].astype(float)
    df["in_deg_from"] = ind[af[arc]].astype(float)
    return df


def process_trip(trip_dir: str):
    t0 = time.perf_counter()
    trip = load_trip(trip_dir, with_route=True)
    cands = generate_candidates(trip)
    true_arc, aligned, adist = align_route(trip)
    df = trip_features(trip, cands)
    df["trip_id"] = trip.trip_id
    df["aligned"] = aligned[df.point_idx.to_numpy()]
    df["label"] = (df.arc_id.to_numpy() == true_arc[df.point_idx.to_numpy()]).astype(np.int8)
    n = len(trip.xy)
    stats = dict(trip_id=trip.trip_id, n_points=n, n_arcs=trip.n_arcs, n_route_arcs=len(trip.route),
                 n_candidate_rows=len(df), mean_candidates=len(df) / n,
                 empty_candidate_points=int(n - df.point_idx.nunique()),
                 aligned_frac=float(aligned.mean()),
                 median_align_dist=float(np.median(adist[aligned])) if aligned.any() else np.nan,
                 candidate_miss_rate=float(miss_rate(cands, true_arc, aligned)),
                 seconds=time.perf_counter() - t0)
    keep = ["trip_id", "point_idx", "arc_id", "aligned", "label"] + FEATURES
    return df[keep], stats


def main():
    os.makedirs("cache", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)
    all_stats = []
    for split in ("train", "student_test"):
        t0 = time.time()
        out = Parallel(n_jobs=-1)(delayed(process_trip)(d) for d in list_trips(split))
        df = pd.concat([o[0] for o in out], ignore_index=True)
        for c in FEATURES:
            df[c] = df[c].astype(np.float32)
        df.to_parquet(f"cache/features_{split}.parquet", index=False)
        st = pd.DataFrame([o[1] for o in out]).assign(split=split)
        all_stats.append(st)
        al = df[df.aligned]
        pts = df.groupby(["trip_id", "point_idx"]).size().shape[0]
        print(f"{split}: {len(df):,} rows, {pts:,} points with candidates, pos rate {al.label.mean():.3f}, "
              f"miss rate (pooled) {(st.candidate_miss_rate * st.n_points * st.aligned_frac).sum() / (st.n_points * st.aligned_frac).sum():.4f}, "
              f"{time.time() - t0:.1f}s")
    pd.concat(all_stats).to_csv("results/candidate_stats.csv", index=False)
    json.dump({"features": FEATURES, "candidate_params": DEFAULTS, "gap_s": GAP_S, "crs": "per-trip UTM zone from track centroid (EPSG:326xx/327xx)",
               "label_alignment": {"search_radius": 60.0, "max_dist": 40.0, "skip_penalty": 0.5}},
              open("artifacts/feature_config.json", "w"), indent=1)


if __name__ == "__main__":
    main()
