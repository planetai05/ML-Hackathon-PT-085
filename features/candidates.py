"""Candidate generation (issue #8).

For every GPS point, return the arcs within `radius` metres (UTM), capped at the `k` nearest.
Spatial index: arcs near the track are densified into sample points every `step` m and
indexed with a cKDTree; exact point-to-segment distances are then computed for the hits.
Candidate arc ids are the original row indices of the `.arcs` file.
Points with no arc inside `radius` get a second query with `2 * radius`; if still empty they
have no candidates (handled downstream by the HMM as a segment break).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from features.trip_io import Trip, point_segment

DEFAULTS = dict(radius=40.0, k=12, step=10.0)


class ArcIndex:
    """Spatial index over the arcs of one trip that lie near its GPS track (cached per trip)."""

    def __init__(self, trip: Trip, margin: float = 200.0, step: float = 10.0):
        a, b = trip.arc_a, trip.arc_b
        lo = trip.xy.min(0) - margin
        hi = trip.xy.max(0) + margin
        amin, amax = np.minimum(a, b), np.maximum(a, b)
        sel = np.where((amax[:, 0] >= lo[0]) & (amin[:, 0] <= hi[0]) & (amax[:, 1] >= lo[1]) & (amin[:, 1] <= hi[1]))[0]
        self.arc_ids = sel
        L = trip.arc_len[sel]
        n = np.maximum(1, np.ceil(L / step).astype(int)) + 1
        owner = np.repeat(np.arange(len(sel)), n)
        start = np.repeat(np.cumsum(n) - n, n)
        frac = (np.arange(n.sum()) - start) / np.repeat(n - 1, n)
        pa, pb = a[sel][owner], b[sel][owner]
        self.pts = pa + (pb - pa) * frac[:, None]
        self.owner = owner
        self.tree = cKDTree(self.pts)
        self.step = step
        self.trip = trip

    def query(self, xy: np.ndarray, radius: float, k: int):
        """Return DataFrame(point_idx, arc_id, dist, t, t_raw, rank) for the given points."""
        hits = self.tree.query_ball_point(xy, r=radius + self.step / 2 + 1e-6, workers=-1)
        lens = np.fromiter((len(h) for h in hits), int, len(hits))
        if lens.sum() == 0:
            return pd.DataFrame(columns=["point_idx", "arc_id", "dist", "t", "t_raw", "rank"])
        pi = np.repeat(np.arange(len(xy)), lens)
        loc = self.owner[np.concatenate([np.asarray(h, dtype=np.int64) for h in hits if len(h)])]
        pairs = np.unique(pi * (len(self.arc_ids) + 1) + loc)
        pi, loc = pairs // (len(self.arc_ids) + 1), pairs % (len(self.arc_ids) + 1)
        arc = self.arc_ids[loc]
        tr = self.trip
        a, b = tr.arc_a[arc], tr.arc_b[arc]
        d, t, t_raw = point_segment(xy[pi, 0], xy[pi, 1], a[:, 0], a[:, 1], b[:, 0], b[:, 1])
        df = pd.DataFrame({"point_idx": pi, "arc_id": arc, "dist": d, "t": t, "t_raw": t_raw})
        df = df[df.dist <= radius].sort_values(["point_idx", "dist", "arc_id"], kind="stable")
        df["rank"] = df.groupby("point_idx").cumcount()
        return df[df["rank"] < k].reset_index(drop=True)


def generate_candidates(trip: Trip, radius=DEFAULTS["radius"], k=DEFAULTS["k"], index: ArcIndex | None = None):
    index = index or ArcIndex(trip, margin=2 * radius + 50, step=DEFAULTS["step"])
    df = index.query(trip.xy, radius, k)
    missing = np.setdiff1d(np.arange(len(trip.xy)), df.point_idx.unique())
    if len(missing):
        df2 = index.query(trip.xy[missing], 2 * radius, k)
        if len(df2):
            df2["point_idx"] = missing[df2.point_idx.to_numpy()]
            df = pd.concat([df, df2], ignore_index=True).sort_values(["point_idx", "rank"], kind="stable")
    return df.reset_index(drop=True)


def miss_rate(cands: pd.DataFrame, true_arc: np.ndarray, aligned: np.ndarray):
    """Share of aligned points whose true arc is NOT among their candidates."""
    idx = np.where(aligned)[0]
    if len(idx) == 0:
        return np.nan
    hit = cands[cands.arc_id.to_numpy() == true_arc[cands.point_idx.to_numpy()]].point_idx.unique()
    return 1.0 - np.isin(idx, hit).mean()
