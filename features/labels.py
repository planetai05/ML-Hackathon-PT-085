"""Route-to-point label alignment (issue #9).

The `.route` file is an ordered list of arc ids and is NOT aligned one-to-one with GPS points.
We assign each GPS point i a position j(i) in the route such that j is non-decreasing in time
(the vehicle never moves backwards along its own route) and the total point-to-arc distance is
minimal, with a small penalty per skipped route position. Solved with dynamic programming over a
sparse state space: for point i only route positions whose arc lies within `search_radius` m are
considered. Points with no such position, or whose aligned distance exceeds `max_dist`, are marked
unaligned and excluded from training and evaluation.

Leakage: labels are used only as targets / for scoring. No feature reads the route.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from features.trip_io import Trip, point_segment


def align_route(trip: Trip, search_radius: float = 60.0, max_dist: float = 40.0, skip_penalty: float = 0.5, step: float = 5.0):
    """Return (true_arc[N] (-1 if unaligned), aligned[N] bool, align_dist[N])."""
    route = trip.route
    N = len(trip.xy)
    true_arc = np.full(N, -1, dtype=np.int64)
    dist_out = np.full(N, np.inf)
    if route is None or len(route) == 0:
        return true_arc, np.zeros(N, bool), dist_out
    M = len(route)
    a, b = trip.node_xy[trip.arc_from[route]], trip.node_xy[trip.arc_to[route]]
    L = np.linalg.norm(b - a, axis=1)
    n = np.maximum(1, np.ceil(L / step).astype(int)) + 1
    owner = np.repeat(np.arange(M), n)
    start = np.repeat(np.cumsum(n) - n, n)
    frac = (np.arange(n.sum()) - start) / np.repeat(n - 1, n)
    pts = a[owner] + (b - a)[owner] * frac[:, None]
    tree = cKDTree(pts)
    hits = tree.query_ball_point(trip.xy, r=search_radius + step, workers=-1)

    states, costs, backs = [], [], []
    prev_pos = prev_val = None
    prev_i = -1
    for i in range(N):
        h = hits[i]
        if not h:
            states.append(None); costs.append(None); backs.append(None)
            continue
        pos = np.unique(owner[np.asarray(h)])
        d, _, _ = point_segment(trip.xy[i, 0], trip.xy[i, 1], a[pos, 0], a[pos, 1], b[pos, 0], b[pos, 1])
        ok = d <= search_radius
        pos, d = pos[ok], d[ok]
        if len(pos) == 0:
            states.append(None); costs.append(None); backs.append(None)
            continue
        if prev_pos is None:
            val = d + skip_penalty * pos
            back = np.full(len(pos), -1)
        else:
            # min_{j' <= j} prev_val(j') + penalty*(j - j')
            adj = prev_val - skip_penalty * prev_pos
            pm = np.minimum.accumulate(adj)
            pa = np.zeros(len(adj), int)
            best = 0
            for q in range(1, len(adj)):
                if adj[q] < adj[best]:
                    best = q
                pa[q] = best
            k = np.searchsorted(prev_pos, pos, side="right") - 1
            val = np.where(k >= 0, d + pm[np.maximum(k, 0)] + skip_penalty * pos, np.inf)
            back = np.where(k >= 0, pa[np.maximum(k, 0)], -1)
            if not np.isfinite(val).any():  # cannot continue monotonically -> restart
                val = d + skip_penalty * pos
                back = np.full(len(pos), -1)
        states.append((pos, d, prev_i)); costs.append(val); backs.append(back)
        prev_pos, prev_val, prev_i = pos, val, i

    # backtrack from the last aligned point
    i = prev_i
    if i < 0:
        return true_arc, np.zeros(N, bool), dist_out
    q = int(np.argmin(costs[i]))
    while i >= 0:
        pos, d, pi = states[i]
        true_arc[i] = route[pos[q]]
        dist_out[i] = d[q]
        q = backs[i][q]
        if q < 0:
            # restart point: continue backtracking from best state of previous aligned point
            if pi >= 0:
                q = int(np.argmin(costs[pi]))
        i = pi
    aligned = (true_arc >= 0) & (dist_out <= max_dist)
    true_arc[~aligned] = -1
    return true_arc, aligned, dist_out
