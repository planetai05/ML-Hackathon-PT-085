"""Viterbi decoding and connected route reconstruction (issue #17).

decode(trip, cand_df, prob, cfg) -> dict with
  per_point_arc  : decoded arc per GPS point (-1 when the point has no candidate)
  per_point_prob : classifier probability of the decoded arc
  segments       : list of continuous arc sequences (gap-filled with shortest paths)
  route_arcs     : concatenation of segments (ordered, consecutive duplicates removed)
  n_breaks       : number of chain restarts (time gaps / empty candidate sets / unreachable)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from features.trip_io import Trip
from hmm.model import HMMConfig, RoadGraph, emission_logp, transition_logp


def _steps(trip: Trip, cand_df: pd.DataFrame, prob: np.ndarray):
    """Group candidates per point -> list over points of (arcs, t, prob, dist) or None."""
    n = len(trip.xy)
    steps = [None] * n
    d = cand_df.assign(_p=prob)
    pi = d.point_idx.to_numpy()
    order = np.argsort(pi, kind="stable")
    pi = pi[order]
    arcs, ts, ps, ds = (d.arc_id.to_numpy()[order], d.t.to_numpy()[order], d._p.to_numpy()[order], d.dist.to_numpy()[order])
    bounds = np.flatnonzero(np.r_[True, pi[1:] != pi[:-1], True])
    for s, e in zip(bounds[:-1], bounds[1:]):
        steps[pi[s]] = (arcs[s:e], ts[s:e], ps[s:e], ds[s:e])
    return steps


def decode(trip: Trip, cand_df: pd.DataFrame, prob: np.ndarray, cfg: HMMConfig | None = None, graph: RoadGraph | None = None):
    cfg = cfg or HMMConfig()
    g = graph or RoadGraph(trip)
    steps = _steps(trip, cand_df, prob)
    n = len(trip.xy)
    t = trip.track[:, 2]
    per_arc = np.full(n, -1, dtype=np.int64)
    per_prob = np.full(n, np.nan)
    n_breaks = 0

    # split observations into chains
    chains, cur = [], []
    for i in range(n):
        if steps[i] is None:
            if cur:
                chains.append(cur); cur = []
            continue
        if cur and t[i] - t[cur[-1]] > cfg.gap_s:
            chains.append(cur); cur = []
        cur.append(i)
    if cur:
        chains.append(cur)

    for chain in chains:
        # Viterbi with restart when every transition is unreachable
        sub = [chain[0]]
        delta = emission_logp(steps[chain[0]][2], steps[chain[0]][3], cfg)
        backs = []
        def flush(sub, delta, backs):
            q = int(np.argmax(delta))
            for k in range(len(sub) - 1, -1, -1):
                arcs, _, ps, _ = steps[sub[k]]
                per_arc[sub[k]] = arcs[q]
                per_prob[sub[k]] = ps[q]
                if k > 0:
                    q = int(backs[k - 1][q])
        for i_prev, i in zip(chain[:-1], chain[1:]):
            a0, t0, _, _ = steps[i_prev]
            a1, t1, p1, d1 = steps[i]
            disp = float(np.linalg.norm(trip.xy[i] - trip.xy[i_prev]))
            tr = transition_logp(g, a0, t0, a1, t1, disp, cfg)
            if (tr <= cfg.log_eps).all():  # disconnected: close this sub-chain and restart
                flush(sub, delta, backs)
                n_breaks += 1
                sub, backs = [i], []
                delta = emission_logp(p1, d1, cfg)
                continue
            score = delta[:, None] + tr
            bp = score.argmax(0)
            delta = score[bp, np.arange(len(a1))] + emission_logp(p1, d1, cfg)
            backs.append(bp)
            sub.append(i)
        flush(sub, delta, backs)
    n_breaks += max(0, len(chains) - 1)
    segments = reconstruct(g, per_arc, t, cfg)
    route = []
    for s in segments:
        for a in s:
            if not route or route[-1] != a:
                route.append(a)
    return dict(per_point_arc=per_arc, per_point_prob=per_prob, segments=segments, route_arcs=route, n_breaks=n_breaks)


def classifier_only(trip: Trip, cand_df: pd.DataFrame, prob: np.ndarray):
    """Per-point argmax baseline (no HMM)."""
    n = len(trip.xy)
    per_arc = np.full(n, -1, dtype=np.int64)
    per_prob = np.full(n, np.nan)
    d = cand_df.assign(_p=prob)
    best = d.loc[d.groupby("point_idx")._p.idxmax().to_numpy()]
    per_arc[best.point_idx.to_numpy()] = best.arc_id.to_numpy()
    per_prob[best.point_idx.to_numpy()] = best._p.to_numpy()
    return per_arc, per_prob


def reconstruct(g: RoadGraph, per_arc: np.ndarray, t: np.ndarray, cfg: HMMConfig, max_fill: float = 2000.0):
    """Collapse decoded arcs and fill gaps between consecutive arcs with shortest paths.
    A new segment starts when the next arc is unreachable within the cutoff."""
    segs, cur = [], []
    last_i = None
    for i, a in enumerate(per_arc):
        if a < 0:
            continue
        a = int(a)
        if not cur:
            cur = [a]; last_i = i
            continue
        if a == cur[-1]:
            last_i = i
            continue
        path = g.path_arcs(cur[-1], a, cutoff=min(max_fill, cfg.base_cutoff + 50.0 * (t[i] - t[last_i])))
        if path is None:
            segs.append(cur); cur = [a]
        else:
            cur.extend(path); cur.append(a)
        last_i = i
    if cur:
        segs.append(cur)
    return segs


def segments_to_lonlat(trip: Trip, segments):
    lines = []
    for s in segments:
        nodes = [int(trip.arc_from[s[0]])] + [int(trip.arc_to[a]) for a in s]
        lines.append(trip.node_ll[nodes].round(6).tolist())
    return lines
