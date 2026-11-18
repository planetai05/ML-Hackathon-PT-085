"""HMM for map matching (issue #16).

Hidden states   : the candidate arcs of each GPS observation (from features/candidates.py).
Emission        : log P(obs | arc) = w_e * log(classifier probability), optionally plus a Gaussian
                  distance term  -0.5 * (dist / sigma)^2 * w_d.
Transition      : Newson & Krumm style. For candidates (a, t_a) at step i and (b, t_b) at step i+1
                  the network distance d is
                    same arc        : |t_b - t_a| * len(a)
                    different arcs  : (1 - t_a) len(a) + shortest_path(to(a), from(b)) + t_b len(b)
                  found with a bounded Dijkstra (cutoff) over the directed road graph (cached per
                  source node). log P = -|d - gps_displacement| / beta. Unreachable within the
                  cutoff -> log_eps (a small epsilon probability, not -inf).
Breaks          : a time gap > gap_s or an observation with no candidates starts a new segment.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import asdict, dataclass

import numpy as np

from features.trip_io import Trip


@dataclass
class HMMConfig:
    beta: float = 8.0               # transition scale (m)
    emission_weight: float = 1.0    # weight on log classifier probability
    dist_weight: float = 0.0        # weight on Gaussian distance term
    sigma: float = 10.0             # GPS noise (m) for the distance term
    log_eps: float = -30.0          # log prob of an unreachable transition
    gap_s: float = 30.0             # time gap that breaks the chain
    base_cutoff: float = 500.0      # Dijkstra cutoff (m); grows with GPS displacement
    p_floor: float = 1e-6

    def to_dict(self):
        return asdict(self)


class RoadGraph:
    """Directed graph in CSR form with cached bounded Dijkstra searches."""

    def __init__(self, trip: Trip):
        self.arc_from, self.arc_to = trip.arc_from, trip.arc_to
        self.arc_len = trip.arc_len
        n = len(trip.node_xy)
        order = np.argsort(self.arc_from, kind="stable")
        self.out_arc = order
        self.indptr = np.r_[0, np.cumsum(np.bincount(self.arc_from, minlength=n))]
        self._cache: dict = {}
        # python lists are much faster than numpy scalars inside the Dijkstra loop
        self._out = self.out_arc.tolist()
        self._ptr = self.indptr.tolist()
        self._to = self.arc_to.tolist()
        self._len = self.arc_len.tolist()

    def dijkstra(self, src: int, cutoff: float):
        key = src
        hit = self._cache.get(key)
        if hit is not None and hit[0] >= cutoff:
            return hit[1], hit[2]
        dist = {src: 0.0}
        pred = {src: -1}
        heap = [(0.0, src)]
        done = set()
        out, ptr, to, ln = self._out, self._ptr, self._to, self._len
        while heap:
            d, u = heapq.heappop(heap)
            if u in done:
                continue
            done.add(u)
            for e in range(ptr[u], ptr[u + 1]):
                a = out[e]
                v = to[a]
                nd = d + ln[a]
                if nd <= cutoff and nd < dist.get(v, math.inf):
                    dist[v] = nd
                    pred[v] = a
                    heapq.heappush(heap, (nd, v))
        self._cache[key] = (cutoff, dist, pred)
        return dist, pred

    def path_arcs(self, a: int, b: int, cutoff: float):
        """Arcs strictly between arc a and arc b (a's to_node -> b's from_node), or None if unreachable."""
        s, t = int(self.arc_to[a]), int(self.arc_from[b])
        if s == t:
            return []
        dist, pred = self.dijkstra(s, cutoff)
        if t not in dist:
            return None
        path = []
        v = t
        while v != s:
            e = pred[v]
            path.append(e)
            v = int(self.arc_from[e])
        return path[::-1]


def emission_logp(prob: np.ndarray, dist: np.ndarray, cfg: HMMConfig):
    lp = cfg.emission_weight * np.log(np.clip(prob, cfg.p_floor, 1.0))
    if cfg.dist_weight:
        lp = lp - cfg.dist_weight * 0.5 * (dist / cfg.sigma) ** 2
    return lp


def transition_logp(g: RoadGraph, arcs_i, t_i, arcs_j, t_j, gps_disp: float, cfg: HMMConfig):
    """Matrix (len(arcs_i), len(arcs_j)) of log transition probabilities."""
    cutoff = max(cfg.base_cutoff, 3.0 * gps_disp + 100.0)
    L = g.arc_len
    out = np.full((len(arcs_i), len(arcs_j)), cfg.log_eps)
    from_j = g.arc_from[arcs_j]
    for r, (a, ta) in enumerate(zip(arcs_i, t_i)):
        dist, _ = g.dijkstra(int(g.arc_to[a]), cutoff)
        rem = (1.0 - ta) * L[a]
        for c, (b, tb) in enumerate(zip(arcs_j, t_j)):
            if a == b:
                d = abs(tb - ta) * L[a]
            else:
                sp = dist.get(int(from_j[c]))
                if sp is None:
                    continue
                d = rem + sp + tb * L[b]
            out[r, c] = -abs(d - gps_disp) / cfg.beta
    return out
