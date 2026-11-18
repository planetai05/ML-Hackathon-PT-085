"""Trip loading, UTM projection and road-graph helpers used by the ML pipeline.

Minimal loader so the ML pipeline is not blocked; Person A's `data_processing/loader.py` (#3)
can replace/extend it. Files are tab-separated; `.track` = lon, lat, t; `.nodes` = lon, lat
(row index = node id); `.arcs` = from_node, to_node (row index = arc id); `.route` = arc ids.
All distances are metres in the trip's UTM zone (EPSG:326xx/327xx, chosen from the track
centroid; 32N/EPSG:32632 for Darmstadt), never degrees. Train trip 00000057 lies in Ecuador
(lon -78), which is why the zone is chosen per trip.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from pyproj import Transformer

DATA_DIR = os.environ.get("PATHSNAP_DATA_DIR", os.path.join("data", "Student folder"))
_TF: dict = {}


def utm_epsg(lon: float, lat: float) -> int:
    zone = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


def project(lon, lat, epsg: int = 32632):
    if epsg not in _TF:
        _TF[epsg] = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    x, y = _TF[epsg].transform(np.asarray(lon, dtype=float), np.asarray(lat, dtype=float))
    return np.column_stack([x, y])


def _read(path, ncol):
    return pd.read_csv(path, sep=r"\s+", header=None, engine="c", dtype=float).to_numpy()[:, :ncol]


@dataclass
class Trip:
    trip_id: str
    track: np.ndarray          # (N, 3) lon, lat, t
    xy: np.ndarray             # (N, 2) projected GPS
    node_ll: np.ndarray        # (Nn, 2) lon, lat
    node_xy: np.ndarray        # (Nn, 2)
    arc_from: np.ndarray       # (Na,) int
    arc_to: np.ndarray         # (Na,) int
    route: np.ndarray | None = None
    epsg: int = 32632
    _cache: dict = field(default_factory=dict, repr=False)

    @property
    def n_arcs(self):
        return len(self.arc_from)

    @property
    def arc_a(self):
        return self.node_xy[self.arc_from]

    @property
    def arc_b(self):
        return self.node_xy[self.arc_to]

    @property
    def arc_len(self):
        if "len" not in self._cache:
            self._cache["len"] = np.linalg.norm(self.arc_b - self.arc_a, axis=1)
        return self._cache["len"]

    @property
    def arc_bearing(self):
        """Bearing in degrees clockwise from north (0..360)."""
        if "brg" not in self._cache:
            d = self.arc_b - self.arc_a
            self._cache["brg"] = np.degrees(np.arctan2(d[:, 0], d[:, 1])) % 360
        return self._cache["brg"]

    def degrees(self):
        if "deg" not in self._cache:
            n = len(self.node_xy)
            self._cache["deg"] = (np.bincount(self.arc_from, minlength=n), np.bincount(self.arc_to, minlength=n))
        return self._cache["deg"]  # out_degree, in_degree


def load_trip(trip_dir: str, with_route: bool = False) -> Trip:
    tid = os.path.basename(os.path.normpath(trip_dir))
    p = lambda ext: os.path.join(trip_dir, f"{tid}.{ext}")
    track = _read(p("track"), 3)
    # sort by time and drop exact duplicate timestamps (keep first)
    track = track[np.argsort(track[:, 2], kind="stable")]
    keep = np.r_[True, np.diff(track[:, 2]) > 0]
    track = track[keep]
    nodes = _read(p("nodes"), 2)
    arcs = _read(p("arcs"), 2).astype(np.int64)
    route = None
    if with_route and os.path.exists(p("route")):
        route = _read(p("route"), 1)[:, 0].astype(np.int64)
    epsg = utm_epsg(float(np.median(track[:, 0])), float(np.median(track[:, 1])))
    return Trip(tid, track, project(track[:, 0], track[:, 1], epsg), nodes, project(nodes[:, 0], nodes[:, 1], epsg),
                arcs[:, 0], arcs[:, 1], route, epsg)


def list_trips(split: str, data_dir: str = DATA_DIR):
    return sorted(d for d in glob.glob(os.path.join(data_dir, split, "*")) if os.path.isdir(d))


def point_segment(px, py, ax, ay, bx, by):
    """Vectorised point-to-segment distance. Returns (dist, t_clipped, t_raw)."""
    dx, dy = bx - ax, by - ay
    l2 = dx * dx + dy * dy
    t_raw = np.where(l2 > 0, ((px - ax) * dx + (py - ay) * dy) / np.where(l2 > 0, l2, 1), 0.0)
    t = np.clip(t_raw, 0, 1)
    qx, qy = ax + t * dx, ay + t * dy
    return np.hypot(px - qx, py - qy), t, t_raw
