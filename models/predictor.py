"""PathSnap inference interface (contract: docs/API_CONTRACT.md; issues #2 and #19).

    predictor = Predictor("artifacts")          # load once (API startup)
    pred = predictor.predict_trip(trip_dir)     # reads ONLY .track/.nodes/.arcs
    metrics = predictor.score(pred, trip_dir)   # server-side: reads .route AFTER prediction

Pipeline: load trip -> candidates (features/candidates.py) -> features (features/build_features.py,
same code as training) -> classifier probabilities -> HMM/Viterbi (hmm/) -> gap-filled route.
`Predictor(mock=True)` returns a schema-valid response without any model (for UI/API development).
"""
from __future__ import annotations

import json
import os
import threading
import time
from collections import OrderedDict

import joblib
import numpy as np

from evaluation.metrics import continuity, route_overlap
from features.build_features import FEATURES, trip_features
from features.candidates import ArcIndex, DEFAULTS, generate_candidates
from features.labels import align_route
from features.trip_io import Trip, load_trip
from hmm.model import HMMConfig, RoadGraph
from hmm.viterbi import classifier_only, decode, segments_to_lonlat


def _trip_id(trip_dir: str) -> str:
    return os.path.basename(os.path.normpath(trip_dir))


def _geojson(lines, layer):
    return {"type": "Feature", "properties": {"layer": layer}, "geometry": {"type": "MultiLineString", "coordinates": lines}}


class Predictor:
    def __init__(self, artifacts_dir: str = "artifacts", mock: bool = False, cache_trips: int = 12):
        self.artifacts_dir = artifacts_dir
        self.mock = mock
        self._trips: OrderedDict = OrderedDict()   # trip_dir -> (Trip, ArcIndex, RoadGraph)
        self._cache_n = cache_trips
        self._lock = threading.Lock()
        if mock:
            self.model_name, self.model, self.version = "mock", None, "mock"
            return
        meta = json.load(open(os.path.join(artifacts_dir, "model_meta.json")))
        self.meta = meta
        self.model_name = meta["model_name"]
        self.version = meta.get("artifact_version", "v1")
        self.model = joblib.load(os.path.join(artifacts_dir, meta["model_file"]))
        fc = json.load(open(os.path.join(artifacts_dir, "feature_config.json")))
        if fc["features"] != FEATURES:
            raise RuntimeError("feature_config.json does not match features/build_features.py FEATURES")
        self.cand_params = fc.get("candidate_params", DEFAULTS)
        hc = json.load(open(os.path.join(artifacts_dir, "hmm_config.json")))
        self.hmm_cfg = HMMConfig(**{k: v for k, v in hc.items() if k in HMMConfig.__dataclass_fields__})

    # ---------------- caching of per-trip heavy objects ----------------
    def _get_trip(self, trip_dir: str):
        key = os.path.abspath(trip_dir)
        with self._lock:
            if key in self._trips:
                self._trips.move_to_end(key)
                return self._trips[key], True
        trip = load_trip(trip_dir, with_route=False)
        obj = (trip, ArcIndex(trip, margin=2 * self.cand_params["radius"] + 50, step=self.cand_params["step"]), RoadGraph(trip))
        with self._lock:
            self._trips[key] = obj
            while len(self._trips) > self._cache_n:
                self._trips.popitem(last=False)
        return obj, False

    def warmup(self, trip_dirs):
        """Pre-build spatial index + graph for the evaluation trips (call at API startup)."""
        for d in trip_dirs:
            self._get_trip(d)

    # ---------------- public API ----------------
    def predict_trip(self, trip_dir: str, decoder: str = "hmm_viterbi") -> dict:
        if self.mock:
            return self._predict_mock(trip_dir)
        T = {}
        t0 = time.perf_counter()
        (trip, index, graph), cached = self._get_trip(trip_dir)
        T["load"] = time.perf_counter() - t0
        t = time.perf_counter()
        cands = generate_candidates(trip, self.cand_params["radius"], self.cand_params["k"], index=index)
        T["candidates"] = time.perf_counter() - t
        t = time.perf_counter()
        feats = trip_features(trip, cands)
        T["features"] = time.perf_counter() - t
        t = time.perf_counter()
        prob = self.model.predict_proba(feats[FEATURES].to_numpy(np.float32))[:, 1] if len(feats) else np.zeros(0)
        T["model"] = time.perf_counter() - t
        t = time.perf_counter()
        clf_arc, _ = classifier_only(trip, feats, prob) if len(feats) else (np.full(len(trip.xy), -1), None)
        if decoder == "hmm_viterbi":
            r = decode(trip, feats, prob, self.hmm_cfg, graph)
            per_arc, per_prob, segments = r["per_point_arc"], r["per_point_prob"], r["segments"]
            route = r["route_arcs"]
        else:
            from hmm.viterbi import reconstruct
            per_arc, per_prob = classifier_only(trip, feats, prob)
            segments = reconstruct(graph, per_arc, trip.track[:, 2], self.hmm_cfg)
            route = [a for s in segments for a in s]
        T["viterbi"] = time.perf_counter() - t
        T["total"] = time.perf_counter() - t0
        n = len(trip.xy)
        out = {
            "trip_id": trip.trip_id,
            "n_points": n,
            "gps_points": np.round(trip.track[:, :3], 6).tolist(),
            "per_point_arc": [int(a) if a >= 0 else None for a in per_arc],
            "per_point_prob": [None if not np.isfinite(p) else round(float(p), 4) for p in per_prob],
            "predicted_arcs": [int(a) for a in route],
            "predicted_route_geojson": _geojson(segments_to_lonlat(trip, segments), "predicted"),
            "metadata": {"model_name": self.model_name, "decoder": decoder, "n_segments": len(segments),
                         "mean_candidates": round(len(feats) / max(n, 1), 3),
                         "empty_candidate_points": int(n - feats.point_idx.nunique()) if len(feats) else n,
                         "artifact_version": self.version, "trip_cache_hit": cached},
            "timings_ms": {k: round(v * 1000, 1) for k, v in T.items()},
        }
        out["_classifier_only_arc"] = [int(a) for a in clf_arc]  # internal, removed by score()/API
        return out

    def score(self, pred: dict, trip_dir: str) -> dict:
        """Server-side scoring against the reference route. Never call before predict_trip."""
        if self.mock:
            return {"top1_accuracy": None, "classifier_only_top1": None, "route_precision": None, "route_recall": None,
                    "route_f1": None, "route_continuity": None, "n_points_aligned": 0, "reference_route_geojson": None}
        trip = load_trip(trip_dir, with_route=True)
        true_arc, aligned, _ = align_route(trip)
        hm = np.array([a if a is not None else -1 for a in pred["per_point_arc"]])
        co = np.array(pred.pop("_classifier_only_arc", [-1] * len(hm)))
        if len(hm) != len(trip.xy):
            raise ValueError("prediction does not match trip length")
        m = {
            "top1_accuracy": float((hm[aligned] == true_arc[aligned]).mean()) if aligned.any() else None,
            "classifier_only_top1": float((co[aligned] == true_arc[aligned]).mean()) if aligned.any() else None,
            "route_continuity": continuity(hm[hm >= 0], trip.arc_from, trip.arc_to),
            "n_points_aligned": int(aligned.sum()),
        }
        m.update(route_overlap(pred["predicted_arcs"], trip.route, trip.arc_len))
        m["reference_route_geojson"] = _geojson(segments_to_lonlat(trip, [list(map(int, trip.route))]), "reference")
        return {k: (round(v, 5) if isinstance(v, float) else v) for k, v in m.items()}

    # ---------------- mock ----------------
    def _predict_mock(self, trip_dir: str) -> dict:
        t0 = time.perf_counter()
        tid = _trip_id(trip_dir)
        track = np.loadtxt(os.path.join(trip_dir, f"{tid}.track"), ndmin=2)
        n = len(track)
        return {
            "trip_id": tid, "n_points": n, "gps_points": track[:, :3].tolist(),
            "per_point_arc": [None] * n, "per_point_prob": [None] * n, "predicted_arcs": [],
            "predicted_route_geojson": _geojson([track[:, :2].tolist()], "predicted"),
            "metadata": {"model_name": "mock", "decoder": "none", "n_segments": 1, "mean_candidates": 0.0,
                         "empty_candidate_points": n, "artifact_version": "mock", "trip_cache_hit": False},
            "timings_ms": {"load": 0.0, "candidates": 0.0, "features": 0.0, "model": 0.0, "viterbi": 0.0,
                           "total": round((time.perf_counter() - t0) * 1000, 1)},
        }


if __name__ == "__main__":
    import sys
    p = Predictor(mock="--mock" in sys.argv)
    for d in [a for a in sys.argv[1:] if not a.startswith("--")]:
        out = p.predict_trip(d)
        met = p.score(out, d)
        met.pop("reference_route_geojson", None)
        print(out["trip_id"], out["metadata"]["n_segments"], out["timings_ms"], met)
