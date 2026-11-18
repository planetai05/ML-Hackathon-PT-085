"""PathSnap inference interface (contract: docs/API_CONTRACT.md, issue #2).

Stub version: `Predictor(mock=True)` returns a schema-valid response so the API/UI can be
built before the trained model exists. The real pipeline replaces `_predict_real` (issue #19).
"""
from __future__ import annotations

import os
import time

import numpy as np


def _trip_id(trip_dir: str) -> str:
    return os.path.basename(os.path.normpath(trip_dir))


class Predictor:
    def __init__(self, artifacts_dir: str = "artifacts", mock: bool = False):
        self.artifacts_dir = artifacts_dir
        self.mock = mock
        self.model_name = "mock" if mock else None

    # ---------------- public API ----------------
    def predict_trip(self, trip_dir: str) -> dict:
        if self.mock:
            return self._predict_mock(trip_dir)
        return self._predict_real(trip_dir)

    def score(self, pred: dict, trip_dir: str) -> dict:
        if self.mock:
            return {"top1_accuracy": None, "classifier_only_top1": None, "route_precision": None,
                    "route_recall": None, "route_f1": None, "route_continuity": None,
                    "n_points_aligned": 0, "reference_route_geojson": None}
        raise NotImplementedError("real scoring lands with issue #18/#19")

    # ---------------- implementations ----------------
    def _predict_mock(self, trip_dir: str) -> dict:
        t0 = time.perf_counter()
        tid = _trip_id(trip_dir)
        track = np.loadtxt(os.path.join(trip_dir, f"{tid}.track"), ndmin=2)
        coords = track[:, :2].tolist()
        n = len(track)
        return {
            "trip_id": tid,
            "n_points": n,
            "gps_points": track[:, :3].tolist(),
            "per_point_arc": [None] * n,
            "per_point_prob": [None] * n,
            "predicted_arcs": [],
            "predicted_route_geojson": {"type": "Feature", "properties": {"layer": "predicted"},
                                        "geometry": {"type": "MultiLineString", "coordinates": [coords]}},
            "metadata": {"model_name": "mock", "decoder": "none", "n_segments": 1,
                         "mean_candidates": 0.0, "empty_candidate_points": n, "artifact_version": "mock"},
            "timings_ms": {"load": 0.0, "candidates": 0.0, "features": 0.0, "model": 0.0, "viterbi": 0.0,
                           "total": round((time.perf_counter() - t0) * 1000, 1)},
        }

    def _predict_real(self, trip_dir: str) -> dict:
        raise NotImplementedError("real predictor lands with issue #19; use Predictor(mock=True) for now")


if __name__ == "__main__":
    import json, sys
    p = Predictor(mock=True)
    out = p.predict_trip(sys.argv[1])
    out["metrics"] = p.score(out, sys.argv[1])
    print(json.dumps({k: (v if k not in ("gps_points",) else v[:3]) for k, v in out.items()}, indent=1)[:2000])
