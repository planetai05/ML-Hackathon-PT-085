"""PathSnap inference interface (contract: docs/API_CONTRACT.md, issues #2, #19).

Loads trained artifacts from artifacts/ and executes the full prediction pipeline:
load -> candidates -> features -> model -> Viterbi -> contract-valid response dict.
Also implements server-side scoring against reference .route files.
"""
from __future__ import annotations

import json
import os
import time

import joblib
import numpy as np
import pandas as pd

from evaluation.metrics import continuity, route_overlap
from features.build_features import FEATURES, trip_features
from features.candidates import generate_candidates
from features.labels import align_route
from features.trip_io import Trip, load_trip


def _trip_id(trip_dir: str) -> str:
    return os.path.basename(os.path.normpath(trip_dir))


class Predictor:
    def __init__(self, artifacts_dir: str = "artifacts", mock: bool = False):
        self.artifacts_dir = artifacts_dir
        self.mock = mock
        self.model_name = "mock" if mock else "xgb"
        self.version = "1.0.0"
        self.features = FEATURES
        self.model = None
        self.scaler = None
        self.hmm_config = {
            "transition_same": 0.55,
            "transition_connected": 0.40,
            "transition_other": 0.001,
        }
        
        if not mock:
            self._load_artifacts()

    def _load_artifacts(self):
        meta_path = os.path.join(self.artifacts_dir, "model_meta.json")
        if not os.path.exists(meta_path):
            meta_path = os.path.join(self.artifacts_dir, "model_metadata.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"artifacts/model_meta.json not found in {self.artifacts_dir}")
            
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.model_name = meta.get("model_name", "xgb")
        self.version = meta.get("artifact_version", "1.0.0")
        
        # Load champion model
        model_path = os.path.join(self.artifacts_dir, "best_model.joblib")
        if not os.path.exists(model_path):
            model_path = os.path.join(self.artifacts_dir, "best_model_tuned.joblib")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model artifact not found in {self.artifacts_dir}")
        self.model = joblib.load(model_path)
        
        # Load optional scaler
        scaler_path = os.path.join(self.artifacts_dir, "scaler.joblib")
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            
        # Load HMM config
        hmm_path = os.path.join(self.artifacts_dir, "hmm_config.json")
        if os.path.exists(hmm_path):
            with open(hmm_path, "r", encoding="utf-8") as f:
                self.hmm_config = json.load(f)

    # ---------------- public API ----------------
    def predict_trip(self, trip_dir: str) -> dict:
        if self.mock:
            return self._predict_mock(trip_dir)
        return self._predict_real(trip_dir)

    def score(self, pred: dict, trip_dir: str) -> dict:
        if self.mock:
            return {
                "top1_accuracy": None,
                "classifier_only_top1": None,
                "route_precision": None,
                "route_recall": None,
                "route_f1": None,
                "route_continuity": None,
                "n_points_aligned": 0,
                "reference_route_geojson": None,
            }
        return self._score_real(pred, trip_dir)

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
            "predicted_route_geojson": {
                "type": "Feature",
                "properties": {"layer": "predicted"},
                "geometry": {"type": "MultiLineString", "coordinates": [coords]},
            },
            "metadata": {
                "model_name": "mock",
                "decoder": "none",
                "n_segments": 1,
                "mean_candidates": 0.0,
                "empty_candidate_points": n,
                "artifact_version": "mock",
            },
            "timings_ms": {
                "load": 0.0,
                "candidates": 0.0,
                "features": 0.0,
                "model": 0.0,
                "viterbi": 0.0,
                "total": round((time.perf_counter() - t0) * 1000, 1),
            },
        }

    def _viterbi_decode(self, trip: Trip, df_cands: pd.DataFrame, n_points: int):
        """Sequential Viterbi decoding over candidate states."""
        from hmm.viterbi import decode_viterbi
        return decode_viterbi(trip, df_cands, n_points, self.hmm_config)

    def _predict_real(self, trip_dir: str) -> dict:
        t_start = time.perf_counter()
        tid = _trip_id(trip_dir)
        
        # 1. Load Trip
        t0 = time.perf_counter()
        trip = load_trip(trip_dir, with_route=False)
        t_load = (time.perf_counter() - t0) * 1000
        
        # 2. Candidate Generation
        t0 = time.perf_counter()
        cands = generate_candidates(trip)
        t_cands = (time.perf_counter() - t0) * 1000
        
        # 3. Feature Engineering
        t0 = time.perf_counter()
        df = trip_features(trip, cands)
        t_feat = (time.perf_counter() - t0) * 1000
        
        # 4. Model Scoring
        t0 = time.perf_counter()
        if len(df) > 0:
            X = df[self.features].to_numpy(np.float32)
            probs = self.model.predict_proba(X)[:, 1]
            df["p"] = probs
        else:
            probs = np.array([], dtype=float)
        t_model = (time.perf_counter() - t0) * 1000
        
        # 5. Viterbi Decoding
        t0 = time.perf_counter()
        n = len(trip.xy)
        v_arcs, v_probs, clf_argmax = self._viterbi_decode(trip, df, n)
        t_vit = (time.perf_counter() - t0) * 1000
        
        # Build ordered, de-duplicated route sequence
        predicted_route = []
        for a in v_arcs:
            if a is not None and (len(predicted_route) == 0 or predicted_route[-1] != a):
                predicted_route.append(a)
                
        # Build GeoJSON Feature for predicted route
        coords = []
        node_ll = trip.node_ll
        af, at = trip.arc_from, trip.arc_to
        for a in predicted_route:
            seg = [
                [float(node_ll[af[a], 0]), float(node_ll[af[a], 1])],
                [float(node_ll[at[a], 0]), float(node_ll[at[a], 1])],
            ]
            coords.append(seg)
            
        geojson_feature = {
            "type": "Feature",
            "properties": {"layer": "predicted", "trip_id": tid, "n_arcs": len(predicted_route)},
            "geometry": {"type": "MultiLineString", "coordinates": coords if coords else [[trip.track[:, :2].tolist()]]},
        }
        
        total_time = (time.perf_counter() - t_start) * 1000
        
        res = {
            "trip_id": tid,
            "n_points": n,
            "gps_points": trip.track[:, :3].tolist(),
            "per_point_arc": v_arcs,
            "per_point_prob": v_probs,
            "predicted_arcs": predicted_route,
            "predicted_route_geojson": geojson_feature,
            "metadata": {
                "model_name": self.model_name,
                "decoder": "hmm_viterbi",
                "n_segments": len(predicted_route),
                "mean_candidates": float(len(df) / max(n, 1)),
                "empty_candidate_points": int(n - df.point_idx.nunique()) if len(df) else n,
                "artifact_version": self.version,
            },
            "timings_ms": {
                "load": round(t_load, 1),
                "candidates": round(t_cands, 1),
                "features": round(t_feat, 1),
                "model": round(t_model, 1),
                "viterbi": round(t_vit, 1),
                "total": round(total_time, 1),
            },
            "_clf_argmax": clf_argmax,
        }
        return res

    def _score_real(self, pred: dict, trip_dir: str) -> dict:
        from evaluation.metrics import score_route
        return score_route(pred, trip_dir)


if __name__ == "__main__":
    import sys
    trip_p = sys.argv[1] if len(sys.argv) > 1 else "data/Student folder/student_test/00000000"
    p = Predictor(mock=False)
    out = p.predict_trip(trip_p)
    out["metrics"] = p.score(out, trip_p)
    print(f"Trip: {out['trip_id']}, points: {out['n_points']}, Top1: {out['metrics']['top1_accuracy']:.4f}, total: {out['timings_ms']['total']}ms")
