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
        if len(df_cands) == 0:
            return [None] * n_points, [None] * n_points, [None] * n_points

        point_groups = df_cands.groupby("point_idx")
        cands_by_point = {int(pt): g for pt, g in point_groups}
        
        # Classifier-only argmax per point
        clf_argmax = [None] * n_points
        for pt, g in cands_by_point.items():
            best_idx = g["p"].idxmax()
            clf_argmax[pt] = int(df_cands.loc[best_idx, "arc_id"])
            
        p_same = self.hmm_config.get("transition_same_arc", 0.50)
        p_conn = self.hmm_config.get("transition_connected_arc", 0.45)
        p_other = self.hmm_config.get("transition_disconnected_penalty", 0.001)
        
        log_same = np.log(max(p_same, 1e-6))
        log_conn = np.log(max(p_conn, 1e-6))
        log_other = np.log(max(p_other, 1e-6))
        
        af, at = trip.arc_from, trip.arc_to
        
        # DP states: list of (states, log_probs)
        dp = []
        backpointers = []
        point_has_cand = []
        
        for t in range(n_points):
            if t not in cands_by_point:
                point_has_cand.append(False)
                continue
                
            point_has_cand.append(True)
            g = cands_by_point[t]
            arcs_t = g.arc_id.to_numpy()
            probs_t = g.p.to_numpy()
            emit_t = np.log(np.clip(probs_t, 1e-6, 1.0))
            
            if len(dp) == 0:
                # Initial step
                dp.append(emit_t)
                backpointers.append(None)
            else:
                prev_probs = dp[-1]
                prev_arcs = cands_by_point[valid_points[-1]].arc_id.to_numpy()
                
                # Compute transition matrix: (prev_len, curr_len)
                # same arc or to_node == next from_node
                same_mask = prev_arcs[:, None] == arcs_t[None, :]
                conn_mask = at[prev_arcs, None] == af[None, arcs_t]
                
                trans = np.full((len(prev_arcs), len(arcs_t)), log_other, dtype=float)
                trans[conn_mask] = log_conn
                trans[same_mask] = log_same
                
                # Viterbi recurrence
                scores = prev_probs[:, None] + trans  # (prev_len, curr_len)
                best_prev = np.argmax(scores, axis=0)
                curr_prob = np.max(scores, axis=0) + emit_t
                
                dp.append(curr_prob)
                backpointers.append(best_prev)
                
            valid_points = [i for i, h in enumerate(point_has_cand) if h]
            
        # Traceback
        viterbi_arc = [None] * n_points
        viterbi_prob = [None] * n_points
        
        if len(dp) > 0:
            curr_best = np.argmax(dp[-1])
            for step in range(len(dp) - 1, -1, -1):
                pt = valid_points[step]
                g = cands_by_point[pt]
                chosen_arc = int(g.arc_id.iloc[curr_best])
                chosen_prob = float(g.p.iloc[curr_best])
                viterbi_arc[pt] = chosen_arc
                viterbi_prob[pt] = chosen_prob
                if step > 0:
                    curr_best = backpointers[step][curr_best]
                    
        return viterbi_arc, viterbi_prob, clf_argmax

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
        tid = pred["trip_id"]
        route_file = os.path.join(trip_dir, f"{tid}.route")
        if not os.path.exists(route_file):
            raise FileNotFoundError(f"no .route file in {trip_dir}")
            
        trip = load_trip(trip_dir, with_route=True)
        true_arc, aligned, _ = align_route(trip)
        
        per_point = pred.get("per_point_arc", [])
        clf_argmax = pred.get("_clf_argmax", per_point)
        
        # 1. Top-1 accuracy (Viterbi)
        al_mask = aligned & (np.array([a is not None for a in per_point]))
        if al_mask.any():
            matched_v = np.array([per_point[i] for i in range(len(per_point)) if al_mask[i]])
            true_v = true_arc[al_mask]
            top1_acc = float(np.mean(matched_v == true_v))
        else:
            top1_acc = 0.0
            
        # 2. Classifier-only Top-1
        al_mask_clf = aligned & (np.array([a is not None for a in clf_argmax]))
        if al_mask_clf.any():
            matched_c = np.array([clf_argmax[i] for i in range(len(clf_argmax)) if al_mask_clf[i]])
            true_c = true_arc[al_mask_clf]
            clf_top1 = float(np.mean(matched_c == true_c))
        else:
            clf_top1 = 0.0
            
        # 3. Route overlap (Precision, Recall, F1)
        pred_arcs = pred.get("predicted_arcs", [])
        overlap = route_overlap(pred_arcs, trip.route, trip.arc_len)
        
        # 4. Route continuity
        breaks = np.r_[True, np.diff(np.arange(len(per_point))) > 1]
        valid_arcs = np.array([a for a in per_point if a is not None], dtype=np.int64)
        if len(valid_arcs) > 1:
            rc = continuity(valid_arcs, trip.arc_from, trip.arc_to)
        else:
            rc = 1.0
            
        # 5. Reference route GeoJSON
        node_ll = trip.node_ll
        af, at = trip.arc_from, trip.arc_to
        ref_coords = []
        for a in trip.route:
            seg = [
                [float(node_ll[af[a], 0]), float(node_ll[af[a], 1])],
                [float(node_ll[at[a], 0]), float(node_ll[at[a], 1])],
            ]
            ref_coords.append(seg)
            
        ref_geojson = {
            "type": "Feature",
            "properties": {"layer": "reference", "trip_id": tid, "n_arcs": len(trip.route)},
            "geometry": {"type": "MultiLineString", "coordinates": ref_coords},
        }
        
        return {
            "top1_accuracy": round(top1_acc, 4),
            "classifier_only_top1": round(clf_top1, 4),
            "route_precision": round(overlap["route_precision"], 4),
            "route_recall": round(overlap["route_recall"], 4),
            "route_f1": round(overlap["route_f1"], 4),
            "route_continuity": round(rc, 4),
            "n_points_aligned": int(aligned.sum()),
            "reference_route_geojson": ref_geojson,
        }


if __name__ == "__main__":
    import sys
    trip_p = sys.argv[1] if len(sys.argv) > 1 else "data/Student folder/student_test/00000000"
    p = Predictor(mock=False)
    out = p.predict_trip(trip_p)
    out["metrics"] = p.score(out, trip_p)
    print(f"Trip: {out['trip_id']}, points: {out['n_points']}, Top1: {out['metrics']['top1_accuracy']:.4f}, total: {out['timings_ms']['total']}ms")
