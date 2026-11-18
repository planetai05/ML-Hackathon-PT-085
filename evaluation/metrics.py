"""Shared evaluation metrics (issues #11 and #18). Used offline and by the API (server-side scoring)."""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_score, recall_score


def pair_metrics(y, p, thr=0.5):
    """Pair-level (point, candidate) binary metrics. Macro F1 averages the F1 of both classes."""
    yhat = (p >= thr).astype(int)
    return dict(accuracy=accuracy_score(y, yhat), macro_f1=f1_score(y, yhat, average="macro"),
                precision=precision_score(y, yhat, zero_division=0), recall=recall_score(y, yhat, zero_division=0),
                log_loss=log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1]))


def top1_table(df: pd.DataFrame, p: np.ndarray):
    """Per point: argmax-probability candidate. df needs trip_id, point_idx, arc_id, label, aligned."""
    d = df[["trip_id", "point_idx", "arc_id", "label", "aligned"]].copy()
    d["p"] = p
    idx = d.groupby(["trip_id", "point_idx"], sort=False).p.idxmax()
    best = d.loc[idx.to_numpy()]
    return best.sort_values(["trip_id", "point_idx"]).reset_index(drop=True)


def top1_accuracy(best: pd.DataFrame):
    a = best[best.aligned]
    return float(a.label.mean()) if len(a) else float("nan")


def continuity(arcs: np.ndarray, arc_from: np.ndarray, arc_to: np.ndarray, breaks: np.ndarray | None = None):
    """Share of consecutive (distinct) decoded arcs that are connected: same arc or to_node == next from_node.
    `breaks[i]` True means a gap before position i (not counted)."""
    if len(arcs) < 2:
        return float("nan")
    a, b = arcs[:-1], arcs[1:]
    valid = np.ones(len(a), bool) if breaks is None else ~breaks[1:]
    ok = (a == b) | (arc_to[a] == arc_from[b])
    change = (a != b) & valid
    return float(ok[change].mean()) if change.any() else 1.0


def route_overlap(pred_arcs, ref_arcs, arc_len):
    """Length-weighted precision / recall / F1 between the sets of predicted and reference arcs."""
    P, R = set(map(int, pred_arcs)), set(map(int, ref_arcs))
    if not P or not R:
        return dict(route_precision=0.0, route_recall=0.0, route_f1=0.0)
    inter = sum(arc_len[a] for a in P & R)
    prec = inter / max(sum(arc_len[a] for a in P), 1e-9)
    rec = inter / max(sum(arc_len[a] for a in R), 1e-9)
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    return dict(route_precision=float(prec), route_recall=float(rec), route_f1=float(f1))


def score_route(pred: dict, trip_dir: str) -> dict:
    """Score predicted route against ground-truth reference route (.route).

    Computes:
    - top1_accuracy: Viterbi top-1 accuracy on aligned points
    - classifier_only_top1: Argmax top-1 accuracy on aligned points
    - classifier_only_continuity: Continuity of argmax sequence
    - route_continuity: Continuity of Viterbi sequence
    - route_precision: Length-weighted precision
    - route_recall: Length-weighted recall
    - route_f1: Length-weighted route F1 overlap
    - n_points_aligned: Number of aligned GPS points
    - reference_route_geojson: GeoJSON Feature of the ground-truth route
    """
    from features.trip_io import Trip, load_trip
    from features.labels import align_route

    tid = pred.get("trip_id") or os.path.basename(trip_dir)
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

    # 4. Route continuity (Viterbi & Classifier-only)
    valid_arcs_v = np.array([a for a in per_point if a is not None], dtype=np.int64)
    rc_v = continuity(valid_arcs_v, trip.arc_from, trip.arc_to) if len(valid_arcs_v) > 1 else 1.0

    valid_arcs_c = np.array([a for a in clf_argmax if a is not None], dtype=np.int64)
    rc_c = continuity(valid_arcs_c, trip.arc_from, trip.arc_to) if len(valid_arcs_c) > 1 else 1.0

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
        "classifier_only_continuity": round(rc_c, 4),
        "route_precision": round(overlap["route_precision"], 4),
        "route_recall": round(overlap["route_recall"], 4),
        "route_f1": round(overlap["route_f1"], 4),
        "route_continuity": round(rc_v, 4),
        "n_points_aligned": int(aligned.sum()),
        "reference_route_geojson": ref_geojson,
    }

