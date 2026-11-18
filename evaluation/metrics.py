"""Shared evaluation metrics (issues #11 and #18). Used offline and by the API (server-side scoring)."""
from __future__ import annotations

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
