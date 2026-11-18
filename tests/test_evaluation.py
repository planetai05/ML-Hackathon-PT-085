"""Unit tests for evaluation metrics (Issue #18)."""
import os
import numpy as np
import pytest

from evaluation.metrics import continuity, pair_metrics, route_overlap, score_route
from models.predictor import Predictor


def test_continuity():
    # Arc 0: 0 -> 1, Arc 1: 1 -> 2, Arc 2: 2 -> 3, Arc 3: 5 -> 6 (disconnected)
    arc_from = np.array([0, 1, 2, 5])
    arc_to = np.array([1, 2, 3, 6])

    # Continuous sequence: 0 -> 1 -> 2
    c = continuity(np.array([0, 1, 2]), arc_from, arc_to)
    assert c == 1.0

    # Repeating same arc: 0 -> 0 -> 1 -> 1
    c_same = continuity(np.array([0, 0, 1, 1]), arc_from, arc_to)
    assert c_same == 1.0

    # Disconnected transition: 0 -> 3 (to_node of 0 is 1 != from_node of 3 which is 5)
    c_disc = continuity(np.array([0, 3]), arc_from, arc_to)
    assert c_disc == 0.0


def test_route_overlap():
    arc_len = np.array([10.0, 20.0, 30.0, 40.0])
    # Exact match
    res = route_overlap([0, 1], [0, 1], arc_len)
    assert res["route_precision"] == 1.0
    assert res["route_recall"] == 1.0
    assert res["route_f1"] == 1.0

    # Partial match: pred has 0, 1 (total 30); ref has 1, 2 (total 50); inter is 1 (20)
    res_part = route_overlap([0, 1], [1, 2], arc_len)
    assert np.isclose(res_part["route_precision"], 20.0 / 30.0)
    assert np.isclose(res_part["route_recall"], 20.0 / 50.0)

    # Disjoint match
    res_none = route_overlap([0], [3], arc_len)
    assert res_none["route_f1"] == 0.0


def test_pair_metrics():
    y = np.array([1, 0, 1, 0])
    p = np.array([0.9, 0.1, 0.8, 0.2])
    m = pair_metrics(y, p)
    assert m["accuracy"] == 1.0
    assert m["macro_f1"] == 1.0
    assert m["log_loss"] < 0.3


def test_score_route_student_test():
    trip_dir = "data/Student folder/student_test/00000052"
    if not os.path.exists(trip_dir):
        pytest.skip("student_test dataset not present")

    p = Predictor(mock=False)
    pred = p.predict_trip(trip_dir)
    scores = score_route(pred, trip_dir)

    assert "top1_accuracy" in scores
    assert "classifier_only_top1" in scores
    assert "classifier_only_continuity" in scores
    assert "route_continuity" in scores
    assert "route_f1" in scores
    assert "reference_route_geojson" in scores

    assert scores["top1_accuracy"] > 0.8
    assert scores["route_continuity"] > 0.8
    assert scores["route_f1"] > 0.8
