"""Unit tests for HMM Viterbi decoding (Issues #16 and #17)."""
import os
import numpy as np
import pandas as pd
import pytest

from features.trip_io import Trip
from hmm.viterbi import decode_viterbi


def test_viterbi_empty():
    trip = Trip(
        trip_id="toy",
        track=np.zeros((3, 3)),
        xy=np.zeros((3, 2)),
        node_ll=np.zeros((2, 2)),
        node_xy=np.zeros((2, 2)),
        arc_from=np.array([0]),
        arc_to=np.array([1]),
    )
    v_arc, v_prob, clf = decode_viterbi(trip, pd.DataFrame(), n_points=3)
    assert v_arc == [None, None, None]
    assert v_prob == [None, None, None]
    assert clf == [None, None, None]


def test_viterbi_connected_preference():
    """Verify Viterbi prefers a topologically connected path over disconnected high emission."""
    # Network: Arc 0: 0->1, Arc 1: 1->2 (connected to 0), Arc 2: 3->4 (disconnected from 0)
    trip = Trip(
        trip_id="toy",
        track=np.zeros((2, 3)),
        xy=np.zeros((2, 2)),
        node_ll=np.zeros((5, 2)),
        node_xy=np.zeros((5, 2)),
        arc_from=np.array([0, 1, 3]),
        arc_to=np.array([1, 2, 4]),
    )

    # Point 0 has Candidate Arc 0 (prob 0.9)
    # Point 1 has Candidate Arc 1 (connected, prob 0.6) and Arc 2 (disconnected, prob 0.8)
    df_cands = pd.DataFrame([
        {"point_idx": 0, "arc_id": 0, "p": 0.9},
        {"point_idx": 1, "arc_id": 1, "p": 0.6},
        {"point_idx": 1, "arc_id": 2, "p": 0.8},
    ])

    v_arc, v_prob, clf = decode_viterbi(trip, df_cands, n_points=2)

    # Classifier argmax picks Arc 2 for Point 1 because 0.8 > 0.6
    assert clf[1] == 2
    # Viterbi DP picks Arc 1 for Point 1 because connected transition bonus outweighs emission delta
    assert v_arc[0] == 0
    assert v_arc[1] == 1


def test_viterbi_student_test():
    trip_dir = "data/Student folder/student_test/00000052"
    if not os.path.exists(trip_dir):
        pytest.skip("student_test dataset not present")

    from models.predictor import Predictor
    p = Predictor(mock=False)
    pred = p.predict_trip(trip_dir)
    v_arc = pred["per_point_arc"]
    assert len(v_arc) == pred["n_points"]
    assert any(a is not None for a in v_arc)
