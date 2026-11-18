"""Unit tests for the ML pipeline (issues #8, #9, #10, #16, #17). Run: python -m pytest tests -q"""
import os

import numpy as np
import pytest

from features.trip_io import Trip, list_trips, load_trip, point_segment

HAS_DATA = len(list_trips("student_test")) > 0


def toy_trip():
    # straight road west->east: nodes 0-1-2-3 (100 m spacing) with arcs both directions + a parallel road
    nodes = np.array([[0, 0], [100, 0], [200, 0], [300, 0], [0, 30], [300, 30]], float)
    arcs = np.array([[0, 1], [1, 2], [2, 3], [1, 0], [2, 1], [3, 2], [4, 5]])
    t = np.arange(0, 30, 1.0)
    xy = np.column_stack([t * 10 + 1, np.full_like(t, 3.0)])  # 10 m/s east, 3 m north of the road
    track = np.column_stack([xy, t])
    return Trip("toy", track, xy, nodes, nodes, arcs[:, 0], arcs[:, 1], np.array([0, 1, 2]))


def test_point_segment():
    d, t, _ = point_segment(np.array([5.0]), np.array([5.0]), 0.0, 0.0, 10.0, 0.0)
    assert d[0] == pytest.approx(5.0) and t[0] == pytest.approx(0.5)


def test_candidates_toy():
    from features.candidates import generate_candidates
    c = generate_candidates(toy_trip(), radius=40, k=12)
    assert set(c.point_idx) == set(range(30))
    assert c.groupby("point_idx").dist.apply(lambda s: s.is_monotonic_increasing).all()
    # nearest candidates for point 0 are the arc 0->1 and its reverse at 3 m
    first = c[c.point_idx == 0].sort_values("dist")
    assert set(first.arc_id.iloc[:2]) == {0, 3} and first.dist.iloc[0] == pytest.approx(3.0)


def test_alignment_monotonic_toy():
    from features.labels import align_route
    ta, al, _ = align_route(toy_trip())
    assert al.all()
    pos = [list(toy_trip().route).index(a) for a in ta]
    assert pos == sorted(pos) and pos[0] == 0 and pos[-1] == 2


def test_features_toy_heading():
    from features.build_features import FEATURES, trip_features
    df = trip_features(toy_trip())
    assert set(FEATURES) <= set(df.columns) and not df[FEATURES].isna().any().any()
    # eastbound vehicle: forward arc 1->2 has heading diff ~0, reverse arc 2->1 ~180
    p = df[df.point_idx == 15].set_index("arc_id")
    assert p.loc[1, "heading_diff"] < 5 and p.loc[4, "heading_diff"] > 175


@pytest.mark.skipif(not HAS_DATA, reason="dataset not present")
def test_real_trip_candidates_and_labels():
    from features.candidates import generate_candidates, miss_rate
    from features.labels import align_route
    tr = load_trip(list_trips("student_test")[0], with_route=True)
    c = generate_candidates(tr)
    assert c.arc_id.between(0, tr.n_arcs - 1).all()
    ta, al, d = align_route(tr)
    assert al.mean() > 0.9 and miss_rate(c, ta, al) < 0.05
