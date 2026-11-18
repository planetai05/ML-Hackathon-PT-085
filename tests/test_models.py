"""Unit tests for ML models, training harness, metrics, and serialization (Issue #13)."""
import os
import tempfile
import numpy as np
import pandas as pd
import pytest
import joblib

from models.train import make_model, FAMILY, fit, evaluate
from evaluation.metrics import pair_metrics, top1_table, top1_accuracy, continuity
from features.build_features import FEATURES


@pytest.fixture
def toy_data():
    np.random.seed(42)
    n = 200
    X = np.random.randn(n, len(FEATURES)).astype(np.float32)
    y = np.random.choice([0, 1], size=n, p=[0.8, 0.2]).astype(np.int8)
    # mock dataframe for evaluation
    df = pd.DataFrame(X, columns=FEATURES)
    df["trip_id"] = np.repeat([f"trip_{i}" for i in range(10)], 20)
    df["point_idx"] = np.tile(np.repeat(range(5), 4), 10)
    df["arc_id"] = np.random.randint(0, 50, size=n)
    df["label"] = y
    df["aligned"] = True
    return df, X, y


@pytest.mark.parametrize("m_name", ["logreg", "knn", "rf", "hgb", "lgbm", "xgb", "mlp"])
def test_all_models_instantiate_and_fit(toy_data, m_name):
    df, X, y = toy_data
    model = make_model(m_name)
    assert m_name in FAMILY
    
    # Train
    model.fit(X, y)
    probs = model.predict_proba(X)
    
    # Verify shape and probability properties
    assert probs.shape == (len(X), 2)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()


def test_top1_ranking_logic(toy_data):
    df, X, y = toy_data
    # create deterministic probabilities: candidate 2 in point 0 has highest prob
    probs = np.zeros(len(df))
    probs[2] = 0.99
    
    table = top1_table(df, probs)
    assert len(table) == df.groupby(["trip_id", "point_idx"]).ngroups
    best_pt0 = table[(table.trip_id == df.iloc[0].trip_id) & (table.point_idx == df.iloc[0].point_idx)]
    assert best_pt0.iloc[0].arc_id == df.iloc[2].arc_id


def test_groupkfold_disjoint_guarantee():
    from sklearn.model_selection import GroupKFold
    trips = np.array([f"trip_{i}" for i in range(20) for _ in range(50)])
    gkf = GroupKFold(n_splits=5)
    
    for tr, va in gkf.split(trips, groups=trips):
        tr_trips = set(trips[tr])
        va_trips = set(trips[va])
        assert len(tr_trips & va_trips) == 0, "Data leakage detected: overlapping trips between train and val!"


def test_model_serialization(toy_data):
    df, X, y = toy_data
    model = make_model("logreg")
    model.fit(X, y)
    orig_p = model.predict_proba(X)
    
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
        tmp_path = tmp.name
        
    try:
        joblib.dump(model, tmp_path)
        reloaded = joblib.load(tmp_path)
        new_p = reloaded.predict_proba(X)
        np.testing.assert_allclose(orig_p, new_p, atol=1e-6)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_metrics_functions():
    y = np.array([0, 1, 0, 1, 0])
    p = np.array([0.1, 0.9, 0.2, 0.8, 0.4])
    m = pair_metrics(y, p)
    assert m["accuracy"] == 1.0
    assert m["macro_f1"] == 1.0
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["log_loss"] < 0.5
