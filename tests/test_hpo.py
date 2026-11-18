"""Unit tests for Bayesian HPO search and objectives (Issue #15)."""
import os
import tempfile
import numpy as np
import pandas as pd
import pytest
import optuna

from features.build_features import FEATURES
from hpo.optuna_search import objective, plot_optuna_study


@pytest.fixture
def hpo_toy_df():
    np.random.seed(42)
    n = 300
    X = np.random.randn(n, len(FEATURES)).astype(np.float32)
    y = np.random.choice([0, 1], size=n, p=[0.8, 0.2]).astype(np.int8)
    df = pd.DataFrame(X, columns=FEATURES)
    df["trip_id"] = np.repeat([f"trip_{i}" for i in range(15)], 20)
    df["point_idx"] = np.tile(np.repeat(range(5), 4), 15)
    df["arc_id"] = np.random.randint(0, 50, size=n)
    df["label"] = y
    df["aligned"] = True
    return df


def test_optuna_objective_executes(hpo_toy_df):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(lambda trial: objective(trial, hpo_toy_df, FEATURES), n_trials=2)
    
    assert len(study.trials) == 2
    best = study.best_trial
    assert best.value >= 0.0 and best.value <= 1.0
    assert "learning_rate" in best.params
    assert "max_depth" in best.params


def test_optuna_plotting(hpo_toy_df):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(lambda trial: objective(trial, hpo_toy_df, FEATURES), n_trials=3)
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        plot_optuna_study(study, out_dir=tmp_dir)
        assert os.path.exists(os.path.join(tmp_dir, "optimization_history.png"))
        assert os.path.exists(os.path.join(tmp_dir, "param_importances.png"))
