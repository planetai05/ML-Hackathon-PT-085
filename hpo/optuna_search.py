"""Bayesian hyperparameter optimisation with Optuna TPE (issue #15).

python -m hpo.optuna_search --base lgbm --trials 30

Objective: mean GroupKFold (by trip, 5 folds, 80 train trips) point-level Top-1 accuracy.
Study persisted to hpo/optuna_<base>.db (SQLite) and results/optuna_trials.csv.
Best params -> results/hpo_best_params.json; tuned model is then evaluated with the same
harness as the untuned model (models/train.py) and compared in results/hpo_comparison.csv.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import optuna
import pandas as pd

from models.train import SEED, comparison_table, cross_validate, load_split, run_model

SPACES = {
    "lgbm": lambda t: dict(
        n_estimators=t.suggest_int("n_estimators", 150, 800, step=50),
        learning_rate=t.suggest_float("learning_rate", 0.01, 0.2, log=True),
        num_leaves=t.suggest_int("num_leaves", 15, 255, log=True),
        min_child_samples=t.suggest_int("min_child_samples", 5, 200, log=True),
        subsample=t.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=t.suggest_float("colsample_bytree", 0.4, 1.0),
        reg_lambda=t.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    ),
    "hgb": lambda t: dict(
        max_iter=t.suggest_int("max_iter", 100, 600, step=50),
        learning_rate=t.suggest_float("learning_rate", 0.01, 0.3, log=True),
        max_leaf_nodes=t.suggest_int("max_leaf_nodes", 15, 255, log=True),
    ),
    "rf": lambda t: dict(
        n_estimators=t.suggest_int("n_estimators", 100, 300, step=50),
        max_depth=t.suggest_int("max_depth", 8, 30),
        min_samples_leaf=t.suggest_int("min_samples_leaf", 1, 50, log=True),
        max_samples=t.suggest_float("max_samples", 0.1, 0.5),
    ),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="lgbm")
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--folds", type=int, default=5)
    a = ap.parse_args()
    train_df, test_df = load_split("train"), load_split("student_test")
    os.makedirs("hpo", exist_ok=True)
    storage = f"sqlite:///hpo/optuna_{a.base}.db"
    study = optuna.create_study(study_name=f"pathsnap_{a.base}", direction="maximize", storage=storage, load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=8))

    def objective(trial):
        params = SPACES[a.base](trial)
        t0 = time.time()
        cv = cross_validate(a.base, train_df, params, n_splits=a.folds, verbose=False)
        trial.set_user_attr("cv_macro_f1", float(cv.macro_f1.mean()))
        trial.set_user_attr("cv_log_loss", float(cv.log_loss.mean()))
        trial.set_user_attr("seconds", time.time() - t0)
        print(f"trial {trial.number}: top1={cv.top1.mean():.4f} f1={cv.macro_f1.mean():.4f} {params} ({time.time() - t0:.0f}s)", flush=True)
        return float(cv.top1.mean())

    remaining = a.trials - len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    if remaining > 0:
        study.optimize(objective, n_trials=remaining)
    study.trials_dataframe().to_csv("results/optuna_trials.csv", index=False)
    best = study.best_params
    json.dump(best, open("results/hpo_best_params.json", "w"), indent=1)
    print("best", study.best_value, best, flush=True)

    # plots (matplotlib backend so they render without a browser)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances
        os.makedirs("results/figures/hpo", exist_ok=True)
        ax = plot_optimization_history(study); ax.figure.tight_layout(); ax.figure.savefig("results/figures/hpo/optimization_history.png", dpi=100); plt.close("all")
        ax = plot_param_importances(study); ax.figure.tight_layout(); ax.figure.savefig("results/figures/hpo/param_importances.png", dpi=100); plt.close("all")
    except Exception as e:  # plotting must never lose the study
        print("plotting failed:", e)

    # tuned model through the identical harness, then compare with the untuned run
    run_model(a.base, train_df, test_df, best, tag=f"{a.base}_tuned")
    comp = comparison_table().set_index("model")
    keep = ["cv_macro_f1", "cv_top1", "cv_log_loss", "test_top1", "test_macro_f1", "test_log_loss", "inference_ms_per_1k_rows", "train_time_s"]
    out = comp.loc[[a.base, f"{a.base}_tuned"], keep]
    out.index = ["untuned", "tuned"]
    out.to_csv("results/hpo_comparison.csv", float_format="%.5f")
    print(out.to_string())


if __name__ == "__main__":
    main()
