"""Bayesian Hyperparameter Optimisation using Optuna (TPE Sampler) for XGBoost (Issue #15).

Tunes XGBoost hyperparameters on training data using GroupKFold cross-validation.
Evaluates the best parameter configuration on full 80-trip training and 10 student-test trips.
Exports study database, trial history, comparison tables, figures, and serialized models.
"""
from __future__ import annotations

import json
import os
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import optuna
import xgboost as xgb
from sklearn.model_selection import GroupKFold

from evaluation.metrics import pair_metrics, top1_accuracy, top1_table
from features.build_features import FEATURES
from models.train import load_split, run_model, cross_validate, evaluate, SEED, argmax_continuity

STUDY_DB_DIR = "hpo"
STUDY_NAME = "xgb_bayesian_hpo"
N_TRIALS = 30
HPO_SEED = 42


def objective(trial: optuna.Trial, train_df: pd.DataFrame, features: list[str]) -> float:
    # Hyperparameter search space
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 150, 450, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.20, log=True),
        "max_depth": trial.suggest_int("max_depth", 4, 9),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0, step=0.05),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0, step=0.05),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 8),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 10.0, log=True),
        "tree_method": "hist",
        "device": "cpu",
        "eval_metric": "logloss",
    }
    
    # Stratified 3-fold GroupKFold subsample during HPO exploration to balance speed and accuracy
    # Subsample 300,000 candidate rows for the trial evaluations
    aligned_df = train_df[train_df.aligned].reset_index(drop=True)
    if len(aligned_df) > 300_000:
        # Sample by trip to ensure full trajectory integrity
        unique_trips = aligned_df.trip_id.unique()
        rng = np.random.default_rng(HPO_SEED + trial.number)
        sample_trips = rng.choice(unique_trips, size=int(len(unique_trips) * 0.7), replace=False)
        sub_df = aligned_df[aligned_df.trip_id.isin(sample_trips)].reset_index(drop=True)
    else:
        sub_df = aligned_df
        
    gkf = GroupKFold(n_splits=3)
    scores = []
    groups = sub_df.trip_id.to_numpy()
    X = sub_df[features].to_numpy(np.float32)
    y = sub_df.label.to_numpy()
    
    for fold, (tr, va) in enumerate(gkf.split(sub_df, groups=groups)):
        clf = xgb.XGBClassifier(**params, random_state=HPO_SEED, n_jobs=-1)
        clf.fit(X[tr], y[tr])
        p_val = clf.predict_proba(X[va])[:, 1]
        
        # Calculate validation observation Top-1 score
        d_val = sub_df.iloc[va].copy()
        d_val["p"] = p_val
        best = top1_table(d_val, p_val)
        t1 = top1_accuracy(best)
        scores.append(t1)
        
        # Report intermediate value for pruning
        trial.report(t1, fold)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            
    return float(np.mean(scores))


def plot_optuna_study(study: optuna.Study, out_dir: str = "results/figures/hpo"):
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Optimization History
    trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not trials:
        return
    trial_nums = [t.number for t in trials]
    values = [t.value for t in trials]
    best_values = np.maximum.accumulate(values)
    
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(trial_nums, values, "o", color="#1f77b4", alpha=0.6, label="Trial Value")
    ax.plot(trial_nums, best_values, "-", color="#d62728", linewidth=2, label="Best Value (Cumulative)")
    ax.set_xlabel("Trial Number")
    ax.set_ylabel("Validation Top-1 Accuracy")
    ax.set_title("Optuna Bayesian Optimisation History (TPE Sampler)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "optimization_history.png"), dpi=150)
    plt.close(fig)
    
    # 2. Parameter Importances
    try:
        importances = optuna.importance.get_param_importances(study)
        params = list(importances.keys())
        scores = list(importances.values())
        
        fig, ax = plt.subplots(figsize=(9, 5))
        y_pos = np.arange(len(params))
        ax.barh(y_pos, scores, color="#2ca02c")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(params)
        ax.invert_yaxis()
        ax.set_xlabel("Relative Importance Score")
        ax.set_title("Hyperparameter Importance (fANOVA)")
        ax.grid(axis="x", linestyle="--", alpha=0.5)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "param_importances.png"), dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"Skipping param importance plot: {e}")


def run_hpo():
    os.makedirs(STUDY_DB_DIR, exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("results/figures/hpo", exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)
    
    print("Loading feature parquet files...", flush=True)
    train_df = load_split("train")
    test_df = load_split("student_test")
    
    db_path = f"sqlite:///{STUDY_DB_DIR}/optuna_study.db"
    sampler = optuna.samplers.TPESampler(seed=HPO_SEED)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=1)
    
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=db_path,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True
    )
    
    print(f"Starting Bayesian HPO for {N_TRIALS} trials...", flush=True)
    t0 = time.time()
    study.optimize(lambda trial: objective(trial, train_df, FEATURES), n_trials=N_TRIALS, timeout=3600)
    duration = time.time() - t0
    print(f"Completed {len(study.trials)} trials in {duration:.1f}s.", flush=True)
    
    best_trial = study.best_trial
    print(f"Best Trial: #{best_trial.number} with Top-1 = {best_trial.value:.4f}")
    print("Best Hyperparameters:")
    for k, v in best_trial.params.items():
        print(f"  {k}: {v}")
        
    # Save trial history dataframe
    df_trials = study.trials_dataframe()
    df_trials.to_csv("results/optuna_trials.csv", index=False)
    
    # Save best parameters
    best_params = dict(best_trial.params)
    best_params.update({"tree_method": "hist", "device": "cpu", "eval_metric": "logloss"})
    
    with open("results/optuna_best.json", "w") as f:
        json.dump({"trial": best_trial.number, "value": best_trial.value, "params": best_params}, f, indent=2)
    with open("artifacts/best_hyperparameters.json", "w") as f:
        json.dump(best_params, f, indent=2)
    with open("artifacts/model_config.json", "w") as f:
        json.dump({"model": "xgb", "family": "Gradient boosting (XGBoost)", "tuned": True, "params": best_params}, f, indent=2)
        
    plot_optuna_study(study)
    
    # Full Evaluation of Untuned vs Tuned on Full 80 Train Trips & 10 Test Trips
    print("\nEvaluating Untuned vs Tuned XGBoost on full partitions...", flush=True)
    
    # Untuned default model
    untuned_run = run_model("xgb", train_df, test_df, params=None, tag="xgb_untuned")
    # Tuned model with optimal hyperparameters
    tuned_run = run_model("xgb", train_df, test_df, params=best_params, tag="xgb_tuned")
    
    # Copy/save final artifacts
    joblib.dump(joblib.load("models/saved/xgb_untuned.joblib"), "artifacts/best_model_untuned.joblib", compress=3)
    joblib.dump(joblib.load("models/saved/xgb_tuned.joblib"), "artifacts/best_model_tuned.joblib", compress=3)
    joblib.dump(joblib.load("models/saved/xgb_tuned.joblib"), "artifacts/best_model.joblib", compress=3)
    
    # Create Tuned vs Untuned comparison dataframe
    u_cv, u_test = untuned_run["cv_mean"], untuned_run["test"]
    t_cv, t_test = tuned_run["cv_mean"], tuned_run["test"]
    
    comp_data = [
        {
            "configuration": "Untuned (Defaults)",
            "cv_macro_f1": u_cv["macro_f1"],
            "cv_top1": u_cv["top1"],
            "cv_log_loss": u_cv["log_loss"],
            "test_top1": u_test["top1"],
            "test_macro_f1": u_test["macro_f1"],
            "test_log_loss": u_test["log_loss"],
            "test_route_continuity": u_test["route_continuity"],
            "train_time_s": untuned_run["full_train_s"],
            "inference_ms_per_1k_rows": u_test["inference_ms_per_1k_rows"]
        },
        {
            "configuration": "Tuned (Optuna Bayesian)",
            "cv_macro_f1": t_cv["macro_f1"],
            "cv_top1": t_cv["top1"],
            "cv_log_loss": t_cv["log_loss"],
            "test_top1": t_test["top1"],
            "test_macro_f1": t_test["macro_f1"],
            "test_log_loss": t_test["log_loss"],
            "test_route_continuity": t_test["route_continuity"],
            "train_time_s": tuned_run["full_train_s"],
            "inference_ms_per_1k_rows": t_test["inference_ms_per_1k_rows"]
        }
    ]
    df_hpo = pd.DataFrame(comp_data)
    df_hpo.to_csv("results/tuned_vs_untuned.csv", index=False, float_format="%.5f")
    df_hpo.to_csv("results/hpo_comparison.csv", index=False, float_format="%.5f")
    
    print("\n=== TUNED VS UNTUNED COMPARISON ===")
    print(df_hpo.to_string())
    return df_hpo


if __name__ == "__main__":
    run_hpo()
