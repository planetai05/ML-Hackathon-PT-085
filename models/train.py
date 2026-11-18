"""GroupKFold training + evaluation harness (issue #11) and model zoo (issue #13).

python -m models.train --models logreg knn rf hgb lgbm mlp

For every model:
  * 5-fold GroupKFold over the 80 TRAIN trips (groups = trip_id; asserted disjoint)
  * per fold: pair-level accuracy / macro F1 / precision / recall / log loss, point-level Top-1,
    route continuity of the per-point argmax sequence, train & inference time, and the same
    metrics on the training fold (over/under-fitting evidence)
  * refit on all 80 train trips -> evaluate on the 10 STUDENT_TEST trips
  * write results/runs/<model>.json, results/model_comparison.csv, confusion matrix, saved model
Seeds are fixed (SEED=42). No evaluator data is used anywhere.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evaluation.metrics import continuity, pair_metrics, top1_accuracy, top1_table
from features.build_features import FEATURES, GAP_S
from features.trip_io import DATA_DIR

SEED = 42
N_SPLITS = 5


def make_model(name: str, params: dict | None = None):
    p = params or {}
    if name == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(C=p.get("C", 1.0), max_iter=500, random_state=SEED))
    if name == "knn":
        return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=p.get("n_neighbors", 25), weights="distance", n_jobs=-1))
    if name == "rf":
        return RandomForestClassifier(n_estimators=p.get("n_estimators", 150), max_depth=p.get("max_depth", 18),
                                      min_samples_leaf=p.get("min_samples_leaf", 5), max_samples=p.get("max_samples", 0.3),
                                      n_jobs=-1, random_state=SEED)
    if name == "hgb":
        return HistGradientBoostingClassifier(max_iter=p.get("max_iter", 300), learning_rate=p.get("learning_rate", 0.1),
                                              max_leaf_nodes=p.get("max_leaf_nodes", 31), random_state=SEED)
    if name == "lgbm":
        import lightgbm as lgb
        base = dict(n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=20, subsample=0.8,
                    subsample_freq=1, colsample_bytree=0.8, reg_lambda=0.0)
        base.update(p)
        return lgb.LGBMClassifier(**base, random_state=SEED, n_jobs=-1, verbose=-1)
    if name == "xgb":
        import xgboost as xgb
        base = dict(n_estimators=300, learning_rate=0.08, max_depth=6, subsample=0.8,
                    colsample_bytree=0.8, tree_method="hist", device="cpu", eval_metric="logloss")
        base.update(p)
        return xgb.XGBClassifier(**base, random_state=SEED, n_jobs=-1)
    if name == "mlp":
        return make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=p.get("hidden", (64, 32)), alpha=p.get("alpha", 1e-4),
                                                             batch_size=1024, learning_rate_init=1e-3, max_iter=40,
                                                             early_stopping=True, n_iter_no_change=5, random_state=SEED))
    raise ValueError(name)


# Models whose fitting cost is super-linear get a fixed-size, seeded subsample of the training fold
TRAIN_SUBSAMPLE = {"knn": 200_000, "mlp": 600_000}
FAMILY = {
    "logreg": "Linear (logistic regression)",
    "knn": "Instance-based (k-NN)",
    "rf": "Bagged trees (random forest)",
    "hgb": "Gradient boosting (sklearn HistGB)",
    "lgbm": "Gradient boosting (LightGBM)",
    "xgb": "Gradient boosting (XGBoost)",
    "mlp": "Neural network (MLP)",
}


def load_split(split):
    df = pd.read_parquet(f"cache/features_{split}.parquet")
    return df


def train_rows(df):
    """Rows used for fitting: points with a reliable label only."""
    return df[df.aligned].reset_index(drop=True)


_ARCS: dict = {}


def trip_arcs(split, tid):
    key = (split, tid)
    if key not in _ARCS:
        a = pd.read_csv(f"{DATA_DIR}/{split}/{tid}/{tid}.arcs", sep=r"\s+", header=None, dtype=np.int64).to_numpy()
        _ARCS[key] = (a[:, 0], a[:, 1])
    return _ARCS[key]


def argmax_continuity(best: pd.DataFrame, df: pd.DataFrame, split: str):
    """Mean over trips of route continuity of the classifier-only argmax sequence."""
    vals = []
    for tid, g in best.groupby("trip_id"):
        af, at = trip_arcs(split, tid)
        pi = g.point_idx.to_numpy()
        breaks = np.r_[True, np.diff(pi) > 1]
        vals.append(continuity(g.arc_id.to_numpy(), af, at, breaks))
    return float(np.nanmean(vals))


def fit(model_name, X, y, params=None):
    m = make_model(model_name, params)
    n = TRAIN_SUBSAMPLE.get(model_name)
    if n and len(X) > n:
        idx = np.random.default_rng(SEED).choice(len(X), n, replace=False)
        X, y = X[idx], y[idx]
    t0 = time.perf_counter()
    m.fit(X, y)
    return m, time.perf_counter() - t0


def evaluate(m, df, split):
    X = df[FEATURES].to_numpy(np.float32)
    t0 = time.perf_counter()
    p = m.predict_proba(X)[:, 1]
    inf = time.perf_counter() - t0
    al = df.aligned.to_numpy()
    out = pair_metrics(df.label.to_numpy()[al], p[al])
    best = top1_table(df, p)
    out["top1"] = top1_accuracy(best)
    out["route_continuity"] = argmax_continuity(best, df, split)
    out["inference_s"] = inf
    out["inference_ms_per_1k_rows"] = 1000 * inf / max(len(df), 1)
    return out, p


def cross_validate(model_name, train_df, params=None, n_splits=N_SPLITS, verbose=True):
    gkf = GroupKFold(n_splits=n_splits)
    folds = []
    groups = train_df.trip_id.to_numpy()
    for k, (tr, va) in enumerate(gkf.split(train_df, groups=groups)):
        assert not set(groups[tr]) & set(groups[va]), "trip leakage between folds"
        dtr, dva = train_df.iloc[tr], train_df.iloc[va]
        fit_df = dtr[dtr.aligned]
        m, tt = fit(model_name, fit_df[FEATURES].to_numpy(np.float32), fit_df.label.to_numpy(), params)
        res, _ = evaluate(m, dva, "train")
        # training-fold metrics on a seeded sample of training trips (over/underfitting evidence)
        trips = np.random.default_rng(SEED + k).choice(dtr.trip_id.unique(), size=min(12, dtr.trip_id.nunique()), replace=False)
        rtr, _ = evaluate(m, dtr[dtr.trip_id.isin(trips)], "train")
        res.update({f"train_{kk}": v for kk, v in rtr.items() if kk in ("macro_f1", "top1", "log_loss")})
        res.update(fold=k, train_s=tt)
        folds.append(res)
        if verbose:
            print(f"  [{model_name}] fold {k}: top1={res['top1']:.4f} macroF1={res['macro_f1']:.4f} "
                  f"logloss={res['log_loss']:.4f} train={tt:.1f}s", flush=True)
    return pd.DataFrame(folds)


def run_model(model_name, train_df, test_df, params=None, tag=None):
    tag = tag or model_name
    print(f"== {tag}", flush=True)
    cv = cross_validate(model_name, train_df, params)
    fit_df = train_df[train_df.aligned]
    m, tt = fit(model_name, fit_df[FEATURES].to_numpy(np.float32), fit_df.label.to_numpy(), params)
    test, p_test = evaluate(m, test_df, "student_test")
    os.makedirs("models/saved", exist_ok=True)
    joblib.dump(m, f"models/saved/{tag}.joblib", compress=3)
    dev = "CPU"
    run = dict(model=tag, family=FAMILY.get(model_name, model_name), device=dev, params=params or {},
               config=repr(make_model(model_name, params)),
               seed=SEED, n_splits=N_SPLITS, train_subsample=TRAIN_SUBSAMPLE.get(model_name),
               cv_mean=cv.drop(columns="fold").mean().to_dict(), cv_std=cv.drop(columns="fold").std().to_dict(),
               cv_folds=cv.to_dict("records"), test=test, full_train_s=tt)
    if model_name == "mlp":
        clf = m[-1]
        run["loss_curve"] = list(map(float, clf.loss_curve_))
        run["validation_scores"] = list(map(float, clf.validation_scores_))
    os.makedirs("results/runs", exist_ok=True)
    json.dump(run, open(f"results/runs/{tag}.json", "w"), indent=1, default=float)
    # confusion matrix on student_test (pair level, aligned points)
    _confusion(test_df, p_test, tag)
    os.makedirs("cache", exist_ok=True)
    np.save(f"cache/p_test_{tag}.npy", p_test)
    print(f"   CV top1={run['cv_mean']['top1']:.4f}±{run['cv_std']['top1']:.4f} macroF1={run['cv_mean']['macro_f1']:.4f} | "
          f"TEST top1={test['top1']:.4f} macroF1={test['macro_f1']:.4f} logloss={test['log_loss']:.4f}", flush=True)
    return run


def _confusion(df, p, tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    al = df.aligned.to_numpy()
    os.makedirs("results/figures/confusion", exist_ok=True)
    disp = ConfusionMatrixDisplay.from_predictions(df.label.to_numpy()[al], (p[al] >= 0.5).astype(int),
                                                   display_labels=["wrong arc", "true arc"], normalize="true", values_format=".3f")
    disp.ax_.set_title(f"{tag}: student_test pair-level confusion (row-normalised)")
    disp.figure_.tight_layout()
    disp.figure_.savefig(f"results/figures/confusion/{tag}.png", dpi=100)
    plt.close(disp.figure_)


def comparison_table():
    from models.reporting import export_auxiliary_metrics, generate_comparison_plots, generate_model_notes
    rows = []
    for f in sorted(os.listdir("results/runs")):
        if not f.endswith(".json"):
            continue
        r = json.load(open(f"results/runs/{f}"))
        c, s, t = r["cv_mean"], r["cv_std"], r["test"]
        rows.append({"model": r["model"], "family": r["family"], "device": r.get("device", "CPU"),
                     "key_config": json.dumps(r["params"]) if r["params"] else "defaults (see run json)",
                     "train_subsample": r.get("train_subsample"),
                     "cv_macro_f1": c["macro_f1"], "cv_macro_f1_std": s["macro_f1"], "cv_top1": c["top1"], "cv_top1_std": s["top1"],
                     "cv_log_loss": c["log_loss"], "cv_precision": c["precision"], "cv_recall": c["recall"], "cv_accuracy": c["accuracy"],
                     "cv_route_continuity": c["route_continuity"], "cv_train_top1": c.get("train_top1"), "cv_train_log_loss": c.get("train_log_loss"),
                     "test_top1": t["top1"], "test_macro_f1": t["macro_f1"], "test_log_loss": t["log_loss"], "test_precision": t["precision"],
                     "test_recall": t["recall"], "test_accuracy": t["accuracy"], "test_route_continuity": t["route_continuity"],
                     "train_time_s": r["full_train_s"], "test_inference_s": t["inference_s"],
                     "inference_ms_per_1k_rows": t["inference_ms_per_1k_rows"]})
    df = pd.DataFrame(rows).sort_values("cv_top1", ascending=False)
    df.to_csv("results/model_comparison.csv", index=False, float_format="%.5f")
    export_auxiliary_metrics()
    generate_comparison_plots()
    generate_model_notes()
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["logreg", "knn", "rf", "hgb", "lgbm", "xgb", "mlp"])
    a = ap.parse_args()
    train_df, test_df = load_split("train"), load_split("student_test")
    print(f"train rows {len(train_df):,} ({train_df.trip_id.nunique()} trips), test rows {len(test_df):,} "
          f"({test_df.trip_id.nunique()} trips)", flush=True)
    for name in a.models:
        run_model(name, train_df, test_df)
        comparison_table()
    print(comparison_table()[["model", "cv_macro_f1", "cv_top1", "cv_log_loss", "test_top1", "test_macro_f1", "test_log_loss"]].to_string())


if __name__ == "__main__":
    main()
