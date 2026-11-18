"""Route-level evaluation: classifier-only vs classifier + HMM/Viterbi (issue #18).

python -m evaluation.route_eval --base lgbm --params results/hpo_best_params.json --tag lgbm_tuned

1. Out-of-fold (GroupKFold by trip) probabilities for the TRAIN trips, so decoder tuning never
   sees probabilities from a model trained on the same trip.
2. Grid-search the HMM parameters (beta, emission weight, distance weight) on a seeded subset of
   train trips using those OOF probabilities (objective: mean HMM Top-1).
3. Evaluate classifier-only vs HMM on the 10 student_test trips with the model refit on all
   train trips. Writes results/hmm_comparison.csv, results/hmm_tuning.csv, artifacts/hmm_config.json.
Evaluator data is never used.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from evaluation.metrics import continuity, route_overlap
from features.build_features import FEATURES
from features.labels import align_route
from features.trip_io import DATA_DIR, load_trip
from hmm.model import HMMConfig, RoadGraph
from hmm.viterbi import classifier_only, decode, reconstruct
from models.train import SEED, fit, train_rows


def oof_probs(base, params, df):
    p = np.zeros(len(df))
    for tr, va in GroupKFold(5).split(df, groups=df.trip_id):
        f = df.iloc[tr]
        f = f[f.aligned]
        m, _ = fit(base, f[FEATURES].to_numpy(np.float32), f.label.to_numpy(), params)
        p[va] = m.predict_proba(df.iloc[va][FEATURES].to_numpy(np.float32))[:, 1]
    return p


def trip_eval(trip, g, cand, prob, true_arc, aligned, cfg):
    t = trip.track[:, 2]
    co, _ = classifier_only(trip, cand, prob)
    t0 = time.perf_counter()
    r = decode(trip, cand, prob, cfg, g)
    dec_s = time.perf_counter() - t0
    hm = r["per_point_arc"]
    co_route = [a for s in reconstruct(g, co, t, cfg) for a in s]
    out = dict(trip_id=trip.trip_id, n_points=len(t), n_aligned=int(aligned.sum()),
               clf_top1=float((co[aligned] == true_arc[aligned]).mean()),
               hmm_top1=float((hm[aligned] == true_arc[aligned]).mean()),
               clf_continuity=continuity(co[co >= 0], trip.arc_from, trip.arc_to),
               hmm_continuity=continuity(hm[hm >= 0], trip.arc_from, trip.arc_to),
               clf_segments=len(reconstruct(g, co, t, cfg)), hmm_segments=len(r["segments"]),
               decode_s=dec_s)
    for k, v in route_overlap(co_route, trip.route, trip.arc_len).items():
        out["clf_" + k] = v
    for k, v in route_overlap(r["route_arcs"], trip.route, trip.arc_len).items():
        out["hmm_" + k] = v
    return out


def prepare(split, df, prob):
    """Yield (trip, graph, cand_df, prob, true_arc, aligned) for each trip in df."""
    for tid, idx in df.groupby("trip_id", sort=True).indices.items():
        trip = load_trip(f"{DATA_DIR}/{split}/{tid}", with_route=True)
        ta, al, _ = align_route(trip)
        cand = df.iloc[idx][["point_idx", "arc_id", "dist", "proj_t"]].rename(columns={"proj_t": "t"}).reset_index(drop=True)
        yield trip, RoadGraph(trip), cand, prob[idx], ta, al


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="lgbm")
    ap.add_argument("--params", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--n-tune-trips", type=int, default=16)
    a = ap.parse_args()
    params = json.load(open(a.params)) if a.params else None
    tag = a.tag or a.base
    train = pd.read_parquet("cache/features_train.parquet")
    test = pd.read_parquet("cache/features_student_test.parquet")

    # ---- 1. OOF probabilities on train
    t0 = time.time()
    p_oof = oof_probs(a.base, params, train)
    print(f"OOF probs in {time.time() - t0:.0f}s", flush=True)

    # ---- 2. tune decoder on a subset of train trips
    rng = np.random.default_rng(SEED)
    tune_trips = sorted(rng.choice(train.trip_id.unique(), a.n_tune_trips, replace=False))
    sub = train[train.trip_id.isin(tune_trips)]
    prepped = list(prepare("train", sub, p_oof[train.trip_id.isin(tune_trips).to_numpy()]))
    grid = list(itertools.product([2.0, 4.0, 8.0, 16.0, 32.0], [0.5, 1.0, 2.0], [0.0, 1.0]))
    rows = []
    for beta, ew, dw in grid:
        cfg = HMMConfig(beta=beta, emission_weight=ew, dist_weight=dw)
        res = [trip_eval(*x, cfg) for x in prepped]
        r = pd.DataFrame(res)
        rows.append(dict(beta=beta, emission_weight=ew, dist_weight=dw, hmm_top1=r.hmm_top1.mean(), clf_top1=r.clf_top1.mean(),
                         hmm_route_f1=r.hmm_route_f1.mean(), hmm_continuity=r.hmm_continuity.mean()))
        print(f"  beta={beta} ew={ew} dw={dw}: hmm_top1={rows[-1]['hmm_top1']:.4f} (clf {rows[-1]['clf_top1']:.4f})", flush=True)
    tune = pd.DataFrame(rows).sort_values("hmm_top1", ascending=False)
    tune.to_csv("results/hmm_tuning.csv", index=False, float_format="%.5f")
    b = tune.iloc[0]
    cfg = HMMConfig(beta=float(b.beta), emission_weight=float(b.emission_weight), dist_weight=float(b.dist_weight))
    json.dump(dict(cfg.to_dict(), tuned_on=f"OOF probabilities, {a.n_tune_trips} train trips: {', '.join(tune_trips)}"),
              open("artifacts/hmm_config.json", "w"), indent=1)
    print("best", cfg, flush=True)

    # ---- 3. student_test: model refit on all train trips
    f = train_rows(train)
    m, _ = fit(a.base, f[FEATURES].to_numpy(np.float32), f.label.to_numpy(), params)
    p_test = m.predict_proba(test[FEATURES].to_numpy(np.float32))[:, 1]
    res = pd.DataFrame([trip_eval(*x, cfg) for x in prepare("student_test", test, p_test)])
    w = res.n_aligned / res.n_aligned.sum()
    summ = {c: (float((res[c] * w).sum()) if c.endswith("top1") else float(res[c].mean())) for c in res.columns if c not in ("trip_id",)}
    summ["trip_id"] = "ALL (top1 point-weighted, others trip-mean)"
    res = pd.concat([res, pd.DataFrame([summ])], ignore_index=True)
    res.insert(1, "model", tag)
    res.to_csv("results/hmm_comparison.csv", index=False, float_format="%.5f")
    print(res[["trip_id", "clf_top1", "hmm_top1", "clf_continuity", "hmm_continuity", "clf_route_f1", "hmm_route_f1", "hmm_segments", "decode_s"]].to_string())


if __name__ == "__main__":
    main()
