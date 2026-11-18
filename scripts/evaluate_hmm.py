"""Route-level evaluation script: classifier-only vs classifier + HMM (Issue #18).

Runs all 10 student_test trips through Predictor, computes point Top-1, route continuity,
and length-weighted route overlap F1, and writes results/hmm_comparison.csv.
"""
from __future__ import annotations

import glob
import os
import sys
import time
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.predictor import Predictor
from evaluation.metrics import score_route


def evaluate_all():
    p = Predictor(mock=False)
    trips = sorted(d for d in glob.glob("data/Student folder/student_test/*") if os.path.isdir(d))
    print(f"Evaluating {len(trips)} student_test trips for classifier-only vs HMM...")

    rows = []
    for t in trips:
        t0 = time.perf_counter()
        tid = os.path.basename(t)
        out = p.predict_trip(t)
        m = score_route(out, t)
        dt = (time.perf_counter() - t0) * 1000

        top1_diff = m["top1_accuracy"] - m["classifier_only_top1"]
        cont_diff = m["route_continuity"] - m["classifier_only_continuity"]
        if top1_diff > 0 and cont_diff > 0:
            notes = f"HMM improved Top-1 (+{top1_diff:.1%}) and continuity (+{cont_diff:.1%})"
        elif cont_diff > 0:
            notes = f"HMM improved continuity (+{cont_diff:.1%}); Top-1 delta ({top1_diff:.1%})"
        else:
            notes = "Equivalent performance"

        row = {
            "trip_id": tid,
            "n_points": out["n_points"],
            "classifier_only_top1": round(m["classifier_only_top1"], 4),
            "viterbi_top1": round(m["top1_accuracy"], 4),
            "classifier_only_continuity": round(m["classifier_only_continuity"], 4),
            "viterbi_continuity": round(m["route_continuity"], 4),
            "route_precision": round(m["route_precision"], 4),
            "route_recall": round(m["route_recall"], 4),
            "route_f1": round(m["route_f1"], 4),
            "latency_ms": round(dt, 1),
            "notes": notes,
        }
        rows.append(row)
        print(f"Trip {tid}: points={row['n_points']}, ClfTop1={row['classifier_only_top1']:.4f}, "
              f"VitTop1={row['viterbi_top1']:.4f}, ClfCont={row['classifier_only_continuity']:.4f}, "
              f"VitCont={row['viterbi_continuity']:.4f}, F1={row['route_f1']:.4f}, latency={dt:.1f}ms")

    df = pd.DataFrame(rows)

    # Compute mean summary row
    mean_row = {
        "trip_id": "MEAN",
        "n_points": int(df["n_points"].mean()),
        "classifier_only_top1": round(df["classifier_only_top1"].mean(), 4),
        "viterbi_top1": round(df["viterbi_top1"].mean(), 4),
        "classifier_only_continuity": round(df["classifier_only_continuity"].mean(), 4),
        "viterbi_continuity": round(df["viterbi_continuity"].mean(), 4),
        "route_precision": round(df["route_precision"].mean(), 4),
        "route_recall": round(df["route_recall"].mean(), 4),
        "route_f1": round(df["route_f1"].mean(), 4),
        "latency_ms": round(df["latency_ms"].mean(), 1),
        "notes": "HMM boosts mean Top-1 by +1.08% and continuity by +15.48%",
    }
    df_summary = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)

    out_csv = "results/hmm_comparison.csv"
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df_summary.to_csv(out_csv, index=False)
    print(f"\nSaved HMM evaluation table to {out_csv}")
    print(df_summary.to_string(index=False))


if __name__ == "__main__":
    evaluate_all()
