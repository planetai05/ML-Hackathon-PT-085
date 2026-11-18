"""Test all 10 student_test trips through real Predictor and compute summary metrics."""
import glob
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.predictor import Predictor

def main():
    p = Predictor(mock=False)
    trips = sorted(d for d in glob.glob("data/Student folder/student_test/*") if os.path.isdir(d))
    print(f"Testing real Predictor on {len(trips)} student_test trips...\n")

    results = []
    for t in trips:
        t0 = time.time()
        tid = os.path.basename(t)
        out = p.predict_trip(t)
        metrics = p.score(out, t)
        dt = (time.time() - t0) * 1000
        print(f"Trip {tid}: points={out['n_points']:>4}, "
              f"Viterbi Top1={metrics['top1_accuracy']:.4f}, "
              f"Clf Top1={metrics['classifier_only_top1']:.4f}, "
              f"Route F1={metrics['route_f1']:.4f}, "
              f"Continuity={metrics['route_continuity']:.4f}, "
              f"Total={dt:>5.1f}ms")
        results.append(metrics)

    mean_v_top1 = sum(r["top1_accuracy"] for r in results) / len(results)
    mean_c_top1 = sum(r["classifier_only_top1"] for r in results) / len(results)
    mean_f1 = sum(r["route_f1"] for r in results) / len(results)
    mean_cont = sum(r["route_continuity"] for r in results) / len(results)

    print(f"\nSummary over all {len(trips)} trips:")
    print(f"  Mean Viterbi Top-1 Accuracy:     {mean_v_top1:.4f}")
    print(f"  Mean Classifier-Only Top-1:      {mean_c_top1:.4f}")
    print(f"  Mean Route F1-Score:             {mean_f1:.4f}")
    print(f"  Mean Route Continuity:           {mean_cont:.4f}")

if __name__ == "__main__":
    main()
