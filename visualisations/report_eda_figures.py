"""Generate the EDA figures used in the technical report (sections 2.2).

Outputs (results/figures/report/):
  eda_trip_00000005_gps_network.png  GPS trajectory over the road network (train trip 00000005)
  mlp_training_curve.png             MLP epoch-wise training history
Prints dataset/missing-value statistics computed from results/eda_inventory.csv.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

OUT = "results/figures/report"
TRIP = "00000005"
TRIP_DIR = os.path.join("data", "Student folder", "train", TRIP)


def read(name, ncol):
    a = np.loadtxt(os.path.join(TRIP_DIR, f"{TRIP}.{name}"), ndmin=2)
    return a[:, :ncol]


def gps_network_plot():
    track = read("track", 3)
    nodes = read("nodes", 2)
    arcs = read("arcs", 2).astype(int)
    route = np.loadtxt(os.path.join(TRIP_DIR, f"{TRIP}.route"), dtype=int, ndmin=1)
    pad = 0.004
    x0, x1 = track[:, 0].min() - pad, track[:, 0].max() + pad
    y0, y1 = track[:, 1].min() - pad, track[:, 1].max() + pad
    # widen the shorter axis so the view is roughly square on the ground
    k = np.cos(np.radians(track[:, 1].mean()))
    half = max((x1 - x0) * k, y1 - y0) / 2
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    x0, x1, y0, y1 = cx - half / k, cx + half / k, cy - half, cy + half
    a, b = nodes[arcs[:, 0]], nodes[arcs[:, 1]]
    inside = ((a[:, 0] > x0) & (a[:, 0] < x1) & (a[:, 1] > y0) & (a[:, 1] < y1))
    segs = np.stack([a[inside], b[inside]], axis=1)
    rsegs = np.stack([nodes[arcs[route, 0]], nodes[arcs[route, 1]]], axis=1)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.add_collection(LineCollection(segs, colors="#9ca3af", linewidths=0.4, label=f"Road network ({inside.sum():,} arcs in view)"))
    ax.add_collection(LineCollection(rsegs, colors="#16a34a", linewidths=6, alpha=0.45, label=f"Reference route ({len(route)} arcs)"))
    ax.scatter(track[:, 0], track[:, 1], s=1.5, c="#1d4ed8", label=f"GPS observations ({len(track):,})", zorder=3)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_aspect(1 / np.cos(np.radians(track[:, 1].mean())))
    ax.set_title(f"Train trip {TRIP}: GPS trajectory over its road network")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=8)
    fig.tight_layout()
    path = os.path.join(OUT, f"eda_trip_{TRIP}_gps_network.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print("wrote", path)


def stats():
    inv = pd.read_csv("results/eda_inventory.csv", dtype={"trip_id": str})
    for part, g in inv.groupby("partition"):
        print(f"[{part}] trips={len(g)} gps={g.n_gps.sum():,} nan(track/nodes/arcs)="
              f"{g.track_nan.sum()}/{g.nodes_nan.sum()}/{g.arcs_nan.sum()} invalid_coords={(~g.valid_track_coords).sum()} "
              f"non_increasing_ts={g.n_non_increasing_ts.sum()} gaps>1s={g.n_gaps_gt_1s.sum():,} "
              f"median_pct_dt1s={g.pct_dt_1s.median():.2f} max_dt={g.max_dt_s.max():,.0f} "
              f"route_invalid_ids={(~g.route_valid_ids).sum()} trips_with_route_discontinuity={(g.n_route_discontinuities>0).sum()} "
              f"median_components={g.n_components.median():.0f} median_largest_comp_pct={g.largest_comp_pct.median():.1f} "
              f"lon[{g.lon_min.min():.2f},{g.lon_max.max():.2f}] lat[{g.lat_min.min():.2f},{g.lat_max.max():.2f}]")
    print("top max_dt:\n", inv.nlargest(3, "max_dt_s")[["partition", "trip_id", "max_dt_s"]].to_string(index=False))
    print("lon<0 trips:", inv[inv.lon_min < 0].trip_id.tolist())
    cs = pd.read_csv("results/candidate_stats.csv", dtype={"trip_id": str})
    for part, g in cs.groupby("split"):
        print(f"[cand {part}] rows={g.n_candidate_rows.sum():,} mean_cands/pt={g.n_candidate_rows.sum()/g.n_points.sum():.2f} "
              f"empty_pts={g.empty_candidate_points.sum():,} aligned={(g.aligned_frac*g.n_points).sum()/g.n_points.sum():.4f} "
              f"median_align_dist={g.median_align_dist.median():.2f}")
    mr = pd.read_csv("results/candidate_miss_rate.csv", dtype={"trip_id": str})
    for part, g in mr.groupby("partition"):
        print(f"[miss {part}] valid_miss_rate={100*g.n_misses_valid.sum()/g.n_valid_pts.sum():.2f}% "
              f"raw_miss_rate={100*g.n_misses.sum()/g.n_gps.sum():.2f}%")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    gps_network_plot()
    stats()


def mlp_curve():
    """MLP is the only model with an epoch-wise history (loss_curve / validation_scores in results/runs/mlp.json)."""
    import json
    d = json.load(open("results/runs/mlp.json"))
    loss, val = d.get("loss_curve") or [], d.get("validation_scores") or []
    fig, ax1 = plt.subplots(figsize=(7, 3.6))
    ax1.plot(range(1, len(loss) + 1), loss, "o-", color="#2563eb", label="Training loss (log loss)")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Training loss")
    ax2 = ax1.twinx()
    ax2.plot(range(1, len(val) + 1), val, "s--", color="#ea580c", label="Validation accuracy (early-stopping split)")
    ax2.set_ylabel("Validation accuracy")
    h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="center right", fontsize=8)
    ax1.set_title("MLP training history (final full-train fit)")
    fig.tight_layout()
    path = os.path.join(OUT, "mlp_training_curve.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print("wrote", path, "epochs", len(loss), "val", len(val))


def positives():
    mr = pd.read_csv("results/candidate_miss_rate.csv", dtype={"trip_id": str})
    for part, g in mr.groupby("partition"):
        print(f"[pos {part}] positive share ~= hits/rows = {g.n_hits.sum()/g.n_candidate_rows.sum():.4f}")


if __name__ == "__main__":
    mlp_curve()
    positives()
