"""Visual check of route-to-point label alignment (issue #9).
python -m features.plot_alignment <split> <trip_id> [<trip_id> ...]"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from features.labels import align_route
from features.trip_io import DATA_DIR, load_trip


def plot(split, tid, out):
    tr = load_trip(f"{DATA_DIR}/{split}/{tid}", with_route=True)
    ta, al, d = align_route(tr)
    fig, ax = plt.subplots(figsize=(8, 8))
    a, b = tr.node_xy[tr.arc_from[tr.route]], tr.node_xy[tr.arc_to[tr.route]]
    for p, q in zip(a, b):
        ax.plot([p[0], q[0]], [p[1], q[1]], color="green", lw=2, alpha=.6)
    ax.scatter(*tr.xy[al].T, s=4, c="blue", label="GPS (aligned)")
    ax.scatter(*tr.xy[~al].T, s=8, c="red", label="GPS (unaligned)")
    idx = np.where(al)[0][::5]  # connector from point to the midpoint of its aligned arc
    m = (tr.node_xy[tr.arc_from[ta[idx]]] + tr.node_xy[tr.arc_to[ta[idx]]]) / 2
    for i, mm in zip(idx, m):
        ax.plot([tr.xy[i, 0], mm[0]], [tr.xy[i, 1], mm[1]], color="orange", lw=.5)
    ax.set_title(f"{split} trip {tid}: label alignment ({al.mean():.1%} aligned, median {np.median(d[al]):.1f} m)\n"
                 "green = reference route, orange = point -> aligned-arc midpoint (every 5th)")
    ax.set_aspect("equal"); ax.legend(loc="best"); ax.set_xlabel("UTM x (m)"); ax.set_ylabel("UTM y (m)")
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)


if __name__ == "__main__":
    for t in sys.argv[2:]:
        plot(sys.argv[1], t, f"results/figures/alignment/{sys.argv[1]}_{t}.png")
