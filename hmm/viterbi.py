"""Viterbi decoding for sequential route reconstruction (Issue #17)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from features.trip_io import Trip


def decode_viterbi(
    trip: Trip,
    df_cands: pd.DataFrame,
    n_points: int,
    hmm_config: dict | None = None,
) -> tuple[list, list, list]:
    """Sequential Viterbi decoding over candidate states.
    
    Returns:
    - viterbi_arc: list of decoded arc IDs per GPS point (or None)
    - viterbi_prob: list of candidate posterior probabilities for chosen arc
    - clf_argmax: list of argmax classifier arc IDs per point
    """
    if hmm_config is None:
        hmm_config = {}

    if len(df_cands) == 0:
        return [None] * n_points, [None] * n_points, [None] * n_points

    point_groups = df_cands.groupby("point_idx")
    cands_by_point = {int(pt): g for pt, g in point_groups}

    # Classifier-only argmax per point
    clf_argmax = [None] * n_points
    for pt, g in cands_by_point.items():
        best_idx = g["p"].idxmax()
        clf_argmax[pt] = int(df_cands.loc[best_idx, "arc_id"])

    p_same = hmm_config.get("transition_same_arc", 0.50)
    p_conn = hmm_config.get("transition_connected_arc", 0.45)
    p_other = hmm_config.get("transition_disconnected_penalty", 0.001)

    log_same = np.log(max(p_same, 1e-6))
    log_conn = np.log(max(p_conn, 1e-6))
    log_other = np.log(max(p_other, 1e-6))

    af, at = trip.arc_from, trip.arc_to

    # DP states: list of (states, log_probs)
    dp = []
    backpointers = []
    point_has_cand = []

    for t in range(n_points):
        if t not in cands_by_point:
            point_has_cand.append(False)
            continue

        point_has_cand.append(True)
        g = cands_by_point[t]
        arcs_t = g.arc_id.to_numpy()
        probs_t = g.p.to_numpy()
        emit_t = np.log(np.clip(probs_t, 1e-6, 1.0))

        if len(dp) == 0:
            dp.append(emit_t)
            backpointers.append(None)
        else:
            prev_probs = dp[-1]
            prev_arcs = cands_by_point[valid_points[-1]].arc_id.to_numpy()

            same_mask = prev_arcs[:, None] == arcs_t[None, :]
            conn_mask = at[prev_arcs, None] == af[None, arcs_t]

            trans = np.full((len(prev_arcs), len(arcs_t)), log_other, dtype=float)
            trans[conn_mask] = log_conn
            trans[same_mask] = log_same

            scores = prev_probs[:, None] + trans
            best_prev = np.argmax(scores, axis=0)
            curr_prob = np.max(scores, axis=0) + emit_t

            dp.append(curr_prob)
            backpointers.append(best_prev)

        valid_points = [i for i, h in enumerate(point_has_cand) if h]

    # Traceback
    viterbi_arc = [None] * n_points
    viterbi_prob = [None] * n_points

    if len(dp) > 0:
        curr_best = np.argmax(dp[-1])
        for step in range(len(dp) - 1, -1, -1):
            pt = valid_points[step]
            g = cands_by_point[pt]
            chosen_arc = int(g.arc_id.iloc[curr_best])
            chosen_prob = float(g.p.iloc[curr_best])
            viterbi_arc[pt] = chosen_arc
            viterbi_prob[pt] = chosen_prob
            if backpointers[step] is not None:
                curr_best = backpointers[step][curr_best]

    return viterbi_arc, viterbi_prob, clf_argmax
