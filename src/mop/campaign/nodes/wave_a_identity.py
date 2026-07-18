"""Wave A science node: universal event-referent identity and intake.

This bed asks a single universal-intake question: does binding events to persistent referents by
feature continuity recover referent identity better than the same binder with continuity destroyed?

We build N independent synthetic sessions (the experimental units). Each session is a multi-referent
event stream over discrete time. Several referents emit noisy feature observations (a small position plus
embedding vector) that drift slowly, so identity is carried by cross-time continuity rather than by any
single frame. Each session deliberately contains one OCCLUSION gap (a referent goes dark for a few frames
then reappears) and one SPLIT then MERGE (a referent spawns a diverging child, and two referents fuse into
one). Ground-truth referent ids are fixed by construction, so identity accuracy is unambiguous.

The candidate is a simple online nearest-in-feature referent-binder: it maintains one running feature
centroid per track, associates each frame's events to distinct tracks greedily by feature distance under a
threshold, survives short gaps with a max-gap window, and spawns a new track when nothing is near. Score is
best-map identity-tracking accuracy against the ground-truth referent ids.

Named control (shuffled-lineage): the exact same binder, but per frame the feature vectors are permuted
among the co-existing events, which destroys cross-time feature continuity while preserving per-frame
appearance separability. That control isolates continuity from appearance: it drops the binder to chance
without changing how separable the referents look inside any single frame.

Wrong-time control (recorded, secondary): the same binder fed the frames in a scrambled temporal order, so
the continuity signal is present but temporally misaligned. It is a diagnostic, not the primary test.

Per-unit paired delta is candidate_accuracy minus shuffled-lineage_accuracy (positive favors the binder).
The verdict comes from the shared exact sign-flip and a small structural SESOI. A tie or a wrong-direction
result is a legitimate null; nothing here is tuned toward a positive.

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mop.campaign.nodes.framework import (
    exact_sign_flip_one_sided,
    honest_envelope,
    rng,
    verdict_from,
)
from mop.campaign.runners import NodeContext, RunResult, register_runner

# Structural geometry of a session. Kept fixed so the seal depends only on ctx.seed.
_N_SESSIONS = 12
_K_BASE = 4
_T = 20
_D_POS = 2
_D_EMB = 4
_D = _D_POS + _D_EMB
_RADIUS = 10.0
_EMB_SCALE = 8.0
_OBS_SIGMA = 0.5
_DRIFT_MAG = 0.15
_CHILD_DIVERGE = 0.6
_OCC_START, _OCC_END = 5, 8
_T_SPLIT = 10
_T_MERGE = 14
# Binder hyperparameters, fixed for both arms so the comparison is a fair paired contrast.
_THRESHOLD = 4.0
_EMA = 0.5
_MAX_GAP = 5
# Small structural SESOI on identity accuracy (ten accuracy points).
_SESOI = 0.10


def _unit_vector(gen: np.random.Generator) -> np.ndarray:
    """A deterministic random unit vector in feature space."""

    v = gen.standard_normal(_D)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def _base_center(k: int) -> np.ndarray:
    """A well-separated base center for referent k: a ring position plus a distinct embedding axis."""

    angle = 2.0 * np.pi * k / _K_BASE
    center = np.zeros(_D, dtype=np.float64)
    center[0] = _RADIUS * np.cos(angle)
    center[1] = _RADIUS * np.sin(angle)
    center[_D_POS + (k % _D_EMB)] = _EMB_SCALE
    return center


def _generate_session(gen: np.random.Generator) -> list[tuple[np.ndarray, list[int]]]:
    """Build one session as a list of frames in true time order: (feature_matrix, true_referent_ids).

    Referent ids: 0..3 are the base referents. Referent 1 splits a diverging child (id 4) at _T_SPLIT.
    Referents 2 and 3 merge into id 5 at _T_MERGE. Referent 0 is occluded on [_OCC_START, _OCC_END).
    """

    centers0 = [_base_center(k) for k in range(_K_BASE)]
    drifts = [_DRIFT_MAG * _unit_vector(gen) for _ in range(_K_BASE)]
    child_dir = _unit_vector(gen)

    def center_of(k: int, t: int) -> np.ndarray:
        return centers0[k] + t * drifts[k]

    frames: list[tuple[np.ndarray, list[int]]] = []
    for t in range(_T):
        feats: list[np.ndarray] = []
        ids: list[int] = []

        # Referent 0 (occluded on the gap window).
        if not (_OCC_START <= t < _OCC_END):
            feats.append(center_of(0, t))
            ids.append(0)

        # Referent 1 (parent of the split).
        feats.append(center_of(1, t))
        ids.append(1)

        # Referents 2 and 3 before the merge, else the merged referent 5.
        if t < _T_MERGE:
            feats.append(center_of(2, t))
            ids.append(2)
            feats.append(center_of(3, t))
            ids.append(3)
        else:
            feats.append(0.5 * (center_of(2, t) + center_of(3, t)))
            ids.append(5)

        # Child referent 4 appears at the split and diverges from its parent over time.
        if t >= _T_SPLIT:
            child = center_of(1, t) + _CHILD_DIVERGE * (t - _T_SPLIT) * child_dir
            feats.append(child)
            ids.append(4)

        mat = np.asarray(feats, dtype=np.float64)
        mat = mat + _OBS_SIGMA * gen.standard_normal(mat.shape)
        frames.append((mat, ids))
    return frames


def _bind(frame_feats: list[np.ndarray]) -> list[list[int]]:
    """Online nearest-in-feature binder with per-frame exclusivity and a max-gap survival window.

    Returns a per-frame list of assigned track ids. Tracks are formed from the feature stream only; no
    ground-truth id is ever consulted here.
    """

    tracks: list[dict[str, Any]] = []
    preds: list[list[int]] = []
    for fi, feats in enumerate(frame_feats):
        n = feats.shape[0]
        active = [j for j, tr in enumerate(tracks) if fi - tr["last"] <= _MAX_GAP]
        pairs: list[tuple[float, int, int]] = []
        for ei in range(n):
            for j in active:
                d = float(np.linalg.norm(feats[ei] - tracks[j]["centroid"]))
                if d <= _THRESHOLD:
                    pairs.append((d, ei, j))
        pairs.sort()
        assigned: dict[int, int] = {}
        used: set[int] = set()
        for _d, ei, j in pairs:
            if ei in assigned or j in used:
                continue
            assigned[ei] = j
            used.add(j)
        frame_pred: list[int] = []
        for ei in range(n):
            if ei in assigned:
                j = assigned[ei]
                tr = tracks[j]
                tr["centroid"] = _EMA * tr["centroid"] + (1.0 - _EMA) * feats[ei]
                tr["last"] = fi
                frame_pred.append(j)
            else:
                tracks.append({"centroid": feats[ei].copy(), "last": fi})
                frame_pred.append(len(tracks) - 1)
        preds.append(frame_pred)
    return preds


def _cluster_accuracy(true_labels: list[int], pred_labels: list[int]) -> float:
    """Best one-to-one map identity accuracy: fraction of events under the optimal injective track-to-id map.

    Solved exactly by a bitmask DP over the true-id rows (at most a handful), so it is deterministic and has
    no dependency beyond numpy.
    """

    true_ids = sorted(set(true_labels))
    pred_ids = sorted(set(pred_labels))
    ti = {v: i for i, v in enumerate(true_ids)}
    pj = {v: i for i, v in enumerate(pred_ids)}
    tn, pn = len(true_ids), len(pred_ids)
    contingency = np.zeros((tn, pn), dtype=np.int64)
    for tl, pl in zip(true_labels, pred_labels, strict=True):
        contingency[ti[tl], pj[pl]] += 1
    dp: dict[int, int] = {0: 0}
    for j in range(pn):
        col = contingency[:, j]
        ndp: dict[int, int] = {}
        for mask, val in dp.items():
            if ndp.get(mask, -1) < val:
                ndp[mask] = val
            for i in range(tn):
                if not (mask >> i) & 1 and col[i] > 0:
                    nm = mask | (1 << i)
                    nv = val + int(col[i])
                    if ndp.get(nm, -1) < nv:
                        ndp[nm] = nv
        dp = ndp
    best = max(dp.values())
    return best / len(true_labels)


def _flatten(preds: list[list[int]], id_frames: list[list[int]]) -> tuple[list[int], list[int]]:
    """Flatten aligned per-frame predictions and true ids into two parallel event-level label lists."""

    trues: list[int] = []
    p: list[int] = []
    for fp, fi in zip(preds, id_frames, strict=True):
        p.extend(fp)
        trues.extend(fi)
    return trues, p


def _run_session(seed: int, s: int) -> dict[str, Any]:
    """Score one session under the candidate, the shuffled-lineage control, and the wrong-time control."""

    frames = _generate_session(rng(seed, "session", s))
    feats = [f for f, _ in frames]
    id_frames = [ids for _, ids in frames]

    # Candidate: true temporal order, features intact.
    cand_preds = _bind(feats)
    t_true, t_pred = _flatten(cand_preds, id_frames)
    acc_cand = _cluster_accuracy(t_true, t_pred)

    # Named control (shuffled-lineage): permute features among co-existing events per frame; ids unchanged.
    cgen = rng(seed, "shuffle", s)
    ctrl_feats = []
    for mat in feats:
        perm = cgen.permutation(mat.shape[0])
        ctrl_feats.append(mat[perm])
    ctrl_preds = _bind(ctrl_feats)
    c_true, c_pred = _flatten(ctrl_preds, id_frames)
    acc_ctrl = _cluster_accuracy(c_true, c_pred)

    # Wrong-time control (recorded): scramble the temporal order of whole frames, features intact.
    wgen = rng(seed, "wrongtime", s)
    order = list(wgen.permutation(_T))
    wt_feats = [feats[o] for o in order]
    wt_ids = [id_frames[o] for o in order]
    wt_preds = _bind(wt_feats)
    w_true, w_pred = _flatten(wt_preds, wt_ids)
    acc_wt = _cluster_accuracy(w_true, w_pred)

    n_true = len(set(t_true))
    return {
        "session": int(s),
        "n_events": int(len(t_true)),
        "n_true_referents": int(n_true),
        "chance_accuracy": round(1.0 / n_true, 6),
        "candidate_accuracy": round(float(acc_cand), 6),
        "shuffled_lineage_accuracy": round(float(acc_ctrl), 6),
        "wrong_time_accuracy": round(float(acc_wt), 6),
        "delta": round(float(acc_cand - acc_ctrl), 6),
        "delta_wrong_time": round(float(acc_cand - acc_wt), 6),
    }


@register_runner("wave_a.event_referent_identity")
def wave_a_identity_runner(params: dict[str, Any], ctx: NodeContext) -> RunResult:
    """Universal event-referent identity bed: continuity binder vs a shuffled-lineage control."""

    per_unit = [_run_session(ctx.seed, s) for s in range(_N_SESSIONS)]
    deltas = [u["delta"] for u in per_unit]
    wrong_time_deltas = [u["delta_wrong_time"] for u in per_unit]

    sign_flip = exact_sign_flip_one_sided(deltas)
    wrong_time_sign_flip = exact_sign_flip_one_sided(wrong_time_deltas)
    mean_delta = float(sign_flip["mean_delta"])
    one_sided_p = float(sign_flip["one_sided_p"])
    verdict = verdict_from(mean_delta, one_sided_p, _SESOI)
    is_null = verdict != "survives"

    coverage = {
        "form_family": "events",
        "phenomenon": "event_formation_identity",
        "mechanism_family": "nearest_feature_referent_binding",
        "unit_class": "synthetic_event_session",
        "evidence_level": "M1",
    }
    content = honest_envelope(ctx.node_id, "mop-campaign-wave_a_identity/v1", coverage)
    content.update(
        {
            "n_units": int(_N_SESSIONS),
            "sesoi": _SESOI,
            "control_description": (
                "shuffled-lineage: the identical nearest-in-feature binder, but per frame the feature "
                "vectors are permuted among the co-existing events, destroying cross-time feature "
                "continuity while preserving per-frame appearance separability (chance identity)."
            ),
            "wrong_time_control_description": (
                "the identical binder fed the frames in a scrambled temporal order, so continuity is "
                "present but temporally misaligned; recorded as a secondary diagnostic only."
            ),
            "per_unit": per_unit,
            "deltas": [round(float(d), 6) for d in deltas],
            "sign_flip": sign_flip,
            "wrong_time_deltas": [round(float(d), 6) for d in wrong_time_deltas],
            "wrong_time_sign_flip": wrong_time_sign_flip,
            "mean_delta": round(mean_delta, 6),
            "one_sided_p": round(one_sided_p, 12),
            "verdict": verdict,
            "is_null": bool(is_null),
            "alternative_explanation": (
                "appearance-only clustering: coexisting referents are separable inside any single frame, so "
                "a per-frame appearance clusterer could split them without any temporal binding. The "
                "shuffled-lineage control preserves that per-frame separability while destroying cross-time "
                "continuity, so a candidate advantage over it cannot be explained by appearance alone."
            ),
            "failure_domain": (
                "observation noise large relative to inter-referent separation, drift per frame exceeding "
                "the association threshold, occlusion gaps longer than the track max-gap window, or dense "
                "near-simultaneous split and merge events collapse the continuity signal and drive the "
                "binder toward the shuffled-lineage floor."
            ),
        }
    )

    path, seal = ctx.seal_json(f"{ctx.node_id}.json", content)
    return RunResult(
        artifact_path=str(path),
        seal=seal,
        verdict=verdict,
        is_null=is_null,
        detail={
            "mean_delta": round(mean_delta, 6),
            "one_sided_p": round(one_sided_p, 12),
            "n_units": int(_N_SESSIONS),
            "mean_wrong_time_delta": round(float(wrong_time_sign_flip["mean_delta"]), 6),
        },
    )
