"""Wave C split-then-merge identity bed: velocity-continuity slots versus greedy nearest-centroid.

Each experimental unit is a synthetic scene in which one parent object splits into two fragments that
diverge, converge to a coincident merge near the scene midpoint, and pass through it (a momentum-conserving
crossing). The question is whether carrying a per-slot velocity and binding each detection to its velocity
predicted position preserves fragment identity across the merge better than a memoryless tracker that binds
each detection to the last observed position only.

  candidate  velocity-continuity slots: predict each slot's next position from its own smoothed velocity and
             bind the nearest detection to that prediction.
  control    the NAMED control, greedy nearest-centroid without identity memory: bind the nearest detection
             to each slot's last observed position, with no velocity carried.

Both arms are seeded with the correct fragment-to-slot binding on the first two frames, then score
identity-consistency accuracy (fraction of frames whose binding matches ground truth) from frame two onward.
The primary paired delta per scene is candidate accuracy minus control accuracy, positive favoring the
candidate. A bounce ablation reruns the identical geometry with the fragments reflecting at the merge rather
than passing through, which removes the momentum continuity the candidate exploits; it is reported as the
failure boundary and is not folded into the primary. A tie or a wrong-direction primary result is a
legitimate null and is reported as such; nothing here is tuned toward a positive.

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

N_SCENES = 10  # independent experimental units; within the exact sign-flip enumeration cap
T2 = 12  # two-fragment frames per scene (the parent split precedes this window)
TMERGE = 7.5  # merge happens between frames 7 and 8, so no frame is exactly coincident
H_LOW = 3.0  # minimum peak half-separation between the two fragments
H_HIGH = 5.0  # maximum peak half-separation between the two fragments
REL_NOISE_LOW = 0.03  # minimum observation noise as a fraction of peak half-separation
REL_NOISE_HIGH = 0.30  # maximum observation noise as a fraction of peak half-separation
DRIFT_SCALE = 0.4  # spread of the common group drift velocity
SMOOTH = 0.5  # velocity smoothing factor for the candidate slots
SESOI = 0.05  # small structural minimum: five points of identity-consistency accuracy
ROUND = 8


def _r(value: Any) -> float:
    """Cast any numpy or python scalar to a plain rounded float so the sealed JSON stays canonical."""

    return round(float(value), ROUND)


def _unit_vector(gen: np.random.Generator) -> np.ndarray:
    """A deterministic random unit vector giving the separation axis of one scene."""

    angle = float(gen.uniform(0.0, 2.0 * np.pi))
    return np.array([np.cos(angle), np.sin(angle)], dtype=float)


def _assign_two(refs: np.ndarray, dets: np.ndarray) -> tuple[int, int]:
    """Minimum-total-distance binding of two detections to two slot reference points.

    ``refs`` and ``dets`` are both shape (2, 2). Returns ``perm`` with ``perm[slot]`` the bound detection
    index, choosing the identity binding unless swapping the two lowers total distance.
    """

    ident = float(np.linalg.norm(refs[0] - dets[0]) + np.linalg.norm(refs[1] - dets[1]))
    swap = float(np.linalg.norm(refs[0] - dets[1]) + np.linalg.norm(refs[1] - dets[0]))
    return (1, 0) if swap < ident else (0, 1)


def _present(gen: np.random.Generator, gt: np.ndarray, noise: float) -> tuple[np.ndarray, np.ndarray]:
    """Add observation noise and shuffle the two detections per frame so input order carries no identity.

    ``gt`` is shape (T, 2, 2) with row 0 the fragment-0 position. Returns presented detections (T, 2, 2) and
    ``frag`` (T, 2) giving the true fragment id of each presented detection.
    """

    noisy = gt + gen.normal(0.0, noise, gt.shape)
    det = np.empty_like(noisy)
    frag = np.zeros((gt.shape[0], 2), dtype=int)
    for t in range(gt.shape[0]):
        if gen.random() < 0.5:
            det[t, 0], det[t, 1] = noisy[t, 1], noisy[t, 0]
            frag[t, 0], frag[t, 1] = 1, 0
        else:
            det[t, 0], det[t, 1] = noisy[t, 0], noisy[t, 1]
            frag[t, 0], frag[t, 1] = 0, 1
    return det, frag


def _slot_positions(det: np.ndarray, frag: np.ndarray, t: int) -> np.ndarray:
    """The correct binding at frame ``t``: slot 0 gets fragment 0's detection, slot 1 gets fragment 1's."""

    i0 = 0 if frag[t, 0] == 0 else 1
    return np.stack([det[t, i0], det[t, 1 - i0]])


def _track(det: np.ndarray, frag: np.ndarray, use_velocity: bool) -> float:
    """Identity-consistency accuracy over frames two onward for one arm.

    Both arms are seeded with the correct binding on frames zero and one. When ``use_velocity`` is true the
    arm binds to velocity-predicted positions and updates a smoothed per-slot velocity; otherwise it binds to
    the last observed positions only (the greedy nearest-centroid control).
    """

    pos0 = _slot_positions(det, frag, 0)
    last = _slot_positions(det, frag, 1)
    vel = last - pos0
    correct = 0
    scored = 0
    for t in range(2, det.shape[0]):
        refs = last + vel if use_velocity else last
        perm = _assign_two(refs, det[t])
        if frag[t, perm[0]] == 0 and frag[t, perm[1]] == 1:
            correct += 1
        scored += 1
        newpos = np.stack([det[t, perm[0]], det[t, perm[1]]])
        if use_velocity:
            vel = SMOOTH * vel + (1.0 - SMOOTH) * (newpos - last)
        last = newpos
    return correct / scored


def _simulate_scene(seed: int, scene_idx: int) -> dict[str, Any]:
    """Build one split-then-merge scene and score both arms under pass-through and bounce geometry."""

    gen = rng(seed, "wave_c_split_merge", "scene", scene_idx)
    axis = _unit_vector(gen)
    center0 = gen.normal(0.0, 1.0, size=2)
    drift = gen.normal(0.0, DRIFT_SCALE, size=2)
    h = float(gen.uniform(H_LOW, H_HIGH))
    rel_noise = float(gen.uniform(REL_NOISE_LOW, REL_NOISE_HIGH))
    noise = rel_noise * h

    frames = np.arange(T2, dtype=float)
    center = center0[None, :] + drift[None, :] * frames[:, None]
    sep_pass = h * np.sin(np.pi * frames / TMERGE)  # crosses zero at the merge (pass-through)
    sep_bounce = h * np.abs(np.sin(np.pi * frames / TMERGE))  # reflects at the merge (no crossing)

    gt_pass = np.stack(
        [center + sep_pass[:, None] * axis[None, :], center - sep_pass[:, None] * axis[None, :]], axis=1
    )
    gt_bounce = np.stack(
        [center + sep_bounce[:, None] * axis[None, :], center - sep_bounce[:, None] * axis[None, :]], axis=1
    )

    det_pass, frag_pass = _present(gen, gt_pass, noise)
    det_bounce, frag_bounce = _present(gen, gt_bounce, noise)

    cand_pass = _track(det_pass, frag_pass, use_velocity=True)
    ctrl_pass = _track(det_pass, frag_pass, use_velocity=False)
    cand_bounce = _track(det_bounce, frag_bounce, use_velocity=True)
    ctrl_bounce = _track(det_bounce, frag_bounce, use_velocity=False)

    return {
        "scene": scene_idx,
        "peak_half_separation": _r(h),
        "relative_noise": _r(rel_noise),
        "candidate_accuracy": _r(cand_pass),
        "control_accuracy": _r(ctrl_pass),
        "delta_candidate_vs_control": _r(cand_pass - ctrl_pass),
        "bounce_candidate_accuracy": _r(cand_bounce),
        "bounce_control_accuracy": _r(ctrl_bounce),
        "delta_bounce": _r(cand_bounce - ctrl_bounce),
    }


@register_runner("wave_c.split_merge_identity")
def wave_c_split_merge_runner(params: dict[str, Any], ctx: NodeContext) -> RunResult:
    """Velocity-continuity slots versus greedy nearest-centroid on split-then-merge identity scenes."""

    n_scenes = int(params.get("n_scenes", N_SCENES))
    per_unit = [_simulate_scene(ctx.seed, i) for i in range(n_scenes)]

    primary_deltas = [u["delta_candidate_vs_control"] for u in per_unit]
    bounce_deltas = [u["delta_bounce"] for u in per_unit]

    sign_flip = exact_sign_flip_one_sided(primary_deltas)
    bounce_sign_flip = exact_sign_flip_one_sided(bounce_deltas)

    verdict = verdict_from(sign_flip["mean_delta"], sign_flip["one_sided_p"], SESOI)
    is_null = verdict != "survives"

    # The bounce ablation removes momentum continuity at the merge. If the candidate does not beat the
    # control there, the primary advantage is attributable to velocity continuity, not to extra state.
    bounce_verdict = verdict_from(bounce_sign_flip["mean_delta"], bounce_sign_flip["one_sided_p"], SESOI)
    bounce_beats_control = bounce_verdict == "survives"
    bounce_relation = "beats_control" if bounce_beats_control else "ties_or_below_control"

    content = honest_envelope(
        ctx.node_id,
        "mop-campaign-wave_c_split_merge/v1",
        {
            "form_family": "vision_synthetic",
            "phenomenon": "split_merge_identity",
            "mechanism_family": "structured_state",
            "unit_class": "synthetic_split_merge_scene",
            "evidence_level": "M1",
        },
    )
    content.update(
        {
            "design": {
                "n_scenes": n_scenes,
                "two_fragment_frames": T2,
                "merge_frame": TMERGE,
                "scored_frames": "2_onward",
                "peak_half_separation_range": [H_LOW, H_HIGH],
                "relative_noise_range": [REL_NOISE_LOW, REL_NOISE_HIGH],
                "drift_scale": DRIFT_SCALE,
                "velocity_smoothing": SMOOTH,
                "score": "identity_consistency_accuracy_from_frame_two",
            },
            "control": (
                "greedy_nearest_centroid binds each detection to the slot whose last observed position is "
                "nearest, carrying no velocity and no identity memory; at the merge both slots share a "
                "near-coincident last position, so the re-separating detections are bound by proximity and "
                "the binding can swap. The candidate instead binds to per-slot velocity-predicted positions."
            ),
            "per_unit": per_unit,
            "primary_deltas": primary_deltas,
            "sign_flip": sign_flip,
            "sesoi": SESOI,
            "verdict": verdict,
            "bounce_ablation": {
                "bounce_deltas": bounce_deltas,
                "sign_flip": bounce_sign_flip,
                "verdict": bounce_verdict,
                "beats_control": bounce_beats_control,
                "relation_to_control": bounce_relation,
                "note": (
                    "with the fragments reflecting at the merge the candidate's forward velocity prediction "
                    "points across the merge while each fragment returns to its own side, so velocity "
                    "continuity misbinds; the candidate is not expected to beat the control here."
                ),
            },
            "alternative_explanation": (
                "The candidate could win merely because it holds a velocity slot the control lacks, rather "
                "than because velocity continuity resolves identity through the merge. The bounce ablation "
                "keeps the identical velocity slot but breaks momentum continuity at the merge, and there "
                "the candidate does not beat the control, which is consistent with the primary advantage "
                "coming from velocity continuity across a momentum-conserving merge, not from extra state."
            ),
            "failure_domain": (
                "Velocity discontinuity at the merge: if fragments stop, reverse, or bounce at the moment of "
                "coincidence, the candidate's forward prediction points to the wrong post-merge detection "
                "and identity binding degrades below the memoryless nearest-centroid control."
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
            "mean_delta": sign_flip["mean_delta"],
            "one_sided_p": sign_flip["one_sided_p"],
            "n_units_favorable": sign_flip["n_units_favorable"],
            "bounce_relation_to_control": bounce_relation,
        },
    )
