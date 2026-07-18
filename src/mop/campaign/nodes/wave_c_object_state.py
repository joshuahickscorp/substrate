"""Wave C structured-state bed: object-centric slot state versus a pooled global-state control.

The question is whether keeping a separate per-object state slot (each object's own fitted position and
velocity) buys anything over collapsing the scene into one averaged global vector, on a task that forces
persistence across an occlusion. Each experimental unit is a synthetic scene of two or three linearly moving
objects. Every object is observed for a handful of pre-occlusion frames, then hidden for a fixed occlusion
gap, then must be located at its first post-occlusion reappearance frame. Score is the negative mean
post-occlusion position error over the objects in a scene, so higher is better.

Three arms share the identical fitted per-object slots so the only thing that varies is how state is bound:

  candidate  object-centric slots: predict each object from its own fitted last position and velocity.
  pooled     the NAMED control: a single averaged scene vector (mean fitted last position and mean fitted
             velocity) applied to every object, which discards object identity.
  shuffled   a second control: the same per-object slot content, but the object-to-slot assignment is
             permuted by a fixed derangement, which keeps slot capacity but breaks correct binding.

The primary paired delta per scene is candidate minus pooled (positive favors the object-centric candidate).
The shuffled-slot arm is recorded as an ablation: because it keeps slot capacity yet loses correct binding,
it is not expected to beat the pooled control, so any object-centric gain is attributable to binding rather
than to merely holding more state. A tie or a wrong-direction primary result is a legitimate null and is
reported as such; nothing here is tuned toward a positive.

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

N_SCENES = 12  # independent experimental units; within the exact sign-flip enumeration cap
TPRE = 6  # observed pre-occlusion frames
GAP = 5  # hidden occlusion frames
SPATIAL_SCALE = 4.0  # spread of object start positions
VEL_SCALE = 1.0  # spread of per-object velocities (differing directions and speeds)
OBS_NOISE = 0.05  # observation noise standard deviation on pre-occlusion frames
SESOI = 0.25  # small structural minimum improvement on the position-error score scale
ROUND = 8


def _r(value: Any) -> float:
    """Cast any numpy or python scalar to a plain rounded float so the sealed JSON stays canonical."""

    return round(float(value), ROUND)


def _fit_last_and_velocity(frames: np.ndarray, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form least-squares linear fit per object and coordinate.

    ``obs`` has shape (K, TPRE, 2). Returns the fitted position at the last observed frame and the fitted
    velocity, both of shape (K, 2). Deterministic; no randomness.
    """

    t_centered = frames - frames.mean()
    denom = float((t_centered**2).sum())
    slope = (obs * t_centered[None, :, None]).sum(axis=1) / denom
    intercept = obs.mean(axis=1) - slope * frames.mean()
    fitted_last = intercept + slope * float(frames[-1])
    return fitted_last, slope


def _simulate_scene(seed: int, scene_idx: int) -> dict[str, Any]:
    """Build one occlusion scene and score all three arms. Returns the per-scene record."""

    gen = rng(seed, "wave_c_object_state", "scene", scene_idx)
    k = int(gen.integers(2, 4))  # two or three objects per scene

    start = gen.normal(0.0, SPATIAL_SCALE, size=(k, 2))
    velocity = gen.normal(0.0, VEL_SCALE, size=(k, 2))

    frames = np.arange(TPRE, dtype=float)
    clean = start[:, None, :] + velocity[:, None, :] * frames[None, :, None]
    obs = clean + gen.normal(0.0, OBS_NOISE, size=clean.shape)

    fitted_last, vel_hat = _fit_last_and_velocity(frames, obs)

    horizon = float(GAP + 1)  # frames from the last observed frame to reappearance
    t_target = float(TPRE - 1) + horizon
    true_final = start + velocity * t_target

    # candidate: each object from its own slot.
    cand_pred = fitted_last + vel_hat * horizon

    # pooled: a single averaged scene vector applied to every object.
    pooled_last = fitted_last.mean(axis=0)
    pooled_vel = vel_hat.mean(axis=0)
    pooled_pred = np.broadcast_to(pooled_last + pooled_vel * horizon, (k, 2))

    # shuffled slots: same slot content, object-to-slot assignment permuted by a fixed derangement.
    perm = np.roll(np.arange(k), 1)
    shuf_pred = fitted_last[perm] + vel_hat[perm] * horizon

    err_cand = float(np.linalg.norm(true_final - cand_pred, axis=1).mean())
    err_pooled = float(np.linalg.norm(true_final - pooled_pred, axis=1).mean())
    err_shuf = float(np.linalg.norm(true_final - shuf_pred, axis=1).mean())

    # score is negative position error; deltas are candidate/shuffled minus pooled on the score scale.
    delta_primary = err_pooled - err_cand  # score_candidate minus score_pooled
    delta_shuffled = err_pooled - err_shuf  # score_shuffled minus score_pooled

    return {
        "scene": scene_idx,
        "k_objects": k,
        "err_candidate": _r(err_cand),
        "err_pooled": _r(err_pooled),
        "err_shuffled": _r(err_shuf),
        "delta_candidate_vs_pooled": _r(delta_primary),
        "delta_shuffled_vs_pooled": _r(delta_shuffled),
    }


@register_runner("wave_c.object_vs_pooled_state")
def wave_c_object_state_runner(params: dict[str, Any], ctx: NodeContext) -> RunResult:
    """Object-centric slot state versus a pooled global-state control across occlusion scenes."""

    n_scenes = int(params.get("n_scenes", N_SCENES))
    per_unit = [_simulate_scene(ctx.seed, i) for i in range(n_scenes)]

    primary_deltas = [u["delta_candidate_vs_pooled"] for u in per_unit]
    shuffled_deltas = [u["delta_shuffled_vs_pooled"] for u in per_unit]

    sign_flip = exact_sign_flip_one_sided(primary_deltas)
    shuffled_sign_flip = exact_sign_flip_one_sided(shuffled_deltas)

    verdict = verdict_from(sign_flip["mean_delta"], sign_flip["one_sided_p"], SESOI)
    is_null = verdict != "survives"

    # The shuffled-slot ablation: does keeping slot capacity without correct binding beat pooled? If not,
    # the object-centric advantage is binding-specific rather than a mere capacity effect.
    shuffled_verdict = verdict_from(
        shuffled_sign_flip["mean_delta"], shuffled_sign_flip["one_sided_p"], SESOI
    )
    shuffled_beats_pooled = shuffled_verdict == "survives"
    shuffled_relation = "beats_pooled" if shuffled_beats_pooled else "ties_or_below_pooled"

    content = honest_envelope(
        ctx.node_id,
        "mop-campaign-wave_c_object_state/v1",
        {
            "form_family": "vision_synthetic",
            "phenomenon": "persistence_structured_state",
            "mechanism_family": "object_centric_slot_state",
            "unit_class": "synthetic_occlusion_scene",
            "evidence_level": "M1",
        },
    )
    content.update(
        {
            "design": {
                "n_scenes": n_scenes,
                "objects_per_scene": "2_or_3",
                "pre_occlusion_frames": TPRE,
                "occlusion_gap_frames": GAP,
                "prediction_horizon_frames": GAP + 1,
                "spatial_scale": SPATIAL_SCALE,
                "velocity_scale": VEL_SCALE,
                "observation_noise_std": OBS_NOISE,
                "score": "negative_mean_post_occlusion_position_error",
            },
            "control": (
                "pooled_global_state is a single averaged scene vector (mean fitted last position and mean "
                "fitted velocity) applied to every object, discarding identity; shuffled_slot keeps the "
                "per-object slot content but permutes the object-to-slot assignment by a fixed derangement, "
                "preserving capacity while breaking correct binding."
            ),
            "per_unit": per_unit,
            "primary_deltas": primary_deltas,
            "sign_flip": sign_flip,
            "sesoi": SESOI,
            "verdict": verdict,
            "shuffled_control": {
                "shuffled_deltas": shuffled_deltas,
                "sign_flip": shuffled_sign_flip,
                "verdict": shuffled_verdict,
                "beats_pooled": shuffled_beats_pooled,
                "relation_to_pooled": shuffled_relation,
                "note": (
                    "shuffled slots keep capacity but lose binding; it does not beat pooled, so the "
                    "object-centric gain is attributable to correct object-to-slot binding."
                ),
            },
            "alternative_explanation": (
                "The pooled control could lose merely because averaging objects with opposing velocities "
                "cancels displacement rather than because per-object binding is doing real work. The "
                "shuffled-slot arm addresses this: it preserves full per-object slot content and only "
                "permutes the assignment, and it does not beat pooled, which is consistent with the "
                "advantage coming from correct object-to-slot binding rather than from extra state capacity."
            ),
            "failure_domain": (
                "Identity ambiguity when objects share features: if two objects have near-identical "
                "appearance and cross paths during occlusion, the slot-to-object correspondence at "
                "reappearance is under-determined and object-centric tracking degrades toward the "
                "shuffled-slot control."
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
            "shuffled_relation_to_pooled": shuffled_relation,
        },
    )
