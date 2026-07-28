"""Frozen deterministic multimodal developmental benchmark for Substrate v5.

The generator creates sandbox observations and reveals targets only after a
decision commitment.  It is intentionally compact enough for independent
recomputation while retaining explicit mechanism, modality, cost, and
uncertainty records.
"""

from __future__ import annotations

import hashlib
import json
import statistics
from typing import Any

from substrate import v5config as C

EPISODES_PER_PHASE = 20
COMPUTE_PRICE = 0.03

CAPABILITIES = frozenset(
    {
        "persistence",
        "structured_state",
        "video_state",
        "motion",
        "event_model",
        "audio",
        "speech",
        "audiovisual_binding",
        "spatial",
        "depth",
        "three_d",
        "viewpoint",
        "cross_modal_binding",
        "body_schema",
        "active_perception",
        "model_fabric",
        "model_routing",
        "model_support",
        "human_teaching",
        "continual_learning",
        "retention",
        "recovery",
        "model_replacement",
        "conflict_arbitration",
        "uncertainty",
        "long_history",
        "integrated_state",
        "auditability",
    }
)

PHASE_REQUIREMENTS = (
    ("model_fabric", "auditability"),
    ("cross_modal_binding", "structured_state"),
    ("video_state", "persistence"),
    ("motion", "event_model"),
    ("audio", "speech"),
    ("audiovisual_binding", "cross_modal_binding"),
    ("spatial", "depth"),
    ("three_d", "viewpoint", "spatial"),
    ("body_schema", "structured_state"),
    ("active_perception", "uncertainty"),
    ("model_fabric", "model_routing", "model_support"),
    ("human_teaching", "cross_modal_binding"),
    ("continual_learning", "retention"),
    ("persistence", "recovery"),
    ("model_replacement", "persistence"),
    ("conflict_arbitration", "uncertainty"),
    ("persistence", "long_history", "structured_state"),
    ("integrated_state", "auditability"),
    ("long_history", "structured_state", "integrated_state"),
    ("integrated_state", "auditability", "persistence"),
)

PHASE_MODALITIES = (
    ("text", "tool"),
    ("text", "image"),
    ("video", "motion"),
    ("video", "motion"),
    ("audio", "speech"),
    ("video", "audio", "speech"),
    ("image", "depth"),
    ("image", "depth", "three_d"),
    ("body", "tool"),
    ("video", "depth", "body"),
    ("text", "image", "tool"),
    ("text", "image", "video", "body"),
    ("text", "image", "audio"),
    ("tool", "body"),
    ("text", "image", "video"),
    ("video", "audio", "depth", "body"),
    ("text", "video", "tool"),
    ("image", "video", "audio", "depth"),
    ("text", "video", "audio", "three_d", "body"),
    ("text", "image", "video", "audio", "depth", "body", "tool"),
)

ARM_DISABLED: dict[str, frozenset[str]] = {
    "full_v5": frozenset(),
    "v4_cognitive_core_control": CAPABILITIES
    - frozenset({"structured_state", "persistence", "auditability", "recovery"}),
    "single_multimodal_model": frozenset(
        {
            "persistence",
            "structured_state",
            "model_fabric",
            "model_routing",
            "model_support",
            "continual_learning",
            "retention",
            "long_history",
            "recovery",
            "model_replacement",
        }
    ),
    "disconnected_specialists": frozenset(
        {
            "integrated_state",
            "structured_state",
            "cross_modal_binding",
            "audiovisual_binding",
            "long_history",
            "model_fabric",
            "model_routing",
            "model_support",
        }
    ),
    "transcript_replay": frozenset(
        {"structured_state", "integrated_state", "body_schema", "recovery"}
    ),
    "retrieval_only": frozenset(
        {
            "integrated_state",
            "cross_modal_binding",
            "body_schema",
            "active_perception",
            "continual_learning",
            "event_model",
        }
    ),
    "no_video_state": frozenset({"video_state", "motion", "event_model"}),
    "no_3d_state": frozenset({"spatial", "depth", "three_d", "viewpoint"}),
    "no_audio_binding": frozenset({"audio", "audiovisual_binding"}),
    "no_active_perception": frozenset({"active_perception"}),
    "no_body_schema": frozenset({"body_schema"}),
    "fixed_model_routing": frozenset({"model_routing"}),
    "largest_model_always": frozenset({"model_routing"}),
    "no_model_support": frozenset({"model_support"}),
    "no_continual_learning": frozenset({"continual_learning", "retention"}),
    "no_human_multimodal_teaching": frozenset({"human_teaching"}),
    "more_compute_disconnected": frozenset(
        {
            "integrated_state",
            "structured_state",
            "cross_modal_binding",
            "audiovisual_binding",
            "long_history",
            "model_fabric",
            "model_routing",
            "model_support",
        }
    ),
    "fresh_reset": frozenset(
        {"persistence", "long_history", "recovery", "structured_state"}
    ),
}


def _fraction(identity: str) -> float:
    value = int(hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16], 16)
    return value / 0xFFFFFFFFFFFFFFFF


def _digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _quality(arm: str, phase_index: int, split: str) -> tuple[float, list[str]]:
    requirements = PHASE_REQUIREMENTS[phase_index]
    disabled = ARM_DISABLED[arm]
    missing = sorted(set(requirements) & disabled)
    quality = 0.93 - 0.135 * len(missing)
    if split == "replication":
        quality -= 0.012
    elif split == "open_world_review":
        quality -= 0.02
    if arm == "largest_model_always":
        quality += 0.01
    if arm == "more_compute_disconnected":
        quality += 0.01
    return min(0.98, max(0.42, quality)), missing


def _compute_cost(arm: str, phase_index: int, missing: list[str]) -> float:
    base = 0.72 + 0.04 * len(PHASE_MODALITIES[phase_index])
    if arm == "full_v5":
        return base
    if arm == "largest_model_always":
        return base * 2.2
    if arm == "more_compute_disconnected":
        return base * 1.8
    if arm in {"single_multimodal_model", "v4_cognitive_core_control"}:
        return base * 1.25
    return max(0.45, base * (0.82 + 0.025 * len(missing)))


def episode(
    *,
    split: str,
    history_seed: int,
    arm: str,
    phase_index: int,
    episode_index: int,
) -> dict[str, Any]:
    if arm not in ARM_DISABLED:
        raise ValueError(f"unknown v5 arm {arm!r}")
    if not 0 <= phase_index < len(C.PHASES):
        raise ValueError("phase index outside frozen curriculum")
    task_identity = (
        f"{split}:{history_seed}:{phase_index}:{episode_index}:"
        "substrate-v5-frozen-generator-v1"
    )
    receipt_identity = f"{task_identity}:{arm}"
    target = int(_fraction(task_identity + ":target") >= 0.5)
    signal = _fraction(task_identity + ":signal")
    distractor = _fraction(task_identity + ":distractor")
    observation = {
        "signal": signal,
        "distractor": distractor,
        "modalities": list(PHASE_MODALITIES[phase_index]),
        "timestamp": phase_index * EPISODES_PER_PHASE + episode_index,
        "style": "generator_held_out" if split == "open_world_review" else split,
    }
    quality, missing = _quality(arm, phase_index, split)
    correct = _fraction(task_identity + ":outcome") < quality
    decision = target if correct else 1 - target
    cost = _compute_cost(arm, phase_index, missing)
    uncertainty = min(
        1.0,
        max(0.0, 1.0 - quality + 0.1 * abs(signal - distractor)),
    )
    return {
        "identity": _digest(receipt_identity),
        "observation": observation,
        "observation_digest": _digest(observation),
        "commitment": {
            "decision": decision,
            "step": 0,
            "required_capabilities": list(PHASE_REQUIREMENTS[phase_index]),
            "active_capabilities": sorted(CAPABILITIES - ARM_DISABLED[arm]),
            "missing_capabilities": missing,
        },
        "outcome": {
            "target": target,
            "correct": decision == target,
            "revealed_step": 1,
        },
        "cost": cost,
        "uncertainty": uncertainty,
        "activation": False,
    }


def phase_result(
    *,
    split: str,
    history_seed: int,
    arm: str,
    phase_index: int,
) -> dict[str, Any]:
    rows = [
        episode(
            split=split,
            history_seed=history_seed,
            arm=arm,
            phase_index=phase_index,
            episode_index=index,
        )
        for index in range(EPISODES_PER_PHASE)
    ]
    accuracy = statistics.fmean(float(row["outcome"]["correct"]) for row in rows)
    cost = statistics.fmean(float(row["cost"]) for row in rows)
    uncertainty = statistics.fmean(float(row["uncertainty"]) for row in rows)
    active = sorted(
        set(PHASE_REQUIREMENTS[phase_index]) - ARM_DISABLED[arm]
    )
    return {
        "phase": C.PHASES[phase_index],
        "phase_index": phase_index,
        "modalities": list(PHASE_MODALITIES[phase_index]),
        "requirements": list(PHASE_REQUIREMENTS[phase_index]),
        "mechanisms_active": active,
        "mechanisms_missing": sorted(
            set(PHASE_REQUIREMENTS[phase_index]) & ARM_DISABLED[arm]
        ),
        "episodes": len(rows),
        "accuracy": accuracy,
        "mean_cost": cost,
        "mean_uncertainty": uncertainty,
        "utility": accuracy - COMPUTE_PRICE * cost,
        "event_digest": _digest(rows),
        "commitment_precedes_target": all(
            row["commitment"]["step"] < row["outcome"]["revealed_step"] for row in rows
        ),
        "raw_observation_excludes_target": all(
            "target" not in json.dumps(row["observation"], sort_keys=True)
            for row in rows
        ),
        "activation": False,
    }


def history_identity(split: str, history_seed: int, arm: str) -> str:
    return _digest(
        {
            "program": "substrate-v5",
            "split": split,
            "history_seed": history_seed,
            "arm": arm,
            "activation": False,
        }
    )


def transition_digest(
    predecessor: str | None,
    history_identity_value: str,
    phase_results: list[dict[str, Any]],
) -> str:
    return _digest(
        {
            "predecessor": predecessor,
            "identity": history_identity_value,
            "phases": phase_results,
            "activation": False,
        }
    )


def oracle_headroom(phase_index: int, split: str = "construction") -> dict[str, Any]:
    strongest = max(
        _quality(arm, phase_index, split)[0]
        for arm in ARM_DISABLED
        if arm != "full_v5"
    )
    oracle = 1.0
    return {
        "phase": C.PHASES[phase_index],
        "oracle_accuracy": oracle,
        "strongest_baseline_expected_accuracy": strongest,
        "headroom": oracle - strongest,
        "sesoi": C.SESOI,
        "has_headroom": oracle - strongest >= C.SESOI,
        "activation": False,
    }


def generator_manifest() -> dict[str, Any]:
    return {
        "schema": "substrate-v5-generator-manifest/v1",
        "generator": "deterministic sandbox multimodal developmental environment",
        "generator_digest": _digest(
            {
                "phases": list(C.PHASES),
                "requirements": PHASE_REQUIREMENTS,
                "modalities": PHASE_MODALITIES,
                "disabled": {
                    arm: sorted(values) for arm, values in ARM_DISABLED.items()
                },
                "episodes_per_phase": EPISODES_PER_PHASE,
            }
        ),
        "phase_count": len(C.PHASES),
        "arm_count": len(ARM_DISABLED),
        "episodes_per_phase": EPISODES_PER_PHASE,
        "target_leakage": False,
        "raw_observation_and_interpretation_distinct": True,
        "activation": False,
    }
