"""Structurally separate verifier for the Generation 1 General Run closure artifact.

This verifier recomputes the closure's load-bearing claims independently. It imports NOTHING from
``mop.closure``; it uses only the standard library, and it reimplements the canonical seal so that a
seal produced by the producer cannot be trusted on the producer's own word. It recomputes the
admission classification and the mechanics lane counts from the raw bound artifacts, re-derives the
canonical seal, and runs semantic mutation attacks that flip the admitted flag, corrupt a refusal,
change a lane count, and rewrite the general-run state. Each mutation must be detected as either a
seal failure or a consistency failure against the raw artifacts.

The receipt hardcodes ``independent_scientific_confirmation = false``: independent artifact
verification is not a second scientific generator.

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

VERIFICATION_SCHEMA = "mop-generation1-general-run-closure-verification/v1"

CLEAN_TERMINAL_STATE = "complete"

# The full-generations wave admits exactly three new mechanism lanes through the W08 canary gate.
# The verifier encodes this design constant independently rather than trusting the closure artifact.
NEW_LANE_COUNT = 3


def _canonical_bytes(value: Any) -> bytes:
    """Reimplement the strict canonical JSON encoding independently of ``mop.substrate.events``."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _reseal(artifact: dict[str, Any]) -> str:
    content = {k: v for k, v in artifact.items() if k != "seal"}
    return _canonical_sha256(content)


def _recompute_lane_counts(h_final: dict[str, Any], v2_admission: dict[str, Any]) -> dict[str, int]:
    """Recompute the mechanics lane counts from the raw terminal classification and v2 admission."""

    mechanics = h_final.get("mechanics") or {}
    surviving: list[str] = []
    pruned: list[str] = []
    warned: list[str] = []
    failed: list[str] = []
    for lane_id in sorted(mechanics):
        info = mechanics[lane_id] or {}
        klass = str(info.get("classification") or "")
        continue_lane = bool(info.get("continue_lane"))
        if "warn" in klass:
            warned.append(lane_id)
        if "fail" in klass:
            failed.append(lane_id)
        if klass == "not_run_pruned" or not continue_lane:
            pruned.append(lane_id)
        elif klass == "mechanics_noninferential" or continue_lane:
            surviving.append(lane_id)

    dependency_pruned = list(v2_admission.get("mechanics_dependency_pruned_lanes") or [])
    carried = list(v2_admission.get("mechanics_initially_eligible_lanes") or [])
    pruned_union = sorted(set(pruned) | set(dependency_pruned))
    blocked = pruned_union

    return {
        "carried": len(carried),
        "surviving": len(surviving),
        "pruned": len(pruned_union),
        "warned": len(warned),
        "failed": len(failed),
        "blocked": len(blocked),
        "untested": NEW_LANE_COUNT,
    }


def _evaluate(closure_artifact: dict[str, Any], raw_artifacts: dict[str, Any]) -> dict[str, Any]:
    """Recompute every checkable claim. Returns seal_intact and the list of consistency mismatches."""

    mismatches: list[str] = []

    recorded_seal = (closure_artifact.get("seal") or {}).get("sha256")
    recomputed_seal = _reseal(closure_artifact)
    seal_intact = recorded_seal == recomputed_seal
    if not seal_intact:
        mismatches.append("seal sha256 does not match a recomputation over the artifact content")

    raw_status = raw_artifacts.get("general_run_status") or {}
    raw_state = raw_status.get("state")

    # Admission classification: a closure may only be admitted when the general run is the clean
    # terminal state. Any other state (including the running horizon state) must be admitted=false.
    expected_admitted = raw_state == CLEAN_TERMINAL_STATE
    recorded_admitted = bool(closure_artifact.get("admitted"))
    if recorded_admitted != expected_admitted:
        mismatches.append(
            f"admitted={recorded_admitted} is inconsistent with raw general_run_state={raw_state!r}"
        )

    if closure_artifact.get("general_run_state") != raw_state:
        mismatches.append("recorded general_run_state does not match the raw general run status state")

    # Lane counts: recompute from the raw terminal classification and the v2 admission boundary.
    lineage = closure_artifact.get("terminal_lineage") or {}
    recorded_counts = (lineage.get("lane_universe") or {}).get("counts") or {}
    recomputed_counts = _recompute_lane_counts(
        raw_artifacts.get("horizon_v1_final_classification") or {},
        raw_artifacts.get("horizon_v2_admission") or {},
    )
    for key, expected in recomputed_counts.items():
        if recorded_counts.get(key) != expected:
            mismatches.append(
                f"lane count {key}={recorded_counts.get(key)!r} does not match recomputed {expected}"
            )

    return {
        "seal_intact": seal_intact,
        "mismatches": mismatches,
        "recomputed_counts": recomputed_counts,
        "expected_admitted": expected_admitted,
    }


def _mutation_detected(
    closure_artifact: dict[str, Any],
    raw_artifacts: dict[str, Any],
    *,
    mutate,
    reseal: bool,
) -> bool:
    """A mutation is detected when the mutated artifact fails the seal check or a consistency check."""

    mutated = copy.deepcopy(closure_artifact)
    mutate(mutated)
    if reseal:
        # The strongest attacker re-seals the mutated content so the seal check passes; the
        # independent recomputation from the raw artifacts must still catch it.
        mutated["seal"] = {"sha256": _reseal(mutated)}
    outcome = _evaluate(mutated, raw_artifacts)
    return (not outcome["seal_intact"]) or bool(outcome["mismatches"])


def _flip_admitted(artifact: dict[str, Any]) -> None:
    artifact["admitted"] = not bool(artifact.get("admitted"))


def _corrupt_refusal(artifact: dict[str, Any]) -> None:
    refusals = artifact.setdefault("refusals", {})
    refusals["activation_allowed"] = True


def _change_lane_count(artifact: dict[str, Any]) -> None:
    counts = (
        artifact.setdefault("terminal_lineage", {}).setdefault("lane_universe", {}).setdefault("counts", {})
    )
    counts["surviving"] = int(counts.get("surviving") or 0) + 1


def _rewrite_general_run_state(artifact: dict[str, Any]) -> None:
    artifact["general_run_state"] = CLEAN_TERMINAL_STATE


def verify_closure(closure_artifact: dict[str, Any], raw_artifacts: dict[str, Any]) -> dict[str, Any]:
    """Independently verify the sealed closure and return a self-sealed verification receipt."""

    baseline = _evaluate(closure_artifact, raw_artifacts)
    seal_intact = baseline["seal_intact"]
    classifications_reproduced = seal_intact and not baseline["mismatches"]

    mutations = {
        "admitted_flag": _mutation_detected(
            closure_artifact, raw_artifacts, mutate=_flip_admitted, reseal=True
        ),
        "refusal": _mutation_detected(closure_artifact, raw_artifacts, mutate=_corrupt_refusal, reseal=False),
        "lane_count": _mutation_detected(
            closure_artifact, raw_artifacts, mutate=_change_lane_count, reseal=True
        ),
        "general_run_state": _mutation_detected(
            closure_artifact, raw_artifacts, mutate=_rewrite_general_run_state, reseal=True
        ),
    }
    mutations["all_detected"] = all(mutations.values())

    receipt_core: dict[str, Any] = {
        "schema": VERIFICATION_SCHEMA,
        "program_id": "generation1-general-run-closure",
        "seal_intact": seal_intact,
        "classifications_reproduced": classifications_reproduced,
        "mutations_detected": mutations,
        "recomputed_lane_counts": baseline["recomputed_counts"],
        "recomputed_admitted": baseline["expected_admitted"],
        "mismatches": baseline["mismatches"],
        "claim_scope": (
            "independent seal, admission-classification, and lane-count recomputation with semantic "
            "mutation rejection; not a second scientific generator"
        ),
        "activation_allowed": False,
        "scientific_promotion": False,
        "independent_scientific_confirmation": False,
    }
    return {**receipt_core, "seal": {"sha256": _canonical_sha256(receipt_core)}}


__all__ = ["verify_closure", "VERIFICATION_SCHEMA"]
