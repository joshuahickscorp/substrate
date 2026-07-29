"""The long sequential developmental lane.

This is the one part of the program that cannot be parallelised or compressed,
because what it tests is continuity itself: whether a material that has been
developing for many hours still holds what it learned early, survives being
interrupted and restored, survives migration to a different representation, and
keeps exact identity across all of it.

The lane is paced against the real clock. It does not simulate elapsed time,
and it does not advance its own clock on the material's behalf beyond the
harness tick, because a material that advanced its own clock would be running a
declared mutation.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from substrate import genesis_config as C
from substrate import genesis_harness as H
from substrate import genesis_history as HI
from substrate import genesis_io as io
from substrate import genesis_material as M
from substrate import genesis_tournament as T

HEARTBEAT_SECONDS = 300.0
CHECKPOINT_EVERY_SECONDS = 900.0
INTERRUPTION_FRACTIONS = (0.25, 0.55, 0.80)
MIGRATION_FRACTION = 0.65


class ContinuityRefused(RuntimeError):
    """The lane violated a condition that would void the continuity claim."""


@dataclass
class LaneState:
    """Everything the lane accumulates, and everything it must be able to resume from."""

    arm: str
    started_at: float
    elapsed_seconds: float = 0.0
    cycles: int = 0
    observations_delivered: int = 0
    committed: int = 0
    rolled_back: int = 0
    checkpoints: int = 0
    interruptions: int = 0
    migrations: int = 0
    early_score_first: float | None = None
    early_score_latest: float | None = None
    heartbeats: list[dict[str, Any]] = field(default_factory=list)
    identity_digests: list[str] = field(default_factory=list)


def _score(material: M.CognitiveMaterial, unit: T.Unit) -> float:
    return unit.judge.score_scoring([material.answer(probe) for probe in unit.probes.scoring])


def _retention(material: M.CognitiveMaterial, unit: T.Unit) -> float:
    return unit.judge.score_retention([material.answer(probe) for probe in unit.probes.retention])


def _cycle(material: M.CognitiveMaterial, unit: T.Unit, state: LaneState) -> None:
    """One developmental pass: observe the history, then verify proposals."""
    for observation in unit.observations:
        material.observe(observation)
        state.observations_delivered += 1
    proposals = material.propose()
    for proposal in proposals:
        before = unit.judge.score_development([material.answer(probe) for probe in unit.probes.development])
        retention_before = _retention(material, unit)
        emitted = material.apply([M.Verdict(proposal.proposal_id, True, 0.0, 0.0)])
        if not emitted:
            continue
        receipt = emitted[0]
        improvement = unit.judge.score_development([material.answer(probe) for probe in unit.probes.development]) - before
        retention = _retention(material, unit) - retention_before
        if improvement > 0.0 and retention >= 0.0:
            state.committed += 1
        else:
            material.rollback(receipt)
            state.rolled_back += 1


def run_lane(
    *,
    arm: str,
    seed_namespace: str,
    duration_seconds: float,
    families: Sequence[str] | None = None,
    envelope: str = "1GB",
    operation_budget: int = 4_000_000_000,
    durable_write_budget: int = 4_000_000,
    heartbeat_path: os.PathLike[str] | None = None,
    disclosed_short_run: bool = False,
) -> dict[str, Any]:
    """Develop one material continuously for the requested wall-clock duration.

    ``disclosed_short_run`` permits a lane below the frozen minimum. It does not
    make the lane conforming: the report carries the shortfall and its
    ``all_pass`` is false, so a short lane can never be mistaken for the twelve
    hour one the constitution requires. The substantive mechanism checks are
    reported separately as ``mechanisms_pass``.
    """
    frozen_minimum = float(C.CONTINUITY_LANE_MINIMUM_SECONDS)
    if duration_seconds < frozen_minimum and not disclosed_short_run:
        raise ContinuityRefused(
            f"the continuity lane must run at least {C.CONTINUITY_LANE_MINIMUM_SECONDS} seconds, not {duration_seconds}"
        )
    if duration_seconds > C.CONTINUITY_LANE_MAXIMUM_SECONDS:
        raise ContinuityRefused(f"the continuity lane is capped at {C.CONTINUITY_LANE_MAXIMUM_SECONDS} seconds")

    families = list(families or C.CHALLENGE_FAMILIES)
    started = time.monotonic()
    wall_started = time.time()
    state = LaneState(arm=arm, started_at=wall_started)

    # The early history. Its score is measured at the start and again at the
    # end: this is what "the material still holds what it learned early" means.
    early = HI.build_history(family=families[0], split="train", history_id=0, seed_namespace=seed_namespace)

    opportunity = M.equal_opportunity(
        envelope=envelope,
        observations=early.observations,
        sensor_channels=tuple(sorted({observation.channel for observation in early.observations})),
        operation_budget=operation_budget,
        durable_write_budget=durable_write_budget,
    )
    material = M.build(arm, opportunity)

    _cycle(material, early, state)
    state.early_score_first = _score(material, early)
    state.identity_digests.append(material.durable_state_digest())

    last_heartbeat = started
    last_checkpoint = started
    checkpoint: dict[str, Any] | None = None
    interruptions_done: set[float] = set()
    migration_done = False
    restore_mismatch: str | None = None
    migration_mismatch: str | None = None

    while True:
        elapsed = time.monotonic() - started
        if elapsed >= duration_seconds:
            break
        fraction = elapsed / duration_seconds

        family = families[state.cycles % len(families)]
        history_id = 1 + (state.cycles % 64)
        unit = HI.build_history(family=family, split="train", history_id=history_id, seed_namespace=seed_namespace)
        _cycle(material, unit, state)
        state.cycles += 1

        now = time.monotonic()

        if now - last_checkpoint >= CHECKPOINT_EVERY_SECONDS:
            checkpoint = material.checkpoint()
            state.checkpoints += 1
            last_checkpoint = now

        # Interruption and recovery. The material is torn down and rebuilt from
        # its checkpoint, and the durable identity must survive exactly.
        for point in INTERRUPTION_FRACTIONS:
            if fraction >= point and point not in interruptions_done:
                interruptions_done.add(point)
                if checkpoint is None:
                    checkpoint = material.checkpoint()
                    state.checkpoints += 1
                before_digest = material.durable_state_digest()
                revived = M.build(arm, opportunity)
                revived.restore(checkpoint)
                material = revived
                if checkpoint["durable"] != material.checkpoint()["durable"]:
                    restore_mismatch = f"restore at fraction {point} did not reproduce the checkpointed durable state"
                state.interruptions += 1
                state.identity_digests.append(before_digest)

        # Migration: export, rebuild into a fresh instance, and require the
        # durable identity to be preserved across the move.
        if not migration_done and fraction >= MIGRATION_FRACTION:
            migration_done = True
            exported = material.checkpoint()
            before_digest = material.durable_state_digest()
            migrated = M.build(arm, opportunity)
            migrated.restore(exported)
            if migrated.durable_state_digest() != before_digest:
                migration_mismatch = "migration did not preserve the durable state digest"
            material = migrated
            state.migrations += 1

        if now - last_heartbeat >= HEARTBEAT_SECONDS:
            last_heartbeat = now
            beat = {
                "elapsed_seconds": round(now - started, 1),
                "cycles": state.cycles,
                "observations": state.observations_delivered,
                "committed": state.committed,
                "rolled_back": state.rolled_back,
                "checkpoints": state.checkpoints,
                "interruptions": state.interruptions,
                "migrations": state.migrations,
                "early_score": _score(material, early),
                "durable_digest": material.durable_state_digest()[:16],
                "activation": False,
            }
            state.heartbeats.append(beat)
            if heartbeat_path is not None:
                with open(heartbeat_path, "a") as handle:
                    handle.write(json.dumps(beat) + "\n")
                    handle.flush()

    state.elapsed_seconds = time.monotonic() - started
    state.early_score_latest = _score(material, early)
    state.identity_digests.append(material.durable_state_digest())

    # Mechanism checks are what the lane substantively measures. Duration
    # conformance is tracked separately so a short lane reports as short rather
    # than as a slightly smaller version of the required one.
    mechanisms = {
        "sequential_not_parallel": True,
        "checkpoints_taken": state.checkpoints > 0,
        "interruptions_survived": state.interruptions >= len(INTERRUPTION_FRACTIONS),
        "restore_exact": restore_mismatch is None,
        "migration_performed": state.migrations > 0,
        "migration_preserved_identity": migration_mismatch is None,
        "early_learning_retained": (
            state.early_score_first is not None
            and state.early_score_latest is not None
            and state.early_score_latest >= state.early_score_first
        ),
        "development_occurred": state.committed > 0,
    }
    conforms = state.elapsed_seconds >= frozen_minimum
    checks = {**mechanisms, "ran_at_least_the_frozen_minimum": conforms}

    return {
        "arm": arm,
        "seed_namespace": seed_namespace,
        "requested_seconds": duration_seconds,
        "frozen_minimum_seconds": frozen_minimum,
        "conforms_to_frozen_minimum": conforms,
        "disclosed_short_run": bool(disclosed_short_run),
        "shortfall_seconds": max(0.0, frozen_minimum - state.elapsed_seconds),
        "mechanisms_pass": all(mechanisms.values()),
        "elapsed_seconds": state.elapsed_seconds,
        "started_at": state.started_at,
        "cycles": state.cycles,
        "observations_delivered": state.observations_delivered,
        "committed_rewrites": state.committed,
        "rolled_back_rewrites": state.rolled_back,
        "checkpoints": state.checkpoints,
        "interruptions": state.interruptions,
        "migrations": state.migrations,
        "early_score_first": state.early_score_first,
        "early_score_latest": state.early_score_latest,
        "restore_mismatch": restore_mismatch,
        "migration_mismatch": migration_mismatch,
        "heartbeats": state.heartbeats,
        "identity_digests": state.identity_digests,
        "final_durable_digest": material.durable_state_digest(),
        "cost": material.cost(),
        "checks": checks,
        "all_pass": all(checks.values()),
        "activation": False,
    }


def publish(report: dict[str, Any]) -> dict[str, Any]:
    document = io.authority("substrate-genesis-continuity/v1", report)
    io.write_json(io.EVIDENCE / "SUBSTRATE_GENESIS_CONTINUITY.json", document)
    return document


def demo() -> None:
    """The duration floor is the point of this lane; prove it is enforced."""
    import substrate.genesis_k_basic  # noqa: F401
    try:
        run_lane(arm="K1_monolithic_plastic_field", seed_namespace="demo", duration_seconds=60)
    except ContinuityRefused as error:
        assert "at least" in str(error)
    else:  # pragma: no cover
        raise AssertionError("the twelve hour floor was not enforced")

    # A disclosed short run is permitted but can never report as conforming.
    short = run_lane(
        arm="K1_monolithic_plastic_field",
        seed_namespace="demo",
        duration_seconds=2,
        disclosed_short_run=True,
    )
    assert short["disclosed_short_run"] is True
    assert short["conforms_to_frozen_minimum"] is False
    assert short["all_pass"] is False, "a short lane must not report as a passing continuity lane"
    assert short["shortfall_seconds"] > 0

    try:
        run_lane(arm="K1_monolithic_plastic_field", seed_namespace="demo", duration_seconds=200_000)
    except ContinuityRefused as error:
        assert "capped" in str(error)
    else:  # pragma: no cover
        raise AssertionError("the twenty-four hour cap was not enforced")

    assert H.DEVELOPMENT != H.SCORING
    print("genesis continuity self-check passed: duration floor and cap enforced")


if __name__ == "__main__":
    demo()
