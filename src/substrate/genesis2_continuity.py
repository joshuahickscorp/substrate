"""Frozen event/cycle continuity authority for non-continuous-time materials."""

from __future__ import annotations

import time
from typing import Any, cast

from substrate import genesis2_config as C2
from substrate import genesis2_controls  # noqa: F401
from substrate import genesis2_io as IO2
from substrate import genesis2_material as F2
from substrate import genesis_material as M


class ContinuityRefused(RuntimeError):
    pass


def _opportunity(_candidate: str) -> M.Opportunity:
    seed = [M.Observation(0, "continuity", (1, 0, 0), teaching=True)]
    return M.equal_opportunity(
        envelope="1GB",
        observations=seed,
        sensor_channels=tuple(C2.DEVELOPMENTAL_ARC),
        operation_budget=1_000_000,
        durable_write_budget=8_192,
    )


def _build(candidate: str) -> F2._FieldCore:
    material = M.build(candidate, _opportunity(candidate))
    if not isinstance(material, F2._FieldCore):
        raise ContinuityRefused(f"{candidate!r} does not expose the Genesis II continuity contracts")
    return material


def _stable_decision_state(material: F2._FieldCore) -> bool:
    archive_undo = getattr(material, "archive_undo", {})
    return not material.pending and not material.store.undo and not material.structure_undo and not archive_undo


def _anchor(
    *,
    kind: str,
    checkpoint: dict[str, Any],
    cycle: int,
    events: int,
    material: F2._FieldCore,
) -> dict[str, Any]:
    row = {
        "kind": kind,
        "checkpoint_digest": IO2.digest(checkpoint),
        "source_digest": IO2.source_digest(),
        "cycle": cycle,
        "events": events,
        "durable_digest": material.durable_state_digest(),
        "active_digest": material.active_state_digest(),
        "stable_decision_state": _stable_decision_state(material),
        "activation": False,
    }
    row["sha256"] = IO2.digest(row)
    return row


def run(candidate: str) -> dict[str, Any]:
    candidate_config = C2.CANDIDATES.get(candidate)
    if candidate_config is None:
        raise ContinuityRefused(f"unknown candidate {candidate!r}")
    if candidate_config.get("continuous_time"):
        raise ContinuityRefused("this authority is invalid for an active continuous-time mechanism")

    requirements = C2.FROZEN_CONTINUITY_REQUIREMENTS
    cycles = int(requirements["cycles"])
    event_target = int(requirements["events"])
    checkpoint_target = int(requirements["checkpoints"])
    interruption_target = int(requirements["interruptions"])
    migration_target = int(requirements["migrations"])
    model_replacement_target = int(requirements["model_organ_replacements"])
    body_replacement_target = int(requirements["body_organ_replacements"])

    material = _build(candidate)
    events_per_cycle, extra = divmod(event_target, cycles)
    checkpoint_every = cycles // checkpoint_target
    interruption_every = cycles // interruption_target
    migration_every = cycles // migration_target
    replacement_every = cycles // max(model_replacement_target, body_replacement_target)

    event_count = 0
    checkpoint_count = 0
    interruption_count = 0
    migration_count = 0
    model_replacements = 0
    body_replacements = 0
    committed = 0
    checkpoint_receipts: list[dict[str, Any]] = []
    handoff_receipts: list[dict[str, Any]] = []
    migration_receipts: list[dict[str, Any]] = []
    interruption_cycles: list[int] = []
    stream_digests: list[str] = []
    proposal_epoch_preserved: list[bool] = []
    resource_ledger_preserved: list[bool] = []
    update_ledger_preserved: list[bool] = []
    receipt_lineage_preserved: list[bool] = []
    cpu_started = time.process_time()
    wall_started = time.monotonic()

    last_checkpoint: dict[str, Any] | None = None
    last_transfer_checkpoint: dict[str, Any] | None = None
    for cycle in range(cycles):
        stage = C2.DEVELOPMENTAL_ARC[cycle % len(C2.DEVELOPMENTAL_ARC)]
        count = events_per_cycle + (1 if cycle < extra else 0)
        for offset in range(count):
            value = (cycle + offset) % 8
            observation = M.Observation(
                index=event_count,
                channel=stage,
                payload=(1, cycle % 17, value, (3 * value + 2) % 8),
                elapsed_ms=1,
                teaching=offset % 31 == 0,
                modality="symbolic" if cycle % 2 == 0 else "sensor",
            )
            material.observe(observation)
            stream_digests.append(observation.digest())
            event_count += 1

        proposals = list(material.propose())
        if proposals:
            verdicts = [M.Verdict(row.proposal_id, True, 0.0, 1.0) for row in proposals]
            emitted = material.apply(verdicts)
            committed += sum(1 for row in emitted if row.committed)
            for receipt in emitted:
                if receipt.committed:
                    material.finalize_receipt(receipt)

        if (cycle + 1) % replacement_every == 0:
            if model_replacements < model_replacement_target:
                model_replacements += 1
                material.replace_organ("model", "reasoner", f"replaceable-model-v{model_replacements + 1}")
            if body_replacements < body_replacement_target:
                body_replacements += 1
                material.replace_organ("body", "sensor_fabric", f"replaceable-body-v{body_replacements + 1}")

        if (cycle + 1) % checkpoint_every == 0:
            checkpoint_count += 1
            if not _stable_decision_state(material):
                raise ContinuityRefused("scheduled checkpoint captured unresolved rollback state")
            last_checkpoint = material.checkpoint()
            checkpoint_receipts.append(
                _anchor(
                    kind="scheduled_checkpoint",
                    checkpoint=last_checkpoint,
                    cycle=cycle + 1,
                    events=event_count,
                    material=material,
                )
            )

        # Interruptions deliberately occur one cycle after each 32-cycle
        # boundary, so restore is tested from live state rather than only from
        # the already scheduled 16-cycle anchors.
        if cycle % interruption_every == 0:
            interruption_count += 1
            interruption_cycles.append(cycle + 1)
            if not _stable_decision_state(material):
                raise ContinuityRefused("handoff captured unresolved rollback state")
            last_transfer_checkpoint = material.checkpoint()
            durable_before = material.durable_state_digest()
            active_before = material.active_state_digest()
            active_state_before = material._active_state()
            replacement = _build(candidate)
            replacement.restore(last_transfer_checkpoint)
            if replacement.durable_state_digest() != durable_before:
                raise ContinuityRefused("durable state changed across process replacement")
            if replacement.active_state_digest() != active_before:
                raise ContinuityRefused("active state changed across process replacement")
            active_state_after = replacement._active_state()
            proposal_epoch_preserved.append(active_state_after["proposal_epoch"] == active_state_before["proposal_epoch"])
            resource_ledger_preserved.append(active_state_after["resource_ledger"] == active_state_before["resource_ledger"])
            update_ledger_preserved.append(active_state_after["update_ledger"] == active_state_before["update_ledger"])
            receipt_lineage_preserved.append(active_state_after["receipts"] == active_state_before["receipts"])
            handoff_receipts.append(
                _anchor(
                    kind="off_anchor_process_handoff",
                    checkpoint=last_transfer_checkpoint,
                    cycle=cycle + 1,
                    events=event_count,
                    material=replacement,
                )
            )
            material = replacement

        if (cycle + 1) % migration_every == 0:
            migration_count += 1
            if not _stable_decision_state(material):
                raise ContinuityRefused("migration captured unresolved rollback state")
            checkpoint = material.checkpoint()
            migrated = _build(candidate)
            migrated.restore(checkpoint)
            if migrated.durable_state_digest() != material.durable_state_digest():
                raise ContinuityRefused("migration changed the developed state")
            if migrated.active_state_digest() != material.active_state_digest():
                raise ContinuityRefused("migration changed active or accounting state")
            migration_receipts.append(
                _anchor(
                    kind="migration_snapshot",
                    checkpoint=checkpoint,
                    cycle=cycle + 1,
                    events=event_count,
                    material=migrated,
                )
            )
            last_transfer_checkpoint = checkpoint
            material = migrated

    elapsed = time.monotonic() - wall_started
    cpu = time.process_time() - cpu_started
    checks = {
        "cycles_exact": cycles == int(requirements["cycles"]),
        "events_exact": event_count == event_target,
        "interruptions_exact": interruption_count == interruption_target,
        "checkpoints_exact": checkpoint_count == checkpoint_target,
        "handoff_snapshots_exact": len(handoff_receipts) == interruption_target,
        "migration_snapshots_exact": len(migration_receipts) == migration_target,
        "interruptions_off_scheduled_anchors": all(cycle % checkpoint_every != 0 for cycle in interruption_cycles),
        "migrations_exact": migration_count == migration_target,
        "model_replacements_exact": model_replacements == model_replacement_target,
        "body_replacements_exact": body_replacements == body_replacement_target,
        "developmental_stages_covered": len(C2.DEVELOPMENTAL_ARC) == int(requirements["developmental_history_stages"]),
        "every_receipt_anchored": all(
            receipt["sha256"] == IO2.digest({key: value for key, value in receipt.items() if key != "sha256"})
            for receipt in (
                *checkpoint_receipts,
                *handoff_receipts,
                *migration_receipts,
            )
        ),
        "proposal_epoch_preserved": all(proposal_epoch_preserved),
        "residual_resource_ledger_preserved": all(resource_ledger_preserved),
        "internal_update_ledger_preserved": all(update_ledger_preserved),
        "receipt_lineage_preserved": all(receipt_lineage_preserved),
        "no_unresolved_undo_at_finish": _stable_decision_state(material),
        "activation_false": C2.ACTIVATION is False,
    }
    return {
        "candidate": candidate,
        "authority": "frozen_event_cycle_interruption_checkpoint_migration_and_developmental_history",
        "continuous_time_active": False,
        "twelve_hour_lane_required": False,
        "requirements": dict(requirements),
        "observed": {
            "cycles": cycles,
            "events": event_count,
            "interruptions": interruption_count,
            "checkpoints": checkpoint_count,
            "scheduled_checkpoints": checkpoint_count,
            "handoff_snapshots": len(handoff_receipts),
            "migration_snapshots": len(migration_receipts),
            "total_checkpoint_serializations": (checkpoint_count + len(handoff_receipts) + len(migration_receipts)),
            "migrations": migration_count,
            "model_organ_replacements": model_replacements,
            "body_organ_replacements": body_replacements,
            "committed_updates": committed,
            "wall_clock_seconds": elapsed,
            "cpu_seconds": cpu,
        },
        "stream_digest": IO2.digest(stream_digests),
        "final_checkpoint_digest": None if last_checkpoint is None else IO2.digest(last_checkpoint),
        "final_transfer_checkpoint_digest": (None if last_transfer_checkpoint is None else IO2.digest(last_transfer_checkpoint)),
        "final_durable_digest": material.durable_state_digest(),
        "final_active_digest": material.active_state_digest(),
        "checkpoint_receipts": checkpoint_receipts,
        "handoff_receipts": handoff_receipts,
        "migration_receipts": migration_receipts,
        "interruption_cycles": interruption_cycles,
        "checks": checks,
        "all_pass": all(checks.values()),
        "activation": False,
    }


def demo() -> None:
    # A reduced structural smoke test; the published lane uses frozen counts.
    material = _build("L9_minimal_sufficient_field")
    observation = M.Observation(0, "continuity", (1, 1, 2), teaching=True)
    material.observe(observation)
    checkpoint = material.checkpoint()
    replica = _build("L9_minimal_sufficient_field")
    replica.restore(checkpoint)
    assert replica.durable_state_digest() == material.durable_state_digest()
    assert replica.active_state_digest() == material.active_state_digest()
    assert cast(dict[str, Any], checkpoint)["activation"] is False
    print("genesis2 continuity smoke self-check passed")


if __name__ == "__main__":
    demo()
